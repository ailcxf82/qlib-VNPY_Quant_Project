"""阶段 F.1：家族分类规则 loader。

``factor_families.yaml`` 是主事实源；本模块负责加载、校验、缓存、兜底。

用法：
    >>> from factor_lab.config import classify_family, get_family_rules
    >>> classify_family("VolRev_5d")
    'volume_price_reversal'
    >>> [r.label for r in get_family_rules()]
    ['volume_price_reversal', ...]

回退行为：
* YAML 文件不存在 / 解析失败 / schema 校验失败 → 使用 ``DEFAULT_FAMILY_RULES``（Python
  硬编码）并 ``logger.warning``。
* 运行期 import yaml 失败 → 同样回退（研究员未装 pyyaml 也不阻塞聚合器）。

性能：
* 首次调用 ``get_family_rules()`` 做 IO；后续调用用进程级缓存。
* ``reload_family_rules()`` 提供显式重新加载（单测用）。
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = "1.0"


# ----------------------------------------------------------------- data classes


@dataclass(frozen=True)
class FamilyRule:
    """单条家族分类规则。"""

    label: str
    pattern: re.Pattern[str]

    def matches(self, name: str) -> bool:
        return self.pattern.search(name) is not None


# ------------------------------------------------------------------- defaults


# Python 兜底副本：当 YAML 不可用时使用；必须与 factor_families.yaml 的内容保持一致。
# **修改时请同步更新两份**。
DEFAULT_FAMILY_RULES: tuple[FamilyRule, ...] = (
    FamilyRule(
        label="volume_price_reversal",
        pattern=re.compile(r"(VolRev|VolRet|VolTwist|RangeRatio|VolumeTrend|VolRatio)", re.IGNORECASE),
    ),
    FamilyRule(
        label="volume_price_momentum",
        pattern=re.compile(r"(VolumePriceTrend|VolPriceMom|VolMom)", re.IGNORECASE),
    ),
    FamilyRule(label="quality_persist", pattern=re.compile(r"(QualPersist|ROEPersist|Quality)", re.IGNORECASE)),
    FamilyRule(label="value_mean_revert", pattern=re.compile(r"(ValueMR|ValueRev|PEMR|PBMR)", re.IGNORECASE)),
    FamilyRule(label="margin_trend", pattern=re.compile(r"(MarginTrend|MarginFlow|Rzye)", re.IGNORECASE)),
    FamilyRule(label="earnings_revision", pattern=re.compile(r"(EarnRev|EPSRev|ProfitRev)", re.IGNORECASE)),
    FamilyRule(label="residual_momentum", pattern=re.compile(r"(ResidMom|ResidualMom|BetaAdj)", re.IGNORECASE)),
    FamilyRule(label="liquidity_stability", pattern=re.compile(r"(LiqStab|LiqVol|TurnStab)", re.IGNORECASE)),
    FamilyRule(label="low_volatility", pattern=re.compile(r"(LowVol|LowRisk|LowBeta)", re.IGNORECASE)),
)


# ----------------------------------------------------------------- YAML loader


_DEFAULT_YAML_PATH = Path(__file__).with_name("factor_families.yaml")


def _validate_label(label: str) -> str:
    if not isinstance(label, str) or not label.strip():
        raise ValueError(f"家族 label 必须是非空字符串：{label!r}")
    if any(ch.isspace() for ch in label):
        raise ValueError(f"家族 label 不允许含空白：{label!r}")
    if "/" in label or "\\" in label:
        raise ValueError(f"家族 label 不允许含路径分隔符：{label!r}")
    return label


def _parse_rule_entry(entry: dict[str, object]) -> FamilyRule:
    if not isinstance(entry, dict):
        raise ValueError(f"规则条目必须是 mapping，实际：{type(entry).__name__}")
    label_raw = entry.get("label")
    pattern_raw = entry.get("pattern")
    if label_raw is None or pattern_raw is None:
        raise ValueError(f"规则条目缺 label 或 pattern：{entry!r}")
    if not isinstance(pattern_raw, str):
        raise ValueError(f"pattern 必须是字符串：{pattern_raw!r}")
    label = _validate_label(str(label_raw))
    try:
        pat = re.compile(pattern_raw, re.IGNORECASE)
    except re.error as exc:
        raise ValueError(f"pattern 正则非法 label={label} err={exc}") from exc
    return FamilyRule(label=label, pattern=pat)


def load_family_rules(path: Path | None = None) -> tuple[FamilyRule, ...]:
    """
    从 YAML 加载家族规则；任何故障都返回 ``DEFAULT_FAMILY_RULES``。

    Parameters
    ----------
    path : Path | None
        自定义 YAML 路径；None 表示使用内置 ``factor_families.yaml``。
    """
    yaml_path = Path(path) if path is not None else _DEFAULT_YAML_PATH

    if not yaml_path.exists():
        logger.warning("家族规则 YAML 不存在，回退到内置默认值 path=%s", yaml_path)
        return DEFAULT_FAMILY_RULES

    try:
        import yaml  # type: ignore
    except ImportError:
        logger.warning("pyyaml 不可用，回退到内置家族默认规则")
        return DEFAULT_FAMILY_RULES

    try:
        with yaml_path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("家族规则 YAML 解析失败，回退到默认值 path=%s err=%s", yaml_path, exc)
        return DEFAULT_FAMILY_RULES

    if not isinstance(payload, dict):
        logger.warning("家族规则 YAML 根节点必须是 mapping，回退到默认值 path=%s", yaml_path)
        return DEFAULT_FAMILY_RULES

    version = str(payload.get("version", "")).strip()
    if version != _SCHEMA_VERSION:
        logger.warning(
            "家族规则 YAML schema_version=%r 不等于 %r，回退到默认值",
            version,
            _SCHEMA_VERSION,
        )
        return DEFAULT_FAMILY_RULES

    raw_rules = payload.get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        logger.warning("家族规则 YAML rules 字段必须是非空列表，回退到默认值")
        return DEFAULT_FAMILY_RULES

    rules: list[FamilyRule] = []
    seen_labels: set[str] = set()
    try:
        for entry in raw_rules:
            rule = _parse_rule_entry(entry)
            if rule.label in seen_labels:
                raise ValueError(f"label 重复：{rule.label}")
            seen_labels.add(rule.label)
            rules.append(rule)
    except Exception as exc:  # noqa: BLE001
        logger.warning("家族规则 YAML 校验失败，回退到默认值 err=%s", exc)
        return DEFAULT_FAMILY_RULES

    return tuple(rules)


# ---------------------------------------------------------------------- cache


_cache_lock = threading.Lock()
_cached_rules: tuple[FamilyRule, ...] | None = None


def get_family_rules() -> tuple[FamilyRule, ...]:
    """返回进程级缓存的家族规则；首次调用时从 YAML 加载。"""
    global _cached_rules
    if _cached_rules is not None:
        return _cached_rules
    with _cache_lock:
        if _cached_rules is None:
            _cached_rules = load_family_rules()
    return _cached_rules


def reload_family_rules(path: Path | None = None) -> tuple[FamilyRule, ...]:
    """
    强制重新加载家族规则（绕过缓存）；单测 / 研究员热更新时使用。
    """
    global _cached_rules
    rules = load_family_rules(path)
    with _cache_lock:
        _cached_rules = rules
    return rules


# ----------------------------------------------------------------- classifier


def classify_family(name: str, rules: Sequence[FamilyRule] | None = None) -> str:
    """
    从因子名推导 family 标签；无命中返回 ``unknown``。

    Parameters
    ----------
    name : str
        因子名，例如 ``VolRev_5d``。
    rules : Sequence[FamilyRule] | None
        注入规则列表，默认用 ``get_family_rules()``。单测可以传自定义规则绕过缓存。
    """
    active_rules = rules if rules is not None else get_family_rules()
    for rule in active_rules:
        if rule.matches(name):
            return rule.label
    return "unknown"
