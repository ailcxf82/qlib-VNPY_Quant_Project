"""阶段 F.3：RD-Agent RAG 静态宪法的 YAML → 文本渲染器 + Python 兜底。

事实源：``factor_lab/config/rag_constitution.yaml``
兜底：当 YAML 不可用时，使用 ``_FALLBACK_CONSTITUTION_TEXT``（本模块末尾，
与 YAML 默认渲染输出 **byte-for-byte** 一致；阶段 E.3 的原始硬编码字符串）。

公开 API：

* :func:`render_constitution`  —— YAML mapping → 最终 markdown 字符串（纯函数）
* :func:`load_constitution_text` —— 读 YAML → 渲染；任何故障回退
* :func:`get_constitution_text` —— 进程级缓存
* :func:`reload_constitution_text` —— 显式重载（单测用）

向后兼容：
* ``factor_lab.adapters.quant_proposal._STATIC_CONSTITUTION`` 通过本模块
  ``get_constitution_text()`` 懒加载得到；外部引用该符号仍可正常工作，且
  ``_PROJECT_FACTOR_RAG`` 别名继续保留。
"""

from __future__ import annotations

import logging
import math
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = "1.0"
_DEFAULT_YAML_PATH = Path(__file__).with_name("rag_constitution.yaml")


# ------------------------------------------------------------------- renderer


def _as_str_list(v: Any, field: str) -> list[str]:
    if not isinstance(v, list):
        raise ValueError(f"{field} 必须是 list，实际 {type(v).__name__}")
    out: list[str] = []
    for item in v:
        if not isinstance(item, (str, int, float)):
            raise ValueError(f"{field} 元素必须是标量，实际 {item!r}")
        out.append(str(item))
    return out


def _as_str_lines(v: Any, field: str) -> list[str]:
    """多行文本 → rstrip 后按 \\n 切分。"""
    if not isinstance(v, str):
        raise ValueError(f"{field} 必须是字符串，实际 {type(v).__name__}")
    return v.rstrip("\n").split("\n")


def _as_dict_of_lists(v: Any, field: str) -> dict[str, list[str]]:
    if not isinstance(v, dict):
        raise ValueError(f"{field} 必须是 mapping，实际 {type(v).__name__}")
    out: dict[str, list[str]] = {}
    for k, vv in v.items():
        out[str(k)] = _as_str_list(vv, f"{field}.{k}")
    return out


def render_constitution(cfg: dict[str, Any]) -> str:
    """
    把结构化宪法 YAML → 最终 markdown 字符串。

    字段校验在此一并完成；校验失败抛 ``ValueError``，调用方负责降级兜底。
    """
    if not isinstance(cfg, dict):
        raise ValueError("cfg 必须是 mapping")
    version = str(cfg.get("version", "")).strip()
    if version != _SCHEMA_VERSION:
        raise ValueError(f"rag_constitution.yaml schema_version={version!r} 不等于 {_SCHEMA_VERSION!r}")

    windows = _as_str_list(cfg.get("allowed_windows"), "allowed_windows")
    primitives = _as_str_list(cfg.get("allowed_primitives"), "allowed_primitives")
    forbidden = _as_str_list(cfg.get("forbidden_patterns"), "forbidden_patterns")
    columns = _as_dict_of_lists(cfg.get("columns"), "columns")
    columns_nan_note = _as_str_lines(cfg.get("columns_nan_note"), "columns_nan_note")

    scoring = cfg.get("scoring") or {}
    if not isinstance(scoring, dict):
        raise ValueError("scoring 必须是 mapping")
    scoring_formula = str(scoring.get("formula") or "").strip()
    if not scoring_formula:
        raise ValueError("scoring.formula 缺失")
    opt_targets = _as_str_list(scoring.get("optimization_targets"), "scoring.optimization_targets")
    lesson_lines = _as_str_lines(scoring.get("historical_lesson"), "scoring.historical_lesson")

    fu = cfg.get("feature_universe") or {}
    if not isinstance(fu, dict):
        raise ValueError("feature_universe 必须是 mapping")
    fu_name = str(fu.get("name") or "").strip()
    fu_count = int(fu.get("column_count") or 0)
    fu_groups = _as_dict_of_lists(fu.get("groups"), "feature_universe.groups")

    ortho_rules = cfg.get("orthogonality_rules") or []
    if not isinstance(ortho_rules, list):
        raise ValueError("orthogonality_rules 必须是 list")

    enc = cfg.get("encouraged_families") or []
    dis = cfg.get("discouraged_families") or []
    if not isinstance(enc, list) or not isinstance(dis, list):
        raise ValueError("encouraged_families / discouraged_families 必须是 list")

    naming = str(cfg.get("naming_convention") or "").rstrip("\n")
    if not naming:
        raise ValueError("naming_convention 缺失")

    # 阶段 I.2：必填，保证 YAML 渲染输出与 _FALLBACK_CONSTITUTION_TEXT byte-parity。
    ref_impl = cfg.get("reference_implementation")
    if not isinstance(ref_impl, str) or not ref_impl.strip():
        raise ValueError("reference_implementation 缺失")
    ref_lines = _as_str_lines(ref_impl, "reference_implementation")

    # --- 渲染 -------------------------------------------------------------
    lines: list[str] = []
    lines.append("")  # 开头空行与原硬编码保持一致
    lines.append("------Project factor hypothesis constraints (mandatory)------")
    lines.append("1) Propose at most 2 new factors per hypothesis when trace length < 8; at most 3 afterward.")
    lines.append(
        f"2) Each factor MUST use a single integer window W chosen from {{{', '.join(windows)}}} (state W explicitly)."
    )
    lines.append(f"3) Allowed primitives: {', '.join(primitives)}.")
    lines.append(f"4) FORBIDDEN: {', '.join(forbidden)}.")
    lines.append("5) Available columns in daily_pv.h5 (use EXACTLY these names, no others):")
    for cat, cols in columns.items():
        # 保持冒号后对齐，原硬编码里每行缩进 3 空格
        lines.append(f"   {cat + ':':<14}{', '.join(cols)}")
    for nan_line in columns_nan_note:
        lines.append(f"   {nan_line}")

    lines.append("")
    lines.append("------Reward / objective (P3a, mandatory reading)------")
    lines.append(f"You are scored by `{scoring_formula}`.")
    lines.append("This means three things you MUST optimise simultaneously, NOT just IC:")
    for t in opt_targets:
        lines.append(f"  \u00b7 {t}")
    lines.append("")
    for ll in lesson_lines:
        lines.append(ll)

    lines.append("")
    lines.append(f"------Existing feature universe (`{fu_name}`, {fu_count} columns)------")
    lines.append("The project already exposes these columns to LGB; new factors must add INCREMENTAL")
    lines.append("information, i.e. low correlation with these:")
    for cat, cols in fu_groups.items():
        lines.append(f"  {cat + ':':<18}{', '.join(cols)}")

    lines.append("")
    lines.append("Hard requirement (mandatory):")
    for rule in ortho_rules:
        rule_str = str(rule).rstrip("\n")
        for line in rule_str.split("\n"):
            lines.append(f"  {line}")

    lines.append("")
    lines.append("------Encouraged factor families (HIGH composite-score expectation)------")
    for item in enc:
        if not isinstance(item, dict):
            raise ValueError(f"encouraged_families 元素必须是 mapping: {item!r}")
        lines.append(f"   ({item['id']}) {item['text']}")

    lines.append("")
    lines.append("------Discouraged factor families (LOW composite-score, DO NOT propose)------")
    for item in dis:
        if not isinstance(item, dict):
            raise ValueError(f"discouraged_families 元素必须是 mapping: {item!r}")
        line = f"   ({item['id']}) {item['text']}"
        # 阶段 G.4：可选的 ``penalty`` 字段，控制是硬黑名单还是软降权。
        # 缺省 / inf → 与 F 阶段硬黑语义完全一致，不追加任何注解（保证默认 YAML 与
        # ``_FALLBACK_CONSTITUTION_TEXT`` 的 byte-for-byte 相等契约不破）。
        # 具体数值（例如 -0.5 / -1.0）→ 追加一段"软降权"注解，提示 LLM 仍可尝试
        # 但需附加显式理由。
        raw_pen = item.get("penalty")
        if raw_pen is not None:
            try:
                pen = float(raw_pen)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"discouraged_families[{item.get('id')!r}].penalty 必须是数值，实际 {raw_pen!r}"
                ) from exc
            if math.isnan(pen):
                raise ValueError(
                    f"discouraged_families[{item.get('id')!r}].penalty 不能是 NaN"
                )
            if not math.isinf(pen):
                line += (
                    f"   [penalty={pen:g}; soft — new attempts allowed only with "
                    "explicit justification of how this proposal differs]"
                )
        lines.append(line)

    lines.append("")
    lines.append(
        "------Reference implementation (copy this skeleton; only edit the 3 marked lines)------"
    )
    for rl in ref_lines:
        lines.append(rl)

    lines.append("")
    for line in naming.split("\n"):
        lines.append(line)
    lines.append("")  # 末尾空行

    return "\n".join(lines)


# --------------------------------------------------------------- YAML loading


def _load_yaml(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        logger.warning("静态宪法 YAML 不存在，使用兜底文本 path=%s", path)
        return None
    try:
        import yaml  # type: ignore
    except ImportError:
        logger.warning("pyyaml 不可用，使用兜底静态宪法")
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("静态宪法 YAML 解析失败 path=%s err=%s", path, exc)
        return None
    if not isinstance(payload, dict):
        logger.warning("静态宪法 YAML 根节点不是 mapping path=%s", path)
        return None
    return payload


def load_constitution_text(path: Path | None = None) -> str:
    """
    从 YAML 加载并渲染静态宪法文本；任何故障回退到 ``_FALLBACK_CONSTITUTION_TEXT``。
    """
    yaml_path = Path(path) if path is not None else _DEFAULT_YAML_PATH
    cfg = _load_yaml(yaml_path)
    if cfg is None:
        return _FALLBACK_CONSTITUTION_TEXT
    try:
        rendered = render_constitution(cfg)
    except Exception as exc:  # noqa: BLE001
        logger.warning("静态宪法渲染失败，使用兜底 err=%s", exc)
        return _FALLBACK_CONSTITUTION_TEXT
    return rendered


# ---------------------------------------------------------------------- cache


_cache_lock = threading.Lock()
_cached_text: str | None = None


def get_constitution_text() -> str:
    """进程级缓存；首次访问时加载。"""
    global _cached_text
    if _cached_text is not None:
        return _cached_text
    with _cache_lock:
        if _cached_text is None:
            _cached_text = load_constitution_text()
    return _cached_text


def reload_constitution_text(path: Path | None = None) -> str:
    """强制重载，绕过缓存；单测或研究员热更新时使用。"""
    global _cached_text
    text = load_constitution_text(path)
    with _cache_lock:
        _cached_text = text
    return text


# ---------------------------------------------------------------- fallback text
#
# 与 rag_constitution.yaml 渲染出的默认输出 byte-for-byte 一致（F.3 起）。
# 如果你修改 YAML 或 render_constitution，同步回写本字符串 **并** 跑
# ``tests/factor_lab/config/test_constitution.py::test_yaml_matches_fallback``。

_FALLBACK_CONSTITUTION_TEXT = """
------Project factor hypothesis constraints (mandatory)------
1) Propose at most 2 new factors per hypothesis when trace length < 8; at most 3 afterward.
2) Each factor MUST use a single integer window W chosen from {5, 10, 20, 30, 60} (state W explicitly).
3) Allowed primitives: pct_change, shift, rolling(W).mean/std/sum/min/max/corr, rank, clip.
4) FORBIDDEN: nested rolling correlations across many series, loops, \"10 pairs\" patterns.
5) Available columns in daily_pv.h5 (use EXACTLY these names, no others):
   Price/Volume: $close, $open, $high, $low, $volume, $amount
   Liquidity:    $turnover_rate, $turnover_rate_f, $volume_ratio
   Valuation:    $pe_ttm, $pb, $ps_ttm, $total_mv, $dv_ratio
   Quality:      $roe, $roa, $q_profit_yoy, $q_eps
   Technical:    $rsi12, $macd, $macd_dif, $kdj_k, $kdj_d, $atr
   Margin:       $rzye, $rqye
   NOTE: $pe_ttm/$pb/$roe/$q_profit_yoy may have NaN for some instruments (quarterly data).
         Always use .fillna(method='ffill') or rolling mean as fallback for fundamental columns.
   NOTE: Instrument index is Qlib-style: SH/SZ/BJ prefix + 6-digit code (e.g. 'SH600000', 'SZ000001', 'SZ300059'). Preserve it as-is.
         DO NOT rewrite to dotted / suffix formats like '000001.SZ', 'SZ.000001' or '600000.SH'.
         Output result.h5 MUST keep the exact same (datetime, instrument) MultiIndex as df.

------Reward / objective (P3a, mandatory reading)------
You are scored by `1day.composite_score = 1.0*IR + 2.0*IC_IR - 0.5*log(1+annualized_turnover)`.
This means three things you MUST optimise simultaneously, NOT just IC:
  \u00b7 IR  (information_ratio of excess_return_with_cost) \u2014 real backtest signal-to-noise
  \u00b7 IC_IR (Rank IC mean / IC std) \u2014 signal stability
  \u00b7 annualized_turnover \u2014 penalised; high-frequency switching hurts the score.

Empirical lesson from previous loops (csi300_RD_v2, 2025-11~2026-03):
  Adding 5 short-cycle volume-price-reversal factors (RangeRatio_10D, VolRatio_20D,
  VolumePriceTrend_10D, VolumeTrend_10D, VolRet_5D) raised LGB valid-IC from 0.148
  to 0.226 BUT the realised Sharpe DROPPED from 3.59 to 1.69 because annualized
  turnover almost doubled (4.7 -> 8.75) and the strategy beta collapsed from 1.50
  to 0.90 (signals fought each other instead of stacking). DO NOT propose more
  short-cycle volume-price-reversal factors; they are the failure family.

------Existing feature universe (`lgb_short_cycle`, 32 columns)------
The project already exposes these columns to LGB; new factors must add INCREMENTAL
information, i.e. low correlation with these:
  Returns/Momentum: RET1, MOM5, MOM10, MOM20, MOM1_5
  Volatility:       VOL5, VOL10, VOL20
  Volume ratios:    VRATIO5, VRATIO10, VRATIO20, VOLVOL
  Range:            HLRANGE, HLRANGE5, HLRANGE10
  Liquidity:        TURN, TURN_F, TURN_REL
  Valuation:        PE_TTM, PB, PS_TTM, LOG_MV
  Quality:          ROE, ROA, PROFIT_YOY
  Technical:        RSI12, MACD, KDJ_DIFF, ATR
  Margin:           MARGIN_L, MARGIN_S

Hard requirement (mandatory):
  6.a) New factor MUST satisfy |Spearman(new_factor, X)| <= 0.50 for every X above
       (when X is a strict superset, prove orthogonality through transformation).
  6.b) New factor's day-over-day cross-sectional rank auto-correlation MUST be
       >= 0.60 (slow-moving) \u2014 this directly bounds turnover.
  6.c) Avoid signals that primarily fire on the SAME day as a price jump; prefer
       lagged / smoothed transformations.

------Encouraged factor families (HIGH composite-score expectation)------
   (a) Quality persistence: rolling_mean($roe, 4 quarters) ranked vs sector
   (b) Valuation mean-reversion (slow): ($pe_ttm - rolling_median($pe_ttm, 60)) / rolling_std($pe_ttm, 60)
   (c) Margin financing trend: rolling_mean($rzye / $total_mv, 20) - rolling_mean($rzye / $total_mv, 60)
   (d) Earnings revision strength: $q_profit_yoy minus its 4-quarter rolling median (use shift)
   (e) Low-volatility quality: rank($roe) / (rank(VOL20) + 1)
   (f) Long-horizon residual momentum: pct_change($close, 60) - beta * pct_change(benchmark, 60)
   (g) Liquidity stability: 1 / rolling_std($turnover_rate / Mean($turnover_rate, 60), 20)

------Discouraged factor families (LOW composite-score, DO NOT propose)------
   (x) Short-cycle volume-price reversals on W in {5, 10}
   (y) Same-day volume spike + price reversal patterns
   (z) Anything that ranks the universe with >50% weekly turnover

------Reference implementation (copy this skeleton; only edit the 3 marked lines)------
The single most common failure is rewriting the (datetime, instrument) MultiIndex.
Use groupby(level='instrument').transform so the index is preserved intact. Do NOT
reset_index, do NOT split/rebuild instrument strings, do NOT rename index levels.

    import pandas as pd

    W = 60                                       # EDIT 1: window in {5,10,20,30,60}
    df = pd.read_hdf("daily_pv.h5", key="data")  # MultiIndex (datetime, instrument)

    # Forward-fill fundamental columns first (quarterly NaN); keep the MultiIndex.
    x = df["$pe_ttm"].groupby(level="instrument").transform(lambda s: s.ffill())
    med = x.groupby(level="instrument").transform(
        lambda s: s.rolling(W, min_periods=W).median()
    )
    std = x.groupby(level="instrument").transform(
        lambda s: s.rolling(W, min_periods=W).std()
    )
    factor = (x - med) / std                     # same MultiIndex as df -- do NOT reset_index

    out = factor.to_frame("YourFactorName_%dD" % W)  # EDIT 2: factor name
    out.to_hdf("result.h5", key="data", mode="w")    # EDIT 3: nothing else

7) Factor names must encode type and window, e.g. QualPersist_60D, ValueMR_60D,
   MarginTrend_20D, EarnRev_4Q, LowVolQual_20D, ResidMom_60D, LiqStab_20D.
"""
