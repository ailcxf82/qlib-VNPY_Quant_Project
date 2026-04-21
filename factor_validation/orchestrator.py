"""L2 验证编排器。

职责（见 ``docs/ARCHITECTURE_FACTOR_LAB.md`` §4）：

1. 加载 ``ValidationProfile`` yaml + 计算 ``profile_hash`` (sha256)。
2. 逐一运行 profile 启用的 check（按注册表查类），每个 check 独立 try/except，
   异常 → 标记 passed=False 并把错误写进 detail。
3. 按 ``aggregation.method`` 聚合得到 ``overall_score`` → 决定 decision。
4. 组装 ``CertifiedFactorRecord`` (契约 C2)。

**本文件不落盘**：调用方（orchestrator 的 CLI / migrate 脚本 / 未来的 auto-promote）
决定证书写哪里。

**不支持 L3 联动 check**：backtest / marginal 需要独立入口，阶段 C 再加。
"""

from __future__ import annotations

import hashlib
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from factor_lab.exporters.schema import CandidateFactorPackage
from factor_validation.checks import CheckContext, get_check
from factor_validation.schema import (
    CertifiedFactorRecord,
    CheckResult,
    Decision,
)

logger = logging.getLogger(__name__)

#: orchestrator 版本——写入 CertifiedFactorRecord.validator_version；改动协议时 bump。
VALIDATOR_VERSION = "0.1.0"

_ALLOWED_AGGREGATION_METHODS = {"weighted_sum"}
_WEIGHT_TOLERANCE = 1e-6


class ProfileError(ValueError):
    """Profile YAML 加载 / 校验失败。"""


class ValidationProfile:
    """加载后的 profile 对象（不可变）。

    本类仅做 **结构** 校验；不校验 check 配置的语义（那是各 check 的职责）。
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path).resolve()
        raw_bytes = self.path.read_bytes()
        self.profile_hash = hashlib.sha256(raw_bytes).hexdigest()

        try:
            data = yaml.safe_load(raw_bytes.decode("utf-8"))
        except yaml.YAMLError as exc:
            raise ProfileError(f"profile YAML 解析失败: {self.path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ProfileError(f"profile 根节点必须是 mapping: {self.path}")
        self._raw = data

        self._validate_structure(data, self.path)

        # 高频访问字段缓存
        self.name: str = data["profile_name"]
        self.description: str = data["description"]
        self.universe: str = data["universe"]
        self.benchmark: str = data["benchmark"]
        self.oos_window: tuple = tuple(data["oos_window"])  # noqa: UP015

        agg = data["aggregation"]
        self.aggregation_method: str = agg["method"]
        self.pass_threshold: float = float(agg["pass_threshold"])
        self.fail_threshold: float = float(agg["fail_threshold"])

        self.checks: dict[str, dict[str, Any]] = data["checks"]

        # data_sources 节为**可选**；存在时用于注入 CheckContext 里的外部 parquet 路径。
        # 每个 check 自己决定是否 require（None 时直接 FAIL 并带诊断）。
        ds = data.get("data_sources") or {}
        if not isinstance(ds, dict):
            raise ProfileError(f"{self.path}: data_sources 必须是 mapping 或缺省")
        self.data_sources: dict[str, Any] = ds

    @staticmethod
    def _validate_structure(data: dict, path: Path) -> None:
        required = ["profile_name", "description", "universe", "oos_window",
                    "benchmark", "aggregation", "checks"]
        missing = [k for k in required if k not in data]
        if missing:
            raise ProfileError(f"{path}: 缺少必须字段 {missing}")

        # profile_name 必须与文件名一致（去 .yaml）
        expected = path.stem
        if data["profile_name"] != expected:
            raise ProfileError(
                f"{path}: profile_name ({data['profile_name']!r}) 必须与文件名 ({expected!r}) 一致"
            )

        # oos_window 必须 [start, end]
        ow = data["oos_window"]
        if not isinstance(ow, list) or len(ow) != 2:
            raise ProfileError(f"{path}: oos_window 必须是 [start, end] 两项列表")

        # aggregation
        agg = data.get("aggregation")
        if not isinstance(agg, dict):
            raise ProfileError(f"{path}: aggregation 必须是 mapping")
        for k in ("method", "pass_threshold", "fail_threshold"):
            if k not in agg:
                raise ProfileError(f"{path}: aggregation 缺字段 {k!r}")
        if agg["method"] not in _ALLOWED_AGGREGATION_METHODS:
            raise ProfileError(
                f"{path}: 不支持的 aggregation.method: {agg['method']!r}；"
                f"当前仅支持 {_ALLOWED_AGGREGATION_METHODS}"
            )
        if float(agg["fail_threshold"]) > float(agg["pass_threshold"]):
            raise ProfileError(
                f"{path}: fail_threshold 必须 <= pass_threshold"
            )

        # checks
        checks = data.get("checks")
        if not isinstance(checks, dict) or not checks:
            raise ProfileError(f"{path}: checks 必须是非空 mapping")
        enabled_weights = 0.0
        enabled_count = 0
        for name, cfg in checks.items():
            if not isinstance(cfg, dict):
                raise ProfileError(f"{path}: checks[{name}] 必须是 mapping")
            if cfg.get("enabled", False):
                enabled_count += 1
                if "weight" not in cfg:
                    raise ProfileError(
                        f"{path}: checks[{name}].weight 在 enabled=true 时必填"
                    )
                w = float(cfg["weight"])
                if not 0.0 <= w <= 1.0:
                    raise ProfileError(
                        f"{path}: checks[{name}].weight 必须在 [0,1]: {w}"
                    )
                enabled_weights += w
        if enabled_count == 0:
            raise ProfileError(f"{path}: 至少要有一个 enabled check")
        if abs(enabled_weights - 1.0) > _WEIGHT_TOLERANCE:
            raise ProfileError(
                f"{path}: 启用的 check.weight 总和必须 == 1.0，实际 {enabled_weights:.6f}"
            )

    def enabled_checks(self) -> dict[str, dict[str, Any]]:
        """返回 ``{name: private_cfg}``，``private_cfg`` 不含 ``enabled``/``weight``。"""
        out: dict[str, dict[str, Any]] = {}
        for name, cfg in self.checks.items():
            if cfg.get("enabled", False):
                out[name] = {
                    k: v for k, v in cfg.items() if k not in ("enabled", "weight")
                }
        return out

    def weight_of(self, check_name: str) -> float:
        return float(self.checks[check_name]["weight"])


# ========================================================================= run


def validate_candidate(
    candidate: CandidateFactorPackage,
    profile_path: Path,
    *,
    project_root: Path,
    now: datetime | None = None,
    notes: str | None = None,
) -> CertifiedFactorRecord:
    """跑一次 L2 验证，产出 ``CertifiedFactorRecord``。

    Parameters
    ----------
    candidate : CandidateFactorPackage
        候选因子包（契约 C1）。调用方应先 ``candidate.validate_artifacts()``。
    profile_path : Path
        Profile YAML 文件路径。
    project_root : Path
        项目根，用于 CheckContext。
    now : datetime | None
        验证时间戳；None 用 ``datetime.now(UTC)``。测试注入用。
    notes : str | None
        人工备注，写入 certificate.
    """
    profile = ValidationProfile(profile_path)
    root = Path(project_root).resolve()

    def _resolve_ds_path(key: str) -> Path | None:
        raw = profile.data_sources.get(key)
        if raw is None:
            return None
        p = Path(str(raw))
        if not p.is_absolute():
            p = root / p
        return p

    context = CheckContext(
        universe=profile.universe,
        oos_window=_parse_oos_window(profile.oos_window),
        benchmark=profile.benchmark,
        project_root=root,
        label_parquet=_resolve_ds_path("label_parquet"),
        reference_factors_parquet=_resolve_ds_path("reference_factors_parquet"),
        baseline_prediction_parquet=_resolve_ds_path("baseline_prediction_parquet"),
    )

    results: list[CheckResult] = []
    for name, private_cfg in profile.enabled_checks().items():
        result = _run_single_check(name, candidate, context, private_cfg)
        results.append(result)

    overall_score = _aggregate(profile, results)
    decision = _decide(profile, results, overall_score)

    return CertifiedFactorRecord(
        factor_id=candidate.factor_id,
        candidate=candidate,
        profile_name=profile.name,
        profile_hash=profile.profile_hash,
        decision=decision,
        overall_score=overall_score,
        check_results=results,
        backtest_metrics={},
        validated_at=(now or datetime.now(timezone.utc)),
        validator_version=VALIDATOR_VERSION,
        notes=notes,
    )


def _run_single_check(
    name: str,
    candidate: CandidateFactorPackage,
    context: CheckContext,
    private_cfg: dict[str, Any],
) -> CheckResult:
    """包装单个 check：异常不中断其他 check，而是转成 passed=False。"""
    try:
        check_cls = get_check(name)
        check = check_cls()
        return check.run(candidate, context, private_cfg)
    except Exception as exc:  # noqa: BLE001 — 我们就是要拦住所有异常
        logger.warning("check %s 抛异常：%s", name, exc)
        return CheckResult(
            name=name,
            passed=False,
            score=0.0,
            threshold=None,
            detail={
                "error": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=4),
            },
            elapsed_ms=0,
        )


def _aggregate(profile: ValidationProfile, results: list[CheckResult]) -> float:
    """按 profile.aggregation.method 聚合 score。"""
    if profile.aggregation_method != "weighted_sum":
        raise ProfileError(
            f"未实现的 aggregation.method: {profile.aggregation_method!r}"
        )
    total = 0.0
    for r in results:
        w = profile.weight_of(r.name)
        s = r.score if r.score is not None else 0.0
        total += w * s
    # 数值舍入到 [0,1]（浮点误差防护）
    total = max(0.0, min(1.0, total))
    return round(total, 6)


def _decide(
    profile: ValidationProfile,
    results: list[CheckResult],
    overall_score: float,
) -> Decision:
    """按 overall + check.passed 决议；同时确保满足 CertifiedFactorRecord 的 schema 约束。"""
    all_passed = all(r.passed for r in results)
    if all_passed and overall_score >= profile.pass_threshold:
        return Decision.PASS
    if not all_passed:
        return Decision.FAIL  # schema 要求：FAIL 必须至少一个 check 未通过
    # all_passed 但 overall 低 → HOLD（即使 < fail_threshold 也不能 FAIL）
    return Decision.HOLD


def _parse_oos_window(raw: tuple) -> tuple:
    """把 oos_window 的 yaml 原值转成 (date, date)。"""
    import datetime as _dt

    def _to_date(x) -> _dt.date:
        if isinstance(x, _dt.date) and not isinstance(x, _dt.datetime):
            return x
        if isinstance(x, _dt.datetime):
            return x.date()
        if isinstance(x, str):
            return _dt.date.fromisoformat(x)
        raise ProfileError(f"oos_window 元素类型错: {x!r}")

    start, end = raw
    return (_to_date(start), _to_date(end))
