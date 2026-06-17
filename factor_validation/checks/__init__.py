"""L2 check 注册表。

新增 check 的步骤：
1. 在本包内新增 ``<name>_check.py``，继承 ``CheckBase``
2. 在本文件的 ``CHECK_REGISTRY`` 字典里登记 ``{name: CheckClass}``
3. 在 profile YAML 中用相同 ``name`` 引用

此注册表是 L2 唯一的"check 可见性"入口。orchestrator 不允许绕过它动态 import。
"""

from __future__ import annotations

from factor_validation.checks.backtest_check import BacktestCheck
from factor_validation.checks.base import CheckBase, CheckContext
from factor_validation.checks.coverage_check import CoverageCheck
from factor_validation.checks.ic_check import IcCheck
from factor_validation.checks.marginal_check import MarginalCheck
from factor_validation.checks.marginal_training_check import MarginalTrainingCheck
from factor_validation.checks.orthogonality_check import OrthogonalityCheck
from factor_validation.checks.rqalpha_backtest_check import RqalphaBacktestCheck
from factor_validation.checks.turnover_check import TurnoverCheck

#: ``name → check class`` 只读映射。
CHECK_REGISTRY: dict[str, type[CheckBase]] = {
    CoverageCheck.name: CoverageCheck,
    IcCheck.name: IcCheck,
    OrthogonalityCheck.name: OrthogonalityCheck,
    TurnoverCheck.name: TurnoverCheck,
    BacktestCheck.name: BacktestCheck,
    RqalphaBacktestCheck.name: RqalphaBacktestCheck,
    MarginalCheck.name: MarginalCheck,
    MarginalTrainingCheck.name: MarginalTrainingCheck,
}


def get_check(name: str) -> type[CheckBase]:
    """按 profile 里的 key 取 check 类，未注册时抛 ``KeyError``。"""
    if name not in CHECK_REGISTRY:
        raise KeyError(
            f"check 未注册：{name!r}；已注册：{sorted(CHECK_REGISTRY)}"
        )
    return CHECK_REGISTRY[name]


__all__ = [
    "CHECK_REGISTRY",
    "BacktestCheck",
    "CheckBase",
    "CheckContext",
    "CoverageCheck",
    "IcCheck",
    "MarginalCheck",
    "MarginalTrainingCheck",
    "OrthogonalityCheck",
    "RqalphaBacktestCheck",
    "TurnoverCheck",
    "get_check",
]
