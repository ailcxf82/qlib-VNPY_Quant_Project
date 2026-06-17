"""L2 check 的基类与上下文。

每个 check = 一个独立 callable：给定 ``CandidateFactorPackage`` + ``CheckContext`` +
check 私有配置 → 产出 ``CheckResult``。

设计：

* **纯函数式**：check 不维持状态，也不写任何磁盘（写入是 orchestrator 的职责）。
* **异常即 FAIL**：若 check 内部抛错，orchestrator 会包装成 ``CheckResult(passed=False, ...)``
  并把 exception repr 放进 detail，不影响其他 check 继续跑。
* **不可依赖网络 / LLM**：L2 必须离线可复现。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from factor_lab.exporters.schema import CandidateFactorPackage
from factor_validation.schema import CheckResult


@dataclass(frozen=True)
class CheckContext:
    """跨 check 共享的运行上下文。

    Attributes
    ----------
    universe : str
        股票池，例如 ``'csi300'``。
    oos_window : tuple[date, date]
        OOS 时间窗 [start, end]（闭区间）。
    benchmark : str
        基准指数代码，例如 ``'SH000300'``。
    project_root : Path
        项目根；check 需要读取相对路径配置时使用。
    label_parquet : Path | None
        OOS 标签 parquet（由 ``scripts/validate/prepare_oos_labels.py`` 离线生成）。
        MultiIndex ``(datetime, instrument)``，单列 ``label: float64``。
        ``ic_check`` / 未来的 ``stability_check`` 从此读 label；None 时这些 check 会 FAIL
        并给出明确错误，提示先跑 prepare_oos_labels。
    reference_factors_parquet : Path | None
        参照因子矩阵 parquet（通常指向 ``factor_registry/parquet/factors_v<N>.parquet``
        即"当前 L3 active 因子集"）。``orthogonality_check`` 从此读参照列；
        None 时 orthogonality check 会 FAIL。
    baseline_prediction_parquet : Path | None
        baseline ensemble 在 OOS 窗内的预测 parquet，MultiIndex ``(datetime, instrument)``，
        单列 ``prediction: float64``。``marginal`` check 从此读取 "当前生产模型预测" 作为
        边际贡献评估的基线。典型来源：``trainer/trainer.py`` 滚动训练产出的 OOS 预测，
        或 ``qlib.workflow.R.predict`` 导出到 parquet。
        None 时 marginal check 会 FAIL 并提示缺哪个文件。
    """

    universe: str
    oos_window: tuple[date, date]
    benchmark: str
    project_root: Path
    label_parquet: Path | None = None
    reference_factors_parquet: Path | None = None
    baseline_prediction_parquet: Path | None = None


class CheckBase(ABC):
    """所有 L2 check 的基类。"""

    #: 类变量——check 的规范名；必须与 profile yaml 里的 key 相同。
    name: str = ""

    @abstractmethod
    def run(
        self,
        candidate: CandidateFactorPackage,
        context: CheckContext,
        config: dict[str, Any],
    ) -> CheckResult:
        """执行 check。

        Parameters
        ----------
        candidate : CandidateFactorPackage
            候选因子包（契约 C1）。**不得修改**。
        context : CheckContext
            profile 内统一注入的全局上下文。
        config : dict
            该 check 在 profile 中的私有配置（不含 ``enabled``/``weight``）。
        """
