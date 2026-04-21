"""factor_lab.adapters.experiments

Project-local ``QlibFactorExperiment`` / ``QlibModelExperiment`` 子类，使用本仓下
``rdagent_overrides/`` 的 YAML 模板（与 ``config/data.yaml`` 的 provider_uri / csi500
保持一致）。

**阶段 E 搬迁自** ``rdagent_integration/project_experiments.py`` —— 语义等价。
"""

from __future__ import annotations

from pathlib import Path

from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment
from rdagent.scenarios.qlib.experiment.workspace import QlibFBWorkspace

# 当前文件位于 factor_lab/adapters/experiments.py，项目根 = parents[2]
_ROOT = Path(__file__).resolve().parents[2]
_FACTOR_TPL = _ROOT / "rdagent_overrides" / "factor_template"
_MODEL_TPL = _ROOT / "rdagent_overrides" / "model_template"


class ProjectQlibFactorExperiment(QlibFactorExperiment):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.experiment_workspace = QlibFBWorkspace(template_folder_path=_FACTOR_TPL)


class ProjectQlibModelExperiment(QlibModelExperiment):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.experiment_workspace = QlibFBWorkspace(template_folder_path=_MODEL_TPL)
