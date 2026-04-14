"""Use rdagent_overrides templates aligned with config/data.yaml (provider_uri, csi500)."""

from __future__ import annotations

from pathlib import Path

from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment
from rdagent.scenarios.qlib.experiment.workspace import QlibFBWorkspace

_ROOT = Path(__file__).resolve().parent.parent
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
