"""Point RD-Agent's Qlib conda runs at env QLIB_RDAGENT_CONDA_ENV (default: qlib_zhengshi)."""

from __future__ import annotations

import os

import rdagent.utils.env as env_mod


def apply_qlib_conda_env_patch() -> None:
    """Must run before any import of rdagent.scenarios.qlib.experiment.workspace."""
    name = os.environ.get("QLIB_RDAGENT_CONDA_ENV", "qlib_zhengshi")
    # Windows has no select.poll; disable live stream mode in LocalEnv to avoid that code path.
    env_mod.LocalConf.live_output = False
    env_mod.CondaConf.live_output = False

    class _QlibCondaConf(env_mod.QlibCondaConf):  # type: ignore[misc]
        conda_env_name: str = name

    env_mod.QlibCondaConf = _QlibCondaConf
