"""REMOVED (stage G.2) — see ``factor_lab.adapters.experiments``.

Any import of this module raises ``RuntimeError`` on purpose. Update to::

    from factor_lab.adapters.experiments import (
        ProjectQlibFactorExperiment,
        ProjectQlibModelExperiment,
    )
"""

from __future__ import annotations

raise RuntimeError(
    "rdagent_integration.project_experiments 已在阶段 G.2 下线；"
    "请 import factor_lab.adapters.experiments"
)
