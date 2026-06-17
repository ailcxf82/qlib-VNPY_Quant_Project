"""REMOVED (stage G.2) — see ``factor_lab.adapters.patch_qlib_conda``.

Any import of this module raises ``RuntimeError`` on purpose. Update to::

    from factor_lab.adapters.patch_qlib_conda import (
        apply_qlib_conda_env_patch,
        _patch_qlib_runner_env,
    )
"""

from __future__ import annotations

raise RuntimeError(
    "rdagent_integration.patch_qlib_conda 已在阶段 G.2 下线；"
    "请 import factor_lab.adapters.patch_qlib_conda"
)
