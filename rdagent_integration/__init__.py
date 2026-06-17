"""rdagent_integration —— **已下线** shim package（阶段 G.2）。

自阶段 E 起所有模块已搬到 ``factor_lab.adapters.*``；阶段 F 期间保留
DeprecationWarning shim 作过渡。阶段 G.2（2026-04）起本包升级为 **硬报错**：
任何 ``import rdagent_integration`` 或子模块都会立即抛 ``RuntimeError``，以防
过渡期悄悄延长。

迁移指引（一对一替换即可，保留 import 符号名不变）：

  - ``from rdagent_integration.experiments``
        → ``from factor_lab.adapters.experiments``
  - ``from rdagent_integration.project_proposal``
        → ``from factor_lab.adapters.proposal``
  - ``from rdagent_integration.project_quant_proposal``
        → ``from factor_lab.adapters.quant_proposal``
  - ``from rdagent_integration.patch_qlib_conda``
        → ``from factor_lab.adapters.patch_qlib_conda``
  - ``python scripts/run_fin_quant.py``
        → ``python -m scripts.lab.run_rdagent_loop``

如果你在 CI/脚本里看到本条异常，请按上表更新 import 路径；无需回滚到 F 阶段。
"""

from __future__ import annotations

_MIGRATION_HINT = (
    "rdagent_integration.* 已在阶段 G.2 下线（2026-04）。请把 import 路径改成 "
    "factor_lab.adapters.*："
    "\n  - rdagent_integration.project_experiments -> factor_lab.adapters.experiments"
    "\n  - rdagent_integration.project_proposal     -> factor_lab.adapters.proposal"
    "\n  - rdagent_integration.project_quant_proposal -> factor_lab.adapters.quant_proposal"
    "\n  - rdagent_integration.patch_qlib_conda     -> factor_lab.adapters.patch_qlib_conda"
    "\n  - scripts/run_fin_quant.py                 -> python -m scripts.lab.run_rdagent_loop"
)

raise RuntimeError(_MIGRATION_HINT)
