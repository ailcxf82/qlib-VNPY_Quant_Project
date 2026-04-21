"""REMOVED (stage G.2) —— 请改用 ``python -m scripts.lab.run_rdagent_loop``。

原实现自阶段 E 起已搬到 ``factor_lab.runners.rdagent_loop``；阶段 F 期间保留
DeprecationWarning shim；阶段 G.2 起 import / 运行本脚本都会立刻抛 ``RuntimeError``，
以强制外部 CI / 任务链更新命令。

新命令::

    python -m scripts.lab.run_rdagent_loop --loop_n=1
"""

from __future__ import annotations

_MIGRATION_HINT = (
    "scripts/run_fin_quant.py 已在阶段 G.2 下线；"
    "请改用 `python -m scripts.lab.run_rdagent_loop`"
)

raise RuntimeError(_MIGRATION_HINT)
