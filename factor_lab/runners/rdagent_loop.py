"""factor_lab.runners.rdagent_loop

RD-Agent `fin_quant` 循环的本项目入口。阶段 E 从 ``scripts/run_fin_quant.py`` 搬进
``factor_lab.runners``，让 scripts 层回归"薄壳"，便于重用（也让本循环成为 L1 正式
对外出口的一部分）。

使用方式：

    # 从 CLI：
    python -m scripts.lab.run_rdagent_loop --loop_n=1
    # 或旧路径（shim，仍可工作）：
    python scripts/run_fin_quant.py --loop_n=1

    # 从 Python：
    from factor_lab.runners.rdagent_loop import run_rdagent_loop
    run_rdagent_loop(loop_n=1)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 项目根 = 本文件 parents[2]（factor_lab/runners/rdagent_loop.py → root）
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _bootstrap_env() -> None:
    """把项目根加入 sys.path、加载 .env、chdir 到项目根。

    RD-Agent 的若干路径假设 `cwd == 项目根`，所以这里做了 ``os.chdir``。
    """
    root_str = str(_PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    os.chdir(root_str)

    try:
        from dotenv import load_dotenv
    except Exception:  # noqa: BLE001
        logger.warning("python-dotenv 不可用，跳过 .env 加载")
    else:
        load_dotenv(_PROJECT_ROOT / ".env")


def _apply_patches() -> None:
    """应用本仓对 RD-Agent 的补丁（conda env / embedding / feedback metrics 等）。"""
    from factor_lab.adapters.patch_qlib_conda import (
        apply_qlib_conda_env_patch,
        _patch_qlib_runner_env,
    )

    apply_qlib_conda_env_patch()
    _patch_qlib_runner_env()


def run_rdagent_loop(*args: Any, **kwargs: Any) -> Any:
    """执行 RD-Agent 的 ``rdagent.app.qlib_rd_loop.quant.main``，转交所有参数。"""
    _bootstrap_env()
    _apply_patches()

    from rdagent.app.qlib_rd_loop.quant import main

    return main(*args, **kwargs)
