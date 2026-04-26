"""factor_lab.runners.rdagent_loop

RD-Agent 循环的本项目入口。阶段 E 从 ``scripts/run_fin_quant.py`` 搬进
``factor_lab.runners``，让 scripts 层回归"薄壳"，便于重用（也让本循环成为 L1 正式
对外出口的一部分）。

自 阶段 I：支持 ``mode`` 参数切换底层 RD-Agent 工作流

    * ``mode="quant"``（默认，向后兼容）
      走 ``rdagent.app.qlib_rd_loop.quant.main``，即 RD-Agent 原生的
      ``QuantRDLoop``：每轮由 LLM 自行决定是做 Factor 实验还是 Model 实验。
      适合「全链路稳定性冒烟」+「模型调参」场景。

    * ``mode="factor"``
      走 ``rdagent.app.qlib_rd_loop.factor.main``，即 RD-Agent 的
      ``FactorRDLoop``：**只做 Factor 实验**，用 LightGBM 固定基线评分，
      禁用所有 Model 假设。适合「快速迭代因子库」场景——单轮 ~7 min，
      不会被 60 min 级别的 GRU 训练拖慢。

使用方式：

    # 混合模式 1 轮（等同阶段 E / H 的旧行为）：
    python -m scripts.lab.run_rdagent_loop --loop_n=1

    # 纯 factor 模式 10 轮：
    python -m scripts.lab.run_rdagent_loop --mode=factor --loop_n=10

    # Python API：
    from factor_lab.runners.rdagent_loop import run_rdagent_loop
    run_rdagent_loop(loop_n=10, mode="factor")

环境变量（同时配合补丁 ``_patch_cap_n_epochs``）：
    * ``FACTOR_LAB_MAX_N_EPOCHS``: 上限 LLM 给神经网络模型提议的 n_epochs，
      防止 qrun 命中 RD-Agent 内置 3600s 硬超时。仅在 ``mode="quant"`` 生效。
      未设置或 ``<=0`` 时不裁剪。
    * ``RUNNING_TIMEOUT_PERIOD``: 透传给 RD-Agent 的 qrun 超时（秒）。
      默认 3600。写到 launcher 脚本里即可。
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_SUPPORTED_MODES = ("quant", "factor")


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


def _resolve_entry(mode: str):
    """按 ``mode`` 返回 RD-Agent 的 ``main`` 入口函数。"""
    if mode == "quant":
        from rdagent.app.qlib_rd_loop.quant import main as _main
        return _main
    if mode == "factor":
        from rdagent.app.qlib_rd_loop.factor import main as _main
        return _main
    raise ValueError(
        f"run_rdagent_loop: unsupported mode={mode!r}. "
        f"Expected one of {_SUPPORTED_MODES}."
    )


def run_rdagent_loop(
    *args: Any,
    mode: Literal["quant", "factor"] = "factor",
    loop_n: int = 10,
    **kwargs: Any,
) -> Any:
    """执行 RD-Agent 循环；``mode`` 控制底层走 Quant/Factor 两种 workflow 之一。

    Parameters
    ----------
    mode:
        * ``"factor"``（默认）: 纯 Factor 循环，速度快，无模型训练。
        * ``"quant"``: Factor + Model 混合循环（RD-Agent 原生 QuantRDLoop）。
    loop_n:
        循环轮数，默认 10。传 ``None`` 则无限运行（需手动 Ctrl+C）。
    *args, **kwargs:
        透传给底层 ``main`` 函数。支持的 kwargs 包括 ``step_n``、
        ``path``、``all_duration``、``checkout`` 等，详见 RD-Agent 同名模块。
    """
    _bootstrap_env()
    _apply_patches()

    if mode not in _SUPPORTED_MODES:
        raise ValueError(
            f"run_rdagent_loop: unsupported mode={mode!r}. "
            f"Expected one of {_SUPPORTED_MODES}."
        )
    logger.info("run_rdagent_loop: mode=%s loop_n=%s args=%s kwargs=%s", mode, loop_n, args, kwargs)

    entry = _resolve_entry(mode)
    return entry(*args, loop_n=loop_n, **kwargs)
