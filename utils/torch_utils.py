"""
PyTorch 训练通用工具。
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_global_seed(seed: int, *, deterministic: bool = True) -> None:
    """
    固定随机种子：python / numpy / torch / cudnn。

    注意：
    - deterministic=True 会影响性能，但利于复现
    - 若未安装 torch，本函数会跳过 torch 部分
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # cudnn
        try:
            torch.backends.cudnn.deterministic = bool(deterministic)
            torch.backends.cudnn.benchmark = False if deterministic else True
        except Exception:
            pass
        # 进一步的确定性（可选）
        if deterministic and hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                # 某些算子不支持 deterministic，会抛异常；这里不强制
                pass
    except Exception:
        # torch 不可用时忽略
        return



