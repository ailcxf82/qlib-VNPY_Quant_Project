"""
按日期横截面标准化工具。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterable, List


def normalize_by_date(
    df: pd.DataFrame,
    cols: Iterable[str],
    *,
    date_col: str = "date",
    mode: str = "zscore",
    eps: float = 1e-6,
) -> pd.DataFrame:
    """
    对指定列按日期分组做横截面标准化。

    参数:
        df: 含日期列的 DataFrame。若没有 date_col，则会尝试从 MultiIndex level 名为 date/datetime 提取。
        cols: 需要标准化的列名列表
        date_col: 日期列名称，默认 "date"
        mode: "demean" 或 "zscore"
        eps: 避免除零的小常数
    返回:
        新的 DataFrame（拷贝）
    """
    if isinstance(cols, str):
        cols = [cols]
    cols = list(cols)

    work = df.copy()

    # 若缺少 date_col，尝试从 MultiIndex 提取
    if date_col not in work.columns:
        if isinstance(work.index, pd.MultiIndex):
            for lvl in ["date", "datetime"]:
                if lvl in work.index.names:
                    work[date_col] = work.index.get_level_values(lvl)
                    break
        if date_col not in work.columns:
            raise ValueError(f"normalize_by_date 需要 {date_col} 列或 MultiIndex(date/datetime)")

    def _norm(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()
        mean = g[cols].mean()
        if mode.lower() == "demean":
            g[cols] = g[cols] - mean
        elif mode.lower() == "zscore":
            std = g[cols].std().replace(0, eps)
            g[cols] = (g[cols] - mean) / (std + eps)
        else:
            raise ValueError(f"未知 mode={mode}，支持 'demean' 或 'zscore'")
        return g

    work = work.groupby(date_col, group_keys=False).apply(_norm)
    return work



