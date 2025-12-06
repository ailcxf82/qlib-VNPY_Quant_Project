"""
标签转换工具：支持将收益转换为排名百分位（Rank）。
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 如果 logger 未配置，使用默认配置
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


def transform_to_rank(
    label: pd.Series,
    method: str = "percentile",
    groupby: Optional[str] = "datetime",
) -> pd.Series:
    """
    将标签转换为截面排名百分位。
    
    Args:
        label: 原始标签（收益）
        method: 转换方法
            - "percentile": 转换为 [0, 1] 的百分位（推荐）
            - "rank": 转换为排名（1 到 N）
        groupby: 分组列名，用于截面排名
            - "datetime": 按日期分组（同一日期的股票排名）
            - None: 全局排名（不推荐，因为不同时期的市场环境不同）
    
    Returns:
        转换后的标签 Series
    """
    if method == "percentile":
        if groupby == "datetime" and isinstance(label.index, pd.MultiIndex):
            # 按日期分组，计算截面排名百分位
            grouped = label.groupby(level="datetime")
            rank_pct = grouped.transform(lambda x: x.rank(pct=True, method="average"))
            logger.info("标签已转换为截面排名百分位（按日期分组）")
            return rank_pct
        elif groupby is None:
            # 全局排名
            rank_pct = label.rank(pct=True, method="average")
            logger.info("标签已转换为全局排名百分位")
            return rank_pct
        else:
            raise ValueError(f"不支持的 groupby 参数: {groupby}")
    
    elif method == "rank":
        if groupby == "datetime" and isinstance(label.index, pd.MultiIndex):
            grouped = label.groupby(level="datetime")
            rank = grouped.transform(lambda x: x.rank(method="average"))
            logger.info("标签已转换为截面排名（按日期分组）")
            return rank
        elif groupby is None:
            rank = label.rank(method="average")
            logger.info("标签已转换为全局排名")
            return rank
        else:
            raise ValueError(f"不支持的 groupby 参数: {groupby}")
    
    else:
        raise ValueError(f"不支持的转换方法: {method}")


def inverse_rank_to_return(
    rank_label: pd.Series,
    original_label: pd.Series,
    groupby: Optional[str] = "datetime",
) -> pd.Series:
    """
    将排名标签逆变换回原始收益（用于预测后还原）。
    
    注意：这是一个近似逆变换，因为排名到收益的映射不是唯一的。
    实际使用中，预测的排名可以直接用于选股，不需要逆变换。
    
    Args:
        rank_label: 排名标签（预测值）
        original_label: 原始收益标签（用于参考分布）
        groupby: 分组列名
    
    Returns:
        近似的收益值
    """
    if groupby == "datetime" and isinstance(original_label.index, pd.MultiIndex):
        # 按日期分组，使用分位数映射
        result = pd.Series(index=rank_label.index, dtype=float)
        for dt in original_label.index.get_level_values("datetime").unique():
            mask_dt = original_label.index.get_level_values("datetime") == dt
            mask_rank = rank_label.index.get_level_values("datetime") == dt
            
            if mask_dt.sum() == 0 or mask_rank.sum() == 0:
                continue
            
            orig_vals = original_label[mask_dt].values
            rank_vals = rank_label[mask_rank].values
            
            # 使用分位数映射
            result.loc[mask_rank] = np.quantile(orig_vals, rank_vals)
        
        logger.warning("排名标签已逆变换为收益（近似值），建议直接使用排名进行选股")
        return result
    else:
        # 全局映射
        result = np.quantile(original_label.values, rank_label.values)
        logger.warning("排名标签已逆变换为收益（近似值），建议直接使用排名进行选股")
        return pd.Series(result, index=rank_label.index)

