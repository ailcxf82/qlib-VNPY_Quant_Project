"""
基于 chanSignal 预测结果的简单 RQAlpha 策略。

策略逻辑：
1. 读取预测文件（pred_chen.csv）
2. 每日根据预测分数选择 Top K 只股票
3. 等权重分配
4. 定期调仓
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict, List, Optional

import pandas as pd

# 允许被 RQAlpha 以"脚本文件"方式加载时也能正确导入项目模块
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from rqalpha.api import *  # type: ignore
    RQALPHA_IMPORTED = True
except Exception:
    RQALPHA_IMPORTED = False

from backtest.msa.prediction_loader import load_prediction_csv, topk

logger = logging.getLogger(__name__)


def _load_context_vars(context) -> Dict:
    """从 RQAlpha 配置中加载策略参数"""
    extra_cfg = getattr(context.config, "extra", None)
    if extra_cfg and hasattr(extra_cfg, "context_vars"):
        cv = extra_cfg.context_vars
        return cv if isinstance(cv, dict) else cv.__dict__
    if isinstance(extra_cfg, dict) and "context_vars" in extra_cfg:
        return extra_cfg["context_vars"] or {}
    return {}


def _safe_plot(name: str, value):
    """安全调用 plot 函数"""
    if not RQALPHA_IMPORTED:
        return
    try:
        plot(name, value)  # type: ignore[name-defined]
    except Exception:
        pass


def init(context):
    """策略初始化"""
    logger.info("初始化 chanSignal 策略")
    
    # 从配置中读取参数
    cv = _load_context_vars(context)
    context.prediction_file = cv.get("prediction_file", "")
    context.score_col = str(cv.get("score_col", "final"))
    context.dates_are = str(cv.get("dates_are", "auto"))
    context.top_k = int(cv.get("top_k", 50))
    context.max_position = float(cv.get("max_position", 0.3))
    context.max_stock_weight = float(cv.get("max_stock_weight", 0.05))
    context.rebalance_interval_days = int(cv.get("rebalance_interval_days", 5))
    context.full_invested = bool(cv.get("full_invested", False))
    
    # 加载预测文件
    if not context.prediction_file or not os.path.exists(context.prediction_file):
        raise FileNotFoundError(f"预测文件不存在: {context.prediction_file}")
    
    logger.info(
        "加载预测文件: %s (score_col=%s, dates_are=%s)",
        context.prediction_file,
        context.score_col,
        context.dates_are,
    )
    context.pred_book = load_prediction_csv(
        context.prediction_file,
        score_col=context.score_col,
        dates_are=context.dates_are,
    )
    logger.info(f"预测文件加载完成，共 {len(context.pred_book.df)} 条记录")
    
    # 初始化状态
    context.last_rebalance_date = None
    context.target_weights = {}
    
    # 订阅基准（CSI300）
    subscribe("000300.XSHG")  # type: ignore[name-defined]


def before_trading(context):
    """每日开盘前"""
    pass


def _should_rebalance(context, today) -> bool:
    """判断是否需要调仓"""
    if context.last_rebalance_date is None:
        return True
    
    days_passed = (today - context.last_rebalance_date).days
    return days_passed >= context.rebalance_interval_days


def _build_target_weights(context, today: pd.Timestamp) -> Dict[str, float]:
    """构建目标权重"""
    # 获取当日的预测
    preds = context.pred_book.get_predictions(today)
    if not preds:
        return {}
    
    # 选择 Top K
    top_codes = topk(preds, context.top_k)
    if not top_codes:
        return {}
    
    # 等权重分配
    if context.full_invested:
        # 满仓：等权重，归一化到 100%
        weight_per_stock = 1.0 / len(top_codes)
        return {code: weight_per_stock for code in top_codes}
    else:
        # 限制仓位：等权重，但不超过 max_position
        weight_per_stock = min(
            context.max_position / len(top_codes),
            context.max_stock_weight
        )
        return {code: weight_per_stock for code in top_codes}


def _rebalance_to_target(target_weights: Dict[str, float]):
    """调仓到目标权重"""
    for code, target_weight in target_weights.items():
        try:
            order_target_percent(code, target_weight)  # type: ignore[name-defined]
        except Exception as e:
            logger.warning(f"调仓失败 {code}: {e}")


def handle_bar(context, bar_dict):
    """每个 bar 调用"""
    today = context.now.date() if hasattr(context, "now") else pd.Timestamp.now().date()
    today_ts = pd.Timestamp(today)
    
    # 绘制每日指标
    try:
        portfolio = context.portfolio
        total_value = float(getattr(portfolio, "total_value", 0.0) or 0.0)
        cash = float(getattr(portfolio, "cash", 0.0) or 0.0)
        market_value = float(getattr(portfolio, "market_value", 0.0) or 0.0)
        unit_nav = float(getattr(portfolio, "unit_net_value", 0.0) or 0.0)
        
        exposure = (market_value / total_value) if total_value > 0 else 0.0
        cash_ratio = (cash / total_value) if total_value > 0 else 0.0
        
        _safe_plot("chan/nav", unit_nav)
        _safe_plot("chan/exposure", exposure)
        _safe_plot("chan/cash_ratio", cash_ratio)
        
        # 持仓数
        try:
            positions = get_positions()  # type: ignore[name-defined]
            holding_n = sum(1 for pos in positions if float(getattr(pos, "quantity", 0) or 0) > 0)
            _safe_plot("chan/holdings", int(holding_n))
        except Exception:
            pass
    except Exception:
        pass
    
    # 判断是否需要调仓
    if _should_rebalance(context, today_ts):
        # 构建目标权重
        target_weights = _build_target_weights(context, today_ts)
        
        if target_weights:
            # 执行调仓
            _rebalance_to_target(target_weights)
            context.target_weights = target_weights
            context.last_rebalance_date = today_ts
            
            logger.info(f"调仓完成: {today}, 持仓数: {len(target_weights)}")



