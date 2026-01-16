"""
基于 chanSignal 预测结果的 RQAlpha 回测脚本。

功能：
1. 读取 data/predictions/pred_chen.csv 预测结果
2. 使用 RQAlpha 进行回测
3. 生成买卖明细（trades_detail.csv）
4. 生成图表（包含策略净值 vs CSI300 基准）
5. 输出回测报告

用法示例：
python run_backtest_chan.py
python run_backtest_chan.py --start 2025-01-10 --end 2025-01-20
python run_backtest_chan.py --prediction data/predictions/pred_chen.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from backtest.rqalpha_backtest import run_rqalpha_backtest
from utils import load_yaml_config


def parse_args():
    parser = argparse.ArgumentParser(description="基于 chanSignal 预测结果的 RQAlpha 回测")
    parser.add_argument(
        "--rqalpha-config",
        type=str,
        default="config/rqalpha_config.yaml",
        help="RQAlpha 配置文件路径",
    )
    parser.add_argument(
        "--prediction",
        type=str,
        default="data/predictions/pred_chen.csv",
        help="预测结果文件路径（默认: data/predictions/pred_chen.csv）",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="回测起始日期（默认从预测文件中读取）",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="回测结束日期（默认从预测文件中读取）",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="策略脚本路径（默认使用内置策略）",
    )
    parser.add_argument(
        "--full-invested",
        action="store_true",
        help="满仓回测：忽略仓位/单股/行业限制，权重归一化为100%",
    )
    parser.add_argument(
        "--score-col",
        type=str,
        default="final",
        help="回测使用的预测列（默认: final，可选 lgb/gru/stack/qlib_ensemble 等）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="回测输出目录（可选，用于区分不同模型回测结果）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    logger = logging.getLogger(__name__)
    
    # 检查预测文件是否存在
    prediction_path = args.prediction
    if not os.path.isabs(prediction_path):
        prediction_path = os.path.join(project_root, prediction_path)
    
    if not os.path.exists(prediction_path):
        raise FileNotFoundError(
            f"预测文件不存在: {prediction_path}\n"
            f"请先运行 run_predict_chan.py 生成预测结果"
        )
    
    logger.info("=" * 80)
    logger.info("开始回测 chanSignal 预测结果")
    logger.info("=" * 80)
    logger.info(f"预测文件: {prediction_path}")
    
    # 加载 RQAlpha 配置
    rqalpha_config_path = args.rqalpha_config
    if not os.path.isabs(rqalpha_config_path):
        rqalpha_config_path = os.path.join(project_root, rqalpha_config_path)
    
    rqalpha_cfg = load_yaml_config(rqalpha_config_path)
    
    # 设置基准为 CSI300（000300.XSHG）
    base_config = rqalpha_cfg.get("base", {})
    if "benchmark" not in base_config or base_config.get("benchmark") != "000300.XSHG":
        logger.info("设置基准为 CSI300 (000300.XSHG)")
        if "base" not in rqalpha_cfg:
            rqalpha_cfg["base"] = {}
        rqalpha_cfg["base"]["benchmark"] = "000300.XSHG"
    
    # 策略脚本路径（默认使用内置的简单策略）
    if args.strategy is None:
        strategy_path = os.path.join(project_root, "backtest", "rqalpha_chan_strategy.py")
    else:
        strategy_path = args.strategy
        if not os.path.isabs(strategy_path):
            strategy_path = os.path.join(project_root, strategy_path)
    
    # 如果策略文件不存在，创建一个简单的策略
    if not os.path.exists(strategy_path):
        logger.info(f"策略文件不存在，将创建: {strategy_path}")
        _create_simple_strategy(strategy_path)
    
    # 执行回测
    try:
        result = run_rqalpha_backtest(
            rqalpha_config_path=rqalpha_config_path,
            prediction_path=prediction_path,
            industry_path=None,  # chanSignal 不需要行业映射
            strategy_path=strategy_path,
            full_invested=args.full_invested,
            score_col=args.score_col,
            output_dir=args.output_dir,
        )
        
        logger.info("=" * 80)
        logger.info("回测完成！")
        logger.info("=" * 80)
        logger.info("输出文件位置:")
        logger.info("  - 交易明细: data/backtest/rqalpha/trades_detail.csv")
        logger.info("  - 持仓明细: data/backtest/rqalpha/positions_detail.csv")
        logger.info("  - 回测报告: data/backtest/rqalpha/report.json")
        logger.info("  - 回测图表: data/backtest/rqalpha/rqalpha_strategy_plot.png")
        logger.info("  - 详细结果: data/backtest/rqalpha/detailed_results.json")
        
        return result
    except Exception as e:
        logger.error(f"回测失败: {e}", exc_info=True)
        raise


def _create_simple_strategy(strategy_path: str):
    """创建一个简单的策略脚本，用于读取 pred_chen.csv 并执行交易"""
    strategy_content = '''"""
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
    context.top_k = int(cv.get("top_k", 50))
    context.max_position = float(cv.get("max_position", 0.3))
    context.max_stock_weight = float(cv.get("max_stock_weight", 0.05))
    context.rebalance_interval_days = int(cv.get("rebalance_interval_days", 5))
    context.full_invested = bool(cv.get("full_invested", False))
    
    # 加载预测文件
    if not context.prediction_file or not os.path.exists(context.prediction_file):
        raise FileNotFoundError(f"预测文件不存在: {context.prediction_file}")
    
    logger.info(f"加载预测文件: {context.prediction_file}")
    context.pred_book = load_prediction_csv(context.prediction_file)
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
'''
    
    # 确保目录存在
    os.makedirs(os.path.dirname(strategy_path), exist_ok=True)
    
    # 写入策略文件
    with open(strategy_path, "w", encoding="utf-8") as f:
        f.write(strategy_content)
    
    logging.info(f"策略文件已创建: {strategy_path}")


if __name__ == "__main__":
    main()



