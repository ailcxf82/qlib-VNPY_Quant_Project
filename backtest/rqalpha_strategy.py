"""
RQAlpha 策略脚本：基于预测信号执行真实 T+1 交易回测。

策略逻辑：
1. 每日开盘前读取当日预测信号
2. 根据信号构建目标组合（Top-K + 权重分配）
3. 计算调仓量并执行交易（考虑手续费、滑点）
4. T+1 交易规则：当日买入，次日可卖出
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
logger = logging.getLogger(__name__)

    
try:
    from rqalpha.api import *
    from rqalpha.apis import *
    #from rqalpha.utils import *
        # order_target_percent,
        
        # get_positions,
        # get_open_orders,
        # subscribe,
        # plot,


        # get_universe,
        # # unsubscribe,
        # # get_position,
        # logger,
        # context,
        # config,
        # order_value,
        # get_current_bar_dict,
    
    RQALPHA_IMPORTED = True
except ImportError:
    # 如果 RQAlpha 未安装，提供占位实现
    
    def order_target_percent(*args, **kwargs):
        pass
    
    def get_position(*args):
        class Position:
            quantity = 0
            market_value = 0
        return Position()

    def get_positions():
        return []
    
    def get_universe():
        return []
    
    def subscribe(*args):
        pass
    
    def plot(*args, **kwargs):
        pass
    
    context = None
    config = None
    DEFAULT_ACCOUNT_TYPE = None
    RQALPHA_IMPORTED = False


def get_account():
    """
    获取股票账户对象。
    
    根据 RQAlpha 文档，账户信息通过 context.portfolio.stock_account 获取。
    这是一个辅助函数，提供类似原生 API 的接口。
    
    参考: https://rqalpha.readthedocs.io/zh-cn/latest/api/base_api.html
    
    注意: 此函数必须在策略函数（init, before_trading, handle_bar, after_trading）内部调用，
    因为需要访问 context 对象。
    
    返回:
        StockAccount 或 AccountWrapper: 股票账户对象，包含 cash, total_value, market_value 等属性
    """
    if not RQALPHA_IMPORTED:
        # RQAlpha 未安装，返回占位实现
        class Account:
            cash = 0.0
            total_value = 0.0
            market_value = 0.0
        return Account()
    
    # 在策略函数中，context 是作为参数传入的
    # 我们需要从执行上下文中获取当前的 context
    try:
        from rqalpha.core.execution_context import ExecutionContext
        exec_ctx = ExecutionContext.get_instance()
        if exec_ctx and hasattr(exec_ctx, 'user_context'):
            current_context = exec_ctx.user_context
        else:
            # 如果无法从 ExecutionContext 获取，尝试使用全局 context（可能不可靠）
            current_context = context
    except Exception:
        # 如果无法获取 context，返回占位对象
        class Account:
            cash = 0.0
            total_value = 0.0
            market_value = 0.0
        return Account()
    
    if current_context is None:
        class Account:
            cash = 0.0
            total_value = 0.0
            market_value = 0.0
        return Account()
    
    try:
        portfolio = current_context.portfolio
        
        # 方法1: 直接通过 stock_account 属性获取（推荐方式）
        if hasattr(portfolio, 'stock_account'):
            stock_account = portfolio.stock_account
            if stock_account is not None:
                return stock_account
        
        # 方法2: 通过 accounts 字典获取
        if hasattr(portfolio, 'accounts'):
            accounts = portfolio.accounts
            if accounts:
                # 尝试获取股票账户
                if DEFAULT_ACCOUNT_TYPE and hasattr(DEFAULT_ACCOUNT_TYPE, 'STOCK'):
                    stock_account = accounts.get(DEFAULT_ACCOUNT_TYPE.STOCK, None)
                    if stock_account:
                        return stock_account
                # 如果没有找到，返回第一个账户
                if accounts:
                    return list(accounts.values())[0]
        
        # 方法3: 创建一个包装对象，从 portfolio 获取属性
        class AccountWrapper:
            """账户包装类，从 portfolio 获取账户信息"""
            def __init__(self, portfolio):
                self._portfolio = portfolio
            
            @property
            def cash(self):
                """现金余额"""
                # 优先从 stock_account 获取
                if hasattr(self._portfolio, 'stock_account') and self._portfolio.stock_account:
                    return getattr(self._portfolio.stock_account, 'cash', 0.0)
                # 备用：从 portfolio 直接获取
                return getattr(self._portfolio, 'cash', 0.0)
            
            @property
            def total_value(self):
                """总资产（现金 + 持仓市值）"""
                return getattr(self._portfolio, 'total_value', 0.0)
            
            @property
            def market_value(self):
                """持仓市值"""
                return getattr(self._portfolio, 'market_value', 0.0)
        
        return AccountWrapper(portfolio)
        
    except Exception as e:
        if logger:
            logger.debug(f"获取账户失败: {e}，返回占位对象")
        # 返回占位对象
        class Account:
            cash = 0.0
            total_value = 0.0
            market_value = 0.0
        return Account()


def init(context):
    """策略初始化。"""
    # 从配置读取预测信号文件路径
    # RQAlpha 通过 extra.context_vars 传递自定义参数
    strategy_params = None
    extra_cfg = getattr(context.config, "extra", None)
    if extra_cfg and hasattr(extra_cfg, "context_vars"):
        strategy_params = extra_cfg.context_vars
    elif extra_cfg and isinstance(extra_cfg, dict) and "context_vars" in extra_cfg:
        strategy_params = extra_cfg["context_vars"]
    
    def _sp_get(key, default=None):
        if strategy_params is None:
            return default
        if isinstance(strategy_params, dict):
            return strategy_params.get(key, default)
        return getattr(strategy_params, key, default)
    
    # 获取预测文件路径
    prediction_file = _sp_get("prediction_file") or getattr(context.config, "prediction_file", None)
    
    if not prediction_file:
        raise ValueError("预测文件路径未配置，请通过 prediction_file 参数指定")
    
    if not os.path.exists(prediction_file):
        raise FileNotFoundError(f"预测文件不存在: {prediction_file}")
    
    # 加载预测信号
    df = pd.read_csv(prediction_file)
    
    # ==================== 关键：检测并修复日期偏移问题 ====================
    # 如果预测文件包含 _meta_shifted_next_day=1，说明日期已经偏移到 trade_date
    # 但 RQAlpha 使用 next_bar 模式，会导致实际使用 T+2 数据（严重数据泄露）
    # 需要将 trade_date 反向还原为 signal_date
    shifted_flag = False
    if "_meta_shifted_next_day" in df.columns:
        try:
            shifted_flag = int(pd.to_numeric(df["_meta_shifted_next_day"], errors="coerce").fillna(0).max()) == 1
        except Exception:
            shifted_flag = False
    
    if shifted_flag:
        # 使用 print 确保输出可见（logger 可能被过滤）
        print("=" * 60)
        print("⚠️ 检测到预测文件已偏移到 trade_date (_meta_shifted_next_day=1)")
        print("⚠️ RQAlpha 使用 next_bar 模式，会导致 T+2 数据泄露！")
        print("⚠️ 正在将日期反向还原为 signal_date...")
        print("=" * 60)
        if logger:
            logger.warning("=" * 60)
            logger.warning("⚠️ 检测到预测文件已偏移到 trade_date (_meta_shifted_next_day=1)")
            logger.warning("⚠️ RQAlpha 使用 next_bar 模式，会导致 T+2 数据泄露！")
            logger.warning("⚠️ 正在将日期反向还原为 signal_date...")
            logger.warning("=" * 60)
        
        # 获取唯一日期并构建交易日历
        df["datetime"] = pd.to_datetime(df["datetime"])
        unique_dts = pd.Index(df["datetime"].unique()).sort_values()
        
        # 构建交易日历
        cal = None
        try:
            from qlib.data import D
            min_dt = pd.Timestamp(unique_dts.min()).normalize()
            max_dt = pd.Timestamp(unique_dts.max()).normalize()
            cal = D.calendar(start_time=min_dt - pd.Timedelta(days=60), end_time=max_dt + pd.Timedelta(days=5), freq="day")
            cal = pd.to_datetime(list(cal)).normalize()
            cal = pd.Index(cal).sort_values().unique()
        except Exception:
            cal = pd.bdate_range(start=unique_dts.min() - pd.Timedelta(days=60), 
                                end=unique_dts.max() + pd.Timedelta(days=5)).normalize()
        
        # 映射：trade_date -> signal_date（前一交易日）
        import bisect
        cal_list = list(cal)
        mapping_prev = {}
        for d in unique_dts:
            i = bisect.bisect_left(cal_list, pd.Timestamp(d))
            if i <= 0:
                mapping_prev[pd.Timestamp(d)] = None
            else:
                mapping_prev[pd.Timestamp(d)] = pd.Timestamp(cal_list[i - 1])
        
        # 反向还原日期
        before_count = len(df)
        df["datetime"] = df["datetime"].map(lambda x: mapping_prev.get(pd.Timestamp(x), None))
        df = df.dropna(subset=["datetime"])
        after_count = len(df)
        
        if logger:
            logger.info(f"日期反向还原完成: {before_count} -> {after_count} 行（丢弃 {before_count - after_count} 行）")
    
    # 检查文件格式：可能是原始格式（datetime, instrument, final）或已转换格式（datetime, rq_code, final）
    if "rq_code" in df.columns:
        # 已转换格式
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index(["datetime", "rq_code"], inplace=True)
        signal_col = "final"
    else:
        # 原始格式，需要转换代码
        df["datetime"] = pd.to_datetime(df["datetime"])
        
        # 转换为 RQAlpha 格式的股票代码（如 SH600000 -> 600000.XSHG）
        def convert_code(instrument: str) -> str:
            """将 Qlib 格式代码转换为 RQAlpha 格式。"""
            if instrument.startswith("SH"):
                return f"{instrument[2:]}.XSHG"
            elif instrument.startswith("SZ"):
                return f"{instrument[2:]}.XSHE"
            else:
                return instrument
        
        df["rq_code"] = df["instrument"].apply(convert_code)
        df.set_index(["datetime", "rq_code"], inplace=True)
        signal_col = "final"
    
    # 存储预测信号（按日期索引）
    context.prediction_signals = {}
    for dt in df.index.get_level_values("datetime").unique():
        try:
            signals = df.xs(dt)[signal_col].to_dict()
            context.prediction_signals[dt.date()] = signals
        except Exception as e:
            logger.warning(f"日期 {dt} 信号加载失败: {e}")
            continue
    
    # 组合构建参数（从配置读取）
    strategy_params = None
    extra_cfg = getattr(context.config, "extra", None)
    if extra_cfg and hasattr(extra_cfg, "context_vars"):
        strategy_params = extra_cfg.context_vars
    elif extra_cfg and isinstance(extra_cfg, dict) and "context_vars" in extra_cfg:
        strategy_params = extra_cfg["context_vars"]
    
    if strategy_params:
        context.portfolio_config = {
            "max_position": _sp_get("max_position", 0.3),
            "max_stock_weight": _sp_get("max_stock_weight", 0.05),
            "max_industry_weight": _sp_get("max_industry_weight", 0.2),
            "top_k": _sp_get("top_k", 50),
            "full_invested": bool(_sp_get("full_invested", False)),
        }
        context.industry_map = _sp_get("industry_map")
    else:
        # 兼容直接传递的方式
        context.portfolio_config = {
            "max_position": getattr(context.config, "max_position", 0.3),
            "max_stock_weight": getattr(context.config, "max_stock_weight", 0.05),
            "max_industry_weight": getattr(context.config, "max_industry_weight", 0.2),
            "top_k": getattr(context.config, "top_k", 50),
            "full_invested": bool(getattr(context.config, "full_invested", False)),
        }
        context.industry_map = getattr(context.config, "industry_map", None)
    
    # 订阅所有可能交易的股票
    all_codes = set()
    for signals in context.prediction_signals.values():
        all_codes.update(signals.keys())
    if all_codes:
        subscribe(list(all_codes))
    
    # 初始化调仓标记与频率（默认每日，可通过 context_vars.rebalance_interval 配置）
    context.target_weights = {}
    context.rebalanced = False
    context.last_rebalance_date = None
    
    # 从配置中读取调仓频率
    rebalance_interval = _sp_get("rebalance_interval", 5)
    if rebalance_interval is None or rebalance_interval < 1:
        rebalance_interval = 5
    context.rebalance_interval = int(rebalance_interval)
    
    # ==================== 新增：卖出优化配置 ====================
    # 卖出阈值：当前预测值低于买入预测值的比例时才卖出
    # 例如：sell_threshold=0.8 表示当前预测值 < 买入预测值 * 0.8 时才卖出
    context.sell_threshold = float(_sp_get("sell_threshold", 0.5))
    
    # 最小持仓天数：避免频繁交易
    context.min_holding_days = int(_sp_get("min_holding_days", 5))
    
    # 是否启用智能卖出（基于预测值下降）
    context.smart_sell_enabled = bool(_sp_get("smart_sell_enabled", True))
    
    # ========== CSI101 专用优化参数 ==========
    # 预测值下降阈值：从高点下降超过此比例才考虑卖出
    context.prediction_decline_threshold = float(_sp_get("prediction_decline_threshold", 0.2))
    
    # 高预测值保护：预测值高于此阈值时不主动卖出
    context.high_prediction_protection = float(_sp_get("high_prediction_protection", 0.6))
    
    # 止盈止损参数
    context.take_profit_ratio = float(_sp_get("take_profit_ratio", 0.3))
    context.stop_loss_ratio = float(_sp_get("stop_loss_ratio", -0.15))
    
    # 波动率调整参数
    context.volatility_adjusted_sell = bool(_sp_get("volatility_adjusted_sell", False))
    context.volatility_window = int(_sp_get("volatility_window", 20))
    context.high_volatility_threshold_multiplier = float(_sp_get("high_volatility_threshold_multiplier", 0.7))
    
    # 记录每只股票的买入信息：{股票代码: {"date": 买入日期, "pred": 买入预测值, "weight": 权重, "high_pred": 最高预测值}}
    context.buy_info = {}
    
    # 记录回测结束标记
    context.backtest_ended = False
    
    if logger:
        logger.info(f"调仓频率设置为: 每 {context.rebalance_interval} 天调仓一次")
        logger.info(f"卖出优化: 阈值={context.sell_threshold}, 最小持仓天数={context.min_holding_days}, 智能卖出={context.smart_sell_enabled}")
        logger.info(f"CSI101优化: 预测下降阈值={context.prediction_decline_threshold}, 高预测保护={context.high_prediction_protection}")
        logger.info(f"止盈止损: 止盈={context.take_profit_ratio}, 止损={context.stop_loss_ratio}")
    
    # 保存初始资金用于计算净值
    try:
        # 尝试从配置中获取初始资金
        base_config = getattr(context.config, "base", None)
        if base_config:
            accounts = getattr(base_config, "accounts", {})
            if isinstance(accounts, dict):
                stock_account = accounts.get("stock", {})
                if isinstance(stock_account, (int, float)):
                    context.initial_cash = float(stock_account)
                elif isinstance(stock_account, dict):
                    context.initial_cash = float(stock_account.get("initial_cash", 10000000))
                else:
                    context.initial_cash = 10000000.0
            else:
                context.initial_cash = 10000000.0
        else:
            context.initial_cash = 10000000.0
    except Exception:
        context.initial_cash = 10000000.0
    
    # 检查账户初始化
    try:
        account = get_account()
        total_value = getattr(context.portfolio, "total_value", 0.0)
        if logger:
            logger.info(f"账户初始化检查 - 总资产: {total_value:.2f}, 初始资金: {context.initial_cash:.2f}, 账户对象: {account}")
            if total_value == 0:
                logger.warning("警告：账户总资产为 0，可能账户未正确初始化")
    except Exception as e:
        if logger:
            logger.error(f"账户初始化检查失败: {e}")
    
    if logger:
        logger.info(
            "策略初始化完成，预测信号日期范围: %s 至 %s",
            min(context.prediction_signals.keys()),
            max(context.prediction_signals.keys()),
        )
        logger.info(f"组合参数: {context.portfolio_config}")


def before_trading(context):
    """每日开盘前执行：构建当日目标组合。"""
    current_date = context.now.date()
    
    # 检查账户状态
    try:
        account = get_account()
        total_value = getattr(context.portfolio, "total_value", 0.0)
        cash = getattr(account, "cash", 0.0) if account else 0.0
        if logger and current_date == min(context.prediction_signals.keys()):
            # 只在第一天记录详细账户信息
            logger.info(f"开盘前账户检查 - 日期: {current_date}, 总资产: {total_value:.2f}, 现金: {cash:.2f}")
            if total_value == 0:
                logger.error("错误：账户总资产为 0，账户未正确初始化！")
    except Exception as e:
        if logger:
            logger.error(f"账户检查失败: {e}")
    
    # 获取当日预测信号
    if current_date not in context.prediction_signals:
        if logger:
            logger.warning(f"日期 {current_date} 无预测信号，跳过调仓")
        context.target_weights = {}
        return
    
    signals = context.prediction_signals[current_date]
    if not signals:
        if logger:
            logger.warning(f"日期 {current_date} 信号为空，跳过调仓")
        context.target_weights = {}
        return
    
    # 构建目标组合权重
    target_weights = build_portfolio(
        signals,
        context.portfolio_config,
        getattr(context, "industry_map", None),
    )
    
    # ==================== 智能卖出优化（CSI101增强版）====================
    # 如果启用智能卖出，检查当前持仓是否应该保留
    smart_sell_enabled = getattr(context, "smart_sell_enabled", True)
    sell_threshold = getattr(context, "sell_threshold", 0.5)
    min_holding_days = getattr(context, "min_holding_days", 5)
    buy_info = getattr(context, "buy_info", {})
    
    # CSI101 专用参数
    prediction_decline_threshold = getattr(context, "prediction_decline_threshold", 0.2)
    high_prediction_protection = getattr(context, "high_prediction_protection", 0.6)
    take_profit_ratio = getattr(context, "take_profit_ratio", 0.3)
    stop_loss_ratio = getattr(context, "stop_loss_ratio", -0.15)
    volatility_adjusted_sell = getattr(context, "volatility_adjusted_sell", False)
    
    # 获取当前持仓信息
    current_positions = set()
    position_weights = {}
    position_pnl = {}
    try:
        positions = get_positions()
        total_value = getattr(context.portfolio, "total_value", 0.0)
        for pos in positions:
            if hasattr(pos, "order_book_id") and getattr(pos, "quantity", 0) > 0:
                code = pos.order_book_id
                current_positions.add(code)
                market_value = getattr(pos, "market_value", 0)
                if total_value > 0:
                    position_weights[code] = market_value / total_value
                # 计算盈亏比例
                avg_price = getattr(pos, "avg_price", 0)
                last_price = getattr(pos, "last_price", 0)
                if avg_price > 0 and last_price > 0:
                    position_pnl[code] = (last_price - avg_price) / avg_price
    except Exception:
        pass
    
    if smart_sell_enabled and current_positions:
        # 检查每只持仓股票是否应该保留
        stocks_to_keep = set()
        stocks_to_sell = set()
        
        for code in current_positions:
            current_pred = signals.get(code, 0.0)
            
            # 如果股票在目标组合中，保留
            if code in target_weights:
                stocks_to_keep.add(code)
                # 更新最高预测值
                if code in buy_info:
                    if "high_pred" not in buy_info[code]:
                        buy_info[code]["high_pred"] = buy_info[code].get("pred", current_pred)
                    buy_info[code]["high_pred"] = max(buy_info[code]["high_pred"], current_pred)
                continue
            
            # 检查买入信息
            if code in buy_info:
                buy_pred = buy_info[code].get("pred", 1.0)
                buy_date = buy_info[code].get("date")
                high_pred = buy_info[code].get("high_pred", buy_pred)
                holding_days = (current_date - buy_date).days if buy_date else 0
                pnl_ratio = position_pnl.get(code, 0.0)
                
                # 更新最高预测值
                if "high_pred" not in buy_info[code]:
                    buy_info[code]["high_pred"] = buy_pred
                buy_info[code]["high_pred"] = max(buy_info[code]["high_pred"], current_pred)
                high_pred = buy_info[code]["high_pred"]
                
                # 判断是否应该卖出
                should_sell = False
                sell_reason = ""
                
                # ========== CSI101 优化：高预测值保护 ==========
                # 如果当前预测值仍然很高，不卖出
                if current_pred >= high_prediction_protection:
                    stocks_to_keep.add(code)
                    if code in position_weights:
                        target_weights[code] = position_weights[code]
                    if logger:
                        logger.debug(f"高预测保护: {code} - 当前预测值={current_pred:.4f} >= {high_prediction_protection}")
                    continue
                
                # ========== CSI101 优化：预测值下降阈值 ==========
                # 计算预测值从高点的下降幅度
                pred_decline = (high_pred - current_pred) / high_pred if high_pred > 0 else 0
                
                # ========== 止损检查 ==========
                if pnl_ratio <= stop_loss_ratio:
                    should_sell = True
                    sell_reason = f"触发止损 (盈亏={pnl_ratio*100:.2f}% <= {stop_loss_ratio*100:.2f}%)"
                
                # ========== 止盈检查 ==========
                elif pnl_ratio >= take_profit_ratio:
                    # 如果盈利达到止盈线，且预测值下降，则卖出
                    if pred_decline >= prediction_decline_threshold:
                        should_sell = True
                        sell_reason = f"止盈且预测下降 (盈亏={pnl_ratio*100:.2f}%, 预测下降={pred_decline*100:.2f}%)"
                    else:
                        # 盈利但预测值仍高，继续持有
                        stocks_to_keep.add(code)
                        if code in position_weights:
                            target_weights[code] = position_weights[code]
                        continue
                
                # ========== 预测值下降检查 ==========
                elif pred_decline >= prediction_decline_threshold and current_pred < buy_pred * sell_threshold:
                    should_sell = True
                    sell_reason = f"预测值大幅下降 (从{high_pred:.4f}降至{current_pred:.4f}, 下降{pred_decline*100:.2f}%)"
                
                # ========== 原有逻辑：预测值低于阈值 ==========
                elif current_pred < buy_pred * sell_threshold:
                    # CSI101 优化：只有持仓天数足够长才卖出
                    if holding_days >= min_holding_days:
                        should_sell = True
                        sell_reason = f"预测值下降且持仓足够 ({current_pred:.4f} < {buy_pred:.4f} * {sell_threshold}, 持仓{holding_days}天)"
                    else:
                        # 持仓时间不够，继续观察
                        stocks_to_keep.add(code)
                        if code in position_weights:
                            target_weights[code] = position_weights[code]
                        if logger:
                            logger.debug(f"持仓不足{min_holding_days}天，继续观察: {code}")
                        continue
                
                # ========== 原有逻辑：持仓天数足够且预测值低 ==========
                elif holding_days >= min_holding_days and current_pred < 0.4:
                    should_sell = True
                    sell_reason = f"持仓{holding_days}天且预测值很低 ({current_pred:.4f})"
                
                if should_sell:
                    stocks_to_sell.add(code)
                    if logger:
                        logger.info(f"智能卖出: {code} - {sell_reason}")
                else:
                    # 保留持仓，维持原有权重
                    stocks_to_keep.add(code)
                    if code in position_weights:
                        target_weights[code] = position_weights[code]
                    if logger:
                        logger.debug(f"保留持仓: {code} - 当前预测值={current_pred:.4f}, 买入预测值={buy_pred:.4f}, 高点预测值={high_pred:.4f}, 持仓天数={holding_days}, 盈亏={pnl_ratio*100:.2f}%")
            else:
                # 没有买入信息，如果预测值较高则保留
                if current_pred >= high_prediction_protection:
                    stocks_to_keep.add(code)
                    if code in position_weights:
                        target_weights[code] = position_weights[code]
                elif current_pred >= 0.5:
                    stocks_to_keep.add(code)
                    if code in position_weights:
                        target_weights[code] = position_weights[code]
                else:
                    stocks_to_sell.add(code)
        
        if logger and stocks_to_sell:
            logger.info(f"智能卖出: {len(stocks_to_sell)} 只股票将被卖出")
        if logger and stocks_to_keep:
            logger.info(f"智能保留: {len(stocks_to_keep)} 只股票不在目标组合但被保留")
    
    # 调仓频率控制：间隔 rebalance_interval 天才调一次仓
    interval = getattr(context, "rebalance_interval", 5) or 5
    last_date = getattr(context, "last_rebalance_date", None)
    delta_days = (current_date - last_date).days if last_date else None
    should_rebalance = (last_date is None) or (delta_days >= interval)
    
    if should_rebalance:
        context.target_weights = target_weights
        context.rebalanced = False
        context.need_rebalance = True
        if logger:
            logger.info(
                f"日期 {current_date}，目标组合 {len(target_weights)} 只股票，"
                f"距上次调仓 {0 if last_date is None else delta_days} 天，执行调仓"
            )
    else:
        context.target_weights = {}
        context.rebalanced = True
        context.need_rebalance = False
        if logger:
            logger.info(
                f"日期 {current_date}，距上次调仓 {delta_days} 天 < 间隔 {interval} 天，跳过调仓"
            )


def handle_bar(context, bar_dict):
    """
    handle_bar 在每根 K 线更新时触发，为避免重复调仓，仅在每日首次满足条件时执行。
    """
    current_date = context.now.date()
    
    # 绘制净值曲线（必须在每个 bar 都调用，不能只在调仓时调用）
    # 计算策略净值（使用 RQAlpha 的 unit_net_value）
    try:
        # 优先使用 portfolio 的 unit_net_value（这是 RQAlpha 的标准方式）
        portfolio = context.portfolio
        if hasattr(portfolio, "unit_net_value"):
            nav = portfolio.unit_net_value
            plot("strategy_nav", nav)
        else:
            # 如果没有 unit_net_value，手动计算
            total_value = getattr(portfolio, "total_value", 0.0)
            initial_cash = getattr(context, "initial_cash", 10000000.0)
            if initial_cash > 0 and total_value > 0:
                nav = total_value / initial_cash
                plot("strategy_nav", nav)
    except Exception as e:
        if logger:
            logger.debug(f"计算策略净值失败: {e}")
    
    # 计算基准净值（如果有基准）
    try:
        benchmark_portfolio = getattr(context, "benchmark_portfolio", None)
        if benchmark_portfolio:
            benchmark_nav = getattr(benchmark_portfolio, "unit_net_value", None)
            if benchmark_nav is not None:
                plot("benchmark_nav", benchmark_nav)
    except Exception as e:
        if logger:
            logger.debug(f"计算基准净值失败: {e}")
    
    # 调仓逻辑：仅在满足条件时执行
    if (
        not getattr(context, "need_rebalance", False)
        or getattr(context, "rebalanced", False)
        or not getattr(context, "target_weights", None)
    ):
        return
    
    # 仅在该交易日首次调仓
    if context.last_rebalance_date == current_date:
        return
    
    rebalance_portfolio(context, bar_dict)
    context.rebalanced = True
    context.need_rebalance = False
    context.last_rebalance_date = current_date


def build_portfolio(
    signals: Dict[str, float],
    config: Dict,
    industry_map: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """
    根据预测信号构建组合权重。
    
    参数:
        signals: {股票代码: 信号值}
        config: 组合配置参数
        industry_map: {股票代码: 行业名称}，可选
    
    返回:
        {股票代码: 目标权重}
    """
    # 1. 排序并取 Top-K
    sorted_signals = sorted(signals.items(), key=lambda x: x[1], reverse=True)
    top_k = config["top_k"]
    filtered = dict(sorted_signals[:top_k])
    
    if not filtered:
        return {}
    
    # 2. 线性衰减权重
    ranks = list(range(1, len(filtered) + 1))
    weights = {}
    for i, (code, signal) in enumerate(filtered.items()):
        weight = (top_k - ranks[i] + 1) / top_k
        weights[code] = weight

    # 满仓模式：忽略仓位/单股/行业约束，权重归一化为 100%
    if config.get("full_invested", False):
        total_weight = sum(weights.values())
        if total_weight > 0:
            return {code: w / total_weight for code, w in weights.items()}
        return {}
    
    # 3. 归一化并应用仓位限制
    total_weight = sum(weights.values())
    if total_weight > 0:
        max_position = config["max_position"]
        weights = {code: w / total_weight * max_position for code, w in weights.items()}
    
    # 4. 单股权重裁剪
    max_stock_weight = config["max_stock_weight"]
    weights = {code: min(w, max_stock_weight) for code, w in weights.items()}
    
    # 重新归一化
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {code: w / total_weight * max_position for code, w in weights.items()}
    
    # 5. 行业权重裁剪（如果提供行业映射）
    if industry_map:
        weights = apply_industry_constraint(weights, industry_map, config["max_industry_weight"], max_position)
    
    return weights


def apply_industry_constraint(
    weights: Dict[str, float],
    industry_map: Dict[str, str],
    max_industry_weight: float,
    max_position: float,
) -> Dict[str, float]:
    """应用行业权重约束。"""
    # 按行业分组
    industry_weights = {}
    for code, weight in weights.items():
        industry = industry_map.get(code)
        if industry:
            if industry not in industry_weights:
                industry_weights[industry] = {}
            industry_weights[industry][code] = weight
    
    # 裁剪超限行业
    constrained = weights.copy()
    for industry, codes in industry_weights.items():
        total = sum(constrained.get(code, 0) for code in codes)
        if total > max_industry_weight:
            scale = max_industry_weight / total
            for code in codes:
                if code in constrained:
                    constrained[code] *= scale
    
    # 重新归一化
    total_weight = sum(constrained.values())
    if total_weight > 0:
        constrained = {code: w / total_weight * max_position for code, w in constrained.items()}
    
    return constrained


def rebalance_portfolio(context, bar_dict: Dict):
    """
    根据目标权重执行调仓。
    
    注意：bar_dict 只包含已订阅且当前有数据的股票。
    停牌、退市或数据不可用的股票不会出现在 bar_dict 中。
    我们直接尝试下单，让 RQAlpha 的撮合引擎处理不可交易的情况。
    """
    target_weights = getattr(context, "target_weights", {}) or {}
    current_date = context.now.date()
    
    # 获取当日预测信号
    signals = {}
    if current_date in context.prediction_signals:
        signals = context.prediction_signals[current_date]
    
    # 使用 get_positions() API 获取当前持仓
    current_positions = set()
    try:
        positions = get_positions()
        for pos in positions:
            if hasattr(pos, "order_book_id") and getattr(pos, "quantity", 0) > 0:
                current_positions.add(pos.order_book_id)
    except Exception as e:
        if logger:
            logger.warning(f"获取持仓列表失败: {e}")
    
    all_codes = set(target_weights.keys()) | current_positions
    
    if logger:
        logger.info(f"调仓股票数量: {len(all_codes)}, 目标权重股票: {len(target_weights)}")
        logger.debug(f"bar_dict 包含 {len(bar_dict)} 只股票的数据")
        logger.debug(f"当前持仓: {current_positions}")
    
    # 获取账户信息用于计算目标金额
    account = get_account()
    total_value = getattr(context.portfolio, "total_value", 0.0)
    if total_value == 0:
        total_value = getattr(account, "total_value", 0.0) if account else 0.0
    
    if logger:
        logger.info(f"账户总资产: {total_value:.2f}")
    
    # 直接尝试下单，不检查 bar_dict
    # RQAlpha 会自动处理停牌、涨跌停等不可交易的情况
    success_count = 0
    order_details = []
    
    # 记录买入信息
    buy_info = getattr(context, "buy_info", {})
    
    for code in all_codes:
        target_weight = target_weights.get(code, 0.0)
        
        # 如果目标权重为 0 且当前无持仓，跳过
        if target_weight == 0.0 and code not in current_positions:
            continue
        
        try:
            # 下单前记录信息
            target_value = total_value * target_weight if total_value > 0 else 0
            if logger:
                logger.debug(f"下单 {code}: 目标权重={target_weight:.4f}, 目标金额={target_value:.2f}")
            
            # 记录买入信息（在买入前）
            if target_weight > 0 and code not in current_positions:
                # 新买入
                buy_info[code] = {
                    "date": current_date,
                    "pred": signals.get(code, 0.0),
                    "weight": target_weight,
                    "high_pred": signals.get(code, 0.0)  # 初始化最高预测值
                }
                if logger:
                    logger.debug(f"记录买入: {code} - 预测值={signals.get(code, 0.0):.4f}")
            elif target_weight == 0 and code in current_positions:
                # 卖出，清除买入信息
                if code in buy_info:
                    del buy_info[code]
                    if logger:
                        logger.debug(f"清除买入信息: {code}")
            
            order = order_target_percent(code, target_weight)
            
            if order is not None:
                # 检查订单状态
                order_status = getattr(order, "status", None)
                order_id = getattr(order, "order_id", None)
                order_details.append({
                    "code": code,
                    "weight": target_weight,
                    "order_id": order_id,
                    "status": str(order_status) if order_status else "unknown"
                })
                success_count += 1
                if logger:
                    logger.debug(f"订单提交成功: {code}, order_id={order_id}, status={order_status}")
            else:
                if logger:
                    logger.warning(f"订单返回 None: {code}, 目标权重={target_weight:.4f}")
                    
        except Exception as e:
            # RQAlpha 会在订单被拒绝时抛出异常（如停牌、涨跌停等）
            if logger:
                logger.warning(f"调仓 {code} 到权重 {target_weight:.4f} 失败: {e}")
    
    # 更新买入信息
    context.buy_info = buy_info
    
    # 检查未成交订单
    try:
        open_orders = get_open_orders()
        if open_orders:
            if logger:
                logger.info(f"当前有 {len(open_orders)} 个未成交订单")
                for order in open_orders:
                    logger.debug(f"未成交订单: {getattr(order, 'order_book_id', 'unknown')}, "
                               f"status={getattr(order, 'status', 'unknown')}")
    except Exception as e:
        if logger:
            logger.debug(f"获取未成交订单失败: {e}")
    
    if logger:
        logger.info(f"成功下单 {success_count}/{len(all_codes)} 只股票")
        if order_details:
            logger.debug(f"订单详情: {order_details[:5]}...")  # 只显示前5个


def after_trading(context):
    """收盘后记录持仓信息并绘制净值曲线。"""
    # 在 RQAlpha 中，总资产应该从 context.portfolio 获取
    total_value = getattr(context.portfolio, "total_value", 0.0)
    account = get_account()
    cash = getattr(account, "cash", 0.0) if account else 0.0
    
    # 获取实际持仓数量
    positions = get_positions()
    position_count = len([p for p in positions if getattr(p, "quantity", 0) > 0])
    
    # 检查未成交订单（T+1 交易：当日下单，次日成交）
    open_orders_count = 0
    try:
        open_orders = get_open_orders()
        if open_orders:
            open_orders_count = len(open_orders)
            if logger:
                logger.info(f"当前有 {open_orders_count} 个未成交订单（T+1 交易，次日成交）")
    except Exception:
        pass
    
    # 详细记录持仓信息
    if positions:
        position_details = []
        for pos in positions[:5]:  # 只显示前5个
            if getattr(pos, "quantity", 0) > 0:
                position_details.append(
                    f"{getattr(pos, 'order_book_id', 'unknown')}: "
                    f"{getattr(pos, 'quantity', 0)}股, "
                    f"市值={getattr(pos, 'market_value', 0):.2f}"
                )
        if logger and position_details:
            logger.debug(f"持仓详情: {position_details}")
    
    if logger:
        logger.info(
            f"日期 {context.now.date()}，总资产: {total_value:.2f}, "
            f"现金: {cash:.2f}, "
            f"持仓数量: {position_count}, "
            f"未成交订单: {open_orders_count}, "
            f"目标组合股票数: {len(getattr(context, 'target_weights', {}))}"
        )
        
        # 如果持仓为0但有未成交订单，说明是 T+1 交易正常情况
        if position_count == 0 and open_orders_count > 0:
            logger.info("注意：当前无持仓但有未成交订单，这是 T+1 交易的正常情况（当日下单，次日成交）")
        elif position_count == 0 and open_orders_count == 0:
            logger.warning("警告：当前无持仓且无未成交订单，可能订单被全部拒绝或未提交成功")
    
    # ==================== 回测结束处理 ====================
    # 检查是否是最后一个交易日
    current_date = context.now.date()
    all_dates = sorted(context.prediction_signals.keys())
    if all_dates and current_date == all_dates[-1]:
        if not getattr(context, "backtest_ended", False):
            context.backtest_ended = True
            
            # 记录最终持仓信息
            if logger:
                logger.info("=" * 60)
                logger.info("回测结束，统计最终持仓")
                logger.info("=" * 60)
                
                # 统计未平仓持仓
                if position_count > 0:
                    logger.info(f"回测结束时仍有 {position_count} 只股票未平仓:")
                    total_market_value = 0
                    for pos in positions:
                        if getattr(pos, "quantity", 0) > 0:
                            code = getattr(pos, "order_book_id", "unknown")
                            qty = getattr(pos, "quantity", 0)
                            market_value = getattr(pos, "market_value", 0)
                            total_market_value += market_value
                            
                            # 获取买入信息
                            buy_info = getattr(context, "buy_info", {})
                            if code in buy_info:
                                buy_date = buy_info[code].get("date", "unknown")
                                buy_pred = buy_info[code].get("pred", 0)
                                logger.info(f"  {code}: {qty}股, 市值={market_value:.2f}, "
                                          f"买入日期={buy_date}, 买入预测值={buy_pred:.4f}")
                            else:
                                logger.info(f"  {code}: {qty}股, 市值={market_value:.2f}")
                    
                    logger.info(f"未平仓股票总市值: {total_market_value:.2f}")
                    logger.info(f"注意：这些持仓的盈亏已计入回测结果（按当日收盘价计算）")
                
                # 计算最终收益
                initial_cash = getattr(context, "initial_cash", 10000000.0)
                final_return = (total_value - initial_cash) / initial_cash * 100
                logger.info(f"初始资金: {initial_cash:.2f}")
                logger.info(f"最终资产: {total_value:.2f}")
                logger.info(f"总收益率: {final_return:.2f}%")
                logger.info("=" * 60)

