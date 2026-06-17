"""
调试回测资金流转问题 - 完整版
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from monitor.strategy_library.fusion import (
    FusionBacktester,
    FusionConfig,
    ModelPrediction,
    StrategySignalWrapper,
    MarketContext,
    MarketRegime,
)

def debug_backtest():
    """调试回测资金流转"""
    print("=" * 60)
    print("调试回测资金流转")
    print("=" * 60)
    
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 31)
    
    dates = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:
            dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    
    codes = ["000001.SZ"]
    names = ["平安银行"]
    
    np.random.seed(42)
    
    # 生成简单的价格数据 - 每天涨1%
    price_records = []
    base_price = 10.0
    
    for i, date in enumerate(dates):
        base_price *= 1.01
        
        price_records.append({
            "date": date,
            "code": codes[0],
            "open": base_price,
            "high": base_price * 1.01,
            "low": base_price * 0.99,
            "close": base_price,
            "volume": 1000000,
        })
    
    price_data = pd.DataFrame(price_records)
    print(f"\n价格数据: {len(price_data)} 条")
    print(f"起始价格: {price_data['close'].iloc[0]:.2f}")
    print(f"结束价格: {price_data['close'].iloc[-1]:.2f}")
    print(f"价格涨幅: {(price_data['close'].iloc[-1] / price_data['close'].iloc[0] - 1) * 100:.2f}%")
    
    # 只生成一个买入信号
    model_predictions = {
        codes[0]: [
            ModelPrediction(
                code=codes[0],
                name=names[0],
                score=0.7,
                confidence=0.8,
                horizon=5,
                timestamp=f"{dates[0]}T09:30:00",
            ),
        ]
    }
    
    strategy_signals = {
        codes[0]: [
            StrategySignalWrapper(
                code=codes[0],
                name=names[0],
                action="buy",
                strength=0.8,
                reason="测试买入",
                strategy_id="test",
                strategy_name="测试策略",
                timestamp=f"{dates[0]}T15:00:00",
            ),
        ]
    }
    
    market_contexts = {}
    for date in dates:
        market_contexts[date] = MarketContext(
            index_value=3000,
            index_change=0.01,
            volatility=0.02,
            trend="up",
            regime=MarketRegime.BULL_TRENDING,
            uncertainty=0.3,
            timestamp=f"{date}T15:00:00",
        )
    
    print(f"\n模型预测: {sum(len(v) for v in model_predictions.values())} 条")
    print(f"策略信号: {sum(len(v) for v in strategy_signals.values())} 条")
    
    print("\n运行回测...")
    
    config = FusionConfig(
        consistency_boost=1.3,
        conflict_penalty=0.7,
    )
    
    backtester = FusionBacktester(
        config=config,
        initial_capital=1000000,
        commission_rate=0.0003,
        slippage=0.001,
    )
    
    fused_result, model_result, strategy_result, gain_metrics = backtester.run_backtest(
        price_data=price_data,
        model_predictions=model_predictions,
        strategy_signals=strategy_signals,
        market_contexts=market_contexts,
        start_date=dates[0],
        end_date=dates[-1],
    )
    
    print("\n" + "=" * 60)
    print("回测结果")
    print("=" * 60)
    
    print(f"\n初始资金: 1,000,000")
    print(f"最终资金: {fused_result.total_return * 1000000 + 1000000:.2f}")
    print(f"总收益: {fused_result.total_return * 100:.2f}%")
    print(f"年化收益: {fused_result.annual_return * 100:.2f}%")
    print(f"夏普比率: {fused_result.sharpe_ratio:.2f}")
    print(f"最大回撤: {fused_result.max_drawdown * 100:.2f}%")
    print(f"交易次数: {fused_result.total_trades}")
    
    # 打印交易详情
    if backtester._trades:
        print("\n交易详情:")
        for t in backtester._trades:
            print(f"  {t.code}: 买入{t.entry_price:.2f}, 卖出{t.exit_price:.2f}, 盈亏{t.pnl:.2f} ({t.pnl_pct*100:.2f}%)")
    
    # 打印每日资金
    if backtester._daily_values:
        print("\n每日资金变化:")
        for i, dv in enumerate(backtester._daily_values[:5]):
            print(f"  {dv['date']}: {dv['value']:.2f}")
        if len(backtester._daily_values) > 5:
            print(f"  ... ({len(backtester._daily_values) - 5} more days)")
            print(f"  {backtester._daily_values[-1]['date']}: {backtester._daily_values[-1]['value']:.2f}")
    
    # 验证结果合理性
    print("\n" + "=" * 60)
    print("验证结果")
    print("=" * 60)
    
    price_return = (price_data['close'].iloc[-1] / price_data['close'].iloc[0] - 1) * 100
    print(f"价格涨幅: {price_return:.2f}%")
    print(f"回测收益: {fused_result.total_return * 100:.2f}%")
    
    if abs(fused_result.total_return * 100 - price_return) < price_return * 0.5:
        print("✓ 回测收益与价格涨幅接近，结果合理")
    else:
        print("✗ 回测收益与价格涨幅差异较大，可能存在问题")
    
    if fused_result.total_return < 1.0:  # 收益小于100%
        print("✓ 收益率在合理范围内")
    else:
        print("✗ 收益率异常，需要检查")


if __name__ == "__main__":
    debug_backtest()
