"""
融合系统测试入口 - Fusion System Test Runner

运行方式:
    cd d:/lianghuatouzi/Qlib1124/project
    python -m monitor.strategy_library.fusion.run_tests              # 运行所有测试
    python -m monitor.strategy_library.fusion.run_tests --layer 1    # 测试Layer 1
    python -m monitor.strategy_library.fusion.run_tests --demo       # 运行演示
    python -m monitor.strategy_library.fusion.run_tests --backtest   # 运行回测演示
"""

import argparse
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd


def run_all_tests():
    """运行所有测试"""
    import subprocess
    
    print("=" * 60)
    print("运行融合系统完整测试")
    print("=" * 60)
    
    result = subprocess.run(
        [sys.executable, "-m", "pytest", 
         "monitor/strategy_library/fusion/tests/test_layered_fusion.py", 
         "-v", "--tb=short"],
        cwd=Path(__file__).parent.parent.parent.parent,
    )
    
    return result.returncode


def run_layer_test(layer: int):
    """运行指定层的测试"""
    import subprocess
    
    layer_tests = {
        1: "TestLayer1SignalPreprocessor",
        2: "TestLayer2ConsistencyAnalyzer",
        3: "TestLayer3GainCalculator",
        4: "TestLayer4DecisionMaker",
    }
    
    test_class = layer_tests.get(layer)
    if not test_class:
        print(f"无效的层: {layer}, 有效值: 1, 2, 3, 4")
        return 1
    
    print(f"\n{'=' * 60}")
    print(f"测试 Layer {layer}: {test_class}")
    print("=" * 60)
    
    result = subprocess.run(
        [sys.executable, "-m", "pytest", 
         f"monitor/strategy_library/fusion/tests/test_layered_fusion.py::{test_class}",
         "-v", "--tb=short"],
        cwd=Path(__file__).parent.parent.parent.parent,
    )
    
    return result.returncode


def run_demo():
    """运行融合系统演示"""
    print("\n" + "=" * 60)
    print("融合系统演示 - Fusion System Demo")
    print("=" * 60)
    
    from monitor.strategy_library.fusion import (
        LayeredFusionProcessor,
        FusionConfig,
        ModelPrediction,
        StrategySignalWrapper,
        MarketContext,
        MarketRegime,
        ConsistencyType,
    )
    
    config = FusionConfig(
        consistency_boost=1.3,
        conflict_penalty=0.7,
        enable_dynamic_weights=True,
        enable_regime_adjustment=True,
    )
    
    processor = LayeredFusionProcessor(config)
    
    print("\n--- 场景1: 模型+策略一致看多 ---")
    model_predictions = {
        "000001.SZ": ModelPrediction(
            code="000001.SZ",
            name="平安银行",
            score=0.72,
            confidence=0.85,
            horizon=5,
        ),
    }
    
    strategy_signals = {
        "000001.SZ": StrategySignalWrapper(
            code="000001.SZ",
            name="平安银行",
            action="buy",
            strength=0.75,
            reason="多因子策略看多",
            strategy_id="multi_factor",
            strategy_name="多因子策略",
        ),
    }
    
    market_context = MarketContext(
        index_value=3050,
        index_change=0.015,
        volatility=0.018,
        trend="up",
        regime=MarketRegime.BULL_TRENDING,
        uncertainty=0.35,
    )
    
    decisions = processor.process(model_predictions, strategy_signals, market_context)
    
    for d in decisions:
        print(f"\n股票: {d.signal.code} - {d.signal.name}")
        print(f"  动作: {d.signal.action}")
        print(f"  一致性: {d.signal.consistency.value}")
        print(f"  融合分数: {d.signal.fused_score:.3f}")
        print(f"  置信度: {d.signal.confidence:.2f}")
        print(f"  仓位比例: {d.position_ratio:.2%}")
        print(f"  止损: {d.stop_loss:.2%}")
        print(f"  止盈: {d.take_profit:.2%}")
        print(f"  执行优先级: {d.execution_priority}/10")
        print(f"  增益潜力: {d.gain_potential:.3f}")
    
    print("\n--- 场景2: 模型看多 vs 策略看空 (冲突) ---")
    model_predictions = {
        "600036.SH": ModelPrediction(
            code="600036.SH",
            name="招商银行",
            score=0.75,
            confidence=0.70,
            horizon=10,
        ),
    }
    
    strategy_signals = {
        "600036.SH": StrategySignalWrapper(
            code="600036.SH",
            name="招商银行",
            action="sell",
            strength=0.65,
            reason="趋势策略看空",
            strategy_id="turtle_trading",
            strategy_name="海龟策略",
        ),
    }
    
    decisions = processor.process(model_predictions, strategy_signals, market_context)
    
    for d in decisions:
        print(f"\n股票: {d.signal.code} - {d.signal.name}")
        print(f"  动作: {d.signal.action}")
        print(f"  一致性: {d.signal.consistency.value}")
        print(f"  融合分数: {d.signal.fused_score:.3f}")
        print(f"  模型权重: {d.signal.model_weight:.2f}")
        print(f"  策略权重: {d.signal.strategy_weight:.2f}")
        print(f"  执行优先级: {d.execution_priority}/10")
    
    print("\n--- 场景3: 仅模型信号 ---")
    model_predictions = {
        "601318.SH": ModelPrediction(
            code="601318.SH",
            name="中国平安",
            score=0.68,
            confidence=0.75,
            horizon=5,
        ),
    }
    
    decisions = processor.process(model_predictions, {}, market_context)
    
    for d in decisions:
        print(f"\n股票: {d.signal.code} - {d.signal.name}")
        print(f"  动作: {d.signal.action}")
        print(f"  一致性: {d.signal.consistency.value}")
        print(f"  融合分数: {d.signal.fused_score:.3f}")
    
    stats = processor.get_statistics()
    print("\n--- 统计信息 ---")
    print(f"总决策数: {stats['total_decisions']}")
    print(f"平均仓位: {stats['avg_position_ratio']:.2%}")
    print(f"平均增益潜力: {stats['avg_gain_potential']:.3f}")
    print(f"动作分布: {stats['action_distribution']}")
    print(f"置信度分布: {stats['confidence_distribution']}")
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)
    
    return 0


def run_backtest_demo():
    """运行回测演示"""
    print("\n" + "=" * 60)
    print("融合回测演示 - Fusion Backtest Demo")
    print("=" * 60)
    
    from monitor.strategy_library.fusion import (
        FusionBacktester,
        FusionBacktestReport,
        FusionConfig,
        ModelPrediction,
        StrategySignalWrapper,
        MarketContext,
        MarketRegime,
    )
    
    print("\n生成模拟数据...")
    
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 3, 31)
    
    dates = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:
            dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    
    codes = ["000001.SZ", "000002.SZ", "600000.SH", "600036.SH", "601318.SH"]
    names = ["平安银行", "万科A", "浦发银行", "招商银行", "中国平安"]
    
    np.random.seed(42)
    
    price_records = []
    base_prices = {code: np.random.uniform(10, 50) for code in codes}
    
    for date in dates:
        for code in codes:
            change = np.random.uniform(-0.03, 0.03)
            base_prices[code] *= (1 + change)
            
            price_records.append({
                "date": date,
                "code": code,
                "open": base_prices[code] * (1 + np.random.uniform(-0.01, 0.01)),
                "high": base_prices[code] * (1 + np.random.uniform(0, 0.02)),
                "low": base_prices[code] * (1 + np.random.uniform(-0.02, 0)),
                "close": base_prices[code],
                "volume": np.random.randint(1000000, 10000000),
            })
    
    price_data = pd.DataFrame(price_records)
    print(f"价格数据: {len(price_data)} 条记录")
    
    model_predictions = {}
    strategy_signals = {}
    market_contexts = {}
    
    for code in codes:
        model_predictions[code] = []
        strategy_signals[code] = []
    
    for i, date in enumerate(dates):
        if i % 5 == 0:
            for j, code in enumerate(codes):
                pred = ModelPrediction(
                    code=code,
                    name=names[j],
                    score=np.random.uniform(0.4, 0.7),
                    confidence=np.random.uniform(0.5, 0.9),
                    horizon=np.random.choice([3, 5, 10]),
                    timestamp=f"{date}T09:30:00",
                )
                model_predictions[code].append(pred)
        
        if i % 7 == 0:
            for j, code in enumerate(codes):
                if np.random.random() > 0.3:
                    sig = StrategySignalWrapper(
                        code=code,
                        name=names[j],
                        action=np.random.choice(["buy", "sell", "hold"], p=[0.4, 0.3, 0.3]),
                        strength=np.random.uniform(0.3, 0.9),
                        reason="模拟信号",
                        strategy_id=np.random.choice(["multi_factor", "turtle_trading", "grid_trading"]),
                        strategy_name="模拟策略",
                        timestamp=f"{date}T15:00:00",
                    )
                    strategy_signals[code].append(sig)
        
        trend = "up" if np.random.random() > 0.5 else "down" if np.random.random() > 0.5 else "neutral"
        market_contexts[date] = MarketContext(
            index_value=3000 + np.random.uniform(-200, 200),
            index_change=np.random.uniform(-0.02, 0.02),
            volatility=np.random.uniform(0.01, 0.03),
            trend=trend,
            regime=np.random.choice(list(MarketRegime)),
            uncertainty=np.random.uniform(0.3, 0.7),
            timestamp=f"{date}T15:00:00",
        )
    
    print(f"模型预测: {sum(len(v) for v in model_predictions.values())} 条")
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
    print("回测结果对比")
    print("=" * 60)
    
    print(f"\n{'指标':<15} {'融合策略':>12} {'模型单独':>12} {'策略单独':>12}")
    print("-" * 55)
    print(f"{'总收益':<15} {fused_result.total_return*100:>11.2f}% {model_result.total_return*100:>11.2f}% {strategy_result.total_return*100:>11.2f}%")
    print(f"{'年化收益':<15} {fused_result.annual_return*100:>11.2f}% {model_result.annual_return*100:>11.2f}% {strategy_result.annual_return*100:>11.2f}%")
    print(f"{'夏普比率':<15} {fused_result.sharpe_ratio:>12.2f} {model_result.sharpe_ratio:>12.2f} {strategy_result.sharpe_ratio:>12.2f}")
    print(f"{'最大回撤':<15} {fused_result.max_drawdown*100:>11.2f}% {model_result.max_drawdown*100:>11.2f}% {strategy_result.max_drawdown*100:>11.2f}%")
    print(f"{'胜率':<15} {fused_result.win_rate*100:>11.2f}% {model_result.win_rate*100:>11.2f}% {strategy_result.win_rate*100:>11.2f}%")
    print(f"{'交易次数':<15} {fused_result.total_trades:>12} {model_result.total_trades:>12} {strategy_result.total_trades:>12}")
    
    print("\n" + "=" * 60)
    print("增益效应分析")
    print("=" * 60)
    
    print(f"\n{'指标':<15} {'增益值':>12} {'状态':>10}")
    print("-" * 40)
    print(f"{'收益增益':<15} {gain_metrics.return_gain*100:>11.2f}% {'✓ 正向' if gain_metrics.return_gain > 0 else '✗ 负向':>10}")
    print(f"{'夏普增益':<15} {gain_metrics.sharpe_gain:>12.2f} {'✓ 正向' if gain_metrics.sharpe_gain > 0 else '✗ 负向':>10}")
    print(f"{'回撤增益':<15} {gain_metrics.drawdown_gain*100:>11.2f}% {'✓ 改善' if gain_metrics.drawdown_gain < 0 else '✗ 恶化':>10}")
    print(f"{'胜率增益':<15} {gain_metrics.winrate_gain*100:>11.2f}% {'✓ 正向' if gain_metrics.winrate_gain > 0 else '✗ 负向':>10}")
    print(f"{'卡玛增益':<15} {gain_metrics.calmar_gain:>12.2f} {'✓ 正向' if gain_metrics.calmar_gain > 0 else '✗ 负向':>10}")
    print(f"{'索提诺增益':<15} {gain_metrics.sortino_gain:>12.2f} {'✓ 正向' if gain_metrics.sortino_gain > 0 else '✗ 负向':>10}")
    
    print("\n" + "=" * 60)
    print(f"增益效应: {'存在' if gain_metrics.has_gain() else '不存在'}")
    print("=" * 60)
    
    output_dir = Path("data/fusion_reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_gen = FusionBacktestReport(str(output_dir))
    report = report_gen.generate_report(
        fused_result=fused_result,
        model_result=model_result,
        strategy_result=strategy_result,
        gain_metrics=gain_metrics,
        save_json=True,
        save_html=True,
    )
    
    print(f"\n报告已生成:")
    print(f"  JSON: {report.get('json_path', 'N/A')}")
    print(f"  HTML: {report.get('html_path', 'N/A')}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(description="融合系统测试入口")
    parser.add_argument("--layer", type=int, choices=[1, 2, 3, 4],
                        help="测试指定层 (1-4)")
    parser.add_argument("--demo", action="store_true",
                        help="运行融合系统演示")
    parser.add_argument("--backtest", action="store_true",
                        help="运行回测演示")
    parser.add_argument("--real", action="store_true",
                        help="运行真实数据回测 (akshare)")
    parser.add_argument("--start", type=str, default="2024-01-01",
                        help="真实数据开始日期 (默认: 2024-01-01)")
    parser.add_argument("--end", type=str, default="2024-03-31",
                        help="真实数据结束日期 (默认: 2024-03-31)")
    
    args = parser.parse_args()
    
    if args.demo:
        return run_demo()
    elif args.backtest:
        return run_backtest_demo()
    elif args.real:
        return run_real_data_backtest(args.start, args.end)
    elif args.layer:
        return run_layer_test(args.layer)
    else:
        return run_all_tests()


def run_real_data_backtest(start_date: str, end_date: str):
    """运行真实数据回测"""
    from monitor.strategy_library.fusion.real_data_provider import run_real_data_backtest as run
    return run(start_date, end_date)


if __name__ == "__main__":
    sys.exit(main())
