#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回测框架测试脚本

验证回测框架各模块是否正常工作
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """测试模块导入"""
    print("=" * 60)
    print("测试模块导入...")
    print("=" * 60)
    
    errors = []
    
    try:
        from monitor.data_source import AkshareAdapter, get_data_source
        print("✓ data_source 模块导入成功")
    except Exception as e:
        errors.append(f"data_source 导入失败: {e}")
        print(f"✗ data_source 导入失败: {e}")
    
    try:
        from monitor.backtest_engine import (
            BacktestConfig,
            BacktestEngine,
            MomentumStrategy,
            MeanReversionStrategy,
            Position,
            Trade,
        )
        print("✓ backtest_engine 模块导入成功")
    except Exception as e:
        errors.append(f"backtest_engine 导入失败: {e}")
        print(f"✗ backtest_engine 导入失败: {e}")
    
    try:
        from monitor.performance import (
            PerformanceAnalyzer,
            PerformanceMetrics,
            BacktestReportGenerator,
        )
        print("✓ performance 模块导入成功")
    except Exception as e:
        errors.append(f"performance 导入失败: {e}")
        print(f"✗ performance 导入失败: {e}")
    
    if errors:
        print(f"\n❌ 发现 {len(errors)} 个导入错误")
        return False
    else:
        print("\n✅ 所有模块导入成功")
        return True


def test_config():
    """测试配置类"""
    print("\n" + "=" * 60)
    print("测试配置类...")
    print("=" * 60)
    
    from monitor.backtest_engine import BacktestConfig
    
    config = BacktestConfig(
        start_date="2023-01-01",
        end_date="2023-12-31",
        initial_capital=1000000.0,
        commission_rate=0.0003,
        stamp_duty=0.001,
        slippage=0.001,
    )
    
    config_dict = config.to_dict()
    
    assert config_dict["start_date"] == "2023-01-01"
    assert config_dict["initial_capital"] == 1000000.0
    
    print(f"✓ BacktestConfig 创建成功")
    print(f"  - 开始日期: {config.start_date}")
    print(f"  - 结束日期: {config.end_date}")
    print(f"  - 初始资金: {config.initial_capital:,.0f}")
    print(f"  - 佣金率: {config.commission_rate:.2%}")
    
    return True


def test_position():
    """测试持仓类"""
    print("\n" + "=" * 60)
    print("测试持仓类...")
    print("=" * 60)
    
    from monitor.backtest_engine import Position
    
    position = Position(
        code="600519",
        name="贵州茅台",
        shares=1000,
        entry_price=1800.0,
        entry_date="2023-01-01",
        current_price=1900.0,
        highest_price=1950.0,
    )
    
    print(f"✓ Position 创建成功")
    print(f"  - 代码: {position.code}")
    print(f"  - 名称: {position.name}")
    print(f"  - 持仓: {position.shares} 股")
    print(f"  - 成本: ¥{position.cost:,.0f}")
    print(f"  - 市值: ¥{position.market_value:,.0f}")
    print(f"  - 盈亏: {position.profit_pct:.2%}")
    
    assert position.market_value == 1000 * 1900.0
    assert abs(position.profit_pct - (1900.0 - 1800.0) / 1800.0) < 0.0001
    
    print("✓ Position 计算正确")
    
    return True


def test_strategy():
    """测试策略类"""
    print("\n" + "=" * 60)
    print("测试策略类...")
    print("=" * 60)
    
    from monitor.backtest_engine import MomentumStrategy, MeanReversionStrategy
    
    momentum_config = {
        "name": "momentum",
        "lookback_period": 20,
        "top_k": 10,
        "min_momentum": 0.05,
        "stop_loss": -0.08,
        "take_profit": 0.15,
        "max_holding_days": 10,
        "position_size": 0.10,
    }
    
    momentum = MomentumStrategy(momentum_config)
    print(f"✓ MomentumStrategy 创建成功")
    print(f"  - 策略名称: {momentum.name}")
    print(f"  - 回看周期: {momentum.lookback_period} 天")
    print(f"  - 止损阈值: {momentum.stop_loss:.1%}")
    print(f"  - 止盈阈值: {momentum.take_profit:.1%}")
    
    mr_config = {
        "name": "mean_reversion",
        "lookback_period": 20,
        "oversold_threshold": -0.15,
        "overbought_threshold": 0.15,
        "top_k": 10,
        "position_size": 0.10,
    }
    
    mr = MeanReversionStrategy(mr_config)
    print(f"✓ MeanReversionStrategy 创建成功")
    print(f"  - 策略名称: {mr.name}")
    print(f"  - 超卖阈值: {mr.oversold_threshold:.1%}")
    print(f"  - 超买阈值: {mr.overbought_threshold:.1%}")
    
    return True


def test_performance():
    """测试绩效分析类"""
    print("\n" + "=" * 60)
    print("测试绩效分析类...")
    print("=" * 60)
    
    from monitor.performance import PerformanceAnalyzer, PerformanceMetrics
    from monitor.backtest_engine import BacktestConfig
    
    equity_curve = [1000000, 1010000, 1020000, 1015000, 1030000, 1025000, 1040000]
    
    config = BacktestConfig(
        start_date="2023-01-01",
        end_date="2023-01-07",
        initial_capital=1000000.0,
    )
    
    analyzer = PerformanceAnalyzer(
        equity_curve=equity_curve,
        trades=[],
        config=config,
    )
    
    metrics = analyzer._calculate_metrics()
    
    print(f"✓ PerformanceAnalyzer 创建成功")
    print(f"  - 总收益: {metrics.total_return:.2%}")
    print(f"  - 年化收益: {metrics.annual_return:.2%}")
    print(f"  - 最大回撤: {metrics.max_drawdown:.2%}")
    print(f"  - 夏普比率: {metrics.sharpe_ratio:.2f}")
    
    return True


def test_report_generator():
    """测试报告生成器"""
    print("\n" + "=" * 60)
    print("测试报告生成器...")
    print("=" * 60)
    
    from monitor.performance import BacktestReportGenerator
    
    results = {
        "config": {
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "initial_capital": 1000000.0,
            "commission_rate": 0.0003,
            "stamp_duty": 0.001,
            "slippage": 0.001,
        },
        "metrics": {
            "total_return": 0.15,
            "annual_return": 0.15,
            "max_drawdown": 0.08,
            "sharpe_ratio": 1.5,
            "sortino_ratio": 2.0,
            "calmar_ratio": 1.875,
            "volatility": 0.15,
            "win_rate": 0.55,
            "profit_loss_ratio": 1.8,
            "total_trades": 100,
            "winning_trades": 55,
            "losing_trades": 45,
            "avg_profit": 0.03,
            "avg_loss": -0.02,
            "avg_holding_days": 5.0,
        },
        "trades": [],
    }
    
    report_gen = BacktestReportGenerator(results)
    report = report_gen.generate_text_report()
    
    print("✓ BacktestReportGenerator 创建成功")
    print("\n生成的报告预览:")
    print(report[:500] + "...")
    
    return True


def main():
    """运行所有测试"""
    print("=" * 60)
    print("回测框架测试")
    print("=" * 60)
    
    tests = [
        ("模块导入", test_imports),
        ("配置类", test_config),
        ("持仓类", test_position),
        ("策略类", test_strategy),
        ("绩效分析", test_performance),
        ("报告生成器", test_report_generator),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n❌ {name} 测试失败: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print("测试结果")
    print("=" * 60)
    print(f"通过: {passed}/{len(tests)}")
    print(f"失败: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✅ 所有测试通过！")
        return 0
    else:
        print(f"\n❌ {failed} 个测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
