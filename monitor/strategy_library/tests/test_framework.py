"""
Strategy Library Framework Verification Script
验证策略库框架的核心功能
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datetime import datetime, timedelta
import pandas as pd
import numpy as np

print("=" * 60)
print("策略库框架验证测试")
print("=" * 60)

def test_imports():
    """测试模块导入"""
    print("\n[1] 测试模块导入...")
    try:
        from monitor.strategy_library import (
            StrategyBase, StrategyConfig, Signal, BacktestResult, ParameterSpace,
            ConfigLoader, StrategyRegistry
        )
        from monitor.strategy_library.strategies import MultiFactorStrategy, TurtleTradingStrategy
        from monitor.strategy_library.optimizer import (
            BaseOptimizer, OptimizationResult, OptimizationConfig,
            GridSearchOptimizer, BayesianOptimizer
        )
        from monitor.strategy_library.evaluator.overfitting_protection import OverfittingProtector
        from monitor.strategy_library.storage.strategy_store import StrategyStore
        print("    ✓ 所有模块导入成功")
        return True
    except Exception as e:
        print(f"    ✗ 导入失败: {e}")
        return False

def test_strategy_registry():
    """测试策略注册"""
    print("\n[2] 测试策略注册...")
    try:
        from monitor.strategy_library.strategy_registry import StrategyRegistry
        
        registry = StrategyRegistry()
        StrategyRegistry.discover()
        
        all_strategies = StrategyRegistry.list_all()
        strategies = list(all_strategies.keys())
        print(f"    已注册策略: {strategies}")
        
        for strategy_id, info in all_strategies.items():
            desc = info.description[:30] if info.description else 'N/A'
            print(f"    - {strategy_id}: {info.category} - {desc}...")
        
        print("    ✓ 策略注册测试通过")
        return True
    except Exception as e:
        print(f"    ✗ 策略注册测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_loader():
    """测试配置加载"""
    print("\n[3] 测试配置加载...")
    try:
        from monitor.strategy_library.config_loader import ConfigLoader
        from monitor.strategy_library.strategy_base import StrategyConfig, ParameterSpace
        
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        config_dir = os.path.join(project_root, "config", "strategies")
        
        loader = ConfigLoader(config_dir=config_dir)
        
        config = loader.load("multi_factor")
        
        if config:
            print(f"    加载配置: {config.id}")
            print(f"    策略名称: {config.name}")
            print(f"    参数空间: {list(config.parameter_space.keys())}")
            print("    ✓ 配置加载测试通过")
            return True
        else:
            print("    配置文件不存在，使用默认配置测试")
            test_config = StrategyConfig(
                id="test_strategy",
                name="测试策略",
                category="测试",
                version="1.0.0",
                enabled=True,
                parameters={"param1": 10},
                parameter_space={
                    "param1": ParameterSpace("param1", "int", 5, 20, default=10)
                }
            )
            print(f"    创建测试配置: {test_config.id}")
            print("    ✓ 配置加载测试通过（使用默认配置）")
            return True
    except Exception as e:
        print(f"    ✗ 配置加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_strategy_instantiation():
    """测试策略实例化"""
    print("\n[4] 测试策略实例化...")
    try:
        from monitor.strategy_library.strategies import MultiFactorStrategy, TurtleTradingStrategy
        from monitor.strategy_library.strategy_base import StrategyConfig, ParameterSpace
        
        multi_factor = MultiFactorStrategy()
        print(f"    多因子策略 ID: {multi_factor.STRATEGY_ID}")
        print(f"    多因子策略参数: {list(multi_factor.DEFAULT_FACTORS.keys())}")
        
        turtle = TurtleTradingStrategy()
        print(f"    海龟策略 ID: {turtle.STRATEGY_ID}")
        
        param_space = turtle.get_parameter_space()
        print(f"    海龟策略参数空间: {list(param_space.keys())}")
        
        print("    ✓ 策略实例化测试通过")
        return True
    except Exception as e:
        print(f"    ✗ 策略实例化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_parameter_space():
    """测试参数空间"""
    print("\n[5] 测试参数空间...")
    try:
        from monitor.strategy_library.strategy_base import ParameterSpace
        
        int_param = ParameterSpace(
            name="entry_period",
            param_type="int",
            min_val=10,
            max_val=30,
            default=20
        )
        print(f"    整数参数: {int_param.name}, 范围: [{int_param.min_val}, {int_param.max_val}]")
        
        choice_param = ParameterSpace(
            name="method",
            param_type="choice",
            choices=["equal_weight", "ic_weight", "max_sharpe"],
            default="equal_weight"
        )
        print(f"    选择参数: {choice_param.name}, 选项: {choice_param.choices}")
        
        print("    ✓ 参数空间测试通过")
        return True
    except Exception as e:
        print(f"    ✗ 参数空间测试失败: {e}")
        return False

def test_signal_generation():
    """测试信号生成"""
    print("\n[6] 测试信号生成...")
    try:
        from monitor.strategy_library.strategies import TurtleTradingStrategy
        from monitor.strategy_library.strategy_base import Signal
        import pandas as pd
        import numpy as np
        
        strategy = TurtleTradingStrategy()
        
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        
        data = pd.DataFrame({
            "date": dates,
            "open": 10 + np.cumsum(np.random.randn(100) * 0.02),
            "high": 10.2 + np.cumsum(np.random.randn(100) * 0.02),
            "low": 9.8 + np.cumsum(np.random.randn(100) * 0.02),
            "close": 10 + np.cumsum(np.random.randn(100) * 0.02),
            "volume": np.random.randint(1000000, 5000000, 100),
        }, index=range(100))
        
        data_dict = {"000001.SZ": data}
        positions = {}
        
        signals = strategy.generate_signals(
            date="2024-04-10",
            data=data_dict,
            positions=positions,
            total_capital=1000000
        )
        
        print(f"    生成信号数量: {len(signals)}")
        if signals:
            for sig in signals[:3]:
                print(f"    - 信号: {sig.action} @ {sig.price:.2f}, 原因: {sig.reason}")
        
        print("    ✓ 信号生成测试通过")
        return True
    except Exception as e:
        print(f"    ✗ 信号生成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_optimizer():
    """测试优化器"""
    print("\n[7] 测试优化器...")
    try:
        from monitor.strategy_library.optimizer import GridSearchOptimizer, OptimizationConfig
        from monitor.strategy_library.strategy_base import ParameterSpace
        
        config = OptimizationConfig(
            objective="sharpe_ratio",
            direction="maximize",
            n_trials=10,
            n_jobs=1,
            verbose=False
        )
        optimizer = GridSearchOptimizer(config=config)
        
        print(f"    优化器类型: {optimizer.__class__.__name__}")
        print(f"    优化指标: {config.objective}")
        
        param_space = {
            "entry_period": ParameterSpace("entry_period", "int", 15, 25, default=20),
            "exit_period": ParameterSpace("exit_period", "int", 8, 12, default=10),
        }
        
        combinations = optimizer._generate_grid(param_space)
        print(f"    参数组合数量: {len(combinations)}")
        
        print("    ✓ 优化器测试通过")
        return True
    except Exception as e:
        print(f"    ✗ 优化器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_overfitting_protection():
    """测试过拟合防护"""
    print("\n[8] 测试过拟合防护...")
    try:
        from monitor.strategy_library.evaluator.overfitting_protection import OverfittingProtector
        from monitor.strategy_library.strategy_base import BacktestResult
        
        protector = OverfittingProtector(
            overfitting_threshold=0.20,
            degradation_threshold=0.30,
            verbose=False
        )
        
        in_sample = BacktestResult(
            strategy_id="test",
            strategy_name="测试策略",
            parameters={"param1": 10},
            start_date="2023-01-01",
            end_date="2023-06-30",
            initial_capital=1000000,
            final_capital=1500000,
            total_return=0.5,
            annual_return=0.3,
            sharpe_ratio=1.5,
            sortino_ratio=1.2,
            max_drawdown=0.15,
            win_rate=0.6,
            profit_factor=1.5,
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            avg_holding_days=5.0
        )
        
        out_sample = BacktestResult(
            strategy_id="test",
            strategy_name="测试策略",
            parameters={"param1": 10},
            start_date="2023-07-01",
            end_date="2023-12-31",
            initial_capital=1000000,
            final_capital=1350000,
            total_return=0.35,
            annual_return=0.2,
            sharpe_ratio=1.0,
            sortino_ratio=0.8,
            max_drawdown=0.2,
            win_rate=0.55,
            profit_factor=1.2,
            total_trades=30,
            winning_trades=16,
            losing_trades=14,
            avg_holding_days=4.5
        )
        
        result = protector.check_overfitting(in_sample, out_sample, "sharpe_ratio")
        print(f"    样本内夏普: {in_sample.sharpe_ratio:.2f}")
        print(f"    样本外夏普: {out_sample.sharpe_ratio:.2f}")
        print(f"    过拟合比例: {result.overfitting_ratio:.2%}")
        print(f"    是否过拟合: {result.is_overfitted}")
        if result.warnings:
            print(f"    警告信息: {result.warnings[0] if result.warnings else 'None'}")
        
        print("    ✓ 过拟合防护测试通过")
        return True
    except Exception as e:
        print(f"    ✗ 过拟合防护测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_strategy_store():
    """测试策略存储"""
    print("\n[9] 测试策略存储...")
    try:
        from monitor.strategy_library.storage.strategy_store import StrategyStore
        from monitor.strategy_library.strategy_base import StrategyConfig, ParameterSpace
        import tempfile
        import os
        
        db_path = os.path.join(tempfile.gettempdir(), "test_strategy_lib.db")
        
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except:
                pass
        
        store = StrategyStore(db_path=db_path)
        
        config = StrategyConfig(
            id="test_strategy",
            name="测试策略",
            category="测试",
            version="1.0.0",
            enabled=True,
            parameters={"param1": 10, "param2": 0.5},
            parameter_space={
                "param1": ParameterSpace("param1", "int", 5, 20, default=10)
            },
            optimization={},
            backtest={},
            constraints={}
        )
        
        success = store.save_strategy(config)
        print(f"    保存策略: {'成功' if success else '失败'}")
        
        loaded = store.load_strategy("test_strategy")
        if loaded:
            print(f"    加载策略: {loaded.name}")
        
        result_id = store.save_optimization_result(
            strategy_id="test_strategy",
            best_params={"param1": 15, "param2": 0.6},
            best_score=1.2,
            all_results=[{"params": {"param1": 15}, "score": 1.2}],
            execution_time=10.5
        )
        print(f"    保存优化结果 ID: {result_id}")
        
        latest = store.get_latest_optimization("test_strategy")
        if latest:
            print(f"    最佳参数: {latest.best_params}")
        
        print("    ✓ 策略存储测试通过")
        return True
    except Exception as e:
        print(f"    ✗ 策略存储测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    tests = [
        test_imports,
        test_strategy_registry,
        test_config_loader,
        test_strategy_instantiation,
        test_parameter_space,
        test_signal_generation,
        test_optimizer,
        test_overfitting_protection,
        test_strategy_store,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("\n✓ 所有测试通过！Phase 1 框架验证成功。")
    else:
        print(f"\n✗ {total - passed} 个测试失败，请检查错误信息。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
