"""
Backtest Module - 回测模块

提供策略回测、绩效分析、监控和报告生成功能。
"""

from monitor.strategy_library.backtest.backtest_engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestLogger,
    BacktestResult,
    DailySnapshot,
    TradeLog,
    TradeSimulator,
)
from monitor.strategy_library.backtest.performance_analyzer import (
    PerformanceAnalyzer,
    PerformanceMetrics,
    TradeAnalysis,
)
from monitor.strategy_library.backtest.strategy_monitor import (
    MonitoringConfig,
    StrategyMonitor,
    StrategyStatus,
    create_monitor_from_config,
)
from monitor.strategy_library.backtest.report_generator import (
    ReportConfig,
    ReportGenerator,
)

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestLogger",
    "BacktestResult",
    "DailySnapshot",
    "TradeLog",
    "TradeSimulator",
    "PerformanceAnalyzer",
    "PerformanceMetrics",
    "TradeAnalysis",
    "MonitoringConfig",
    "StrategyMonitor",
    "StrategyStatus",
    "create_monitor_from_config",
    "ReportConfig",
    "ReportGenerator",
]
