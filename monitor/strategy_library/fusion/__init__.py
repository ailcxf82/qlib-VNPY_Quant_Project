"""
Fusion Module - 融合模块

提供模型预测与策略信号的智能融合功能，实现增益效应。
"""

from monitor.strategy_library.fusion.fusion_engine import (
    ConflictResolver,
    ConsistencyType,
    DynamicWeightAdjuster,
    FusionConfig,
    FusionEngine,
    FusionWeights,
    FusedSignal,
    MarketContext,
    MarketRegime,
    MarketRegimeDetector,
    ModelPrediction,
    StrategySignalWrapper,
    create_fusion_engine,
)

from monitor.strategy_library.fusion.layered_fusion import (
    SignalStrength,
    TimeHorizon,
    NormalizedSignal,
    ConsistencyResult,
    GainMetrics,
    FusionDecision,
    Layer1SignalPreprocessor,
    Layer2ConsistencyAnalyzer,
    Layer3GainCalculator,
    Layer4DecisionMaker,
    GainValidator,
    LayeredFusionProcessor,
)

from monitor.strategy_library.fusion.fusion_backtest import (
    BacktestPosition,
    BacktestTrade,
    BacktestResult,
    FusionBacktester,
    FusionBacktestReport,
    run_fusion_backtest,
)

from monitor.strategy_library.fusion.data_provider import (
    DataSourceConfig,
    ModelPredictionProvider,
    StrategySignalProvider,
    MarketContextProvider,
    PriceDataProvider,
    FusionDataProvider,
)

__all__ = [
    "ConflictResolver",
    "ConsistencyType",
    "DynamicWeightAdjuster",
    "FusionConfig",
    "FusionEngine",
    "FusionWeights",
    "FusedSignal",
    "MarketContext",
    "MarketRegime",
    "MarketRegimeDetector",
    "ModelPrediction",
    "StrategySignalWrapper",
    "create_fusion_engine",
    "SignalStrength",
    "TimeHorizon",
    "NormalizedSignal",
    "ConsistencyResult",
    "GainMetrics",
    "FusionDecision",
    "Layer1SignalPreprocessor",
    "Layer2ConsistencyAnalyzer",
    "Layer3GainCalculator",
    "Layer4DecisionMaker",
    "GainValidator",
    "LayeredFusionProcessor",
    "BacktestPosition",
    "BacktestTrade",
    "BacktestResult",
    "FusionBacktester",
    "FusionBacktestReport",
    "run_fusion_backtest",
    "DataSourceConfig",
    "ModelPredictionProvider",
    "StrategySignalProvider",
    "MarketContextProvider",
    "PriceDataProvider",
    "FusionDataProvider",
]
