"""
Fusion Engine Tests - 融合引擎测试

测试模型预测与策略信号的融合功能。
"""

import pytest
import numpy as np
from datetime import datetime

from monitor.strategy_library.fusion import (
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


class TestModelPrediction:
    def test_model_prediction_creation(self):
        pred = ModelPrediction(
            code="600519.SH",
            name="贵州茅台",
            score=0.75,
            confidence=0.8,
        )
        
        assert pred.code == "600519.SH"
        assert pred.score == 0.75
        assert pred.confidence == 0.8
        assert pred.direction == "up"
        assert pred.timestamp != ""
    
    def test_model_prediction_down_direction(self):
        pred = ModelPrediction(
            code="000001.SZ",
            name="平安银行",
            score=0.35,
            confidence=0.7,
        )
        
        assert pred.direction == "down"
    
    def test_model_prediction_neutral_direction(self):
        pred = ModelPrediction(
            code="000002.SZ",
            name="万科A",
            score=0.50,
            confidence=0.6,
        )
        
        assert pred.direction == "neutral"


class TestStrategySignalWrapper:
    def test_strategy_signal_creation(self):
        signal = StrategySignalWrapper(
            code="600519.SH",
            name="贵州茅台",
            action="buy",
            strength=0.8,
            reason="突破20日高点",
            strategy_id="turtle_trading",
            strategy_name="海龟交易策略",
        )
        
        assert signal.code == "600519.SH"
        assert signal.action == "buy"
        assert signal.strength == 0.8
        assert signal.timestamp != ""


class TestMarketRegimeDetector:
    def test_detect_neutral_regime(self):
        detector = MarketRegimeDetector()
        regime = detector.detect()
        
        assert regime in [
            MarketRegime.BULL_TRENDING,
            MarketRegime.BULL_CHOPPY,
            MarketRegime.BEAR_TRENDING,
            MarketRegime.BEAR_CHOPPY,
            MarketRegime.NEUTRAL,
        ]
    
    def test_detect_from_context(self):
        detector = MarketRegimeDetector()
        
        context = {
            "trend": "up",
            "volatility": 0.02,
            "index_change": 0.05,
        }
        
        regime = detector.detect(context)
        assert regime == MarketRegime.BULL_TRENDING
    
    def test_detect_bull_choppy(self):
        detector = MarketRegimeDetector()
        
        context = {
            "trend": "up",
            "volatility": 0.03,
            "index_change": 0.02,
        }
        
        regime = detector.detect(context)
        assert regime == MarketRegime.BULL_CHOPPY


class TestDynamicWeightAdjuster:
    def test_default_weights(self):
        config = FusionConfig()
        adjuster = DynamicWeightAdjuster(config)
        
        weights = adjuster.get_weights(MarketRegime.NEUTRAL)
        
        assert abs(weights.model_weight - 0.5) < 0.01
        assert abs(weights.strategy_weight - 0.5) < 0.01
    
    def test_regime_adjusted_weights(self):
        config = FusionConfig(enable_regime_adjustment=True)
        adjuster = DynamicWeightAdjuster(config)
        
        weights = adjuster.get_weights(MarketRegime.BULL_TRENDING)
        
        assert weights.strategy_weight > weights.model_weight
    
    def test_performance_adjusted_weights(self):
        config = FusionConfig(enable_dynamic_weights=True)
        adjuster = DynamicWeightAdjuster(config)
        
        recent_perf = {
            "model_ic": 0.10,
            "strategy_return": 0.02,
        }
        
        weights = adjuster.get_weights(MarketRegime.NEUTRAL, recent_perf)
        
        assert weights.model_weight > 0.5


class TestConflictResolver:
    def test_strategy_only_signal(self):
        config = FusionConfig()
        resolver = ConflictResolver(config)
        
        signal = StrategySignalWrapper(
            code="600519.SH",
            name="贵州茅台",
            action="buy",
            strength=0.7,
            reason="测试信号",
            strategy_id="test",
            strategy_name="测试策略",
        )
        
        fused = resolver.resolve(None, signal, None)
        
        assert fused.action == "buy"
        assert fused.consistency == ConsistencyType.PARTIAL
        assert fused.strategy_weight == 1.0
    
    def test_model_only_signal(self):
        config = FusionConfig()
        resolver = ConflictResolver(config)
        
        pred = ModelPrediction(
            code="600519.SH",
            name="贵州茅台",
            score=0.75,
            confidence=0.8,
        )
        
        fused = resolver.resolve(pred, None, None)
        
        assert fused.action == "buy"
        assert fused.consistency == ConsistencyType.PARTIAL
        assert fused.model_weight == 1.0


class TestFusionEngine:
    def test_consistent_fusion(self):
        engine = FusionEngine()
        
        model_preds = {
            "600519.SH": ModelPrediction(
                code="600519.SH",
                name="贵州茅台",
                score=0.75,
                confidence=0.8,
            ),
        }
        
        strategy_signals = {
            "600519.SH": StrategySignalWrapper(
                code="600519.SH",
                name="贵州茅台",
                action="buy",
                strength=0.7,
                reason="突破信号",
                strategy_id="turtle",
                strategy_name="海龟交易",
            ),
        }
        
        fused_signals = engine.fuse(model_preds, strategy_signals)
        
        assert len(fused_signals) == 1
        assert fused_signals[0].consistency == ConsistencyType.CONSISTENT
        assert fused_signals[0].action == "buy"
        assert fused_signals[0].confidence > 0.7
    
    def test_conflict_fusion(self):
        engine = FusionEngine()
        
        model_preds = {
            "600519.SH": ModelPrediction(
                code="600519.SH",
                name="贵州茅台",
                score=0.75,
                confidence=0.8,
            ),
        }
        
        strategy_signals = {
            "600519.SH": StrategySignalWrapper(
                code="600519.SH",
                name="贵州茅台",
                action="sell",
                strength=0.7,
                reason="止损信号",
                strategy_id="turtle",
                strategy_name="海龟交易",
            ),
        }
        
        fused_signals = engine.fuse(model_preds, strategy_signals)
        
        assert len(fused_signals) == 1
        assert fused_signals[0].consistency == ConsistencyType.CONFLICT
    
    def test_partial_fusion_model_only(self):
        engine = FusionEngine()
        
        model_preds = {
            "600519.SH": ModelPrediction(
                code="600519.SH",
                name="贵州茅台",
                score=0.75,
                confidence=0.8,
            ),
        }
        
        strategy_signals = {}
        
        fused_signals = engine.fuse(model_preds, strategy_signals)
        
        assert len(fused_signals) == 1
        assert fused_signals[0].consistency == ConsistencyType.PARTIAL
    
    def test_partial_fusion_strategy_only(self):
        engine = FusionEngine()
        
        model_preds = {}
        
        strategy_signals = {
            "600519.SH": StrategySignalWrapper(
                code="600519.SH",
                name="贵州茅台",
                action="buy",
                strength=0.7,
                reason="突破信号",
                strategy_id="turtle",
                strategy_name="海龟交易",
            ),
        }
        
        fused_signals = engine.fuse(model_preds, strategy_signals)
        
        assert len(fused_signals) == 1
        assert fused_signals[0].consistency == ConsistencyType.PARTIAL
    
    def test_multiple_signals(self):
        engine = FusionEngine()
        
        model_preds = {
            "600519.SH": ModelPrediction(
                code="600519.SH",
                name="贵州茅台",
                score=0.75,
                confidence=0.8,
            ),
            "000858.SZ": ModelPrediction(
                code="000858.SZ",
                name="五粮液",
                score=0.35,
                confidence=0.7,
            ),
        }
        
        strategy_signals = {
            "600519.SH": StrategySignalWrapper(
                code="600519.SH",
                name="贵州茅台",
                action="buy",
                strength=0.7,
                reason="突破",
                strategy_id="turtle",
                strategy_name="海龟",
            ),
            "000858.SZ": StrategySignalWrapper(
                code="000858.SZ",
                name="五粮液",
                action="sell",
                strength=0.6,
                reason="止损",
                strategy_id="turtle",
                strategy_name="海龟",
            ),
        }
        
        fused_signals = engine.fuse(model_preds, strategy_signals)
        
        assert len(fused_signals) == 2
        
        maotai_signal = next(s for s in fused_signals if s.code == "600519.SH")
        wuliangye_signal = next(s for s in fused_signals if s.code == "000858.SZ")
        
        assert maotai_signal.consistency == ConsistencyType.CONSISTENT
        assert wuliangye_signal.consistency == ConsistencyType.CONSISTENT
    
    def test_market_context_adjustment(self):
        engine = FusionEngine()
        
        market_context = MarketContext(
            index_value=3100,
            index_change=0.02,
            volatility=0.015,
            trend="up",
            regime=MarketRegime.BULL_TRENDING,
            uncertainty=0.3,
        )
        
        model_preds = {
            "600519.SH": ModelPrediction(
                code="600519.SH",
                name="贵州茅台",
                score=0.75,
                confidence=0.8,
            ),
        }
        
        strategy_signals = {
            "600519.SH": StrategySignalWrapper(
                code="600519.SH",
                name="贵州茅台",
                action="buy",
                strength=0.7,
                reason="突破",
                strategy_id="turtle",
                strategy_name="海龟",
            ),
        }
        
        fused_signals = engine.fuse(model_preds, strategy_signals, market_context)
        
        assert len(fused_signals) == 1
        assert fused_signals[0].strategy_weight > fused_signals[0].model_weight
    
    def test_performance_update(self):
        engine = FusionEngine()
        
        engine.update_performance({
            "model_ic": 0.08,
            "strategy_return": 0.05,
        })
        
        stats = engine.get_fusion_statistics()
        
        assert isinstance(stats, dict)


class TestFusionStatistics:
    def test_fusion_statistics(self):
        engine = FusionEngine()
        
        for i in range(5):
            model_preds = {
                f"code{i}": ModelPrediction(
                    code=f"code{i}",
                    name=f"股票{i}",
                    score=0.6 + i * 0.05,
                    confidence=0.7,
                ),
            }
            
            strategy_signals = {
                f"code{i}": StrategySignalWrapper(
                    code=f"code{i}",
                    name=f"股票{i}",
                    action="buy",
                    strength=0.6,
                    reason="测试",
                    strategy_id="test",
                    strategy_name="测试",
                ),
            }
            
            engine.fuse(model_preds, strategy_signals)
        
        stats = engine.get_fusion_statistics()
        
        assert stats["total_fusions"] == 5
        assert "consistency_distribution" in stats
        assert "action_distribution" in stats


def test_create_fusion_engine():
    engine = create_fusion_engine()
    
    assert isinstance(engine, FusionEngine)
    assert isinstance(engine.config, FusionConfig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
