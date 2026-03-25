"""
分层融合系统完整测试 - Comprehensive Layered Fusion Tests

测试四层融合架构和增益效应验证。
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from monitor.strategy_library.fusion import (
    FusionConfig,
    ModelPrediction,
    StrategySignalWrapper,
    MarketContext,
    MarketRegime,
    ConsistencyType,
    Layer1SignalPreprocessor,
    Layer2ConsistencyAnalyzer,
    Layer3GainCalculator,
    Layer4DecisionMaker,
    GainValidator,
    LayeredFusionProcessor,
    FusionBacktester,
    FusionDataProvider,
    run_fusion_backtest,
)


class TestLayer1SignalPreprocessor:
    """测试Layer 1: 信号预处理"""

    def setup_method(self):
        self.preprocessor = Layer1SignalPreprocessor()

    def test_preprocess_model_prediction(self):
        pred = ModelPrediction(
            code="000001.SZ",
            name="平安银行",
            score=0.7,
            confidence=0.8,
            std=0.1,
            horizon=5,
        )
        
        normalized = self.preprocessor.preprocess_model_prediction(pred)
        
        assert normalized.code == "000001.SZ"
        assert normalized.source == "model"
        assert -1 <= normalized.normalized_score <= 1
        assert 0 <= normalized.confidence <= 1
        assert normalized.direction in ["up", "down", "neutral"]

    def test_preprocess_strategy_signal(self):
        signal = StrategySignalWrapper(
            code="000001.SZ",
            name="平安银行",
            action="buy",
            strength=0.8,
            reason="测试信号",
            strategy_id="multi_factor",
            strategy_name="多因子策略",
        )
        
        normalized = self.preprocessor.preprocess_strategy_signal(signal)
        
        assert normalized.code == "000001.SZ"
        assert normalized.source == "strategy"
        assert normalized.normalized_score > 0
        assert normalized.direction == "up"

    def test_normalize_score_range(self):
        for score in [0.3, 0.5, 0.7, 0.9]:
            pred = ModelPrediction(
                code="test",
                name="test",
                score=score,
                confidence=0.5,
            )
            normalized = self.preprocessor.preprocess_model_prediction(pred)
            assert -1 <= normalized.normalized_score <= 1

    def test_signal_alignment(self):
        model_signals = {
            "000001.SZ": self.preprocessor.preprocess_model_prediction(
                ModelPrediction(code="000001.SZ", name="A", score=0.6, confidence=0.5)
            ),
            "000002.SZ": self.preprocessor.preprocess_model_prediction(
                ModelPrediction(code="000002.SZ", name="B", score=0.5, confidence=0.5)
            ),
        }
        
        strategy_signals = {
            "000001.SZ": self.preprocessor.preprocess_strategy_signal(
                StrategySignalWrapper(
                    code="000001.SZ", name="A", action="buy", strength=0.6,
                    reason="", strategy_id="test", strategy_name="test"
                )
            ),
            "000003.SZ": self.preprocessor.preprocess_strategy_signal(
                StrategySignalWrapper(
                    code="000003.SZ", name="C", action="sell", strength=0.7,
                    reason="", strategy_id="test", strategy_name="test"
                )
            ),
        }
        
        aligned_model, aligned_strategy = self.preprocessor.align_signals(
            model_signals, strategy_signals
        )
        
        assert "000001.SZ" in aligned_model
        assert "000002.SZ" in aligned_model
        assert "000001.SZ" in aligned_strategy
        assert "000003.SZ" in aligned_strategy


class TestLayer2ConsistencyAnalyzer:
    """测试Layer 2: 一致性分析"""

    def setup_method(self):
        self.analyzer = Layer2ConsistencyAnalyzer()
        self.preprocessor = Layer1SignalPreprocessor()

    def test_consistent_signals(self):
        model_sig = self.preprocessor.preprocess_model_prediction(
            ModelPrediction(code="test", name="test", score=0.7, confidence=0.8)
        )
        strategy_sig = self.preprocessor.preprocess_strategy_signal(
            StrategySignalWrapper(
                code="test", name="test", action="buy", strength=0.8,
                reason="", strategy_id="test", strategy_name="test"
            )
        )
        
        result = self.analyzer.analyze(model_sig, strategy_sig)
        
        assert result.consistency_type == ConsistencyType.CONSISTENT
        assert result.direction_consistency == 1.0
        assert result.overall_consistency >= 0.7

    def test_conflict_signals(self):
        model_sig = self.preprocessor.preprocess_model_prediction(
            ModelPrediction(code="test", name="test", score=0.7, confidence=0.8)
        )
        strategy_sig = self.preprocessor.preprocess_strategy_signal(
            StrategySignalWrapper(
                code="test", name="test", action="sell", strength=0.8,
                reason="", strategy_id="test", strategy_name="test"
            )
        )
        
        result = self.analyzer.analyze(model_sig, strategy_sig)
        
        assert result.consistency_type == ConsistencyType.CONFLICT
        assert result.direction_consistency == 0.0

    def test_partial_signals(self):
        model_sig = self.preprocessor.preprocess_model_prediction(
            ModelPrediction(code="test", name="test", score=0.7, confidence=0.8)
        )
        
        result = self.analyzer.analyze(model_sig, None)
        
        assert result.consistency_type == ConsistencyType.PARTIAL
        assert result.overall_consistency == 0.5

    def test_no_signals(self):
        result = self.analyzer.analyze(None, None)
        
        assert result.consistency_type == ConsistencyType.NO_SIGNAL
        assert result.overall_consistency == 0


class TestLayer3GainCalculator:
    """测试Layer 3: 增益计算"""

    def setup_method(self):
        self.config = FusionConfig()
        self.calculator = Layer3GainCalculator(self.config)
        self.preprocessor = Layer1SignalPreprocessor()

    def test_calculate_gain_consistent(self):
        model_sig = self.preprocessor.preprocess_model_prediction(
            ModelPrediction(code="test", name="test", score=0.7, confidence=0.8)
        )
        strategy_sig = self.preprocessor.preprocess_strategy_signal(
            StrategySignalWrapper(
                code="test", name="test", action="buy", strength=0.8,
                reason="", strategy_id="test", strategy_name="test"
            )
        )
        
        from monitor.strategy_library.fusion.layered_fusion import ConsistencyResult
        consistency = ConsistencyResult(
            direction_consistency=1.0,
            strength_consistency=0.8,
            time_consistency=0.9,
            overall_consistency=0.9,
            consistency_type=ConsistencyType.CONSISTENT,
        )
        
        from monitor.strategy_library.fusion import FusionWeights
        weights = FusionWeights(model_weight=0.5, strategy_weight=0.5)
        
        gain, components = self.calculator.calculate_gain(
            model_sig, strategy_sig, consistency, MarketRegime.BULL_TRENDING, weights
        )
        
        assert gain > 0
        assert components["consistency_gain"] > 0
        assert components["total_gain"] == gain

    def test_regime_gain_bull_trending(self):
        model_sig = self.preprocessor.preprocess_model_prediction(
            ModelPrediction(code="test", name="test", score=0.7, confidence=0.8)
        )
        strategy_sig = self.preprocessor.preprocess_strategy_signal(
            StrategySignalWrapper(
                code="test", name="test", action="buy", strength=0.8,
                reason="", strategy_id="test", strategy_name="test"
            )
        )
        
        from monitor.strategy_library.fusion.layered_fusion import ConsistencyResult
        consistency = ConsistencyResult(
            direction_consistency=1.0,
            strength_consistency=0.8,
            time_consistency=0.9,
            overall_consistency=0.9,
            consistency_type=ConsistencyType.CONSISTENT,
        )
        
        from monitor.strategy_library.fusion import FusionWeights
        weights = FusionWeights(model_weight=0.5, strategy_weight=0.5)
        
        gain_bull, _ = self.calculator.calculate_gain(
            model_sig, strategy_sig, consistency, MarketRegime.BULL_TRENDING, weights
        )
        
        gain_bear, _ = self.calculator.calculate_gain(
            model_sig, strategy_sig, consistency, MarketRegime.BEAR_TRENDING, weights
        )
        
        assert gain_bull > gain_bear

    def test_gain_statistics(self):
        model_sig = self.preprocessor.preprocess_model_prediction(
            ModelPrediction(code="test", name="test", score=0.7, confidence=0.8)
        )
        strategy_sig = self.preprocessor.preprocess_strategy_signal(
            StrategySignalWrapper(
                code="test", name="test", action="buy", strength=0.8,
                reason="", strategy_id="test", strategy_name="test"
            )
        )
        
        from monitor.strategy_library.fusion.layered_fusion import ConsistencyResult
        consistency = ConsistencyResult(
            direction_consistency=1.0,
            strength_consistency=0.8,
            time_consistency=0.9,
            overall_consistency=0.9,
            consistency_type=ConsistencyType.CONSISTENT,
        )
        
        from monitor.strategy_library.fusion import FusionWeights
        weights = FusionWeights(model_weight=0.5, strategy_weight=0.5)
        
        for _ in range(5):
            self.calculator.calculate_gain(
                model_sig, strategy_sig, consistency, MarketRegime.NEUTRAL, weights
            )
        
        stats = self.calculator.get_gain_statistics()
        
        assert stats["total_calculations"] == 5
        assert "avg_total_gain" in stats


class TestLayer4DecisionMaker:
    """测试Layer 4: 决策输出"""

    def setup_method(self):
        self.decision_maker = Layer4DecisionMaker()

    def test_make_buy_decision(self):
        from monitor.strategy_library.fusion import FusedSignal
        signal = FusedSignal(
            code="000001.SZ",
            name="平安银行",
            action="buy",
            confidence=0.8,
            fused_score=0.6,
            consistency=ConsistencyType.CONSISTENT,
            model_contribution=0.3,
            strategy_contribution=0.3,
            model_weight=0.5,
            strategy_weight=0.5,
        )
        
        gain = 0.5
        components = {
            "base_gain": 0.4,
            "consistency_gain": 0.1,
            "regime_gain": 0,
            "synergy_gain": 0,
        }
        
        decision = self.decision_maker.make_decision(signal, gain, components, None)
        
        assert decision.signal.action == "buy"
        assert decision.position_ratio > 0
        assert decision.stop_loss > 0
        assert decision.take_profit > 0
        assert decision.execution_priority >= 1

    def test_position_sizing(self):
        from monitor.strategy_library.fusion import FusedSignal
        
        high_conf_signal = FusedSignal(
            code="test", name="test", action="buy", confidence=0.9,
            fused_score=0.7, consistency=ConsistencyType.CONSISTENT,
            model_contribution=0.3, strategy_contribution=0.3,
            model_weight=0.5, strategy_weight=0.5,
        )
        
        low_conf_signal = FusedSignal(
            code="test", name="test", action="buy", confidence=0.2,
            fused_score=0.3, consistency=ConsistencyType.PARTIAL,
            model_contribution=0.3, strategy_contribution=0.3,
            model_weight=0.5, strategy_weight=0.5,
        )
        
        high_decision = self.decision_maker.make_decision(
            high_conf_signal, 0.7, {"base_gain": 0.5}, None
        )
        low_decision = self.decision_maker.make_decision(
            low_conf_signal, 0.2, {"base_gain": 0.1}, None
        )
        
        assert high_decision.position_ratio > low_decision.position_ratio

    def test_priority_calculation(self):
        from monitor.strategy_library.fusion import FusedSignal
        
        consistent_signal = FusedSignal(
            code="test", name="test", action="buy", confidence=0.9,
            fused_score=0.7, consistency=ConsistencyType.CONSISTENT,
            model_contribution=0.3, strategy_contribution=0.3,
            model_weight=0.5, strategy_weight=0.5,
        )
        
        conflict_signal = FusedSignal(
            code="test", name="test", action="hold", confidence=0.3,
            fused_score=0.0, consistency=ConsistencyType.CONFLICT,
            model_contribution=0.0, strategy_contribution=0.0,
            model_weight=0.5, strategy_weight=0.5,
        )
        
        consistent_decision = self.decision_maker.make_decision(
            consistent_signal, 0.7, {"base_gain": 0.5}, None
        )
        conflict_decision = self.decision_maker.make_decision(
            conflict_signal, 0.0, {"base_gain": 0.0}, None
        )
        
        assert consistent_decision.execution_priority > conflict_decision.execution_priority


class TestGainValidator:
    """测试增益效应验证器"""

    def setup_method(self):
        self.validator = GainValidator()

    def test_validate_positive_gain(self):
        np.random.seed(42)
        
        model_returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        strategy_returns = pd.Series(np.random.normal(0.0008, 0.018, 100))
        
        fused_returns = pd.Series(np.random.normal(0.0015, 0.015, 100))
        
        metrics = self.validator.validate(
            model_returns, strategy_returns, fused_returns
        )
        
        assert isinstance(metrics.return_gain, float)
        assert isinstance(metrics.sharpe_gain, float)
        assert isinstance(metrics.drawdown_gain, float)
        assert isinstance(metrics.winrate_gain, float)

    def test_has_gain_detection(self):
        model_returns = pd.Series([0.01] * 50)
        strategy_returns = pd.Series([0.008] * 50)
        fused_returns = pd.Series([0.015] * 50)
        
        metrics = self.validator.validate(
            model_returns, strategy_returns, fused_returns
        )
        
        assert metrics.has_gain() == True
        assert metrics.return_gain > 0

    def test_no_gain_detection(self):
        model_returns = pd.Series([0.02] * 50)
        strategy_returns = pd.Series([0.018] * 50)
        fused_returns = pd.Series([0.01] * 50)
        
        metrics = self.validator.validate(
            model_returns, strategy_returns, fused_returns
        )
        
        assert metrics.return_gain < 0


class TestLayeredFusionProcessor:
    """测试完整的分层融合处理器"""

    def setup_method(self):
        self.config = FusionConfig()
        self.processor = LayeredFusionProcessor(self.config)

    def test_process_signals(self):
        model_predictions = {
            "000001.SZ": ModelPrediction(
                code="000001.SZ",
                name="平安银行",
                score=0.7,
                confidence=0.8,
                horizon=5,
            ),
        }
        
        strategy_signals = {
            "000001.SZ": StrategySignalWrapper(
                code="000001.SZ",
                name="平安银行",
                action="buy",
                strength=0.7,
                reason="测试信号",
                strategy_id="multi_factor",
                strategy_name="多因子策略",
            ),
        }
        
        market_context = MarketContext(
            index_value=3000,
            index_change=0.01,
            volatility=0.02,
            trend="up",
            regime=MarketRegime.BULL_TRENDING,
            uncertainty=0.3,
        )
        
        decisions = self.processor.process(
            model_predictions, strategy_signals, market_context
        )
        
        assert len(decisions) > 0
        assert decisions[0].signal.code == "000001.SZ"
        assert decisions[0].signal.action in ["buy", "sell", "hold"]

    def test_process_with_conflict(self):
        model_predictions = {
            "000001.SZ": ModelPrediction(
                code="000001.SZ",
                name="平安银行",
                score=0.7,
                confidence=0.8,
            ),
        }
        
        strategy_signals = {
            "000001.SZ": StrategySignalWrapper(
                code="000001.SZ",
                name="平安银行",
                action="sell",
                strength=0.8,
                reason="反向信号",
                strategy_id="test",
                strategy_name="test",
            ),
        }
        
        decisions = self.processor.process(model_predictions, strategy_signals)
        
        assert len(decisions) > 0
        assert decisions[0].signal.consistency == ConsistencyType.CONFLICT

    def test_statistics_tracking(self):
        for i in range(5):
            model_predictions = {
                f"code{i}": ModelPrediction(
                    code=f"code{i}",
                    name=f"股票{i}",
                    score=0.6,
                    confidence=0.7,
                ),
            }
            
            strategy_signals = {
                f"code{i}": StrategySignalWrapper(
                    code=f"code{i}",
                    name=f"股票{i}",
                    action="buy",
                    strength=0.6,
                    reason="test",
                    strategy_id="test",
                    strategy_name="test",
                ),
            }
            
            self.processor.process(model_predictions, strategy_signals)
        
        stats = self.processor.get_statistics()
        
        assert stats["total_decisions"] == 5
        assert "avg_position_ratio" in stats
        assert "confidence_distribution" in stats


class TestFusionBacktester:
    """测试融合回测系统"""

    def setup_method(self):
        self.backtester = FusionBacktester(
            initial_capital=1000000,
            commission_rate=0.0003,
            slippage=0.001,
        )

    def test_backtest_execution(self):
        dates = pd.date_range("2024-01-01", "2024-01-31", freq="B")
        codes = ["000001.SZ", "000002.SZ"]
        
        price_records = []
        for date in dates:
            for code in codes:
                price_records.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "code": code,
                    "open": 10.0,
                    "high": 10.5,
                    "low": 9.5,
                    "close": 10.0,
                    "volume": 1000000,
                })
        
        price_data = pd.DataFrame(price_records)
        
        model_predictions = {}
        for code in codes:
            model_predictions[code] = [
                ModelPrediction(
                    code=code,
                    name=f"股票{code}",
                    score=0.65,
                    confidence=0.7,
                    timestamp=date.strftime("%Y-%m-%dT09:30:00"),
                )
                for date in dates[::5]
            ]
        
        strategy_signals = {}
        for code in codes:
            strategy_signals[code] = [
                StrategySignalWrapper(
                    code=code,
                    name=f"股票{code}",
                    action="buy",
                    strength=0.6,
                    reason="test",
                    strategy_id="test",
                    strategy_name="test",
                    timestamp=date.strftime("%Y-%m-%dT15:00:00"),
                )
                for date in dates[::7]
            ]
        
        fused_result, model_result, strategy_result, gain_metrics = self.backtester.run_backtest(
            price_data=price_data,
            model_predictions=model_predictions,
            strategy_signals=strategy_signals,
        )
        
        assert fused_result.source == "fused"
        assert model_result.source == "model"
        assert strategy_result.source == "strategy"
        assert isinstance(gain_metrics.return_gain, float)


class TestFusionDataProvider:
    """测试数据提供者"""

    def setup_method(self):
        self.provider = FusionDataProvider()

    def test_get_daily_data(self):
        model_preds, strategy_sigs, market_ctx = self.provider.get_daily_data(
            date="2024-01-15"
        )
        
        assert isinstance(model_preds, dict)
        assert isinstance(strategy_sigs, dict)
        assert market_ctx is not None

    def test_prepare_backtest_data(self):
        price_data, model_preds, strategy_sigs, market_ctxs = self.provider.prepare_backtest_data(
            start_date="2024-01-01",
            end_date="2024-01-10",
        )
        
        assert isinstance(price_data, pd.DataFrame)
        assert isinstance(model_preds, dict)
        assert isinstance(strategy_sigs, dict)
        assert isinstance(market_ctxs, dict)

    def test_mock_data_generation(self):
        model_preds = self.provider.model_provider.get_predictions("2024-01-15")
        
        assert len(model_preds) > 0
        for code, pred in model_preds.items():
            assert pred.code == code
            assert 0 <= pred.score <= 1
            assert 0 <= pred.confidence <= 1


class TestIntegration:
    """集成测试"""

    def test_full_fusion_workflow(self):
        config = FusionConfig(
            consistency_boost=1.3,
            conflict_penalty=0.7,
        )
        
        processor = LayeredFusionProcessor(config)
        
        model_predictions = {
            "000001.SZ": ModelPrediction(
                code="000001.SZ",
                name="平安银行",
                score=0.72,
                confidence=0.85,
                horizon=5,
            ),
            "600036.SH": ModelPrediction(
                code="600036.SH",
                name="招商银行",
                score=0.75,
                confidence=0.7,
                horizon=10,
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
            "600036.SH": StrategySignalWrapper(
                code="600036.SH",
                name="招商银行",
                action="sell",
                strength=0.6,
                reason="趋势策略看空",
                strategy_id="turtle_trading",
                strategy_name="海龟策略",
            ),
        }
        
        market_context = MarketContext(
            index_value=3050,
            index_change=0.015,
            volatility=0.018,
            trend="up",
            regime=MarketRegime.BULL_TRENDING,
            uncertainty=0.35,
            north_money_flow=25.5,
            sentiment_score=0.65,
        )
        
        decisions = processor.process(
            model_predictions, strategy_signals, market_context
        )
        
        assert len(decisions) == 2
        
        decision_000001 = next(d for d in decisions if d.signal.code == "000001.SZ")
        assert decision_000001.signal.consistency == ConsistencyType.CONSISTENT
        assert decision_000001.signal.action == "buy"
        assert decision_000001.execution_priority >= 7
        
        decision_600036 = next(d for d in decisions if d.signal.code == "600036.SH")
        assert decision_600036.signal.consistency == ConsistencyType.CONFLICT
        assert decision_600036.execution_priority <= 5
        
        stats = processor.get_statistics()
        assert stats["total_decisions"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
