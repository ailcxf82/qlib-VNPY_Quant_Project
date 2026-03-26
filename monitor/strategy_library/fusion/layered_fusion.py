"""
分层融合处理器 - Layered Fusion Processor

实现四层融合架构，确保增益效应：
Layer 1: 信号预处理 - 标准化、去噪、对齐
Layer 2: 一致性分析 - 方向、强度、时效一致性检测
Layer 3: 增益计算 - 市场状态、动态权重、增益计算
Layer 4: 决策输出 - 最终信号、仓位、止损止盈
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .fusion_engine import (
    ConsistencyType,
    MarketRegime,
    ModelPrediction,
    StrategySignalWrapper,
    FusedSignal,
    MarketContext,
    FusionWeights,
    FusionConfig,
    MarketRegimeDetector,
)

logger = logging.getLogger(__name__)


class SignalStrength(Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    WEAK_BUY = "weak_buy"
    NEUTRAL = "neutral"
    WEAK_SELL = "weak_sell"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class TimeHorizon(Enum):
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"


@dataclass
class NormalizedSignal:
    code: str
    name: str
    source: str
    normalized_score: float
    confidence: float
    direction: str
    strength: SignalStrength
    horizon: TimeHorizon
    noise_level: float
    timestamp: str
    raw_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "source": self.source,
            "normalized_score": self.normalized_score,
            "confidence": self.confidence,
            "direction": self.direction,
            "strength": self.strength.value,
            "horizon": self.horizon.value,
            "noise_level": self.noise_level,
            "timestamp": self.timestamp,
        }


@dataclass
class ConsistencyResult:
    direction_consistency: float
    strength_consistency: float
    time_consistency: float
    overall_consistency: float
    consistency_type: ConsistencyType
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "direction_consistency": self.direction_consistency,
            "strength_consistency": self.strength_consistency,
            "time_consistency": self.time_consistency,
            "overall_consistency": self.overall_consistency,
            "consistency_type": self.consistency_type.value,
            "details": self.details,
        }


@dataclass
class GainMetrics:
    return_gain: float
    sharpe_gain: float
    drawdown_gain: float
    winrate_gain: float
    information_ratio: float
    calmar_gain: float
    sortino_gain: float
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "return_gain": self.return_gain,
            "sharpe_gain": self.sharpe_gain,
            "drawdown_gain": self.drawdown_gain,
            "winrate_gain": self.winrate_gain,
            "information_ratio": self.information_ratio,
            "calmar_gain": self.calmar_gain,
            "sortino_gain": self.sortino_gain,
            "details": self.details,
        }

    def has_gain(self) -> bool:
        return (
            self.return_gain > 0 or
            self.sharpe_gain > 0 or
            self.drawdown_gain < 0 or
            self.winrate_gain > 0
        )


@dataclass
class FusionDecision:
    signal: FusedSignal
    position_ratio: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    gain_potential: float
    confidence_level: str
    execution_priority: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal": self.signal.to_dict(),
            "position_ratio": self.position_ratio,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "risk_reward_ratio": self.risk_reward_ratio,
            "gain_potential": self.gain_potential,
            "confidence_level": self.confidence_level,
            "execution_priority": self.execution_priority,
            "metadata": self.metadata,
        }


class Layer1SignalPreprocessor:
    """Layer 1: 信号预处理"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.noise_threshold = self.config.get("noise_threshold", 0.1)
        self.smoothing_window = self.config.get("smoothing_window", 3)

    def preprocess_model_prediction(self, pred: ModelPrediction) -> NormalizedSignal:
        normalized_score = self._normalize_score(pred.score, "model")
        confidence = self._adjust_confidence(pred.confidence, pred.std)
        direction = self._get_direction(normalized_score)
        strength = self._get_strength(normalized_score, confidence)
        horizon = self._get_horizon(pred.horizon)
        noise_level = self._estimate_noise(pred.std)

        return NormalizedSignal(
            code=pred.code,
            name=pred.name,
            source="model",
            normalized_score=normalized_score,
            confidence=confidence,
            direction=direction,
            strength=strength,
            horizon=horizon,
            noise_level=noise_level,
            timestamp=pred.timestamp,
            raw_data=pred.to_dict(),
        )

    def preprocess_strategy_signal(self, signal: StrategySignalWrapper) -> NormalizedSignal:
        normalized_score = self._normalize_strategy_score(signal)
        confidence = signal.strength
        direction = self._get_direction(normalized_score)
        strength = self._get_strength(normalized_score, confidence)
        horizon = self._infer_horizon(signal.strategy_id)
        noise_level = self._estimate_strategy_noise(signal)

        return NormalizedSignal(
            code=signal.code,
            name=signal.name,
            source="strategy",
            normalized_score=normalized_score,
            confidence=confidence,
            direction=direction,
            strength=strength,
            horizon=horizon,
            noise_level=noise_level,
            timestamp=signal.timestamp,
            raw_data=signal.to_dict(),
        )

    def _normalize_score(self, score: float, source: str) -> float:
        return (score - 0.5) * 2

    def _normalize_strategy_score(self, signal: StrategySignalWrapper) -> float:
        base_score = signal.strength
        if signal.action == "sell":
            base_score = -base_score
        return np.clip(base_score, -1, 1)

    def _adjust_confidence(self, confidence: float, std: float) -> float:
        uncertainty_penalty = min(std * 2, 0.3)
        return max(confidence - uncertainty_penalty, 0.1)

    def _get_direction(self, score: float) -> str:
        if score > 0.15:
            return "up"
        elif score < -0.15:
            return "down"
        return "neutral"

    def _get_strength(self, score: float, confidence: float) -> SignalStrength:
        abs_score = abs(score) * confidence
        
        if abs_score > 0.7:
            return SignalStrength.STRONG_BUY if score > 0 else SignalStrength.STRONG_SELL
        elif abs_score > 0.5:
            return SignalStrength.BUY if score > 0 else SignalStrength.SELL
        elif abs_score > 0.3:
            return SignalStrength.WEAK_BUY if score > 0 else SignalStrength.WEAK_SELL
        else:
            return SignalStrength.NEUTRAL

    def _get_horizon(self, horizon_days: int) -> TimeHorizon:
        if horizon_days <= 5:
            return TimeHorizon.SHORT_TERM
        elif horizon_days <= 20:
            return TimeHorizon.MEDIUM_TERM
        return TimeHorizon.LONG_TERM

    def _infer_horizon(self, strategy_id: str) -> TimeHorizon:
        short_term_strategies = ["grid_trading", "day_trading"]
        long_term_strategies = ["turtle_trading", "sector_rotation"]
        
        if strategy_id in short_term_strategies:
            return TimeHorizon.SHORT_TERM
        elif strategy_id in long_term_strategies:
            return TimeHorizon.LONG_TERM
        return TimeHorizon.MEDIUM_TERM

    def _estimate_noise(self, std: float) -> float:
        return min(std * 3, 1.0)

    def _estimate_strategy_noise(self, signal: StrategySignalWrapper) -> float:
        base_noise = 0.1
        if signal.strength < 0.3:
            base_noise += 0.2
        return min(base_noise, 0.5)

    def align_signals(
        self,
        model_signals: Dict[str, NormalizedSignal],
        strategy_signals: Dict[str, NormalizedSignal],
    ) -> Tuple[Dict[str, NormalizedSignal], Dict[str, NormalizedSignal]]:
        aligned_model = {}
        aligned_strategy = {}
        
        all_codes = set(model_signals.keys()) | set(strategy_signals.keys())
        
        for code in all_codes:
            if code in model_signals:
                aligned_model[code] = model_signals[code]
            if code in strategy_signals:
                aligned_strategy[code] = strategy_signals[code]
        
        return aligned_model, aligned_strategy


class Layer2ConsistencyAnalyzer:
    """Layer 2: 一致性分析"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.direction_threshold = self.config.get("direction_threshold", 0.15)
        self.strength_threshold = self.config.get("strength_threshold", 0.3)

    def analyze(
        self,
        model_signal: Optional[NormalizedSignal],
        strategy_signal: Optional[NormalizedSignal],
    ) -> ConsistencyResult:
        if model_signal is None and strategy_signal is None:
            return ConsistencyResult(
                direction_consistency=0,
                strength_consistency=0,
                time_consistency=0,
                overall_consistency=0,
                consistency_type=ConsistencyType.NO_SIGNAL,
            )

        if model_signal is None or strategy_signal is None:
            return ConsistencyResult(
                direction_consistency=0.5,
                strength_consistency=0.5,
                time_consistency=0.5,
                overall_consistency=0.5,
                consistency_type=ConsistencyType.PARTIAL,
                details={"missing_source": "model" if model_signal is None else "strategy"},
            )

        direction_consistency = self._check_direction_consistency(model_signal, strategy_signal)
        strength_consistency = self._check_strength_consistency(model_signal, strategy_signal)
        time_consistency = self._check_time_consistency(model_signal, strategy_signal)
        
        overall = self._calculate_overall_consistency(
            direction_consistency, strength_consistency, time_consistency
        )
        
        consistency_type = self._determine_consistency_type(overall)

        return ConsistencyResult(
            direction_consistency=direction_consistency,
            strength_consistency=strength_consistency,
            time_consistency=time_consistency,
            overall_consistency=overall,
            consistency_type=consistency_type,
            details={
                "model_direction": model_signal.direction,
                "strategy_direction": strategy_signal.direction,
                "model_strength": model_signal.strength.value,
                "strategy_strength": strategy_signal.strength.value,
            },
        )

    def _check_direction_consistency(
        self, model: NormalizedSignal, strategy: NormalizedSignal
    ) -> float:
        if model.direction == strategy.direction:
            return 1.0
        elif model.direction == "neutral" or strategy.direction == "neutral":
            return 0.5
        return 0.0

    def _check_strength_consistency(
        self, model: NormalizedSignal, strategy: NormalizedSignal
    ) -> float:
        score_diff = abs(model.normalized_score - strategy.normalized_score)
        return max(1 - score_diff / 2, 0)

    def _check_time_consistency(
        self, model: NormalizedSignal, strategy: NormalizedSignal
    ) -> float:
        horizon_map = {
            TimeHorizon.SHORT_TERM: 1,
            TimeHorizon.MEDIUM_TERM: 2,
            TimeHorizon.LONG_TERM: 3,
        }
        
        model_horizon = horizon_map.get(model.horizon, 2)
        strategy_horizon = horizon_map.get(strategy.horizon, 2)
        
        diff = abs(model_horizon - strategy_horizon)
        return max(1 - diff * 0.3, 0.4)

    def _calculate_overall_consistency(
        self, direction: float, strength: float, time: float
    ) -> float:
        weights = {
            "direction": 0.5,
            "strength": 0.3,
            "time": 0.2,
        }
        return (
            direction * weights["direction"] +
            strength * weights["strength"] +
            time * weights["time"]
        )

    def _determine_consistency_type(self, overall: float) -> ConsistencyType:
        if overall >= 0.7:
            return ConsistencyType.CONSISTENT
        elif overall >= 0.4:
            return ConsistencyType.PARTIAL
        return ConsistencyType.CONFLICT


class Layer3GainCalculator:
    """Layer 3: 增益计算"""

    def __init__(self, config: FusionConfig):
        self.config = config
        self.regime_detector = MarketRegimeDetector()
        self._gain_history: List[Dict] = []

    def calculate_gain(
        self,
        model_signal: Optional[NormalizedSignal],
        strategy_signal: Optional[NormalizedSignal],
        consistency: ConsistencyResult,
        regime: MarketRegime,
        weights: FusionWeights,
    ) -> Tuple[float, Dict[str, float]]:
        base_gain = self._calculate_base_gain(model_signal, strategy_signal, weights)
        
        consistency_gain = self._calculate_consistency_gain(consistency)
        regime_gain = self._calculate_regime_gain(regime, model_signal, strategy_signal)
        synergy_gain = self._calculate_synergy_gain(model_signal, strategy_signal, consistency)
        
        total_gain = base_gain * (1 + consistency_gain + regime_gain + synergy_gain)
        
        gain_components = {
            "base_gain": base_gain,
            "consistency_gain": consistency_gain,
            "regime_gain": regime_gain,
            "synergy_gain": synergy_gain,
            "total_gain": total_gain,
        }
        
        self._record_gain(gain_components)
        
        return total_gain, gain_components

    def _calculate_base_gain(
        self,
        model: Optional[NormalizedSignal],
        strategy: Optional[NormalizedSignal],
        weights: FusionWeights,
    ) -> float:
        model_score = model.normalized_score if model else 0
        strategy_score = strategy.normalized_score if strategy else 0
        model_conf = model.confidence if model else 0
        strategy_conf = strategy.confidence if strategy else 0
        
        weighted_score = (
            weights.model_weight * model_score * model_conf +
            weights.strategy_weight * strategy_score * strategy_conf
        )
        
        return weighted_score

    def _calculate_consistency_gain(self, consistency: ConsistencyResult) -> float:
        if consistency.consistency_type == ConsistencyType.CONSISTENT:
            return self.config.consistency_boost - 1
        elif consistency.consistency_type == ConsistencyType.CONFLICT:
            return self.config.conflict_penalty - 1
        return 0

    def _calculate_regime_gain(
        self,
        regime: MarketRegime,
        model: Optional[NormalizedSignal],
        strategy: Optional[NormalizedSignal],
    ) -> float:
        regime_gain_map = {
            MarketRegime.BULL_TRENDING: {
                "up": 0.15,
                "down": -0.1,
                "neutral": 0,
            },
            MarketRegime.BULL_CHOPPY: {
                "up": 0.05,
                "down": -0.05,
                "neutral": 0,
            },
            MarketRegime.BEAR_TRENDING: {
                "up": -0.1,
                "down": 0.15,
                "neutral": 0,
            },
            MarketRegime.BEAR_CHOPPY: {
                "up": -0.05,
                "down": 0.05,
                "neutral": 0,
            },
            MarketRegime.NEUTRAL: {
                "up": 0,
                "down": 0,
                "neutral": 0,
            },
        }
        
        direction = "neutral"
        if model and strategy:
            if model.direction == strategy.direction:
                direction = model.direction
        elif model:
            direction = model.direction
        elif strategy:
            direction = strategy.direction
        
        return regime_gain_map.get(regime, {}).get(direction, 0)

    def _calculate_synergy_gain(
        self,
        model: Optional[NormalizedSignal],
        strategy: Optional[NormalizedSignal],
        consistency: ConsistencyResult,
    ) -> float:
        if model is None or strategy is None:
            return 0
        
        if consistency.consistency_type != ConsistencyType.CONSISTENT:
            return 0
        
        confidence_product = model.confidence * strategy.confidence
        strength_alignment = 1 - abs(abs(model.normalized_score) - abs(strategy.normalized_score)) / 2
        
        synergy = confidence_product * strength_alignment * 0.2
        
        return synergy

    def _record_gain(self, components: Dict[str, float]):
        self._gain_history.append({
            "timestamp": datetime.now().isoformat(),
            **components,
        })
        
        if len(self._gain_history) > 500:
            self._gain_history = self._gain_history[-500:]

    def get_gain_statistics(self) -> Dict[str, Any]:
        if not self._gain_history:
            return {}
        
        df = pd.DataFrame(self._gain_history)
        
        return {
            "total_calculations": len(self._gain_history),
            "avg_total_gain": df["total_gain"].mean(),
            "avg_consistency_gain": df["consistency_gain"].mean(),
            "avg_regime_gain": df["regime_gain"].mean(),
            "avg_synergy_gain": df["synergy_gain"].mean(),
            "positive_gain_ratio": (df["total_gain"] > 0).mean(),
        }


class Layer4DecisionMaker:
    """Layer 4: 决策输出"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.max_position = self.config.get("max_position", 1.0)
        self.min_position = self.config.get("min_position", 0.1)
        self.default_stop_loss = self.config.get("default_stop_loss", 0.08)
        self.default_take_profit = self.config.get("default_take_profit", 0.15)

    def make_decision(
        self,
        fused_signal: FusedSignal,
        gain: float,
        gain_components: Dict[str, float],
        market_context: Optional[MarketContext],
    ) -> FusionDecision:
        position_ratio = self._calculate_position(gain, fused_signal.confidence)
        stop_loss, take_profit = self._calculate_stops(
            gain, fused_signal, market_context
        )
        risk_reward = self._calculate_risk_reward(stop_loss, take_profit, fused_signal.action)
        gain_potential = self._estimate_gain_potential(gain, gain_components)
        confidence_level = self._determine_confidence_level(fused_signal.confidence)
        priority = self._calculate_priority(fused_signal, gain, confidence_level)

        return FusionDecision(
            signal=fused_signal,
            position_ratio=position_ratio,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=risk_reward,
            gain_potential=gain_potential,
            confidence_level=confidence_level,
            execution_priority=priority,
            metadata={
                "gain_components": gain_components,
                "market_regime": market_context.regime.value if market_context else "unknown",
            },
        )

    def _calculate_position(self, gain: float, confidence: float) -> float:
        base_position = abs(gain) * confidence
        
        position = np.clip(base_position, self.min_position, self.max_position)
        
        if confidence < 0.3:
            position *= 0.5
        elif confidence > 0.7:
            position *= 1.2
        
        return min(position, self.max_position)

    def _calculate_stops(
        self,
        gain: float,
        signal: FusedSignal,
        market_context: Optional[MarketContext],
    ) -> Tuple[float, float]:
        volatility = market_context.volatility if market_context else 0.02
        
        if signal.action == "buy":
            stop_loss = self.default_stop_loss * (1 + volatility * 5)
            take_profit = self.default_take_profit * (1 + abs(gain))
        elif signal.action == "sell":
            stop_loss = self.default_stop_loss * (1 + volatility * 5)
            take_profit = self.default_take_profit * (1 + abs(gain))
        else:
            stop_loss = 0.05
            take_profit = 0.05
        
        return stop_loss, take_profit

    def _calculate_risk_reward(
        self, stop_loss: float, take_profit: float, action: str
    ) -> float:
        if action == "hold":
            return 0
        return take_profit / stop_loss if stop_loss > 0 else 0

    def _estimate_gain_potential(
        self, gain: float, components: Dict[str, float]
    ) -> float:
        base_potential = abs(gain)
        
        consistency_bonus = max(components.get("consistency_gain", 0), 0) * 0.5
        regime_bonus = max(components.get("regime_gain", 0), 0) * 0.3
        synergy_bonus = components.get("synergy_gain", 0) * 0.2
        
        return base_potential + consistency_bonus + regime_bonus + synergy_bonus

    def _determine_confidence_level(self, confidence: float) -> str:
        if confidence >= 0.8:
            return "very_high"
        elif confidence >= 0.6:
            return "high"
        elif confidence >= 0.4:
            return "medium"
        elif confidence >= 0.2:
            return "low"
        return "very_low"

    def _calculate_priority(
        self, signal: FusedSignal, gain: float, confidence_level: str
    ) -> int:
        priority = 5
        
        if signal.consistency == ConsistencyType.CONSISTENT:
            priority += 2
        elif signal.consistency == ConsistencyType.CONFLICT:
            priority -= 2
        
        if confidence_level in ["very_high", "high"]:
            priority += 1
        elif confidence_level in ["low", "very_low"]:
            priority -= 1
        
        if abs(gain) > 0.5:
            priority += 1
        
        return max(1, min(10, priority))


class GainValidator:
    """增益效应验证器"""

    def __init__(self, risk_free_rate: float = 0.03):
        self.risk_free_rate = risk_free_rate

    def validate(
        self,
        model_returns: pd.Series,
        strategy_returns: pd.Series,
        fused_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> GainMetrics:
        model_metrics = self._calculate_metrics(model_returns)
        strategy_metrics = self._calculate_metrics(strategy_returns)
        fused_metrics = self._calculate_metrics(fused_returns)
        
        return_gain = fused_metrics["return"] - max(
            model_metrics["return"], strategy_metrics["return"]
        )
        
        sharpe_gain = fused_metrics["sharpe"] - max(
            model_metrics["sharpe"], strategy_metrics["sharpe"]
        )
        
        drawdown_gain = fused_metrics["max_drawdown"] - min(
            model_metrics["max_drawdown"], strategy_metrics["max_drawdown"]
        )
        
        winrate_gain = fused_metrics["win_rate"] - max(
            model_metrics["win_rate"], strategy_metrics["win_rate"]
        )
        
        information_ratio = 0
        if benchmark_returns is not None:
            tracking_error = (fused_returns - benchmark_returns).std()
            if tracking_error > 0:
                excess_return = fused_metrics["return"] - self._calculate_metrics(benchmark_returns)["return"]
                information_ratio = excess_return / tracking_error
        
        calmar_gain = fused_metrics["calmar"] - max(
            model_metrics["calmar"], strategy_metrics["calmar"]
        )
        
        sortino_gain = fused_metrics["sortino"] - max(
            model_metrics["sortino"], strategy_metrics["sortino"]
        )

        return GainMetrics(
            return_gain=return_gain,
            sharpe_gain=sharpe_gain,
            drawdown_gain=drawdown_gain,
            winrate_gain=winrate_gain,
            information_ratio=information_ratio,
            calmar_gain=calmar_gain,
            sortino_gain=sortino_gain,
            details={
                "model_metrics": model_metrics,
                "strategy_metrics": strategy_metrics,
                "fused_metrics": fused_metrics,
            },
        )

    def _calculate_metrics(self, returns: pd.Series) -> Dict[str, float]:
        if returns.empty:
            return {
                "return": 0,
                "sharpe": 0,
                "max_drawdown": 0,
                "win_rate": 0,
                "calmar": 0,
                "sortino": 0,
            }
        
        total_return = (1 + returns).prod() - 1
        annual_return = returns.mean() * 252
        
        excess_returns = returns - self.risk_free_rate / 252
        sharpe = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        win_rate = (returns > 0).mean()
        
        calmar = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = annual_return / downside_std if downside_std > 0 else 0

        return {
            "return": total_return,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "calmar": calmar,
            "sortino": sortino,
        }


class LayeredFusionProcessor:
    """分层融合处理器 - 整合四层架构"""

    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()
        
        self.layer1 = Layer1SignalPreprocessor()
        self.layer2 = Layer2ConsistencyAnalyzer()
        self.layer3 = Layer3GainCalculator(self.config)
        self.layer4 = Layer4DecisionMaker()
        
        self.regime_detector = MarketRegimeDetector()
        self.gain_validator = GainValidator()
        
        self._decision_history: List[FusionDecision] = []

    def process(
        self,
        model_predictions: Dict[str, ModelPrediction],
        strategy_signals: Dict[str, StrategySignalWrapper],
        market_context: Optional[MarketContext] = None,
    ) -> List[FusionDecision]:
        model_normalized = {
            code: self.layer1.preprocess_model_prediction(pred)
            for code, pred in model_predictions.items()
        }
        
        strategy_normalized = {
            code: self.layer1.preprocess_strategy_signal(sig)
            for code, sig in strategy_signals.items()
        }
        
        model_aligned, strategy_aligned = self.layer1.align_signals(
            model_normalized, strategy_normalized
        )
        
        regime = self._detect_regime(market_context)
        weights = self._get_weights(regime)
        
        decisions = []
        all_codes = set(model_aligned.keys()) | set(strategy_aligned.keys())
        
        for code in all_codes:
            model_sig = model_aligned.get(code)
            strategy_sig = strategy_aligned.get(code)
            
            consistency = self.layer2.analyze(model_sig, strategy_sig)
            
            gain, gain_components = self.layer3.calculate_gain(
                model_sig, strategy_sig, consistency, regime, weights
            )
            
            fused_signal = self._create_fused_signal(
                code, model_sig, strategy_sig, consistency, gain, weights
            )
            
            decision = self.layer4.make_decision(
                fused_signal, gain, gain_components, market_context
            )
            
            decisions.append(decision)
            self._decision_history.append(decision)
        
        decisions.sort(key=lambda x: x.execution_priority, reverse=True)
        
        if len(self._decision_history) > 1000:
            self._decision_history = self._decision_history[-1000:]
        
        return decisions

    def _detect_regime(self, market_context: Optional[MarketContext]) -> MarketRegime:
        if market_context:
            return market_context.regime
        return self.regime_detector.detect()

    def _get_weights(self, regime: MarketRegime) -> FusionWeights:
        regime_weights = self.config.regime_weights.get(regime.value, {})
        
        return FusionWeights(
            model_weight=regime_weights.get("model", self.config.base_model_weight),
            strategy_weight=regime_weights.get("strategy", self.config.base_strategy_weight),
        )

    def _create_fused_signal(
        self,
        code: str,
        model_sig: Optional[NormalizedSignal],
        strategy_sig: Optional[NormalizedSignal],
        consistency: ConsistencyResult,
        gain: float,
        weights: FusionWeights,
    ) -> FusedSignal:
        name = model_sig.name if model_sig else strategy_sig.name if strategy_sig else code
        
        action = "buy" if gain > 0.05 else "sell" if gain < -0.05 else "hold"
        
        model_pred = None
        strategy_signal = None
        if model_sig and "raw_data" in model_sig.raw_data:
            model_pred = ModelPrediction(**model_sig.raw_data)
        if strategy_sig and "raw_data" in strategy_sig.raw_data:
            strategy_signal = StrategySignalWrapper(**strategy_sig.raw_data)
        
        return FusedSignal(
            code=code,
            name=name,
            action=action,
            confidence=min(abs(gain) + 0.3, 1.0),
            fused_score=gain,
            consistency=consistency.consistency_type,
            model_contribution=weights.model_weight * (model_sig.normalized_score if model_sig else 0),
            strategy_contribution=weights.strategy_weight * (strategy_sig.normalized_score if strategy_sig else 0),
            model_weight=weights.model_weight,
            strategy_weight=weights.strategy_weight,
            model_prediction=model_pred,
            strategy_signal=strategy_signal,
        )

    def validate_gain_effect(
        self,
        model_returns: pd.Series,
        strategy_returns: pd.Series,
        fused_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> GainMetrics:
        return self.gain_validator.validate(
            model_returns, strategy_returns, fused_returns, benchmark_returns
        )

    def get_statistics(self) -> Dict[str, Any]:
        if not self._decision_history:
            return {}
        
        decisions = self._decision_history
        
        return {
            "total_decisions": len(decisions),
            "avg_position_ratio": np.mean([d.position_ratio for d in decisions]),
            "avg_gain_potential": np.mean([d.gain_potential for d in decisions]),
            "confidence_distribution": {
                level: sum(1 for d in decisions if d.confidence_level == level)
                for level in ["very_high", "high", "medium", "low", "very_low"]
            },
            "action_distribution": {
                action: sum(1 for d in decisions if d.signal.action == action)
                for action in ["buy", "sell", "hold"]
            },
            "avg_risk_reward": np.mean([d.risk_reward_ratio for d in decisions if d.risk_reward_ratio > 0]),
            "layer3_stats": self.layer3.get_gain_statistics(),
        }
