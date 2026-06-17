"""
Signal Fusion Engine - 信号融合引擎

将模型预测与策略信号进行智能融合，实现增益效应。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from monitor.strategy_library.strategy_base import Signal as StrategySignal

logger = logging.getLogger(__name__)


class ConsistencyType(Enum):
    CONSISTENT = "consistent"
    CONFLICT = "conflict"
    PARTIAL = "partial"
    NO_SIGNAL = "no_signal"


class MarketRegime(Enum):
    BULL_TRENDING = "bull_trending"
    BULL_CHOPPY = "bull_choppy"
    BEAR_TRENDING = "bear_trending"
    BEAR_CHOPPY = "bear_choppy"
    NEUTRAL = "neutral"


@dataclass
class ModelPrediction:
    code: str
    name: str
    score: float
    confidence: float
    std: float = 0.1
    direction: str = "neutral"
    horizon: int = 5
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if self.score > 0.55:
            self.direction = "up"
        elif self.score < 0.45:
            self.direction = "down"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "score": self.score,
            "confidence": self.confidence,
            "std": self.std,
            "direction": self.direction,
            "horizon": self.horizon,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class StrategySignalWrapper:
    code: str
    name: str
    action: str
    strength: float
    reason: str
    strategy_id: str
    strategy_name: str
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "action": self.action,
            "strength": self.strength,
            "reason": self.reason,
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class FusedSignal:
    code: str
    name: str
    action: str
    confidence: float
    fused_score: float
    consistency: ConsistencyType
    model_contribution: float
    strategy_contribution: float
    model_weight: float
    strategy_weight: float
    model_prediction: Optional[ModelPrediction] = None
    strategy_signal: Optional[StrategySignalWrapper] = None
    reason: str = ""
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.reason:
            self.reason = self._generate_reason()

    def _generate_reason(self) -> str:
        parts = []
        if self.consistency == ConsistencyType.CONSISTENT:
            parts.append("模型+策略共振")
        elif self.consistency == ConsistencyType.CONFLICT:
            parts.append("信号冲突")
        else:
            parts.append("部分信号")
        
        if self.model_prediction:
            parts.append(f"模型预测{self.model_prediction.score:.2f}")
        if self.strategy_signal:
            parts.append(f"策略{self.strategy_signal.action}")
        
        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "action": self.action,
            "confidence": self.confidence,
            "fused_score": self.fused_score,
            "consistency": self.consistency.value,
            "model_contribution": self.model_contribution,
            "strategy_contribution": self.strategy_contribution,
            "model_weight": self.model_weight,
            "strategy_weight": self.strategy_weight,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class MarketContext:
    index_value: float
    index_change: float
    volatility: float
    trend: str
    regime: MarketRegime
    uncertainty: float
    north_money_flow: float = 0.0
    sentiment_score: float = 0.0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index_value": self.index_value,
            "index_change": self.index_change,
            "volatility": self.volatility,
            "trend": self.trend,
            "regime": self.regime.value,
            "uncertainty": self.uncertainty,
            "north_money_flow": self.north_money_flow,
            "sentiment_score": self.sentiment_score,
            "timestamp": self.timestamp,
        }


@dataclass
class FusionWeights:
    model_weight: float = 0.5
    strategy_weight: float = 0.5
    confidence_boost: float = 1.0
    regime_adjustment: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_weight": self.model_weight,
            "strategy_weight": self.strategy_weight,
            "confidence_boost": self.confidence_boost,
            "regime_adjustment": self.regime_adjustment,
        }


@dataclass
class FusionConfig:
    base_model_weight: float = 0.5
    base_strategy_weight: float = 0.5
    consistency_boost: float = 1.3
    conflict_penalty: float = 0.7
    min_confidence_threshold: float = 0.3
    high_confidence_threshold: float = 0.7
    enable_dynamic_weights: bool = True
    enable_regime_adjustment: bool = True
    enable_feedback_loop: bool = True

    regime_weights: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        MarketRegime.BULL_TRENDING.value: {"model": 0.3, "strategy": 0.7},
        MarketRegime.BULL_CHOPPY.value: {"model": 0.6, "strategy": 0.4},
        MarketRegime.BEAR_TRENDING.value: {"model": 0.3, "strategy": 0.7},
        MarketRegime.BEAR_CHOPPY.value: {"model": 0.7, "strategy": 0.3},
        MarketRegime.NEUTRAL.value: {"model": 0.5, "strategy": 0.5},
    })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_model_weight": self.base_model_weight,
            "base_strategy_weight": self.base_strategy_weight,
            "consistency_boost": self.consistency_boost,
            "conflict_penalty": self.conflict_penalty,
            "min_confidence_threshold": self.min_confidence_threshold,
            "high_confidence_threshold": self.high_confidence_threshold,
            "enable_dynamic_weights": self.enable_dynamic_weights,
            "enable_regime_adjustment": self.enable_regime_adjustment,
            "enable_feedback_loop": self.enable_feedback_loop,
            "regime_weights": self.regime_weights,
        }


class MarketRegimeDetector:
    def __init__(self, lookback: int = 60):
        self.lookback = lookback
        self._index_data: Optional[pd.DataFrame] = None

    def set_index_data(self, data: pd.DataFrame):
        self._index_data = data

    def detect(self, market_data: Optional[Dict] = None) -> MarketRegime:
        if market_data:
            return self._detect_from_context(market_data)
        return self._detect_from_index()

    def _detect_from_context(self, context: Dict) -> MarketRegime:
        trend = context.get("trend", "neutral")
        volatility = context.get("volatility", 0.02)
        index_change = context.get("index_change", 0)

        if trend == "up":
            if volatility > 0.025:
                return MarketRegime.BULL_CHOPPY
            else:
                return MarketRegime.BULL_TRENDING
        elif trend == "down":
            if volatility > 0.025:
                return MarketRegime.BEAR_CHOPPY
            else:
                return MarketRegime.BEAR_TRENDING
        else:
            return MarketRegime.NEUTRAL

    def _detect_from_index(self) -> MarketRegime:
        if self._index_data is None or self._index_data.empty:
            return MarketRegime.NEUTRAL

        close = self._index_data["close"].tail(self.lookback)
        if len(close) < 20:
            return MarketRegime.NEUTRAL

        ma20 = close.rolling(20).mean().iloc[-1]
        ma60 = close.rolling(60).mean().iloc[-1] if len(close) >= 60 else ma20
        current = close.iloc[-1]

        returns = close.pct_change().dropna()
        volatility = returns.tail(20).std() * np.sqrt(252)

        trend = "up" if current > ma20 > ma60 else "down" if current < ma20 < ma60 else "neutral"

        if trend == "up":
            if volatility > 0.25:
                return MarketRegime.BULL_CHOPPY
            else:
                return MarketRegime.BULL_TRENDING
        elif trend == "down":
            if volatility > 0.25:
                return MarketRegime.BEAR_CHOPPY
            else:
                return MarketRegime.BEAR_TRENDING
        else:
            return MarketRegime.NEUTRAL


class DynamicWeightAdjuster:
    def __init__(self, config: FusionConfig):
        self.config = config
        self._performance_history: List[Dict] = []
        self._current_weights = FusionWeights()

    def get_weights(
        self,
        regime: MarketRegime,
        recent_performance: Optional[Dict] = None,
    ) -> FusionWeights:
        weights = FusionWeights(
            model_weight=self.config.base_model_weight,
            strategy_weight=self.config.base_strategy_weight,
        )

        if self.config.enable_regime_adjustment:
            regime_weights = self.config.regime_weights.get(regime.value, {})
            if regime_weights:
                weights.model_weight = regime_weights.get("model", weights.model_weight)
                weights.strategy_weight = regime_weights.get("strategy", weights.strategy_weight)
                weights.regime_adjustment = weights.model_weight / self.config.base_model_weight

        if self.config.enable_dynamic_weights and recent_performance:
            model_ic = recent_performance.get("model_ic", 0.5)
            strategy_return = recent_performance.get("strategy_return", 0)

            if model_ic > 0.08:
                weights.model_weight *= 1.2
                weights.strategy_weight *= 0.8
            elif strategy_return > 0.05:
                weights.model_weight *= 0.8
                weights.strategy_weight *= 1.2

        total = weights.model_weight + weights.strategy_weight
        if total > 0:
            weights.model_weight /= total
            weights.strategy_weight /= total

        self._current_weights = weights
        return weights

    def update_performance(self, performance: Dict):
        self._performance_history.append({
            "timestamp": datetime.now().isoformat(),
            **performance,
        })

        if len(self._performance_history) > 100:
            self._performance_history = self._performance_history[-100:]


class ConflictResolver:
    def __init__(self, config: FusionConfig):
        self.config = config

    def resolve(
        self,
        model_pred: Optional[ModelPrediction],
        strategy_signal: Optional[StrategySignalWrapper],
        market_context: Optional[MarketContext],
    ) -> FusedSignal:
        code = model_pred.code if model_pred else strategy_signal.code if strategy_signal else ""
        name = model_pred.name if model_pred else strategy_signal.name if strategy_signal else ""

        if model_pred is None:
            return self._strategy_only_signal(strategy_signal)
        if strategy_signal is None:
            return self._model_only_signal(model_pred)

        model_strength = abs(model_pred.score - 0.5) * 2 * model_pred.confidence
        strategy_strength = strategy_signal.strength

        if model_strength > strategy_strength * 1.5:
            return self._model_dominant(model_pred, strategy_signal)
        elif strategy_strength > model_strength * 1.5:
            return self._strategy_dominant(model_pred, strategy_signal)
        else:
            return self._neutral_conflict(model_pred, strategy_signal, market_context)

    def _strategy_only_signal(self, signal: StrategySignalWrapper) -> FusedSignal:
        return FusedSignal(
            code=signal.code,
            name=signal.name,
            action=signal.action,
            confidence=signal.strength * 0.8,
            fused_score=signal.strength if signal.action == "buy" else -signal.strength,
            consistency=ConsistencyType.PARTIAL,
            model_contribution=0.0,
            strategy_contribution=signal.strength,
            model_weight=0.0,
            strategy_weight=1.0,
            strategy_signal=signal,
            reason=f"仅策略信号: {signal.reason}",
        )

    def _model_only_signal(self, pred: ModelPrediction) -> FusedSignal:
        action = "buy" if pred.score > 0.55 else "sell" if pred.score < 0.45 else "hold"
        return FusedSignal(
            code=pred.code,
            name=pred.name,
            action=action,
            confidence=pred.confidence * 0.8,
            fused_score=(pred.score - 0.5) * 2,
            consistency=ConsistencyType.PARTIAL,
            model_contribution=(pred.score - 0.5) * 2,
            strategy_contribution=0.0,
            model_weight=1.0,
            strategy_weight=0.0,
            model_prediction=pred,
            reason=f"仅模型预测: 分数{pred.score:.2f}",
        )

    def _model_dominant(
        self, model_pred: ModelPrediction, strategy_signal: StrategySignalWrapper
    ) -> FusedSignal:
        action = "buy" if model_pred.score > 0.5 else "sell"
        return FusedSignal(
            code=model_pred.code,
            name=model_pred.name,
            action=action,
            confidence=model_pred.confidence * 0.9,
            fused_score=(model_pred.score - 0.5) * 2 * 0.7,
            consistency=ConsistencyType.CONFLICT,
            model_contribution=(model_pred.score - 0.5) * 2,
            strategy_contribution=-0.2 if action == "buy" else 0.2,
            model_weight=0.7,
            strategy_weight=0.3,
            model_prediction=model_pred,
            strategy_signal=strategy_signal,
            reason=f"模型主导(冲突): 模型{model_pred.score:.2f} vs 策略{strategy_signal.action}",
        )

    def _strategy_dominant(
        self, model_pred: ModelPrediction, strategy_signal: StrategySignalWrapper
    ) -> FusedSignal:
        return FusedSignal(
            code=strategy_signal.code,
            name=strategy_signal.name,
            action=strategy_signal.action,
            confidence=strategy_signal.strength * 0.9,
            fused_score=strategy_signal.strength * 0.7 if strategy_signal.action == "buy" else -strategy_signal.strength * 0.7,
            consistency=ConsistencyType.CONFLICT,
            model_contribution=-0.2 if strategy_signal.action == "buy" else 0.2,
            strategy_contribution=strategy_signal.strength,
            model_weight=0.3,
            strategy_weight=0.7,
            model_prediction=model_pred,
            strategy_signal=strategy_signal,
            reason=f"策略主导(冲突): 策略{strategy_signal.action} vs 模型{model_pred.score:.2f}",
        )

    def _neutral_conflict(
        self,
        model_pred: ModelPrediction,
        strategy_signal: StrategySignalWrapper,
        market_context: Optional[MarketContext],
    ) -> FusedSignal:
        return FusedSignal(
            code=model_pred.code,
            name=model_pred.name,
            action="hold",
            confidence=0.3,
            fused_score=0.0,
            consistency=ConsistencyType.CONFLICT,
            model_contribution=0.0,
            strategy_contribution=0.0,
            model_weight=0.5,
            strategy_weight=0.5,
            model_prediction=model_pred,
            strategy_signal=strategy_signal,
            reason=f"信号冲突,保持观望: 模型{model_pred.score:.2f} vs 策略{strategy_signal.action}",
        )


class FusionEngine:
    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()
        self.regime_detector = MarketRegimeDetector()
        self.weight_adjuster = DynamicWeightAdjuster(self.config)
        self.conflict_resolver = ConflictResolver(self.config)
        
        self._fusion_history: List[Dict] = []
        self._performance_tracker: Dict[str, List] = {}

    def fuse(
        self,
        model_predictions: Dict[str, ModelPrediction],
        strategy_signals: Dict[str, StrategySignalWrapper],
        market_context: Optional[MarketContext] = None,
    ) -> List[FusedSignal]:
        regime = self._detect_regime(market_context)
        weights = self._get_weights(regime)
        fused_signals = []

        all_codes = set(model_predictions.keys()) | set(strategy_signals.keys())

        for code in all_codes:
            model_pred = model_predictions.get(code)
            strategy_signal = strategy_signals.get(code)

            consistency = self._check_consistency(model_pred, strategy_signal)

            if consistency == ConsistencyType.CONSISTENT:
                fused = self._consistent_fusion(model_pred, strategy_signal, weights)
            elif consistency == ConsistencyType.CONFLICT:
                fused = self.conflict_resolver.resolve(model_pred, strategy_signal, market_context)
            elif consistency == ConsistencyType.PARTIAL:
                fused = self._partial_fusion(model_pred, strategy_signal, weights)
            else:
                continue

            fused_signals.append(fused)
            self._record_fusion(fused)

        fused_signals.sort(key=lambda x: abs(x.fused_score), reverse=True)
        return fused_signals

    def _detect_regime(self, market_context: Optional[MarketContext]) -> MarketRegime:
        if market_context:
            return market_context.regime
        return self.regime_detector.detect()

    def _get_weights(self, regime: MarketRegime) -> FusionWeights:
        recent_perf = self._get_recent_performance()
        return self.weight_adjuster.get_weights(regime, recent_perf)

    def _get_recent_performance(self) -> Dict[str, float]:
        if not self._performance_tracker:
            return {}

        model_ics = self._performance_tracker.get("model_ic", [])
        strategy_returns = self._performance_tracker.get("strategy_return", [])

        return {
            "model_ic": np.mean(model_ics[-20:]) if model_ics else 0.0,
            "strategy_return": np.mean(strategy_returns[-20:]) if strategy_returns else 1.0,
        }

    def _check_consistency(
        self,
        model_pred: Optional[ModelPrediction],
        strategy_signal: Optional[StrategySignalWrapper],
    ) -> ConsistencyType:
        if model_pred is None and strategy_signal is None:
            return ConsistencyType.NO_SIGNAL
        if model_pred is None or strategy_signal is None:
            return ConsistencyType.PARTIAL

        model_direction = model_pred.direction
        strategy_direction = "up" if strategy_signal.action == "buy" else "down" if strategy_signal.action == "sell" else "neutral"

        if model_direction == strategy_direction:
            return ConsistencyType.CONSISTENT
        else:
            return ConsistencyType.CONFLICT

    def _consistent_fusion(
        self,
        model_pred: ModelPrediction,
        strategy_signal: StrategySignalWrapper,
        weights: FusionWeights,
    ) -> FusedSignal:
        model_score = (model_pred.score - 0.5) * 2
        strategy_score = strategy_signal.strength if strategy_signal.action == "buy" else -strategy_signal.strength

        fused_score = (
            weights.model_weight * model_score * model_pred.confidence +
            weights.strategy_weight * strategy_score
        ) * self.config.consistency_boost

        action = "buy" if fused_score > 0.2 else "sell" if fused_score < -0.2 else "hold"
        confidence = min(abs(fused_score) * self.config.consistency_boost, 1.0)

        return FusedSignal(
            code=model_pred.code,
            name=model_pred.name,
            action=action,
            confidence=confidence,
            fused_score=fused_score,
            consistency=ConsistencyType.CONSISTENT,
            model_contribution=weights.model_weight * model_score * model_pred.confidence,
            strategy_contribution=weights.strategy_weight * strategy_score,
            model_weight=weights.model_weight,
            strategy_weight=weights.strategy_weight,
            model_prediction=model_pred,
            strategy_signal=strategy_signal,
            reason=f"模型+策略共振: 模型{model_pred.score:.2f}, 策略{strategy_signal.action}",
        )

    def _partial_fusion(
        self,
        model_pred: Optional[ModelPrediction],
        strategy_signal: Optional[StrategySignalWrapper],
        weights: FusionWeights,
    ) -> FusedSignal:
        if model_pred is None:
            return self.conflict_resolver._strategy_only_signal(strategy_signal)
        if strategy_signal is None:
            return self.conflict_resolver._model_only_signal(model_pred)

        return self.conflict_resolver.resolve(model_pred, strategy_signal, None)

    def _record_fusion(self, fused: FusedSignal):
        record = {
            "timestamp": fused.timestamp,
            "code": fused.code,
            "action": fused.action,
            "confidence": fused.confidence,
            "fused_score": fused.fused_score,
            "consistency": fused.consistency.value,
            "model_weight": fused.model_weight,
            "strategy_weight": fused.strategy_weight,
        }
        self._fusion_history.append(record)

        if len(self._fusion_history) > 1000:
            self._fusion_history = self._fusion_history[-1000:]

    def update_performance(self, performance: Dict[str, float]):
        for key, value in performance.items():
            if key not in self._performance_tracker:
                self._performance_tracker[key] = []
            self._performance_tracker[key].append(value)

            if len(self._performance_tracker[key]) > 100:
                self._performance_tracker[key] = self._performance_tracker[key][-100:]

        self.weight_adjuster.update_performance(performance)

    def get_fusion_statistics(self) -> Dict[str, Any]:
        if not self._fusion_history:
            return {}

        df = pd.DataFrame(self._fusion_history)

        return {
            "total_fusions": len(self._fusion_history),
            "consistency_distribution": df["consistency"].value_counts().to_dict(),
            "action_distribution": df["action"].value_counts().to_dict(),
            "avg_confidence": df["confidence"].mean(),
            "avg_fused_score": df["fused_score"].mean(),
            "avg_model_weight": df["model_weight"].mean(),
            "avg_strategy_weight": df["strategy_weight"].mean(),
        }

    def get_top_signals(self, n: int = 10) -> List[FusedSignal]:
        if not self._fusion_history:
            return []

        df = pd.DataFrame(self._fusion_history)
        df = df.sort_values("fused_score", key=lambda x: abs(x), ascending=False)

        return df.head(n).to_dict("records")


def create_fusion_engine(config_path: Optional[str] = None) -> FusionEngine:
    if config_path:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        
        fusion_config = config_dict.get("fusion", {})
        config = FusionConfig(
            base_model_weight=fusion_config.get("base_model_weight", 0.5),
            base_strategy_weight=fusion_config.get("base_strategy_weight", 0.5),
            consistency_boost=fusion_config.get("consistency_boost", 1.3),
            conflict_penalty=fusion_config.get("conflict_penalty", 0.7),
            min_confidence_threshold=fusion_config.get("min_confidence_threshold", 0.3),
            high_confidence_threshold=fusion_config.get("high_confidence_threshold", 0.7),
            enable_dynamic_weights=fusion_config.get("enable_dynamic_weights", True),
            enable_regime_adjustment=fusion_config.get("enable_regime_adjustment", True),
            enable_feedback_loop=fusion_config.get("enable_feedback_loop", True),
            regime_weights=fusion_config.get("regime_weights", {}),
        )
    else:
        config = FusionConfig()
    
    return FusionEngine(config=config)
