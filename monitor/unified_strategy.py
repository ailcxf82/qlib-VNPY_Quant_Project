from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from monitor.data_source import DataSourceAdapter

logger = logging.getLogger(__name__)


@dataclass
class UnifiedSignal:
    code: str
    name: str
    signal_type: str
    score: float
    confidence: float
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "signal_type": self.signal_type,
            "score": self.score,
            "confidence": self.confidence,
            "reason": self.reason,
            "details": self.details,
        }


class UnifiedStrategyBase(ABC):
    STRATEGY_NAME = "base"
    
    def __init__(
        self, 
        config: Dict[str, Any],
        data_source: Optional["DataSourceAdapter"] = None
    ):
        self.config = config
        self._data_source = data_source
        self._factor_cache: Dict[str, Dict[str, float]] = {}
    
    @property
    def data_source(self) -> Optional["DataSourceAdapter"]:
        return self._data_source
    
    @data_source.setter
    def data_source(self, value: "DataSourceAdapter"):
        self._data_source = value
    
    @abstractmethod
    def generate_signals(
        self, 
        date: str, 
        data: Dict[str, pd.DataFrame],
        positions: Dict[str, Any]
    ) -> Dict[str, str]:
        pass
    
    @abstractmethod
    def calculate_score(self, code: str, factors: Dict[str, float]) -> float:
        pass
    
    def get_position_size(
        self, 
        code: str, 
        price: float, 
        total_capital: float,
        risk_controller: Optional[Any] = None
    ) -> int:
        base_position_size = self.config.get("position_size", 0.10)
        position_value = total_capital * base_position_size
        shares = int(position_value / price / 100) * 100
        return max(shares, 100)
    
    def get_stop_loss_level(self, entry_price: float, factors: Dict[str, float]) -> float:
        default_stop = self.config.get("stop_loss", -0.08)
        return entry_price * (1 + default_stop)
    
    def get_take_profit_level(self, entry_price: float, factors: Dict[str, float]) -> float:
        default_profit = self.config.get("take_profit", 0.15)
        return entry_price * (1 + default_profit)
    
    def get_max_holding_days(self) -> int:
        return self.config.get("max_holding_days", 10)
    
    def fetch_factors(self, code: str) -> Dict[str, float]:
        if code in self._factor_cache:
            return self._factor_cache[code]
        
        if self.data_source is not None:
            try:
                factors = self.data_source.get_stock_factors(code)
                self._factor_cache[code] = factors
                return factors
            except Exception as e:
                logger.debug(f"Failed to fetch factors for {code}: {e}")
        
        return self._get_default_factors()
    
    def _get_default_factors(self) -> Dict[str, float]:
        return {
            "pb": 1.0,
            "pe": 15.0,
            "roe": 0.10,
            "roa": 0.05,
            "debt_ratio": 0.40,
            "price_momentum_5d": 0.0,
            "price_momentum_20d": 0.0,
            "volume_momentum": 1.0,
            "rsi_14": 50.0,
            "price_deviation_ma20": 0.0,
        }
    
    def clear_cache(self):
        self._factor_cache.clear()


class UnifiedMomentumStrategy(UnifiedStrategyBase):
    STRATEGY_NAME = "momentum"
    
    def __init__(
        self, 
        config: Dict[str, Any],
        data_source: Optional["DataSourceAdapter"] = None
    ):
        super().__init__(config, data_source)
        self.lookback_period = config.get("lookback_period", 20)
        self.top_k = config.get("top_k", 10)
        self.min_momentum = config.get("min_momentum", 0.05)
        self.stop_loss = config.get("stop_loss", -0.08)
        self.take_profit = config.get("take_profit", 0.15)
        self.max_holding_days = config.get("max_holding_days", 10)
    
    def generate_signals(
        self, 
        date: str, 
        data: Dict[str, pd.DataFrame],
        positions: Dict[str, Any]
    ) -> Dict[str, str]:
        signals = {}
        
        for code, pos in positions.items():
            if code not in data:
                continue
            
            df = data[code]
            if df.empty:
                continue
            
            current_date = pd.to_datetime(date)
            df_before = df[df["date"] <= current_date]
            
            if df_before.empty:
                continue
            
            current_price = float(df_before.iloc[-1]["close"])
            entry_price = pos.get("entry_price", pos.get("buy_price", current_price))
            holding_days = (current_date - pd.to_datetime(pos.get("entry_date", pos.get("buy_date", date)))).days
            
            profit_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0
            
            if profit_pct <= self.stop_loss:
                signals[code] = "sell"
            elif profit_pct >= self.take_profit:
                signals[code] = "sell"
            elif holding_days >= self.max_holding_days:
                signals[code] = "sell"
        
        momentum_scores = {}
        for code, df in data.items():
            if df.empty:
                continue
            
            current_date = pd.to_datetime(date)
            df_before = df[df["date"] <= current_date]
            
            if len(df_before) < self.lookback_period:
                continue
            
            if code in positions:
                continue
            
            close = df_before["close"].astype(float)
            momentum = (close.iloc[-1] - close.iloc[-self.lookback_period]) / close.iloc[-self.lookback_period]
            
            if momentum >= self.min_momentum:
                momentum_scores[code] = momentum
        
        sorted_stocks = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        
        max_stocks = self.config.get("max_stocks", 10)
        for code, _ in sorted_stocks[:self.top_k]:
            if len(positions) + len([s for s in signals.values() if s == "buy"]) >= max_stocks:
                break
            signals[code] = "buy"
        
        return signals
    
    def calculate_score(self, code: str, factors: Dict[str, float]) -> float:
        mom_5d = factors.get("price_momentum_5d", 0.0)
        mom_20d = factors.get("price_momentum_20d", 0.0)
        vol_mom = factors.get("volume_momentum", 1.0)
        
        mom_5d_score = max(0, min(1, (mom_5d + 0.05) / 0.10))
        mom_20d_score = max(0, min(1, (mom_20d + 0.10) / 0.20))
        vol_score = min(vol_mom / 2.0, 1.0)
        
        momentum_score = mom_5d_score * 0.35 + mom_20d_score * 0.50 + vol_score * 0.15
        return momentum_score


class UnifiedMeanReversionStrategy(UnifiedStrategyBase):
    STRATEGY_NAME = "mean_reversion"
    
    def __init__(
        self, 
        config: Dict[str, Any],
        data_source: Optional["DataSourceAdapter"] = None
    ):
        super().__init__(config, data_source)
        self.lookback_period = config.get("lookback_period", 20)
        self.oversold_threshold = config.get("oversold_threshold", -0.15)
        self.overbought_threshold = config.get("overbought_threshold", 0.15)
        self.top_k = config.get("top_k", 10)
    
    def generate_signals(
        self, 
        date: str, 
        data: Dict[str, pd.DataFrame],
        positions: Dict[str, Any]
    ) -> Dict[str, str]:
        signals = {}
        
        for code, pos in positions.items():
            if code not in data:
                continue
            
            df = data[code]
            if df.empty:
                continue
            
            current_date = pd.to_datetime(date)
            df_before = df[df["date"] <= current_date]
            
            if len(df_before) < self.lookback_period:
                continue
            
            close = df_before["close"].astype(float)
            ma = close.rolling(self.lookback_period).mean().iloc[-1]
            current_price = float(df_before.iloc[-1]["close"])
            
            deviation = (current_price - ma) / ma if ma > 0 else 0
            
            if deviation > self.overbought_threshold:
                signals[code] = "sell"
        
        deviation_scores = {}
        for code, df in data.items():
            if df.empty:
                continue
            
            current_date = pd.to_datetime(date)
            df_before = df[df["date"] <= current_date]
            
            if len(df_before) < self.lookback_period:
                continue
            
            if code in positions:
                continue
            
            close = df_before["close"].astype(float)
            ma = close.rolling(self.lookback_period).mean().iloc[-1]
            current_price = float(df_before.iloc[-1]["close"])
            
            deviation = (current_price - ma) / ma if ma > 0 else 0
            
            if deviation < self.oversold_threshold:
                deviation_scores[code] = deviation
        
        sorted_stocks = sorted(deviation_scores.items(), key=lambda x: x[0])
        
        max_stocks = self.config.get("max_stocks", 10)
        for code, _ in sorted_stocks[:self.top_k]:
            if len(positions) + len([s for s in signals.values() if s == "buy"]) >= max_stocks:
                break
            signals[code] = "buy"
        
        return signals
    
    def calculate_score(self, code: str, factors: Dict[str, float]) -> float:
        rsi = factors.get("rsi_14", 50.0)
        deviation = factors.get("price_deviation_ma20", 0.0)
        vol_spike = factors.get("volume_spike", 1.0)
        
        rsi_score = max(0, (40 - rsi) / 40) if rsi < 40 else max(0, (rsi - 60) / 40)
        deviation_score = max(0, min(1, abs(deviation) / 0.10))
        vol_score = min(vol_spike / 2.0, 1.0)
        
        mr_score = rsi_score * 0.40 + deviation_score * 0.35 + vol_score * 0.25
        return mr_score


class UnifiedValueStrategy(UnifiedStrategyBase):
    STRATEGY_NAME = "value"
    
    def calculate_score(self, code: str, factors: Dict[str, float]) -> float:
        pb = factors.get("pb", 1.0)
        pe = factors.get("pe", 15.0)
        roe = factors.get("roe", 0.10)
        dividend = factors.get("dividend_yield", 0.02)
        
        pb_score = 1.0 - min(pb / 3.0, 1.0)
        pe_score = 1.0 - min(pe / 30.0, 1.0)
        roe_score = min(roe / 0.20, 1.0)
        div_score = min(dividend / 0.05, 1.0)
        
        value_score = pb_score * 0.35 + pe_score * 0.30 + roe_score * 0.20 + div_score * 0.15
        return value_score
    
    def generate_signals(
        self, 
        date: str, 
        data: Dict[str, pd.DataFrame],
        positions: Dict[str, Any]
    ) -> Dict[str, str]:
        signals = {}
        
        value_scores = {}
        for code in data.keys():
            if code in positions:
                continue
            
            factors = self.fetch_factors(code)
            score = self.calculate_score(code, factors)
            
            min_score = self.config.get("min_score", 0.5)
            if score >= min_score:
                value_scores[code] = score
        
        sorted_stocks = sorted(value_scores.items(), key=lambda x: x[1], reverse=True)
        
        top_k = self.config.get("top_k", 10)
        max_stocks = self.config.get("max_stocks", 10)
        
        for code, _ in sorted_stocks[:top_k]:
            if len(positions) + len([s for s in signals.values() if s == "buy"]) >= max_stocks:
                break
            signals[code] = "buy"
        
        return signals


class UnifiedQualityStrategy(UnifiedStrategyBase):
    STRATEGY_NAME = "quality"
    
    def calculate_score(self, code: str, factors: Dict[str, float]) -> float:
        roe = factors.get("roe", 0.10)
        roa = factors.get("roa", 0.05)
        debt = factors.get("debt_ratio", 0.40)
        
        roe_score = min(roe / 0.20, 1.0)
        roa_score = min(roa / 0.08, 1.0)
        debt_score = 1.0 - min(debt / 0.60, 1.0)
        
        quality_score = roe_score * 0.40 + roa_score * 0.30 + debt_score * 0.30
        return quality_score
    
    def generate_signals(
        self, 
        date: str, 
        data: Dict[str, pd.DataFrame],
        positions: Dict[str, Any]
    ) -> Dict[str, str]:
        signals = {}
        
        quality_scores = {}
        for code in data.keys():
            if code in positions:
                continue
            
            factors = self.fetch_factors(code)
            score = self.calculate_score(code, factors)
            
            min_score = self.config.get("min_score", 0.5)
            if score >= min_score:
                quality_scores[code] = score
        
        sorted_stocks = sorted(quality_scores.items(), key=lambda x: x[1], reverse=True)
        
        top_k = self.config.get("top_k", 10)
        max_stocks = self.config.get("max_stocks", 10)
        
        for code, _ in sorted_stocks[:top_k]:
            if len(positions) + len([s for s in signals.values() if s == "buy"]) >= max_stocks:
                break
            signals[code] = "buy"
        
        return signals


def create_strategy(
    strategy_name: str,
    config: Dict[str, Any],
    data_source: Optional["DataSourceAdapter"] = None
) -> UnifiedStrategyBase:
    strategy_map = {
        "momentum": UnifiedMomentumStrategy,
        "mean_reversion": UnifiedMeanReversionStrategy,
        "value": UnifiedValueStrategy,
        "quality": UnifiedQualityStrategy,
    }
    
    strategy_class = strategy_map.get(strategy_name.lower())
    if strategy_class is None:
        logger.warning(f"Unknown strategy: {strategy_name}, using momentum")
        strategy_class = UnifiedMomentumStrategy
    
    return strategy_class(config, data_source)
