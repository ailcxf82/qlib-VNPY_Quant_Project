"""
Turtle Trading Strategy - 海龟交易策略

基于经典海龟交易法则的趋势跟踪策略。
核心逻辑：突破N日高点买入，跌破N日低点卖出。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

from monitor.strategy_library.strategy_base import (
    StrategyBase,
    StrategyConfig,
    Signal,
    ParameterSpace,
)
from monitor.strategy_library.strategy_registry import register_strategy

if TYPE_CHECKING:
    from monitor.data_source import DataSourceAdapter

logger = logging.getLogger(__name__)


@dataclass
class TurtlePosition:
    code: str
    name: str
    shares: int
    entry_price: float
    entry_date: str
    entry_signal: str
    highest_price: float
    lowest_price: float
    stop_price: float
    unit_size: int
    pyramids: int = 0
    max_pyramids: int = 4
    
    def update_price(self, high: float, low: float) -> None:
        self.highest_price = max(self.highest_price, high)
        self.lowest_price = min(self.lowest_price, low)
    
    def can_add_pyramid(self, current_price: float, entry_price: float) -> bool:
        if self.pyramids >= self.max_pyramids:
            return False
        
        price_increase = (current_price - entry_price) / entry_price
        return price_increase >= 0.5 * self._get_atr_estimate(entry_price)
    
    def _get_atr_estimate(self, entry_price: float) -> float:
        return 0.02
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "shares": self.shares,
            "entry_price": self.entry_price,
            "entry_date": self.entry_date,
            "entry_signal": self.entry_signal,
            "highest_price": self.highest_price,
            "lowest_price": self.lowest_price,
            "stop_price": self.stop_price,
            "unit_size": self.unit_size,
            "pyramids": self.pyramids,
            "max_pyramids": self.max_pyramids,
        }


@register_strategy(strategy_id="turtle_trading", category="趋势策略", description="基于经典海龟交易法则的趋势跟踪策略")
class TurtleTradingStrategy(StrategyBase):
    STRATEGY_ID = "turtle_trading"
    STRATEGY_NAME = "海龟交易策略"
    STRATEGY_CATEGORY = "趋势策略"
    STRATEGY_VERSION = "1.0.0"
    
    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        parameters: Optional[Dict[str, Any]] = None,
        data_source: Optional["DataSourceAdapter"] = None,
    ):
        super().__init__(config=config, parameters=parameters)
        
        self._data_source = data_source
        self._turtle_positions: Dict[str, TurtlePosition] = {}
        self._breakout_highs: Dict[str, float] = {}
        self._breakout_lows: Dict[str, float] = {}
    
    def get_parameter_space(self) -> Dict[str, ParameterSpace]:
        return {
            "entry_breakout": ParameterSpace(
                name="entry_breakout",
                param_type="int",
                min_val=10,
                max_val=60,
                default=20,
                description="入场突破周期",
            ),
            "exit_breakout": ParameterSpace(
                name="exit_breakout",
                param_type="int",
                min_val=5,
                max_val=30,
                default=10,
                description="离场突破周期",
            ),
            "atr_period": ParameterSpace(
                name="atr_period",
                param_type="int",
                min_val=10,
                max_val=30,
                default=20,
                description="ATR计算周期",
            ),
            "risk_per_trade": ParameterSpace(
                name="risk_per_trade",
                param_type="float",
                min_val=0.005,
                max_val=0.03,
                default=0.01,
                description="单笔风险比例",
            ),
            "max_units": ParameterSpace(
                name="max_units",
                param_type="int",
                min_val=1,
                max_val=6,
                default=4,
                description="最大加仓次数",
            ),
            "max_positions": ParameterSpace(
                name="max_positions",
                param_type="int",
                min_val=5,
                max_val=20,
                default=10,
                description="最大持仓数量",
            ),
            "pyramid_atr_multiple": ParameterSpace(
                name="pyramid_atr_multiple",
                param_type="float",
                min_val=0.25,
                max_val=1.0,
                default=0.5,
                description="加仓ATR倍数",
            ),
            "stop_atr_multiple": ParameterSpace(
                name="stop_atr_multiple",
                param_type="float",
                min_val=1.0,
                max_val=3.0,
                default=2.0,
                description="止损ATR倍数",
            ),
        }
    
    def set_data_source(self, data_source: "DataSourceAdapter") -> None:
        self._data_source = data_source
    
    def calculate_atr(
        self, 
        df: pd.DataFrame, 
        period: int = 20
    ) -> float:
        if len(df) < period + 1:
            return 0.0
        
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(period).mean().iloc[-1]
        
        return float(atr) if not pd.isna(atr) else 0.0
    
    def calculate_breakout_levels(
        self,
        df: pd.DataFrame,
        entry_period: int = 20,
        exit_period: int = 10
    ) -> Tuple[float, float, float, float]:
        if len(df) < entry_period:
            return 0.0, 0.0, 0.0, 0.0
        
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        
        entry_high = high.rolling(entry_period).max().iloc[-1]
        entry_low = low.rolling(entry_period).min().iloc[-1]
        
        exit_high = high.rolling(exit_period).max().iloc[-1]
        exit_low = low.rolling(exit_period).min().iloc[-1]
        
        return (
            float(entry_high) if not pd.isna(entry_high) else 0.0,
            float(entry_low) if not pd.isna(entry_low) else 0.0,
            float(exit_high) if not pd.isna(exit_high) else 0.0,
            float(exit_low) if not pd.isna(exit_low) else 0.0,
        )
    
    def calculate_position_size(
        self,
        atr: float,
        current_price: float,
        total_capital: float
    ) -> int:
        if atr <= 0 or current_price <= 0:
            return 100
        
        risk_per_trade = self.get_parameter("risk_per_trade", 0.01)
        stop_atr_multiple = self.get_parameter("stop_atr_multiple", 2.0)
        
        dollar_risk = total_capital * risk_per_trade
        stop_distance = atr * stop_atr_multiple
        
        unit_size = int(dollar_risk / stop_distance)
        
        shares = int(unit_size / current_price / 100) * 100
        
        return max(shares, 100)
    
    def calculate_score(
        self,
        code: str,
        factors: Dict[str, float],
        **kwargs
    ) -> float:
        df = kwargs.get("df")
        if df is None or df.empty:
            return 0.5
        
        entry_period = self.get_parameter("entry_breakout", 20)
        
        if len(df) < entry_period:
            return 0.5
        
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        
        current_price = close.iloc[-1]
        
        entry_high = high.rolling(entry_period).max().iloc[-2]
        entry_low = low.rolling(entry_period).min().iloc[-2]
        
        high_distance = (current_price - entry_low) / (entry_high - entry_low) if entry_high != entry_low else 0.5
        low_distance = (entry_high - current_price) / (entry_high - entry_low) if entry_high != entry_low else 0.5
        
        if current_price > entry_high:
            score = 0.7 + min(0.3, (current_price - entry_high) / entry_high * 10)
        elif current_price < entry_low:
            score = 0.3 - min(0.3, (entry_low - current_price) / entry_low * 10)
        else:
            score = high_distance * 0.4 + 0.3
        
        return max(0.0, min(1.0, score))
    
    def generate_signals(
        self,
        date: str,
        data: Dict[str, pd.DataFrame],
        positions: Dict[str, Any],
        **kwargs
    ) -> List[Signal]:
        signals = []
        
        entry_period = self.get_parameter("entry_breakout", 20)
        exit_period = self.get_parameter("exit_breakout", 10)
        atr_period = self.get_parameter("atr_period", 20)
        max_positions = self.get_parameter("max_positions", 10)
        max_units = self.get_parameter("max_units", 4)
        pyramid_atr_multiple = self.get_parameter("pyramid_atr_multiple", 0.5)
        stop_atr_multiple = self.get_parameter("stop_atr_multiple", 2.0)
        
        current_date = pd.to_datetime(date)
        current_date_str = str(date)[:10]
        total_capital = kwargs.get("total_capital", 1000000)
        
        for code, pos_data in positions.items():
            if code not in data:
                continue
            
            df = data[code]
            date_col = df["date"].astype(str) if "date" in df.columns else df.index.astype(str)
            df_before = df[date_col <= current_date_str].copy()
            
            if len(df_before) < exit_period + 1:
                continue
            
            current_price = float(df_before.iloc[-1]["close"])
            current_high = float(df_before.iloc[-1]["high"])
            current_low = float(df_before.iloc[-1]["low"])
            
            exit_high = df_before["high"].rolling(exit_period).max().iloc[-2]
            exit_low = df_before["low"].rolling(exit_period).min().iloc[-2]
            
            atr = self.calculate_atr(df_before, atr_period)
            
            entry_price = pos_data.get("entry_price", pos_data.get("buy_price", current_price))
            entry_signal = pos_data.get("entry_signal", "long")
            
            if entry_signal == "long":
                if current_price <= exit_low or current_price <= entry_price - atr * stop_atr_multiple:
                    signals.append(Signal(
                        code=code,
                        name=pos_data.get("name", code),
                        action="sell",
                        price=current_price,
                        shares=pos_data.get("shares", 0),
                        confidence=0.8,
                        reason=f"多头离场: 跌破{exit_period}日低点或止损",
                        strategy_id=self.strategy_id,
                        strategy_name=self.strategy_name,
                    ))
                elif current_price >= entry_price + atr * pyramid_atr_multiple:
                    pyramids = pos_data.get("pyramids", 0)
                    if pyramids < max_units:
                        signals.append(Signal(
                            code=code,
                            name=pos_data.get("name", code),
                            action="buy",
                            price=current_price,
                            shares=0,
                            confidence=0.7,
                            reason=f"多头加仓: 价格上涨{pyramid_atr_multiple}倍ATR",
                            strategy_id=self.strategy_id,
                            strategy_name=self.strategy_name,
                        ))
            
            elif entry_signal == "short":
                if current_price >= exit_high or current_price >= entry_price + atr * stop_atr_multiple:
                    signals.append(Signal(
                        code=code,
                        name=pos_data.get("name", code),
                        action="buy",
                        price=current_price,
                        shares=pos_data.get("shares", 0),
                        confidence=0.8,
                        reason=f"空头平仓: 突破{exit_period}日高点或止损",
                        strategy_id=self.strategy_id,
                        strategy_name=self.strategy_name,
                    ))
        
        current_position_count = len([p for p in positions.values() if p.get("entry_signal") == "long"])
        
        if current_position_count < max_positions:
            breakout_candidates = []
            
            for code, df in data.items():
                if df.empty:
                    continue
                
                date_col = df["date"].astype(str) if "date" in df.columns else df.index.astype(str)
                df_before = df[date_col <= current_date_str].copy()
                
                if len(df_before) < entry_period + 1:
                    continue
                
                if code in positions:
                    continue
                
                current_price = float(df_before.iloc[-1]["close"])
                current_high = float(df_before.iloc[-1]["high"])
                current_low = float(df_before.iloc[-1]["low"])
                
                entry_high = df_before["high"].rolling(entry_period).max().iloc[-2]
                entry_low = df_before["low"].rolling(entry_period).min().iloc[-2]
                
                if current_high >= entry_high:
                    atr = self.calculate_atr(df_before, atr_period)
                    breakout_candidates.append({
                        "code": code,
                        "price": current_price,
                        "high": entry_high,
                        "atr": atr,
                        "type": "long",
                        "strength": (current_price - entry_high) / entry_high if entry_high > 0 else 0,
                    })
                
                elif current_low <= entry_low:
                    atr = self.calculate_atr(df_before, atr_period)
                    breakout_candidates.append({
                        "code": code,
                        "price": current_price,
                        "low": entry_low,
                        "atr": atr,
                        "type": "short",
                        "strength": (entry_low - current_price) / entry_low if entry_low > 0 else 0,
                    })
            
            breakout_candidates.sort(key=lambda x: x["strength"], reverse=True)
            
            for candidate in breakout_candidates[:max_positions - current_position_count]:
                code = candidate["code"]
                current_price = candidate["price"]
                atr = candidate["atr"]
                
                df = data[code]
                df_before = df[date_col <= current_date_str]
                stock_name = df_before.iloc[-1].get("name", code) if not df_before.empty else code
                
                position_size = self.calculate_position_size(atr, current_price, total_capital)
                
                signals.append(Signal(
                    code=code,
                    name=stock_name,
                    action="buy",
                    price=current_price,
                    shares=position_size,
                    confidence=min(0.9, 0.6 + candidate["strength"] * 10),
                    reason=f"突破{entry_period}日{'高' if candidate['type'] == 'long' else '低'}点",
                    strategy_id=self.strategy_id,
                    strategy_name=self.strategy_name,
                    metadata={
                        "entry_signal": candidate["type"],
                        "atr": atr,
                        "breakout_level": candidate.get("high", candidate.get("low", 0)),
                    }
                ))
        
        return signals
    
    def get_turtle_position(self, code: str) -> Optional[TurtlePosition]:
        return self._turtle_positions.get(code)
    
    def update_turtle_position(
        self,
        code: str,
        high: float,
        low: float
    ) -> None:
        if code in self._turtle_positions:
            self._turtle_positions[code].update_price(high, low)
    
    def clear_cache(self) -> None:
        super().clear_cache()
        self._turtle_positions.clear()
        self._breakout_highs.clear()
        self._breakout_lows.clear()
