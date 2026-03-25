"""
Grid Trading Strategy - 网格交易策略

在设定的价格区间内，按照固定间隔设置买卖点，低买高卖。
适用于震荡市场。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from monitor.strategy_library.strategy_base import (
    StrategyBase,
    StrategyConfig,
    Signal,
    ParameterSpace,
)
from monitor.strategy_library.strategy_registry import register_strategy

logger = logging.getLogger(__name__)


@dataclass
class GridLevel:
    price: float
    shares: int
    is_buy: bool
    triggered: bool = False


@register_strategy(
    strategy_id="grid_trading",
    category="震荡策略",
    description="网格交易策略，在价格区间内低买高卖",
    tags=["网格", "震荡", "套利"],
)
class GridTradingStrategy(StrategyBase):
    STRATEGY_ID = "grid_trading"
    STRATEGY_NAME = "网格交易策略"
    STRATEGY_CATEGORY = "震荡策略"
    STRATEGY_VERSION = "1.0.0"
    
    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config=config, parameters=parameters)
        
        self._grids: Dict[str, List[GridLevel]] = {}
        self._base_prices: Dict[str, float] = {}
        self._grid_triggers: Dict[str, Dict[int, bool]] = {}
    
    def get_parameter_space(self) -> Dict[str, ParameterSpace]:
        return {
            "grid_count": ParameterSpace(
                name="grid_count",
                param_type="int",
                min_val=5,
                max_val=20,
                default=10,
                description="网格数量",
            ),
            "grid_spacing": ParameterSpace(
                name="grid_spacing",
                param_type="float",
                min_val=0.01,
                max_val=0.05,
                default=0.02,
                description="网格间距(%)",
            ),
            "position_per_grid": ParameterSpace(
                name="position_per_grid",
                param_type="float",
                min_val=0.05,
                max_val=0.20,
                default=0.10,
                description="每格仓位比例",
            ),
            "rebalance_threshold": ParameterSpace(
                name="rebalance_threshold",
                param_type="float",
                min_val=0.05,
                max_val=0.15,
                default=0.10,
                description="重新平衡阈值(%)",
            ),
        }
    
    def _initialize_grids(
        self,
        code: str,
        base_price: float,
        total_capital: float,
    ):
        grid_count = self.get_parameter("grid_count", 10)
        grid_spacing = self.get_parameter("grid_spacing", 0.02)
        position_per_grid = self.get_parameter("position_per_grid", 0.10)
        
        grids = []
        position_size = total_capital * position_per_grid
        
        for i in range(-grid_count, grid_count + 1):
            if i == 0:
                continue
            
            price = base_price * (1 + i * grid_spacing)
            shares = int(position_size / price / 100) * 100
            is_buy = i < 0
            
            grids.append(GridLevel(
                price=price,
                shares=shares,
                is_buy=is_buy,
                triggered=False,
            ))
        
        grids.sort(key=lambda x: x.price, reverse=True)
        self._grids[code] = grids
        self._base_prices[code] = base_price
        self._grid_triggers[code] = {i: False for i in range(len(grids))}
    
    def generate_signals(
        self,
        date: str,
        data: Dict[str, pd.DataFrame],
        positions: Dict[str, Any],
        **kwargs
    ) -> List[Signal]:
        signals = []
        
        grid_spacing = self.get_parameter("grid_spacing", 0.02)
        rebalance_threshold = self.get_parameter("rebalance_threshold", 0.10)
        total_capital = kwargs.get("total_capital", 1000000)
        
        current_date_str = str(date)[:10]
        
        for code, pos_data in positions.items():
            if code not in data:
                continue
            
            df = data[code]
            date_col = df["date"].astype(str) if "date" in df.columns else df.index.astype(str)
            df_before = df[date_col <= current_date_str]
            
            if df_before.empty:
                continue
            
            current_price = float(df_before.iloc[-1]["close"])
            entry_price = pos_data.get("entry_price", pos_data.get("avg_cost", current_price))
            
            if code in self._base_prices:
                base_price = self._base_prices[code]
                price_change = abs(current_price - base_price) / base_price
                
                if price_change > rebalance_threshold:
                    self._initialize_grids(code, current_price, total_capital)
            
            if code in self._grids:
                for i, grid in enumerate(self._grids[code]):
                    if self._grid_triggers[code].get(i, False):
                        continue
                    
                    if grid.is_buy and current_price <= grid.price:
                        signals.append(Signal(
                            code=code,
                            name=pos_data.get("name", code),
                            action="buy",
                            price=current_price,
                            shares=grid.shares,
                            confidence=0.7,
                            reason=f"网格买入: 价格{current_price:.2f} <= 网格{grid.price:.2f}",
                            strategy_id=self.strategy_id,
                            strategy_name=self.strategy_name,
                        ))
                        self._grid_triggers[code][i] = True
                    
                    elif not grid.is_buy and current_price >= grid.price:
                        signals.append(Signal(
                            code=code,
                            name=pos_data.get("name", code),
                            action="sell",
                            price=current_price,
                            shares=grid.shares,
                            confidence=0.7,
                            reason=f"网格卖出: 价格{current_price:.2f} >= 网格{grid.price:.2f}",
                            strategy_id=self.strategy_id,
                            strategy_name=self.strategy_name,
                        ))
                        self._grid_triggers[code][i] = True
        
        max_new_positions = 5 - len(positions)
        if max_new_positions <= 0:
            return signals
        
        for code, df in data.items():
            if code in positions:
                continue
            
            if df.empty:
                continue
            
            date_col = df["date"].astype(str) if "date" in df.columns else df.index.astype(str)
            df_before = df[date_col <= current_date_str]
            
            if df_before.empty or len(df_before) < 20:
                continue
            
            current_price = float(df_before.iloc[-1]["close"])
            
            recent_prices = df_before["close"].tail(20)
            volatility = recent_prices.pct_change().std()
            
            if volatility < 0.03:
                self._initialize_grids(code, current_price, total_capital)
                
                signals.append(Signal(
                    code=code,
                    name=df_before.iloc[-1].get("name", code),
                    action="buy",
                    price=current_price,
                    shares=0,
                    confidence=0.6,
                    reason=f"网格初始化: 波动率{volatility:.2%}适合网格",
                    strategy_id=self.strategy_id,
                    strategy_name=self.strategy_name,
                ))
                
                max_new_positions -= 1
                if max_new_positions <= 0:
                    break
        
        return signals
    
    def calculate_score(
        self,
        code: str,
        factors: Dict[str, float],
        **kwargs
    ) -> float:
        volatility = factors.get("volatility", 0.02)
        
        if volatility < 0.02:
            return 0.8
        elif volatility < 0.03:
            return 0.6
        elif volatility < 0.05:
            return 0.4
        else:
            return 0.2
    
    def clear_cache(self) -> None:
        super().clear_cache()
        self._grids.clear()
        self._base_prices.clear()
        self._grid_triggers.clear()
