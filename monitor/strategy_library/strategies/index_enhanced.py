"""
Index Enhanced Strategy - 指数增强策略

以指数为基准，通过多因子选股增强收益。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
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


INDEX_COMPONENTS = {
    "000300.SH": {
        "name": "沪深300",
        "codes": [
            "600519.SH", "601318.SH", "600036.SH", "601166.SH",
            "000858.SZ", "600000.SH", "601398.SH", "600276.SH",
            "000333.SZ", "600030.SH", "601288.SH", "600887.SH",
            "000651.SZ", "601939.SH", "600016.SH", "601328.SH",
        ],
    },
    "000905.SH": {
        "name": "中证500",
        "codes": [
            "002415.SZ", "002594.SZ", "300750.SZ", "002475.SZ",
            "000063.SZ", "002230.SZ", "300059.SZ", "002352.SZ",
            "600460.SH", "002371.SZ", "603501.SH", "002714.SZ",
            "000661.SZ", "002007.SZ", "600438.SH", "300015.SZ",
        ],
    },
    "000852.SH": {
        "name": "中证1000",
        "codes": [
            "688981.SH", "688111.SH", "688012.SH", "688599.SH",
            "300124.SZ", "300014.SZ", "300033.SZ", "002920.SZ",
            "603160.SH", "603259.SH", "002938.SZ", "300146.SZ",
            "688169.SH", "688256.SH", "300347.SZ", "002821.SZ",
        ],
    },
}


@dataclass
class StockFactor:
    code: str
    name: str
    factors: Dict[str, float]
    score: float
    weight: float


@register_strategy(
    strategy_id="index_enhanced",
    category="增强策略",
    description="以指数为基准，通过多因子选股增强收益",
    tags=["指数", "增强", "因子"],
)
class IndexEnhancedStrategy(StrategyBase):
    STRATEGY_ID = "index_enhanced"
    STRATEGY_NAME = "指数增强策略"
    STRATEGY_CATEGORY = "增强策略"
    STRATEGY_VERSION = "1.0.0"
    
    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config=config, parameters=parameters)
        
        self._stock_scores: Dict[str, StockFactor] = {}
        self._current_index: str = "000300.SH"
    
    def get_parameter_space(self) -> Dict[str, ParameterSpace]:
        return {
            "target_index": ParameterSpace(
                name="target_index",
                param_type="choice",
                choices=["000300.SH", "000905.SH", "000852.SH"],
                default="000300.SH",
                description="目标指数",
            ),
            "enhancement_factor": ParameterSpace(
                name="enhancement_factor",
                param_type="float",
                min_val=0.5,
                max_val=2.0,
                default=1.0,
                description="增强因子",
            ),
            "max_tracking_error": ParameterSpace(
                name="max_tracking_error",
                param_type="float",
                min_val=0.02,
                max_val=0.08,
                default=0.04,
                description="最大跟踪误差",
            ),
            "rebalance_freq": ParameterSpace(
                name="rebalance_freq",
                param_type="int",
                min_val=5,
                max_val=30,
                default=20,
                description="再平衡频率(天)",
            ),
            "top_n_stocks": ParameterSpace(
                name="top_n_stocks",
                param_type="int",
                min_val=10,
                max_val=50,
                default=30,
                description="持仓股票数量",
            ),
        }
    
    def _calculate_stock_score(
        self,
        code: str,
        df: pd.DataFrame,
        current_date_str: str,
    ) -> Optional[StockFactor]:
        date_col = df["date"].astype(str) if "date" in df.columns else df.index.astype(str)
        df_before = df[date_col <= current_date_str]
        
        if len(df_before) < 30:
            return None
        
        close = df_before["close"]
        volume = df_before.get("volume", df_before.get("vol", pd.Series([1] * len(df_before))))
        
        returns = close.pct_change().dropna()
        
        momentum_20 = returns.tail(20).mean() if len(returns) >= 20 else 0
        volatility = returns.tail(20).std() if len(returns) >= 20 else 0
        
        price_change = (close.iloc[-1] / close.iloc[-20] - 1) if len(close) >= 20 else 0
        
        vol_ma = volume.tail(20).mean()
        vol_ratio = volume.iloc[-1] / vol_ma if vol_ma > 0 else 1
        
        factors = {
            "momentum": momentum_20,
            "volatility": volatility,
            "price_change": price_change,
            "volume_ratio": vol_ratio,
        }
        
        score = (
            0.3 * momentum_20 +
            0.2 * (1 - volatility if volatility < 1 else 0) +
            0.3 * price_change +
            0.2 * (vol_ratio - 1 if vol_ratio > 1 else 0)
        )
        
        return StockFactor(
            code=code,
            name=df_before.iloc[-1].get("name", code),
            factors=factors,
            score=score,
            weight=0.0,
        )
    
    def _rank_stocks(
        self,
        data: Dict[str, pd.DataFrame],
        current_date_str: str,
    ) -> List[StockFactor]:
        target_index = self.get_parameter("target_index", "000300.SH")
        index_info = INDEX_COMPONENTS.get(target_index, INDEX_COMPONENTS["000300.SH"])
        index_codes = index_info["codes"]
        
        stock_scores = []
        
        for code in index_codes:
            if code not in data:
                continue
            
            df = data[code]
            score = self._calculate_stock_score(code, df, current_date_str)
            if score:
                stock_scores.append(score)
        
        stock_scores.sort(key=lambda x: x.score, reverse=True)
        
        top_n = self.get_parameter("top_n_stocks", 30)
        top_stocks = stock_scores[:top_n]
        
        total_score = sum(s.score for s in top_stocks) if top_stocks else 1
        for stock in top_stocks:
            stock.weight = stock.score / total_score if total_score > 0 else 1 / len(top_stocks)
        
        return top_stocks
    
    def generate_signals(
        self,
        date: str,
        data: Dict[str, pd.DataFrame],
        positions: Dict[str, Any],
        **kwargs
    ) -> List[Signal]:
        signals = []
        
        rebalance_freq = self.get_parameter("rebalance_freq", 20)
        current_date_str = str(date)[:10]
        current_date = pd.to_datetime(date)
        
        should_rebalance = False
        if not positions:
            should_rebalance = True
        else:
            first_pos = list(positions.values())[0]
            entry_date = first_pos.get("entry_date", first_pos.get("buy_date", date))
            holding_days = (current_date - pd.to_datetime(entry_date)).days
            if holding_days >= rebalance_freq:
                should_rebalance = True
        
        if not should_rebalance:
            return signals
        
        ranked_stocks = self._rank_stocks(data, current_date_str)
        
        if not ranked_stocks:
            return signals
        
        target_codes = {s.code for s in ranked_stocks}
        
        for code, pos_data in list(positions.items()):
            if code not in target_codes:
                df = data.get(code)
                if df is not None:
                    date_col = df["date"].astype(str) if "date" in df.columns else df.index.astype(str)
                    df_before = df[date_col <= current_date_str]
                    if not df_before.empty:
                        current_price = float(df_before.iloc[-1]["close"])
                        signals.append(Signal(
                            code=code,
                            name=pos_data.get("name", code),
                            action="sell",
                            price=current_price,
                            shares=pos_data.get("shares", 0),
                            confidence=0.8,
                            reason="指数增强再平衡: 调出成分股",
                            strategy_id=self.strategy_id,
                            strategy_name=self.strategy_name,
                        ))
        
        for stock in ranked_stocks:
            if stock.code in positions:
                continue
            
            df = data.get(stock.code)
            if df is None:
                continue
            
            date_col = df["date"].astype(str) if "date" in df.columns else df.index.astype(str)
            df_before = df[date_col <= current_date_str]
            
            if df_before.empty:
                continue
            
            current_price = float(df_before.iloc[-1]["close"])
            
            signals.append(Signal(
                code=stock.code,
                name=stock.name,
                action="buy",
                price=current_price,
                shares=0,
                confidence=stock.score,
                reason=f"指数增强: 评分{stock.score:.3f}, 权重{stock.weight:.2%}",
                strategy_id=self.strategy_id,
                strategy_name=self.strategy_name,
            ))
        
        return signals
    
    def calculate_score(
        self,
        code: str,
        factors: Dict[str, float],
        **kwargs
    ) -> float:
        if code in self._stock_scores:
            return self._stock_scores[code].score
        
        momentum = factors.get("momentum", 0)
        volatility = factors.get("volatility", 0)
        price_change = factors.get("price_change", 0)
        volume_ratio = factors.get("volume_ratio", 1)
        
        score = (
            0.3 * momentum +
            0.2 * (1 - volatility if volatility < 1 else 0) +
            0.3 * price_change +
            0.2 * (volume_ratio - 1 if volume_ratio > 1 else 0)
        )
        
        return score
    
    def clear_cache(self) -> None:
        super().clear_cache()
        self._stock_scores.clear()
