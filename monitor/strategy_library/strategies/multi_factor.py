"""
Multi-Factor Strategy - 多因子选股策略

基于多个因子综合评分进行选股的策略。
支持价值因子、动量因子、质量因子、成长因子等多种因子组合。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

from monitor.strategy_library.strategy_base import (
    StrategyBase,
    StrategyConfig,
    Signal,
    BacktestResult,
    ParameterSpace,
)
from monitor.strategy_library.strategy_registry import StrategyRegistry, register_strategy

if TYPE_CHECKING:
    from monitor.data_source import DataSourceAdapter

logger = logging.getLogger(__name__)


@dataclass
class FactorDefinition:
    name: str
    factor_type: str
    weight: float
    direction: int = 1
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.factor_type,
            "weight": self.weight,
            "direction": self.direction,
            "min": self.min_value,
            "max": self.max_value,
            "description": self.description,
        }


@register_strategy(
    strategy_id="multi_factor",
    category="选股策略",
    description="基于多因子综合评分进行选股的策略",
    tags=["选股", "因子", "量化"],
)
class MultiFactorStrategy(StrategyBase):
    STRATEGY_ID = "multi_factor"
    STRATEGY_NAME = "多因子选股策略"
    STRATEGY_CATEGORY = "选股策略"
    STRATEGY_VERSION = "1.0.0"
    
    DEFAULT_FACTORS = {
        "value": {
            "pb": FactorDefinition("pb", "value", 0.25, -1, 0.1, 10.0, "市净率"),
            "pe": FactorDefinition("pe", "value", 0.25, -1, 0.0, 100.0, "市盈率"),
            "dividend_yield": FactorDefinition("dividend_yield", "value", 0.10, 1, 0.0, 0.1, "股息率"),
        },
        "momentum": {
            "price_momentum_20d": FactorDefinition("price_momentum_20d", "momentum", 0.15, 1, -0.3, 0.5, "20日价格动量"),
            "volume_momentum": FactorDefinition("volume_momentum", "momentum", 0.05, 1, 0.5, 3.0, "成交量动量"),
        },
        "quality": {
            "roe": FactorDefinition("roe", "quality", 0.10, 1, 0.0, 0.5, "净资产收益率"),
            "debt_ratio": FactorDefinition("debt_ratio", "quality", 0.05, -1, 0.0, 0.8, "资产负债率"),
        },
        "technical": {
            "rsi_14": FactorDefinition("rsi_14", "technical", 0.05, -1, 20.0, 80.0, "RSI指标"),
        },
    }
    
    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        parameters: Optional[Dict[str, Any]] = None,
        data_source: Optional["DataSourceAdapter"] = None,
    ):
        super().__init__(config=config, parameters=parameters)
        
        self._data_source = data_source
        self._factor_definitions = self._build_factor_definitions()
        self._factor_cache: Dict[str, Dict[str, float]] = {}
        self._score_cache: Dict[str, float] = {}
    
    def _build_factor_definitions(self) -> Dict[str, FactorDefinition]:
        factors = {}
        
        for category, category_factors in self.DEFAULT_FACTORS.items():
            for name, factor_def in category_factors.items():
                weight_key = f"{category}_weight"
                if weight_key in self._parameters:
                    factor_def = FactorDefinition(
                        name=factor_def.name,
                        factor_type=factor_def.factor_type,
                        weight=self._parameters[weight_key] * factor_def.weight,
                        direction=factor_def.direction,
                        min_value=factor_def.min_value,
                        max_value=factor_def.max_value,
                        description=factor_def.description,
                    )
                factors[name] = factor_def
        
        return factors
    
    def get_parameter_space(self) -> Dict[str, ParameterSpace]:
        return {
            "value_weight": ParameterSpace(
                name="value_weight",
                param_type="float",
                min_val=0.0,
                max_val=1.0,
                default=0.3,
                description="价值因子权重",
            ),
            "momentum_weight": ParameterSpace(
                name="momentum_weight",
                param_type="float",
                min_val=0.0,
                max_val=1.0,
                default=0.3,
                description="动量因子权重",
            ),
            "quality_weight": ParameterSpace(
                name="quality_weight",
                param_type="float",
                min_val=0.0,
                max_val=1.0,
                default=0.2,
                description="质量因子权重",
            ),
            "technical_weight": ParameterSpace(
                name="technical_weight",
                param_type="float",
                min_val=0.0,
                max_val=1.0,
                default=0.2,
                description="技术因子权重",
            ),
            "top_k": ParameterSpace(
                name="top_k",
                param_type="int",
                min_val=5,
                max_val=50,
                default=10,
                description="选股数量",
            ),
            "min_score": ParameterSpace(
                name="min_score",
                param_type="float",
                min_val=0.3,
                max_val=0.7,
                default=0.5,
                description="最低评分阈值",
            ),
            "holding_days": ParameterSpace(
                name="holding_days",
                param_type="int",
                min_val=1,
                max_val=20,
                default=5,
                description="持仓天数",
            ),
            "stop_loss": ParameterSpace(
                name="stop_loss",
                param_type="float",
                min_val=-0.15,
                max_val=-0.03,
                default=-0.08,
                description="止损比例",
            ),
            "take_profit": ParameterSpace(
                name="take_profit",
                param_type="float",
                min_val=0.05,
                max_val=0.30,
                default=0.15,
                description="止盈比例",
            ),
        }
    
    def set_data_source(self, data_source: "DataSourceAdapter") -> None:
        self._data_source = data_source
    
    def fetch_factors(self, code: str) -> Dict[str, float]:
        if code in self._factor_cache:
            return self._factor_cache[code]
        
        factors = {}
        
        if self._data_source is not None:
            try:
                factors = self._data_source.get_stock_factors(code)
            except Exception as e:
                logger.debug(f"获取因子失败 {code}: {e}")
        
        if not factors:
            factors = self._get_default_factors()
        
        self._factor_cache[code] = factors
        return factors
    
    def _get_default_factors(self) -> Dict[str, float]:
        return {
            "pb": 1.5,
            "pe": 15.0,
            "dividend_yield": 0.025,
            "price_momentum_20d": 0.0,
            "volume_momentum": 1.0,
            "roe": 0.10,
            "debt_ratio": 0.40,
            "rsi_14": 50.0,
        }
    
    def normalize_factor(
        self, 
        value: float, 
        factor_def: FactorDefinition
    ) -> float:
        if factor_def.min_value is not None and factor_def.max_value is not None:
            min_val = factor_def.min_value
            max_val = factor_def.max_value
            
            normalized = (value - min_val) / (max_val - min_val)
            normalized = max(0.0, min(1.0, normalized))
        else:
            normalized = 0.5
        
        if factor_def.direction < 0:
            normalized = 1.0 - normalized
        
        return normalized
    
    def calculate_score(
        self,
        code: str,
        factors: Dict[str, float],
        **kwargs
    ) -> float:
        cache_key = f"{code}_{hash(tuple(sorted(factors.items())))}"
        if cache_key in self._score_cache:
            return self._score_cache[cache_key]
        
        total_score = 0.0
        total_weight = 0.0
        
        for factor_name, factor_def in self._factor_definitions.items():
            if factor_name in factors:
                value = factors[factor_name]
                
                if pd.isna(value) or not np.isfinite(value):
                    continue
                
                normalized = self.normalize_factor(value, factor_def)
                
                total_score += normalized * factor_def.weight
                total_weight += factor_def.weight
        
        final_score = total_score / total_weight if total_weight > 0 else 0.5
        
        self._score_cache[cache_key] = final_score
        return final_score
    
    def calculate_all_scores(
        self,
        codes: List[str],
        date: Optional[str] = None
    ) -> Dict[str, float]:
        scores = {}
        
        for code in codes:
            try:
                factors = self.fetch_factors(code)
                score = self.calculate_score(code, factors)
                scores[code] = score
            except Exception as e:
                logger.debug(f"计算评分失败 {code}: {e}")
                scores[code] = 0.0
        
        return scores
    
    def generate_signals(
        self,
        date: str,
        data: Dict[str, pd.DataFrame],
        positions: Dict[str, Any],
        **kwargs
    ) -> List[Signal]:
        signals = []
        
        top_k = self.get_parameter("top_k", 10)
        min_score = self.get_parameter("min_score", 0.5)
        holding_days = self.get_parameter("holding_days", 5)
        stop_loss = self.get_parameter("stop_loss", -0.08)
        take_profit = self.get_parameter("take_profit", 0.15)
        
        current_date = pd.to_datetime(date)
        current_date_str = str(date)[:10]
        
        for code, pos in positions.items():
            if code not in data:
                continue
            
            df = data[code]
            date_col = df["date"].astype(str) if "date" in df.columns else df.index.astype(str)
            df_before = df[date_col <= current_date_str]
            
            if df_before.empty:
                continue
            
            current_price = float(df_before.iloc[-1]["close"])
            entry_price = pos.get("entry_price", pos.get("buy_price", current_price))
            entry_date = pos.get("entry_date", pos.get("buy_date", date))
            
            profit_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0
            
            holding = (current_date - pd.to_datetime(entry_date)).days
            
            if profit_pct <= stop_loss:
                signals.append(Signal(
                    code=code,
                    name=pos.get("name", code),
                    action="sell",
                    price=current_price,
                    shares=pos.get("shares", 0),
                    confidence=0.9,
                    reason=f"触发止损: {profit_pct:.2%}",
                    strategy_id=self.strategy_id,
                    strategy_name=self.strategy_name,
                ))
            elif profit_pct >= take_profit:
                signals.append(Signal(
                    code=code,
                    name=pos.get("name", code),
                    action="sell",
                    price=current_price,
                    shares=pos.get("shares", 0),
                    confidence=0.8,
                    reason=f"触发止盈: {profit_pct:.2%}",
                    strategy_id=self.strategy_id,
                    strategy_name=self.strategy_name,
                ))
            elif holding >= holding_days:
                signals.append(Signal(
                    code=code,
                    name=pos.get("name", code),
                    action="sell",
                    price=current_price,
                    shares=pos.get("shares", 0),
                    confidence=0.6,
                    reason=f"持仓到期: {holding}天",
                    strategy_id=self.strategy_id,
                    strategy_name=self.strategy_name,
                ))
        
        scores = {}
        for code, df in data.items():
            if df.empty:
                continue
            
            date_col = df["date"].astype(str) if "date" in df.columns else df.index.astype(str)
            df_before = df[date_col <= current_date_str]
            if df_before.empty:
                continue
            
            if code in positions:
                continue
            
            factors = self.fetch_factors(code)
            score = self.calculate_score(code, factors)
            
            if score >= min_score:
                scores[code] = score
        
        sorted_stocks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        max_stocks = kwargs.get("max_stocks", top_k)
        current_positions = len([s for s in signals if s.action == "buy"])
        
        for code, score in sorted_stocks[:top_k]:
            if current_positions + len([s for s in signals if s.action == "buy"]) >= max_stocks:
                break
            
            df = data[code]
            date_col = df["date"].astype(str) if "date" in df.columns else df.index.astype(str)
            df_before = df[date_col <= current_date_str]
            
            if df_before.empty:
                continue
            
            current_price = float(df_before.iloc[-1]["close"])
            stock_name = df_before.iloc[-1].get("name", code)
            
            signals.append(Signal(
                code=code,
                name=stock_name,
                action="buy",
                price=current_price,
                shares=0,
                confidence=score,
                reason=f"多因子评分: {score:.3f}",
                strategy_id=self.strategy_id,
                strategy_name=self.strategy_name,
            ))
        
        return signals
    
    def get_factor_weights(self) -> Dict[str, float]:
        weights = {}
        for name, factor_def in self._factor_definitions.items():
            weights[name] = factor_def.weight
        return weights
    
    def get_factor_importance(self) -> Dict[str, float]:
        weights = self.get_factor_weights()
        total = sum(weights.values())
        
        if total == 0:
            return weights
        
        return {k: v / total for k, v in weights.items()}
    
    def clear_cache(self) -> None:
        super().clear_cache()
        self._factor_cache.clear()
        self._score_cache.clear()
