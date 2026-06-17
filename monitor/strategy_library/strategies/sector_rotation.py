"""
Sector Rotation Strategy - 行业轮动策略

基于行业动量和相对强度进行行业轮动配置。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

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


SECTOR_MAPPING = {
    "银行": ["601398.SH", "601288.SH", "601939.SH", "601328.SH", "600036.SH"],
    "白酒": ["600519.SH", "000858.SZ", "000568.SZ", "002304.SZ"],
    "医药": ["000661.SZ", "600276.SH", "000538.SZ", "002007.SZ"],
    "新能源": ["002594.SZ", "300750.SZ", "002475.SZ", "600460.SH"],
    "半导体": ["002415.SZ", "603501.SH", "002371.SZ", "688981.SH"],
    "消费": ["000895.SZ", "600887.SH", "000651.SZ", "002714.SZ"],
    "科技": ["000063.SZ", "002230.SZ", "300059.SZ", "688111.SH"],
    "地产": ["000002.SZ", "001979.SZ", "600048.SH", "000069.SZ"],
}


@dataclass
class SectorScore:
    sector_name: str
    momentum: float
    relative_strength: float
    combined_score: float
    codes: List[str]


@register_strategy(
    strategy_id="sector_rotation",
    category="轮动策略",
    description="基于行业动量和相对强度进行行业轮动配置",
    tags=["行业", "轮动", "动量"],
)
class SectorRotationStrategy(StrategyBase):
    STRATEGY_ID = "sector_rotation"
    STRATEGY_NAME = "行业轮动策略"
    STRATEGY_CATEGORY = "轮动策略"
    STRATEGY_VERSION = "1.0.0"
    
    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config=config, parameters=parameters)
        
        self._sector_scores: Dict[str, SectorScore] = {}
        self._last_rotation_date: Optional[str] = None
    
    def get_parameter_space(self) -> Dict[str, ParameterSpace]:
        return {
            "momentum_period": ParameterSpace(
                name="momentum_period",
                param_type="int",
                min_val=10,
                max_val=60,
                default=20,
                description="动量计算周期",
            ),
            "rotation_threshold": ParameterSpace(
                name="rotation_threshold",
                param_type="float",
                min_val=0.02,
                max_val=0.10,
                default=0.05,
                description="轮动阈值",
            ),
            "top_sectors": ParameterSpace(
                name="top_sectors",
                param_type="int",
                min_val=1,
                max_val=4,
                default=3,
                description="持有行业数量",
            ),
            "min_holding_days": ParameterSpace(
                name="min_holding_days",
                param_type="int",
                min_val=3,
                max_val=15,
                default=5,
                description="最小持有天数",
            ),
            "momentum_weight": ParameterSpace(
                name="momentum_weight",
                param_type="float",
                min_val=0.3,
                max_val=0.7,
                default=0.5,
                description="动量权重",
            ),
        }
    
    def _calculate_sector_metrics(
        self,
        sector_name: str,
        codes: List[str],
        data: Dict[str, pd.DataFrame],
        current_date_str: str,
    ) -> Optional[SectorScore]:
        momentum_period = self.get_parameter("momentum_period", 20)
        
        sector_returns = []
        sector_prices = []
        
        for code in codes:
            if code not in data:
                continue
            
            df = data[code]
            date_col = df["date"].astype(str) if "date" in df.columns else df.index.astype(str)
            df_before = df[date_col <= current_date_str]
            
            if len(df_before) < momentum_period + 1:
                continue
            
            close_prices = df_before["close"].tail(momentum_period + 1)
            if len(close_prices) < momentum_period + 1:
                continue
            
            returns = close_prices.pct_change().dropna()
            if len(returns) < momentum_period:
                continue
            
            sector_returns.append(returns.mean())
            sector_prices.append(close_prices.iloc[-1] / close_prices.iloc[0] - 1)
        
        if not sector_returns:
            return None
        
        avg_momentum = np.mean(sector_returns)
        avg_relative_strength = np.mean(sector_prices)
        
        momentum_weight = self.get_parameter("momentum_weight", 0.5)
        combined_score = (
            momentum_weight * avg_momentum +
            (1 - momentum_weight) * avg_relative_strength
        )
        
        return SectorScore(
            sector_name=sector_name,
            momentum=avg_momentum,
            relative_strength=avg_relative_strength,
            combined_score=combined_score,
            codes=codes,
        )
    
    def _rank_sectors(
        self,
        data: Dict[str, pd.DataFrame],
        current_date_str: str,
    ) -> List[SectorScore]:
        sector_scores = []
        
        for sector_name, codes in SECTOR_MAPPING.items():
            score = self._calculate_sector_metrics(
                sector_name, codes, data, current_date_str
            )
            if score:
                sector_scores.append(score)
        
        sector_scores.sort(key=lambda x: x.combined_score, reverse=True)
        
        return sector_scores
    
    def generate_signals(
        self,
        date: str,
        data: Dict[str, pd.DataFrame],
        positions: Dict[str, Any],
        **kwargs
    ) -> List[Signal]:
        signals = []
        
        top_sectors = self.get_parameter("top_sectors", 3)
        min_holding_days = self.get_parameter("min_holding_days", 5)
        rotation_threshold = self.get_parameter("rotation_threshold", 0.05)
        
        current_date_str = str(date)[:10]
        current_date = pd.to_datetime(date)
        
        sector_rankings = self._rank_sectors(data, current_date_str)
        
        if not sector_rankings:
            return signals
        
        top_sector_names = [s.sector_name for s in sector_rankings[:top_sectors]]
        
        can_rotate = True
        for code, pos_data in positions.items():
            entry_date = pos_data.get("entry_date", pos_data.get("buy_date", date))
            holding_days = (current_date - pd.to_datetime(entry_date)).days
            if holding_days < min_holding_days:
                can_rotate = False
                break
        
        if can_rotate:
            for code, pos_data in list(positions.items()):
                sector = pos_data.get("sector", "")
                if sector not in top_sector_names:
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
                                confidence=0.7,
                                reason=f"行业轮动: {sector}退出前{top_sectors}",
                                strategy_id=self.strategy_id,
                                strategy_name=self.strategy_name,
                            ))
        
        current_sector_exposure = {}
        for code, pos_data in positions.items():
            sector = pos_data.get("sector", "unknown")
            current_sector_exposure[sector] = current_sector_exposure.get(sector, 0) + 1
        
        for sector_score in sector_rankings[:top_sectors]:
            sector_name = sector_score.sector_name
            
            if current_sector_exposure.get(sector_name, 0) >= 2:
                continue
            
            available_codes = [c for c in sector_score.codes if c not in positions]
            if not available_codes:
                continue
            
            best_code = available_codes[0]
            df = data.get(best_code)
            if df is None:
                continue
            
            date_col = df["date"].astype(str) if "date" in df.columns else df.index.astype(str)
            df_before = df[date_col <= current_date_str]
            
            if df_before.empty:
                continue
            
            current_price = float(df_before.iloc[-1]["close"])
            
            signals.append(Signal(
                code=best_code,
                name=df_before.iloc[-1].get("name", best_code),
                action="buy",
                price=current_price,
                shares=0,
                confidence=sector_score.combined_score,
                reason=f"行业轮动: {sector_name}评分{sector_score.combined_score:.3f}",
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
        sector = kwargs.get("sector", "unknown")
        
        if sector in self._sector_scores:
            return self._sector_scores[sector].combined_score
        
        momentum = factors.get("momentum", 0)
        relative_strength = factors.get("relative_strength", 0)
        
        momentum_weight = self.get_parameter("momentum_weight", 0.5)
        return momentum_weight * momentum + (1 - momentum_weight) * relative_strength
    
    def clear_cache(self) -> None:
        super().clear_cache()
        self._sector_scores.clear()
        self._last_rotation_date = None
