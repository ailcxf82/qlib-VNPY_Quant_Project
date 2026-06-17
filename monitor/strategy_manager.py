from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd
import yaml

if TYPE_CHECKING:
    from monitor.data_source import DataSourceAdapter

logger = logging.getLogger(__name__)


@dataclass
class StrategySignal:
    code: str
    name: str
    strategy: str
    signal_type: str
    score: float
    weight: float
    reason: str
    confidence: float = 0.5
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "strategy": self.strategy,
            "signal_type": self.signal_type,
            "score": self.score,
            "weight": self.weight,
            "reason": self.reason,
            "confidence": self.confidence,
            "details": self.details,
        }


@dataclass
class CombinedSignal:
    code: str
    name: str
    total_score: float
    weighted_score: float
    strategy_scores: Dict[str, float]
    strategy_weights: Dict[str, float]
    signal_type: str
    confidence: float
    reasons: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "total_score": self.total_score,
            "weighted_score": self.weighted_score,
            "strategy_scores": self.strategy_scores,
            "strategy_weights": self.strategy_weights,
            "signal_type": self.signal_type,
            "confidence": self.confidence,
            "reasons": self.reasons,
        }


class StrategyManager:
    STRATEGY_DEFINITIONS = {
        "value": {
            "name": "价值策略",
            "description": "低估值股票，关注PB、PE等估值指标",
            "factors": ["pb", "pe", "dividend_yield", "roe"],
            "weight_range": (0.15, 0.40),
            "min_score": 0.5,
        },
        "momentum": {
            "name": "动量策略",
            "description": "趋势向上股票，关注价格动量和成交量",
            "factors": ["price_momentum", "volume_momentum", "relative_strength"],
            "weight_range": (0.10, 0.40),
            "min_score": 0.6,
        },
        "mean_reversion": {
            "name": "均值回归策略",
            "description": "超跌反弹股票，关注技术指标背离",
            "factors": ["rsi_oversold", "price_deviation", "volume_spike"],
            "weight_range": (0.15, 0.35),
            "min_score": 0.55,
        },
        "quality": {
            "name": "质量策略",
            "description": "高质量股票，关注盈利能力和财务健康",
            "factors": ["roe", "roa", "debt_ratio", "cash_flow"],
            "weight_range": (0.20, 0.40),
            "min_score": 0.5,
        },
    }
    
    def __init__(
        self, 
        config_path: str = "config/monitor.yaml",
        data_source: Optional["DataSourceAdapter"] = None,
        data_source_type: str = "auto"
    ):
        self.config_path = config_path
        self.config = self._load_config()
        self.strategy_config = self.config.get("strategies", {})
        
        self._strategy_weights: Dict[str, float] = {}
        self._strategy_signals: Dict[str, List[StrategySignal]] = {}
        self._factor_cache: Dict[str, Dict[str, float]] = {}
        self._factor_cache_time: Dict[str, datetime] = {}
        self._cache_expiry_hours: int = 4
        
        self._data_source = data_source
        self._data_source_type = data_source_type
        self._ts_client = None
    
    def _load_config(self) -> Dict[str, Any]:
        p = Path(self.config_path)
        if not p.exists():
            return {}
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    
    @property
    def data_source(self) -> Optional["DataSourceAdapter"]:
        if self._data_source is None:
            try:
                from monitor.data_source import get_data_source
                self._data_source = get_data_source("akshare")
                logger.info("Using Akshare data source for StrategyManager")
            except Exception as e:
                logger.warning(f"Failed to initialize Akshare data source: {e}")
        return self._data_source
    
    def _get_ts_client(self):
        if self._ts_client is None and self._data_source_type in ["auto", "tushare"]:
            try:
                from monitor.ts_client import get_ts_client
                self._ts_client = get_ts_client()
            except Exception as e:
                logger.debug(f"Tushare client not available: {e}")
        return self._ts_client
    
    def _is_factor_cache_valid(self, code: str) -> bool:
        if code not in self._factor_cache_time:
            return False
        elapsed = (datetime.now() - self._factor_cache_time[code]).total_seconds() / 3600
        return elapsed < self._cache_expiry_hours
    
    def fetch_stock_factors(self, code: str) -> Dict[str, float]:
        if code in self._factor_cache and self._is_factor_cache_valid(code):
            return self._factor_cache[code]
        
        factors = {
            "pb": 1.0,
            "pe": 15.0,
            "dividend_yield": 0.02,
            "roe": 0.10,
            "roa": 0.05,
            "debt_ratio": 0.40,
            "cash_flow": 1.0,
            "price_momentum_5d": 0.0,
            "price_momentum_20d": 0.0,
            "volume_momentum": 1.0,
            "relative_strength": 0.5,
            "rsi_14": 50.0,
            "price_deviation_ma20": 0.0,
            "volume_spike": 1.0,
        }
        
        if self.data_source is not None:
            try:
                ds_factors = self.data_source.get_stock_factors(code)
                if ds_factors:
                    factors.update(ds_factors)
                    logger.debug(f"Got factors from Akshare for {code}")
                    self._factor_cache[code] = factors
                    self._factor_cache_time[code] = datetime.now()
                    return factors
            except Exception as e:
                logger.debug(f"Akshare factor fetch failed for {code}: {e}")
        
        ts = self._get_ts_client()
        if ts is None:
            logger.debug(f"No data source available for {code}, using defaults")
            return factors
        
        try:
            code_clean = str(code).replace(".XSHE", "").replace(".XSHG", "").zfill(6)
            suffix = ".SZ" if code_clean.startswith(("0", "3")) else ".SH"
            ts_code = code_clean + suffix
            
            daily_basic = ts._pro.daily_basic(ts_code=ts_code, fields="pb,pe,turnover_rate")
            if not daily_basic.empty:
                last_row = daily_basic.iloc[-1]
                factors["pb"] = float(last_row.get("pb", 1.0) or 1.0)
                factors["pe"] = float(last_row.get("pe", 15.0) or 15.0)
            
            try:
                fins = ts._pro.fina_indicator(ts_code=ts_code, fields="roe,roa,debt_to_assets")
                if not fins.empty:
                    last_fin = fins.iloc[-1]
                    factors["roe"] = float(last_fin.get("roe", 0.10) or 0.10) / 100 if last_fin.get("roe") else 0.10
                    factors["roa"] = float(last_fin.get("roa", 0.05) or 0.05) / 100 if last_fin.get("roa") else 0.05
                    factors["debt_ratio"] = float(last_fin.get("debt_to_assets", 0.40) or 0.40)
            except Exception:
                pass
            
            try:
                end_date = datetime.now().strftime("%Y%m%d")
                start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
                daily = ts._pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
                
                if not daily.empty and len(daily) >= 20:
                    daily = daily.sort_values("trade_date")
                    close = daily["close"].astype(float)
                    vol = daily["vol"].astype(float)
                    
                    momentum_5d = (close.iloc[-1] / close.iloc[-5] - 1) if len(close) >= 5 else 0
                    momentum_20d = (close.iloc[-1] / close.iloc[-20] - 1) if len(close) >= 20 else 0
                    factors["price_momentum_5d"] = momentum_5d
                    factors["price_momentum_20d"] = momentum_20d
                    
                    vol_ma5 = vol.rolling(5).mean().iloc[-1]
                    factors["volume_momentum"] = vol.iloc[-1] / vol_ma5 if vol_ma5 > 0 else 1.0
                    
                    delta = close.diff()
                    gain = delta.where(delta > 0, 0)
                    loss = (-delta).where(delta < 0, 0)
                    avg_gain = gain.rolling(14).mean().iloc[-1]
                    avg_loss = loss.rolling(14).mean().iloc[-1]
                    if avg_loss > 0:
                        rs = avg_gain / avg_loss
                        rsi = 100 - (100 / (1 + rs))
                        factors["rsi_14"] = float(rsi) if not pd.isna(rsi) else 50.0
                    
                    ma20 = close.rolling(20).mean().iloc[-1]
                    factors["price_deviation_ma20"] = (close.iloc[-1] - ma20) / ma20 if ma20 > 0 else 0
                    
                    vol_ma20 = vol.rolling(20).mean().iloc[-1]
                    factors["volume_spike"] = vol.iloc[-1] / vol_ma20 if vol_ma20 > 0 else 1.0
            except Exception:
                pass
            
            logger.debug(f"Got factors from Tushare for {code}")
            
        except Exception as e:
            logger.debug(f"获取股票因子失败 {code}: {e}")
        
        self._factor_cache[code] = factors
        self._factor_cache_time[code] = datetime.now()
        return factors
    
    def calculate_value_score(self, factors: Dict[str, float]) -> float:
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
    
    def calculate_momentum_score(self, factors: Dict[str, float]) -> float:
        mom_5d = factors.get("price_momentum_5d", 0.0)
        mom_20d = factors.get("price_momentum_20d", 0.0)
        vol_mom = factors.get("volume_momentum", 1.0)
        rs = factors.get("relative_strength", 0.5)
        
        mom_5d_score = max(0, min(1, (mom_5d + 0.05) / 0.10))
        mom_20d_score = max(0, min(1, (mom_20d + 0.10) / 0.20))
        vol_score = min(vol_mom / 2.0, 1.0)
        rs_score = rs
        
        momentum_score = mom_5d_score * 0.30 + mom_20d_score * 0.35 + vol_score * 0.15 + rs_score * 0.20
        return momentum_score
    
    def calculate_mean_reversion_score(self, factors: Dict[str, float]) -> float:
        rsi = factors.get("rsi_14", 50.0)
        deviation = factors.get("price_deviation_ma20", 0.0)
        vol_spike = factors.get("volume_spike", 1.0)
        
        rsi_score = max(0, (40 - rsi) / 40) if rsi < 40 else max(0, (rsi - 60) / 40)
        deviation_score = max(0, min(1, abs(deviation) / 0.10))
        vol_score = min(vol_spike / 2.0, 1.0)
        
        mr_score = rsi_score * 0.40 + deviation_score * 0.35 + vol_score * 0.25
        return mr_score
    
    def calculate_quality_score(self, factors: Dict[str, float]) -> float:
        roe = factors.get("roe", 0.10)
        roa = factors.get("roa", 0.05)
        debt = factors.get("debt_ratio", 0.40)
        
        roe_score = min(roe / 0.15, 1.0)
        roa_score = min(roa / 0.08, 1.0)
        debt_score = 1.0 - min(debt / 0.60, 1.0)
        
        quality_score = roe_score * 0.40 + roa_score * 0.30 + debt_score * 0.30
        return quality_score
    
    def calculate_all_strategy_scores(self, code: str) -> Dict[str, float]:
        factors = self.fetch_stock_factors(code)
        
        return {
            "value": self.calculate_value_score(factors),
            "momentum": self.calculate_momentum_score(factors),
            "mean_reversion": self.calculate_mean_reversion_score(factors),
            "quality": self.calculate_quality_score(factors),
        }
    
    def update_weights(self, regime_weights: Dict[str, float]) -> None:
        self._strategy_weights = regime_weights.copy()
        logger.info(f"策略权重更新: {self._strategy_weights}")
    
    def set_strategy_signals(self, strategy: str, signals: List[StrategySignal]) -> None:
        self._strategy_signals[strategy] = signals
    
    def combine_signals(
        self,
        regime_weights: Dict[str, float],
        model_signals: List[Any],
        sentiment_scores: Dict[str, float],
        technical_scores: Dict[str, float],
    ) -> List[CombinedSignal]:
        self.update_weights(regime_weights)
        
        stock_signals: Dict[str, Dict[str, Any]] = {}
        
        for sig in model_signals:
            code = sig.code if hasattr(sig, 'code') else sig.get("code", "")
            name = sig.name if hasattr(sig, 'name') else sig.get("name", "")
            score = sig.score if hasattr(sig, 'score') else sig.get("score", 0.5)
            
            if code not in stock_signals:
                strategy_scores = self.calculate_all_strategy_scores(code)
                
                stock_signals[code] = {
                    "name": name,
                    "model_score": score,
                    "sentiment_score": sentiment_scores.get(code, 0.5),
                    "technical_score": technical_scores.get(code, 0.5),
                    "strategy_scores": strategy_scores,
                    "factors": self.fetch_stock_factors(code),
                }
        
        combined_signals = []
        
        for code, data in stock_signals.items():
            strategy_scores = data["strategy_scores"]
            
            weighted_score = 0.0
            total_weight = 0.0
            
            for strategy, strat_score in strategy_scores.items():
                weight = self._strategy_weights.get(strategy, 0.25)
                min_score = self.STRATEGY_DEFINITIONS.get(strategy, {}).get("min_score", 0.5)
                
                if strat_score >= min_score:
                    weighted_score += strat_score * weight
                    total_weight += weight
            
            if total_weight > 0:
                weighted_score /= total_weight
            
            total_score = (
                data["model_score"] * 0.35 +
                data["sentiment_score"] * 0.20 +
                data["technical_score"] * 0.15 +
                weighted_score * 0.30
            )
            
            signal_type = self._determine_signal_type(total_score, data)
            confidence = self._calculate_confidence(data)
            reasons = self._generate_reasons(data, strategy_scores)
            
            combined = CombinedSignal(
                code=code,
                name=data["name"],
                total_score=total_score,
                weighted_score=weighted_score,
                strategy_scores=strategy_scores,
                strategy_weights=self._strategy_weights,
                signal_type=signal_type,
                confidence=confidence,
                reasons=reasons,
            )
            combined_signals.append(combined)
        
        combined_signals.sort(key=lambda x: x.total_score, reverse=True)
        
        return combined_signals
    
    def _determine_signal_type(self, total_score: float, data: Dict[str, Any]) -> str:
        if total_score >= 0.75:
            return "strong_buy"
        elif total_score >= 0.65:
            return "buy"
        elif total_score >= 0.55:
            return "hold"
        elif total_score >= 0.45:
            return "reduce"
        else:
            return "sell"
    
    def _calculate_confidence(self, data: Dict[str, Any]) -> float:
        strategy_scores = data.get("strategy_scores", {})
        strategy_count = len([s for s in strategy_scores.values() if s >= 0.5])
        
        score_variance = 0.0
        if len(strategy_scores) > 1:
            scores = list(strategy_scores.values())
            mean_score = sum(scores) / len(scores)
            score_variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        
        strategy_confidence = min(1.0, strategy_count / 4.0)
        consistency_confidence = 1.0 - min(score_variance * 4, 1.0)
        
        confidence = strategy_confidence * 0.6 + consistency_confidence * 0.4
        return confidence
    
    def _generate_reasons(self, data: Dict[str, Any], strategy_scores: Dict[str, float]) -> List[str]:
        reasons = []
        factors = data.get("factors", {})
        
        if data["model_score"] >= 0.7:
            reasons.append(f"模型预测分数较高({data['model_score']:.2f})")
        elif data["model_score"] <= 0.3:
            reasons.append(f"模型预测分数较低({data['model_score']:.2f})")
        
        pb = factors.get("pb", 1.0)
        pe = factors.get("pe", 15.0)
        if pb < 1.0 and pe < 15:
            reasons.append(f"低估值(PB:{pb:.1f}, PE:{pe:.0f})")
        
        roe = factors.get("roe", 0.10)
        if roe > 0.15:
            reasons.append(f"高ROE({roe:.1%})")
        
        mom_20d = factors.get("price_momentum_20d", 0.0)
        if mom_20d > 0.10:
            reasons.append(f"动量强劲(20日+{mom_20d:.1%})")
        elif mom_20d < -0.10:
            reasons.append(f"动量疲弱(20日{mom_20d:.1%})")
        
        for strategy, score in strategy_scores.items():
            strategy_def = self.STRATEGY_DEFINITIONS.get(strategy, {})
            strategy_name = strategy_def.get("name", strategy)
            if score >= 0.7:
                reasons.append(f"{strategy_name}信号强烈({score:.2f})")
        
        return reasons if reasons else ["综合评分一般"]
    
    def get_top_signals(
        self, 
        signals: List[CombinedSignal], 
        top_k: int = 20,
        min_score: float = 0.5
    ) -> List[CombinedSignal]:
        filtered = [s for s in signals if s.total_score >= min_score]
        return filtered[:top_k]
    
    def filter_by_signal_type(
        self, 
        signals: List[CombinedSignal], 
        signal_types: List[str]
    ) -> List[CombinedSignal]:
        return [s for s in signals if s.signal_type in signal_types]
    
    def get_strategy_summary(self, signals: List[CombinedSignal]) -> Dict[str, Any]:
        summary = {
            "total_signals": len(signals),
            "by_type": {},
            "by_strategy": {},
            "avg_scores": {},
        }
        
        for signal in signals:
            st = signal.signal_type
            summary["by_type"][st] = summary["by_type"].get(st, 0) + 1
            
            for strategy in signal.strategy_scores:
                summary["by_strategy"][strategy] = summary["by_strategy"].get(strategy, 0) + 1
        
        if signals:
            summary["avg_scores"]["total"] = sum(s.total_score for s in signals) / len(signals)
            summary["avg_scores"]["confidence"] = sum(s.confidence for s in signals) / len(signals)
        
        return summary
