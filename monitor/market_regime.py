from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
import yaml

if TYPE_CHECKING:
    from monitor.data_source import DataSourceAdapter

logger = logging.getLogger(__name__)


@dataclass
class MarketRegime:
    date: str
    regime: str
    trend_strength: float
    volatility: float
    breadth: float
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "regime": self.regime,
            "trend_strength": self.trend_strength,
            "volatility": self.volatility,
            "breadth": self.breadth,
            "confidence": self.confidence,
            "details": self.details,
        }


class MarketRegimeDetector:
    BULL_THRESHOLD = 0.6
    BEAR_THRESHOLD = 0.4
    HIGH_VOL_THRESHOLD = 0.25
    LOW_VOL_THRESHOLD = 0.12
    
    DEFAULT_REGIME = MarketRegime(
        date="",
        regime="range_bound",
        trend_strength=0.5,
        volatility=0.15,
        breadth=0.5,
        confidence=0.3,
        details={"source": "default", "reason": "数据不可用，使用默认值"}
    )
    
    def __init__(
        self, 
        config_path: str = "config/monitor.yaml",
        data_source: Optional["DataSourceAdapter"] = None,
        data_source_type: str = "auto"
    ):
        self.config_path = config_path
        self.config = self._load_config()
        self.regime_config = self.config.get("market_regime", {})
        
        self.lookback_days = self.regime_config.get("lookback_days", 60)
        self._cache: Optional[MarketRegime] = None
        self._cache_date: str = ""
        self._fallback_mode: bool = False
        self._last_error_time: Optional[datetime] = None
        self._error_count: int = 0
        
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
                logger.info("Using Akshare data source for MarketRegimeDetector")
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
    
    def _should_use_fallback(self) -> bool:
        if self._error_count >= 3:
            if self._last_error_time:
                minutes_since_error = (datetime.now() - self._last_error_time).total_seconds() / 60
                if minutes_since_error < 30:
                    return True
                else:
                    self._error_count = 0
                    self._fallback_mode = False
        return self._fallback_mode
    
    def _record_error(self):
        self._error_count += 1
        self._last_error_time = datetime.now()
        if self._error_count >= 3:
            self._fallback_mode = True
            logger.warning("数据源连续失败，启用降级模式")
    
    def _load_cached_regime(self) -> Optional[MarketRegime]:
        cache_file = Path(self.config.get("paths", {}).get("data_dir", "data/monitor")) / "regime_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return MarketRegime(
                    date=data.get("date", ""),
                    regime=data.get("regime", "range_bound"),
                    trend_strength=data.get("trend_strength", 0.5),
                    volatility=data.get("volatility", 0.15),
                    breadth=data.get("breadth", 0.5),
                    confidence=data.get("confidence", 0.3) * 0.8,
                    details={**data.get("details", {}), "source": "cache"}
                )
            except Exception as e:
                logger.warning(f"加载缓存市场环境失败: {e}")
        return None
    
    def _save_regime_cache(self, regime: MarketRegime):
        cache_file = Path(self.config.get("paths", {}).get("data_dir", "data/monitor")) / "regime_cache.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(regime.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存市场环境缓存失败: {e}")
    
    def fetch_index_data(self, index_code: str = "000001.SH", days: int = 120) -> pd.DataFrame:
        if self.data_source is not None:
            try:
                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=days + 30)).strftime("%Y-%m-%d")
                
                df = self.data_source.get_index_data(index_code, start_date, end_date)
                
                if not df.empty:
                    df = df.tail(days)
                    logger.debug(f"Got index data from Akshare for {index_code}")
                    return df
            except Exception as e:
                logger.debug(f"Akshare index data fetch failed: {e}")
        
        ts = self._get_ts_client()
        if ts is None:
            return pd.DataFrame()
        
        try:
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=days + 30)).strftime("%Y%m%d")
            
            df = ts._pro.index_daily(
                ts_code=index_code,
                start_date=start_date,
                end_date=end_date
            )
            
            if df.empty:
                return pd.DataFrame()
            
            df = df.sort_values("trade_date").reset_index(drop=True)
            return df.tail(days)
        except Exception as e:
            logger.warning(f"获取指数数据失败: {e}")
            return pd.DataFrame()
    
    def calculate_trend_strength(self, df: pd.DataFrame) -> float:
        if df.empty or len(df) < 20:
            return 0.5
        
        close = df["close"].astype(float)
        
        ma5 = close.rolling(5).mean()
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean() if len(close) >= 60 else ma20
        
        last_close = close.iloc[-1]
        last_ma5 = ma5.iloc[-1]
        last_ma20 = ma20.iloc[-1]
        last_ma60 = ma60.iloc[-1]
        
        trend_score = 0.5
        
        if last_close > last_ma5 > last_ma20 > last_ma60:
            trend_score = 0.85
        elif last_close > last_ma5 > last_ma20:
            trend_score = 0.70
        elif last_close > last_ma5:
            trend_score = 0.60
        elif last_close < last_ma5 < last_ma20 < last_ma60:
            trend_score = 0.15
        elif last_close < last_ma5 < last_ma20:
            trend_score = 0.30
        elif last_close < last_ma5:
            trend_score = 0.40
        
        returns = close.pct_change().dropna()
        if len(returns) >= 20:
            momentum = (close.iloc[-1] / close.iloc[-20] - 1)
            if momentum > 0.05:
                trend_score = min(1.0, trend_score + 0.1)
            elif momentum < -0.05:
                trend_score = max(0.0, trend_score - 0.1)
        
        return trend_score
    
    def calculate_volatility(self, df: pd.DataFrame) -> float:
        if df.empty or len(df) < 20:
            return 0.15
        
        close = df["close"].astype(float)
        returns = close.pct_change().dropna()
        
        if len(returns) < 20:
            return 0.15
        
        vol = returns.rolling(20).std().iloc[-1]
        annualized_vol = vol * np.sqrt(252)
        
        return float(annualized_vol) if not pd.isna(annualized_vol) else 0.15
    
    def calculate_market_breadth(self) -> float:
        if self.data_source is not None:
            try:
                breadth_data = self.data_source.get_market_breadth(datetime.now().strftime("%Y-%m-%d"))
                if breadth_data and "breadth" in breadth_data:
                    logger.debug("Got market breadth from Akshare")
                    return breadth_data["breadth"]
            except Exception as e:
                logger.debug(f"Akshare market breadth fetch failed: {e}")
        
        ts = self._get_ts_client()
        if ts is None:
            return 0.5
        
        try:
            today = datetime.now().strftime("%Y%m%d")
            df = ts._pro.daily(trade_date=today)
            
            if df.empty:
                return 0.5
            
            up_count = len(df[df["pct_chg"] > 0])
            down_count = len(df[df["pct_chg"] < 0])
            total = up_count + down_count
            
            if total == 0:
                return 0.5
            
            breadth = up_count / total
            return float(breadth)
        except Exception as e:
            logger.warning(f"计算市场宽度失败: {e}")
            return 0.5
    
    def detect_regime(self, force_refresh: bool = False) -> MarketRegime:
        today = datetime.now().strftime("%Y-%m-%d")
        
        if not force_refresh and self._cache_date == today and self._cache:
            return self._cache
        
        if self._should_use_fallback():
            logger.info("使用降级模式获取市场环境")
            cached = self._load_cached_regime()
            if cached:
                return cached
            default = MarketRegime(
                date=today,
                regime=self.DEFAULT_REGIME.regime,
                trend_strength=self.DEFAULT_REGIME.trend_strength,
                volatility=self.DEFAULT_REGIME.volatility,
                breadth=self.DEFAULT_REGIME.breadth,
                confidence=0.2,
                details={"source": "fallback", "reason": "数据源不可用"}
            )
            return default
        
        try:
            df = self.fetch_index_data("000001.SH", self.lookback_days + 60)
            
            if df.empty:
                self._record_error()
                cached = self._load_cached_regime()
                if cached:
                    return cached
                return self._get_default_regime(today)
            
            trend_strength = self.calculate_trend_strength(df)
            volatility = self.calculate_volatility(df)
            breadth = self.calculate_market_breadth()
            
            regime = self._classify_regime(trend_strength, volatility, breadth)
            confidence = self._calculate_confidence(trend_strength, volatility, breadth)
            
            regime_result = MarketRegime(
                date=today,
                regime=regime,
                trend_strength=trend_strength,
                volatility=volatility,
                breadth=breadth,
                confidence=confidence,
                details={
                    "ma_alignment": self._get_ma_alignment(df),
                    "momentum_5d": self._get_momentum(df, 5),
                    "momentum_20d": self._get_momentum(df, 20),
                    "vol_level": self._get_vol_level(volatility),
                    "source": "live"
                }
            )
            
            self._cache = regime_result
            self._cache_date = today
            self._error_count = 0
            self._fallback_mode = False
            
            self._save_regime_cache(regime_result)
            
            return regime_result
            
        except Exception as e:
            logger.error(f"检测市场环境失败: {e}")
            self._record_error()
            
            cached = self._load_cached_regime()
            if cached:
                return cached
            
            return self._get_default_regime(today)
    
    def _get_default_regime(self, date: str) -> MarketRegime:
        return MarketRegime(
            date=date,
            regime=self.DEFAULT_REGIME.regime,
            trend_strength=self.DEFAULT_REGIME.trend_strength,
            volatility=self.DEFAULT_REGIME.volatility,
            breadth=self.DEFAULT_REGIME.breadth,
            confidence=0.2,
            details={"source": "default", "reason": "数据获取失败，使用默认值"}
        )
    
    def _classify_regime(self, trend: float, vol: float, breadth: float) -> str:
        if trend >= self.BULL_THRESHOLD and breadth > 0.55:
            if vol < self.HIGH_VOL_THRESHOLD:
                return "bull_low_vol"
            else:
                return "bull_high_vol"
        elif trend <= self.BEAR_THRESHOLD or breadth < 0.45:
            if vol > self.HIGH_VOL_THRESHOLD:
                return "bear_high_vol"
            else:
                return "bear_low_vol"
        else:
            if vol > self.HIGH_VOL_THRESHOLD:
                return "choppy_high_vol"
            else:
                return "range_bound"
    
    def _calculate_confidence(self, trend: float, vol: float, breadth: float) -> float:
        trend_conf = abs(trend - 0.5) * 2
        breadth_conf = abs(breadth - 0.5) * 2
        vol_conf = 1.0 - min(vol / self.HIGH_VOL_THRESHOLD, 1.0)
        
        confidence = (trend_conf * 0.4 + breadth_conf * 0.35 + vol_conf * 0.25)
        return min(1.0, confidence)
    
    def _get_ma_alignment(self, df: pd.DataFrame) -> str:
        if df.empty or len(df) < 60:
            return "unknown"
        
        close = df["close"].astype(float)
        ma5 = close.rolling(5).mean().iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]
        ma60 = close.rolling(60).mean().iloc[-1]
        last_close = close.iloc[-1]
        
        if last_close > ma5 > ma20 > ma60:
            return "bullish_aligned"
        elif last_close < ma5 < ma20 < ma60:
            return "bearish_aligned"
        else:
            return "mixed"
    
    def _get_momentum(self, df: pd.DataFrame, period: int) -> float:
        if df.empty or len(df) < period:
            return 0.0
        
        close = df["close"].astype(float)
        momentum = (close.iloc[-1] / close.iloc[-period] - 1) * 100
        return float(momentum) if not pd.isna(momentum) else 0.0
    
    def _get_vol_level(self, vol: float) -> str:
        if vol < self.LOW_VOL_THRESHOLD:
            return "low"
        elif vol > self.HIGH_VOL_THRESHOLD:
            return "high"
        else:
            return "normal"
    
    def get_strategy_weights(self, regime: MarketRegime) -> Dict[str, float]:
        weights = {
            "value": 0.25,
            "momentum": 0.25,
            "mean_reversion": 0.25,
            "quality": 0.25,
        }
        
        if regime.regime == "bull_low_vol":
            weights = {"value": 0.20, "momentum": 0.40, "mean_reversion": 0.15, "quality": 0.25}
        elif regime.regime == "bull_high_vol":
            weights = {"value": 0.25, "momentum": 0.30, "mean_reversion": 0.15, "quality": 0.30}
        elif regime.regime == "bear_high_vol":
            weights = {"value": 0.30, "momentum": 0.10, "mean_reversion": 0.20, "quality": 0.40}
        elif regime.regime == "bear_low_vol":
            weights = {"value": 0.35, "momentum": 0.15, "mean_reversion": 0.25, "quality": 0.25}
        elif regime.regime == "choppy_high_vol":
            weights = {"value": 0.25, "momentum": 0.15, "mean_reversion": 0.35, "quality": 0.25}
        else:
            weights = {"value": 0.30, "momentum": 0.20, "mean_reversion": 0.30, "quality": 0.20}
        
        return weights
    
    def get_position_size_multiplier(self, regime: MarketRegime) -> float:
        if regime.regime in ["bull_low_vol", "bull_high_vol"]:
            return 1.0
        elif regime.regime in ["bear_high_vol"]:
            return 0.5
        elif regime.regime in ["bear_low_vol"]:
            return 0.7
        elif regime.regime in ["choppy_high_vol"]:
            return 0.6
        else:
            return 0.8
    
    def should_reduce_risk(self, regime: MarketRegime) -> Tuple[bool, str]:
        if regime.regime == "bear_high_vol":
            return True, "熊市高波动环境，建议降低仓位"
        elif regime.regime == "choppy_high_vol" and regime.confidence > 0.6:
            return True, "震荡高波动环境，建议谨慎操作"
        elif regime.trend_strength < 0.35 and regime.volatility > 0.20:
            return True, "趋势走弱且波动加大，建议减仓"
        
        return False, ""
