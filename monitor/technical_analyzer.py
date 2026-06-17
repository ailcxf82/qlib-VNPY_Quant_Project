from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from monitor.ts_client import get_ts_client

logger = logging.getLogger(__name__)


@dataclass
class TechnicalIndicators:
    code: str
    date: str
    
    ma5: float = 0.0
    ma10: float = 0.0
    ma20: float = 0.0
    ma60: float = 0.0
    
    rsi_14: float = 50.0
    
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_hist: float = 0.0
    
    close: float = 0.0
    high: float = 0.0
    low: float = 0.0
    volume: float = 0.0
    volume_ma5: float = 0.0
    
    support_level: float = 0.0
    resistance_level: float = 0.0
    
    trend: str = "neutral"
    signal: str = "hold"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "date": self.date,
            "ma5": self.ma5,
            "ma10": self.ma10,
            "ma20": self.ma20,
            "ma60": self.ma60,
            "rsi_14": self.rsi_14,
            "macd": self.macd,
            "macd_signal": self.macd_signal,
            "macd_hist": self.macd_hist,
            "close": self.close,
            "high": self.high,
            "low": self.low,
            "volume": self.volume,
            "volume_ma5": self.volume_ma5,
            "support_level": self.support_level,
            "resistance_level": self.resistance_level,
            "trend": self.trend,
            "signal": self.signal,
        }


@dataclass
class TechnicalScore:
    code: str
    date: str
    ma_score: float = 0.5
    rsi_score: float = 0.5
    macd_score: float = 0.5
    volume_score: float = 0.5
    trend_score: float = 0.5
    total_score: float = 0.5
    breakdown_alert: bool = False
    breakout_alert: bool = False
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "date": self.date,
            "ma_score": self.ma_score,
            "rsi_score": self.rsi_score,
            "macd_score": self.macd_score,
            "volume_score": self.volume_score,
            "trend_score": self.trend_score,
            "total_score": self.total_score,
            "breakdown_alert": self.breakdown_alert,
            "breakout_alert": self.breakout_alert,
            "details": self.details,
        }


class TechnicalAnalyzer:
    def __init__(self, config_path: str = "config/monitor.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.tech_config = self.config.get("technical", {})
        
        self.lookback_days = self.tech_config.get("lookback_days", 120)
        self.support_threshold = self.tech_config.get("support_threshold", 0.03)
        self.breakdown_confirm_days = self.tech_config.get("breakdown_confirm_days", 2)
        
        self._cache: Dict[str, TechnicalIndicators] = {}
    
    def _load_config(self) -> Dict[str, Any]:
        p = Path(self.config_path)
        if not p.exists():
            return {}
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    
    def _get_ts_client(self):
        return get_ts_client()
    
    def fetch_kline_data(self, code: str, days: int = 120) -> pd.DataFrame:
        ts = self._get_ts_client()
        if ts is None:
            return pd.DataFrame()
        
        try:
            code_clean = str(code).replace(".XSHE", "").replace(".XSHG", "").zfill(6)
            suffix = ".SZ" if code_clean.startswith(("0", "3")) else ".SH"
            ts_code = code_clean + suffix
            
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=days + 30)).strftime("%Y%m%d")
            
            df = ts._pro.daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )
            
            if df.empty:
                return pd.DataFrame()
            
            df = df.sort_values("trade_date").reset_index(drop=True)
            df = df.tail(days)
            
            return df
        except Exception as e:
            logger.warning(f"获取K线数据失败: code={code}, error={e}")
            return pd.DataFrame()
    
    def calculate_ma(self, df: pd.DataFrame) -> Dict[str, float]:
        if df.empty or len(df) < 5:
            return {"ma5": 0, "ma10": 0, "ma20": 0, "ma60": 0}
        
        close = df["close"]
        
        ma5 = close.rolling(5).mean().iloc[-1] if len(close) >= 5 else 0
        ma10 = close.rolling(10).mean().iloc[-1] if len(close) >= 10 else 0
        ma20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else 0
        ma60 = close.rolling(60).mean().iloc[-1] if len(close) >= 60 else 0
        
        return {
            "ma5": float(ma5) if not pd.isna(ma5) else 0,
            "ma10": float(ma10) if not pd.isna(ma10) else 0,
            "ma20": float(ma20) if not pd.isna(ma20) else 0,
            "ma60": float(ma60) if not pd.isna(ma60) else 0,
        }
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        if df.empty or len(df) < period + 1:
            return 50.0
        
        close = df["close"].astype(float)
        delta = close.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        
        last_rsi = rsi.iloc[-1]
        return float(last_rsi) if not pd.isna(last_rsi) else 50.0
    
    def calculate_macd(self, df: pd.DataFrame) -> Dict[str, float]:
        if df.empty or len(df) < 35:
            return {"macd": 0, "signal": 0, "hist": 0}
        
        close = df["close"].astype(float)
        
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        
        return {
            "macd": float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else 0,
            "signal": float(signal.iloc[-1]) if not pd.isna(signal.iloc[-1]) else 0,
            "hist": float(hist.iloc[-1]) if not pd.isna(hist.iloc[-1]) else 0,
        }
    
    def calculate_support_resistance(self, df: pd.DataFrame) -> Dict[str, float]:
        if df.empty or len(df) < 20:
            return {"support": 0, "resistance": 0}
        
        recent = df.tail(20)
        
        lows = recent["low"].astype(float)
        highs = recent["high"].astype(float)
        
        support = lows.min()
        resistance = highs.max()
        
        return {
            "support": float(support),
            "resistance": float(resistance),
        }
    
    def detect_trend(self, df: pd.DataFrame, ma_data: Dict[str, float]) -> str:
        if df.empty:
            return "neutral"
        
        close = float(df["close"].iloc[-1])
        ma5 = ma_data.get("ma5", 0)
        ma10 = ma_data.get("ma10", 0)
        ma20 = ma_data.get("ma20", 0)
        
        if ma5 == 0 or ma10 == 0 or ma20 == 0:
            return "neutral"
        
        if close > ma5 > ma10 > ma20:
            return "strong_up"
        elif close > ma5 > ma10:
            return "up"
        elif close < ma5 < ma10 < ma20:
            return "strong_down"
        elif close < ma5 < ma10:
            return "down"
        else:
            return "neutral"
    
    def detect_breakdown(self, df: pd.DataFrame, support: float, close: float) -> bool:
        if df.empty or support == 0:
            return False
        
        breakdown_threshold = support * (1 - self.support_threshold)
        
        if close < breakdown_threshold:
            recent_lows = df["low"].tail(self.breakdown_confirm_days)
            if all(low < breakdown_threshold for low in recent_lows):
                return True
        
        return False
    
    def analyze(self, code: str) -> TechnicalIndicators:
        if code in self._cache:
            cached = self._cache[code]
            cache_date = cached.date
            if cache_date == datetime.now().strftime("%Y-%m-%d"):
                return cached
        
        df = self.fetch_kline_data(code, self.lookback_days)
        
        if df.empty:
            return TechnicalIndicators(code=code, date=datetime.now().strftime("%Y-%m-%d"))
        
        ma_data = self.calculate_ma(df)
        rsi = self.calculate_rsi(df)
        macd_data = self.calculate_macd(df)
        sr_data = self.calculate_support_resistance(df)
        
        close = float(df["close"].iloc[-1])
        high = float(df["high"].iloc[-1])
        low = float(df["low"].iloc[-1])
        volume = float(df["vol"].iloc[-1]) if "vol" in df.columns else 0
        
        volume_ma5 = float(df["vol"].tail(5).mean()) if "vol" in df.columns and len(df) >= 5 else 0
        
        trend = self.detect_trend(df, ma_data)
        
        is_breakdown = self.detect_breakdown(df, sr_data["support"], close)
        
        signal = "hold"
        if is_breakdown:
            signal = "sell"
        elif trend in ["up", "strong_up"] and macd_data["hist"] > 0:
            signal = "buy"
        elif trend in ["down", "strong_down"]:
            signal = "caution"
        
        indicators = TechnicalIndicators(
            code=code,
            date=datetime.now().strftime("%Y-%m-%d"),
            ma5=ma_data["ma5"],
            ma10=ma_data["ma10"],
            ma20=ma_data["ma20"],
            ma60=ma_data["ma60"],
            rsi_14=rsi,
            macd=macd_data["macd"],
            macd_signal=macd_data["signal"],
            macd_hist=macd_data["hist"],
            close=close,
            high=high,
            low=low,
            volume=volume,
            volume_ma5=volume_ma5,
            support_level=sr_data["support"],
            resistance_level=sr_data["resistance"],
            trend=trend,
            signal=signal,
        )
        
        self._cache[code] = indicators
        
        return indicators
    
    def calculate_score(self, indicators: TechnicalIndicators) -> TechnicalScore:
        ma_score = 0.5
        if indicators.ma5 > 0 and indicators.ma20 > 0:
            if indicators.close > indicators.ma5 > indicators.ma20:
                ma_score = 0.8
            elif indicators.close > indicators.ma5:
                ma_score = 0.65
            elif indicators.close < indicators.ma5 < indicators.ma20:
                ma_score = 0.2
            elif indicators.close < indicators.ma5:
                ma_score = 0.35
        
        rsi_score = 0.5
        if indicators.rsi_14 > 70:
            rsi_score = 0.25
        elif indicators.rsi_14 > 60:
            rsi_score = 0.65
        elif indicators.rsi_14 < 30:
            rsi_score = 0.75
        elif indicators.rsi_14 < 40:
            rsi_score = 0.6
        
        macd_score = 0.5
        if indicators.macd_hist > 0:
            macd_score = 0.7
            if indicators.macd > indicators.macd_signal:
                macd_score = 0.8
        elif indicators.macd_hist < 0:
            macd_score = 0.3
            if indicators.macd < indicators.macd_signal:
                macd_score = 0.2
        
        volume_score = 0.5
        if indicators.volume_ma5 > 0:
            volume_ratio = indicators.volume / indicators.volume_ma5
            if volume_ratio > 2.0:
                if indicators.close > indicators.ma5:
                    volume_score = 0.75
                else:
                    volume_score = 0.25
            elif volume_ratio > 1.5:
                volume_score = 0.6 if indicators.close > indicators.ma5 else 0.4
        
        trend_score = 0.5
        if indicators.trend == "strong_up":
            trend_score = 0.9
        elif indicators.trend == "up":
            trend_score = 0.7
        elif indicators.trend == "strong_down":
            trend_score = 0.1
        elif indicators.trend == "down":
            trend_score = 0.3
        
        total_score = (
            ma_score * 0.3 +
            rsi_score * 0.25 +
            macd_score * 0.25 +
            volume_score * 0.2
        )
        
        breakdown_alert = False
        breakout_alert = False
        
        if indicators.support_level > 0:
            breakdown_threshold = indicators.support_level * 0.97
            if indicators.close < breakdown_threshold:
                breakdown_alert = True
        
        if indicators.resistance_level > 0:
            breakout_threshold = indicators.resistance_level * 0.98
            if indicators.close > breakout_threshold and indicators.volume > indicators.volume_ma5 * 1.5:
                breakout_alert = True
        
        return TechnicalScore(
            code=indicators.code,
            date=indicators.date,
            ma_score=ma_score,
            rsi_score=rsi_score,
            macd_score=macd_score,
            volume_score=volume_score,
            trend_score=trend_score,
            total_score=total_score,
            breakdown_alert=breakdown_alert,
            breakout_alert=breakout_alert,
            details={
                "trend": indicators.trend,
                "signal": indicators.signal,
                "close": indicators.close,
                "support": indicators.support_level,
                "resistance": indicators.resistance_level,
            },
        )
    
    def analyze_multiple(self, codes: List[str]) -> Dict[str, TechnicalScore]:
        results = {}
        for code in codes:
            try:
                indicators = self.analyze(code)
                score = self.calculate_score(indicators)
                results[code] = score
            except Exception as e:
                logger.warning(f"分析 {code} 技术面失败: {e}")
        return results
    
    def confirm_extreme_condition(
        self, 
        code: str, 
        sentiment_type: str
    ) -> Dict[str, Any]:
        indicators = self.analyze(code)
        score = self.calculate_score(indicators)
        
        confirmed = False
        reason = ""
        
        if sentiment_type == "extreme_negative":
            if score.breakdown_alert:
                confirmed = True
                reason = "跌破支撑位确认"
            elif indicators.trend in ["down", "strong_down"]:
                confirmed = True
                reason = "下跌趋势确认"
            elif indicators.macd_hist < 0 and indicators.macd < indicators.macd_signal:
                confirmed = True
                reason = "MACD死叉确认"
            elif indicators.volume > indicators.volume_ma5 * 1.5 and indicators.close < indicators.ma5:
                confirmed = True
                reason = "放量下跌确认"
        
        elif sentiment_type == "extreme_positive":
            if score.breakout_alert:
                confirmed = True
                reason = "突破阻力位确认"
            elif indicators.trend in ["up", "strong_up"]:
                confirmed = True
                reason = "上涨趋势确认"
            elif indicators.macd_hist > 0 and indicators.macd > indicators.macd_signal:
                confirmed = True
                reason = "MACD金叉确认"
        
        return {
            "code": code,
            "sentiment_type": sentiment_type,
            "confirmed": confirmed,
            "reason": reason,
            "technical_score": score.total_score,
            "trend": indicators.trend,
            "breakdown_alert": score.breakdown_alert,
            "breakout_alert": score.breakout_alert,
            "details": score.to_dict(),
        }
