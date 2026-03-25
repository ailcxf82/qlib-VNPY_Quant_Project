from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    code: str
    name: str
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float
    turnover_rate: Optional[float] = None
    pct_change: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "date": self.date,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "amount": self.amount,
            "turnover_rate": self.turnover_rate,
            "pct_change": self.pct_change,
        }


@dataclass
class FinancialData:
    code: str
    name: str
    report_date: str
    roe: float
    roa: float
    pe: float
    pb: float
    debt_ratio: float
    net_profit_growth: float
    revenue_growth: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "report_date": self.report_date,
            "roe": self.roe,
            "roa": self.roa,
            "pe": self.pe,
            "pb": self.pb,
            "debt_ratio": self.debt_ratio,
            "net_profit_growth": self.net_profit_growth,
            "revenue_growth": self.revenue_growth,
        }


class DataSourceAdapter(ABC):
    @abstractmethod
    def get_daily_data(
        self, 
        code: str, 
        start_date: str, 
        end_date: str
    ) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def get_stock_list(self) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def get_financial_data(self, code: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def get_index_data(
        self, 
        index_code: str, 
        start_date: str, 
        end_date: str
    ) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def get_stock_factors(self, code: str) -> Dict[str, float]:
        pass
    
    @abstractmethod
    def get_market_breadth(self, date: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def get_all_daily_data(self, date: str) -> pd.DataFrame:
        pass


class AkshareAdapter(DataSourceAdapter):
    def __init__(self, config_path: str = "config/monitor.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.akshare_config = self.config.get("akshare", {})
        
        self._cache_dir = Path(self.config.get("paths", {}).get("data_dir", "data/monitor")) / "akshare_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._akshare = None
        self._cache: Dict[str, pd.DataFrame] = {}
        self._dict_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_expiry: Dict[str, datetime] = {}
        self._cache_hours = self.akshare_config.get("cache_hours", 4)
    
    def _load_config(self) -> Dict[str, Any]:
        p = Path(self.config_path)
        if not p.exists():
            return {}
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    
    @property
    def akshare(self):
        if self._akshare is None:
            try:
                import akshare as ak
                self._akshare = ak
                logger.info("Akshare initialized successfully")
            except ImportError:
                logger.error("Akshare not installed. Run: pip install akshare")
                raise ImportError("Akshare not installed. Run: pip install akshare")
        return self._akshare
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        if cache_key not in self._cache_expiry:
            return False
        return datetime.now() < self._cache_expiry[cache_key]
    
    def _set_cache(self, key: str, data: pd.DataFrame):
        self._cache[key] = data
        self._cache_expiry[key] = datetime.now() + timedelta(hours=self._cache_hours)
    
    def _get_cache(self, key: str) -> Optional[pd.DataFrame]:
        if self._is_cache_valid(key):
            return self._cache.get(key)
        return None
    
    def _set_dict_cache(self, key: str, data: Dict[str, Any]):
        self._dict_cache[key] = data
        self._cache_expiry[key] = datetime.now() + timedelta(hours=self._cache_hours)
    
    def _get_dict_cache(self, key: str) -> Optional[Dict[str, Any]]:
        if self._is_cache_valid(key):
            return self._dict_cache.get(key)
        return None
    
    def _normalize_code(self, code: str) -> str:
        code = str(code).replace(".XSHE", "").replace(".XSHG", "").zfill(6)
        return code
    
    def _get_akshare_code(self, code: str) -> str:
        code = self._normalize_code(code)
        if code.startswith("6"):
            return f"sh{code}"
        else:
            return f"sz{code}"
    
    def get_daily_data(
        self, 
        code: str, 
        start_date: str, 
        end_date: str
    ) -> pd.DataFrame:
        cache_key = f"daily_{code}_{start_date}_{end_date}"
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cached
        
        try:
            ak_code = self._get_akshare_code(code)
            
            df = self.akshare.stock_zh_a_hist(
                symbol=self._normalize_code(code),
                period="daily",
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                adjust="qfq"
            )
            
            if df.empty:
                logger.warning(f"No data found for {code}")
                return pd.DataFrame()
            
            df = df.rename(columns={
                "日期": "date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
                "换手率": "turnover_rate",
                "涨跌幅": "pct_change",
            })
            
            df["code"] = self._normalize_code(code)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            
            for col in ["open", "high", "low", "close", "volume", "amount"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            
            self._set_cache(cache_key, df)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get daily data for {code}: {e}")
            return pd.DataFrame()
    
    def get_stock_list(self) -> pd.DataFrame:
        cache_key = "stock_list"
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cached
        
        try:
            df = self.akshare.stock_zh_a_spot_em()
            
            df = df.rename(columns={
                "代码": "code",
                "名称": "name",
                "最新价": "price",
                "涨跌幅": "pct_change",
                "涨跌额": "change",
                "成交量": "volume",
                "成交额": "amount",
                "市盈率-动态": "pe",
                "市净率": "pb",
                "总市值": "market_cap",
                "流通市值": "float_market_cap",
            })
            
            self._set_cache(cache_key, df)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get stock list: {e}")
            return pd.DataFrame()
    
    def get_financial_data(self, code: str) -> Dict[str, Any]:
        cache_key = f"financial_{code}"
        
        try:
            code = self._normalize_code(code)
            
            result = {
                "code": code,
                "roe": 0.0,
                "roa": 0.0,
                "pe": 0.0,
                "pb": 0.0,
                "debt_ratio": 0.0,
                "net_profit_growth": 0.0,
                "revenue_growth": 0.0,
            }
            
            try:
                df = self.akshare.stock_financial_analysis_indicator(symbol=code)
                if not df.empty:
                    latest = df.iloc[0]
                    result["roe"] = float(latest.get("净资产收益率(%)", 0) or 0)
                    result["roa"] = float(latest.get("总资产净利率(%)", 0) or 0)
            except Exception:
                pass
            
            try:
                df = self.akshare.stock_a_lg_indicator(symbol=code)
                if not df.empty:
                    latest = df.iloc[0]
                    result["pe"] = float(latest.get("市盈率", 0) or 0)
                    result["pb"] = float(latest.get("市净率", 0) or 0)
            except Exception:
                pass
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get financial data for {code}: {e}")
            return {"code": code}
    
    def get_index_data(
        self, 
        index_code: str, 
        start_date: str, 
        end_date: str
    ) -> pd.DataFrame:
        cache_key = f"index_{index_code}_{start_date}_{end_date}"
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cached
        
        try:
            index_map = {
                "000001.SH": "sh000001",
                "000300.SH": "sh000300",
                "399001.SZ": "sz399001",
                "399006.SZ": "sz399006",
            }
            
            symbol = index_map.get(index_code, index_code.lower())
            
            df = self.akshare.stock_zh_index_daily(symbol=symbol)
            
            if df.empty:
                return pd.DataFrame()
            
            df = df.rename(columns={
                "date": "date",
                "open": "open",
                "close": "close",
                "high": "high",
                "low": "low",
                "volume": "volume",
            })
            
            df["date"] = pd.to_datetime(df["date"])
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            df = df[(df["date"] >= start) & (df["date"] <= end)]
            df = df.sort_values("date").reset_index(drop=True)
            
            self._set_cache(cache_key, df)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get index data for {index_code}: {e}")
            return pd.DataFrame()
    
    def get_realtime_quotes(self, codes: List[str]) -> pd.DataFrame:
        try:
            all_stocks = self.get_stock_list()
            codes_normalized = [self._normalize_code(c) for c in codes]
            return all_stocks[all_stocks["code"].isin(codes_normalized)]
        except Exception as e:
            logger.error(f"Failed to get realtime quotes: {e}")
            return pd.DataFrame()
    
    def get_trading_calendar(self, start_date: str, end_date: str) -> List[str]:
        try:
            df = self.akshare.tool_trade_date_hist_sina()
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            dates = df[(df["trade_date"] >= start) & (df["trade_date"] <= end)]
            return [d.strftime("%Y-%m-%d") for d in dates["trade_date"]]
        except Exception as e:
            logger.error(f"Failed to get trading calendar: {e}")
            return []
    
    def get_stock_factors(self, code: str) -> Dict[str, float]:
        cache_key = f"factors_{code}"
        cached = self._get_dict_cache(cache_key)
        if cached is not None:
            return cached
        
        code = self._normalize_code(code)
        
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
        
        try:
            financial = self.get_financial_data(code)
            factors["pb"] = financial.get("pb", 1.0)
            factors["pe"] = financial.get("pe", 15.0)
            factors["roe"] = financial.get("roe", 0.10) / 100 if financial.get("roe", 0) > 1 else financial.get("roe", 0.10)
            factors["roa"] = financial.get("roa", 0.05) / 100 if financial.get("roa", 0) > 1 else financial.get("roa", 0.05)
            factors["debt_ratio"] = financial.get("debt_ratio", 0.40)
        except Exception:
            pass
        
        try:
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=60)).strftime("%Y%m%d")
            
            df = self.akshare.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                adjust="qfq"
            )
            
            if not df.empty and len(df) >= 20:
                df = df.rename(columns={
                    "日期": "date",
                    "收盘": "close",
                    "成交量": "volume",
                })
                df = df.sort_values("date").reset_index(drop=True)
                
                close = pd.to_numeric(df["close"], errors="coerce")
                vol = pd.to_numeric(df["volume"], errors="coerce")
                
                if len(close) >= 5:
                    factors["price_momentum_5d"] = float((close.iloc[-1] / close.iloc[-5] - 1))
                
                if len(close) >= 20:
                    factors["price_momentum_20d"] = float((close.iloc[-1] / close.iloc[-20] - 1))
                    
                    vol_ma5 = vol.rolling(5).mean().iloc[-1]
                    factors["volume_momentum"] = float(vol.iloc[-1] / vol_ma5) if vol_ma5 > 0 else 1.0
                    
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
                    factors["price_deviation_ma20"] = float((close.iloc[-1] - ma20) / ma20) if ma20 > 0 else 0
                    
                    vol_ma20 = vol.rolling(20).mean().iloc[-1]
                    factors["volume_spike"] = float(vol.iloc[-1] / vol_ma20) if vol_ma20 > 0 else 1.0
        except Exception as e:
            logger.debug(f"Failed to get price factors for {code}: {e}")
        
        self._set_dict_cache(cache_key, factors)
        return factors
    
    def get_market_breadth(self, date: str) -> Dict[str, Any]:
        try:
            df = self.akshare.stock_zh_a_spot_em()
            
            if df.empty:
                return {"up_count": 0, "down_count": 0, "breadth": 0.5}
            
            pct_col = None
            for col in ["涨跌幅", "pct_change", "pct_chg"]:
                if col in df.columns:
                    pct_col = col
                    break
            
            if pct_col is None:
                return {"up_count": 0, "down_count": 0, "breadth": 0.5}
            
            pct = pd.to_numeric(df[pct_col], errors="coerce")
            up_count = int((pct > 0).sum())
            down_count = int((pct < 0).sum())
            total = up_count + down_count
            
            breadth = up_count / total if total > 0 else 0.5
            
            return {
                "up_count": up_count,
                "down_count": down_count,
                "breadth": breadth,
                "total": len(df),
            }
        except Exception as e:
            logger.error(f"Failed to get market breadth: {e}")
            return {"up_count": 0, "down_count": 0, "breadth": 0.5}
    
    def get_all_daily_data(self, date: str) -> pd.DataFrame:
        try:
            df = self.akshare.stock_zh_a_spot_em()
            
            if df.empty:
                return pd.DataFrame()
            
            rename_map = {
                "代码": "code",
                "名称": "name",
                "最新价": "close",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
                "涨跌幅": "pct_change",
                "市盈率-动态": "pe",
                "市净率": "pb",
            }
            
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
            df["date"] = date
            
            return df
        except Exception as e:
            logger.error(f"Failed to get all daily data: {e}")
            return pd.DataFrame()


class DataSourceFactory:
    _adapters: Dict[str, DataSourceAdapter] = {}
    
    @classmethod
    def get_adapter(
        cls, 
        source: str = "akshare", 
        config_path: str = "config/monitor.yaml"
    ) -> DataSourceAdapter:
        if source not in cls._adapters:
            if source == "akshare":
                cls._adapters[source] = AkshareAdapter(config_path)
            else:
                raise ValueError(f"Unknown data source: {source}")
        return cls._adapters[source]
    
    @classmethod
    def register_adapter(cls, name: str, adapter: DataSourceAdapter):
        cls._adapters[name] = adapter


def get_data_source(source: str = "akshare") -> DataSourceAdapter:
    return DataSourceFactory.get_adapter(source)
