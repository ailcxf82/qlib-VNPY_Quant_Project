"""
融合数据支持模块 - Fusion Data Support

提供模型预测数据和策略信号数据的获取、转换和缓存功能。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .fusion_engine import (
    ModelPrediction,
    StrategySignalWrapper,
    MarketContext,
    MarketRegime,
)

logger = logging.getLogger(__name__)


@dataclass
class DataSourceConfig:
    model_output_dir: str = "data/model_outputs"
    strategy_signal_dir: str = "data/strategy_signals"
    market_data_dir: str = "data/market_data"
    cache_dir: str = "data/cache"
    use_cache: bool = True
    cache_expire_hours: int = 24


class ModelPredictionProvider:
    """模型预测数据提供者"""

    def __init__(self, config: Optional[DataSourceConfig] = None):
        self.config = config or DataSourceConfig()
        self._cache: Dict[str, List[ModelPrediction]] = {}

    def get_predictions(
        self,
        date: str,
        codes: Optional[List[str]] = None,
        model_type: str = "default",
    ) -> Dict[str, ModelPrediction]:
        cache_key = f"{date}_{model_type}"
        if cache_key in self._cache:
            predictions = self._cache[cache_key]
        else:
            predictions = self._load_predictions(date, model_type)
            self._cache[cache_key] = predictions
        
        result = {}
        for pred in predictions:
            if codes is None or pred.code in codes:
                result[pred.code] = pred
        
        return result

    def get_predictions_range(
        self,
        start_date: str,
        end_date: str,
        codes: Optional[List[str]] = None,
        model_type: str = "default",
    ) -> Dict[str, List[ModelPrediction]]:
        result = {}
        
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        current = start
        
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            daily_preds = self.get_predictions(date_str, codes, model_type)
            
            for code, pred in daily_preds.items():
                if code not in result:
                    result[code] = []
                result[code].append(pred)
            
            current += timedelta(days=1)
        
        return result

    def _load_predictions(
        self, date: str, model_type: str
    ) -> List[ModelPrediction]:
        pred_file = Path(self.config.model_output_dir) / model_type / f"{date}.csv"
        
        if not pred_file.exists():
            return self._generate_mock_predictions(date)
        
        try:
            df = pd.read_csv(pred_file)
            predictions = []
            
            for _, row in df.iterrows():
                pred = ModelPrediction(
                    code=str(row.get("code", "")),
                    name=str(row.get("name", "")),
                    score=float(row.get("score", 0.5)),
                    confidence=float(row.get("confidence", 0.5)),
                    std=float(row.get("std", 0.1)),
                    horizon=int(row.get("horizon", 5)),
                    timestamp=f"{date}T09:30:00",
                    metadata={
                        "model_type": model_type,
                        "features": row.get("features", ""),
                    },
                )
                predictions.append(pred)
            
            return predictions
        except Exception as e:
            logger.warning(f"加载模型预测失败 {pred_file}: {e}")
            return self._generate_mock_predictions(date)

    def _generate_mock_predictions(self, date: str) -> List[ModelPrediction]:
        mock_codes = [
            ("000001.SZ", "平安银行"),
            ("000002.SZ", "万科A"),
            ("600000.SH", "浦发银行"),
            ("600036.SH", "招商银行"),
            ("601318.SH", "中国平安"),
        ]
        
        np.random.seed(hash(date) % 2**32)
        
        predictions = []
        for code, name in mock_codes:
            score = np.random.uniform(0.3, 0.7)
            confidence = np.random.uniform(0.4, 0.9)
            
            pred = ModelPrediction(
                code=code,
                name=name,
                score=score,
                confidence=confidence,
                std=np.random.uniform(0.05, 0.2),
                horizon=np.random.choice([3, 5, 10, 20]),
                timestamp=f"{date}T09:30:00",
            )
            predictions.append(pred)
        
        return predictions


class StrategySignalProvider:
    """策略信号数据提供者"""

    def __init__(self, config: Optional[DataSourceConfig] = None):
        self.config = config or DataSourceConfig()
        self._cache: Dict[str, List[StrategySignalWrapper]] = {}

    def get_signals(
        self,
        date: str,
        codes: Optional[List[str]] = None,
        strategy_ids: Optional[List[str]] = None,
    ) -> Dict[str, StrategySignalWrapper]:
        cache_key = f"{date}_{strategy_ids}"
        if cache_key in self._cache:
            signals = self._cache[cache_key]
        else:
            signals = self._load_signals(date, strategy_ids)
            self._cache[cache_key] = signals
        
        result = {}
        for sig in signals:
            if codes is None or sig.code in codes:
                if sig.code in result:
                    if sig.strength > result[sig.code].strength:
                        result[sig.code] = sig
                else:
                    result[sig.code] = sig
        
        return result

    def get_signals_range(
        self,
        start_date: str,
        end_date: str,
        codes: Optional[List[str]] = None,
        strategy_ids: Optional[List[str]] = None,
    ) -> Dict[str, List[StrategySignalWrapper]]:
        result = {}
        
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        current = start
        
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            daily_signals = self.get_signals(date_str, codes, strategy_ids)
            
            for code, sig in daily_signals.items():
                if code not in result:
                    result[code] = []
                result[code].append(sig)
            
            current += timedelta(days=1)
        
        return result

    def _load_signals(
        self, date: str, strategy_ids: Optional[List[str]]
    ) -> List[StrategySignalWrapper]:
        signal_dir = Path(self.config.strategy_signal_dir)
        
        if not signal_dir.exists():
            return self._generate_mock_signals(date)
        
        signals = []
        
        try:
            for signal_file in signal_dir.glob(f"{date}_*.csv"):
                strategy_id = signal_file.stem.split("_")[1] if "_" in signal_file.stem else "unknown"
                
                if strategy_ids and strategy_id not in strategy_ids:
                    continue
                
                df = pd.read_csv(signal_file)
                
                for _, row in df.iterrows():
                    sig = StrategySignalWrapper(
                        code=str(row.get("code", "")),
                        name=str(row.get("name", "")),
                        action=str(row.get("action", "hold")),
                        strength=float(row.get("strength", 0.5)),
                        reason=str(row.get("reason", "")),
                        strategy_id=strategy_id,
                        strategy_name=str(row.get("strategy_name", strategy_id)),
                        timestamp=f"{date}T15:00:00",
                    )
                    signals.append(sig)
        except Exception as e:
            logger.warning(f"加载策略信号失败: {e}")
            return self._generate_mock_signals(date)
        
        return signals if signals else self._generate_mock_signals(date)

    def _generate_mock_signals(self, date: str) -> List[StrategySignalWrapper]:
        mock_codes = [
            ("000001.SZ", "平安银行"),
            ("000002.SZ", "万科A"),
            ("600000.SH", "浦发银行"),
            ("600036.SH", "招商银行"),
            ("601318.SH", "中国平安"),
        ]
        
        strategies = ["multi_factor", "turtle_trading", "grid_trading"]
        
        np.random.seed(hash(date + "strategy") % 2**32)
        
        signals = []
        for code, name in mock_codes:
            if np.random.random() > 0.3:
                action = np.random.choice(["buy", "sell", "hold"], p=[0.4, 0.3, 0.3])
                strength = np.random.uniform(0.3, 0.9) if action != "hold" else 0
                
                sig = StrategySignalWrapper(
                    code=code,
                    name=name,
                    action=action,
                    strength=strength,
                    reason=f"模拟策略信号 - {action}",
                    strategy_id=np.random.choice(strategies),
                    strategy_name=np.random.choice(strategies),
                    timestamp=f"{date}T15:00:00",
                )
                signals.append(sig)
        
        return signals


class MarketContextProvider:
    """市场环境数据提供者"""

    def __init__(self, config: Optional[DataSourceConfig] = None):
        self.config = config or DataSourceConfig()
        self._cache: Dict[str, MarketContext] = {}

    def get_context(self, date: str) -> MarketContext:
        if date in self._cache:
            return self._cache[date]
        
        context = self._load_context(date)
        self._cache[date] = context
        return context

    def get_context_range(
        self, start_date: str, end_date: str
    ) -> Dict[str, MarketContext]:
        result = {}
        
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        current = start
        
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            result[date_str] = self.get_context(date_str)
            current += timedelta(days=1)
        
        return result

    def _load_context(self, date: str) -> MarketContext:
        context_file = Path(self.config.market_data_dir) / "context" / f"{date}.json"
        
        if context_file.exists():
            try:
                import json
                with open(context_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                return MarketContext(
                    index_value=data.get("index_value", 3000),
                    index_change=data.get("index_change", 0),
                    volatility=data.get("volatility", 0.02),
                    trend=data.get("trend", "neutral"),
                    regime=MarketRegime(data.get("regime", "neutral")),
                    uncertainty=data.get("uncertainty", 0.5),
                    north_money_flow=data.get("north_money_flow", 0),
                    sentiment_score=data.get("sentiment_score", 0.5),
                    timestamp=f"{date}T15:00:00",
                )
            except Exception as e:
                logger.warning(f"加载市场环境失败 {context_file}: {e}")
        
        return self._generate_mock_context(date)

    def _generate_mock_context(self, date: str) -> MarketContext:
        np.random.seed(hash(date + "market") % 2**32)
        
        index_value = 3000 + np.random.uniform(-200, 200)
        index_change = np.random.uniform(-0.03, 0.03)
        volatility = np.random.uniform(0.01, 0.04)
        
        if index_change > 0.01:
            trend = "up"
        elif index_change < -0.01:
            trend = "down"
        else:
            trend = "neutral"
        
        if trend == "up":
            if volatility > 0.025:
                regime = MarketRegime.BULL_CHOPPY
            else:
                regime = MarketRegime.BULL_TRENDING
        elif trend == "down":
            if volatility > 0.025:
                regime = MarketRegime.BEAR_CHOPPY
            else:
                regime = MarketRegime.BEAR_TRENDING
        else:
            regime = MarketRegime.NEUTRAL
        
        return MarketContext(
            index_value=index_value,
            index_change=index_change,
            volatility=volatility,
            trend=trend,
            regime=regime,
            uncertainty=np.random.uniform(0.3, 0.7),
            north_money_flow=np.random.uniform(-50, 50),
            sentiment_score=np.random.uniform(0.3, 0.7),
            timestamp=f"{date}T15:00:00",
        )


class PriceDataProvider:
    """价格数据提供者"""

    def __init__(self, config: Optional[DataSourceConfig] = None):
        self.config = config or DataSourceConfig()
        self._cache: Optional[pd.DataFrame] = None

    def get_prices(
        self,
        start_date: str,
        end_date: str,
        codes: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        price_file = Path(self.config.market_data_dir) / "prices.csv"
        
        if price_file.exists():
            try:
                df = pd.read_csv(price_file)
                df["date"] = df["date"].astype(str)
                
                df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
                
                if codes:
                    df = df[df["code"].isin(codes)]
                
                return df
            except Exception as e:
                logger.warning(f"加载价格数据失败 {price_file}: {e}")
        
        return self._generate_mock_prices(start_date, end_date, codes)

    def _generate_mock_prices(
        self, start_date: str, end_date: str, codes: Optional[List[str]]
    ) -> pd.DataFrame:
        mock_codes = codes or [
            "000001.SZ", "000002.SZ", "600000.SH", "600036.SH", "601318.SH"
        ]
        
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        dates = []
        current = start
        while current <= end:
            if current.weekday() < 5:
                dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        
        np.random.seed(42)
        
        records = []
        base_prices = {code: np.random.uniform(10, 100) for code in mock_codes}
        
        for date in dates:
            for code in mock_codes:
                change = np.random.uniform(-0.03, 0.03)
                base_prices[code] *= (1 + change)
                
                records.append({
                    "date": date,
                    "code": code,
                    "open": base_prices[code] * (1 + np.random.uniform(-0.01, 0.01)),
                    "high": base_prices[code] * (1 + np.random.uniform(0, 0.02)),
                    "low": base_prices[code] * (1 + np.random.uniform(-0.02, 0)),
                    "close": base_prices[code],
                    "volume": np.random.randint(1000000, 10000000),
                })
        
        return pd.DataFrame(records)


class FusionDataProvider:
    """融合数据统一提供者"""

    def __init__(self, config: Optional[DataSourceConfig] = None):
        self.config = config or DataSourceConfig()
        
        self.model_provider = ModelPredictionProvider(self.config)
        self.strategy_provider = StrategySignalProvider(self.config)
        self.market_provider = MarketContextProvider(self.config)
        self.price_provider = PriceDataProvider(self.config)

    def prepare_backtest_data(
        self,
        start_date: str,
        end_date: str,
        codes: Optional[List[str]] = None,
        strategy_ids: Optional[List[str]] = None,
        model_type: str = "default",
    ) -> Tuple[
        pd.DataFrame,
        Dict[str, List[ModelPrediction]],
        Dict[str, List[StrategySignalWrapper]],
        Dict[str, MarketContext],
    ]:
        price_data = self.price_provider.get_prices(start_date, end_date, codes)
        
        if codes is None and not price_data.empty:
            codes = price_data["code"].unique().tolist()
        
        model_predictions = self.model_provider.get_predictions_range(
            start_date, end_date, codes, model_type
        )
        
        strategy_signals = self.strategy_provider.get_signals_range(
            start_date, end_date, codes, strategy_ids
        )
        
        market_contexts = self.market_provider.get_context_range(start_date, end_date)
        
        return price_data, model_predictions, strategy_signals, market_contexts

    def get_daily_data(
        self,
        date: str,
        codes: Optional[List[str]] = None,
        strategy_ids: Optional[List[str]] = None,
        model_type: str = "default",
    ) -> Tuple[
        Dict[str, ModelPrediction],
        Dict[str, StrategySignalWrapper],
        MarketContext,
    ]:
        model_preds = self.model_provider.get_predictions(date, codes, model_type)
        strategy_sigs = self.strategy_provider.get_signals(date, codes, strategy_ids)
        market_ctx = self.market_provider.get_context(date)
        
        return model_preds, strategy_sigs, market_ctx

    def clear_cache(self):
        self.model_provider._cache.clear()
        self.strategy_provider._cache.clear()
        self.market_provider._cache.clear()
        self.price_provider._cache = None
