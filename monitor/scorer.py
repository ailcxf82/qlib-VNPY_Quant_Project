from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ScoreDimension:
    name: str
    score: float
    weight: float
    raw_value: Any = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    def weighted_score(self) -> float:
        return self.score * self.weight


@dataclass
class StockScore:
    code: str
    name: str
    date: str
    total_score: float
    dimensions: List[ScoreDimension]
    rank: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "date": self.date,
            "total_score": self.total_score,
            "rank": self.rank,
            "dimensions": [
                {
                    "name": d.name,
                    "score": d.score,
                    "weight": d.weight,
                    "weighted_score": d.weighted_score(),
                    "details": d.details,
                }
                for d in self.dimensions
            ],
        }


class MultiDimensionScorer:
    DEFAULT_WEIGHTS = {
        "model_prediction": 0.40,
        "sentiment": 0.25,
        "strategy": 0.20,
        "technical": 0.15,
    }
    
    def __init__(self, config_path: str = "config/monitor.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.weights = self.config.get("scoring", {}).get("weights", self.DEFAULT_WEIGHTS)
        
        self._ts_client = None
    
    def _load_config(self) -> Dict[str, Any]:
        p = Path(self.config_path)
        if not p.exists():
            return {}
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    
    def _get_ts_client(self):
        if self._ts_client is None:
            try:
                from monitor.ts_client import get_ts_client
                self._ts_client = get_ts_client()
            except Exception as e:
                logger.warning(f"无法获取 TushareClient: {e}")
        return self._ts_client
    
    def _get_stock_name(self, code: str, name_map: Dict[str, str]) -> str:
        code_clean = str(code).replace(".XSHE", "").replace(".XSHG", "").zfill(6)
        return name_map.get(code_clean, code_clean)
    
    def calculate_model_score(self, prediction_score: float) -> ScoreDimension:
        normalized = max(0, min(1, (prediction_score + 1) / 2)) if prediction_score < 0 else prediction_score
        normalized = min(1, max(0, normalized))
        
        return ScoreDimension(
            name="model_prediction",
            score=normalized,
            weight=self.weights.get("model_prediction", 0.40),
            raw_value=prediction_score,
            details={"raw_prediction": prediction_score},
        )
    
    def calculate_sentiment_score(self, code: str, sentiment_data: Optional[Dict] = None) -> ScoreDimension:
        score = 0.5
        details = {}
        
        if sentiment_data:
            stock_sentiment = sentiment_data.get("stocks", {}).get(code, {})
            if stock_sentiment:
                news_count = stock_sentiment.get("news_count", 0)
                sentiment_value = stock_sentiment.get("sentiment", 0.5)
                sentiment_score = stock_sentiment.get("score", 0.5)
                
                if news_count > 0:
                    score = sentiment_score
                    details = {
                        "news_count": news_count,
                        "sentiment": sentiment_value,
                        "sentiment_score": sentiment_score,
                    }
        
        return ScoreDimension(
            name="sentiment",
            score=score,
            weight=self.weights.get("sentiment", 0.25),
            details=details,
        )
    
    def calculate_strategy_score(self, code: str, strategy_rules: Dict, 
                                  prediction_rank: int, total_stocks: int) -> ScoreDimension:
        score = 0.5
        details = {"rank": prediction_rank, "total": total_stocks}
        
        if total_stocks > 0 and prediction_rank > 0:
            rank_score = 1 - (prediction_rank - 1) / total_stocks
            score = rank_score
            details["rank_score"] = rank_score
        
        rule_count = len(strategy_rules.get("buy", []))
        if rule_count > 0:
            details["rule_count"] = rule_count
        
        return ScoreDimension(
            name="strategy",
            score=score,
            weight=self.weights.get("strategy", 0.20),
            details=details,
        )
    
    def calculate_technical_score(self, code: str, technical_data: Optional[Dict] = None) -> ScoreDimension:
        score = 0.5
        details = {}
        
        if technical_data:
            ts = technical_data.get(code, {})
            if ts:
                ma_score = ts.get("ma_score", 0.5)
                rsi_score = ts.get("rsi_score", 0.5)
                macd_score = ts.get("macd_score", 0.5)
                volume_score = ts.get("volume_score", 0.5)
                
                score = (ma_score * 0.3 + rsi_score * 0.25 + 
                        macd_score * 0.25 + volume_score * 0.2)
                
                details = {
                    "ma_score": ma_score,
                    "rsi_score": rsi_score,
                    "macd_score": macd_score,
                    "volume_score": volume_score,
                }
        
        return ScoreDimension(
            name="technical",
            score=score,
            weight=self.weights.get("technical", 0.15),
            details=details,
        )
    
    def calculate_total_score(
        self,
        dimensions: Optional[List[ScoreDimension]] = None,
        model_score: float = 0.5,
        sentiment_score: float = 0.5,
        strategy_score: float = 0.5,
        technical_score: float = 0.5,
    ) -> float:
        if dimensions is not None:
            total_weight = sum(d.weight for d in dimensions)
            if total_weight == 0:
                return 0.0
            weighted_sum = sum(d.weighted_score() for d in dimensions)
            return weighted_sum / total_weight
        
        weights = self.weights
        total_weight = (
            weights.get("model_prediction", 0.40) +
            weights.get("sentiment", 0.25) +
            weights.get("strategy", 0.20) +
            weights.get("technical", 0.15)
        )
        
        if total_weight == 0:
            return 0.0
        
        weighted_sum = (
            model_score * weights.get("model_prediction", 0.40) +
            sentiment_score * weights.get("sentiment", 0.25) +
            strategy_score * weights.get("strategy", 0.20) +
            technical_score * weights.get("technical", 0.15)
        )
        
        return weighted_sum / total_weight
    
    def score_stock(
        self,
        code: str,
        name: str,
        prediction_score: float,
        prediction_rank: int,
        total_stocks: int,
        strategy_rules: Dict,
        sentiment_data: Optional[Dict] = None,
        technical_data: Optional[Dict] = None,
    ) -> StockScore:
        dimensions = [
            self.calculate_model_score(prediction_score),
            self.calculate_sentiment_score(code, sentiment_data),
            self.calculate_strategy_score(code, strategy_rules, prediction_rank, total_stocks),
            self.calculate_technical_score(code, technical_data),
        ]
        
        total_score = self.calculate_total_score(dimensions=dimensions)
        
        return StockScore(
            code=code,
            name=name,
            date=datetime.now().strftime("%Y-%m-%d"),
            total_score=total_score,
            dimensions=dimensions,
        )
    
    def score_stock_pool(
        self,
        predictions: List[Dict],
        strategy_rules: Dict,
        name_map: Dict[str, str],
        sentiment_data: Optional[Dict] = None,
        technical_data: Optional[Dict] = None,
    ) -> List[StockScore]:
        total_stocks = len(predictions)
        scores = []
        
        for pred in predictions:
            code = pred.get("instrument", pred.get("code", ""))
            name = self._get_stock_name(code, name_map)
            prediction_score = pred.get("final", pred.get("score", 0.5))
            prediction_rank = pred.get("rank", 0)
            
            stock_score = self.score_stock(
                code=code,
                name=name,
                prediction_score=prediction_score,
                prediction_rank=prediction_rank,
                total_stocks=total_stocks,
                strategy_rules=strategy_rules,
                sentiment_data=sentiment_data,
                technical_data=technical_data,
            )
            scores.append(stock_score)
        
        scores.sort(key=lambda x: x.total_score, reverse=True)
        
        for i, score in enumerate(scores, 1):
            score.rank = i
        
        return scores
    
    def get_top_stocks(self, scores: List[StockScore], top_k: int = 20) -> List[StockScore]:
        return scores[:top_k]
    
    def filter_by_score(self, scores: List[StockScore], min_score: float = 0.5) -> List[StockScore]:
        return [s for s in scores if s.total_score >= min_score]
