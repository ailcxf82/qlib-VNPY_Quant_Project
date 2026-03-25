from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ExtremeAlert:
    code: str
    name: str
    date: str
    alert_type: str
    severity: str
    sentiment_score: float
    technical_confirmed: bool
    confirmation_reason: str
    action: str
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "date": self.date,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "sentiment_score": self.sentiment_score,
            "technical_confirmed": self.technical_confirmed,
            "confirmation_reason": self.confirmation_reason,
            "action": self.action,
            "details": self.details,
        }


@dataclass
class ExtremeConditionResult:
    date: str
    total_alerts: int
    sell_alerts: List[ExtremeAlert]
    warning_alerts: List[ExtremeAlert]
    watch_alerts: List[ExtremeAlert]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "total_alerts": self.total_alerts,
            "sell_alerts": [a.to_dict() for a in self.sell_alerts],
            "warning_alerts": [a.to_dict() for a in self.warning_alerts],
            "watch_alerts": [a.to_dict() for a in self.watch_alerts],
        }


class ExtremeConditionDetector:
    EXTREME_NEGATIVE_THRESHOLD = 0.25
    EXTREME_POSITIVE_THRESHOLD = 0.75
    
    SELL_TRIGGERS = [
        "breakdown_confirmed",
        "extreme_negative_with_technical",
        "volume_breakdown",
        "trend_reversal",
    ]
    
    def __init__(self, config_path: str = "config/monitor.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.extreme_config = self.config.get("extreme_detection", {})
        
        self.negative_threshold = self.extreme_config.get("negative_threshold", 0.25)
        self.positive_threshold = self.extreme_config.get("positive_threshold", 0.75)
        self.confirmation_required = self.extreme_config.get("confirmation_required", True)
        
        self._cache_file = Path(self.config.get("paths", {}).get(
            "extreme_cache_file", "data/monitor/extreme_cache.json"
        ))
        
        self._sentiment_analyzer = None
        self._technical_analyzer = None
    
    def _load_config(self) -> Dict[str, Any]:
        p = Path(self.config_path)
        if not p.exists():
            return {}
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    
    def _get_sentiment_analyzer(self):
        if self._sentiment_analyzer is None:
            from monitor.enhanced_sentiment import EnhancedSentimentAnalyzer
            self._sentiment_analyzer = EnhancedSentimentAnalyzer(self.config_path)
        return self._sentiment_analyzer
    
    def _get_technical_analyzer(self):
        if self._technical_analyzer is None:
            from monitor.technical_analyzer import TechnicalAnalyzer
            self._technical_analyzer = TechnicalAnalyzer(self.config_path)
        return self._technical_analyzer
    
    def detect_extreme_sentiment(
        self, 
        code: str,
        sentiment_data: Optional[Dict] = None
    ) -> Optional[Dict[str, Any]]:
        analyzer = self._get_sentiment_analyzer()
        
        if sentiment_data:
            stock_sentiment = sentiment_data.get("stocks", {}).get(code)
        else:
            stock_sentiment = analyzer.get_stock_sentiment(code)
        
        if stock_sentiment is None:
            return None
        
        score = stock_sentiment.score
        
        if score <= self.negative_threshold:
            return {
                "code": code,
                "name": stock_sentiment.name,
                "sentiment_type": "extreme_negative",
                "sentiment_score": score,
                "news_count": stock_sentiment.news_count,
                "negative_ratio": stock_sentiment.negative_count / max(stock_sentiment.news_count, 1),
            }
        elif score >= self.positive_threshold:
            return {
                "code": code,
                "name": stock_sentiment.name,
                "sentiment_type": "extreme_positive",
                "sentiment_score": score,
                "news_count": stock_sentiment.news_count,
                "positive_ratio": stock_sentiment.positive_count / max(stock_sentiment.news_count, 1),
            }
        
        return None
    
    def confirm_with_technical(
        self, 
        code: str, 
        sentiment_type: str
    ) -> Dict[str, Any]:
        analyzer = self._get_technical_analyzer()
        return analyzer.confirm_extreme_condition(code, sentiment_type)
    
    def confirm_extreme_condition(
        self, 
        code: str, 
        sentiment_type: str
    ) -> Dict[str, Any]:
        return self.confirm_with_technical(code, sentiment_type)
    
    def generate_alert(
        self,
        code: str,
        name: str,
        sentiment_info: Dict,
        technical_confirmation: Dict,
    ) -> ExtremeAlert:
        sentiment_type = sentiment_info.get("sentiment_type", "")
        sentiment_score = sentiment_info.get("sentiment_score", 0.5)
        technical_confirmed = technical_confirmation.get("confirmed", False)
        confirmation_reason = technical_confirmation.get("reason", "")
        
        if sentiment_type == "extreme_negative":
            if technical_confirmed:
                severity = "critical"
                action = "sell"
                alert_type = "extreme_negative_with_technical"
            elif sentiment_score < 0.15:
                severity = "high"
                action = "sell"
                alert_type = "extreme_negative_severe"
            else:
                severity = "medium"
                action = "watch"
                alert_type = "extreme_negative_unconfirmed"
        
        elif sentiment_type == "extreme_positive":
            if technical_confirmed:
                severity = "info"
                action = "hold_or_buy"
                alert_type = "extreme_positive_confirmed"
            else:
                severity = "low"
                action = "watch"
                alert_type = "extreme_positive_unconfirmed"
        
        else:
            severity = "low"
            action = "hold"
            alert_type = "normal"
        
        return ExtremeAlert(
            code=code,
            name=name,
            date=datetime.now().strftime("%Y-%m-%d"),
            alert_type=alert_type,
            severity=severity,
            sentiment_score=sentiment_score,
            technical_confirmed=technical_confirmed,
            confirmation_reason=confirmation_reason,
            action=action,
            details={
                "sentiment_info": sentiment_info,
                "technical_details": technical_confirmation.get("details", {}),
            },
        )
    
    def detect_for_positions(
        self, 
        positions: Dict[str, Any],
        sentiment_data: Optional[Dict] = None,
    ) -> ExtremeConditionResult:
        sell_alerts = []
        warning_alerts = []
        watch_alerts = []
        
        for code, pos_info in positions.items():
            extreme_sentiment = self.detect_extreme_sentiment(code, sentiment_data)
            
            if extreme_sentiment is None:
                continue
            
            sentiment_type = extreme_sentiment.get("sentiment_type", "")
            
            technical_confirmation = self.confirm_with_technical(code, sentiment_type)
            
            alert = self.generate_alert(
                code=code,
                name=extreme_sentiment.get("name", code),
                sentiment_info=extreme_sentiment,
                technical_confirmation=technical_confirmation,
            )
            
            if alert.action == "sell":
                sell_alerts.append(alert)
            elif alert.severity in ["high", "critical"]:
                warning_alerts.append(alert)
            else:
                watch_alerts.append(alert)
        
        return ExtremeConditionResult(
            date=datetime.now().strftime("%Y-%m-%d"),
            total_alerts=len(sell_alerts) + len(warning_alerts) + len(watch_alerts),
            sell_alerts=sell_alerts,
            warning_alerts=warning_alerts,
            watch_alerts=watch_alerts,
        )
    
    def check_single_stock(
        self, 
        code: str,
        sentiment_data: Optional[Dict] = None,
    ) -> Optional[ExtremeAlert]:
        extreme_sentiment = self.detect_extreme_sentiment(code, sentiment_data)
        
        if extreme_sentiment is None:
            return None
        
        sentiment_type = extreme_sentiment.get("sentiment_type", "")
        technical_confirmation = self.confirm_with_technical(code, sentiment_type)
        
        return self.generate_alert(
            code=code,
            name=extreme_sentiment.get("name", code),
            sentiment_info=extreme_sentiment,
            technical_confirmation=technical_confirmation,
        )
    
    def should_trigger_sell(self, alert: ExtremeAlert) -> bool:
        if alert.action == "sell":
            return True
        
        if alert.alert_type in self.SELL_TRIGGERS:
            return True
        
        if alert.sentiment_score < 0.15 and alert.technical_confirmed:
            return True
        
        return False
    
    def get_sell_recommendations(
        self, 
        positions: Dict[str, Any],
        sentiment_data: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        result = self.detect_for_positions(positions, sentiment_data)
        
        recommendations = []
        for alert in result.sell_alerts:
            if self.should_trigger_sell(alert):
                recommendations.append({
                    "code": alert.code,
                    "name": alert.name,
                    "reason": f"极端情况触发卖出: {alert.alert_type}",
                    "sentiment_score": alert.sentiment_score,
                    "technical_confirmed": alert.technical_confirmed,
                    "confirmation_reason": alert.confirmation_reason,
                    "severity": alert.severity,
                })
        
        return recommendations
    
    def save_alerts(self, result: ExtremeConditionResult):
        self._cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "last_check": datetime.now().isoformat(),
            "result": result.to_dict(),
        }
        
        with open(self._cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_last_alerts(self) -> Optional[Dict[str, Any]]:
        if not self._cache_file.exists():
            return None
        
        try:
            with open(self._cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"加载极端情况缓存失败: {e}")
            return None
