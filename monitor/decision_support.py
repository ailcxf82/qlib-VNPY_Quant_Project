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
class TradeSignal:
    code: str
    name: str
    action: str
    signal_source: str
    confidence: float
    score: float
    reasons: List[str]
    risk_assessment: str
    position_sizing: Optional[Dict[str, Any]] = None
    stop_loss: Optional[Dict[str, Any]] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "action": self.action,
            "signal_source": self.signal_source,
            "confidence": self.confidence,
            "score": self.score,
            "reasons": self.reasons,
            "risk_assessment": self.risk_assessment,
            "position_sizing": self.position_sizing,
            "stop_loss": self.stop_loss,
            "timestamp": self.timestamp,
        }


@dataclass
class DecisionReport:
    date: str
    market_regime: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    buy_signals: List[TradeSignal]
    sell_signals: List[TradeSignal]
    hold_signals: List[TradeSignal]
    summary: str
    recommendations: List[str]
    requires_attention: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "market_regime": self.market_regime,
            "risk_metrics": self.risk_metrics,
            "buy_signals": [s.to_dict() for s in self.buy_signals],
            "sell_signals": [s.to_dict() for s in self.sell_signals],
            "hold_signals": [s.to_dict() for s in self.hold_signals],
            "summary": self.summary,
            "recommendations": self.recommendations,
            "requires_attention": self.requires_attention,
        }


class DecisionSupportSystem:
    def __init__(self, config_path: str = "config/monitor.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.decision_config = self.config.get("decision_support", {})
        
        self.min_confidence = self.decision_config.get("min_confidence", 0.5)
        self.max_signals = self.decision_config.get("max_signals", 10)
        self.auto_execute_threshold = self.decision_config.get("auto_execute_threshold", 0.85)
        
        self._signal_history: List[Dict[str, Any]] = []
        self._report_cache: Optional[DecisionReport] = None
    
    def _load_config(self) -> Dict[str, Any]:
        p = Path(self.config_path)
        if not p.exists():
            return {}
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    
    def generate_buy_signals(
        self,
        combined_signals: List[Any],
        position_sizings: Dict[str, Any],
        stop_losses: Dict[str, Any],
        risk_metrics: Any,
    ) -> List[TradeSignal]:
        signals = []
        
        for sig in combined_signals:
            if sig.signal_type not in ["strong_buy", "buy"]:
                continue
            
            if sig.confidence < self.min_confidence:
                continue
            
            code = sig.code
            position_sizing = position_sizings.get(code, {})
            stop_loss = stop_losses.get(code, {})
            
            risk_assessment = self._assess_buy_risk(sig, risk_metrics)
            
            signal = TradeSignal(
                code=code,
                name=sig.name,
                action="buy",
                signal_source="multi_strategy",
                confidence=sig.confidence,
                score=sig.total_score,
                reasons=sig.reasons,
                risk_assessment=risk_assessment,
                position_sizing=position_sizing.to_dict() if hasattr(position_sizing, 'to_dict') else position_sizing,
                stop_loss=stop_loss.to_dict() if hasattr(stop_loss, 'to_dict') else stop_loss,
            )
            signals.append(signal)
        
        signals.sort(key=lambda x: x.score, reverse=True)
        return signals[:self.max_signals]
    
    def generate_sell_signals(
        self,
        sell_recommendations: List[Dict[str, Any]],
        extreme_alerts: List[Dict[str, Any]],
    ) -> List[TradeSignal]:
        signals = []
        
        for rec in sell_recommendations:
            signal = TradeSignal(
                code=rec["code"],
                name=rec["name"],
                action="sell",
                signal_source=rec.get("stop_type", "risk_control"),
                confidence=0.8 if rec.get("priority", 0) >= 60 else 0.6,
                score=rec.get("profit_pct", 0),
                reasons=[rec["reason"]],
                risk_assessment=f"止损类型: {rec.get('stop_type', 'unknown')}",
            )
            signals.append(signal)
        
        for alert in extreme_alerts:
            signal = TradeSignal(
                code=alert["code"],
                name=alert["name"],
                action="sell",
                signal_source="extreme_detection",
                confidence=0.9,
                score=alert.get("sentiment_score", 0.5),
                reasons=[alert.get("reason", "极端情况")],
                risk_assessment=f"严重程度: {alert.get('severity', 'high')}",
            )
            signals.append(signal)
        
        signals.sort(key=lambda x: x.confidence, reverse=True)
        return signals
    
    def generate_hold_signals(
        self,
        positions: Dict[str, Any],
        combined_signals: List[Any],
    ) -> List[TradeSignal]:
        signals = []
        signal_map = {s.code: s for s in combined_signals}
        
        for code, pos in positions.items():
            if code in signal_map:
                sig = signal_map[code]
                if sig.signal_type in ["hold", "reduce"]:
                    signal = TradeSignal(
                        code=code,
                        name=pos.get("name", code),
                        action="hold" if sig.signal_type == "hold" else "reduce",
                        signal_source="position_monitor",
                        confidence=sig.confidence,
                        score=sig.total_score,
                        reasons=sig.reasons,
                        risk_assessment=f"持仓{pos.get('holding_days', 0)}天, 收益{pos.get('profit_pct', 0):.1%}",
                    )
                    signals.append(signal)
        
        return signals
    
    def _assess_buy_risk(self, signal: Any, risk_metrics: Any) -> str:
        risk_level = risk_metrics.risk_level if hasattr(risk_metrics, 'risk_level') else "unknown"
        
        if risk_level == "critical":
            return "高风险: 市场环境恶劣，不建议买入"
        elif risk_level == "high":
            return "中高风险: 建议降低仓位或观望"
        elif signal.confidence < 0.6:
            return "信号置信度较低，建议谨慎"
        else:
            return "风险可控，可按计划执行"
    
    def generate_decision_report(
        self,
        market_regime: Any,
        risk_metrics: Any,
        combined_signals: List[Any],
        positions: Dict[str, Any],
        sell_recommendations: List[Dict[str, Any]],
        extreme_alerts: List[Dict[str, Any]],
        position_sizings: Dict[str, Any],
        stop_losses: Dict[str, Any],
    ) -> DecisionReport:
        buy_signals = self.generate_buy_signals(
            combined_signals, position_sizings, stop_losses, risk_metrics
        )
        
        sell_signals = self.generate_sell_signals(sell_recommendations, extreme_alerts)
        
        hold_signals = self.generate_hold_signals(positions, combined_signals)
        
        summary = self._generate_summary(
            market_regime, risk_metrics, buy_signals, sell_signals
        )
        
        recommendations = self._generate_recommendations(
            market_regime, risk_metrics, buy_signals, sell_signals
        )
        
        requires_attention = (
            risk_metrics.risk_level in ["high", "critical"] or
            len(sell_signals) > 0 or
            any(s.confidence >= self.auto_execute_threshold for s in buy_signals)
        )
        
        report = DecisionReport(
            date=datetime.now().strftime("%Y-%m-%d"),
            market_regime=market_regime.to_dict() if hasattr(market_regime, 'to_dict') else market_regime,
            risk_metrics=risk_metrics.to_dict() if hasattr(risk_metrics, 'to_dict') else risk_metrics,
            buy_signals=buy_signals,
            sell_signals=sell_signals,
            hold_signals=hold_signals,
            summary=summary,
            recommendations=recommendations,
            requires_attention=requires_attention,
        )
        
        self._report_cache = report
        self._save_report(report)
        
        return report
    
    def _generate_summary(
        self,
        market_regime: Any,
        risk_metrics: Any,
        buy_signals: List[TradeSignal],
        sell_signals: List[TradeSignal],
    ) -> str:
        regime_name = market_regime.regime if hasattr(market_regime, 'regime') else "unknown"
        risk_level = risk_metrics.risk_level if hasattr(risk_metrics, 'risk_level') else "unknown"
        
        lines = [
            f"市场环境: {regime_name}",
            f"风险等级: {risk_level}",
            f"买入信号: {len(buy_signals)} 个",
            f"卖出信号: {len(sell_signals)} 个",
        ]
        
        return " | ".join(lines)
    
    def _generate_recommendations(
        self,
        market_regime: Any,
        risk_metrics: Any,
        buy_signals: List[TradeSignal],
        sell_signals: List[TradeSignal],
    ) -> List[str]:
        recommendations = []
        
        regime_name = market_regime.regime if hasattr(market_regime, 'regime') else "unknown"
        
        if "bear" in regime_name:
            recommendations.append("熊市环境，建议降低仓位，优先保护本金")
        elif "bull" in regime_name:
            recommendations.append("牛市环境，可适当增加仓位，把握趋势")
        
        if risk_metrics.risk_level == "critical":
            recommendations.append("⚠️ 风险水平危急，建议立即减仓")
        elif risk_metrics.risk_level == "high":
            recommendations.append("⚠️ 风险水平较高，建议谨慎操作")
        
        if sell_signals:
            recommendations.append(f"🔴 有 {len(sell_signals)} 个卖出信号需要处理")
        
        if buy_signals:
            top_signal = buy_signals[0]
            if top_signal.confidence >= self.auto_execute_threshold:
                recommendations.append(f"🟢 {top_signal.name} 信号置信度高，可考虑执行")
            else:
                recommendations.append(f"🟡 {top_signal.name} 信号置信度中等，建议人工审核")
        
        return recommendations
    
    def _save_report(self, report: DecisionReport) -> None:
        report_dir = Path(self.config.get("paths", {}).get("data_dir", "data/monitor"))
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = report_dir / f"decision_report_{datetime.now().strftime('%Y%m%d')}.json"
        
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
    
    def format_report_message(self, report: DecisionReport) -> str:
        lines = [
            "📋 **投资决策报告**",
            f"日期: {report.date}",
            "",
            f"**市场环境**: {report.market_regime.get('regime', 'unknown')}",
            f"**风险等级**: {report.risk_metrics.get('risk_level', 'unknown')}",
            "",
        ]
        
        if report.sell_signals:
            lines.append("🔴 **卖出信号**")
            for sig in report.sell_signals[:5]:
                lines.append(f"- {sig.name}({sig.code}): {sig.reasons[0]}")
            lines.append("")
        
        if report.buy_signals:
            lines.append("🟢 **买入信号**")
            for sig in report.buy_signals[:5]:
                lines.append(f"- {sig.name}({sig.code}): 评分{sig.score:.2f}, 置信度{sig.confidence:.0%}")
            lines.append("")
        
        if report.recommendations:
            lines.append("**操作建议**")
            for rec in report.recommendations:
                lines.append(f"- {rec}")
        
        return "\n".join(lines)
    
    def get_execution_checklist(self, report: DecisionReport) -> List[Dict[str, Any]]:
        checklist = []
        
        for sig in report.sell_signals:
            checklist.append({
                "type": "sell",
                "code": sig.code,
                "name": sig.name,
                "action": f"卖出 {sig.name}",
                "reason": sig.reasons[0],
                "priority": "high" if sig.confidence >= 0.8 else "medium",
                "auto_allowed": sig.confidence >= self.auto_execute_threshold,
            })
        
        for sig in report.buy_signals:
            checklist.append({
                "type": "buy",
                "code": sig.code,
                "name": sig.name,
                "action": f"买入 {sig.name}",
                "reason": sig.reasons[0] if sig.reasons else "综合评分",
                "priority": "high" if sig.confidence >= 0.7 else "medium",
                "auto_allowed": sig.confidence >= self.auto_execute_threshold,
                "position_size": sig.position_sizing,
            })
        
        checklist.sort(key=lambda x: (
            0 if x["type"] == "sell" else 1,
            0 if x["priority"] == "high" else 1
        ))
        
        return checklist
