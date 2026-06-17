from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    date: str
    total_risk: float
    position_risk: float
    drawdown_risk: float
    concentration_risk: float
    volatility_risk: float
    risk_level: str
    alerts: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "total_risk": self.total_risk,
            "position_risk": self.position_risk,
            "drawdown_risk": self.drawdown_risk,
            "concentration_risk": self.concentration_risk,
            "volatility_risk": self.volatility_risk,
            "risk_level": self.risk_level,
            "alerts": self.alerts,
        }


@dataclass
class StopLossLevel:
    code: str
    name: str
    current_price: float
    entry_price: float
    profit_pct: float
    initial_stop: float
    trailing_stop: float
    time_stop: Optional[str]
    stop_type: str
    stop_reason: str
    should_stop: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "current_price": self.current_price,
            "entry_price": self.entry_price,
            "profit_pct": self.profit_pct,
            "initial_stop": self.initial_stop,
            "trailing_stop": self.trailing_stop,
            "time_stop": self.time_stop,
            "stop_type": self.stop_type,
            "stop_reason": self.stop_reason,
            "should_stop": self.should_stop,
        }


@dataclass
class PositionSizing:
    code: str
    name: str
    recommended_size: float
    recommended_shares: int
    max_size: float
    risk_per_share: float
    kelly_fraction: float
    reason: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "recommended_size": self.recommended_size,
            "recommended_shares": self.recommended_shares,
            "max_size": self.max_size,
            "risk_per_share": self.risk_per_share,
            "kelly_fraction": self.kelly_fraction,
            "reason": self.reason,
        }


class RiskController:
    DEFAULT_STOP_LOSS = -0.08
    DEFAULT_TAKE_PROFIT = 0.15
    MAX_POSITION_SIZE = 0.15
    MAX_SINGLE_LOSS = 0.02
    MAX_DRAWDOWN = 0.10
    MAX_CONCENTRATION = 0.30
    
    def __init__(self, config_path: str = "config/monitor.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.risk_config = self.config.get("risk_control", {})
        
        self.stop_loss = self.risk_config.get("stop_loss", self.DEFAULT_STOP_LOSS)
        self.take_profit = self.risk_config.get("take_profit", self.DEFAULT_TAKE_PROFIT)
        self.max_position_size = self.risk_config.get("max_position_size", self.MAX_POSITION_SIZE)
        self.max_single_loss = self.risk_config.get("max_single_loss", self.MAX_SINGLE_LOSS)
        self.max_drawdown = self.risk_config.get("max_drawdown", self.MAX_DRAWDOWN)
        self.max_concentration = self.risk_config.get("max_concentration", self.MAX_CONCENTRATION)
        
        self.trailing_stop_enabled = self.risk_config.get("trailing_stop_enabled", True)
        self.trailing_stop_trigger = self.risk_config.get("trailing_stop_trigger", 0.05)
        self.trailing_stop_distance = self.risk_config.get("trailing_stop_distance", 0.03)
        
        self.time_stop_enabled = self.risk_config.get("time_stop_enabled", True)
        self.max_holding_days = self.risk_config.get("max_holding_days", 10)
        
        self._peak_value: float = 0.0
        self._current_drawdown: float = 0.0
    
    def _load_config(self) -> Dict[str, Any]:
        p = Path(self.config_path)
        if not p.exists():
            return {}
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    
    def calculate_stop_loss_levels(
        self,
        code: str,
        name: str,
        entry_price: float,
        current_price: float,
        holding_days: int,
        highest_price: float,
        volatility: float = 0.02,
    ) -> StopLossLevel:
        profit_pct = (current_price - entry_price) / entry_price
        
        initial_stop_price = entry_price * (1 + self.stop_loss)
        
        trailing_stop_price = 0.0
        stop_type = "initial"
        stop_reason = ""
        
        if self.trailing_stop_enabled and profit_pct >= self.trailing_stop_trigger:
            trailing_stop_price = highest_price * (1 - self.trailing_stop_distance)
            stop_type = "trailing"
            stop_reason = f"盈利{profit_pct:.1%}触发移动止损"
        
        time_stop = None
        if self.time_stop_enabled and holding_days >= self.max_holding_days:
            time_stop = f"持仓{holding_days}天超时"
            if not stop_reason:
                stop_type = "time"
                stop_reason = time_stop
        
        if profit_pct <= self.stop_loss:
            stop_type = "hard_stop"
            stop_reason = f"亏损{profit_pct:.1%}触发止损"
        elif profit_pct >= self.take_profit:
            stop_type = "take_profit"
            stop_reason = f"盈利{profit_pct:.1%}触发止盈"
        
        effective_stop = max(initial_stop_price, trailing_stop_price) if trailing_stop_price > 0 else initial_stop_price
        
        vol_adjusted_stop = current_price * (1 - max(volatility * 2, abs(self.stop_loss)))
        effective_stop = min(effective_stop, vol_adjusted_stop)
        
        should_stop = (
            current_price <= effective_stop or
            profit_pct <= self.stop_loss or
            profit_pct >= self.take_profit or
            (self.time_stop_enabled and holding_days >= self.max_holding_days)
        )
        
        return StopLossLevel(
            code=code,
            name=name,
            current_price=current_price,
            entry_price=entry_price,
            profit_pct=profit_pct,
            initial_stop=initial_stop_price,
            trailing_stop=trailing_stop_price,
            time_stop=time_stop,
            stop_type=stop_type,
            stop_reason=stop_reason,
            should_stop=should_stop,
        )
    
    def calculate_position_size(
        self,
        code: str,
        name: str,
        entry_price: float,
        stop_price: float,
        total_capital: float,
        win_rate: float = 0.55,
        avg_win_loss_ratio: float = 1.5,
        confidence: float = 0.5,
    ) -> PositionSizing:
        risk_per_share = abs(entry_price - stop_price)
        
        kelly = win_rate - (1 - win_rate) / avg_win_loss_ratio
        kelly = max(0, min(kelly, 0.25))
        kelly_fraction = kelly * confidence
        
        max_risk_amount = total_capital * self.max_single_loss
        max_shares_by_risk = int(max_risk_amount / risk_per_share) if risk_per_share > 0 else 0
        
        max_position_value = total_capital * self.max_position_size
        max_shares_by_position = int(max_position_value / entry_price)
        
        recommended_shares = min(max_shares_by_risk, max_shares_by_position)
        
        kelly_shares = int(total_capital * kelly_fraction / entry_price)
        recommended_shares = min(recommended_shares, kelly_shares)
        
        recommended_shares = max(recommended_shares, 100)
        recommended_shares = (recommended_shares // 100) * 100
        
        recommended_size = recommended_shares * entry_price
        max_size = max_shares_by_position * entry_price
        
        reason = f"风险控制: 单笔最大亏损{self.max_single_loss:.1%}, 凯利比例{kelly_fraction:.1%}"
        
        return PositionSizing(
            code=code,
            name=name,
            recommended_size=recommended_size,
            recommended_shares=recommended_shares,
            max_size=max_size,
            risk_per_share=risk_per_share,
            kelly_fraction=kelly_fraction,
            reason=reason,
        )
    
    def calculate_portfolio_risk(
        self,
        positions: Dict[str, Any],
        total_capital: float,
        peak_value: float,
    ) -> RiskMetrics:
        self._peak_value = max(self._peak_value, peak_value)
        self._current_drawdown = (self._peak_value - total_capital) / self._peak_value if self._peak_value > 0 else 0
        
        position_risk = 0.0
        alerts = []
        
        for code, pos in positions.items():
            profit_pct = pos.get("profit_pct", 0)
            position_value = pos.get("market_value", 0)
            weight = position_value / total_capital if total_capital > 0 else 0
            
            if profit_pct < 0:
                position_risk += abs(profit_pct) * weight
            
            if weight > self.max_concentration:
                alerts.append(f"{pos.get('name', code)} 仓位{weight:.1%}超过集中度限制")
        
        drawdown_risk = self._current_drawdown
        
        concentration_risk = 0.0
        if positions:
            weights = [pos.get("market_value", 0) / total_capital for pos in positions.values() if total_capital > 0]
            if weights:
                concentration_risk = max(weights)
        
        volatility_risk = 0.0
        
        total_risk = (
            position_risk * 0.35 +
            drawdown_risk * 0.30 +
            concentration_risk * 0.20 +
            volatility_risk * 0.15
        )
        
        if total_risk < 0.2:
            risk_level = "low"
        elif total_risk < 0.4:
            risk_level = "medium"
        elif total_risk < 0.6:
            risk_level = "high"
        else:
            risk_level = "critical"
        
        if drawdown_risk > self.max_drawdown:
            alerts.append(f"回撤{drawdown_risk:.1%}超过阈值{self.max_drawdown:.1%}")
        
        if position_risk > 0.05:
            alerts.append(f"持仓亏损风险{position_risk:.1%}较高")
        
        return RiskMetrics(
            date=datetime.now().strftime("%Y-%m-%d"),
            total_risk=total_risk,
            position_risk=position_risk,
            drawdown_risk=drawdown_risk,
            concentration_risk=concentration_risk,
            volatility_risk=volatility_risk,
            risk_level=risk_level,
            alerts=alerts,
        )
    
    def get_sell_recommendations(
        self,
        positions: Dict[str, Any],
        prices: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        recommendations = []
        
        for code, pos in positions.items():
            entry_price = pos.get("buy_price", pos.get("entry_price", 0))
            current_price = prices.get(code, pos.get("current_price", 0))
            holding_days = pos.get("holding_days", 0)
            highest_price = pos.get("highest_price", current_price)
            
            if entry_price <= 0 or current_price <= 0:
                continue
            
            stop_level = self.calculate_stop_loss_levels(
                code=code,
                name=pos.get("name", code),
                entry_price=entry_price,
                current_price=current_price,
                holding_days=holding_days,
                highest_price=highest_price,
            )
            
            if stop_level.should_stop:
                recommendations.append({
                    "code": code,
                    "name": pos.get("name", code),
                    "action": "sell",
                    "reason": stop_level.stop_reason,
                    "stop_type": stop_level.stop_type,
                    "profit_pct": stop_level.profit_pct,
                    "current_price": current_price,
                    "stop_price": max(stop_level.initial_stop, stop_level.trailing_stop) if stop_level.trailing_stop > 0 else stop_level.initial_stop,
                    "priority": self._get_sell_priority(stop_level),
                })
        
        recommendations.sort(key=lambda x: x["priority"], reverse=True)
        
        return recommendations
    
    def _get_sell_priority(self, stop_level: StopLossLevel) -> int:
        if stop_level.stop_type == "hard_stop":
            return 100
        elif stop_level.stop_type == "take_profit":
            return 80
        elif stop_level.stop_type == "trailing":
            return 60
        elif stop_level.stop_type == "time":
            return 40
        else:
            return 20
    
    def adjust_position_for_risk(
        self,
        base_position_size: float,
        risk_metrics: RiskMetrics,
        regime_multiplier: float = 1.0,
    ) -> float:
        risk_multiplier = 1.0
        
        if risk_metrics.risk_level == "critical":
            risk_multiplier = 0.3
        elif risk_metrics.risk_level == "high":
            risk_multiplier = 0.6
        elif risk_metrics.risk_level == "medium":
            risk_multiplier = 0.8
        
        adjusted_size = base_position_size * risk_multiplier * regime_multiplier
        
        return min(adjusted_size, self.max_position_size)
    
    def should_reduce_exposure(self, risk_metrics: RiskMetrics) -> Tuple[bool, str]:
        if risk_metrics.risk_level == "critical":
            return True, "风险水平危急，建议大幅减仓"
        elif risk_metrics.risk_level == "high":
            if risk_metrics.drawdown_risk > self.max_drawdown * 0.8:
                return True, "回撤接近阈值，建议降低仓位"
            if risk_metrics.position_risk > 0.05:
                return True, "持仓亏损风险较高，建议减仓"
        
        return False, ""
    
    def get_risk_report(self, risk_metrics: RiskMetrics) -> str:
        lines = [
            "📊 **风险控制报告**",
            "",
            f"风险等级: {self._get_risk_emoji(risk_metrics.risk_level)} {risk_metrics.risk_level.upper()}",
            f"综合风险分数: {risk_metrics.total_risk:.2%}",
            "",
            "**风险分解:**",
            f"- 持仓风险: {risk_metrics.position_risk:.2%}",
            f"- 回撤风险: {risk_metrics.drawdown_risk:.2%}",
            f"- 集中度风险: {risk_metrics.concentration_risk:.2%}",
            f"- 波动率风险: {risk_metrics.volatility_risk:.2%}",
        ]
        
        if risk_metrics.alerts:
            lines.append("")
            lines.append("**⚠️ 风险警报:**")
            for alert in risk_metrics.alerts:
                lines.append(f"- {alert}")
        
        return "\n".join(lines)
    
    def _get_risk_emoji(self, level: str) -> str:
        return {
            "low": "🟢",
            "medium": "🟡",
            "high": "🟠",
            "critical": "🔴",
        }.get(level, "⚪")
