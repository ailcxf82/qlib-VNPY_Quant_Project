from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
import yaml

if TYPE_CHECKING:
    from monitor.data_source import DataSourceAdapter
    from monitor.risk_controller import RiskController
    from monitor.market_regime import MarketRegimeDetector, MarketRegime
    from monitor.unified_strategy import UnifiedStrategyBase

logger = logging.getLogger(__name__)


@dataclass
class IntegratedBacktestConfig:
    start_date: str
    end_date: str
    initial_capital: float = 1000000.0
    commission_rate: float = 0.0003
    stamp_duty: float = 0.001
    slippage: float = 0.001
    max_position_size: float = 0.15
    max_stocks: int = 10
    min_trade_amount: int = 100
    use_risk_control: bool = True
    use_market_regime: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_capital": self.initial_capital,
            "commission_rate": self.commission_rate,
            "stamp_duty": self.stamp_duty,
            "slippage": self.slippage,
            "max_position_size": self.max_position_size,
            "max_stocks": self.max_stocks,
            "min_trade_amount": self.min_trade_amount,
            "use_risk_control": self.use_risk_control,
            "use_market_regime": self.use_market_regime,
        }


@dataclass
class IntegratedPosition:
    code: str
    name: str
    shares: int
    entry_price: float
    entry_date: str
    current_price: float = 0.0
    highest_price: float = 0.0
    stop_loss_price: float = 0.0
    
    @property
    def market_value(self) -> float:
        return self.shares * self.current_price
    
    @property
    def cost(self) -> float:
        return self.shares * self.entry_price
    
    @property
    def profit_pct(self) -> float:
        if self.entry_price > 0:
            return (self.current_price - self.entry_price) / self.entry_price
        return 0.0
    
    @property
    def profit(self) -> float:
        return self.market_value - self.cost
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "shares": self.shares,
            "entry_price": self.entry_price,
            "entry_date": self.entry_date,
            "current_price": self.current_price,
            "highest_price": self.highest_price,
            "stop_loss_price": self.stop_loss_price,
            "market_value": self.market_value,
            "profit_pct": self.profit_pct,
            "profit": self.profit,
        }


@dataclass
class IntegratedTrade:
    code: str
    name: str
    action: str
    date: str
    price: float
    shares: int
    amount: float
    commission: float
    stamp_duty: float
    slippage_cost: float
    total_cost: float
    reason: str = ""
    regime: str = ""
    risk_alert: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "action": self.action,
            "date": self.date,
            "price": self.price,
            "shares": self.shares,
            "amount": self.amount,
            "commission": self.commission,
            "stamp_duty": self.stamp_duty,
            "slippage_cost": self.slippage_cost,
            "total_cost": self.total_cost,
            "reason": self.reason,
            "regime": self.regime,
            "risk_alert": self.risk_alert,
        }


@dataclass
class IntegratedDailySnapshot:
    date: str
    cash: float
    position_value: float
    total_value: float
    positions: Dict[str, IntegratedPosition]
    daily_return: float = 0.0
    cumulative_return: float = 0.0
    regime: str = "unknown"
    risk_level: str = "low"
    risk_alerts: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "cash": self.cash,
            "position_value": self.position_value,
            "total_value": self.total_value,
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
            "daily_return": self.daily_return,
            "cumulative_return": self.cumulative_return,
            "regime": self.regime,
            "risk_level": self.risk_level,
            "risk_alerts": self.risk_alerts,
        }


class IntegratedBacktestEngine:
    def __init__(
        self,
        config: IntegratedBacktestConfig,
        data_source: Optional["DataSourceAdapter"] = None,
        strategy: Optional["UnifiedStrategyBase"] = None,
        risk_controller: Optional["RiskController"] = None,
        regime_detector: Optional["MarketRegimeDetector"] = None,
    ):
        self.config = config
        self.data_source = data_source
        self.strategy = strategy
        self.risk_controller = risk_controller
        self.regime_detector = regime_detector
        
        self.cash = config.initial_capital
        self.positions: Dict[str, IntegratedPosition] = {}
        self.trades: List[IntegratedTrade] = []
        self.snapshots: List[IntegratedDailySnapshot] = []
        self.equity_curve: List[float] = [config.initial_capital]
        
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._trading_dates: List[str] = []
        self._position_multiplier: float = 1.0
        self._current_regime: Optional[MarketRegime] = None
    
    def load_data(self, codes: List[str]) -> None:
        logger.info(f"Loading data for {len(codes)} stocks...")
        
        if self.data_source is None:
            from monitor.data_source import get_data_source
            self.data_source = get_data_source("akshare")
        
        for code in codes:
            try:
                df = self.data_source.get_daily_data(
                    code=code,
                    start_date=self.config.start_date,
                    end_date=self.config.end_date
                )
                if not df.empty:
                    self._data_cache[code] = df
            except Exception as e:
                logger.warning(f"Failed to load data for {code}: {e}")
        
        logger.info(f"Loaded data for {len(self._data_cache)} stocks")
    
    def load_trading_calendar(self) -> None:
        if self.data_source is None:
            return
        
        self._trading_dates = self.data_source.get_trading_calendar(
            self.config.start_date,
            self.config.end_date
        )
        
        if not self._trading_dates:
            start = pd.to_datetime(self.config.start_date)
            end = pd.to_datetime(self.config.end_date)
            self._trading_dates = [
                d.strftime("%Y-%m-%d") 
                for d in pd.date_range(start, end, freq="B")
            ]
    
    def set_strategy(self, strategy: "UnifiedStrategyBase") -> None:
        self.strategy = strategy
    
    def set_risk_controller(self, risk_controller: "RiskController") -> None:
        self.risk_controller = risk_controller
    
    def set_regime_detector(self, regime_detector: "MarketRegimeDetector") -> None:
        self.regime_detector = regime_detector
    
    def run(self, codes: Optional[List[str]] = None) -> Dict[str, Any]:
        if self.strategy is None:
            raise ValueError("Strategy not set. Call set_strategy() first.")
        
        if codes:
            self.load_data(codes)
        
        if not self._trading_dates:
            self.load_trading_calendar()
        
        logger.info(f"Running integrated backtest from {self.config.start_date} to {self.config.end_date}")
        logger.info(f"Initial capital: {self.config.initial_capital:,.0f}")
        logger.info(f"Risk control: {'enabled' if self.config.use_risk_control else 'disabled'}")
        logger.info(f"Market regime: {'enabled' if self.config.use_market_regime else 'disabled'}")
        
        for date in self._trading_dates:
            self._process_day(date)
        
        return self._generate_results()
    
    def _process_day(self, date: str) -> None:
        self._update_positions(date)
        
        regime_info = self._get_regime_info(date)
        self._position_multiplier = regime_info.get("position_multiplier", 1.0)
        
        risk_alerts = self._check_risk_alerts(date)
        
        signals = self.strategy.generate_signals(
            date=date,
            data=self._data_cache,
            positions={k: v.to_dict() for k, v in self.positions.items()}
        )
        
        signals = self._apply_risk_filters(signals, risk_alerts)
        
        sell_signals = {k: v for k, v in signals.items() if v == "sell"}
        for code in sell_signals:
            if code in self.positions:
                self._execute_sell(code, date, "Signal")
        
        buy_signals = {k: v for k, v in signals.items() if v == "buy"}
        for code in buy_signals:
            if code not in self.positions:
                self._execute_buy(code, date, "Signal")
        
        self._record_snapshot(date, regime_info, risk_alerts)
    
    def _get_regime_info(self, date: str) -> Dict[str, Any]:
        if not self.config.use_market_regime or self.regime_detector is None:
            return {"regime": "unknown", "position_multiplier": 1.0}
        
        try:
            regime = self.regime_detector.detect_regime()
            self._current_regime = regime
            position_multiplier = self.regime_detector.get_position_size_multiplier(regime)
            return {
                "regime": regime.regime,
                "position_multiplier": position_multiplier,
                "trend_strength": regime.trend_strength,
                "volatility": regime.volatility,
            }
        except Exception as e:
            logger.debug(f"Failed to get regime info: {e}")
            return {"regime": "unknown", "position_multiplier": 1.0}
    
    def _check_risk_alerts(self, current_date: str) -> List[Dict[str, Any]]:
        if not self.config.use_risk_control or self.risk_controller is None:
            return []
        
        alerts = []
        current_dt = pd.to_datetime(current_date)
        
        for code, pos in self.positions.items():
            entry_dt = pd.to_datetime(pos.entry_date)
            holding_days = (current_dt - entry_dt).days
            
            stop_level = self.risk_controller.calculate_stop_loss_levels(
                code=code,
                name=pos.name,
                entry_price=pos.entry_price,
                current_price=pos.current_price,
                holding_days=holding_days,
                highest_price=pos.highest_price,
            )
            
            if stop_level.should_stop:
                alerts.append({
                    "code": code,
                    "type": stop_level.stop_type,
                    "reason": stop_level.stop_reason,
                    "priority": self._get_stop_priority(stop_level.stop_type),
                })
        
        return alerts
    
    def _get_stop_priority(self, stop_type: str) -> int:
        priority_map = {
            "hard_stop": 100,
            "take_profit": 80,
            "trailing": 60,
            "time": 40,
            "initial": 20,
        }
        return priority_map.get(stop_type, 10)
    
    def _apply_risk_filters(
        self, 
        signals: Dict[str, str], 
        risk_alerts: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        filtered_signals = signals.copy()
        
        for alert in risk_alerts:
            code = alert["code"]
            if alert["type"] == "stop_loss":
                filtered_signals[code] = "sell"
            elif alert["type"] == "take_profit":
                filtered_signals[code] = "sell"
        
        return filtered_signals
    
    def _update_positions(self, date: str) -> None:
        current_date = pd.to_datetime(date)
        
        for code, pos in list(self.positions.items()):
            if code not in self._data_cache:
                continue
            
            df = self._data_cache[code]
            df_day = df[df["date"] == current_date]
            
            if df_day.empty:
                df_before = df[df["date"] <= current_date]
                if not df_before.empty:
                    pos.current_price = float(df_before.iloc[-1]["close"])
            else:
                pos.current_price = float(df_day.iloc[-1]["close"])
            
            pos.highest_price = max(pos.highest_price, pos.current_price)
    
    def _execute_buy(self, code: str, date: str, reason: str) -> Optional[IntegratedTrade]:
        if code not in self._data_cache:
            return None
        
        df = self._data_cache[code]
        current_date = pd.to_datetime(date)
        df_day = df[df["date"] == current_date]
        
        if df_day.empty:
            return None
        
        price = float(df_day.iloc[-1]["close"])
        actual_price = price * (1 + self.config.slippage)
        
        total_capital = self.cash + sum(p.market_value for p in self.positions.values())
        
        base_position_size = self.strategy.get_position_size(code, price, total_capital)
        
        if self.risk_controller is not None and self.config.use_risk_control:
            position_sizing = self.risk_controller.calculate_position_size(
                code=code,
                name=code,
                entry_price=actual_price,
                stop_price=actual_price * 0.92,
                total_capital=total_capital,
            )
            shares = min(base_position_size, position_sizing.recommended_shares)
        else:
            shares = base_position_size
        
        shares = int(shares * self._position_multiplier)
        
        if shares < self.config.min_trade_amount:
            return None
        
        amount = shares * actual_price
        commission = amount * self.config.commission_rate
        total_cost = amount + commission
        
        if total_cost > self.cash:
            shares = int(self.cash / actual_price / 100) * 100
            if shares < self.config.min_trade_amount:
                return None
            amount = shares * actual_price
            commission = amount * self.config.commission_rate
            total_cost = amount + commission
        
        self.cash -= total_cost
        
        stock_list = self.data_source.get_stock_list()
        name = code
        if not stock_list.empty and code in stock_list["code"].values:
            name = stock_list[stock_list["code"] == code]["name"].iloc[0]
        
        position = IntegratedPosition(
            code=code,
            name=name,
            shares=shares,
            entry_price=actual_price,
            entry_date=date,
            current_price=actual_price,
            highest_price=actual_price,
        )
        self.positions[code] = position
        
        regime_name = self._current_regime.regime if self._current_regime else "unknown"
        
        trade = IntegratedTrade(
            code=code,
            name=name,
            action="buy",
            date=date,
            price=actual_price,
            shares=shares,
            amount=amount,
            commission=commission,
            stamp_duty=0,
            slippage_cost=amount * self.config.slippage,
            total_cost=total_cost,
            reason=reason,
            regime=regime_name,
        )
        self.trades.append(trade)
        
        logger.debug(f"BUY {code} {shares} shares @ {actual_price:.2f}")
        return trade
    
    def _execute_sell(self, code: str, date: str, reason: str) -> Optional[IntegratedTrade]:
        if code not in self.positions:
            return None
        
        position = self.positions[code]
        
        if code in self._data_cache:
            df = self._data_cache[code]
            current_date = pd.to_datetime(date)
            df_day = df[df["date"] == current_date]
            
            if not df_day.empty:
                position.current_price = float(df_day.iloc[-1]["close"])
        
        price = position.current_price
        actual_price = price * (1 - self.config.slippage)
        
        amount = position.shares * actual_price
        commission = amount * self.config.commission_rate
        stamp_duty = amount * self.config.stamp_duty
        total_proceeds = amount - commission - stamp_duty
        
        self.cash += total_proceeds
        
        regime_name = self._current_regime.regime if self._current_regime else "unknown"
        
        trade = IntegratedTrade(
            code=code,
            name=position.name,
            action="sell",
            date=date,
            price=actual_price,
            shares=position.shares,
            amount=amount,
            commission=commission,
            stamp_duty=stamp_duty,
            slippage_cost=position.shares * price * self.config.slippage,
            total_cost=total_proceeds,
            reason=reason,
            regime=regime_name,
        )
        self.trades.append(trade)
        
        del self.positions[code]
        
        logger.debug(f"SELL {code} {position.shares} shares @ {actual_price:.2f}, P&L: {position.profit_pct:.1%}")
        return trade
    
    def _record_snapshot(
        self, 
        date: str, 
        regime_info: Dict[str, Any],
        risk_alerts: List[Dict[str, Any]]
    ) -> None:
        position_value = sum(p.market_value for p in self.positions.values())
        total_value = self.cash + position_value
        
        self.equity_curve.append(total_value)
        
        daily_return = 0.0
        if len(self.equity_curve) > 1 and self.equity_curve[-2] > 0:
            daily_return = (total_value - self.equity_curve[-2]) / self.equity_curve[-2]
        
        cumulative_return = (total_value - self.config.initial_capital) / self.config.initial_capital
        
        risk_level = "low"
        if risk_alerts:
            high_priority = [a for a in risk_alerts if a.get("priority", 0) >= 80]
            if high_priority:
                risk_level = "high"
            else:
                risk_level = "medium"
        
        snapshot = IntegratedDailySnapshot(
            date=date,
            cash=self.cash,
            position_value=position_value,
            total_value=total_value,
            positions=dict(self.positions),
            daily_return=daily_return,
            cumulative_return=cumulative_return,
            regime=regime_info.get("regime", "unknown"),
            risk_level=risk_level,
            risk_alerts=[a["reason"] for a in risk_alerts],
        )
        self.snapshots.append(snapshot)
    
    def _generate_results(self) -> Dict[str, Any]:
        from monitor.performance import PerformanceAnalyzer
        
        analyzer = PerformanceAnalyzer(
            equity_curve=self.equity_curve,
            trades=self.trades,
            config=self.config,
        )
        
        return {
            "config": self.config.to_dict(),
            "performance": analyzer.analyze(),
            "trades": [t.to_dict() for t in self.trades],
            "snapshots": [s.to_dict() for s in self.snapshots],
            "final_positions": {k: v.to_dict() for k, v in self.positions.items()},
        }
    
    def get_equity_curve_df(self) -> pd.DataFrame:
        if not self.snapshots:
            return pd.DataFrame()
        
        return pd.DataFrame([s.to_dict() for s in self.snapshots])
    
    def get_trades_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame([t.to_dict() for t in self.trades])


def run_integrated_backtest(
    strategy_config: Dict[str, Any],
    backtest_config: Dict[str, Any],
    codes: List[str],
    use_risk_control: bool = True,
    use_market_regime: bool = True,
) -> Dict[str, Any]:
    from monitor.data_source import get_data_source
    from monitor.unified_strategy import create_strategy
    from monitor.risk_controller import RiskController
    from monitor.market_regime import MarketRegimeDetector
    
    bt_config = IntegratedBacktestConfig(
        start_date=backtest_config["start_date"],
        end_date=backtest_config["end_date"],
        initial_capital=backtest_config.get("initial_capital", 1000000),
        commission_rate=backtest_config.get("commission_rate", 0.0003),
        stamp_duty=backtest_config.get("stamp_duty", 0.001),
        slippage=backtest_config.get("slippage", 0.001),
        max_position_size=backtest_config.get("max_position_size", 0.15),
        max_stocks=backtest_config.get("max_stocks", 10),
        min_trade_amount=backtest_config.get("min_trade_amount", 100),
        use_risk_control=use_risk_control,
        use_market_regime=use_market_regime,
    )
    
    data_source = get_data_source("akshare")
    
    strategy = create_strategy(
        strategy_config.get("name", "momentum"),
        strategy_config,
        data_source
    )
    
    risk_controller = None
    if use_risk_control:
        risk_controller = RiskController()
    
    regime_detector = None
    if use_market_regime:
        regime_detector = MarketRegimeDetector()
    
    engine = IntegratedBacktestEngine(
        config=bt_config,
        data_source=data_source,
        strategy=strategy,
        risk_controller=risk_controller,
        regime_detector=regime_detector,
    )
    
    return engine.run(codes)
