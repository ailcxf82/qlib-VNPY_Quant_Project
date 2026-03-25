from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml

from monitor.data_source import DataSourceAdapter, AkshareAdapter, get_data_source

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    start_date: str
    end_date: str
    initial_capital: float = 1000000.0
    commission_rate: float = 0.0003
    stamp_duty: float = 0.001
    slippage: float = 0.001
    max_position_size: float = 0.15
    max_stocks: int = 10
    min_trade_amount: int = 100
    
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
        }


@dataclass
class Position:
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
class Trade:
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
        }


@dataclass
class DailySnapshot:
    date: str
    cash: float
    position_value: float
    total_value: float
    positions: Dict[str, Position]
    daily_return: float = 0.0
    cumulative_return: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "cash": self.cash,
            "position_value": self.position_value,
            "total_value": self.total_value,
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
            "daily_return": self.daily_return,
            "cumulative_return": self.cumulative_return,
        }


class StrategyBase:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get("name", "BaseStrategy")
    
    def generate_signals(
        self, 
        date: str, 
        data: Dict[str, pd.DataFrame],
        positions: Dict[str, Position]
    ) -> Dict[str, str]:
        return {}
    
    def get_position_size(
        self, 
        code: str, 
        price: float, 
        total_capital: float
    ) -> int:
        return 0


class MomentumStrategy(StrategyBase):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.lookback_period = config.get("lookback_period", 20)
        self.top_k = config.get("top_k", 10)
        self.min_momentum = config.get("min_momentum", 0.05)
        self.stop_loss = config.get("stop_loss", -0.08)
        self.take_profit = config.get("take_profit", 0.15)
        self.max_holding_days = config.get("max_holding_days", 10)
    
    def generate_signals(
        self, 
        date: str, 
        data: Dict[str, pd.DataFrame],
        positions: Dict[str, Position]
    ) -> Dict[str, str]:
        signals = {}
        
        for code, pos in positions.items():
            if code not in data:
                continue
            
            df = data[code]
            if df.empty:
                continue
            
            current_date = pd.to_datetime(date)
            df_before = df[df["date"] <= current_date]
            
            if df_before.empty:
                continue
            
            current_price = float(df_before.iloc[-1]["close"])
            pos.current_price = current_price
            pos.highest_price = max(pos.highest_price, current_price)
            
            holding_days = (current_date - pd.to_datetime(pos.entry_date)).days
            
            profit_pct = (current_price - pos.entry_price) / pos.entry_price
            
            if profit_pct <= self.stop_loss:
                signals[code] = "sell"
            elif profit_pct >= self.take_profit:
                signals[code] = "sell"
            elif holding_days >= self.max_holding_days:
                signals[code] = "sell"
        
        momentum_scores = {}
        for code, df in data.items():
            if df.empty:
                continue
            
            current_date = pd.to_datetime(date)
            df_before = df[df["date"] <= current_date]
            
            if len(df_before) < self.lookback_period:
                continue
            
            if code in positions:
                continue
            
            close = df_before["close"].astype(float)
            momentum = (close.iloc[-1] - close.iloc[-self.lookback_period]) / close.iloc[-self.lookback_period]
            
            if momentum >= self.min_momentum:
                momentum_scores[code] = momentum
        
        sorted_stocks = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        
        for code, _ in sorted_stocks[:self.top_k]:
            if len(positions) + len([s for s in signals.values() if s == "buy"]) >= self.config.get("max_stocks", 10):
                break
            signals[code] = "buy"
        
        return signals
    
    def get_position_size(
        self, 
        code: str, 
        price: float, 
        total_capital: float
    ) -> int:
        position_value = total_capital * self.config.get("position_size", 0.10)
        shares = int(position_value / price / 100) * 100
        return max(shares, 100)


class MeanReversionStrategy(StrategyBase):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.lookback_period = config.get("lookback_period", 20)
        self.oversold_threshold = config.get("oversold_threshold", -0.15)
        self.overbought_threshold = config.get("overbought_threshold", 0.15)
        self.top_k = config.get("top_k", 10)
    
    def generate_signals(
        self, 
        date: str, 
        data: Dict[str, pd.DataFrame],
        positions: Dict[str, Position]
    ) -> Dict[str, str]:
        signals = {}
        
        for code, pos in positions.items():
            if code not in data:
                continue
            
            df = data[code]
            if df.empty:
                continue
            
            current_date = pd.to_datetime(date)
            df_before = df[df["date"] <= current_date]
            
            if len(df_before) < self.lookback_period:
                continue
            
            close = df_before["close"].astype(float)
            ma = close.rolling(self.lookback_period).mean().iloc[-1]
            current_price = float(df_before.iloc[-1]["close"])
            
            deviation = (current_price - ma) / ma
            
            if deviation > self.overbought_threshold:
                signals[code] = "sell"
        
        deviation_scores = {}
        for code, df in data.items():
            if df.empty:
                continue
            
            current_date = pd.to_datetime(date)
            df_before = df[df["date"] <= current_date]
            
            if len(df_before) < self.lookback_period:
                continue
            
            if code in positions:
                continue
            
            close = df_before["close"].astype(float)
            ma = close.rolling(self.lookback_period).mean().iloc[-1]
            current_price = float(df_before.iloc[-1]["close"])
            
            deviation = (current_price - ma) / ma
            
            if deviation < self.oversold_threshold:
                deviation_scores[code] = deviation
        
        sorted_stocks = sorted(deviation_scores.items(), key=lambda x: x[0])
        
        for code, _ in sorted_stocks[:self.top_k]:
            if len(positions) + len([s for s in signals.values() if s == "buy"]) >= self.config.get("max_stocks", 10):
                break
            signals[code] = "buy"
        
        return signals
    
    def get_position_size(
        self, 
        code: str, 
        price: float, 
        total_capital: float
    ) -> int:
        position_value = total_capital * self.config.get("position_size", 0.10)
        shares = int(position_value / price / 100) * 100
        return max(shares, 100)


class BacktestEngine:
    def __init__(
        self,
        config: BacktestConfig,
        data_source: Optional[DataSourceAdapter] = None,
        strategy: Optional[StrategyBase] = None,
    ):
        self.config = config
        self.data_source = data_source or get_data_source("akshare")
        self.strategy = strategy
        
        self.cash = config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.snapshots: List[DailySnapshot] = []
        self.equity_curve: List[float] = [config.initial_capital]
        
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._trading_dates: List[str] = []
    
    def load_data(self, codes: List[str]) -> None:
        logger.info(f"Loading data for {len(codes)} stocks...")
        
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
    
    def set_strategy(self, strategy: StrategyBase) -> None:
        self.strategy = strategy
    
    def run(self, codes: Optional[List[str]] = None) -> Dict[str, Any]:
        if self.strategy is None:
            raise ValueError("Strategy not set. Call set_strategy() first.")
        
        if codes:
            self.load_data(codes)
        
        if not self._trading_dates:
            self.load_trading_calendar()
        
        logger.info(f"Running backtest from {self.config.start_date} to {self.config.end_date}")
        logger.info(f"Initial capital: {self.config.initial_capital:,.0f}")
        
        for date in self._trading_dates:
            self._process_day(date)
        
        return self._generate_results()
    
    def _process_day(self, date: str) -> None:
        self._update_positions(date)
        
        signals = self.strategy.generate_signals(
            date=date,
            data=self._data_cache,
            positions=self.positions
        )
        
        sell_signals = {k: v for k, v in signals.items() if v == "sell"}
        for code in sell_signals:
            if code in self.positions:
                self._execute_sell(code, date, "Signal")
        
        buy_signals = {k: v for k, v in signals.items() if v == "buy"}
        for code in buy_signals:
            if code not in self.positions:
                self._execute_buy(code, date, "Signal")
        
        self._record_snapshot(date)
    
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
    
    def _execute_buy(self, code: str, date: str, reason: str) -> Optional[Trade]:
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
        shares = self.strategy.get_position_size(code, price, total_capital)
        
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
        name = stock_list[stock_list["code"] == code]["name"].iloc[0] if not stock_list.empty and code in stock_list["code"].values else code
        
        position = Position(
            code=code,
            name=name,
            shares=shares,
            entry_price=actual_price,
            entry_date=date,
            current_price=actual_price,
            highest_price=actual_price,
        )
        self.positions[code] = position
        
        trade = Trade(
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
        )
        self.trades.append(trade)
        
        logger.debug(f"BUY {code} {shares} shares @ {actual_price:.2f}")
        return trade
    
    def _execute_sell(self, code: str, date: str, reason: str) -> Optional[Trade]:
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
        
        trade = Trade(
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
        )
        self.trades.append(trade)
        
        del self.positions[code]
        
        logger.debug(f"SELL {code} {position.shares} shares @ {actual_price:.2f}, P&L: {position.profit_pct:.1%}")
        return trade
    
    def _record_snapshot(self, date: str) -> None:
        position_value = sum(p.market_value for p in self.positions.values())
        total_value = self.cash + position_value
        
        self.equity_curve.append(total_value)
        
        daily_return = 0.0
        if len(self.equity_curve) > 1 and self.equity_curve[-2] > 0:
            daily_return = (total_value - self.equity_curve[-2]) / self.equity_curve[-2]
        
        cumulative_return = (total_value - self.config.initial_capital) / self.config.initial_capital
        
        snapshot = DailySnapshot(
            date=date,
            cash=self.cash,
            position_value=position_value,
            total_value=total_value,
            positions=dict(self.positions),
            daily_return=daily_return,
            cumulative_return=cumulative_return,
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


def run_backtest(
    strategy_config: Dict[str, Any],
    backtest_config: Dict[str, Any],
    codes: List[str],
) -> Dict[str, Any]:
    bt_config = BacktestConfig(**backtest_config)
    
    strategy_name = strategy_config.get("name", "momentum")
    if strategy_name == "momentum":
        strategy = MomentumStrategy(strategy_config)
    elif strategy_name == "mean_reversion":
        strategy = MeanReversionStrategy(strategy_config)
    else:
        strategy = MomentumStrategy(strategy_config)
    
    engine = BacktestEngine(config=bt_config, strategy=strategy)
    
    return engine.run(codes)
