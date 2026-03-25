"""
Backtest Engine - 策略回测引擎

提供完整的策略回测功能，包括交易模拟、绩效分析、日志记录。
"""

from __future__ import annotations

import json
import logging
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from monitor.strategy_library.strategy_base import StrategyBase, Signal

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    start_date: str
    end_date: str
    initial_capital: float = 1000000.0
    commission_rate: float = 0.0003
    stamp_duty: float = 0.001
    slippage: float = 0.001
    benchmark: str = "000300.SH"
    max_positions: int = 10
    position_size_method: str = "equal_weight"
    min_trade_amount: float = 1000.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_capital": self.initial_capital,
            "commission_rate": self.commission_rate,
            "stamp_duty": self.stamp_duty,
            "slippage": self.slippage,
            "benchmark": self.benchmark,
            "max_positions": self.max_positions,
            "position_size_method": self.position_size_method,
            "min_trade_amount": self.min_trade_amount,
        }


@dataclass
class TradeLog:
    trade_id: int
    date: str
    code: str
    name: str
    action: str
    price: float
    shares: int
    amount: float
    commission: float
    stamp_duty: float
    slippage_cost: float
    reason: str
    strategy_signal: Dict[str, Any]
    portfolio_value: float
    cash: float
    position_value: float
    signal_confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "date": self.date,
            "code": self.code,
            "name": self.name,
            "action": self.action,
            "price": self.price,
            "shares": self.shares,
            "amount": self.amount,
            "commission": self.commission,
            "stamp_duty": self.stamp_duty,
            "slippage_cost": self.slippage_cost,
            "reason": self.reason,
            "strategy_signal": self.strategy_signal,
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "position_value": self.position_value,
            "signal_confidence": self.signal_confidence,
        }


@dataclass
class DailySnapshot:
    date: str
    portfolio_value: float
    cash: float
    position_value: float
    daily_return: float
    cumulative_return: float
    positions: Dict[str, Dict[str, Any]]
    benchmark_value: float = 0.0
    benchmark_return: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "position_value": self.position_value,
            "daily_return": self.daily_return,
            "cumulative_return": self.cumulative_return,
            "positions": self.positions,
            "benchmark_value": self.benchmark_value,
            "benchmark_return": self.benchmark_return,
        }


@dataclass
class BacktestResult:
    backtest_id: str
    strategy_id: str
    strategy_name: str
    config: BacktestConfig
    parameters: Dict[str, Any]
    
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    
    total_return: float
    annual_return: float
    benchmark_return: float
    excess_return: float
    
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    
    trades: List[TradeLog] = field(default_factory=list)
    daily_snapshots: List[DailySnapshot] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    monthly_returns: Dict[str, float] = field(default_factory=dict)
    
    execution_time: float = 0.0
    created_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "backtest_id": self.backtest_id,
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "config": self.config.to_dict(),
            "parameters": self.parameters,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_capital": self.initial_capital,
            "final_capital": self.final_capital,
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "benchmark_return": self.benchmark_return,
            "excess_return": self.excess_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "execution_time": self.execution_time,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }
    
    def get_params_hash(self) -> str:
        params_str = json.dumps(self.parameters, sort_keys=True)
        return hashlib.md5(params_str.encode()).hexdigest()[:8]


class TradeSimulator:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.cash = config.initial_capital
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.trades: List[TradeLog] = []
        self.trade_counter = 0
        self.total_commission = 0.0
        self.total_stamp_duty = 0.0
        self.total_slippage = 0.0
    
    def reset(self):
        self.cash = self.config.initial_capital
        self.positions = {}
        self.trades = []
        self.trade_counter = 0
        self.total_commission = 0.0
        self.total_stamp_duty = 0.0
        self.total_slippage = 0.0
    
    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        position_value = sum(
            pos["shares"] * prices.get(code, pos["avg_cost"])
            for code, pos in self.positions.items()
        )
        return self.cash + position_value
    
    def execute_buy(
        self,
        code: str,
        name: str,
        price: float,
        shares: int,
        date: str,
        reason: str,
        signal: Dict[str, Any],
        confidence: float = 0.0,
    ) -> Optional[TradeLog]:
        actual_price = price * (1 + self.config.slippage)
        amount = actual_price * shares
        commission = max(amount * self.config.commission_rate, 5.0)
        total_cost = amount + commission
        
        if total_cost > self.cash:
            affordable_shares = int((self.cash - commission) / actual_price / 100) * 100
            if affordable_shares < 100:
                return None
            shares = affordable_shares
            amount = actual_price * shares
            commission = max(amount * self.config.commission_rate, 5.0)
            total_cost = amount + commission
        
        self.cash -= total_cost
        self.total_commission += commission
        self.total_slippage += price * self.config.slippage * shares
        
        if code in self.positions:
            old_pos = self.positions[code]
            total_shares = old_pos["shares"] + shares
            total_cost_basis = old_pos["avg_cost"] * old_pos["shares"] + amount
            avg_cost = total_cost_basis / total_shares
            self.positions[code] = {
                "shares": total_shares,
                "avg_cost": avg_cost,
                "name": name,
                "entry_date": old_pos["entry_date"],
                "highest_price": max(old_pos.get("highest_price", actual_price), actual_price),
            }
        else:
            self.positions[code] = {
                "shares": shares,
                "avg_cost": actual_price,
                "name": name,
                "entry_date": date,
                "highest_price": actual_price,
            }
        
        self.trade_counter += 1
        trade = TradeLog(
            trade_id=self.trade_counter,
            date=date,
            code=code,
            name=name,
            action="buy",
            price=actual_price,
            shares=shares,
            amount=amount,
            commission=commission,
            stamp_duty=0.0,
            slippage_cost=price * self.config.slippage * shares,
            reason=reason,
            strategy_signal=signal,
            portfolio_value=self.get_portfolio_value({code: actual_price}),
            cash=self.cash,
            position_value=self.get_portfolio_value({code: actual_price}) - self.cash,
            signal_confidence=confidence,
        )
        self.trades.append(trade)
        
        logger.debug(f"[{date}] 买入 {code} {shares}股 @ {actual_price:.2f}, 原因: {reason}")
        return trade
    
    def execute_sell(
        self,
        code: str,
        price: float,
        shares: Optional[int],
        date: str,
        reason: str,
        signal: Dict[str, Any],
        confidence: float = 0.0,
    ) -> Optional[TradeLog]:
        if code not in self.positions:
            return None
        
        pos = self.positions[code]
        name = pos["name"]
        
        if shares is None or shares > pos["shares"]:
            shares = pos["shares"]
        
        actual_price = price * (1 - self.config.slippage)
        amount = actual_price * shares
        commission = max(amount * self.config.commission_rate, 5.0)
        stamp_duty = amount * self.config.stamp_duty
        net_amount = amount - commission - stamp_duty
        
        self.cash += net_amount
        self.total_commission += commission
        self.total_stamp_duty += stamp_duty
        self.total_slippage += price * self.config.slippage * shares
        
        pos["shares"] -= shares
        if pos["shares"] <= 0:
            del self.positions[code]
        
        self.trade_counter += 1
        trade = TradeLog(
            trade_id=self.trade_counter,
            date=date,
            code=code,
            name=name,
            action="sell",
            price=actual_price,
            shares=shares,
            amount=amount,
            commission=commission,
            stamp_duty=stamp_duty,
            slippage_cost=price * self.config.slippage * shares,
            reason=reason,
            strategy_signal=signal,
            portfolio_value=self.get_portfolio_value({code: actual_price}),
            cash=self.cash,
            position_value=self.get_portfolio_value({code: actual_price}) - self.cash,
            signal_confidence=confidence,
        )
        self.trades.append(trade)
        
        logger.debug(f"[{date}] 卖出 {code} {shares}股 @ {actual_price:.2f}, 原因: {reason}")
        return trade


class BacktestLogger:
    DEFAULT_LOG_DIR = Path("data/strategy_library/backtest_results")
    
    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = Path(log_dir) if log_dir else self.DEFAULT_LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def save_result(self, result: BacktestResult, strategy_id: str) -> Tuple[str, str, str]:
        strategy_dir = self.log_dir / strategy_id
        strategy_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"bt_{timestamp}"
        
        json_path = strategy_dir / f"{base_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        
        trades_path = strategy_dir / f"{base_name}_trades.csv"
        if result.trades:
            trades_df = pd.DataFrame([t.to_dict() for t in result.trades])
            trades_df.to_csv(trades_path, index=False, encoding="utf-8-sig")
        
        snapshots_path = strategy_dir / f"{base_name}_snapshots.csv"
        if result.daily_snapshots:
            snapshots_df = pd.DataFrame([s.to_dict() for s in result.daily_snapshots])
            snapshots_df.to_csv(snapshots_path, index=False, encoding="utf-8-sig")
        
        logger.info(f"回测结果已保存: {json_path}")
        return str(json_path), str(trades_path), str(snapshots_path)
    
    def load_result(self, path: str) -> Optional[BacktestResult]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            config = BacktestConfig(**data["config"])
            
            result = BacktestResult(
                backtest_id=data["backtest_id"],
                strategy_id=data["strategy_id"],
                strategy_name=data["strategy_name"],
                config=config,
                parameters=data["parameters"],
                start_date=data["start_date"],
                end_date=data["end_date"],
                initial_capital=data["initial_capital"],
                final_capital=data["final_capital"],
                total_return=data["total_return"],
                annual_return=data["annual_return"],
                benchmark_return=data.get("benchmark_return", 0.0),
                excess_return=data.get("excess_return", 0.0),
                sharpe_ratio=data["sharpe_ratio"],
                sortino_ratio=data.get("sortino_ratio", 0.0),
                max_drawdown=data["max_drawdown"],
                max_drawdown_duration=data.get("max_drawdown_duration", 0),
                win_rate=data["win_rate"],
                profit_factor=data.get("profit_factor", 0.0),
                total_trades=data["total_trades"],
                winning_trades=data.get("winning_trades", 0),
                losing_trades=data.get("losing_trades", 0),
                execution_time=data.get("execution_time", 0.0),
                created_at=data.get("created_at", ""),
                metadata=data.get("metadata", {}),
            )
            return result
        except Exception as e:
            logger.error(f"加载回测结果失败: {e}")
            return None
    
    def get_latest_result(self, strategy_id: str) -> Optional[BacktestResult]:
        strategy_dir = self.log_dir / strategy_id
        if not strategy_dir.exists():
            return None
        
        json_files = sorted(strategy_dir.glob("bt_*.json"), reverse=True)
        if not json_files:
            return None
        
        return self.load_result(str(json_files[0]))
    
    def list_results(self, strategy_id: str) -> List[Dict[str, Any]]:
        strategy_dir = self.log_dir / strategy_id
        if not strategy_dir.exists():
            return []
        
        results = []
        for json_file in sorted(strategy_dir.glob("bt_*.json"), reverse=True):
            result = self.load_result(str(json_file))
            if result:
                results.append({
                    "path": str(json_file),
                    "backtest_id": result.backtest_id,
                    "created_at": result.created_at,
                    "total_return": result.total_return,
                    "sharpe_ratio": result.sharpe_ratio,
                })
        
        return results


class BacktestEngine:
    def __init__(
        self,
        config: BacktestConfig,
        data_loader: Optional[Callable] = None,
    ):
        self.config = config
        self.data_loader = data_loader
        self.simulator = TradeSimulator(config)
        self.logger = BacktestLogger()
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._benchmark_data: Optional[pd.DataFrame] = None
    
    def set_data_loader(self, loader: Callable):
        self.data_loader = loader
    
    def load_data(self, codes: List[str]) -> Dict[str, pd.DataFrame]:
        if self.data_loader is None:
            logger.warning("数据加载器未设置")
            return {}
        
        data = {}
        for code in codes:
            if code in self._data_cache:
                data[code] = self._data_cache[code]
            else:
                try:
                    df = self.data_loader(code, self.config.start_date, self.config.end_date)
                    if df is not None and not df.empty:
                        self._data_cache[code] = df
                        data[code] = df
                except Exception as e:
                    logger.warning(f"加载数据失败 {code}: {e}")
        
        return data
    
    def load_benchmark(self) -> pd.DataFrame:
        if self._benchmark_data is not None:
            return self._benchmark_data
        
        if self.data_loader is None:
            return pd.DataFrame()
        
        try:
            self._benchmark_data = self.data_loader(
                self.config.benchmark,
                self.config.start_date,
                self.config.end_date
            )
            return self._benchmark_data
        except Exception as e:
            logger.warning(f"加载基准数据失败: {e}")
            return pd.DataFrame()
    
    def run(
        self,
        strategy: "StrategyBase",
        data: Dict[str, pd.DataFrame],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> BacktestResult:
        import time
        start_time = time.time()
        
        self.simulator.reset()
        
        if parameters:
            strategy.set_parameters(parameters)
        
        strategy_id = strategy.strategy_id
        strategy_name = strategy.strategy_name
        
        backtest_id = f"bt_{strategy_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"开始回测: {strategy_name} ({strategy_id})")
        logger.info(f"回测区间: {self.config.start_date} ~ {self.config.end_date}")
        
        all_dates = set()
        for df in data.values():
            if "date" in df.columns:
                dates = df["date"].astype(str).tolist()
            elif df.index.name == "date" or isinstance(df.index, pd.DatetimeIndex):
                dates = df.index.astype(str).tolist()
            else:
                dates = df.index.astype(str).tolist()
            all_dates.update(dates)
        
        trading_dates = sorted([d for d in all_dates if d >= self.config.start_date and d <= self.config.end_date])
        
        if not trading_dates:
            logger.error("没有有效的交易日期")
            return self._create_empty_result(backtest_id, strategy_id, strategy_name, parameters or {})
        
        daily_snapshots: List[DailySnapshot] = []
        prev_portfolio_value = self.config.initial_capital
        cumulative_return = 0.0
        
        benchmark_data = self.load_benchmark()
        benchmark_values = {}
        if not benchmark_data.empty:
            if "date" in benchmark_data.columns:
                benchmark_data = benchmark_data.set_index("date")
            for date in trading_dates:
                if date in benchmark_data.index:
                    benchmark_values[date] = float(benchmark_data.loc[date, "close"])
        
        initial_benchmark = list(benchmark_values.values())[0] if benchmark_values else 100
        
        for i, date in enumerate(trading_dates):
            current_data = {}
            current_prices = {}
            
            for code, df in data.items():
                if "date" in df.columns:
                    df_date = df[df["date"].astype(str) <= date]
                else:
                    df_date = df[df.index.astype(str) <= date]
                
                if not df_date.empty:
                    current_data[code] = df_date
                    last_row = df_date.iloc[-1]
                    current_prices[code] = float(last_row.get("close", last_row.get("price", 0)))
            
            positions_dict = {
                code: {
                    "shares": pos["shares"],
                    "avg_cost": pos["avg_cost"],
                    "name": pos["name"],
                    "entry_date": pos["entry_date"],
                }
                for code, pos in self.simulator.positions.items()
            }
            
            signals = strategy.generate_signals(
                date=date,
                data=current_data,
                positions=positions_dict,
                total_capital=self.simulator.cash,
            )
            
            if signals:
                sell_signals = [s for s in signals if s.action == "sell"]
                for sig in sell_signals:
                    if sig.code in self.simulator.positions:
                        self.simulator.execute_sell(
                            code=sig.code,
                            price=current_prices.get(sig.code, sig.price),
                            shares=sig.shares if sig.shares > 0 else None,
                            date=date,
                            reason=sig.reason,
                            signal={"confidence": sig.confidence, "reason": sig.reason},
                            confidence=sig.confidence,
                        )
                
                buy_signals = [s for s in signals if s.action == "buy"]
                affordable_count = self.config.max_positions - len(self.simulator.positions)
                
                available_cash = self.simulator.cash * 0.95
                position_size = available_cash / affordable_count if affordable_count > 0 else available_cash
                
                for sig in buy_signals[:affordable_count]:
                    price = current_prices.get(sig.code, sig.price)
                    if price <= 0:
                        continue
                    
                    if sig.shares > 0:
                        shares = sig.shares
                    else:
                        shares = int(position_size / price / 100) * 100
                        if shares < 100:
                            shares = 100
                    
                    self.simulator.execute_buy(
                        code=sig.code,
                        name=sig.name,
                        price=price,
                        shares=shares,
                        date=date,
                        reason=sig.reason,
                        signal={"confidence": sig.confidence, "reason": sig.reason},
                        confidence=sig.confidence,
                    )
            
            portfolio_value = self.simulator.get_portfolio_value(current_prices)
            daily_return = (portfolio_value - prev_portfolio_value) / prev_portfolio_value if prev_portfolio_value > 0 else 0
            cumulative_return = (portfolio_value - self.config.initial_capital) / self.config.initial_capital
            
            benchmark_value = benchmark_values.get(date, list(benchmark_values.values())[-1] if benchmark_values else initial_benchmark)
            benchmark_return = (benchmark_value - initial_benchmark) / initial_benchmark if initial_benchmark > 0 else 0
            
            snapshot = DailySnapshot(
                date=date,
                portfolio_value=portfolio_value,
                cash=self.simulator.cash,
                position_value=portfolio_value - self.simulator.cash,
                daily_return=daily_return,
                cumulative_return=cumulative_return,
                positions={
                    code: {
                        "shares": pos["shares"],
                        "avg_cost": pos["avg_cost"],
                        "current_price": current_prices.get(code, pos["avg_cost"]),
                        "market_value": pos["shares"] * current_prices.get(code, pos["avg_cost"]),
                    }
                    for code, pos in self.simulator.positions.items()
                },
                benchmark_value=benchmark_value,
                benchmark_return=benchmark_return,
            )
            daily_snapshots.append(snapshot)
            prev_portfolio_value = portfolio_value
        
        final_capital = prev_portfolio_value
        metrics = self._calculate_metrics(
            daily_snapshots=daily_snapshots,
            trades=self.simulator.trades,
            initial_capital=self.config.initial_capital,
            final_capital=final_capital,
        )
        
        equity_curve = self._build_equity_curve(daily_snapshots)
        monthly_returns = self._calculate_monthly_returns(daily_snapshots)
        
        execution_time = time.time() - start_time
        
        result = BacktestResult(
            backtest_id=backtest_id,
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            config=self.config,
            parameters=parameters or strategy.parameters,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            initial_capital=self.config.initial_capital,
            final_capital=final_capital,
            total_return=metrics["total_return"],
            annual_return=metrics["annual_return"],
            benchmark_return=metrics["benchmark_return"],
            excess_return=metrics["excess_return"],
            sharpe_ratio=metrics["sharpe_ratio"],
            sortino_ratio=metrics["sortino_ratio"],
            max_drawdown=metrics["max_drawdown"],
            max_drawdown_duration=metrics["max_drawdown_duration"],
            win_rate=metrics["win_rate"],
            profit_factor=metrics["profit_factor"],
            total_trades=metrics["total_trades"],
            winning_trades=metrics["winning_trades"],
            losing_trades=metrics["losing_trades"],
            trades=self.simulator.trades,
            daily_snapshots=daily_snapshots,
            equity_curve=equity_curve,
            monthly_returns=monthly_returns,
            execution_time=execution_time,
            metadata={
                "total_commission": self.simulator.total_commission,
                "total_stamp_duty": self.simulator.total_stamp_duty,
                "total_slippage": self.simulator.total_slippage,
            },
        )
        
        self.logger.save_result(result, strategy_id)
        
        logger.info(f"回测完成: 总收益={metrics['total_return']:.2%}, 夏普={metrics['sharpe_ratio']:.2f}, 最大回撤={metrics['max_drawdown']:.2%}")
        
        return result
    
    def run_all_strategies(
        self,
        strategies: List["StrategyBase"],
        data: Dict[str, pd.DataFrame],
    ) -> Dict[str, BacktestResult]:
        results = {}
        for strategy in strategies:
            try:
                result = self.run(strategy, data)
                results[strategy.strategy_id] = result
            except Exception as e:
                logger.error(f"回测策略 {strategy.strategy_id} 失败: {e}")
        
        return results
    
    def _create_empty_result(
        self,
        backtest_id: str,
        strategy_id: str,
        strategy_name: str,
        parameters: Dict[str, Any],
    ) -> BacktestResult:
        return BacktestResult(
            backtest_id=backtest_id,
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            config=self.config,
            parameters=parameters,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            initial_capital=self.config.initial_capital,
            final_capital=self.config.initial_capital,
            total_return=0.0,
            annual_return=0.0,
            benchmark_return=0.0,
            excess_return=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0,
            win_rate=0.0,
            profit_factor=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
        )
    
    def _calculate_metrics(
        self,
        daily_snapshots: List[DailySnapshot],
        trades: List[TradeLog],
        initial_capital: float,
        final_capital: float,
    ) -> Dict[str, float]:
        if not daily_snapshots:
            return {
                "total_return": 0.0,
                "annual_return": 0.0,
                "benchmark_return": 0.0,
                "excess_return": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "max_drawdown_duration": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
            }
        
        total_return = (final_capital - initial_capital) / initial_capital
        
        days = len(daily_snapshots)
        years = days / 252
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        benchmark_return = daily_snapshots[-1].benchmark_return if daily_snapshots else 0
        excess_return = total_return - benchmark_return
        
        daily_returns = [s.daily_return for s in daily_snapshots]
        
        if daily_returns:
            mean_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            sharpe_ratio = (mean_return * 252) / (std_return * np.sqrt(252)) if std_return > 0 else 0
            
            negative_returns = [r for r in daily_returns if r < 0]
            downside_std = np.std(negative_returns) if negative_returns else 0
            sortino_ratio = (mean_return * 252) / (downside_std * np.sqrt(252)) if downside_std > 0 else 0
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
        
        portfolio_values = [s.portfolio_value for s in daily_snapshots]
        peak = portfolio_values[0]
        max_drawdown = 0
        max_drawdown_duration = 0
        current_drawdown_start = 0
        
        for i, value in enumerate(portfolio_values):
            if value > peak:
                peak = value
                current_drawdown_start = i
            else:
                drawdown = (peak - value) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                    max_drawdown_duration = i - current_drawdown_start
        
        buy_trades = [t for t in trades if t.action == "buy"]
        sell_trades = [t for t in trades if t.action == "sell"]
        
        trade_pairs = []
        for buy in buy_trades:
            for sell in sell_trades:
                if buy.code == sell.code and sell.date >= buy.date:
                    profit = (sell.price - buy.price) * min(buy.shares, sell.shares)
                    trade_pairs.append(profit)
                    break
        
        winning_trades = len([p for p in trade_pairs if p > 0])
        losing_trades = len([p for p in trade_pairs if p < 0])
        total_trades = len(trade_pairs)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        gross_profit = sum([p for p in trade_pairs if p > 0])
        gross_loss = abs(sum([p for p in trade_pairs if p < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "benchmark_return": benchmark_return,
            "excess_return": excess_return,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "max_drawdown_duration": max_drawdown_duration,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
        }
    
    def _build_equity_curve(self, daily_snapshots: List[DailySnapshot]) -> pd.DataFrame:
        if not daily_snapshots:
            return pd.DataFrame()
        
        data = {
            "date": [s.date for s in daily_snapshots],
            "portfolio_value": [s.portfolio_value for s in daily_snapshots],
            "cash": [s.cash for s in daily_snapshots],
            "position_value": [s.position_value for s in daily_snapshots],
            "daily_return": [s.daily_return for s in daily_snapshots],
            "cumulative_return": [s.cumulative_return for s in daily_snapshots],
            "benchmark_value": [s.benchmark_value for s in daily_snapshots],
            "benchmark_return": [s.benchmark_return for s in daily_snapshots],
        }
        
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        
        return df
    
    def _calculate_monthly_returns(self, daily_snapshots: List[DailySnapshot]) -> Dict[str, float]:
        if not daily_snapshots:
            return {}
        
        monthly_data = {}
        for snapshot in daily_snapshots:
            month_key = snapshot.date[:7]
            if month_key not in monthly_data:
                monthly_data[month_key] = {
                    "start_value": snapshot.portfolio_value,
                    "end_value": snapshot.portfolio_value,
                }
            else:
                monthly_data[month_key]["end_value"] = snapshot.portfolio_value
        
        monthly_returns = {}
        for month, data in monthly_data.items():
            monthly_returns[month] = (data["end_value"] - data["start_value"]) / data["start_value"]
        
        return monthly_returns
