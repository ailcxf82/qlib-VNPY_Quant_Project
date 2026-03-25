"""
Performance Analyzer - 绩效分析器

计算策略回测的各项绩效指标，包括收益、风险、风险调整收益等。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from monitor.strategy_library.backtest.backtest_engine import (
    BacktestResult,
    DailySnapshot,
    TradeLog,
)

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    total_return: float
    annual_return: float
    monthly_return_avg: float
    monthly_return_std: float
    
    max_drawdown: float
    max_drawdown_duration: int
    avg_drawdown: float
    recovery_time: int
    
    volatility: float
    downside_volatility: float
    var_95: float
    cvar_95: float
    
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    treynor_ratio: float
    
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_holding_days: float
    
    alpha: float
    beta: float
    tracking_error: float
    
    benchmark_return: float
    excess_return: float
    
    monthly_returns: Dict[str, float] = field(default_factory=dict)
    yearly_returns: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "monthly_return_avg": self.monthly_return_avg,
            "monthly_return_std": self.monthly_return_std,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "avg_drawdown": self.avg_drawdown,
            "recovery_time": self.recovery_time,
            "volatility": self.volatility,
            "downside_volatility": self.downside_volatility,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "information_ratio": self.information_ratio,
            "treynor_ratio": self.treynor_ratio,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "avg_holding_days": self.avg_holding_days,
            "alpha": self.alpha,
            "beta": self.beta,
            "tracking_error": self.tracking_error,
            "benchmark_return": self.benchmark_return,
            "excess_return": self.excess_return,
            "monthly_returns": self.monthly_returns,
            "yearly_returns": self.yearly_returns,
        }


@dataclass
class TradeAnalysis:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_profit: float
    avg_loss: float
    max_profit: float
    max_loss: float
    avg_holding_days: float
    max_holding_days: int
    min_holding_days: int
    
    trade_distribution: Dict[str, int] = field(default_factory=dict)
    monthly_trade_counts: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_profit": self.avg_profit,
            "avg_loss": self.avg_loss,
            "max_profit": self.max_profit,
            "max_loss": self.max_loss,
            "avg_holding_days": self.avg_holding_days,
            "max_holding_days": self.max_holding_days,
            "min_holding_days": self.min_holding_days,
            "trade_distribution": self.trade_distribution,
            "monthly_trade_counts": self.monthly_trade_counts,
        }


class PerformanceAnalyzer:
    RISK_FREE_RATE = 0.03
    
    def __init__(self, risk_free_rate: float = 0.03):
        self.risk_free_rate = risk_free_rate
    
    def analyze(self, result: BacktestResult) -> PerformanceMetrics:
        daily_returns = [s.daily_return for s in result.daily_snapshots]
        portfolio_values = [s.portfolio_value for s in result.daily_snapshots]
        benchmark_returns = [s.benchmark_return for s in result.daily_snapshots]
        
        total_return = result.total_return
        annual_return = result.annual_return
        
        monthly_returns = self._calculate_monthly_returns(result.daily_snapshots)
        yearly_returns = self._calculate_yearly_returns(result.daily_snapshots)
        
        monthly_values = list(monthly_returns.values())
        monthly_return_avg = np.mean(monthly_values) if monthly_values else 0
        monthly_return_std = np.std(monthly_values) if monthly_values else 0
        
        drawdown_metrics = self._calculate_drawdown_metrics(portfolio_values)
        
        volatility = self._calculate_volatility(daily_returns)
        downside_volatility = self._calculate_downside_volatility(daily_returns)
        var_95 = self._calculate_var(daily_returns, 0.95)
        cvar_95 = self._calculate_cvar(daily_returns, 0.95)
        
        sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
        sortino_ratio = self._calculate_sortino_ratio(daily_returns)
        calmar_ratio = self._calculate_calmar_ratio(annual_return, drawdown_metrics["max_drawdown"])
        
        benchmark_return = result.benchmark_return
        excess_return = total_return - benchmark_return
        information_ratio = self._calculate_information_ratio(daily_returns, benchmark_returns)
        
        trade_analysis = self._analyze_trades(result.trades)
        
        alpha_beta = self._calculate_alpha_beta(daily_returns, benchmark_returns)
        
        return PerformanceMetrics(
            total_return=total_return,
            annual_return=annual_return,
            monthly_return_avg=monthly_return_avg,
            monthly_return_std=monthly_return_std,
            max_drawdown=drawdown_metrics["max_drawdown"],
            max_drawdown_duration=drawdown_metrics["max_drawdown_duration"],
            avg_drawdown=drawdown_metrics["avg_drawdown"],
            recovery_time=drawdown_metrics["recovery_time"],
            volatility=volatility,
            downside_volatility=downside_volatility,
            var_95=var_95,
            cvar_95=cvar_95,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            information_ratio=information_ratio,
            treynor_ratio=alpha_beta["treynor_ratio"],
            win_rate=trade_analysis.win_rate,
            profit_factor=trade_analysis.profit_factor,
            avg_win=trade_analysis.avg_profit,
            avg_loss=trade_analysis.avg_loss,
            max_consecutive_wins=trade_analysis.trade_distribution.get("max_consecutive_wins", 0),
            max_consecutive_losses=trade_analysis.trade_distribution.get("max_consecutive_losses", 0),
            total_trades=trade_analysis.total_trades,
            winning_trades=trade_analysis.winning_trades,
            losing_trades=trade_analysis.losing_trades,
            avg_holding_days=trade_analysis.avg_holding_days,
            alpha=alpha_beta["alpha"],
            beta=alpha_beta["beta"],
            tracking_error=alpha_beta["tracking_error"],
            benchmark_return=benchmark_return,
            excess_return=excess_return,
            monthly_returns=monthly_returns,
            yearly_returns=yearly_returns,
        )
    
    def analyze_trades(self, result: BacktestResult) -> TradeAnalysis:
        return self._analyze_trades(result.trades)
    
    def compare_strategies(
        self,
        results: Dict[str, BacktestResult],
    ) -> pd.DataFrame:
        comparison_data = []
        
        for strategy_id, result in results.items():
            metrics = self.analyze(result)
            comparison_data.append({
                "strategy_id": strategy_id,
                "strategy_name": result.strategy_name,
                "total_return": metrics.total_return,
                "annual_return": metrics.annual_return,
                "sharpe_ratio": metrics.sharpe_ratio,
                "max_drawdown": metrics.max_drawdown,
                "win_rate": metrics.win_rate,
                "profit_factor": metrics.profit_factor,
                "total_trades": metrics.total_trades,
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values("sharpe_ratio", ascending=False)
        
        return df
    
    def get_rolling_metrics(
        self,
        result: BacktestResult,
        window: int = 20,
    ) -> pd.DataFrame:
        if not result.daily_snapshots:
            return pd.DataFrame()
        
        dates = [s.date for s in result.daily_snapshots]
        returns = [s.daily_return for s in result.daily_snapshots]
        
        df = pd.DataFrame({
            "date": dates,
            "daily_return": returns,
        })
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        
        df["rolling_return"] = df["daily_return"].rolling(window).mean() * window
        df["rolling_volatility"] = df["daily_return"].rolling(window).std() * np.sqrt(252)
        df["rolling_sharpe"] = (df["daily_return"].rolling(window).mean() * 252) / (df["daily_return"].rolling(window).std() * np.sqrt(252))
        
        portfolio_values = [s.portfolio_value for s in result.daily_snapshots]
        df["rolling_max_dd"] = pd.Series(portfolio_values).rolling(window).apply(
            lambda x: self._calculate_max_drawdown_for_series(x)
        ).values
        
        return df
    
    def _calculate_monthly_returns(self, snapshots: List[DailySnapshot]) -> Dict[str, float]:
        monthly_data = {}
        
        for snapshot in snapshots:
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
            if data["start_value"] > 0:
                monthly_returns[month] = (data["end_value"] - data["start_value"]) / data["start_value"]
            else:
                monthly_returns[month] = 0
        
        return monthly_returns
    
    def _calculate_yearly_returns(self, snapshots: List[DailySnapshot]) -> Dict[str, float]:
        yearly_data = {}
        
        for snapshot in snapshots:
            year_key = snapshot.date[:4]
            if year_key not in yearly_data:
                yearly_data[year_key] = {
                    "start_value": snapshot.portfolio_value,
                    "end_value": snapshot.portfolio_value,
                }
            else:
                yearly_data[year_key]["end_value"] = snapshot.portfolio_value
        
        yearly_returns = {}
        for year, data in yearly_data.items():
            if data["start_value"] > 0:
                yearly_returns[year] = (data["end_value"] - data["start_value"]) / data["start_value"]
            else:
                yearly_returns[year] = 0
        
        return yearly_returns
    
    def _calculate_drawdown_metrics(self, portfolio_values: List[float]) -> Dict[str, Any]:
        if not portfolio_values:
            return {
                "max_drawdown": 0,
                "max_drawdown_duration": 0,
                "avg_drawdown": 0,
                "recovery_time": 0,
            }
        
        peak = portfolio_values[0]
        max_drawdown = 0
        max_drawdown_duration = 0
        drawdowns = []
        current_drawdown_start = 0
        in_drawdown = False
        recovery_times = []
        
        for i, value in enumerate(portfolio_values):
            if value > peak:
                if in_drawdown:
                    recovery_times.append(i - current_drawdown_start)
                    in_drawdown = False
                peak = value
            else:
                drawdown = (peak - value) / peak
                drawdowns.append(drawdown)
                
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                    max_drawdown_duration = i - current_drawdown_start
                
                if not in_drawdown:
                    current_drawdown_start = i
                    in_drawdown = True
        
        avg_drawdown = np.mean(drawdowns) if drawdowns else 0
        avg_recovery_time = int(np.mean(recovery_times)) if recovery_times else 0
        
        return {
            "max_drawdown": max_drawdown,
            "max_drawdown_duration": max_drawdown_duration,
            "avg_drawdown": avg_drawdown,
            "recovery_time": avg_recovery_time,
        }
    
    def _calculate_volatility(self, daily_returns: List[float]) -> float:
        if not daily_returns:
            return 0
        return float(np.std(daily_returns) * np.sqrt(252))
    
    def _calculate_downside_volatility(self, daily_returns: List[float]) -> float:
        negative_returns = [r for r in daily_returns if r < 0]
        if not negative_returns:
            return 0
        return float(np.std(negative_returns) * np.sqrt(252))
    
    def _calculate_var(self, daily_returns: List[float], confidence: float) -> float:
        if not daily_returns:
            return 0
        return float(np.percentile(daily_returns, (1 - confidence) * 100))
    
    def _calculate_cvar(self, daily_returns: List[float], confidence: float) -> float:
        if not daily_returns:
            return 0
        var = self._calculate_var(daily_returns, confidence)
        tail_returns = [r for r in daily_returns if r <= var]
        return float(np.mean(tail_returns)) if tail_returns else var
    
    def _calculate_sharpe_ratio(self, daily_returns: List[float]) -> float:
        if not daily_returns:
            return 0
        
        mean_return = np.mean(daily_returns)
        std_return = np.std(daily_returns)
        
        if std_return == 0:
            return 0
        
        daily_rf = self.risk_free_rate / 252
        excess_return = mean_return - daily_rf
        
        return float(excess_return / std_return * np.sqrt(252))
    
    def _calculate_sortino_ratio(self, daily_returns: List[float]) -> float:
        if not daily_returns:
            return 0
        
        mean_return = np.mean(daily_returns)
        downside_returns = [r for r in daily_returns if r < 0]
        
        if not downside_returns:
            return float('inf') if mean_return > 0 else 0
        
        downside_std = np.std(downside_returns)
        
        if downside_std == 0:
            return float('inf') if mean_return > 0 else 0
        
        daily_rf = self.risk_free_rate / 252
        excess_return = mean_return - daily_rf
        
        return float(excess_return / downside_std * np.sqrt(252))
    
    def _calculate_calmar_ratio(self, annual_return: float, max_drawdown: float) -> float:
        if max_drawdown == 0:
            return float('inf') if annual_return > 0 else 0
        return float(annual_return / max_drawdown)
    
    def _calculate_information_ratio(
        self,
        portfolio_returns: List[float],
        benchmark_returns: List[float],
    ) -> float:
        if not portfolio_returns or not benchmark_returns:
            return 0
        
        min_len = min(len(portfolio_returns), len(benchmark_returns))
        portfolio_returns = portfolio_returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]
        
        excess_returns = [p - b for p, b in zip(portfolio_returns, benchmark_returns)]
        
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns)
        
        if std_excess == 0:
            return 0
        
        return float(mean_excess / std_excess * np.sqrt(252))
    
    def _calculate_alpha_beta(
        self,
        portfolio_returns: List[float],
        benchmark_returns: List[float],
    ) -> Dict[str, float]:
        if not portfolio_returns or not benchmark_returns:
            return {"alpha": 0, "beta": 0, "tracking_error": 0, "treynor_ratio": 0}
        
        min_len = min(len(portfolio_returns), len(benchmark_returns))
        portfolio_returns = portfolio_returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]
        
        portfolio_arr = np.array(portfolio_returns)
        benchmark_arr = np.array(benchmark_returns)
        
        covariance = np.cov(portfolio_arr, benchmark_arr)[0, 1]
        benchmark_variance = np.var(benchmark_arr)
        
        if benchmark_variance == 0:
            beta = 0
        else:
            beta = covariance / benchmark_variance
        
        alpha = np.mean(portfolio_arr) - beta * np.mean(benchmark_arr)
        alpha_annual = alpha * 252
        
        tracking_error = np.std(portfolio_arr - benchmark_arr) * np.sqrt(252)
        
        portfolio_mean = np.mean(portfolio_arr)
        daily_rf = self.risk_free_rate / 252
        
        if beta == 0:
            treynor_ratio = 0
        else:
            treynor_ratio = (portfolio_mean - daily_rf) / beta * np.sqrt(252)
        
        return {
            "alpha": float(alpha_annual),
            "beta": float(beta),
            "tracking_error": float(tracking_error),
            "treynor_ratio": float(treynor_ratio),
        }
    
    def _analyze_trades(self, trades: List[TradeLog]) -> TradeAnalysis:
        if not trades:
            return TradeAnalysis(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                profit_factor=0,
                avg_profit=0,
                avg_loss=0,
                max_profit=0,
                max_loss=0,
                avg_holding_days=0,
                max_holding_days=0,
                min_holding_days=0,
            )
        
        buy_trades = [t for t in trades if t.action == "buy"]
        sell_trades = [t for t in trades if t.action == "sell"]
        
        trade_profits = []
        holding_days_list = []
        
        buy_dict = {}
        for buy in buy_trades:
            if buy.code not in buy_dict:
                buy_dict[buy.code] = []
            buy_dict[buy.code].append(buy)
        
        for sell in sell_trades:
            if sell.code in buy_dict and buy_dict[sell.code]:
                buy = buy_dict[sell.code].pop(0)
                profit = (sell.price - buy.price) * min(buy.shares, sell.shares)
                trade_profits.append(profit)
                
                buy_date = datetime.strptime(buy.date, "%Y-%m-%d")
                sell_date = datetime.strptime(sell.date, "%Y-%m-%d")
                holding_days = (sell_date - buy_date).days
                holding_days_list.append(holding_days)
        
        total_trades = len(trade_profits)
        winning_trades = len([p for p in trade_profits if p > 0])
        losing_trades = len([p for p in trade_profits if p < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        profits = [p for p in trade_profits if p > 0]
        losses = [p for p in trade_profits if p < 0]
        
        avg_profit = np.mean(profits) if profits else 0
        avg_loss = np.mean(losses) if losses else 0
        
        gross_profit = sum(profits) if profits else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        max_profit = max(profits) if profits else 0
        max_loss = min(losses) if losses else 0
        
        avg_holding_days = np.mean(holding_days_list) if holding_days_list else 0
        max_holding_days = max(holding_days_list) if holding_days_list else 0
        min_holding_days = min(holding_days_list) if holding_days_list else 0
        
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        
        for profit in trade_profits:
            if profit > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
        
        monthly_trade_counts = {}
        for trade in trades:
            month_key = trade.date[:7]
            monthly_trade_counts[month_key] = monthly_trade_counts.get(month_key, 0) + 1
        
        return TradeAnalysis(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_profit=avg_profit,
            avg_loss=avg_loss,
            max_profit=max_profit,
            max_loss=max_loss,
            avg_holding_days=avg_holding_days,
            max_holding_days=max_holding_days,
            min_holding_days=min_holding_days,
            trade_distribution={
                "max_consecutive_wins": max_consecutive_wins,
                "max_consecutive_losses": max_consecutive_losses,
            },
            monthly_trade_counts=monthly_trade_counts,
        )
    
    def _calculate_max_drawdown_for_series(self, series: pd.Series) -> float:
        if len(series) == 0:
            return 0
        
        peak = series.iloc[0]
        max_dd = 0
        
        for value in series:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        
        return max_dd
