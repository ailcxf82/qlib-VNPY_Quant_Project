from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    total_return: float = 0.0
    annual_return: float = 0.0
    benchmark_return: float = 0.0
    excess_return: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_loss_ratio: float = 0.0
    avg_profit: float = 0.0
    avg_loss: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_holding_days: float = 0.0
    volatility: float = 0.0
    downside_volatility: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0
    information_ratio: float = 0.0
    tracking_error: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "benchmark_return": self.benchmark_return,
            "excess_return": self.excess_return,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "win_rate": self.win_rate,
            "profit_loss_ratio": self.profit_loss_ratio,
            "avg_profit": self.avg_profit,
            "avg_loss": self.avg_loss,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "avg_holding_days": self.avg_holding_days,
            "volatility": self.volatility,
            "downside_volatility": self.downside_volatility,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "beta": self.beta,
            "alpha": self.alpha,
            "information_ratio": self.information_ratio,
            "tracking_error": self.tracking_error,
        }


@dataclass
class DrawdownPeriod:
    start_date: str
    end_date: str
    drawdown: float
    duration: int
    recovery_date: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "drawdown": self.drawdown,
            "duration": self.duration,
            "recovery_date": self.recovery_date,
        }


@dataclass
class TradeAnalysis:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_return: float
    avg_winning_return: float
    avg_losing_return: float
    max_profit_trade: float
    max_loss_trade: float
    profit_loss_ratio: float
    avg_holding_days: float
    trades_by_month: Dict[str, int]
    trades_by_stock: Dict[str, int]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "avg_return": self.avg_return,
            "avg_winning_return": self.avg_winning_return,
            "avg_losing_return": self.avg_losing_return,
            "max_profit_trade": self.max_profit_trade,
            "max_loss_trade": self.max_loss_trade,
            "profit_loss_ratio": self.profit_loss_ratio,
            "avg_holding_days": self.avg_holding_days,
            "trades_by_month": self.trades_by_month,
            "trades_by_stock": self.trades_by_stock,
        }


class PerformanceAnalyzer:
    RISK_FREE_RATE = 0.03
    TRADING_DAYS_PER_YEAR = 252
    
    def __init__(
        self,
        equity_curve: List[float],
        trades: List[Any],
        config: Any,
        benchmark_returns: Optional[List[float]] = None,
    ):
        self.equity_curve = np.array(equity_curve)
        self.trades = trades
        self.config = config
        self.benchmark_returns = np.array(benchmark_returns) if benchmark_returns else None
        
        self._returns: Optional[np.ndarray] = None
        self._cumulative_returns: Optional[np.ndarray] = None
    
    @property
    def returns(self) -> np.ndarray:
        if self._returns is None:
            self._returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        return self._returns
    
    @property
    def cumulative_returns(self) -> np.ndarray:
        if self._cumulative_returns is None:
            self._cumulative_returns = (self.equity_curve / self.equity_curve[0]) - 1
        return self._cumulative_returns
    
    def analyze(self) -> Dict[str, Any]:
        metrics = self._calculate_metrics()
        drawdowns = self._calculate_drawdowns()
        trade_analysis = self._analyze_trades()
        monthly_returns = self._calculate_monthly_returns()
        yearly_returns = self._calculate_yearly_returns()
        
        return {
            "metrics": metrics.to_dict(),
            "drawdowns": [d.to_dict() for d in drawdowns],
            "trade_analysis": trade_analysis.to_dict(),
            "monthly_returns": monthly_returns,
            "yearly_returns": yearly_returns,
            "equity_curve": self.equity_curve.tolist(),
        }
    
    def _calculate_metrics(self) -> PerformanceMetrics:
        metrics = PerformanceMetrics()
        
        if len(self.equity_curve) < 2:
            return metrics
        
        metrics.total_return = (self.equity_curve[-1] - self.equity_curve[0]) / self.equity_curve[0]
        
        days = len(self.equity_curve) - 1
        if days > 0:
            metrics.annual_return = (1 + metrics.total_return) ** (self.TRADING_DAYS_PER_YEAR / days) - 1
        
        metrics.volatility = np.std(self.returns) * np.sqrt(self.TRADING_DAYS_PER_YEAR)
        
        negative_returns = self.returns[self.returns < 0]
        if len(negative_returns) > 0:
            metrics.downside_volatility = np.std(negative_returns) * np.sqrt(self.TRADING_DAYS_PER_YEAR)
        
        if metrics.volatility > 0:
            excess_return = metrics.annual_return - self.RISK_FREE_RATE
            metrics.sharpe_ratio = excess_return / metrics.volatility
        
        if metrics.downside_volatility > 0:
            excess_return = metrics.annual_return - self.RISK_FREE_RATE
            metrics.sortino_ratio = excess_return / metrics.downside_volatility
        
        peak = np.maximum.accumulate(self.equity_curve)
        drawdown = (peak - self.equity_curve) / peak
        metrics.max_drawdown = float(np.max(drawdown))
        
        if metrics.max_drawdown > 0:
            metrics.calmar_ratio = metrics.annual_return / metrics.max_drawdown
        
        drawdown_periods = self._find_drawdown_periods(drawdown)
        if drawdown_periods:
            metrics.max_drawdown_duration = max(p.duration for p in drawdown_periods)
        
        if len(self.returns) > 0:
            metrics.var_95 = float(np.percentile(self.returns, 5))
            
            var_threshold = np.percentile(self.returns, 5)
            tail_returns = self.returns[self.returns <= var_threshold]
            if len(tail_returns) > 0:
                metrics.cvar_95 = float(np.mean(tail_returns))
        
        if self.benchmark_returns is not None and len(self.benchmark_returns) == len(self.returns):
            self._calculate_benchmark_metrics(metrics)
        
        trade_metrics = self._calculate_trade_metrics()
        metrics.total_trades = trade_metrics["total_trades"]
        metrics.winning_trades = trade_metrics["winning_trades"]
        metrics.losing_trades = trade_metrics["losing_trades"]
        metrics.win_rate = trade_metrics["win_rate"]
        metrics.profit_loss_ratio = trade_metrics["profit_loss_ratio"]
        metrics.avg_profit = trade_metrics["avg_profit"]
        metrics.avg_loss = trade_metrics["avg_loss"]
        metrics.avg_holding_days = trade_metrics["avg_holding_days"]
        
        return metrics
    
    def _calculate_benchmark_metrics(self, metrics: PerformanceMetrics) -> None:
        if self.benchmark_returns is None:
            return
        
        metrics.benchmark_return = float(np.prod(1 + self.benchmark_returns) - 1)
        metrics.excess_return = metrics.annual_return - (np.mean(self.benchmark_returns) * self.TRADING_DAYS_PER_YEAR)
        
        excess_returns = self.returns - self.benchmark_returns
        metrics.tracking_error = float(np.std(excess_returns) * np.sqrt(self.TRADING_DAYS_PER_YEAR))
        
        if metrics.tracking_error > 0:
            metrics.information_ratio = metrics.excess_return / metrics.tracking_error
        
        covariance = np.cov(self.returns, self.benchmark_returns)[0, 1]
        benchmark_variance = np.var(self.benchmark_returns)
        if benchmark_variance > 0:
            metrics.beta = float(covariance / benchmark_variance)
        
        if metrics.beta > 0:
            benchmark_annual_return = np.mean(self.benchmark_returns) * self.TRADING_DAYS_PER_YEAR
            metrics.alpha = metrics.annual_return - self.RISK_FREE_RATE - metrics.beta * (benchmark_annual_return - self.RISK_FREE_RATE)
    
    def _find_drawdown_periods(self, drawdown: np.ndarray) -> List[DrawdownPeriod]:
        periods = []
        in_drawdown = False
        start_idx = 0
        max_dd = 0.0
        
        for i, dd in enumerate(drawdown):
            if dd > 0 and not in_drawdown:
                in_drawdown = True
                start_idx = i
                max_dd = dd
            elif dd > 0 and in_drawdown:
                max_dd = max(max_dd, dd)
            elif dd == 0 and in_drawdown:
                periods.append(DrawdownPeriod(
                    start_date=str(start_idx),
                    end_date=str(i - 1),
                    drawdown=max_dd,
                    duration=i - start_idx,
                ))
                in_drawdown = False
                max_dd = 0.0
        
        if in_drawdown:
            periods.append(DrawdownPeriod(
                start_date=str(start_idx),
                end_date=str(len(drawdown) - 1),
                drawdown=max_dd,
                duration=len(drawdown) - start_idx,
            ))
        
        return periods
    
    def _calculate_drawdowns(self) -> List[DrawdownPeriod]:
        if len(self.equity_curve) < 2:
            return []
        
        peak = np.maximum.accumulate(self.equity_curve)
        drawdown = (peak - self.equity_curve) / peak
        
        return self._find_drawdown_periods(drawdown)
    
    def _calculate_trade_metrics(self) -> Dict[str, Any]:
        result = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "profit_loss_ratio": 0.0,
            "avg_profit": 0.0,
            "avg_loss": 0.0,
            "avg_holding_days": 0.0,
        }
        
        sell_trades = [t for t in self.trades if t.action == "sell"]
        
        if not sell_trades:
            return result
        
        result["total_trades"] = len(sell_trades)
        
        profits = []
        holding_days = []
        
        for trade in sell_trades:
            buy_trade = next(
                (t for t in self.trades 
                 if t.code == trade.code and t.action == "buy" and t.date < trade.date),
                None
            )
            
            if buy_trade:
                profit_pct = (trade.price - buy_trade.price) / buy_trade.price
                profits.append(profit_pct)
                
                try:
                    buy_date = datetime.strptime(buy_trade.date, "%Y-%m-%d")
                    sell_date = datetime.strptime(trade.date, "%Y-%m-%d")
                    holding_days.append((sell_date - buy_date).days)
                except:
                    pass
        
        if profits:
            winning = [p for p in profits if p > 0]
            losing = [p for p in profits if p <= 0]
            
            result["winning_trades"] = len(winning)
            result["losing_trades"] = len(losing)
            result["win_rate"] = len(winning) / len(profits)
            
            if winning:
                result["avg_profit"] = float(np.mean(winning))
            if losing:
                result["avg_loss"] = float(np.mean(losing))
            
            if result["avg_loss"] != 0:
                result["profit_loss_ratio"] = abs(result["avg_profit"] / result["avg_loss"])
        
        if holding_days:
            result["avg_holding_days"] = float(np.mean(holding_days))
        
        return result
    
    def _analyze_trades(self) -> TradeAnalysis:
        sell_trades = [t for t in self.trades if t.action == "sell"]
        
        total_trades = len(sell_trades)
        winning_trades = 0
        losing_trades = 0
        returns = []
        winning_returns = []
        losing_returns = []
        holding_days = []
        trades_by_month: Dict[str, int] = {}
        trades_by_stock: Dict[str, int] = {}
        
        for trade in sell_trades:
            buy_trade = next(
                (t for t in self.trades 
                 if t.code == trade.code and t.action == "buy" and t.date < trade.date),
                None
            )
            
            if buy_trade:
                profit_pct = (trade.price - buy_trade.price) / buy_trade.price
                returns.append(profit_pct)
                
                if profit_pct > 0:
                    winning_trades += 1
                    winning_returns.append(profit_pct)
                else:
                    losing_trades += 1
                    losing_returns.append(profit_pct)
                
                try:
                    buy_date = datetime.strptime(buy_trade.date, "%Y-%m-%d")
                    sell_date = datetime.strptime(trade.date, "%Y-%m-%d")
                    holding_days.append((sell_date - buy_date).days)
                except:
                    pass
            
            try:
                month_key = trade.date[:7]
                trades_by_month[month_key] = trades_by_month.get(month_key, 0) + 1
            except:
                pass
            
            trades_by_stock[trade.code] = trades_by_stock.get(trade.code, 0) + 1
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_return = float(np.mean(returns)) if returns else 0
        avg_winning_return = float(np.mean(winning_returns)) if winning_returns else 0
        avg_losing_return = float(np.mean(losing_returns)) if losing_returns else 0
        max_profit_trade = float(max(returns)) if returns else 0
        max_loss_trade = float(min(returns)) if returns else 0
        profit_loss_ratio = abs(avg_winning_return / avg_losing_return) if avg_losing_return != 0 else 0
        avg_holding_days = float(np.mean(holding_days)) if holding_days else 0
        
        return TradeAnalysis(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_return=avg_return,
            avg_winning_return=avg_winning_return,
            avg_losing_return=avg_losing_return,
            max_profit_trade=max_profit_trade,
            max_loss_trade=max_loss_trade,
            profit_loss_ratio=profit_loss_ratio,
            avg_holding_days=avg_holding_days,
            trades_by_month=trades_by_month,
            trades_by_stock=trades_by_stock,
        )
    
    def _calculate_monthly_returns(self) -> Dict[str, float]:
        if len(self.equity_curve) < 2:
            return {}
        
        monthly_returns = {}
        
        for i in range(1, len(self.equity_curve)):
            month_key = str(i)
            monthly_returns[month_key] = float(self.returns[i-1]) if i <= len(self.returns) else 0
        
        return monthly_returns
    
    def _calculate_yearly_returns(self) -> Dict[str, float]:
        if len(self.equity_curve) < 2:
            return {}
        
        yearly_returns = {}
        
        for i in range(1, len(self.equity_curve)):
            year_key = str(i)
            yearly_returns[year_key] = float(self.returns[i-1]) if i <= len(self.returns) else 0
        
        return yearly_returns


class BacktestReportGenerator:
    def __init__(self, results: Dict[str, Any]):
        self.results = results
        self.metrics = results.get("metrics", {})
        self.trades = results.get("trades", [])
        self.config = results.get("config", {})
    
    def generate_text_report(self) -> str:
        lines = [
            "=" * 70,
            "                        回测报告",
            "=" * 70,
            "",
            "一、回测参数",
            "-" * 50,
            f"  回测区间: {self.config.get('start_date', 'N/A')} ~ {self.config.get('end_date', 'N/A')}",
            f"  初始资金: ¥{self.config.get('initial_capital', 0):,.0f}",
            f"  交易成本: 佣金 {self.config.get('commission_rate', 0):.2%}, 印花税 {self.config.get('stamp_duty', 0):.2%}",
            f"  滑点: {self.config.get('slippage', 0):.2%}",
            "",
            "二、收益指标",
            "-" * 50,
            f"  总收益率: {self.metrics.get('total_return', 0):.2%}",
            f"  年化收益率: {self.metrics.get('annual_return', 0):.2%}",
            f"  基准收益率: {self.metrics.get('benchmark_return', 0):.2%}",
            f"  超额收益: {self.metrics.get('excess_return', 0):.2%}",
            "",
            "三、风险指标",
            "-" * 50,
            f"  最大回撤: {self.metrics.get('max_drawdown', 0):.2%}",
            f"  最大回撤持续期: {self.metrics.get('max_drawdown_duration', 0)} 天",
            f"  年化波动率: {self.metrics.get('volatility', 0):.2%}",
            f"  下行波动率: {self.metrics.get('downside_volatility', 0):.2%}",
            f"  VaR(95%): {self.metrics.get('var_95', 0):.2%}",
            f"  CVaR(95%): {self.metrics.get('cvar_95', 0):.2%}",
            "",
            "四、风险调整收益",
            "-" * 50,
            f"  夏普比率: {self.metrics.get('sharpe_ratio', 0):.2f}",
            f"  索提诺比率: {self.metrics.get('sortino_ratio', 0):.2f}",
            f"  卡玛比率: {self.metrics.get('calmar_ratio', 0):.2f}",
            f"  信息比率: {self.metrics.get('information_ratio', 0):.2f}",
            "",
            "五、交易统计",
            "-" * 50,
            f"  总交易次数: {self.metrics.get('total_trades', 0)}",
            f"  盈利次数: {self.metrics.get('winning_trades', 0)}",
            f"  亏损次数: {self.metrics.get('losing_trades', 0)}",
            f"  胜率: {self.metrics.get('win_rate', 0):.1%}",
            f"  盈亏比: {self.metrics.get('profit_loss_ratio', 0):.2f}",
            f"  平均盈利: {self.metrics.get('avg_profit', 0):.2%}",
            f"  平均亏损: {self.metrics.get('avg_loss', 0):.2%}",
            f"  平均持仓天数: {self.metrics.get('avg_holding_days', 0):.1f}",
            "",
            "=" * 70,
        ]
        
        return "\n".join(lines)
    
    def generate_markdown_report(self) -> str:
        lines = [
            "# 回测报告",
            "",
            "## 一、回测参数",
            "",
            "| 参数 | 值 |",
            "|------|-----|",
            f"| 回测区间 | {self.config.get('start_date', 'N/A')} ~ {self.config.get('end_date', 'N/A')} |",
            f"| 初始资金 | ¥{self.config.get('initial_capital', 0):,.0f} |",
            f"| 交易成本 | 佣金 {self.config.get('commission_rate', 0):.2%}, 印花税 {self.config.get('stamp_duty', 0):.2%} |",
            f"| 滑点 | {self.config.get('slippage', 0):.2%} |",
            "",
            "## 二、收益指标",
            "",
            "| 指标 | 值 |",
            "|------|-----|",
            f"| 总收益率 | {self.metrics.get('total_return', 0):.2%} |",
            f"| 年化收益率 | {self.metrics.get('annual_return', 0):.2%} |",
            f"| 基准收益率 | {self.metrics.get('benchmark_return', 0):.2%} |",
            f"| 超额收益 | {self.metrics.get('excess_return', 0):.2%} |",
            "",
            "## 三、风险指标",
            "",
            "| 指标 | 值 |",
            "|------|-----|",
            f"| 最大回撤 | {self.metrics.get('max_drawdown', 0):.2%} |",
            f"| 最大回撤持续期 | {self.metrics.get('max_drawdown_duration', 0)} 天 |",
            f"| 年化波动率 | {self.metrics.get('volatility', 0):.2%} |",
            f"| 下行波动率 | {self.metrics.get('downside_volatility', 0):.2%} |",
            f"| VaR(95%) | {self.metrics.get('var_95', 0):.2%} |",
            f"| CVaR(95%) | {self.metrics.get('cvar_95', 0):.2%} |",
            "",
            "## 四、风险调整收益",
            "",
            "| 指标 | 值 |",
            "|------|-----|",
            f"| 夏普比率 | {self.metrics.get('sharpe_ratio', 0):.2f} |",
            f"| 索提诺比率 | {self.metrics.get('sortino_ratio', 0):.2f} |",
            f"| 卡玛比率 | {self.metrics.get('calmar_ratio', 0):.2f} |",
            f"| 信息比率 | {self.metrics.get('information_ratio', 0):.2f} |",
            "",
            "## 五、交易统计",
            "",
            "| 指标 | 值 |",
            "|------|-----|",
            f"| 总交易次数 | {self.metrics.get('total_trades', 0)} |",
            f"| 盈利次数 | {self.metrics.get('winning_trades', 0)} |",
            f"| 亏损次数 | {self.metrics.get('losing_trades', 0)} |",
            f"| 胜率 | {self.metrics.get('win_rate', 0):.1%} |",
            f"| 盈亏比 | {self.metrics.get('profit_loss_ratio', 0):.2f} |",
            f"| 平均盈利 | {self.metrics.get('avg_profit', 0):.2%} |",
            f"| 平均亏损 | {self.metrics.get('avg_loss', 0):.2%} |",
            f"| 平均持仓天数 | {self.metrics.get('avg_holding_days', 0):.1f} |",
            "",
        ]
        
        return "\n".join(lines)
    
    def save_report(self, output_path: str, format: str = "text") -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "markdown":
            content = self.generate_markdown_report()
        else:
            content = self.generate_text_report()
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        
        logger.info(f"Report saved to {path}")
    
    def save_json(self, output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to {path}")
