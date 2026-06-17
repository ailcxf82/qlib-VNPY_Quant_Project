from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    start_date: str
    end_date: str
    total_return: float
    annual_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    profit_loss_ratio: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_holding_days: float
    trade_details: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "win_rate": self.win_rate,
            "profit_loss_ratio": self.profit_loss_ratio,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "avg_holding_days": self.avg_holding_days,
            "trade_details": self.trade_details[:20],
        }


@dataclass
class StrategyPerformance:
    strategy_name: str
    total_return: float
    win_rate: float
    avg_return_per_trade: float
    max_drawdown: float
    trades_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "total_return": self.total_return,
            "win_rate": self.win_rate,
            "avg_return_per_trade": self.avg_return_per_trade,
            "max_drawdown": self.max_drawdown,
            "trades_count": self.trades_count,
        }


class Backtester:
    def __init__(self, config_path: str = "config/monitor.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.backtest_config = self.config.get("backtest", {})
        
        self.initial_capital = self.backtest_config.get("initial_capital", 1000000)
        self.commission_rate = self.backtest_config.get("commission_rate", 0.0003)
        self.slippage = self.backtest_config.get("slippage", 0.001)
        
        self._trade_history: List[Dict[str, Any]] = []
        self._equity_curve: List[float] = []
    
    def _load_config(self) -> Dict[str, Any]:
        p = Path(self.config_path)
        if not p.exists():
            return {}
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    
    def _get_ts_client(self):
        from monitor.ts_client import get_ts_client
        return get_ts_client()
    
    def fetch_historical_data(
        self, 
        code: str, 
        start_date: str, 
        end_date: str
    ) -> pd.DataFrame:
        ts = self._get_ts_client()
        if ts is None:
            return pd.DataFrame()
        
        try:
            code_clean = str(code).replace(".XSHE", "").replace(".XSHG", "").zfill(6)
            suffix = ".SZ" if code_clean.startswith(("0", "3")) else ".SH"
            ts_code = code_clean + suffix
            
            df = ts._pro.daily(
                ts_code=ts_code,
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", "")
            )
            
            if df.empty:
                return pd.DataFrame()
            
            df = df.sort_values("trade_date").reset_index(drop=True)
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            return df
        except Exception as e:
            logger.warning(f"获取历史数据失败 {code}: {e}")
            return pd.DataFrame()
    
    def run_backtest(
        self,
        signals: List[Dict[str, Any]],
        start_date: str,
        end_date: str,
        position_size: float = 0.10,
        stop_loss: float = -0.08,
        take_profit: float = 0.15,
    ) -> BacktestResult:
        self._trade_history = []
        self._equity_curve = [self.initial_capital]
        
        capital = self.initial_capital
        positions: Dict[str, Dict[str, Any]] = {}
        
        signal_dates = {}
        for sig in signals:
            sig_date = sig.get("date", start_date)
            if sig_date not in signal_dates:
                signal_dates[sig_date] = []
            signal_dates[sig_date].append(sig)
        
        all_dates = pd.date_range(start=start_date, end=end_date, freq="B")
        
        for date in all_dates:
            date_str = date.strftime("%Y-%m-%d")
            
            for code, pos in list(positions.items()):
                df = self.fetch_historical_data(code, date_str, date_str)
                if df.empty:
                    continue
                
                current_price = float(df.iloc[0]["close"])
                pos["current_price"] = current_price
                pos["highest_price"] = max(pos.get("highest_price", current_price), current_price)
                pos["holding_days"] = pos.get("holding_days", 0) + 1
                
                profit_pct = (current_price - pos["entry_price"]) / pos["entry_price"]
                
                should_sell = False
                sell_reason = ""
                
                if profit_pct <= stop_loss:
                    should_sell = True
                    sell_reason = f"止损({profit_pct:.1%})"
                elif profit_pct >= take_profit:
                    should_sell = True
                    sell_reason = f"止盈({profit_pct:.1%})"
                elif pos["holding_days"] >= 10:
                    should_sell = True
                    sell_reason = f"时间止损({pos['holding_days']}天)"
                
                if should_sell:
                    sell_value = pos["shares"] * current_price * (1 - self.commission_rate)
                    capital += sell_value
                    
                    trade_return = (current_price - pos["entry_price"]) / pos["entry_price"]
                    
                    self._trade_history.append({
                        "code": code,
                        "name": pos["name"],
                        "action": "sell",
                        "date": date_str,
                        "price": current_price,
                        "shares": pos["shares"],
                        "value": sell_value,
                        "return": trade_return,
                        "reason": sell_reason,
                        "strategy": pos.get("strategy", "unknown"),
                    })
                    
                    del positions[code]
            
            if date_str in signal_dates:
                for sig in signal_dates[date_str]:
                    code = sig.get("code", "")
                    if code in positions:
                        continue
                    
                    if len(positions) >= 10:
                        break
                    
                    df = self.fetch_historical_data(code, date_str, date_str)
                    if df.empty:
                        continue
                    
                    entry_price = float(df.iloc[0]["close"]) * (1 + self.slippage)
                    position_value = capital * position_size
                    shares = int(position_value / entry_price / 100) * 100
                    
                    if shares < 100:
                        continue
                    
                    buy_value = shares * entry_price * (1 + self.commission_rate)
                    if buy_value > capital:
                        continue
                    
                    capital -= buy_value
                    
                    positions[code] = {
                        "name": sig.get("name", code),
                        "entry_price": entry_price,
                        "shares": shares,
                        "entry_date": date_str,
                        "current_price": entry_price,
                        "highest_price": entry_price,
                        "holding_days": 0,
                        "strategy": sig.get("strategy", "unknown"),
                    }
                    
                    self._trade_history.append({
                        "code": code,
                        "name": sig.get("name", code),
                        "action": "buy",
                        "date": date_str,
                        "price": entry_price,
                        "shares": shares,
                        "value": buy_value,
                        "strategy": sig.get("strategy", "unknown"),
                    })
            
            total_equity = capital
            for pos in positions.values():
                total_equity += pos["shares"] * pos.get("current_price", pos["entry_price"])
            self._equity_curve.append(total_equity)
        
        for code, pos in positions.items():
            df = self.fetch_historical_data(code, end_date, end_date)
            if not df.empty:
                final_price = float(df.iloc[0]["close"])
                capital += pos["shares"] * final_price * (1 - self.commission_rate)
        
        return self._calculate_result(start_date, end_date)
    
    def _calculate_result(self, start_date: str, end_date: str) -> BacktestResult:
        if not self._trade_history:
            return BacktestResult(
                start_date=start_date,
                end_date=end_date,
                total_return=0.0,
                annual_return=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                win_rate=0.0,
                profit_loss_ratio=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                avg_holding_days=0.0,
            )
        
        sell_trades = [t for t in self._trade_history if t["action"] == "sell"]
        
        total_return = (self._equity_curve[-1] - self.initial_capital) / self.initial_capital
        
        days = (datetime.strptime(end_date, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")).days
        annual_return = (1 + total_return) ** (252 / max(days, 1)) - 1
        
        equity_array = np.array(self._equity_curve)
        peak = np.maximum.accumulate(equity_array)
        drawdown = (peak - equity_array) / peak
        max_drawdown = float(np.max(drawdown))
        
        returns = np.diff(equity_array) / equity_array[:-1]
        sharpe_ratio = 0.0
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = float(np.mean(returns) / np.std(returns) * np.sqrt(252))
        
        winning_trades = [t for t in sell_trades if t.get("return", 0) > 0]
        losing_trades = [t for t in sell_trades if t.get("return", 0) <= 0]
        
        win_rate = len(winning_trades) / len(sell_trades) if sell_trades else 0
        
        avg_win = np.mean([t["return"] for t in winning_trades]) if winning_trades else 0
        avg_loss = abs(np.mean([t["return"] for t in losing_trades])) if losing_trades else 0
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        holding_days = [t.get("holding_days", 0) for t in sell_trades]
        avg_holding_days = np.mean(holding_days) if holding_days else 0
        
        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            total_return=total_return,
            annual_return=annual_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            profit_loss_ratio=profit_loss_ratio,
            total_trades=len(sell_trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_holding_days=avg_holding_days,
            trade_details=self._trade_history,
        )
    
    def backtest_strategy(
        self,
        strategy_name: str,
        signals: List[Dict[str, Any]],
        start_date: str,
        end_date: str,
    ) -> StrategyPerformance:
        strategy_signals = [s for s in signals if s.get("strategy") == strategy_name]
        
        if not strategy_signals:
            return StrategyPerformance(
                strategy_name=strategy_name,
                total_return=0.0,
                win_rate=0.0,
                avg_return_per_trade=0.0,
                max_drawdown=0.0,
                trades_count=0,
            )
        
        result = self.run_backtest(strategy_signals, start_date, end_date)
        
        return StrategyPerformance(
            strategy_name=strategy_name,
            total_return=result.total_return,
            win_rate=result.win_rate,
            avg_return_per_trade=result.total_return / max(result.total_trades, 1),
            max_drawdown=result.max_drawdown,
            trades_count=result.total_trades,
        )
    
    def compare_strategies(
        self,
        signals: List[Dict[str, Any]],
        start_date: str,
        end_date: str,
    ) -> Dict[str, Any]:
        strategies = set(s.get("strategy", "unknown") for s in signals)
        
        performances = {}
        for strategy in strategies:
            perf = self.backtest_strategy(strategy, signals, start_date, end_date)
            performances[strategy] = perf.to_dict()
        
        sorted_strategies = sorted(
            performances.items(),
            key=lambda x: x[1]["total_return"],
            reverse=True
        )
        
        return {
            "start_date": start_date,
            "end_date": end_date,
            "strategy_performances": dict(sorted_strategies),
            "best_strategy": sorted_strategies[0][0] if sorted_strategies else None,
            "recommendation": self._generate_recommendation(performances),
        }
    
    def _generate_recommendation(self, performances: Dict[str, Dict]) -> str:
        if not performances:
            return "无足够数据进行策略比较"
        
        best = max(performances.items(), key=lambda x: x[1]["total_return"])
        best_name, best_perf = best
        
        lines = [f"最佳策略: {best_name}"]
        lines.append(f"总收益: {best_perf['total_return']:.1%}")
        lines.append(f"胜率: {best_perf['win_rate']:.1%}")
        lines.append(f"最大回撤: {best_perf['max_drawdown']:.1%}")
        
        return " | ".join(lines)
    
    def validate_current_config(self) -> Dict[str, Any]:
        risk_config = self.config.get("risk_control", {})
        
        validation = {
            "valid": True,
            "warnings": [],
            "recommendations": [],
        }
        
        stop_loss = risk_config.get("stop_loss", -0.08)
        if stop_loss > -0.05:
            validation["warnings"].append(f"止损阈值({stop_loss:.1%})过宽，建议收紧到-8%以内")
        
        take_profit = risk_config.get("take_profit", 0.15)
        if take_profit < 0.10:
            validation["warnings"].append(f"止盈阈值({take_profit:.1%})过窄，可能影响收益")
        
        max_position = risk_config.get("max_position_size", 0.15)
        if max_position > 0.20:
            validation["warnings"].append(f"单只仓位上限({max_position:.1%})过大，建议不超过15%")
        
        max_drawdown = risk_config.get("max_drawdown", 0.10)
        if max_drawdown > 0.15:
            validation["warnings"].append(f"最大回撤容忍({max_drawdown:.1%})过大，建议不超过10%")
        
        if validation["warnings"]:
            validation["valid"] = False
        
        return validation
    
    def generate_report(self, result: BacktestResult) -> str:
        lines = [
            "📊 **回测报告**",
            "",
            f"回测区间: {result.start_date} ~ {result.end_date}",
            "",
            "**收益指标**",
            f"- 总收益: {result.total_return:.1%}",
            f"- 年化收益: {result.annual_return:.1%}",
            f"- 最大回撤: {result.max_drawdown:.1%}",
            f"- 夏普比率: {result.sharpe_ratio:.2f}",
            "",
            "**交易统计**",
            f"- 总交易次数: {result.total_trades}",
            f"- 盈利次数: {result.winning_trades}",
            f"- 亏损次数: {result.losing_trades}",
            f"- 胜率: {result.win_rate:.1%}",
            f"- 盈亏比: {result.profit_loss_ratio:.2f}",
            f"- 平均持仓天数: {result.avg_holding_days:.1f}",
        ]
        
        return "\n".join(lines)
