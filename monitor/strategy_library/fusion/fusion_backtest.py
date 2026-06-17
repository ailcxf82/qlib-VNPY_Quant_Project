"""
融合回测系统 - Fusion Backtest System

用于验证融合层的增益效应，对比融合策略与单独使用模型/策略的表现。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .fusion_engine import (
    ConsistencyType,
    MarketRegime,
    ModelPrediction,
    StrategySignalWrapper,
    FusedSignal,
    MarketContext,
    FusionConfig,
)
from .layered_fusion import (
    LayeredFusionProcessor,
    GainMetrics,
    GainValidator,
    FusionDecision,
)

logger = logging.getLogger(__name__)


@dataclass
class BacktestPosition:
    code: str
    name: str
    direction: str
    entry_price: float
    entry_date: str
    shares: int
    position_value: float
    stop_loss: float
    take_profit: float
    source: str
    decision: Optional[FusionDecision] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "entry_date": self.entry_date,
            "shares": self.shares,
            "position_value": self.position_value,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "source": self.source,
        }


@dataclass
class BacktestTrade:
    code: str
    name: str
    direction: str
    entry_price: float
    exit_price: float
    entry_date: str
    exit_date: str
    shares: int
    pnl: float
    pnl_pct: float
    source: str
    hold_days: int
    decision: Optional[FusionDecision] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "entry_date": self.entry_date,
            "exit_date": self.exit_date,
            "shares": self.shares,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "source": self.source,
            "hold_days": self.hold_days,
        }


@dataclass
class BacktestResult:
    source: str
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profit_trades: int
    loss_trades: int
    avg_profit: float
    avg_loss: float
    profit_factor: float
    calmar_ratio: float
    sortino_ratio: float
    daily_returns: pd.Series = field(default_factory=pd.Series)
    trades: List[BacktestTrade] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "total_trades": self.total_trades,
            "profit_trades": self.profit_trades,
            "loss_trades": self.loss_trades,
            "avg_profit": self.avg_profit,
            "avg_loss": self.avg_loss,
            "profit_factor": self.profit_factor,
            "calmar_ratio": self.calmar_ratio,
            "sortino_ratio": self.sortino_ratio,
        }


class FusionBacktester:
    """融合回测器"""

    def __init__(
        self,
        config: Optional[FusionConfig] = None,
        initial_capital: float = 1000000,
        commission_rate: float = 0.0003,
        slippage: float = 0.001,
    ):
        self.config = config or FusionConfig()
        self.fusion_processor = LayeredFusionProcessor(self.config)
        
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        
        self._positions: Dict[str, BacktestPosition] = {}
        self._trades: List[BacktestTrade] = []
        self._daily_values: List[Dict] = []
        
        self._model_positions: Dict[str, BacktestPosition] = {}
        self._model_trades: List[BacktestTrade] = []
        self._model_daily_values: List[Dict] = []
        
        self._strategy_positions: Dict[str, BacktestPosition] = {}
        self._strategy_trades: List[BacktestTrade] = []
        self._strategy_daily_values: List[Dict] = []

    def run_backtest(
        self,
        price_data: pd.DataFrame,
        model_predictions: Dict[str, List[ModelPrediction]],
        strategy_signals: Dict[str, List[StrategySignalWrapper]],
        market_contexts: Optional[Dict[str, MarketContext]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Tuple[BacktestResult, BacktestResult, BacktestResult, GainMetrics]:
        dates = sorted(price_data["date"].unique())
        
        if start_date:
            dates = [d for d in dates if d >= start_date]
        if end_date:
            dates = [d for d in dates if d <= end_date]
        
        cash_fused = self.initial_capital
        cash_model = self.initial_capital
        cash_strategy = self.initial_capital
        
        for date in dates:
            daily_prices = price_data[price_data["date"] == date].set_index("code")
            
            daily_preds = self._get_predictions_for_date(model_predictions, date)
            daily_signals = self._get_signals_for_date(strategy_signals, date)
            market_ctx = market_contexts.get(date) if market_contexts else None
            
            decisions = self.fusion_processor.process(
                daily_preds, daily_signals, market_ctx
            )
            
            cash_fused = self._execute_decisions(
                decisions, daily_prices, date, cash_fused, "fused"
            )
            
            cash_model = self._execute_model_only(
                daily_preds, daily_prices, date, cash_model
            )
            
            cash_strategy = self._execute_strategy_only(
                daily_signals, daily_prices, date, cash_strategy
            )
            
            total_fused = self._calculate_capital(cash_fused, self._positions, daily_prices)
            total_model = self._calculate_capital(cash_model, self._model_positions, daily_prices)
            total_strategy = self._calculate_capital(cash_strategy, self._strategy_positions, daily_prices)
            
            self._record_daily_value(date, total_fused, total_model, total_strategy)
        
        fused_result = self._calculate_result("fused", self._trades, self._daily_values)
        model_result = self._calculate_result("model", self._model_trades, self._model_daily_values)
        strategy_result = self._calculate_result("strategy", self._strategy_trades, self._strategy_daily_values)
        
        gain_metrics = self.fusion_processor.validate_gain_effect(
            model_result.daily_returns,
            strategy_result.daily_returns,
            fused_result.daily_returns,
        )
        
        return fused_result, model_result, strategy_result, gain_metrics

    def _get_predictions_for_date(
        self, predictions: Dict[str, List[ModelPrediction]], date: str
    ) -> Dict[str, ModelPrediction]:
        result = {}
        for code, preds in predictions.items():
            for pred in preds:
                if pred.timestamp.startswith(date[:10]):
                    result[code] = pred
                    break
        return result

    def _get_signals_for_date(
        self, signals: Dict[str, List[StrategySignalWrapper]], date: str
    ) -> Dict[str, StrategySignalWrapper]:
        result = {}
        for code, sigs in signals.items():
            for sig in sigs:
                if sig.timestamp.startswith(date[:10]):
                    result[code] = sig
                    break
        return result

    def _execute_decisions(
        self,
        decisions: List[FusionDecision],
        prices: pd.DataFrame,
        date: str,
        cash: float,
        source: str,
    ) -> float:
        positions = self._get_positions(source)
        trades = self._get_trades(source)
        
        for code, pos in list(positions.items()):
            if code in prices.index:
                current_price = prices.loc[code, "close"]
                if self._should_close_position(pos, current_price, date):
                    proceeds = self._close_position(pos, current_price, date, trades, source)
                    cash += proceeds
                    del positions[code]
        
        for decision in decisions:
            code = decision.signal.code
            
            if code not in prices.index:
                continue
            
            current_price = prices.loc[code, "close"]
            
            if decision.signal.action == "buy" and code not in positions:
                position_value = cash * decision.position_ratio
                shares = int(position_value / current_price / 100) * 100
                
                if shares > 0:
                    actual_price = current_price * (1 + self.slippage)
                    commission = shares * actual_price * self.commission_rate
                    
                    positions[code] = BacktestPosition(
                        code=code,
                        name=decision.signal.name,
                        direction="long",
                        entry_price=actual_price,
                        entry_date=date,
                        shares=shares,
                        position_value=shares * actual_price,
                        stop_loss=decision.stop_loss,
                        take_profit=decision.take_profit,
                        source=source,
                        decision=decision,
                    )
                    
                    cash -= shares * actual_price + commission
            
            elif decision.signal.action == "sell" and code in positions:
                pos = positions[code]
                proceeds = self._close_position(pos, current_price, date, trades, source)
                cash += proceeds
                del positions[code]
        
        return cash

    def _execute_model_only(
        self,
        predictions: Dict[str, ModelPrediction],
        prices: pd.DataFrame,
        date: str,
        cash: float,
    ) -> float:
        positions = self._model_positions
        trades = self._model_trades
        
        for code, pos in list(positions.items()):
            if code in prices.index:
                current_price = prices.loc[code, "close"]
                hold_days = (datetime.strptime(date, "%Y-%m-%d") - 
                           datetime.strptime(pos.entry_date, "%Y-%m-%d")).days
                if hold_days >= 5:
                    proceeds = self._close_position(pos, current_price, date, trades, "model")
                    cash += proceeds
                    del positions[code]
        
        for code, pred in predictions.items():
            if code not in prices.index:
                continue
            
            current_price = prices.loc[code, "close"]
            
            if pred.score > 0.6 and pred.confidence > 0.5 and code not in positions:
                position_value = cash * 0.2 * pred.confidence
                shares = int(position_value / current_price / 100) * 100
                
                if shares > 0:
                    actual_price = current_price * (1 + self.slippage)
                    commission = shares * actual_price * self.commission_rate
                    
                    positions[code] = BacktestPosition(
                        code=code,
                        name=pred.name,
                        direction="long",
                        entry_price=actual_price,
                        entry_date=date,
                        shares=shares,
                        position_value=shares * actual_price,
                        stop_loss=0.08,
                        take_profit=0.15,
                        source="model",
                    )
                    
                    cash -= shares * actual_price + commission
        
        return cash

    def _execute_strategy_only(
        self,
        signals: Dict[str, StrategySignalWrapper],
        prices: pd.DataFrame,
        date: str,
        cash: float,
    ) -> float:
        positions = self._strategy_positions
        trades = self._strategy_trades
        
        for code, pos in list(positions.items()):
            if code in prices.index:
                current_price = prices.loc[code, "close"]
                hold_days = (datetime.strptime(date, "%Y-%m-%d") - 
                           datetime.strptime(pos.entry_date, "%Y-%m-%d")).days
                if hold_days >= 10:
                    proceeds = self._close_position(pos, current_price, date, trades, "strategy")
                    cash += proceeds
                    del positions[code]
        
        for code, sig in signals.items():
            if code not in prices.index:
                continue
            
            current_price = prices.loc[code, "close"]
            
            if sig.action == "buy" and sig.strength > 0.5 and code not in positions:
                position_value = cash * 0.2 * sig.strength
                shares = int(position_value / current_price / 100) * 100
                
                if shares > 0:
                    actual_price = current_price * (1 + self.slippage)
                    commission = shares * actual_price * self.commission_rate
                    
                    positions[code] = BacktestPosition(
                        code=code,
                        name=sig.name,
                        direction="long",
                        entry_price=actual_price,
                        entry_date=date,
                        shares=shares,
                        position_value=shares * actual_price,
                        stop_loss=0.08,
                        take_profit=0.15,
                        source="strategy",
                    )
                    
                    cash -= shares * actual_price + commission
            
            elif sig.action == "sell" and code in positions:
                pos = positions[code]
                proceeds = self._close_position(pos, current_price, date, trades, "strategy")
                cash += proceeds
                del positions[code]
        
        return cash

    def _should_close_position(
        self, position: BacktestPosition, current_price: float, date: str
    ) -> bool:
        pnl_pct = (current_price - position.entry_price) / position.entry_price
        
        if pnl_pct <= -position.stop_loss:
            return True
        if pnl_pct >= position.take_profit:
            return True
        
        return False

    def _close_position(
        self,
        position: BacktestPosition,
        exit_price: float,
        exit_date: str,
        trades: List[BacktestTrade],
        source: str,
    ) -> float:
        actual_exit_price = exit_price * (1 - self.slippage)
        commission = position.shares * actual_exit_price * self.commission_rate
        
        proceeds = position.shares * actual_exit_price - commission
        
        pnl = (actual_exit_price - position.entry_price) * position.shares - commission
        pnl_pct = pnl / position.position_value
        
        hold_days = (datetime.strptime(exit_date, "%Y-%m-%d") - 
                    datetime.strptime(position.entry_date, "%Y-%m-%d")).days
        
        trade = BacktestTrade(
            code=position.code,
            name=position.name,
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=actual_exit_price,
            entry_date=position.entry_date,
            exit_date=exit_date,
            shares=position.shares,
            pnl=pnl,
            pnl_pct=pnl_pct,
            source=source,
            hold_days=hold_days,
            decision=position.decision,
        )
        
        trades.append(trade)
        
        return proceeds

    def _get_positions(self, source: str) -> Dict[str, BacktestPosition]:
        if source == "fused":
            return self._positions
        elif source == "model":
            return self._model_positions
        return self._strategy_positions

    def _get_trades(self, source: str) -> List[BacktestTrade]:
        if source == "fused":
            return self._trades
        elif source == "model":
            return self._model_trades
        return self._strategy_trades

    def _calculate_capital(
        self,
        cash: float,
        positions: Dict[str, BacktestPosition],
        prices: pd.DataFrame,
    ) -> float:
        position_value = 0
        for code, pos in positions.items():
            if code in prices.index:
                current_price = prices.loc[code, "close"]
                position_value += pos.shares * current_price
        return cash + position_value

    def _record_daily_value(
        self,
        date: str,
        capital_fused: float,
        capital_model: float,
        capital_strategy: float,
    ):
        self._daily_values.append({
            "date": date,
            "value": capital_fused,
        })
        self._model_daily_values.append({
            "date": date,
            "value": capital_model,
        })
        self._strategy_daily_values.append({
            "date": date,
            "value": capital_strategy,
        })

    def _calculate_result(
        self,
        source: str,
        trades: List[BacktestTrade],
        daily_values: List[Dict],
    ) -> BacktestResult:
        if not daily_values:
            return BacktestResult(
                source=source,
                total_return=0,
                annual_return=0,
                sharpe_ratio=0,
                max_drawdown=0,
                win_rate=0,
                total_trades=0,
                profit_trades=0,
                loss_trades=0,
                avg_profit=0,
                avg_loss=0,
                profit_factor=0,
                calmar_ratio=0,
                sortino_ratio=0,
            )
        
        df = pd.DataFrame(daily_values)
        df["return"] = df["value"].pct_change()
        
        total_return = (df["value"].iloc[-1] / self.initial_capital - 1)
        annual_return = df["return"].mean() * 252
        
        excess_returns = df["return"] - 0.03 / 252
        sharpe = excess_returns.mean() / df["return"].std() * np.sqrt(252) if df["return"].std() > 0 else 0
        
        cumulative = (1 + df["return"]).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        profit_trades = [t for t in trades if t.pnl > 0]
        loss_trades = [t for t in trades if t.pnl <= 0]
        
        win_rate = len(profit_trades) / len(trades) if trades else 0
        avg_profit = np.mean([t.pnl_pct for t in profit_trades]) if profit_trades else 0
        avg_loss = np.mean([t.pnl_pct for t in loss_trades]) if loss_trades else 0
        
        total_profit = sum(t.pnl for t in profit_trades)
        total_loss = abs(sum(t.pnl for t in loss_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        
        calmar = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        downside_returns = df["return"][df["return"] < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = annual_return / downside_std if downside_std > 0 else 0
        
        return BacktestResult(
            source=source,
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=len(trades),
            profit_trades=len(profit_trades),
            loss_trades=len(loss_trades),
            avg_profit=avg_profit,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            calmar_ratio=calmar,
            sortino_ratio=sortino,
            daily_returns=df["return"],
            trades=trades,
        )


class FusionBacktestReport:
    """融合回测报告生成器"""

    def __init__(self, output_dir: str = "data/fusion_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        fused_result: BacktestResult,
        model_result: BacktestResult,
        strategy_result: BacktestResult,
        gain_metrics: GainMetrics,
        save_json: bool = True,
        save_html: bool = True,
    ) -> Dict[str, Any]:
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": self._generate_summary(
                fused_result, model_result, strategy_result, gain_metrics
            ),
            "gain_analysis": self._generate_gain_analysis(gain_metrics),
            "fused_performance": fused_result.to_dict(),
            "model_performance": model_result.to_dict(),
            "strategy_performance": strategy_result.to_dict(),
            "conclusion": self._generate_conclusion(gain_metrics),
        }
        
        if save_json:
            json_path = self.output_dir / f"fusion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            report["json_path"] = str(json_path)
        
        if save_html:
            html_path = self._generate_html_report(report)
            report["html_path"] = str(html_path)
        
        return report

    def _generate_summary(
        self,
        fused: BacktestResult,
        model: BacktestResult,
        strategy: BacktestResult,
        gain: GainMetrics,
    ) -> Dict[str, Any]:
        return {
            "fusion_return": f"{fused.total_return * 100:.2f}%",
            "model_return": f"{model.total_return * 100:.2f}%",
            "strategy_return": f"{strategy.total_return * 100:.2f}%",
            "best_source": "fusion" if gain.return_gain > 0 else "model" if model.total_return > strategy.total_return else "strategy",
            "has_gain_effect": gain.has_gain(),
            "gain_summary": f"收益增益: {gain.return_gain * 100:.2f}%, 夏普增益: {gain.sharpe_gain:.2f}",
        }

    def _generate_gain_analysis(self, gain: GainMetrics) -> Dict[str, Any]:
        return {
            "return_gain": {
                "value": gain.return_gain,
                "percentage": f"{gain.return_gain * 100:.2f}%",
                "interpretation": "正向增益" if gain.return_gain > 0 else "负向增益",
            },
            "sharpe_gain": {
                "value": gain.sharpe_gain,
                "interpretation": "风险调整后收益提升" if gain.sharpe_gain > 0 else "风险调整后收益下降",
            },
            "drawdown_gain": {
                "value": gain.drawdown_gain,
                "percentage": f"{gain.drawdown_gain * 100:.2f}%",
                "interpretation": "回撤减少" if gain.drawdown_gain < 0 else "回撤增加",
            },
            "winrate_gain": {
                "value": gain.winrate_gain,
                "percentage": f"{gain.winrate_gain * 100:.2f}%",
                "interpretation": "胜率提升" if gain.winrate_gain > 0 else "胜率下降",
            },
            "information_ratio": gain.information_ratio,
            "calmar_gain": gain.calmar_gain,
            "sortino_gain": gain.sortino_gain,
        }

    def _generate_conclusion(self, gain: GainMetrics) -> Dict[str, Any]:
        positive_gains = sum([
            1 if gain.return_gain > 0 else 0,
            1 if gain.sharpe_gain > 0 else 0,
            1 if gain.drawdown_gain < 0 else 0,
            1 if gain.winrate_gain > 0 else 0,
        ])
        
        if positive_gains >= 3:
            conclusion = "融合策略显著优于单独使用模型或策略，增益效应明显"
            recommendation = "建议采用融合策略"
        elif positive_gains >= 2:
            conclusion = "融合策略在部分指标上优于单独使用，存在一定增益效应"
            recommendation = "建议继续优化融合参数"
        elif positive_gains >= 1:
            conclusion = "融合策略有轻微改善，但增益效应不显著"
            recommendation = "建议调整融合权重或增益计算方法"
        else:
            conclusion = "融合策略未能产生增益效应，需要重新设计融合机制"
            recommendation = "建议重新审视融合层设计"
        
        return {
            "positive_gain_count": positive_gains,
            "total_metrics": 4,
            "conclusion": conclusion,
            "recommendation": recommendation,
        }

    def _generate_html_report(self, report: Dict[str, Any]) -> Path:
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>融合策略回测报告</title>
    <style>
        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .summary-box {{
            background: #e8f5e9;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
        }}
        .metric-label {{
            color: #666;
            margin-top: 5px;
        }}
        .gain-positive {{
            color: #4CAF50;
        }}
        .gain-negative {{
            color: #f44336;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .conclusion-box {{
            background: #fff3e0;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
            border-left: 4px solid #ff9800;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 融合策略回测报告</h1>
        <p>生成时间: {report['generated_at']}</p>
        
        <div class="summary-box">
            <h2>📊 核心摘要</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{report['summary']['fusion_return']}</div>
                    <div class="metric-label">融合策略收益</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{report['summary']['model_return']}</div>
                    <div class="metric-label">模型单独收益</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{report['summary']['strategy_return']}</div>
                    <div class="metric-label">策略单独收益</div>
                </div>
            </div>
            <p><strong>增益效应:</strong> {report['summary']['gain_summary']}</p>
        </div>
        
        <h2>📈 增益分析</h2>
        <table>
            <tr>
                <th>指标</th>
                <th>增益值</th>
                <th>解读</th>
            </tr>
            <tr>
                <td>收益增益</td>
                <td class="{'gain-positive' if report['gain_analysis']['return_gain']['value'] > 0 else 'gain-negative'}">
                    {report['gain_analysis']['return_gain']['percentage']}
                </td>
                <td>{report['gain_analysis']['return_gain']['interpretation']}</td>
            </tr>
            <tr>
                <td>夏普增益</td>
                <td class="{'gain-positive' if report['gain_analysis']['sharpe_gain']['value'] > 0 else 'gain-negative'}">
                    {report['gain_analysis']['sharpe_gain']['value']:.2f}
                </td>
                <td>{report['gain_analysis']['sharpe_gain']['interpretation']}</td>
            </tr>
            <tr>
                <td>回撤增益</td>
                <td class="{'gain-positive' if report['gain_analysis']['drawdown_gain']['value'] < 0 else 'gain-negative'}">
                    {report['gain_analysis']['drawdown_gain']['percentage']}
                </td>
                <td>{report['gain_analysis']['drawdown_gain']['interpretation']}</td>
            </tr>
            <tr>
                <td>胜率增益</td>
                <td class="{'gain-positive' if report['gain_analysis']['winrate_gain']['value'] > 0 else 'gain-negative'}">
                    {report['gain_analysis']['winrate_gain']['percentage']}
                </td>
                <td>{report['gain_analysis']['winrate_gain']['interpretation']}</td>
            </tr>
        </table>
        
        <h2>📋 详细表现对比</h2>
        <table>
            <tr>
                <th>指标</th>
                <th>融合策略</th>
                <th>模型单独</th>
                <th>策略单独</th>
            </tr>
            <tr>
                <td>总收益</td>
                <td>{report['fused_performance']['total_return'] * 100:.2f}%</td>
                <td>{report['model_performance']['total_return'] * 100:.2f}%</td>
                <td>{report['strategy_performance']['total_return'] * 100:.2f}%</td>
            </tr>
            <tr>
                <td>年化收益</td>
                <td>{report['fused_performance']['annual_return'] * 100:.2f}%</td>
                <td>{report['model_performance']['annual_return'] * 100:.2f}%</td>
                <td>{report['strategy_performance']['annual_return'] * 100:.2f}%</td>
            </tr>
            <tr>
                <td>夏普比率</td>
                <td>{report['fused_performance']['sharpe_ratio']:.2f}</td>
                <td>{report['model_performance']['sharpe_ratio']:.2f}</td>
                <td>{report['strategy_performance']['sharpe_ratio']:.2f}</td>
            </tr>
            <tr>
                <td>最大回撤</td>
                <td>{report['fused_performance']['max_drawdown'] * 100:.2f}%</td>
                <td>{report['model_performance']['max_drawdown'] * 100:.2f}%</td>
                <td>{report['strategy_performance']['max_drawdown'] * 100:.2f}%</td>
            </tr>
            <tr>
                <td>胜率</td>
                <td>{report['fused_performance']['win_rate'] * 100:.2f}%</td>
                <td>{report['model_performance']['win_rate'] * 100:.2f}%</td>
                <td>{report['strategy_performance']['win_rate'] * 100:.2f}%</td>
            </tr>
            <tr>
                <td>交易次数</td>
                <td>{report['fused_performance']['total_trades']}</td>
                <td>{report['model_performance']['total_trades']}</td>
                <td>{report['strategy_performance']['total_trades']}</td>
            </tr>
        </table>
        
        <div class="conclusion-box">
            <h2>🎯 结论与建议</h2>
            <p><strong>结论:</strong> {report['conclusion']['conclusion']}</p>
            <p><strong>建议:</strong> {report['conclusion']['recommendation']}</p>
        </div>
    </div>
</body>
</html>
"""
        
        html_path = self.output_dir / f"fusion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        return html_path


def run_fusion_backtest(
    price_data: pd.DataFrame,
    model_predictions: Dict[str, List[ModelPrediction]],
    strategy_signals: Dict[str, List[StrategySignalWrapper]],
    market_contexts: Optional[Dict[str, MarketContext]] = None,
    config: Optional[FusionConfig] = None,
    initial_capital: float = 1000000,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    output_dir: str = "data/fusion_reports",
) -> Dict[str, Any]:
    """
    运行融合回测并生成报告
    
    Args:
        price_data: 价格数据，需包含 date, code, close 列
        model_predictions: 模型预测，按股票代码组织
        strategy_signals: 策略信号，按股票代码组织
        market_contexts: 市场环境，按日期组织
        config: 融合配置
        initial_capital: 初始资金
        start_date: 开始日期
        end_date: 结束日期
        output_dir: 报告输出目录
    
    Returns:
        包含回测结果和报告路径的字典
    """
    backtester = FusionBacktester(
        config=config,
        initial_capital=initial_capital,
    )
    
    fused_result, model_result, strategy_result, gain_metrics = backtester.run_backtest(
        price_data=price_data,
        model_predictions=model_predictions,
        strategy_signals=strategy_signals,
        market_contexts=market_contexts,
        start_date=start_date,
        end_date=end_date,
    )
    
    report_generator = FusionBacktestReport(output_dir=output_dir)
    report = report_generator.generate_report(
        fused_result=fused_result,
        model_result=model_result,
        strategy_result=strategy_result,
        gain_metrics=gain_metrics,
    )
    
    return {
        "fused_result": fused_result,
        "model_result": model_result,
        "strategy_result": strategy_result,
        "gain_metrics": gain_metrics,
        "report": report,
    }
