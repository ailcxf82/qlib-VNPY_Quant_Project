"""
Report Generator - 报告生成器

生成策略回测报告，包括HTML报告、Markdown报告、交易明细等。
"""

from __future__ import annotations

import base64
import io
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from monitor.strategy_library.backtest.backtest_engine import BacktestResult, TradeLog
    from monitor.strategy_library.backtest.performance_analyzer import PerformanceMetrics
    from monitor.strategy_library.backtest.strategy_monitor import StrategyStatus

logger = logging.getLogger(__name__)


@dataclass
class ReportConfig:
    output_dir: str = "data/strategy_library/reports"
    include_trades: bool = True
    include_charts: bool = True
    chart_format: str = "base64"
    language: str = "zh"


class ReportGenerator:
    DEFAULT_OUTPUT_DIR = Path("data/strategy_library/reports")
    
    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_backtest_report(
        self,
        result: "BacktestResult",
        metrics: Optional["PerformanceMetrics"] = None,
        output_path: Optional[str] = None,
    ) -> str:
        from monitor.strategy_library.backtest.performance_analyzer import PerformanceAnalyzer
        
        if metrics is None:
            analyzer = PerformanceAnalyzer()
            metrics = analyzer.analyze(result)
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(
                self.output_dir / result.strategy_id / f"report_{timestamp}.html"
            )
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        equity_chart = self._generate_equity_curve_chart(result)
        drawdown_chart = self._generate_drawdown_chart(result)
        monthly_chart = self._generate_monthly_returns_chart(metrics)
        
        html_content = self._render_html_report(result, metrics, equity_chart, drawdown_chart, monthly_chart)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        logger.info(f"生成回测报告: {output_path}")
        return output_path
    
    def generate_comparison_report(
        self,
        results: Dict[str, "BacktestResult"],
        output_path: Optional[str] = None,
    ) -> str:
        from monitor.strategy_library.backtest.performance_analyzer import PerformanceAnalyzer
        
        analyzer = PerformanceAnalyzer()
        
        comparison_df = analyzer.compare_strategies(results)
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(self.output_dir / f"comparison_{timestamp}.html")
        
        comparison_chart = self._generate_comparison_chart(comparison_df)
        
        html_content = self._render_comparison_html(results, comparison_df, comparison_chart)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        logger.info(f"生成对比报告: {output_path}")
        return output_path
    
    def generate_monitoring_report(
        self,
        statuses: Dict[str, "StrategyStatus"],
        output_path: Optional[str] = None,
    ) -> str:
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(self.output_dir / f"monitoring_{timestamp}.html")
        
        ranking_data = []
        for strategy_id, status in statuses.items():
            ranking_data.append({
                "strategy_id": strategy_id,
                "strategy_name": status.strategy_name,
                "sharpe_ratio": status.sharpe_ratio,
                "total_return": status.total_return,
                "max_drawdown": status.max_drawdown,
                "win_rate": status.win_rate,
                "status": status.status,
                "trend": status.trend,
                "alerts": len(status.alerts),
            })
        
        ranking_df = pd.DataFrame(ranking_data)
        if not ranking_df.empty:
            ranking_df = ranking_df.sort_values("sharpe_ratio", ascending=False)
        
        html_content = self._render_monitoring_html(statuses, ranking_df)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        logger.info(f"生成监控报告: {output_path}")
        return output_path
    
    def generate_trade_detail_report(
        self,
        trades: List["TradeLog"],
        output_path: Optional[str] = None,
    ) -> str:
        if not trades:
            logger.warning("没有交易记录")
            return ""
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(self.output_dir / f"trades_{timestamp}.csv")
        
        trades_data = [t.to_dict() for t in trades]
        df = pd.DataFrame(trades_data)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        
        logger.info(f"生成交易明细: {output_path}")
        return output_path
    
    def _generate_equity_curve_chart(self, result: "BacktestResult") -> str:
        if result.equity_curve.empty:
            return ""
        
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            dates = result.equity_curve.index
            portfolio_values = result.equity_curve["portfolio_value"]
            benchmark_values = result.equity_curve.get("benchmark_value", pd.Series())
            
            ax.plot(dates, portfolio_values, label="策略净值", linewidth=2, color="#2E86AB")
            
            if not benchmark_values.empty:
                initial_portfolio = portfolio_values.iloc[0]
                initial_benchmark = benchmark_values.iloc[0]
                normalized_benchmark = benchmark_values / initial_benchmark * initial_portfolio
                ax.plot(dates, normalized_benchmark, label="基准(沪深300)", linewidth=1.5, color="#A23B72", linestyle="--")
            
            ax.set_title("策略净值曲线", fontsize=14, fontweight="bold")
            ax.set_xlabel("日期", fontsize=12)
            ax.set_ylabel("净值", fontsize=12)
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3)
            
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"¥{x:,.0f}"))
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
            buffer.seek(0)
            chart_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return chart_base64
        except Exception as e:
            logger.error(f"生成净值曲线图失败: {e}")
            return ""
    
    def _generate_drawdown_chart(self, result: "BacktestResult") -> str:
        if result.equity_curve.empty:
            return ""
        
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            portfolio_values = result.equity_curve["portfolio_value"].values
            
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (peak - portfolio_values) / peak
            
            fig, ax = plt.subplots(figsize=(12, 4))
            
            ax.fill_between(
                result.equity_curve.index,
                drawdown * 100,
                0,
                color="#E74C3C",
                alpha=0.5,
                label="回撤"
            )
            
            ax.set_title("策略回撤曲线", fontsize=14, fontweight="bold")
            ax.set_xlabel("日期", fontsize=12)
            ax.set_ylabel("回撤 (%)", fontsize=12)
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}%"))
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
            buffer.seek(0)
            chart_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return chart_base64
        except Exception as e:
            logger.error(f"生成回撤曲线图失败: {e}")
            return ""
    
    def _generate_monthly_returns_chart(self, metrics: "PerformanceMetrics") -> str:
        if not metrics.monthly_returns:
            return ""
        
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            months = sorted(metrics.monthly_returns.keys())
            returns = [metrics.monthly_returns[m] * 100 for m in months]
            
            colors = ["#27AE60" if r >= 0 else "#E74C3C" for r in returns]
            
            fig, ax = plt.subplots(figsize=(14, 5))
            
            bars = ax.bar(range(len(months)), returns, color=colors, alpha=0.8)
            
            ax.set_title("月度收益分布", fontsize=14, fontweight="bold")
            ax.set_xlabel("月份", fontsize=12)
            ax.set_ylabel("收益率 (%)", fontsize=12)
            ax.set_xticks(range(len(months)))
            ax.set_xticklabels([m[5:] for m in months], rotation=45, ha="right")
            ax.axhline(y=0, color="black", linewidth=0.5)
            ax.grid(True, alpha=0.3, axis="y")
            
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}%"))
            
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
            buffer.seek(0)
            chart_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return chart_base64
        except Exception as e:
            logger.error(f"生成月度收益图失败: {e}")
            return ""
    
    def _generate_comparison_chart(self, comparison_df: pd.DataFrame) -> str:
        if comparison_df.empty:
            return ""
        
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            strategies = comparison_df["strategy_name"].tolist()
            x = range(len(strategies))
            
            ax1 = axes[0, 0]
            colors = ["#27AE60" if r >= 0 else "#E74C3C" for r in comparison_df["total_return"]]
            ax1.bar(x, comparison_df["total_return"] * 100, color=colors, alpha=0.8)
            ax1.set_title("总收益率对比", fontweight="bold")
            ax1.set_xticks(x)
            ax1.set_xticklabels(strategies, rotation=45, ha="right")
            ax1.axhline(y=0, color="black", linewidth=0.5)
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}%"))
            
            ax2 = axes[0, 1]
            colors = plt.cm.RdYlGn(comparison_df["sharpe_ratio"].values / comparison_df["sharpe_ratio"].max())
            ax2.bar(x, comparison_df["sharpe_ratio"], color=colors, alpha=0.8)
            ax2.set_title("夏普比率对比", fontweight="bold")
            ax2.set_xticks(x)
            ax2.set_xticklabels(strategies, rotation=45, ha="right")
            ax2.axhline(y=1, color="gray", linewidth=1, linestyle="--", label="基准线")
            
            ax3 = axes[1, 0]
            ax3.bar(x, comparison_df["max_drawdown"] * 100, color="#E74C3C", alpha=0.8)
            ax3.set_title("最大回撤对比", fontweight="bold")
            ax3.set_xticks(x)
            ax3.set_xticklabels(strategies, rotation=45, ha="right")
            ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}%"))
            
            ax4 = axes[1, 1]
            ax4.bar(x, comparison_df["win_rate"] * 100, color="#3498DB", alpha=0.8)
            ax4.set_title("胜率对比", fontweight="bold")
            ax4.set_xticks(x)
            ax4.set_xticklabels(strategies, rotation=45, ha="right")
            ax4.axhline(y=50, color="gray", linewidth=1, linestyle="--")
            ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.0f}%"))
            
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
            buffer.seek(0)
            chart_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return chart_base64
        except Exception as e:
            logger.error(f"生成对比图失败: {e}")
            return ""
    
    def _render_html_report(
        self,
        result: "BacktestResult",
        metrics: "PerformanceMetrics",
        equity_chart: str,
        drawdown_chart: str,
        monthly_chart: str,
    ) -> str:
        trades_html = ""
        if result.trades and self.config.include_trades:
            trades_html = self._render_trades_table(result.trades[:50])
        
        return f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>策略回测报告 - {result.strategy_name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; background: #f5f5f5; color: #333; line-height: 1.6; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #2E86AB 0%, #1a5276 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 20px; }}
        .header h1 {{ font-size: 28px; margin-bottom: 10px; }}
        .header .subtitle {{ opacity: 0.9; font-size: 14px; }}
        .card {{ background: white; border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .card h2 {{ font-size: 18px; margin-bottom: 15px; color: #2E86AB; border-bottom: 2px solid #2E86AB; padding-bottom: 10px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
        .metric-item {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2E86AB; }}
        .metric-value.positive {{ color: #27AE60; }}
        .metric-value.negative {{ color: #E74C3C; }}
        .metric-label {{ font-size: 12px; color: #666; margin-top: 5px; }}
        .chart-container {{ text-align: center; margin: 20px 0; }}
        .chart-container img {{ max-width: 100%; height: auto; border-radius: 8px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        tr:hover {{ background: #f8f9fa; }}
        .tag {{ display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: 500; }}
        .tag-buy {{ background: #d4edda; color: #155724; }}
        .tag-sell {{ background: #f8d7da; color: #721c24; }}
        .summary-row {{ display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #eee; }}
        .summary-label {{ color: #666; }}
        .summary-value {{ font-weight: 600; }}
        .footer {{ text-align: center; padding: 20px; color: #666; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 策略回测报告</h1>
            <div class="subtitle">
                {result.strategy_name} | 回测区间: {result.start_date} ~ {result.end_date}
            </div>
        </div>
        
        <div class="card">
            <h2>📈 核心指标</h2>
            <div class="metrics-grid">
                <div class="metric-item">
                    <div class="metric-value {'positive' if result.total_return >= 0 else 'negative'}">{result.total_return:.2%}</div>
                    <div class="metric-label">总收益率</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value {'positive' if result.annual_return >= 0 else 'negative'}">{result.annual_return:.2%}</div>
                    <div class="metric-label">年化收益</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">{result.sharpe_ratio:.2f}</div>
                    <div class="metric-label">夏普比率</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value negative">{result.max_drawdown:.2%}</div>
                    <div class="metric-label">最大回撤</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">{result.win_rate:.1%}</div>
                    <div class="metric-label">胜率</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">{result.profit_factor:.2f}</div>
                    <div class="metric-label">盈亏比</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>📊 收益风险分析</h2>
            <div class="summary-row">
                <span class="summary-label">基准收益(沪深300)</span>
                <span class="summary-value">{result.benchmark_return:.2%}</span>
            </div>
            <div class="summary-row">
                <span class="summary-label">超额收益</span>
                <span class="summary-value {'positive' if result.excess_return >= 0 else 'negative'}">{result.excess_return:.2%}</span>
            </div>
            <div class="summary-row">
                <span class="summary-label">索提诺比率</span>
                <span class="summary-value">{metrics.sortino_ratio:.2f}</span>
            </div>
            <div class="summary-row">
                <span class="summary-label">卡玛比率</span>
                <span class="summary-value">{metrics.calmar_ratio:.2f}</span>
            </div>
            <div class="summary-row">
                <span class="summary-label">Alpha</span>
                <span class="summary-value">{metrics.alpha:.2%}</span>
            </div>
            <div class="summary-row">
                <span class="summary-label">Beta</span>
                <span class="summary-value">{metrics.beta:.2f}</span>
            </div>
            <div class="summary-row">
                <span class="summary-label">年化波动率</span>
                <span class="summary-value">{metrics.volatility:.2%}</span>
            </div>
            <div class="summary-row">
                <span class="summary-label">VaR(95%)</span>
                <span class="summary-value negative">{metrics.var_95:.2%}</span>
            </div>
        </div>
        
        <div class="card">
            <h2>📉 净值曲线</h2>
            <div class="chart-container">
                <img src="data:image/png;base64,{equity_chart}" alt="净值曲线">
            </div>
        </div>
        
        <div class="card">
            <h2>📉 回撤分析</h2>
            <div class="chart-container">
                <img src="data:image/png;base64,{drawdown_chart}" alt="回撤曲线">
            </div>
            <div class="summary-row">
                <span class="summary-label">最大回撤持续时间</span>
                <span class="summary-value">{result.max_drawdown_duration} 天</span>
            </div>
            <div class="summary-row">
                <span class="summary-label">平均回撤</span>
                <span class="summary-value">{metrics.avg_drawdown:.2%}</span>
            </div>
        </div>
        
        <div class="card">
            <h2>📅 月度收益</h2>
            <div class="chart-container">
                <img src="data:image/png;base64,{monthly_chart}" alt="月度收益">
            </div>
        </div>
        
        <div class="card">
            <h2>🎯 交易统计</h2>
            <div class="metrics-grid">
                <div class="metric-item">
                    <div class="metric-value">{result.total_trades}</div>
                    <div class="metric-label">总交易次数</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value positive">{result.winning_trades}</div>
                    <div class="metric-label">盈利次数</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value negative">{result.losing_trades}</div>
                    <div class="metric-label">亏损次数</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">{metrics.avg_holding_days:.1f}</div>
                    <div class="metric-label">平均持仓天数</div>
                </div>
            </div>
        </div>
        
        {trades_html}
        
        <div class="card">
            <h2>⚙️ 回测配置</h2>
            <div class="summary-row">
                <span class="summary-label">初始资金</span>
                <span class="summary-value">¥{result.initial_capital:,.0f}</span>
            </div>
            <div class="summary-row">
                <span class="summary-label">最终资金</span>
                <span class="summary-value">¥{result.final_capital:,.0f}</span>
            </div>
            <div class="summary-row">
                <span class="summary-label">手续费率</span>
                <span class="summary-value">{result.config.commission_rate:.2%}</span>
            </div>
            <div class="summary-row">
                <span class="summary-label">印花税</span>
                <span class="summary-value">{result.config.stamp_duty:.2%}</span>
            </div>
            <div class="summary-row">
                <span class="summary-label">滑点</span>
                <span class="summary-value">{result.config.slippage:.2%}</span>
            </div>
            <div class="summary-row">
                <span class="summary-label">执行时间</span>
                <span class="summary-value">{result.execution_time:.2f}秒</span>
            </div>
        </div>
        
        <div class="footer">
            <p>报告生成时间: {result.created_at}</p>
            <p>回测ID: {result.backtest_id}</p>
        </div>
    </div>
</body>
</html>
"""
    
    def _render_trades_table(self, trades: List["TradeLog"]) -> str:
        if not trades:
            return ""
        
        rows = []
        for trade in trades:
            action_class = "tag-buy" if trade.action == "buy" else "tag-sell"
            rows.append(f"""
                <tr>
                    <td>{trade.date}</td>
                    <td>{trade.code}</td>
                    <td>{trade.name}</td>
                    <td><span class="tag {action_class}">{trade.action.upper()}</span></td>
                    <td>¥{trade.price:.2f}</td>
                    <td>{trade.shares:,}</td>
                    <td>¥{trade.amount:,.2f}</td>
                    <td>¥{trade.commission:.2f}</td>
                    <td title="{trade.reason}">{trade.reason[:30]}...</td>
                </tr>
            """)
        
        return f"""
        <div class="card">
            <h2>📋 交易明细 (最近{len(trades)}笔)</h2>
            <table>
                <thead>
                    <tr>
                        <th>日期</th>
                        <th>代码</th>
                        <th>名称</th>
                        <th>方向</th>
                        <th>价格</th>
                        <th>数量</th>
                        <th>金额</th>
                        <th>手续费</th>
                        <th>原因</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(rows)}
                </tbody>
            </table>
        </div>
        """
    
    def _render_comparison_html(
        self,
        results: Dict[str, "BacktestResult"],
        comparison_df: pd.DataFrame,
        comparison_chart: str,
    ) -> str:
        strategy_rows = []
        for _, row in comparison_df.iterrows():
            strategy_rows.append(f"""
                <tr>
                    <td>{row['strategy_name']}</td>
                    <td class="{'positive' if row['total_return'] >= 0 else 'negative'}">{row['total_return']:.2%}</td>
                    <td>{row['sharpe_ratio']:.2f}</td>
                    <td class="negative">{row['max_drawdown']:.2%}</td>
                    <td>{row['win_rate']:.1%}</td>
                    <td>{row['profit_factor']:.2f}</td>
                    <td>{row['total_trades']}</td>
                </tr>
            """)
        
        return f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>策略对比报告</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f5f5f5; color: #333; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #8E44AD 0%, #5B2C6F 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 20px; }}
        .header h1 {{ font-size: 28px; margin-bottom: 10px; }}
        .card {{ background: white; border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .card h2 {{ font-size: 18px; margin-bottom: 15px; color: #8E44AD; border-bottom: 2px solid #8E44AD; padding-bottom: 10px; }}
        .chart-container {{ text-align: center; margin: 20px 0; }}
        .chart-container img {{ max-width: 100%; height: auto; border-radius: 8px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        tr:hover {{ background: #f8f9fa; }}
        .positive {{ color: #27AE60; font-weight: 600; }}
        .negative {{ color: #E74C3C; font-weight: 600; }}
        .footer {{ text-align: center; padding: 20px; color: #666; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 策略对比报告</h1>
            <div class="subtitle">对比 {len(results)} 个策略的回测表现</div>
        </div>
        
        <div class="card">
            <h2>📈 策略对比图表</h2>
            <div class="chart-container">
                <img src="data:image/png;base64,{comparison_chart}" alt="策略对比">
            </div>
        </div>
        
        <div class="card">
            <h2>📋 策略排名</h2>
            <table>
                <thead>
                    <tr>
                        <th>策略名称</th>
                        <th>总收益</th>
                        <th>夏普比率</th>
                        <th>最大回撤</th>
                        <th>胜率</th>
                        <th>盈亏比</th>
                        <th>交易次数</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(strategy_rows)}
                </tbody>
            </table>
        </div>
        
        <div class="footer">
            <p>报告生成时间: {datetime.now().isoformat()}</p>
        </div>
    </div>
</body>
</html>
"""
    
    def _render_monitoring_html(
        self,
        statuses: Dict[str, "StrategyStatus"],
        ranking_df: pd.DataFrame,
    ) -> str:
        status_rows = []
        for _, row in ranking_df.iterrows():
            status_class = {
                "active": "status-active",
                "warning": "status-warning",
                "critical": "status-critical",
            }.get(row["status"], "status-pending")
            
            trend_icon = {
                "improving": "📈",
                "stable": "➡️",
                "declining": "📉",
            }.get(row["trend"], "❓")
            
            status_rows.append(f"""
                <tr>
                    <td>{row['strategy_name']}</td>
                    <td><span class="status {status_class}">{row['status'].upper()}</span></td>
                    <td>{row['sharpe_ratio']:.2f}</td>
                    <td class="{'positive' if row['total_return'] >= 0 else 'negative'}">{row['total_return']:.2%}</td>
                    <td class="negative">{row['max_drawdown']:.2%}</td>
                    <td>{row['win_rate']:.1%}</td>
                    <td>{trend_icon} {row['trend']}</td>
                    <td>{row['alerts']}</td>
                </tr>
            """)
        
        return f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>策略监控报告</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f5f5f5; color: #333; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #E74C3C 0%, #C0392B 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 20px; }}
        .header h1 {{ font-size: 28px; margin-bottom: 10px; }}
        .card {{ background: white; border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .card h2 {{ font-size: 18px; margin-bottom: 15px; color: #E74C3C; border-bottom: 2px solid #E74C3C; padding-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        tr:hover {{ background: #f8f9fa; }}
        .positive {{ color: #27AE60; font-weight: 600; }}
        .negative {{ color: #E74C3C; font-weight: 600; }}
        .status {{ padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }}
        .status-active {{ background: #d4edda; color: #155724; }}
        .status-warning {{ background: #fff3cd; color: #856404; }}
        .status-critical {{ background: #f8d7da; color: #721c24; }}
        .status-pending {{ background: #e2e3e5; color: #383d41; }}
        .footer {{ text-align: center; padding: 20px; color: #666; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔔 策略监控报告</h1>
            <div class="subtitle">监控 {len(statuses)} 个策略的实时表现</div>
        </div>
        
        <div class="card">
            <h2>📊 策略状态概览</h2>
            <table>
                <thead>
                    <tr>
                        <th>策略名称</th>
                        <th>状态</th>
                        <th>夏普比率</th>
                        <th>总收益</th>
                        <th>最大回撤</th>
                        <th>胜率</th>
                        <th>趋势</th>
                        <th>告警数</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(status_rows)}
                </tbody>
            </table>
        </div>
        
        <div class="footer">
            <p>报告生成时间: {datetime.now().isoformat()}</p>
        </div>
    </div>
</body>
</html>
"""
