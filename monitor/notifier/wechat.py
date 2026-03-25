from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml

logger = logging.getLogger(__name__)


class WeChatNotifier:
    def __init__(self, config_path: str = "config/monitor.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.wechat_config = self.config.get("notification", {}).get("wechat", {})
        
        self.enabled = self.wechat_config.get("enabled", True)
        self._webhook_url = os.environ.get(
            "WECHAT_WEBHOOK_URL",
            self.wechat_config.get("webhook_url", "").replace("${WECHAT_WEBHOOK_URL}", "")
        )
        self.message_types = self.wechat_config.get("message_types", [
            "daily_signal", "position_change", "pnl_summary", "market_analysis"
        ])
        
        quiet_hours = self.wechat_config.get("quiet_hours", {})
        self.quiet_start = quiet_hours.get("start", "22:00")
        self.quiet_end = quiet_hours.get("end", "08:00")
    
    def _load_config(self) -> Dict[str, Any]:
        p = Path(self.config_path)
        if not p.exists():
            return {}
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    
    def _is_quiet_hours(self) -> bool:
        try:
            now = datetime.now().time()
            start = datetime.strptime(self.quiet_start, "%H:%M").time()
            end = datetime.strptime(self.quiet_end, "%H:%M").time()
            
            if start < end:
                return start <= now <= end
            else:
                return now >= start or now <= end
        except:
            return False
    
    def _send_message(self, content: str, msg_type: str = "text") -> bool:
        if not self.enabled:
            logger.info("企业微信通知已禁用")
            return False
        
        if not self._webhook_url:
            logger.warning("企业微信 webhook URL 未配置")
            return False
        
        if self._is_quiet_hours():
            logger.info("当前为静默时段，跳过推送")
            return False
        
        if msg_type not in self.message_types:
            logger.debug(f"消息类型 {msg_type} 未启用")
            return False
        
        try:
            data = {
                "msgtype": "markdown",
                "markdown": {
                    "content": content
                }
            }
            
            response = requests.post(
                self._webhook_url,
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            result = response.json()
            if result.get("errcode") == 0:
                logger.info(f"企业微信消息发送成功: {msg_type}")
                return True
            else:
                logger.error(f"企业微信消息发送失败: {result}")
                return False
        except Exception as e:
            logger.error(f"发送企业微信消息异常: {e}")
            return False
    
    def format_signal_message(self, signals: Dict[str, List], date: str) -> str:
        lines = [f"📊 **日线监听报告 - {date}**", ""]
        
        all_buy_signals = []
        for pool, sigs in signals.items():
            all_buy_signals.extend(sigs)
        
        if all_buy_signals:
            lines.append("🔴 **买入信号**")
            lines.append("| 代码 | 名称 | 排名 | 分数 | 策略 |")
            lines.append("|------|------|------|------|------|")
            for sig in all_buy_signals[:10]:
                lines.append(
                    f"| {sig.code} | {sig.name} | {sig.rank} | {sig.score:.2f} | {sig.strategy} |"
                )
            lines.append("")
        else:
            lines.append("⚪ 今日无买入信号")
            lines.append("")
        
        return "\n".join(lines)
    
    def format_position_message(self, sell_signals: List[Dict], trades: List[Any]) -> str:
        lines = ["🔄 **持仓变动**", ""]
        
        if sell_signals:
            lines.append("🟢 **卖出信号**")
            lines.append("| 代码 | 名称 | 原因 | 收益率 |")
            lines.append("|------|------|------|--------|")
            for sig in sell_signals:
                profit_str = f"{sig.get('profit_pct', 0):.2%}"
                lines.append(
                    f"| {sig.get('code')} | {sig.get('name')} | {sig.get('reason')} | {profit_str} |"
                )
            lines.append("")
        
        if trades:
            lines.append("📝 **成交记录**")
            for trade in trades[:5]:
                trade_type = "买入" if trade.trade_type == "buy" else "卖出"
                lines.append(
                    f"- {trade_type} {trade.code} {trade.name} "
                    f"{trade.shares}股 @ {trade.price:.2f}"
                )
            lines.append("")
        
        return "\n".join(lines)
    
    def format_pnl_message(self, portfolio: Dict) -> str:
        lines = ["💰 **账户状态**", ""]
        
        total_assets = portfolio.get("total_assets", 0)
        total_return = portfolio.get("total_return", 0)
        cash = portfolio.get("cash", 0)
        position_count = portfolio.get("position_count", 0)
        
        return_emoji = "📈" if total_return >= 0 else "📉"
        return_color = "green" if total_return >= 0 else "red"
        
        lines.append(f"总资产: ¥{total_assets:,.0f}")
        lines.append(f"总收益: {return_emoji} {total_return:.2%}")
        lines.append(f"现金: ¥{cash:,.0f}")
        lines.append(f"持仓股票: {position_count} 只")
        lines.append("")
        
        positions = portfolio.get("positions", [])
        if positions:
            lines.append("**当前持仓**")
            lines.append("| 代码 | 名称 | 盈亏 | 天数 |")
            lines.append("|------|------|------|------|")
            for pos in positions[:10]:
                profit_pct = pos.get("profit_pct", 0)
                profit_emoji = "🟢" if profit_pct >= 0 else "🔴"
                lines.append(
                    f"| {pos.get('code')} | {pos.get('name')} | "
                    f"{profit_emoji} {profit_pct:.2%} | {pos.get('holding_days')}天 |"
                )
        
        return "\n".join(lines)
    
    def format_market_analysis_message(self, sentiment: Any, llm_analysis: Dict) -> str:
        lines = ["📈 **市场分析**", ""]
        
        if hasattr(sentiment, 'overall_sentiment'):
            sentiment_emoji = {
                "positive": "🟢",
                "negative": "🔴",
                "neutral": "⚪"
            }.get(sentiment.overall_sentiment, "⚪")
            lines.append(f"市场情绪: {sentiment_emoji} {sentiment.overall_sentiment}")
            lines.append(f"情绪分数: {sentiment.sentiment_score:.2f}")
            lines.append(f"新闻数量: {sentiment.news_count} 条")
            lines.append("")
        
        market = llm_analysis.get("market_overview", {})
        if market.get("summary"):
            lines.append("**大盘走势**")
            lines.append(market.get("summary", "")[:200])
            lines.append("")
        
        trading = llm_analysis.get("trading_opportunity", {})
        if trading.get("recommendation"):
            lines.append("**交易建议**")
            lines.append(trading.get("recommendation", ""))
        
        return "\n".join(lines)
    
    def send_daily_report(
        self,
        signals: Dict[str, List],
        sell_signals: List[Dict],
        trades: List[Any],
        portfolio: Dict,
        sentiment: Any = None,
        llm_analysis: Dict = None,
    ) -> bool:
        date = datetime.now().strftime("%Y-%m-%d")
        
        results = []
        
        signal_msg = self.format_signal_message(signals, date)
        results.append(self._send_message(signal_msg, "daily_signal"))
        
        if sell_signals or trades:
            position_msg = self.format_position_message(sell_signals, trades)
            results.append(self._send_message(position_msg, "position_change"))
        
        pnl_msg = self.format_pnl_message(portfolio)
        results.append(self._send_message(pnl_msg, "pnl_summary"))
        
        if sentiment and llm_analysis:
            market_msg = self.format_market_analysis_message(sentiment, llm_analysis)
            results.append(self._send_message(market_msg, "market_analysis"))
        
        return any(results)
    
    def send_alert(self, title: str, content: str) -> bool:
        message = f"⚠️ **{title}**\n\n{content}"
        return self._send_message(message, "alert")
    
    def send_error(self, error_msg: str) -> bool:
        message = f"❌ **系统错误**\n\n{error_msg}"
        return self._send_message(message, "error")
    
    def send_message(self, content: str, msg_type: str = "text") -> bool:
        return self._send_message(content, msg_type)
