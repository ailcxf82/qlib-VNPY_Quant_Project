"""
Strategy Monitor - 策略监控器

长期监控策略表现，定期执行回测，检测策略退化，发送告警。
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from monitor.strategy_library.strategy_base import StrategyBase
    from monitor.strategy_library.backtest.backtest_engine import BacktestEngine, BacktestResult
    from monitor.strategy_library.backtest.performance_analyzer import PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class StrategyStatus:
    strategy_id: str
    strategy_name: str
    category: str
    
    last_backtest_date: str
    last_backtest_result: Optional[str]
    
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    
    status: str
    trend: str
    alerts: List[str]
    
    consecutive_declines: int = 0
    days_since_update: int = 0
    
    performance_history: List[Dict[str, float]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "category": self.category,
            "last_backtest_date": self.last_backtest_date,
            "last_backtest_result": self.last_backtest_result,
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "status": self.status,
            "trend": self.trend,
            "alerts": self.alerts,
            "consecutive_declines": self.consecutive_declines,
            "days_since_update": self.days_since_update,
            "performance_history": self.performance_history[-10:],
        }


@dataclass
class MonitoringConfig:
    enabled: bool = True
    backtest_interval: str = "weekly"
    backtest_day: int = 6
    
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "min_sharpe": 0.5,
        "max_drawdown": 0.25,
        "min_win_rate": 0.45,
        "decline_threshold": 0.2,
        "consecutive_declines": 3,
    })
    
    notification_channels: List[str] = field(default_factory=lambda: ["log"])
    webhook_url: Optional[str] = None
    
    lookback_days: int = 365
    min_trades: int = 10
    
    auto_disable_failing: bool = False
    auto_reoptimize: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "backtest_interval": self.backtest_interval,
            "backtest_day": self.backtest_day,
            "alert_thresholds": self.alert_thresholds,
            "notification_channels": self.notification_channels,
            "webhook_url": self.webhook_url,
            "lookback_days": self.lookback_days,
            "min_trades": self.min_trades,
            "auto_disable_failing": self.auto_disable_failing,
            "auto_reoptimize": self.auto_reoptimize,
        }


class StrategyMonitor:
    DEFAULT_STATUS_DIR = Path("data/strategy_library/monitoring")
    
    def __init__(
        self,
        config: Optional[MonitoringConfig] = None,
        status_dir: Optional[Path] = None,
    ):
        self.config = config or MonitoringConfig()
        self.status_dir = Path(status_dir) if status_dir else self.DEFAULT_STATUS_DIR
        self.status_dir.mkdir(parents=True, exist_ok=True)
        
        self._statuses: Dict[str, StrategyStatus] = {}
        self._backtest_engine: Optional["BacktestEngine"] = None
        self._data_loader: Optional[Callable] = None
        self._strategies: Dict[str, "StrategyBase"] = {}
        
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        self._load_statuses()
    
    def set_backtest_engine(self, engine: "BacktestEngine"):
        self._backtest_engine = engine
    
    def set_data_loader(self, loader: Callable):
        self._data_loader = loader
    
    def register_strategy(self, strategy: "StrategyBase"):
        with self._lock:
            self._strategies[strategy.strategy_id] = strategy
            
            if strategy.strategy_id not in self._statuses:
                self._statuses[strategy.strategy_id] = StrategyStatus(
                    strategy_id=strategy.strategy_id,
                    strategy_name=strategy.strategy_name,
                    category=getattr(strategy, "STRATEGY_CATEGORY", "general"),
                    last_backtest_date="",
                    last_backtest_result=None,
                    total_return=0.0,
                    annual_return=0.0,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    win_rate=0.0,
                    status="pending",
                    trend="unknown",
                    alerts=[],
                )
            
            logger.info(f"注册策略监控: {strategy.strategy_id}")
    
    def unregister_strategy(self, strategy_id: str):
        with self._lock:
            if strategy_id in self._strategies:
                del self._strategies[strategy_id]
            logger.info(f"注销策略监控: {strategy_id}")
    
    def get_status(self, strategy_id: str) -> Optional[StrategyStatus]:
        return self._statuses.get(strategy_id)
    
    def get_all_statuses(self) -> Dict[str, StrategyStatus]:
        return self._statuses.copy()
    
    def run_backtest(
        self,
        strategy_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional["BacktestResult"]:
        if self._backtest_engine is None:
            logger.error("回测引擎未设置")
            return None
        
        if strategy_id not in self._strategies:
            logger.error(f"策略未注册: {strategy_id}")
            return None
        
        strategy = self._strategies[strategy_id]
        
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=self.config.lookback_days)).strftime("%Y-%m-%d")
        
        from monitor.strategy_library.backtest.backtest_engine import BacktestConfig
        
        config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
        )
        self._backtest_engine.config = config
        
        if self._data_loader:
            codes = self._get_default_codes()
            data = self._backtest_engine.load_data(codes)
        else:
            logger.error("数据加载器未设置")
            return None
        
        result = self._backtest_engine.run(strategy, data)
        
        self._update_status(strategy_id, result)
        
        return result
    
    def run_all_backtests(self) -> Dict[str, "BacktestResult"]:
        results = {}
        
        for strategy_id in list(self._strategies.keys()):
            try:
                result = self.run_backtest(strategy_id)
                if result:
                    results[strategy_id] = result
            except Exception as e:
                logger.error(f"回测策略 {strategy_id} 失败: {e}")
        
        self._save_statuses()
        
        return results
    
    def check_alerts(self) -> Dict[str, List[str]]:
        alerts = {}
        
        for strategy_id, status in self._statuses.items():
            strategy_alerts = []
            thresholds = self.config.alert_thresholds
            
            if status.sharpe_ratio < thresholds["min_sharpe"]:
                strategy_alerts.append(
                    f"夏普比率 {status.sharpe_ratio:.2f} 低于阈值 {thresholds['min_sharpe']}"
                )
            
            if status.max_drawdown > thresholds["max_drawdown"]:
                strategy_alerts.append(
                    f"最大回撤 {status.max_drawdown:.2%} 超过阈值 {thresholds['max_drawdown']:.2%}"
                )
            
            if status.win_rate < thresholds["min_win_rate"]:
                strategy_alerts.append(
                    f"胜率 {status.win_rate:.2%} 低于阈值 {thresholds['min_win_rate']:.2%}"
                )
            
            if status.consecutive_declines >= thresholds["consecutive_declines"]:
                strategy_alerts.append(
                    f"连续 {status.consecutive_declines} 次绩效下降"
                )
            
            if status.trend == "declining":
                strategy_alerts.append("策略表现呈下降趋势")
            
            if strategy_alerts:
                alerts[strategy_id] = strategy_alerts
                status.alerts = strategy_alerts
                status.status = "warning" if len(strategy_alerts) <= 2 else "critical"
            else:
                status.alerts = []
                status.status = "active"
        
        return alerts
    
    def get_ranking(self, metric: str = "sharpe_ratio") -> List[Dict[str, Any]]:
        rankings = []
        
        for strategy_id, status in self._statuses.items():
            if status.last_backtest_date:
                rankings.append({
                    "strategy_id": strategy_id,
                    "strategy_name": status.strategy_name,
                    "category": status.category,
                    "metric_value": getattr(status, metric, 0),
                    "total_return": status.total_return,
                    "sharpe_ratio": status.sharpe_ratio,
                    "max_drawdown": status.max_drawdown,
                    "win_rate": status.win_rate,
                    "status": status.status,
                    "trend": status.trend,
                })
        
        rankings.sort(key=lambda x: x["metric_value"], reverse=True)
        
        for i, r in enumerate(rankings):
            r["rank"] = i + 1
        
        return rankings
    
    def get_performance_history(
        self,
        strategy_id: str,
        days: int = 90,
    ) -> pd.DataFrame:
        status = self._statuses.get(strategy_id)
        if not status or not status.performance_history:
            return pd.DataFrame()
        
        history = status.performance_history[-days:]
        
        df = pd.DataFrame(history)
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
        
        return df
    
    def start_monitoring(self):
        if self._running:
            logger.warning("监控已在运行")
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        logger.info("策略监控已启动")
    
    def stop_monitoring(self):
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        
        self._save_statuses()
        logger.info("策略监控已停止")
    
    def _monitor_loop(self):
        while self._running:
            try:
                now = datetime.now()
                
                should_run = False
                if self.config.backtest_interval == "daily":
                    should_run = True
                elif self.config.backtest_interval == "weekly":
                    if now.weekday() == self.config.backtest_day:
                        should_run = True
                elif self.config.backtest_interval == "monthly":
                    if now.day == 1:
                        should_run = True
                
                if should_run:
                    logger.info("执行定期回测...")
                    self.run_all_backtests()
                    self.check_alerts()
                    self._send_notifications()
                
                for status in self._statuses.values():
                    if status.last_backtest_date:
                        last_date = datetime.strptime(status.last_backtest_date, "%Y-%m-%d")
                        status.days_since_update = (now - last_date).days
                
            except Exception as e:
                logger.error(f"监控循环出错: {e}")
            
            time.sleep(3600)
    
    def _update_status(self, strategy_id: str, result: "BacktestResult"):
        status = self._statuses.get(strategy_id)
        if not status:
            return
        
        from monitor.strategy_library.backtest.performance_analyzer import PerformanceAnalyzer
        
        analyzer = PerformanceAnalyzer()
        metrics = analyzer.analyze(result)
        
        prev_sharpe = status.sharpe_ratio
        prev_return = status.total_return
        
        status.last_backtest_date = result.end_date
        status.last_backtest_result = result.backtest_id
        status.total_return = metrics.total_return
        status.annual_return = metrics.annual_return
        status.sharpe_ratio = metrics.sharpe_ratio
        status.max_drawdown = metrics.max_drawdown
        status.win_rate = metrics.win_rate
        status.days_since_update = 0
        
        status.performance_history.append({
            "date": result.end_date,
            "total_return": metrics.total_return,
            "annual_return": metrics.annual_return,
            "sharpe_ratio": metrics.sharpe_ratio,
            "max_drawdown": metrics.max_drawdown,
            "win_rate": metrics.win_rate,
        })
        
        if len(status.performance_history) > 100:
            status.performance_history = status.performance_history[-100:]
        
        if metrics.sharpe_ratio < prev_sharpe * (1 - self.config.alert_thresholds["decline_threshold"]):
            status.consecutive_declines += 1
            status.trend = "declining"
        elif metrics.sharpe_ratio > prev_sharpe * (1 + self.config.alert_thresholds["decline_threshold"]):
            status.consecutive_declines = 0
            status.trend = "improving"
        else:
            status.trend = "stable"
        
        if metrics.sharpe_ratio < self.config.alert_thresholds["min_sharpe"]:
            status.status = "warning"
        else:
            status.status = "active"
        
        logger.info(f"更新策略状态: {strategy_id}, 夏普={metrics.sharpe_ratio:.2f}, 趋势={status.trend}")
    
    def _send_notifications(self):
        alerts = self.check_alerts()
        
        if not alerts:
            return
        
        message_lines = ["策略监控告警:\n"]
        
        for strategy_id, strategy_alerts in alerts.items():
            status = self._statuses.get(strategy_id)
            if status:
                message_lines.append(f"\n【{status.strategy_name}】")
                for alert in strategy_alerts:
                    message_lines.append(f"  - {alert}")
        
        message = "\n".join(message_lines)
        
        if "log" in self.config.notification_channels:
            logger.warning(message)
        
        if "webhook" in self.config.notification_channels and self.config.webhook_url:
            self._send_webhook(message)
    
    def _send_webhook(self, message: str):
        import requests
        
        try:
            requests.post(
                self.config.webhook_url,
                json={"content": message},
                timeout=10,
            )
        except Exception as e:
            logger.error(f"发送webhook通知失败: {e}")
    
    def _load_statuses(self):
        status_file = self.status_dir / "strategy_status.json"
        
        if status_file.exists():
            try:
                with open(status_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                for strategy_id, status_data in data.items():
                    self._statuses[strategy_id] = StrategyStatus(
                        strategy_id=status_data.get("strategy_id", strategy_id),
                        strategy_name=status_data.get("strategy_name", ""),
                        category=status_data.get("category", "general"),
                        last_backtest_date=status_data.get("last_backtest_date", ""),
                        last_backtest_result=status_data.get("last_backtest_result"),
                        total_return=status_data.get("total_return", 0.0),
                        annual_return=status_data.get("annual_return", 0.0),
                        sharpe_ratio=status_data.get("sharpe_ratio", 0.0),
                        max_drawdown=status_data.get("max_drawdown", 0.0),
                        win_rate=status_data.get("win_rate", 0.0),
                        status=status_data.get("status", "pending"),
                        trend=status_data.get("trend", "unknown"),
                        alerts=status_data.get("alerts", []),
                        consecutive_declines=status_data.get("consecutive_declines", 0),
                        days_since_update=status_data.get("days_since_update", 0),
                        performance_history=status_data.get("performance_history", []),
                    )
                
                logger.info(f"加载策略状态: {len(self._statuses)} 个")
            except Exception as e:
                logger.error(f"加载策略状态失败: {e}")
    
    def _save_statuses(self):
        status_file = self.status_dir / "strategy_status.json"
        
        try:
            data = {
                strategy_id: status.to_dict()
                for strategy_id, status in self._statuses.items()
            }
            
            with open(status_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"保存策略状态: {len(self._statuses)} 个")
        except Exception as e:
            logger.error(f"保存策略状态失败: {e}")
    
    def _get_default_codes(self) -> List[str]:
        return [
            "000001.SZ", "000002.SZ", "000063.SZ",
            "000333.SZ", "000651.SZ", "000858.SZ",
            "002415.SZ", "002594.SZ",
            "600000.SH", "600036.SH", "600519.SH",
            "600887.SH", "601318.SH", "601398.SH",
        ]


def create_monitor_from_config(config_path: str) -> StrategyMonitor:
    import yaml
    
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
    
    monitoring_config = config_data.get("monitoring", {})
    
    config = MonitoringConfig(
        enabled=monitoring_config.get("enabled", True),
        backtest_interval=monitoring_config.get("backtest_interval", "weekly"),
        backtest_day=monitoring_config.get("backtest_day", 6),
        alert_thresholds=monitoring_config.get("alert_thresholds", {
            "min_sharpe": 0.5,
            "max_drawdown": 0.25,
            "min_win_rate": 0.45,
            "decline_threshold": 0.2,
            "consecutive_declines": 3,
        }),
        notification_channels=monitoring_config.get("notification_channels", ["log"]),
        webhook_url=monitoring_config.get("webhook_url"),
        lookback_days=monitoring_config.get("lookback_days", 365),
        min_trades=monitoring_config.get("min_trades", 10),
        auto_disable_failing=monitoring_config.get("auto_disable_failing", False),
        auto_reoptimize=monitoring_config.get("auto_reoptimize", False),
    )
    
    return StrategyMonitor(config=config)
