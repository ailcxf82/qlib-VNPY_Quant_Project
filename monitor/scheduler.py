from __future__ import annotations

import logging
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


class MonitorScheduler:
    def __init__(self, config_path: str = "config/monitor.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.scheduler_config = self.config.get("scheduler", {})
        
        self.enabled = self.scheduler_config.get("enabled", True)
        self.schedule_time = self.scheduler_config.get("schedule_time", "15:05")
        self.timezone = self.scheduler_config.get("timezone", "Asia/Shanghai")
        self.retry_times = self.scheduler_config.get("retry_times", 3)
        self.retry_interval = self.scheduler_config.get("retry_interval", 300)
        
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._sentiment_thread: Optional[threading.Thread] = None
        self._last_run_date: Optional[str] = None
        self._lock = threading.Lock()
        
        self.signal_engine = None
        self.position_tracker = None
        self.sentiment_analyzer = None
        self.llm_analyzer = None
        self.notifier = None
        self.scorer = None
        self.enhanced_sentiment = None
        self.technical_analyzer = None
        self.extreme_detector = None
        
        self._stock_pool: List[Dict] = []
        self._stock_scores: Dict[str, Dict] = {}
        self._sentiment_schedule_times: List[str] = []
        self._last_sentiment_times: Dict[str, str] = {}
        
        self._init_components()
        self._init_schedules()
    
    def _load_config(self) -> Dict[str, Any]:
        p = Path(self.config_path)
        if not p.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    
    def _init_components(self):
        from monitor.signal_engine import SignalEngine
        from monitor.position_tracker import PositionTracker
        from monitor.sentiment_analyzer import SentimentAnalyzer
        from monitor.llm_analyzer import LLMAnalyzer
        from monitor.notifier.wechat import WeChatNotifier
        from monitor.scorer import MultiDimensionScorer
        from monitor.enhanced_sentiment import EnhancedSentimentAnalyzer
        from monitor.technical_analyzer import TechnicalAnalyzer
        from monitor.extreme_detector import ExtremeConditionDetector
        
        self.signal_engine = SignalEngine(self.config_path)
        self.position_tracker = PositionTracker(self.config_path)
        self.sentiment_analyzer = SentimentAnalyzer(self.config_path)
        self.llm_analyzer = LLMAnalyzer(self.config_path)
        self.notifier = WeChatNotifier(self.config_path)
        
        scoring_config = self.config.get("scoring", {})
        if scoring_config.get("enabled", True):
            self.scorer = MultiDimensionScorer(self.config_path)
            logger.info("多维度评分器初始化完成")
        
        sentiment_schedule_config = self.config.get("sentiment_schedule", {})
        if sentiment_schedule_config.get("enabled", True):
            self.enhanced_sentiment = EnhancedSentimentAnalyzer(self.config_path)
            logger.info("增强舆情分析器初始化完成")
        
        technical_config = self.config.get("technical_analysis", {})
        if technical_config.get("enabled", True):
            self.technical_analyzer = TechnicalAnalyzer(self.config_path)
            logger.info("技术分析器初始化完成")
        
        extreme_config = self.config.get("extreme_detection", {})
        if extreme_config.get("enabled", True):
            self.extreme_detector = ExtremeConditionDetector(self.config_path)
            logger.info("极端情况检测器初始化完成")
        
        logger.info("监听系统组件初始化完成")
    
    def _init_schedules(self):
        sentiment_schedule = self.config.get("sentiment_schedule", {})
        if sentiment_schedule.get("enabled", True):
            self._sentiment_schedule_times = sentiment_schedule.get("times", [
                "09:15", "10:30", "13:30", "14:45"
            ])
            logger.info(f"舆情分析定时任务: {self._sentiment_schedule_times}")
    
    def _should_run_today(self) -> bool:
        today = datetime.now().strftime("%Y-%m-%d")
        if self._last_run_date == today:
            return False
        return True
    
    def _is_schedule_time(self) -> bool:
        now = datetime.now()
        current_time = now.strftime("%H:%M")
        return current_time >= self.schedule_time
    
    def _is_sentiment_time(self) -> bool:
        now = datetime.now()
        current_time = now.strftime("%H:%M")
        current_date = now.strftime("%Y-%m-%d")
        
        for schedule_time in self._sentiment_schedule_times:
            time_key = f"{current_date}_{schedule_time}"
            if current_time >= schedule_time:
                if self._last_sentiment_times.get(schedule_time) != current_date:
                    self._last_sentiment_times[schedule_time] = current_date
                    return True
        return False
    
    def generate_stock_pool(self, date: Optional[str] = None) -> List[Dict]:
        logger.info("生成待选股票池...")
        
        signals = self.signal_engine.generate_all_signals(date)
        all_stocks = []
        
        for pool, sigs in signals.items():
            for sig in sigs:
                stock_info = {
                    "code": sig.code,
                    "name": sig.name,
                    "pool": pool,
                    "prediction_score": sig.score,
                    "reason": sig.reason,
                }
                all_stocks.append(stock_info)
        
        if self.scorer:
            scored_stocks = []
            for stock in all_stocks:
                scores = self.calculate_stock_scores(stock)
                stock.update(scores)
                scored_stocks.append(stock)
            
            scored_stocks.sort(key=lambda x: x.get("total_score", 0), reverse=True)
            
            pool_config = self.config.get("stock_pool", {})
            max_size = pool_config.get("max_pool_size", 50)
            min_score = pool_config.get("min_score", 0.50)
            
            scored_stocks = [s for s in scored_stocks if s.get("total_score", 0) >= min_score]
            scored_stocks = scored_stocks[:max_size]
            
            self._stock_pool = scored_stocks
            self._stock_scores = {s["code"]: s for s in scored_stocks}
            
            logger.info(f"股票池生成完成: {len(scored_stocks)} 只股票")
            return scored_stocks
        
        self._stock_pool = all_stocks
        return all_stocks
    
    def calculate_stock_scores(self, stock: Dict) -> Dict[str, float]:
        if not self.scorer:
            return {"total_score": stock.get("prediction_score", 0.5)}
        
        scores = {
            "model_score": stock.get("prediction_score", 0.5),
            "sentiment_score": 0.5,
            "strategy_score": 0.5,
            "technical_score": 0.5,
        }
        
        if self.enhanced_sentiment:
            sentiment_result = self.enhanced_sentiment.analyze_stock_sentiment(
                stock["code"], stock["name"]
            )
            scores["sentiment_score"] = sentiment_result.get("sentiment_score", 0.5)
            scores["sentiment_data"] = sentiment_result
        
        if self.technical_analyzer:
            tech_indicators = self.technical_analyzer.analyze(stock["code"])
            tech_score = self.technical_analyzer.calculate_score(tech_indicators)
            scores["technical_score"] = tech_score.total_score
            scores["technical_data"] = tech_score.to_dict()
        
        total_score = self.scorer.calculate_total_score(
            model_score=scores["model_score"],
            sentiment_score=scores["sentiment_score"],
            strategy_score=scores["strategy_score"],
            technical_score=scores["technical_score"],
        )
        scores["total_score"] = total_score
        
        return scores
    
    def run_sentiment_analysis(self) -> Dict[str, Any]:
        logger.info("执行定时舆情分析...")
        
        results = {
            "time": datetime.now().isoformat(),
            "position_sentiments": [],
            "extreme_alerts": [],
        }
        
        if not self.enhanced_sentiment:
            return results
        
        positions = self.position_tracker.positions
        for code, position in positions.items():
            try:
                sentiment_result = self.enhanced_sentiment.analyze_stock_sentiment(
                    code, position.name
                )
                results["position_sentiments"].append({
                    "code": code,
                    "name": position.name,
                    "sentiment": sentiment_result,
                })
                
                if sentiment_result.get("is_extreme"):
                    results["extreme_alerts"].append({
                        "code": code,
                        "name": position.name,
                        "type": sentiment_result.get("extreme_type"),
                        "score": sentiment_result.get("sentiment_score"),
                    })
            except Exception as e:
                logger.error(f"分析 {code} 舆情失败: {e}")
        
        if results["extreme_alerts"]:
            logger.warning(f"发现 {len(results['extreme_alerts'])} 个极端舆情警报")
            self._handle_extreme_sentiment_alerts(results["extreme_alerts"])
        
        return results
    
    def _handle_extreme_sentiment_alerts(self, alerts: List[Dict]):
        if not self.extreme_detector:
            return
        
        for alert in alerts:
            code = alert["code"]
            name = alert["name"]
            sentiment_type = alert["type"]
            
            logger.info(f"检查 {code} 极端情况确认...")
            
            confirmation = self.extreme_detector.confirm_extreme_condition(
                code, sentiment_type
            )
            
            if confirmation.get("confirmed"):
                logger.warning(f"{code} 极端情况已确认: {confirmation.get('reason')}")
                
                extreme_config = self.config.get("extreme_detection", {})
                auto_sell_config = extreme_config.get("auto_sell", {})
                
                if auto_sell_config.get("enabled", True):
                    self._execute_extreme_sell(code, name, confirmation)
    
    def _execute_extreme_sell(self, code: str, name: str, confirmation: Dict):
        position = self.position_tracker.positions.get(code)
        if not position:
            return
        
        extreme_config = self.config.get("extreme_detection", {})
        auto_sell_config = extreme_config.get("auto_sell", {})
        min_holding_days = auto_sell_config.get("min_holding_days", 1)
        
        holding_days = (datetime.now() - position.buy_time).days
        if holding_days < min_holding_days:
            logger.info(f"{code} 持仓天数不足 {min_holding_days} 天，暂不执行极端卖出")
            return
        
        price = self.position_tracker.get_current_price(code)
        if price <= 0:
            logger.warning(f"无法获取 {code} 价格，跳过极端卖出")
            return
        
        reason = f"极端情况卖出: {confirmation.get('reason', '未知原因')}"
        
        trade = self.position_tracker.sell(
            code=code,
            price=price,
            reason=reason,
        )
        
        if trade:
            logger.info(f"极端情况卖出执行成功: {code} @ {price}")
            
            notification_config = extreme_config.get("notification", {})
            if notification_config.get("enabled", True):
                self.notifier.send_message(
                    f"⚠️ 极端情况卖出警报\n\n"
                    f"股票: {name} ({code})\n"
                    f"卖出价格: ¥{price:.2f}\n"
                    f"原因: {confirmation.get('reason', '未知')}\n"
                    f"技术确认: {confirmation.get('technical_reason', '无')}"
                )
    
    def check_extreme_conditions(self) -> List[Dict]:
        if not self.extreme_detector:
            return []
        
        positions = self.position_tracker.positions
        sell_recommendations = []
        
        for code, position in positions.items():
            sentiment_data = None
            if self.enhanced_sentiment:
                sentiment_data = self.enhanced_sentiment.get_cached_sentiment(code)
            
            recommendations = self.extreme_detector.get_sell_recommendations(
                {code: position}, 
                {code: sentiment_data} if sentiment_data else {}
            )
            sell_recommendations.extend(recommendations)
        
        return sell_recommendations
    
    def run_monitor_cycle(self, date: Optional[str] = None) -> Dict[str, Any]:
        logger.info("=" * 50)
        logger.info("开始日线监听任务")
        
        results = {
            "start_time": datetime.now().isoformat(),
            "date": date or datetime.now().strftime("%Y-%m-%d"),
            "success": False,
            "steps": {},
        }
        
        try:
            logger.info("[1/8] 生成待选股票池...")
            stock_pool = self.generate_stock_pool(date)
            results["steps"]["stock_pool"] = {
                "success": True,
                "pool_size": len(stock_pool),
                "top_stocks": stock_pool[:5] if stock_pool else [],
            }
            logger.info(f"股票池生成完成: {len(stock_pool)} 只股票")
            
            logger.info("[2/8] 检查极端情况...")
            extreme_sells = self.check_extreme_conditions()
            results["steps"]["extreme_check"] = {
                "success": True,
                "recommendations": extreme_sells,
            }
            if extreme_sells:
                logger.warning(f"发现 {len(extreme_sells)} 个极端卖出建议")
            
            logger.info("[3/8] 检查卖出信号...")
            sell_signals = self.position_tracker.check_sell_signals()
            all_sell_signals = sell_signals + [
                {
                    "code": s["code"],
                    "name": s["name"],
                    "reason": s["reason"],
                    "type": "extreme",
                }
                for s in extreme_sells
            ]
            results["steps"]["sell_signals"] = {
                "success": True,
                "count": len(all_sell_signals),
                "signals": all_sell_signals,
            }
            logger.info(f"发现 {len(all_sell_signals)} 个卖出信号")
            
            logger.info("[4/8] 执行卖出...")
            sell_trades = self.position_tracker.execute_sell_signals(sell_signals)
            for rec in extreme_sells:
                if rec.get("auto_sell"):
                    price = self.position_tracker.get_current_price(rec["code"])
                    if price > 0:
                        trade = self.position_tracker.sell(
                            code=rec["code"],
                            price=price,
                            reason=rec["reason"],
                        )
                        if trade:
                            sell_trades.append(trade)
            
            results["steps"]["sell_execution"] = {
                "success": True,
                "trades": [t.to_dict() for t in sell_trades],
            }
            logger.info(f"执行卖出完成: {len(sell_trades)} 笔")
            
            logger.info("[5/8] 执行买入...")
            all_buy_signals = []
            for stock in stock_pool:
                if len(self.position_tracker.positions) >= self.position_tracker.position_config.get("max_stocks", 10):
                    break
                
                code = stock["code"]
                if code in self.position_tracker.positions:
                    continue
                
                price = self.position_tracker.get_current_price(code)
                if price <= 0:
                    logger.warning(f"无法获取 {code} 的价格，跳过买入")
                    continue
                
                pool = stock.get("pool", "default")
                strategy_config = self.config.get("strategies", {}).get(pool, {})
                position_size = strategy_config.get("position_size", 0.1)
                
                buy_reason = f"多维度评分: {stock.get('total_score', 0):.2f}"
                
                trade = self.position_tracker.buy(
                    code=code,
                    name=stock["name"],
                    price=price,
                    strategy=pool,
                    reason=buy_reason,
                    position_size=position_size,
                )
                if trade:
                    all_buy_signals.append(stock)
            
            results["steps"]["buy_execution"] = {
                "success": True,
                "trades": len(all_buy_signals),
            }
            logger.info(f"执行买入完成: {len(all_buy_signals)} 笔")
            
            logger.info("[6/8] 获取市场舆情...")
            sentiment = self.sentiment_analyzer.analyze()
            results["steps"]["sentiment"] = {
                "success": True,
                "sentiment": sentiment.overall_sentiment,
                "score": sentiment.sentiment_score,
            }
            logger.info(f"舆情分析完成: {sentiment.overall_sentiment}")
            
            logger.info("[7/8] LLM 金融分析...")
            portfolio = self.position_tracker.get_portfolio_summary()
            llm_signals = []
            for stock in stock_pool[:10]:
                llm_signals.append({
                    "code": stock["code"],
                    "name": stock["name"],
                    "score": stock.get("total_score", 0),
                    "reason": stock.get("reason", ""),
                })
            
            llm_analysis = self.llm_analyzer.get_full_analysis(
                llm_signals, portfolio, sentiment.to_dict()
            )
            results["steps"]["llm_analysis"] = {
                "success": True,
            }
            logger.info("LLM 分析完成")
            
            portfolio = self.position_tracker.get_portfolio_summary()
            
            logger.info("[8/8] 发送通知...")
            self.notifier.send_daily_report(
                signals={},
                sell_signals=all_sell_signals,
                trades=sell_trades,
                portfolio=portfolio,
                sentiment=sentiment,
                llm_analysis=llm_analysis,
            )
            
            self._last_run_date = datetime.now().strftime("%Y-%m-%d")
            results["success"] = True
            results["portfolio"] = portfolio
            results["stock_pool"] = stock_pool[:10]
            results["end_time"] = datetime.now().isoformat()
            
            logger.info("=" * 50)
            logger.info("日线监听任务完成")
            logger.info(f"总资产: ¥{portfolio['total_assets']:,.0f}")
            logger.info(f"总收益: {portfolio['total_return']:.2%}")
            
        except Exception as e:
            logger.error(f"监听任务执行失败: {e}", exc_info=True)
            results["error"] = str(e)
            results["end_time"] = datetime.now().isoformat()
            
            self.notifier.send_error(f"监听任务执行失败: {e}")
        
        return results
    
    def _scheduler_loop(self):
        logger.info(f"定时调度器启动，运行时间: {self.schedule_time}")
        
        while self._running:
            try:
                if self._should_run_today() and self._is_schedule_time():
                    logger.info("到达执行时间，开始运行监听任务...")
                    self.run_monitor_cycle()
                    time.sleep(60)
                
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"调度器循环异常: {e}")
                time.sleep(60)
    
    def _sentiment_loop(self):
        logger.info("舆情分析定时任务启动")
        
        while self._running:
            try:
                if self._is_sentiment_time():
                    logger.info("到达舆情分析时间，开始执行...")
                    self.run_sentiment_analysis()
                    time.sleep(60)
                
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"舆情分析循环异常: {e}")
                time.sleep(60)
    
    def start(self):
        if not self.enabled:
            logger.info("定时调度器已禁用")
            return
        
        if self._running:
            logger.warning("调度器已在运行中")
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._monitor_thread.start()
        
        sentiment_schedule = self.config.get("sentiment_schedule", {})
        if sentiment_schedule.get("enabled", True):
            self._sentiment_thread = threading.Thread(target=self._sentiment_loop, daemon=True)
            self._sentiment_thread.start()
        
        logger.info("定时调度器已启动")
    
    def stop(self):
        if not self._running:
            return
        
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        if self._sentiment_thread:
            self._sentiment_thread.join(timeout=5)
        logger.info("定时调度器已停止")
    
    def run_once(self, date: Optional[str] = None) -> Dict[str, Any]:
        return self.run_monitor_cycle(date)
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "schedule_time": self.schedule_time,
            "sentiment_schedule_times": self._sentiment_schedule_times,
            "last_run_date": self._last_run_date,
            "stock_pool_size": len(self._stock_pool),
            "portfolio": self.position_tracker.get_portfolio_summary() if self.position_tracker else None,
        }
    
    def get_stock_pool(self) -> List[Dict]:
        return self._stock_pool
    
    def get_stock_scores(self, code: Optional[str] = None) -> Dict:
        if code:
            return self._stock_scores.get(code, {})
        return self._stock_scores
    
    def reset_positions(self):
        if self.position_tracker:
            self.position_tracker.reset()
            logger.info("持仓已重置")


def create_scheduler(config_path: str = "config/monitor.yaml") -> MonitorScheduler:
    return MonitorScheduler(config_path)


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    scheduler = create_scheduler()
    scheduler.start()
    
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("接收到退出信号，停止调度器...")
        scheduler.stop()
