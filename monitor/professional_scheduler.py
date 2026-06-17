from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from monitor.market_regime import MarketRegimeDetector, MarketRegime
from monitor.strategy_manager import StrategyManager, CombinedSignal
from monitor.risk_controller import RiskController, RiskMetrics
from monitor.decision_support import DecisionSupportSystem, DecisionReport

logger = logging.getLogger(__name__)


class ProfessionalMonitorScheduler:
    def __init__(self, config_path: str = "config/monitor.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.scheduler_config = self.config.get("scheduler", {})
        
        self.enabled = self.scheduler_config.get("enabled", True)
        self.schedule_time = self.scheduler_config.get("schedule_time", "15:05")
        
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._last_run_date: Optional[str] = None
        self._lock = threading.Lock()
        
        self._init_components()
    
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
        
        self.scorer = MultiDimensionScorer(self.config_path)
        self.enhanced_sentiment = EnhancedSentimentAnalyzer(self.config_path)
        self.technical_analyzer = TechnicalAnalyzer(self.config_path)
        self.extreme_detector = ExtremeConditionDetector(self.config_path)
        
        self.market_regime_detector = MarketRegimeDetector(self.config_path)
        self.strategy_manager = StrategyManager(self.config_path)
        self.risk_controller = RiskController(self.config_path)
        self.decision_support = DecisionSupportSystem(self.config_path)
        
        self._stock_pool: List[Dict] = []
        self._combined_signals: List[CombinedSignal] = []
        self._current_regime: Optional[MarketRegime] = None
        self._current_risk: Optional[RiskMetrics] = None
        self._decision_report: Optional[DecisionReport] = None
        
        logger.info("专业投资监听系统初始化完成")
    
    def run_professional_cycle(self, date: Optional[str] = None) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("开始专业投资监听任务")
        
        results = {
            "start_time": datetime.now().isoformat(),
            "date": date or datetime.now().strftime("%Y-%m-%d"),
            "success": False,
            "steps": {},
        }
        
        try:
            logger.info("[1/9] 分析市场环境...")
            self._current_regime = self.market_regime_detector.detect_regime()
            results["steps"]["market_regime"] = {
                "success": True,
                "regime": self._current_regime.regime,
                "trend_strength": self._current_regime.trend_strength,
                "volatility": self._current_regime.volatility,
            }
            logger.info(f"市场环境: {self._current_regime.regime}, 趋势强度: {self._current_regime.trend_strength:.2f}")
            
            logger.info("[2/9] 计算策略权重...")
            strategy_weights = self.market_regime_detector.get_strategy_weights(self._current_regime)
            position_multiplier = self.market_regime_detector.get_position_size_multiplier(self._current_regime)
            results["steps"]["strategy_weights"] = {
                "success": True,
                "weights": strategy_weights,
                "position_multiplier": position_multiplier,
            }
            logger.info(f"策略权重: {strategy_weights}")
            
            logger.info("[3/9] 生成股票池和评分...")
            stock_pool = self._generate_scored_pool()
            self._stock_pool = stock_pool
            results["steps"]["stock_pool"] = {
                "success": True,
                "pool_size": len(stock_pool),
                "top_stocks": stock_pool[:5] if stock_pool else [],
            }
            logger.info(f"股票池生成完成: {len(stock_pool)} 只")
            
            logger.info("[4/9] 组合策略信号...")
            self._combined_signals = self._combine_strategy_signals(stock_pool, strategy_weights)
            results["steps"]["combined_signals"] = {
                "success": True,
                "signal_count": len(self._combined_signals),
            }
            logger.info(f"组合信号: {len(self._combined_signals)} 个")
            
            logger.info("[5/9] 风险评估...")
            portfolio = self.position_tracker.get_portfolio_summary()
            positions = self.position_tracker.positions
            self._current_risk = self.risk_controller.calculate_portfolio_risk(
                {code: pos.to_dict() for code, pos in positions.items()},
                portfolio.get("total_assets", 0),
                portfolio.get("peak_value", portfolio.get("total_assets", 0)),
            )
            results["steps"]["risk_assessment"] = {
                "success": True,
                "risk_level": self._current_risk.risk_level,
                "total_risk": self._current_risk.total_risk,
                "alerts": self._current_risk.alerts,
            }
            logger.info(f"风险等级: {self._current_risk.risk_level}")
            
            logger.info("[6/9] 检查止损和极端情况...")
            prices = {code: self.position_tracker.get_current_price(code) for code in positions}
            sell_recommendations = self.risk_controller.get_sell_recommendations(
                {code: pos.to_dict() for code, pos in positions.items()},
                prices
            )
            extreme_alerts = self._check_extreme_conditions()
            results["steps"]["risk_signals"] = {
                "success": True,
                "sell_recommendations": len(sell_recommendations),
                "extreme_alerts": len(extreme_alerts),
            }
            logger.info(f"卖出建议: {len(sell_recommendations)}, 极端警报: {len(extreme_alerts)}")
            
            logger.info("[7/9] 生成决策报告...")
            position_sizings = self._calculate_position_sizings(stock_pool)
            stop_losses = self._calculate_stop_losses()
            
            self._decision_report = self.decision_support.generate_decision_report(
                market_regime=self._current_regime,
                risk_metrics=self._current_risk,
                combined_signals=self._combined_signals,
                positions={code: pos.to_dict() for code, pos in positions.items()},
                sell_recommendations=sell_recommendations,
                extreme_alerts=extreme_alerts,
                position_sizings=position_sizings,
                stop_losses=stop_losses,
            )
            results["steps"]["decision_report"] = {
                "success": True,
                "buy_signals": len(self._decision_report.buy_signals),
                "sell_signals": len(self._decision_report.sell_signals),
                "requires_attention": self._decision_report.requires_attention,
            }
            logger.info(f"决策报告生成完成: {len(self._decision_report.buy_signals)} 买, {len(self._decision_report.sell_signals)} 卖")
            
            logger.info("[8/9] 执行交易操作...")
            trade_results = self._execute_trades(
                self._decision_report,
                position_multiplier,
            )
            results["steps"]["trades"] = trade_results
            logger.info(f"交易执行完成: 买入 {trade_results['buy_count']}, 卖出 {trade_results['sell_count']}")
            
            logger.info("[9/9] 发送通知...")
            self._send_notifications(self._decision_report, portfolio)
            results["steps"]["notification"] = {"success": True}
            
            self._last_run_date = datetime.now().strftime("%Y-%m-%d")
            results["success"] = True
            results["portfolio"] = self.position_tracker.get_portfolio_summary()
            results["decision_report"] = self._decision_report.to_dict()
            results["end_time"] = datetime.now().isoformat()
            
            logger.info("=" * 60)
            logger.info("专业投资监听任务完成")
            
        except Exception as e:
            logger.error(f"监听任务执行失败: {e}", exc_info=True)
            results["error"] = str(e)
            results["end_time"] = datetime.now().isoformat()
            self.notifier.send_error(f"监听任务执行失败: {e}")
        
        return results
    
    def _generate_scored_pool(self) -> List[Dict]:
        signals = self.signal_engine.generate_all_signals()
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
        
        scored_stocks = []
        for stock in all_stocks:
            scores = self._calculate_stock_scores(stock)
            stock.update(scores)
            scored_stocks.append(stock)
        
        scored_stocks.sort(key=lambda x: x.get("total_score", 0), reverse=True)
        
        pool_config = self.config.get("stock_pool", {})
        max_size = pool_config.get("max_pool_size", 50)
        min_score = pool_config.get("min_score", 0.50)
        
        scored_stocks = [s for s in scored_stocks if s.get("total_score", 0) >= min_score]
        return scored_stocks[:max_size]
    
    def _calculate_stock_scores(self, stock: Dict) -> Dict[str, float]:
        scores = {
            "model_score": stock.get("prediction_score", 0.5),
            "sentiment_score": 0.5,
            "technical_score": 0.5,
        }
        
        if self.enhanced_sentiment:
            try:
                sentiment_result = self.enhanced_sentiment.analyze_stock_sentiment(
                    stock["code"], stock["name"]
                )
                scores["sentiment_score"] = sentiment_result.get("sentiment_score", 0.5)
            except Exception as e:
                logger.debug(f"舆情分析失败 {stock['code']}: {e}")
        
        if self.technical_analyzer:
            try:
                tech_result = self.technical_analyzer.analyze(stock["code"])
                tech_score = self.technical_analyzer.calculate_score(tech_result)
                scores["technical_score"] = tech_score.total_score
            except Exception as e:
                logger.debug(f"技术分析失败 {stock['code']}: {e}")
        
        if self.scorer:
            scores["total_score"] = self.scorer.calculate_total_score(
                model_score=scores["model_score"],
                sentiment_score=scores["sentiment_score"],
                strategy_score=0.5,
                technical_score=scores["technical_score"],
            )
        else:
            scores["total_score"] = (
                scores["model_score"] * 0.4 +
                scores["sentiment_score"] * 0.25 +
                scores["technical_score"] * 0.35
            )
        
        return scores
    
    def _combine_strategy_signals(
        self, 
        stock_pool: List[Dict],
        strategy_weights: Dict[str, float]
    ) -> List[CombinedSignal]:
        signals = []
        
        for stock in stock_pool:
            signal = CombinedSignal(
                code=stock["code"],
                name=stock["name"],
                total_score=stock.get("total_score", 0.5),
                weighted_score=stock.get("total_score", 0.5),
                strategy_scores={"model": stock.get("model_score", 0.5)},
                strategy_weights=strategy_weights,
                signal_type="buy" if stock.get("total_score", 0.5) >= 0.6 else "hold",
                confidence=min(1.0, stock.get("total_score", 0.5) * 1.2),
                reasons=[stock.get("reason", "综合评分")],
            )
            signals.append(signal)
        
        return signals
    
    def _check_extreme_conditions(self) -> List[Dict]:
        alerts = []
        positions = self.position_tracker.positions
        
        for code, position in positions.items():
            try:
                sentiment_data = None
                if self.enhanced_sentiment:
                    sentiment_data = self.enhanced_sentiment.get_cached_sentiment(code)
                
                recommendations = self.extreme_detector.get_sell_recommendations(
                    {code: position.to_dict()},
                    {code: sentiment_data} if sentiment_data else {}
                )
                alerts.extend(recommendations)
            except Exception as e:
                logger.debug(f"极端检测失败 {code}: {e}")
        
        return alerts
    
    def _calculate_position_sizings(self, stock_pool: List[Dict]) -> Dict[str, Any]:
        sizings = {}
        portfolio = self.position_tracker.get_portfolio_summary()
        total_capital = portfolio.get("total_assets", 0)
        
        for stock in stock_pool[:20]:
            try:
                price = self.position_tracker.get_current_price(stock["code"])
                if price <= 0:
                    continue
                
                stop_price = price * 0.92
                
                sizing = self.risk_controller.calculate_position_size(
                    code=stock["code"],
                    name=stock["name"],
                    entry_price=price,
                    stop_price=stop_price,
                    total_capital=total_capital,
                    confidence=stock.get("total_score", 0.5),
                )
                sizings[stock["code"]] = sizing
            except Exception as e:
                logger.debug(f"仓位计算失败 {stock['code']}: {e}")
        
        return sizings
    
    def _calculate_stop_losses(self) -> Dict[str, Any]:
        stop_losses = {}
        positions = self.position_tracker.positions
        
        for code, position in positions.items():
            try:
                current_price = self.position_tracker.get_current_price(code)
                stop_level = self.risk_controller.calculate_stop_loss_levels(
                    code=code,
                    name=position.name,
                    entry_price=position.buy_price,
                    current_price=current_price,
                    holding_days=(datetime.now() - position.buy_time).days,
                    highest_price=position.highest_price if hasattr(position, 'highest_price') else current_price,
                )
                stop_losses[code] = stop_level
            except Exception as e:
                logger.debug(f"止损计算失败 {code}: {e}")
        
        return stop_losses
    
    def _execute_trades(
        self,
        report: DecisionReport,
        position_multiplier: float,
    ) -> Dict[str, Any]:
        results = {"buy_count": 0, "sell_count": 0, "trades": []}
        
        for sig in report.sell_signals:
            try:
                price = self.position_tracker.get_current_price(sig.code)
                if price > 0:
                    trade = self.position_tracker.sell(sig.code, price, sig.reasons[0])
                    if trade:
                        results["sell_count"] += 1
                        results["trades"].append(trade.to_dict())
            except Exception as e:
                logger.error(f"卖出执行失败 {sig.code}: {e}")
        
        for sig in report.buy_signals:
            if len(self.position_tracker.positions) >= self.position_tracker.position_config.get("max_stocks", 10):
                break
            
            if sig.code in self.position_tracker.positions:
                continue
            
            try:
                price = self.position_tracker.get_current_price(sig.code)
                if price <= 0:
                    continue
                
                base_size = self.config.get("strategies", {}).get("default", {}).get("position_size", 0.1)
                adjusted_size = self.risk_controller.adjust_position_for_risk(
                    base_size, self._current_risk, position_multiplier
                )
                
                trade = self.position_tracker.buy(
                    code=sig.code,
                    name=sig.name,
                    price=price,
                    strategy="multi_strategy",
                    reason=sig.reasons[0] if sig.reasons else "综合评分",
                    position_size=adjusted_size,
                )
                if trade:
                    results["buy_count"] += 1
                    results["trades"].append(trade.to_dict())
            except Exception as e:
                logger.error(f"买入执行失败 {sig.code}: {e}")
        
        return results
    
    def _send_notifications(self, report: DecisionReport, portfolio: Dict):
        message = self.decision_support.format_report_message(report)
        self.notifier.send_message(message, "decision_report")
        
        if report.requires_attention:
            self.notifier.send_alert(
                "投资决策需要关注",
                "\n".join(report.recommendations)
            )
    
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
        logger.info(f"专业投资监听系统已启动，运行时间: {self.schedule_time}")
    
    def stop(self):
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("专业投资监听系统已停止")
    
    def _scheduler_loop(self):
        while self._running:
            try:
                now = datetime.now()
                current_time = now.strftime("%H:%M")
                today = now.strftime("%Y-%m-%d")
                
                if self._last_run_date != today and current_time >= self.schedule_time:
                    logger.info("到达执行时间，开始运行监听任务...")
                    self.run_professional_cycle()
                    time.sleep(60)
                
                time.sleep(30)
            except Exception as e:
                logger.error(f"调度器循环异常: {e}")
                time.sleep(60)
    
    def run_once(self, date: Optional[str] = None) -> Dict[str, Any]:
        return self.run_professional_cycle(date)
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "schedule_time": self.schedule_time,
            "last_run_date": self._last_run_date,
            "current_regime": self._current_regime.to_dict() if self._current_regime else None,
            "current_risk": self._current_risk.to_dict() if self._current_risk else None,
            "portfolio": self.position_tracker.get_portfolio_summary() if self.position_tracker else None,
        }


def create_professional_scheduler(config_path: str = "config/monitor.yaml") -> ProfessionalMonitorScheduler:
    return ProfessionalMonitorScheduler(config_path)
