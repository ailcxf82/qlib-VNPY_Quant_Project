from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from monitor.exceptions import LLMError, NetworkError, ConfigurationError

logger = logging.getLogger(__name__)


@dataclass
class LLMAnalysisResult:
    analysis_type: str
    date: str
    summary: str
    key_points: List[str]
    recommendation: str
    confidence: str
    raw_response: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_type": self.analysis_type,
            "date": self.date,
            "summary": self.summary,
            "key_points": self.key_points,
            "recommendation": self.recommendation,
            "confidence": self.confidence,
            "raw_response": self.raw_response,
        }


class LLMAnalyzer:
    def __init__(self, config_path: str = "config/monitor.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.llm_config = self.config.get("llm", {})
        
        self.enabled = self.llm_config.get("enabled", True)
        self.provider = self.llm_config.get("provider", "deepseek")
        self.model = self.llm_config.get("model", "deepseek-chat")
        self.max_tokens = self.llm_config.get("max_tokens", 2000)
        self.temperature = self.llm_config.get("temperature", 0.3)
        self.cache_hours = self.llm_config.get("cache_hours", 4)
        
        if self.provider == "deepseek":
            self._api_key = os.environ.get(
                "DEEPSEEK_API_KEY",
                self.llm_config.get("api_key", "").replace("${DEEPSEEK_API_KEY}", "")
            )
            self._base_url = self.llm_config.get("base_url", "https://api.deepseek.com/v1")
        else:
            self._api_key = os.environ.get(
                "OPENAI_API_KEY",
                self.llm_config.get("api_key", "").replace("${OPENAI_API_KEY}", "")
            )
            self._base_url = os.environ.get(
                "OPENAI_BASE_URL",
                self.llm_config.get("base_url", "").replace("${OPENAI_BASE_URL}", "")
            )
        
        self._cache_file = Path(self.config.get("paths", {}).get(
            "llm_cache_file", "data/monitor/llm_cache.json"
        ))
        
        self._client = None
    
    def _load_config(self) -> Dict[str, Any]:
        p = Path(self.config_path)
        if not p.exists():
            return {}
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    
    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self._api_key,
                    base_url=self._base_url if self._base_url else None,
                )
                logger.info(f"LLM 客户端初始化成功: provider={self.provider}, model={self.model}")
            except ImportError:
                logger.warning("未安装 openai 库，请运行: pip install openai")
            except Exception as e:
                logger.warning(f"初始化 LLM 客户端失败: {e}")
        return self._client
    
    def _load_cache(self) -> Dict[str, Any]:
        if not self._cache_file.exists():
            return {}
        try:
            with open(self._cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            cache_time_str = data.get("_cache_time")
            if cache_time_str:
                cache_time = datetime.fromisoformat(cache_time_str)
                if datetime.now() - cache_time > timedelta(hours=self.cache_hours):
                    logger.info("LLM 缓存已过期，清除缓存")
                    return {}
            
            return data
        except Exception as e:
            logger.warning(f"加载 LLM 缓存失败: {e}")
            return {}
    
    def _save_cache(self, cache: Dict[str, Any]):
        self._cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache["_cache_time"] = datetime.now().isoformat()
        with open(self._cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    
    def _call_llm(self, prompt: str) -> str:
        client = self._get_client()
        if client is None:
            raise LLMError("LLM 客户端未初始化")
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一位专业的A股金融分析师，擅长分析中国股市走势和投资机会。请用简洁专业的中文回答问题，给出具体可操作的建议。"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return response.choices[0].message.content
        except ConnectionError as e:
            raise NetworkError(f"网络连接失败: {e}")
        except TimeoutError as e:
            raise NetworkError(f"请求超时: {e}")
        except Exception as e:
            error_msg = str(e).lower()
            if "api_key" in error_msg or "unauthorized" in error_msg:
                raise ConfigurationError(f"API Key 无效或未配置: {e}")
            elif "rate limit" in error_msg or "quota" in error_msg:
                raise LLMError(f"API 调用频率限制: {e}")
            elif "model" in error_msg or "not found" in error_msg:
                raise ConfigurationError(f"模型不存在或不可用: {e}")
            else:
                raise LLMError(f"调用 LLM 失败: {e}")
    
    def analyze_market_overview(self, sentiment_result: Optional[Dict] = None) -> LLMAnalysisResult:
        cache = self._load_cache()
        today = datetime.now().strftime("%Y-%m-%d")
        cache_key = f"market_overview_{today}"
        
        if cache_key in cache:
            logger.info("使用缓存的市场分析结果")
            return LLMAnalysisResult(**cache[cache_key])
        
        prompt_parts = [
            "请分析今日A股市场整体走势，包括：",
            "1. 大盘走势判断（沪指、深成指、创业板）",
            "2. 市场情绪评估（乐观/中性/悲观）",
            "3. 资金流向分析（北向资金、主力资金）",
            "4. 热点板块和概念",
            "5. 明日市场展望",
        ]
        
        if sentiment_result:
            prompt_parts.append(f"\n今日舆情数据：")
            prompt_parts.append(f"- 新闻总数: {sentiment_result.get('news_count', 0)}")
            prompt_parts.append(f"- 情绪分数: {sentiment_result.get('sentiment_score', 0.5):.2f}")
            prompt_parts.append(f"- 整体情绪: {sentiment_result.get('overall_sentiment', 'neutral')}")
            if sentiment_result.get('keyword_mentions'):
                prompt_parts.append(f"- 热门关键词: {list(sentiment_result.get('keyword_mentions', {}).keys())[:5]}")
        
        prompt = "\n".join(prompt_parts)
        
        response = self._call_llm(prompt)
        
        if response:
            result = LLMAnalysisResult(
                analysis_type="market_overview",
                date=today,
                summary=self._extract_summary(response),
                key_points=self._extract_key_points(response),
                recommendation=self._extract_recommendation(response),
                confidence=self._extract_confidence(response),
                raw_response=response,
            )
            
            cache[cache_key] = result.to_dict()
            self._save_cache(cache)
            
            return result
        
        return LLMAnalysisResult(
            analysis_type="market_overview",
            date=today,
            summary="分析失败",
            key_points=[],
            recommendation="暂无建议",
            confidence="低",
        )
    
    def analyze_trading_opportunity(
        self,
        signals: List[Dict],
        portfolio: Dict,
        sentiment_result: Optional[Dict] = None,
    ) -> LLMAnalysisResult:
        cache = self._load_cache()
        today = datetime.now().strftime("%Y-%m-%d")
        cache_key = f"trading_opportunity_{today}"
        
        if cache_key in cache:
            logger.info("使用缓存的交易机会分析结果")
            return LLMAnalysisResult(**cache[cache_key])
        
        prompt_parts = [
            "基于以下信息，请给出今日的交易建议：\n",
        ]
        
        if signals:
            prompt_parts.append("【买入信号】")
            for sig in signals[:10]:
                prompt_parts.append(f"- {sig.get('code')} {sig.get('name')}: 分数{sig.get('score', 0):.2f}, 排名{sig.get('rank')}")
        
        if portfolio.get("positions"):
            prompt_parts.append("\n【当前持仓】")
            for pos in portfolio.get("positions", [])[:10]:
                prompt_parts.append(
                    f"- {pos.get('code')} {pos.get('name')}: "
                    f"盈亏{pos.get('profit_pct', 0):.2%}, 持仓{pos.get('holding_days')}天"
                )
        
        prompt_parts.append(f"\n【账户状态】")
        prompt_parts.append(f"- 总资产: {portfolio.get('total_assets', 0):.2f}")
        prompt_parts.append(f"- 总收益: {portfolio.get('total_return', 0):.2%}")
        prompt_parts.append(f"- 现金: {portfolio.get('cash', 0):.2f}")
        
        if sentiment_result:
            prompt_parts.append(f"\n【市场情绪】")
            prompt_parts.append(f"- 情绪分数: {sentiment_result.get('sentiment_score', 0.5):.2f}")
            prompt_parts.append(f"- 整体情绪: {sentiment_result.get('overall_sentiment', 'neutral')}")
        
        prompt_parts.append("\n请给出：")
        prompt_parts.append("1. 今日是否适合建仓/加仓")
        prompt_parts.append("2. 推荐买入的股票及理由")
        prompt_parts.append("3. 需要卖出的持仓及理由")
        prompt_parts.append("4. 风险提示")
        
        prompt = "\n".join(prompt_parts)
        
        response = self._call_llm(prompt)
        
        if response:
            result = LLMAnalysisResult(
                analysis_type="trading_opportunity",
                date=today,
                summary=self._extract_summary(response),
                key_points=self._extract_key_points(response),
                recommendation=self._extract_recommendation(response),
                confidence=self._extract_confidence(response),
                raw_response=response,
            )
            
            cache[cache_key] = result.to_dict()
            self._save_cache(cache)
            
            return result
        
        return LLMAnalysisResult(
            analysis_type="trading_opportunity",
            date=today,
            summary="分析失败",
            key_points=[],
            recommendation="暂无建议",
            confidence="低",
        )
    
    def _extract_summary(self, response: str) -> str:
        lines = response.split("\n")
        for line in lines:
            line = line.strip()
            if line and len(line) > 20:
                return line[:200]
        return response[:200] if response else ""
    
    def _extract_key_points(self, response: str) -> List[str]:
        points = []
        lines = response.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith(("1.", "2.", "3.", "4.", "5.", "-", "•", "●")):
                clean = line.lstrip("12345.-•● ").strip()
                if clean and len(clean) > 5:
                    points.append(clean)
        return points[:5]
    
    def _extract_recommendation(self, response: str) -> str:
        keywords = ["建议", "推荐", "操作", "策略"]
        lines = response.split("\n")
        for line in lines:
            for kw in keywords:
                if kw in line:
                    return line.strip()[:100]
        return ""
    
    def _extract_confidence(self, response: str) -> str:
        if "高" in response and "信心" in response:
            return "高"
        elif "中" in response and "信心" in response:
            return "中"
        elif "不确定" in response or "风险" in response:
            return "低"
        return "中"
    
    def get_full_analysis(
        self,
        signals: List[Dict],
        portfolio: Dict,
        sentiment_result: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        market_analysis = self.analyze_market_overview(sentiment_result)
        trading_analysis = self.analyze_trading_opportunity(signals, portfolio, sentiment_result)
        
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "market_overview": market_analysis.to_dict(),
            "trading_opportunity": trading_analysis.to_dict(),
        }
    
    def format_report(self, analysis: Dict[str, Any]) -> str:
        lines = []
        
        market = analysis.get("market_overview", {})
        lines.append(f"📊 {analysis.get('date')} 市场分析报告")
        lines.append("")
        lines.append("【大盘走势】")
        lines.append(market.get("summary", ""))
        lines.append("")
        
        if market.get("key_points"):
            lines.append("【关键要点】")
            for point in market.get("key_points", []):
                lines.append(f"  • {point}")
            lines.append("")
        
        trading = analysis.get("trading_opportunity", {})
        lines.append("【交易建议】")
        lines.append(trading.get("recommendation", ""))
        lines.append("")
        
        if trading.get("key_points"):
            lines.append("【操作要点】")
            for point in trading.get("key_points", []):
                lines.append(f"  • {point}")
        
        return "\n".join(lines)
