from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from monitor.ts_client import get_ts_client

logger = logging.getLogger(__name__)


@dataclass
class NewsItem:
    title: str
    content: str
    source: str
    time: str
    url: str = ""
    keywords: List[str] = field(default_factory=list)
    sentiment: str = "neutral"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content,
            "source": self.source,
            "time": self.time,
            "url": self.url,
            "keywords": self.keywords,
            "sentiment": self.sentiment,
        }


@dataclass
class SentimentResult:
    date: str
    news_count: int
    positive_count: int
    negative_count: int
    neutral_count: int
    overall_sentiment: str
    sentiment_score: float
    top_news: List[NewsItem]
    keyword_mentions: Dict[str, int]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "news_count": self.news_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "neutral_count": self.neutral_count,
            "overall_sentiment": self.overall_sentiment,
            "sentiment_score": self.sentiment_score,
            "top_news": [n.to_dict() for n in self.top_news],
            "keyword_mentions": self.keyword_mentions,
        }


class SentimentAnalyzer:
    POSITIVE_WORDS = [
        "上涨", "大涨", "暴涨", "涨停", "利好", "突破", "新高", "反弹", "走强",
        "增长", "盈利", "收益", "牛市", "看涨", "增持", "买入", "超预期",
        "回暖", "复苏", "宽松", "降息", "刺激", "支持", "利好",
    ]
    
    NEGATIVE_WORDS = [
        "下跌", "大跌", "暴跌", "跌停", "利空", "破位", "新低", "回调", "走弱",
        "亏损", "下滑", "熊市", "看跌", "减持", "卖出", "不及预期",
        "收紧", "加息", "风险", "危机", "恐慌", "抛售", "崩盘",
    ]
    
    def __init__(self, config_path: str = "config/monitor.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.sentiment_config = self.config.get("sentiment", {})
        
        self.enabled = self.sentiment_config.get("enabled", True)
        self.max_news = self.sentiment_config.get("max_news", 50)
        self.cache_hours = self.sentiment_config.get("cache_hours", 4)
        self.keywords = self.sentiment_config.get("keywords", [])
        
        self._cache_file = Path(self.config.get("paths", {}).get(
            "sentiment_cache_file", "data/monitor/sentiment_cache.json"
        ))
    
    def _load_config(self) -> Dict[str, Any]:
        p = Path(self.config_path)
        if not p.exists():
            return {}
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    
    def _load_cache(self) -> Optional[Dict[str, Any]]:
        if not self._cache_file.exists():
            return None
        try:
            with open(self._cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            cache_time = datetime.fromisoformat(data.get("cache_time", "2000-01-01"))
            if datetime.now() - cache_time < timedelta(hours=self.cache_hours):
                return data
        except Exception as e:
            logger.warning(f"加载舆情缓存失败: {e}")
        return None
    
    def _save_cache(self, data: Dict[str, Any]):
        self._cache_file.parent.mkdir(parents=True, exist_ok=True)
        data["cache_time"] = datetime.now().isoformat()
        with open(self._cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def fetch_news_tushare(self) -> List[NewsItem]:
        ts = get_ts_client()
        if ts is None:
            logger.warning("TushareClient 未初始化，无法获取新闻")
            return []
        
        news_list = []
        try:
            today = datetime.now().strftime("%Y%m%d")
            df = ts._pro.news(src="sina", start_date=today, end_date=today)
            
            if df.empty:
                return []
            
            for _, row in df.head(self.max_news).iterrows():
                news = NewsItem(
                    title=str(row.get("title", "")),
                    content=str(row.get("content", "")),
                    source=str(row.get("src", "tushare")),
                    time=str(row.get("datetime", "")),
                    url=str(row.get("url", "")),
                )
                news_list.append(news)
        except Exception as e:
            logger.warning(f"从 Tushare 获取新闻失败: {e}")
        
        return news_list
    
    def fetch_news_web(self) -> List[NewsItem]:
        news_list = []
        
        sources = self.sentiment_config.get("sources", [])
        for source in sources:
            if not source.get("enabled", True):
                continue
            if source.get("type") != "web_scraper":
                continue
            
            try:
                import requests
                from bs4 import BeautifulSoup
                
                url = source.get("url", "")
                if not url:
                    continue
                
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
                resp = requests.get(url, headers=headers, timeout=10)
                resp.encoding = "utf-8"
                
                soup = BeautifulSoup(resp.text, "html.parser")
                
                links = soup.find_all("a", limit=self.max_news)
                for link in links:
                    title = link.get_text(strip=True)
                    href = link.get("href", "")
                    
                    if len(title) < 10 or len(title) > 100:
                        continue
                    
                    if any(kw in title for kw in self.keywords):
                        news = NewsItem(
                            title=title,
                            content="",
                            source=source.get("name", "web"),
                            time=datetime.now().strftime("%Y-%m-%d %H:%M"),
                            url=href,
                        )
                        news_list.append(news)
            except Exception as e:
                logger.warning(f"从 {source.get('name')} 获取新闻失败: {e}")
        
        return news_list
    
    def analyze_sentiment(self, news: NewsItem) -> str:
        text = f"{news.title} {news.content}"
        
        positive_count = sum(1 for word in self.POSITIVE_WORDS if word in text)
        negative_count = sum(1 for word in self.NEGATIVE_WORDS if word in text)
        
        if positive_count > negative_count + 1:
            return "positive"
        elif negative_count > positive_count + 1:
            return "negative"
        else:
            return "neutral"
    
    def extract_keywords(self, news: NewsItem) -> List[str]:
        text = f"{news.title} {news.content}"
        found = []
        for kw in self.keywords:
            if kw in text:
                found.append(kw)
        return found
    
    def analyze(self, force_refresh: bool = False) -> SentimentResult:
        if not self.enabled:
            return SentimentResult(
                date=datetime.now().strftime("%Y-%m-%d"),
                news_count=0,
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                overall_sentiment="neutral",
                sentiment_score=0.5,
                top_news=[],
                keyword_mentions={},
            )
        
        if not force_refresh:
            cache = self._load_cache()
            if cache:
                logger.info("使用缓存的舆情分析结果")
                return SentimentResult(**cache.get("result", {}))
        
        news_list = []
        
        news_list.extend(self.fetch_news_tushare())
        news_list.extend(self.fetch_news_web())
        
        unique_news = {}
        for n in news_list:
            key = n.title[:50]
            if key not in unique_news:
                unique_news[key] = n
        news_list = list(unique_news.values())[:self.max_news]
        
        for news in news_list:
            news.sentiment = self.analyze_sentiment(news)
            news.keywords = self.extract_keywords(news)
        
        positive_count = sum(1 for n in news_list if n.sentiment == "positive")
        negative_count = sum(1 for n in news_list if n.sentiment == "negative")
        neutral_count = sum(1 for n in news_list if n.sentiment == "neutral")
        
        total = len(news_list)
        if total > 0:
            sentiment_score = (positive_count + 0.5 * neutral_count) / total
        else:
            sentiment_score = 0.5
        
        if sentiment_score > 0.6:
            overall_sentiment = "positive"
        elif sentiment_score < 0.4:
            overall_sentiment = "negative"
        else:
            overall_sentiment = "neutral"
        
        keyword_mentions = {}
        for news in news_list:
            for kw in news.keywords:
                keyword_mentions[kw] = keyword_mentions.get(kw, 0) + 1
        
        top_news = sorted(
            news_list,
            key=lambda n: len(n.keywords),
            reverse=True
        )[:10]
        
        result = SentimentResult(
            date=datetime.now().strftime("%Y-%m-%d"),
            news_count=total,
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
            overall_sentiment=overall_sentiment,
            sentiment_score=sentiment_score,
            top_news=top_news,
            keyword_mentions=keyword_mentions,
        )
        
        self._save_cache({"result": result.to_dict()})
        
        logger.info(f"舆情分析完成: {total} 条新闻, 情绪={overall_sentiment}, 分数={sentiment_score:.2f}")
        
        return result
    
    def get_market_summary(self) -> str:
        result = self.analyze()
        
        lines = [
            f"📅 {result.date} 舆情分析",
            f"",
            f"📊 情绪指标: {result.overall_sentiment} ({result.sentiment_score:.2f})",
            f"📰 新闻统计: {result.news_count} 条",
            f"  - 正面: {result.positive_count} 条",
            f"  - 负面: {result.negative_count} 条",
            f"  - 中性: {result.neutral_count} 条",
        ]
        
        if result.keyword_mentions:
            lines.append(f"")
            lines.append(f"🔍 关键词热度:")
            sorted_kw = sorted(result.keyword_mentions.items(), key=lambda x: x[1], reverse=True)[:5]
            for kw, count in sorted_kw:
                lines.append(f"  - {kw}: {count} 次")
        
        if result.top_news:
            lines.append(f"")
            lines.append(f"📋 重要新闻:")
            for i, news in enumerate(result.top_news[:5], 1):
                sentiment_emoji = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(news.sentiment, "⚪")
                lines.append(f"  {i}. {sentiment_emoji} {news.title[:40]}...")
        
        return "\n".join(lines)
