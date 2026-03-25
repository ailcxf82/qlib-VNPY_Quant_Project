from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

from monitor.ts_client import get_ts_client

logger = logging.getLogger(__name__)


@dataclass
class StockNews:
    code: str
    name: str
    title: str
    content: str
    source: str
    time: str
    url: str = ""
    sentiment: str = "neutral"
    sentiment_score: float = 0.5
    keywords: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "title": self.title,
            "content": self.content,
            "source": self.source,
            "time": self.time,
            "url": self.url,
            "sentiment": self.sentiment,
            "sentiment_score": self.sentiment_score,
            "keywords": self.keywords,
        }


@dataclass
class StockSentiment:
    code: str
    name: str
    date: str
    news_count: int
    positive_count: int
    negative_count: int
    neutral_count: int
    sentiment: str
    score: float
    extreme_alert: bool = False
    extreme_type: str = ""
    news_list: List[StockNews] = field(default_factory=list)
    keyword_mentions: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "date": self.date,
            "news_count": self.news_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "neutral_count": self.neutral_count,
            "sentiment": self.sentiment,
            "score": self.score,
            "extreme_alert": self.extreme_alert,
            "extreme_type": self.extreme_type,
            "news_list": [n.to_dict() for n in self.news_list[:5]],
            "keyword_mentions": self.keyword_mentions,
        }


@dataclass
class MarketSentiment:
    date: str
    overall_sentiment: str
    sentiment_score: float
    news_count: int
    stocks: Dict[str, StockSentiment]
    extreme_stocks: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "overall_sentiment": self.overall_sentiment,
            "sentiment_score": self.sentiment_score,
            "news_count": self.news_count,
            "stocks": {k: v.to_dict() for k, v in self.stocks.items()},
            "extreme_stocks": self.extreme_stocks,
        }


class EnhancedSentimentAnalyzer:
    POSITIVE_WORDS = [
        "上涨", "大涨", "暴涨", "涨停", "利好", "突破", "新高", "反弹", "走强",
        "增长", "盈利", "收益", "增持", "买入", "超预期", "回暖", "复苏",
        "业绩大增", "翻倍", "领涨", "强势", "爆发", "利好消息", "重大利好",
    ]
    
    NEGATIVE_WORDS = [
        "下跌", "大跌", "暴跌", "跌停", "利空", "破位", "新低", "回调", "走弱",
        "亏损", "下滑", "减持", "卖出", "不及预期", "风险", "危机", "恐慌",
        "业绩下滑", "腰斩", "领跌", "弱势", "暴跌", "利空消息", "重大利空",
        "暴雷", "违约", "造假", "调查", "处罚", "诉讼", "退市风险",
    ]
    
    DATA_SOURCES = {
        "tushare": {
            "name": "Tushare",
            "priority": 1,
            "enabled": True,
        },
        "eastmoney": {
            "name": "东方财富",
            "priority": 2,
            "enabled": True,
        },
        "sina": {
            "name": "新浪财经",
            "priority": 3,
            "enabled": True,
        },
    }
    
    EXTREME_POSITIVE_THRESHOLD = 0.8
    EXTREME_NEGATIVE_THRESHOLD = 0.2
    
    def __init__(self, config_path: str = "config/monitor.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.sentiment_config = self.config.get("sentiment", {})
        
        self.enabled = self.sentiment_config.get("enabled", True)
        self.max_news = self.sentiment_config.get("max_news", 50)
        self.cache_hours = self.sentiment_config.get("cache_hours", 4)
        self.keywords = self.sentiment_config.get("keywords", [])
        self.extreme_thresholds = self.sentiment_config.get("extreme_thresholds", {
            "positive": 0.8,
            "negative": 0.2,
        })
        
        self.data_sources = self.sentiment_config.get("data_sources", ["tushare", "eastmoney", "sina"])
        
        self._cache_file = Path(self.config.get("paths", {}).get(
            "sentiment_cache_file", "data/monitor/sentiment_cache.json"
        ))
        
        self._stock_name_map: Dict[str, str] = {}
        self._sentiment_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamp: Dict[str, datetime] = {}
    
    def _load_config(self) -> Dict[str, Any]:
        p = Path(self.config_path)
        if not p.exists():
            return {}
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    
    def _load_stock_names(self) -> Dict[str, str]:
        if self._stock_name_map:
            return self._stock_name_map
        
        ts = get_ts_client()
        if ts is None:
            return {}
        
        try:
            sb = ts.stock_basic()
            name_map = {}
            for _, row in sb.iterrows():
                ts_code = str(row.get("ts_code", ""))
                name = str(row.get("name", ""))
                if ts_code and name:
                    code = ts_code.split(".")[0]
                    name_map[code] = name
            self._stock_name_map = name_map
            return name_map
        except Exception as e:
            logger.warning(f"加载股票名称失败: {e}")
            return {}
    
    def _get_stock_code_from_text(self, text: str, name_map: Dict[str, str]) -> Set[str]:
        codes = set()
        
        code_pattern = r'[036]\d{5}|688\d{3}|689\d{3}'
        found_codes = re.findall(code_pattern, text)
        codes.update(found_codes)
        
        for code, name in name_map.items():
            if name in text:
                codes.add(code)
        
        return codes
    
    def analyze_sentiment(self, text: str) -> tuple[str, float]:
        positive_count = sum(1 for word in self.POSITIVE_WORDS if word in text)
        negative_count = sum(1 for word in self.NEGATIVE_WORDS if word in text)
        
        total = positive_count + negative_count
        if total == 0:
            return "neutral", 0.5
        
        score = (positive_count + 0.5 * (total - positive_count - negative_count)) / max(total, 1)
        score = positive_count / (positive_count + negative_count) if (positive_count + negative_count) > 0 else 0.5
        
        if positive_count > negative_count + 1:
            sentiment = "positive"
        elif negative_count > positive_count + 1:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return sentiment, score
    
    def is_extreme_sentiment(self, score: float) -> tuple[bool, str]:
        if score >= self.EXTREME_POSITIVE_THRESHOLD:
            return True, "extreme_positive"
        elif score <= self.EXTREME_NEGATIVE_THRESHOLD:
            return True, "extreme_negative"
        return False, ""
    
    def fetch_news_tushare(self, codes: Optional[List[str]] = None) -> List[StockNews]:
        ts = get_ts_client()
        if ts is None:
            logger.warning("TushareClient 未初始化，无法获取新闻")
            return []
        
        news_list = []
        name_map = self._load_stock_names()
        
        try:
            today = datetime.now().strftime("%Y%m%d")
            df = ts._pro.news(src="sina", start_date=today, end_date=today)
            
            if df.empty:
                return []
            
            for _, row in df.head(self.max_news * 2).iterrows():
                title = str(row.get("title", ""))
                content = str(row.get("content", ""))
                text = f"{title} {content}"
                
                found_codes = self._get_stock_code_from_text(text, name_map)
                
                if codes and not found_codes.intersection(set(codes)):
                    continue
                
                sentiment, score = self.analyze_sentiment(text)
                
                for code in found_codes:
                    news = StockNews(
                        code=code,
                        name=name_map.get(code, code),
                        title=title,
                        content=content[:500],
                        source=str(row.get("src", "tushare")),
                        time=str(row.get("datetime", "")),
                        url=str(row.get("url", "")),
                        sentiment=sentiment,
                        sentiment_score=score,
                    )
                    news_list.append(news)
                
                if len(news_list) >= self.max_news:
                    break
                    
        except Exception as e:
            logger.warning(f"从 Tushare 获取新闻失败: {e}")
        
        return news_list
    
    def fetch_news_eastmoney(self, code: str) -> List[StockNews]:
        news_list = []
        name_map = self._load_stock_names()
        
        try:
            import requests
            
            code_clean = str(code).replace(".XSHE", "").replace(".XSHG", "").zfill(6)
            secid = f"1.{code_clean}" if code_clean.startswith("6") else f"0.{code_clean}"
            
            url = "https://np-listapi.eastmoney.com/comm/web/getFastNewsList"
            params = {
                "client": "web",
                "biz": "web_StockF10",
                "fastColumn": "102",
                "secid": secid,
                "pageSize": 20,
            }
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer": "https://data.eastmoney.com/",
            }
            
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("data"):
                    for item in data["data"][:10]:
                        title = item.get("title", "")
                        content = item.get("digest", "")
                        text = f"{title} {content}"
                        
                        sentiment, score = self.analyze_sentiment(text)
                        
                        news = StockNews(
                            code=code,
                            name=name_map.get(code, code),
                            title=title,
                            content=content[:500],
                            source="eastmoney",
                            time=item.get("showtime", ""),
                            url=item.get("url", ""),
                            sentiment=sentiment,
                            sentiment_score=score,
                        )
                        news_list.append(news)
        except Exception as e:
            logger.debug(f"从东方财富获取新闻失败 {code}: {e}")
        
        return news_list
    
    def fetch_news_sina(self, code: str) -> List[StockNews]:
        news_list = []
        name_map = self._load_stock_names()
        
        try:
            import requests
            
            code_clean = str(code).replace(".XSHE", "").replace(".XSHG", "").zfill(6)
            symbol = f"sh{code_clean}" if code_clean.startswith("6") else f"sz{code_clean}"
            
            url = f"https://feed.sina.com.cn/api/roll/get"
            params = {
                "pageid": "153",
                "lid": "2506",
                "k": name_map.get(code, code_clean),
                "num": 20,
                "page": 1,
            }
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer": "https://finance.sina.com.cn/",
            }
            
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("result", {}).get("data"):
                    for item in data["result"]["data"][:10]:
                        title = item.get("title", "")
                        content = item.get("intro", "")
                        text = f"{title} {content}"
                        
                        sentiment, score = self.analyze_sentiment(text)
                        
                        news = StockNews(
                            code=code,
                            name=name_map.get(code, code),
                            title=title,
                            content=content[:500],
                            source="sina",
                            time=item.get("ctime", ""),
                            url=item.get("url", ""),
                            sentiment=sentiment,
                            sentiment_score=score,
                        )
                        news_list.append(news)
        except Exception as e:
            logger.debug(f"从新浪财经获取新闻失败 {code}: {e}")
        
        return news_list
    
    def fetch_all_news(self, code: str) -> List[StockNews]:
        all_news = []
        
        if "tushare" in self.data_sources:
            ts_news = self.fetch_news_tushare([code])
            all_news.extend(ts_news)
        
        if "eastmoney" in self.data_sources:
            em_news = self.fetch_news_eastmoney(code)
            all_news.extend(em_news)
        
        if "sina" in self.data_sources:
            sina_news = self.fetch_news_sina(code)
            all_news.extend(sina_news)
        
        seen_titles = set()
        unique_news = []
        for news in all_news:
            if news.title not in seen_titles:
                seen_titles.add(news.title)
                unique_news.append(news)
        
        unique_news.sort(key=lambda x: x.time, reverse=True)
        
        return unique_news[:self.max_news]
    
    def analyze_stocks_sentiment(
        self, 
        codes: Optional[List[str]] = None,
        force_refresh: bool = False
    ) -> MarketSentiment:
        name_map = self._load_stock_names()
        
        if codes is None:
            codes = list(name_map.keys())[:100]
        
        news_list = self.fetch_news_tushare(codes)
        
        stock_news_map: Dict[str, List[StockNews]] = {}
        for news in news_list:
            if news.code not in stock_news_map:
                stock_news_map[news.code] = []
            stock_news_map[news.code].append(news)
        
        stocks_sentiment: Dict[str, StockSentiment] = {}
        extreme_stocks: List[str] = []
        
        for code in codes:
            stock_news = stock_news_map.get(code, [])
            
            if not stock_news:
                continue
            
            positive_count = sum(1 for n in stock_news if n.sentiment == "positive")
            negative_count = sum(1 for n in stock_news if n.sentiment == "negative")
            neutral_count = sum(1 for n in stock_news if n.sentiment == "neutral")
            total = len(stock_news)
            
            if total > 0:
                score = (positive_count + 0.5 * neutral_count) / total
            else:
                score = 0.5
            
            if score > 0.6:
                sentiment = "positive"
            elif score < 0.4:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            is_extreme, extreme_type = self.is_extreme_sentiment(score)
            
            keyword_mentions: Dict[str, int] = {}
            for news in stock_news:
                for kw in self.keywords:
                    if kw in news.title or kw in news.content:
                        keyword_mentions[kw] = keyword_mentions.get(kw, 0) + 1
            
            stock_sentiment = StockSentiment(
                code=code,
                name=name_map.get(code, code),
                date=datetime.now().strftime("%Y-%m-%d"),
                news_count=total,
                positive_count=positive_count,
                negative_count=negative_count,
                neutral_count=neutral_count,
                sentiment=sentiment,
                score=score,
                extreme_alert=is_extreme,
                extreme_type=extreme_type,
                news_list=stock_news[:10],
                keyword_mentions=keyword_mentions,
            )
            
            stocks_sentiment[code] = stock_sentiment
            
            if is_extreme:
                extreme_stocks.append(code)
        
        total_news = len(news_list)
        if stocks_sentiment:
            overall_score = sum(s.score for s in stocks_sentiment.values()) / len(stocks_sentiment)
        else:
            overall_score = 0.5
        
        if overall_score > 0.6:
            overall_sentiment = "positive"
        elif overall_score < 0.4:
            overall_sentiment = "negative"
        else:
            overall_sentiment = "neutral"
        
        return MarketSentiment(
            date=datetime.now().strftime("%Y-%m-%d"),
            overall_sentiment=overall_sentiment,
            sentiment_score=overall_score,
            news_count=total_news,
            stocks=stocks_sentiment,
            extreme_stocks=extreme_stocks,
        )
    
    def get_stock_sentiment(self, code: str) -> Optional[StockSentiment]:
        result = self.analyze_stocks_sentiment([code])
        return result.stocks.get(code)
    
    def check_extreme_alert(self, code: str) -> Optional[Dict[str, Any]]:
        stock_sentiment = self.get_stock_sentiment(code)
        
        if stock_sentiment is None:
            return None
        
        if stock_sentiment.extreme_alert:
            return {
                "code": code,
                "name": stock_sentiment.name,
                "alert_type": stock_sentiment.extreme_type,
                "sentiment_score": stock_sentiment.score,
                "news_count": stock_sentiment.news_count,
                "negative_count": stock_sentiment.negative_count,
                "positive_count": stock_sentiment.positive_count,
                "top_news": [n.to_dict() for n in stock_sentiment.news_list[:3]],
            }
        
        return None
    
    def analyze_stock_sentiment(self, code: str, name: str = "") -> Dict[str, Any]:
        stock_sentiment = self.get_stock_sentiment(code)
        
        if stock_sentiment is None:
            return {
                "code": code,
                "name": name or code,
                "sentiment_score": 0.5,
                "sentiment": "neutral",
                "is_extreme": False,
                "extreme_type": "",
                "news_count": 0,
            }
        
        is_extreme, extreme_type = self.is_extreme_sentiment(stock_sentiment.score)
        
        return {
            "code": code,
            "name": stock_sentiment.name,
            "sentiment_score": stock_sentiment.score,
            "sentiment": stock_sentiment.sentiment,
            "is_extreme": is_extreme,
            "extreme_type": extreme_type,
            "news_count": stock_sentiment.news_count,
            "positive_count": stock_sentiment.positive_count,
            "negative_count": stock_sentiment.negative_count,
        }
    
    def get_cached_sentiment(self, code: str) -> Optional[Dict[str, Any]]:
        return self.analyze_stock_sentiment(code)
