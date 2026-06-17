"""
统一数据层 - DataHub

整合akshare/tushare数据源，提供缓存机制
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataSourceConfig:
    akshare_enabled: bool = True
    tushare_enabled: bool = False
    tushare_token: Optional[str] = None
    cache_dir: str = "data/cache"
    cache_expire_hours: int = 24


@dataclass
class StockInfo:
    code: str
    name: str
    weight: float
    industry: str = ""
    market_cap: float = 0.0


class DataHub:
    """统一数据层 - 数据中心"""
    
    def __init__(self, config: Optional[DataSourceConfig] = None):
        self.config = config or DataSourceConfig()
        self._akshare = None
        self._tushare = None
        self._cache: Dict[str, Any] = {}
        self._cache_timestamp: Dict[str, datetime] = {}
    
    @property
    def ak(self):
        if self._akshare is None:
            import akshare as ak
            self._akshare = ak
        return self._akshare
    
    @property
    def ts(self):
        if self._tushare is None and self.config.tushare_enabled:
            import tushare as ts
            self._tushare = ts
        return self._tushare
    
    def get_hs300_constituents(
        self,
        adjust_date: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, StockInfo]]:
        """
        获取沪深300成分股
        
        Args:
            adjust_date: 调整日期，格式YYYY-MM-DD，如果为None则返回最新成分股
        
        Returns:
            constituents_df: 成分股DataFrame
            constituents_dict: 成分股字典 {code: StockInfo}
        """
        cache_key = f"hs300_constituents_{adjust_date}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if datetime.now() - self._cache_timestamp.get(cache_key, datetime.min()) < timedelta(hours=self.config.cache_expire_hours):
                return cached['df'], cached['dict']
        
        try:
            df = self.ak.index_stock_cons_weight_csindex(symbol="000300")
            
            df = df.rename(columns={
                '成分券代码': 'code',
                '成分券名称': 'name',
                '权重': 'weight',
            })
            df['weight'] = df['weight'].astype(float)
            
            constituents_dict = {}
            for _, row in df.iterrows():
                constituents_dict[row['code']] = StockInfo(
                    code=row['code'],
                    name=row['name'],
                    weight=row['weight'],
                    industry="",
                    market_cap=0.0,
                )
            
            self._cache[cache_key] = {
                'df': df,
                'dict': constituents_dict,
            }
            self._cache_timestamp[cache_key] = datetime.now()
            
            logger.info(f"获取沪深300成分股: {len(df)} 只")
            return df, constituents_dict
        except Exception as e:
            logger.warning(f"获取沪深300成分股失败: {e}")
            return pd.DataFrame(), {}
    
    def get_index_data(
        self,
        start_date: str,
        end_date: str,
        index_code: str = "000001",
    ) -> pd.DataFrame:
        """获取指数数据"""
        try:
            df = self.ak.stock_zh_index_daily(symbol=f"sh{index_code}")
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            return df
        except Exception as e:
            logger.warning(f"获取指数数据失败: {e}")
            return pd.DataFrame()
    
    def get_stock_data(
        self,
        start_date: str,
        end_date: str,
        codes: List[str],
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> pd.DataFrame:
        """获取股票行情数据（带重试机制）"""
        import time
        all_data = []
        
        for code in codes:
            for retry in range(max_retries):
                try:
                    df = self.ak.stock_zh_a_hist(
                        symbol=code,
                        period="daily",
                        start_date=start_date.replace("-", ""),
                        end_date=end_date.replace("-", ""),
                        adjust="qfq"
                    )
                    
                    if df is None or df.empty:
                        if retry < max_retries - 1:
                            time.sleep(retry_delay)
                            continue
                        else:
                            logger.warning(f"获取 {code} 数据为空，已重试 {max_retries} 次")
                        continue
                    
                    df = df.rename(columns={
                        '日期': 'date',
                        '开盘': 'open',
                        '最高': 'high',
                        '最低': 'low',
                        '收盘': 'close',
                        '成交量': 'volume',
                        '成交额': 'amount',
                        '换手率': 'turnover_rate',
                        '涨跌幅': 'pct_change',
                    })
                    df['code'] = code
                    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                    
                    cols = ['date', 'code', 'open', 'high', 'low', 'close', 'volume']
                    if 'amount' in df.columns:
                        cols.append('amount')
                    if 'turnover_rate' in df.columns:
                        cols.append('turnover_rate')
                    if 'pct_change' in df.columns:
                        cols.append('pct_change')
                    
                    all_data.append(df[cols])
                    logger.info(f"获取 {code} 数据: {len(df)} 条")
                    break
                except Exception as e:
                    if retry < max_retries - 1:
                        logger.warning(f"获取 {code} 数据失败 (尝试 {retry + 1}/{max_retries}): {e}")
                        time.sleep(retry_delay)
                    else:
                        logger.warning(f"获取 {code} 价格数据失败，已重试 {max_retries} 次: {e}")
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()
    
    def get_market_context(self, date: str):
        """获取市场环境"""
        from monitor.strategy_library.fusion import MarketRegime, MarketContext
        
        try:
            idx_df = self.get_index_data(
                start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                end_date=date
            )
            
            if idx_df.empty:
                return self._default_market_context(date)
            
            idx_value = float(idx_df['close'].iloc[-1])
            
            prev_close = idx_df['close'].iloc[-2] if len(idx_df) > 1 else idx_value
            index_change = (idx_value - prev_close) / prev_close
            
            returns = idx_df['close'].pct_change().dropna()
            volatility = float(returns.std()) if len(returns) > 1 else 0.02
            
            if index_change > 0.01:
                trend = "up"
            elif index_change < -0.01:
                trend = "down"
            else:
                trend = "neutral"
            
            if trend == "up":
                regime = MarketRegime.BULL_TRENDING if volatility < 0.02 else MarketRegime.BULL_CHOPPY
            elif trend == "down":
                regime = MarketRegime.BEAR_TRENDING if volatility < 0.02 else MarketRegime.BEAR_CHOPPY
            else:
                regime = MarketRegime.NEUTRAL
            
            return MarketContext(
                index_value=idx_value,
                index_change=float(index_change) if not pd.isna(index_change) else 0.0,
                volatility=volatility,
                trend=trend,
                regime=regime,
                uncertainty=max(0.3, min(0.7, 0.5 - volatility * 10)),
                timestamp=f"{date}T15:00:00",
            )
        except Exception as e:
            logger.warning(f"获取市场环境失败: {e}")
            return self._default_market_context(date)
    
    def _default_market_context(self, date: str):
        from monitor.strategy_library.fusion import MarketRegime, MarketContext
        
        return MarketContext(
            index_value=3000,
            index_change=0,
            volatility=0.02,
            trend="neutral",
            regime=MarketRegime.NEUTRAL,
            uncertainty=0.5,
            timestamp=f"{date}T15:00:00",
        )
    
    def clear_cache(self):
        self._cache.clear()
        self._cache_timestamp.clear()


class IndexEnhancedSystem:
    """沪深300指数增强系统"""
    
    def __init__(self, data_hub: DataHub):
        self.data_hub = data_hub
        self.constituents: List[StockInfo] = []
        self.weights: Dict[str, float] = {}
    
    def initialize(self, adjust_date: Optional[str] = None) -> int:
        """初始化沪深300成分股"""
        df, constituents_dict = self.data_hub.get_hs300_constituents(adjust_date)
        
        if constituents_dict:
            self.constituents = list(constituents_dict.values())
            self.weights = {s.code: s.weight for s in self.constituents}
            logger.info(f"初始化沪深300成分股: {len(self.constituents)} 只")
            logger.info(f"总权重: {sum(self.weights.values()):.4f}%")
        
        return len(self.constituents)
    
    def get_constituent_codes(self) -> List[str]:
        """获取成分股代码列表"""
        return [s.code for s in self.constituents]
    
    def get_constituent_weights(self) -> Dict[str, float]:
        """获取成分股权重"""
        return self.weights
    
    def get_constituent_data(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """获取所有成分股的历史数据"""
        codes = self.get_constituent_codes()
        return self.data_hub.get_stock_data(start_date, end_date, codes)
    
    def generate_model_predictions(
        self,
        price_data: pd.DataFrame,
        index_data: pd.DataFrame,
    ) -> Dict[str, List]:
        """生成模型预测 (预测相对收益)"""
        from monitor.strategy_library.fusion import ModelPrediction
        
        predictions = {}
        
        for code in price_data['code'].unique():
            code_df = price_data[price_data['code'] == code].copy()
            code_df = code_df.sort_values('date')
            
            code_df['ma5'] = code_df['close'].rolling(5).mean()
            code_df['ma10'] = code_df['close'].rolling(10).mean()
            code_df['ma20'] = code_df['close'].rolling(20).mean()
            
            code_df['return'] = code_df['close'].pct_change()
            code_df['momentum_5'] = code_df['close'].pct_change(5)
            code_df['momentum_10'] = code_df['close'].pct_change(10)
            code_df['momentum_20'] = code_df['close'].pct_change(20)
            code_df['volatility'] = code_df['return'].rolling(10).std()
            
            if 'pct_change' in code_df.columns:
                code_df['daily_return'] = code_df['pct_change'] / 100
            else:
                code_df['daily_return'] = code_df['return']
            
            if len(code_df) < 20:
                continue
            
            predictions[code] = []
            
            for i in range(20, len(code_df)):
                row = code_df.iloc[i]
                
                score = 0.5
                
                ma5 = row['ma5'] if not pd.isna(row['ma5']) else row['close']
                ma10 = row['ma10'] if not pd.isna(row['ma10']) else row['close']
                ma20 = row['ma20'] if not pd.isna(row['ma20']) else row['close']
                
                if ma5 > ma10 > ma20:
                    score += 0.15
                elif ma5 > ma10:
                    score += 0.08
                elif ma5 < ma10 < ma20:
                    score -= 0.15
                elif ma5 < ma10:
                    score -= 0.08
                
                mom5 = row['momentum_5'] if not pd.isna(row['momentum_5']) else 0
                mom10 = row['momentum_10'] if not pd.isna(row['momentum_10']) else 0
                mom20 = row['momentum_20'] if not pd.isna(row['momentum_20']) else 0
                
                avg_momentum = (mom5 + mom10 + mom20) / 3
                score += np.clip(avg_momentum * 2, -0.15, 0.15)
                
                if 'turnover_rate' in code_df.columns:
                    turnover = row['turnover_rate'] if not pd.isna(row['turnover_rate']) else 0
                    if turnover > 5:
                        score -= 0.05
                    elif turnover < 1:
                        score += 0.03
                
                score = np.clip(score, 0.15, 0.85)
                
                volatility_val = row['volatility'] if not pd.isna(row['volatility']) else 0.02
                confidence = max(0.35, min(0.85, 0.7 - volatility_val * 5))
                
                pred = ModelPrediction(
                    code=code,
                    name=code,
                    score=score,
                    confidence=confidence,
                    std=volatility_val,
                    horizon=5,
                    timestamp=f"{row['date']}T09:30:00",
                )
                predictions[code].append(pred)
        
        return predictions
    
    def generate_strategy_signals(
        self,
        price_data: pd.DataFrame,
    ) -> Dict[str, List]:
        """生成策略信号"""
        from monitor.strategy_library.fusion import StrategySignalWrapper
        
        signals = {}
        
        for code in price_data['code'].unique():
            code_df = price_data[price_data['code'] == code].copy()
            code_df = code_df.sort_values('date')
            
            code_df['ma5'] = code_df['close'].rolling(5).mean()
            code_df['ma10'] = code_df['close'].rolling(10).mean()
            code_df['ma20'] = code_df['close'].rolling(20).mean()
            
            code_df['return'] = code_df['close'].pct_change()
            code_df['volatility'] = code_df['return'].rolling(10).std()
            
            if 'pct_change' in code_df.columns:
                code_df['daily_return'] = code_df['pct_change'] / 100
            else:
                code_df['daily_return'] = code_df['return']
            
            signals[code] = []
            
            for i in range(20, len(code_df)):
                row = code_df.iloc[i]
                prev_row = code_df.iloc[i-1]
                
                action = "hold"
                strength = 0.3
                reason = ""
                
                ma5 = row['ma5'] if not pd.isna(row['ma5']) else row['close']
                ma10 = row['ma10'] if not pd.isna(row['ma10']) else row['close']
                ma20 = row['ma20'] if not pd.isna(row['ma20']) else row['close']
                
                prev_ma5 = prev_row['ma5'] if not pd.isna(prev_row['ma5']) else prev_row['close']
                prev_ma10 = prev_row['ma10'] if not pd.isna(prev_row['ma10']) else prev_row['close']
                prev_ma20 = prev_row['ma20'] if not pd.isna(prev_row['ma20']) else prev_row['close']
                
                golden_cross_5_10 = ma5 > ma10 and prev_ma5 <= prev_ma10
                death_cross_5_10 = ma5 < ma10 and prev_ma5 >= prev_ma10
                
                golden_cross_10_20 = ma10 > ma20 and prev_ma10 <= prev_ma20
                death_cross_10_20 = ma10 < ma20 and prev_ma10 >= prev_ma20
                
                if golden_cross_5_10:
                    action = "buy"
                    strength = 0.7
                    reason = "MA5上穿MA10"
                elif death_cross_5_10:
                    action = "sell"
                    strength = 0.6
                    reason = "MA5下穿MA10"
                elif golden_cross_10_20:
                    action = "buy"
                    strength = 0.65
                    reason = "MA10上穿MA20"
                elif death_cross_10_20:
                    action = "sell"
                    strength = 0.55
                    reason = "MA10下穿MA20"
                elif ma5 > ma10 > ma20:
                    if prev_ma5 > prev_ma10 > prev_ma20:
                        pass
                    else:
                        action = "buy"
                        strength = 0.5
                        reason = "多头排列确认"
                elif ma5 < ma10 < ma20:
                    if prev_ma5 < prev_ma10 < prev_ma20:
                        pass
                    else:
                        action = "sell"
                        strength = 0.45
                        reason = "空头排列确认"
                elif ma5 > ma10 * 1.02 and ma5 > ma20 * 1.03:
                    action = "buy"
                    strength = 0.4
                    reason = "强势上涨"
                elif ma5 < ma10 * 0.98 and ma5 < ma20 * 0.97:
                    action = "sell"
                    strength = 0.35
                    reason = "弱势下跌"
                
                if action != "hold":
                    sig = StrategySignalWrapper(
                        code=code,
                        name=code,
                        action=action,
                        strength=strength,
                        reason=reason,
                        strategy_id="ma_cross",
                        strategy_name="均线交叉策略",
                        timestamp=f"{row['date']}T15:00:00",
                    )
                    signals[code].append(sig)
        
        return signals
    
    def run(
        self,
        start_date: str,
        end_date: str,
    ) -> Dict[str, Any]:
        """运行指数增强系统"""
        logger.info(f"开始运行指数增强系统: {start_date} ~ {end_date}")
        
        if not self.constituents:
            self.initialize()
        
        logger.info("获取成分股数据...")
        price_data = self.get_constituent_data(start_date, end_date)
        
        if price_data.empty:
            logger.error("无法获取成分股数据")
            return {}
        
        logger.info(f"获取到 {len(price_data)} 条价格记录")
        
        logger.info("获取指数数据...")
        index_data = self.data_hub.get_index_data(start_date, end_date)
        
        logger.info("生成模型预测...")
        predictions = self.generate_model_predictions(price_data, index_data)
        
        logger.info("生成策略信号...")
        signals = self.generate_strategy_signals(price_data)
        
        logger.info("获取市场环境...")
        dates = sorted(price_data['date'].unique())
        market_contexts = {}
        for date in dates:
            market_contexts[date] = self.data_hub.get_market_context(date)
        
        from monitor.strategy_library.fusion import FusionBacktester, FusionConfig
        
        config = FusionConfig(
            consistency_boost=1.3,
            conflict_penalty=0.7,
        )
        
        backtester = FusionBacktester(
            config=config,
            initial_capital=1000000,
        )
        
        logger.info("运行回测...")
        fused_result, model_result, strategy_result, gain_metrics = backtester.run_backtest(
            price_data=price_data,
            model_predictions=predictions,
            strategy_signals=signals,
            market_contexts=market_contexts,
            start_date=dates[0],
            end_date=dates[-1],
        )
        
        return {
            'fused_result': fused_result,
            'model_result': model_result,
            'strategy_result': strategy_result,
            'gain_metrics': gain_metrics,
            'constituents': self.constituents,
            'weights': self.weights,
        }


def run_index_enhanced(
    start_date: str = "2024-01-01",
    end_date: str = "2024-03-31",
) -> Dict[str, Any]:
    """运行沪深300指数增强"""
    config = DataSourceConfig()
    data_hub = DataHub(config)
    system = IndexEnhancedSystem(data_hub)
    
    result = system.run(start_date, end_date)
    
    if not result:
        logger.error("运行失败")
        return {}
    
    fused = result['fused_result']
    model = result['model_result']
    strategy = result['strategy_result']
    gain = result['gain_metrics']
    
    print("\n" + "=" * 60)
    print("沪深300指数增强回测结果")
    print("=" * 60)
    
    print(f"\n{'指标':<15} {'融合策略':>12} {'模型单独':>12} {'策略单独':>12}")
    print("-" * 55)
    print(f"{'总收益':<15} {fused.total_return*100:>11.2f}% {model.total_return*100:>11.2f}% {strategy.total_return*100:>11.2f}%")
    print(f"{'年化收益':<15} {fused.annual_return*100:>11.2f}% {model.annual_return*100:>11.2f}% {strategy.annual_return*100:>11.2f}%")
    print(f"{'夏普比率':<15} {fused.sharpe_ratio:>12.2f} {model.sharpe_ratio:>12.2f} {strategy.sharpe_ratio:>12.2f}")
    print(f"{'最大回撤':<15} {fused.max_drawdown*100:>11.2f}% {model.max_drawdown*100:>11.2f}% {strategy.max_drawdown*100:>11.2f}%")
    print(f"{'胜率':<15} {fused.win_rate*100:>11.2f}% {model.win_rate*100:>11.2f}% {strategy.win_rate*100:>11.2f}%")
    print(f"{'交易次数':<15} {fused.total_trades:>12} {model.total_trades:>12} {strategy.total_trades:>12}")
    
    print("\n" + "=" * 60)
    print("增益效应分析")
    print("=" * 60)
    
    print(f"\n{'指标':<15} {'增益值':>12} {'状态':>10}")
    print("-" * 40)
    print(f"{'收益增益':<15} {gain.return_gain*100:>11.2f}% {'✓ 正向' if gain.return_gain > 0 else '✗ 负向':>10}")
    print(f"{'夏普增益':<15} {gain.sharpe_gain:>12.2f} {'✓ 正向' if gain.sharpe_gain > 0 else '✗ 负向':>10}")
    print(f"{'回撤增益':<15} {gain.drawdown_gain*100:>11.2f}% {'✓ 改善' if gain.drawdown_gain < 0 else '✗ 恶化':>10}")
    print(f"{'胜率增益':<15} {gain.winrate_gain*100:>11.2f}% {'✓ 正向' if gain.winrate_gain > 0 else '✗ 负向':>10}")
    
    print("\n" + "=" * 60)
    print(f"增益效应: {'存在' if gain.has_gain() else '不存在'}")
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_index_enhanced()
