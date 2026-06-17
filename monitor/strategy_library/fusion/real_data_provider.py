"""
真实数据提供者 - Real Data Provider

基于akshare获取真实市场数据，用于融合系统验证。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from monitor.strategy_library.fusion import (
    ModelPrediction,
    StrategySignalWrapper,
    MarketContext,
    MarketRegime,
    LayeredFusionProcessor,
    FusionBacktester,
)

logger = logging.getLogger(__name__)


class RealDataProvider:
    """基于akshare的真实数据提供者"""
    
    def __init__(self):
        self._akshare = None
        self.default_codes = [
            "000001",
            "000002",
            "600000",
            "600036",
            "601318",
        ]
    
    @property
    def ak(self):
        if self._akshare is None:
            import akshare as ak
            self._akshare = ak
        return self._akshare
    
    def get_price_data(
        self,
        start_date: str,
        end_date: str,
        codes: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        codes = codes or self.default_codes
        all_data = []
        
        for code in codes:
            try:
                df = self.ak.stock_zh_a_hist(
                    symbol=code,
                    period="daily",
                    start_date=start_date.replace("-", ""),
                    end_date=end_date.replace("-", ""),
                    adjust="qfq"
                )
                
                if df is not None and not df.empty:
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
            except Exception as e:
                logger.warning(f"获取 {code} 价格数据失败: {e}")
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()
    
    def get_index_data(
        self,
        start_date: str,
        end_date: str,
        index_code: str = "000001",
    ) -> pd.DataFrame:
        try:
            df = self.ak.stock_zh_index_daily(symbol=f"sh{index_code}")
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            return df
        except Exception as e:
            logger.warning(f"获取指数数据失败: {e}")
            return pd.DataFrame()
    
    def get_market_context(self, date: str) -> MarketContext:
        try:
            df = self.ak.stock_zh_index_daily(symbol="sh000001")
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            
            date_data = df[df['date'] == date]
            if date_data.empty:
                return self._default_market_context(date)
            
            row = date_data.iloc[-1]
            prev_data = df[df['date'] < date].tail(20)
            
            index_value = float(row['close'])
            index_change = float(row.get('pct_change', 0)) / 100 if 'pct_change' in row else 0
            
            if not prev_data.empty:
                returns = prev_data['close'].pct_change().dropna()
                volatility = float(returns.std())
            else:
                volatility = 0.02
            
            if index_change > 0.01:
                trend = "up"
                regime = MarketRegime.BULL_TRENDING if volatility < 0.02 else MarketRegime.BULL_CHOPPY
            elif index_change < -0.01:
                trend = "down"
                regime = MarketRegime.BEAR_TRENDING if volatility < 0.02 else MarketRegime.BEAR_CHOPPY
            else:
                trend = "neutral"
                regime = MarketRegime.NEUTRAL
            
            return MarketContext(
                index_value=index_value,
                index_change=index_change,
                volatility=volatility,
                trend=trend,
                regime=regime,
                uncertainty=0.3,
                timestamp=f"{date}T15:00:00",
            )
        except Exception as e:
            logger.warning(f"获取市场环境失败: {e}")
            return self._default_market_context(date)
    
    def _default_market_context(self, date: str) -> MarketContext:
        return MarketContext(
            index_value=3000,
            index_change=0,
            volatility=0.02,
            trend="neutral",
            regime=MarketRegime.NEUTRAL,
            uncertainty=0.5,
            timestamp=f"{date}T15:00:00",
        )


class RealDataFusionRunner:
    """真实数据融合回测运行器"""
    
    def __init__(self, data_provider: RealDataProvider):
        self.data_provider = data_provider
        self.fusion_processor = LayeredFusionProcessor()
        self.backtester = FusionBacktester(initial_capital=1000000)
    
    def run(
        self,
        start_date: str,
        end_date: str,
        codes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        codes = codes or self.data_provider.default_codes
        
        logger.info(f"获取价格数据: {start_date} ~ {end_date}")
        price_data = self.data_provider.get_price_data(start_date, end_date, codes)
        
        if price_data.empty:
            logger.error("价格数据为空，无法运行回测")
            return {}
        
        logger.info(f"获取到 {len(price_data)} 条价格记录")
        
        logger.info("生成模型预测...")
        model_predictions = self._generate_model_predictions(price_data, codes)
        
        logger.info("生成策略信号...")
        strategy_signals = self._generate_strategy_signals(price_data, codes)
        
        logger.info("获取市场环境...")
        dates = sorted(price_data['date'].unique())
        market_contexts = {}
        for date in dates:
            market_contexts[date] = self.data_provider.get_market_context(date)
        
        logger.info("运行回测...")
        fused_result, model_result, strategy_result, gain_metrics = self.backtester.run_backtest(
            price_data=price_data,
            model_predictions=model_predictions,
            strategy_signals=strategy_signals,
            market_contexts=market_contexts,
            start_date=dates[0],
            end_date=dates[-1],
        )
        
        return {
            "fused_result": fused_result,
            "model_result": model_result,
            "strategy_result": strategy_result,
            "gain_metrics": gain_metrics,
            "price_data": price_data,
        }
    
    def _generate_model_predictions(
        self,
        price_data: pd.DataFrame,
        codes: List[str],
    ) -> Dict[str, List[ModelPrediction]]:
        predictions = {}
        
        for code in codes:
            code_df = price_data[price_data['code'] == code].copy()
            if code_df.empty:
                continue
            
            code_df = code_df.sort_values('date')
            code_df['ma5'] = code_df['close'].rolling(5).mean()
            code_df['ma20'] = code_df['close'].rolling(20).mean()
            code_df['rsi'] = self._calculate_rsi(code_df['close'], 14)
            code_df['macd'], code_df['macd_signal'] = self._calculate_macd(code_df['close'])
            
            predictions[code] = []
            
            for i in range(20, len(code_df)):
                row = code_df.iloc[i]
                
                score = 0.5
                confidence = 0.5
                
                ma_trend = 1 if row['ma5'] > row['ma20'] else -1
                rsi_val = row['rsi'] if not pd.isna(row['rsi']) else 50
                macd_val = row['macd'] if not pd.isna(row['macd']) else 0
                
                score += ma_trend * 0.1
                score += (50 - rsi_val) / 200
                score += np.tanh(macd_val / row['close'] * 100) * 0.1
                
                score = np.clip(score, 0.1, 0.9)
                
                volatility = code_df['close'].pct_change().tail(10).std()
                confidence = max(0.3, min(0.9, 1 - volatility * 10))
                
                pred = ModelPrediction(
                    code=code,
                    name=code,
                    score=score,
                    confidence=confidence,
                    std=volatility if not pd.isna(volatility) else 0.02,
                    horizon=5,
                    timestamp=f"{row['date']}T09:30:00",
                )
                predictions[code].append(pred)
        
        return predictions
    
    def _generate_strategy_signals(
        self,
        price_data: pd.DataFrame,
        codes: List[str],
    ) -> Dict[str, List[StrategySignalWrapper]]:
        signals = {}
        
        for code in codes:
            code_df = price_data[price_data['code'] == code].copy()
            if code_df.empty:
                continue
            
            code_df = code_df.sort_values('date')
            code_df['ma5'] = code_df['close'].rolling(5).mean()
            code_df['ma20'] = code_df['close'].rolling(20).mean()
            
            signals[code] = []
            
            for i in range(20, len(code_df)):
                row = code_df.iloc[i]
                prev_row = code_df.iloc[i-1]
                
                action = "hold"
                strength = 0.3
                reason = ""
                
                golden_cross = row['ma5'] > row['ma20'] and prev_row['ma5'] <= prev_row['ma20']
                death_cross = row['ma5'] < row['ma20'] and prev_row['ma5'] >= prev_row['ma20']
                
                if golden_cross:
                    action = "buy"
                    strength = 0.7
                    reason = "金叉买入"
                elif death_cross:
                    action = "sell"
                    strength = 0.6
                    reason = "死叉卖出"
                elif row['ma5'] > row['ma20'] * 1.02:
                    action = "buy"
                    strength = 0.5
                    reason = "均线多头排列"
                elif row['ma5'] < row['ma20'] * 0.98:
                    action = "sell"
                    strength = 0.5
                    reason = "均线空头排列"
                
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
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(
        self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[pd.Series, pd.Series]:
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        return macd, macd_signal


def run_real_data_backtest(
    start_date: str = "2024-01-01",
    end_date: str = "2024-03-31",
    codes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """运行真实数据回测"""
    data_provider = RealDataProvider()
    runner = RealDataFusionRunner(data_provider)
    
    result = runner.run(start_date, end_date, codes)
    
    if not result:
        logger.error("回测失败")
        return {}
    
    fused = result['fused_result']
    model = result['model_result']
    strategy = result['strategy_result']
    gain = result['gain_metrics']
    
    print("\n" + "=" * 60)
    print("真实数据回测结果")
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
    run_real_data_backtest()
