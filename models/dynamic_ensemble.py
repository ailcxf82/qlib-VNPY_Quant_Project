"""
动态权重集成策略
基于各模型历史IC表现动态调整集成权重，提升ICIR
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class DynamicWeightedEnsemble:
    """
    动态权重集成器
    根据各模型近期IC表现动态调整权重
    """
    
    def __init__(
        self,
        window: int = 60,
        half_life: int = 20,
        min_weight: float = 0.05,
        max_weight: float = 0.5,
        clip_negative: bool = True,
        use_softmax: bool = False,
        temperature: float = 1.0,
    ):
        """
        Args:
            window: IC历史窗口大小
            half_life: 指数衰减半衰期
            min_weight: 单个模型最小权重
            max_weight: 单个模型最大权重
            clip_negative: 是否将负ICIR模型权重设为min_weight
            use_softmax: 是否使用softmax平滑权重
            temperature: softmax温度参数
        """
        self.window = window
        self.half_life = half_life
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.clip_negative = clip_negative
        self.use_softmax = use_softmax
        self.temperature = temperature
        
        # 存储每个模型的IC历史
        self.ic_history: Dict[str, List[float]] = {}
        
        logger.info(
            "动态权重集成器初始化: window=%d, half_life=%d, min_weight=%.2f, max_weight=%.2f",
            window, half_life, min_weight, max_weight,
        )
    
    def calculate_ic(self, pred: pd.Series, label: pd.Series) -> float:
        """计算Rank IC"""
        pred, label = pred.align(label, join="inner")
        if pred.empty or len(pred) < 10:
            return np.nan
        
        # Spearman相关系数
        ic, _ = stats.spearmanr(pred.values, label.values)
        return ic
    
    def update_ic(self, model_name: str, ic: float):
        """更新模型的IC历史"""
        if model_name not in self.ic_history:
            self.ic_history[model_name] = []
        
        self.ic_history[model_name].append(ic)
        
        # 只保留最近window个IC
        if len(self.ic_history[model_name]) > self.window:
            self.ic_history[model_name] = self.ic_history[model_name][-self.window:]
    
    def calculate_icir(self, ic_series: List[float]) -> float:
        """计算ICIR"""
        if len(ic_series) < 2:
            return 0.0
        
        ic_arr = np.array(ic_series)
        mean_ic = np.nanmean(ic_arr)
        std_ic = np.nanstd(ic_arr)
        
        if std_ic < 1e-8:
            return 0.0
        
        return mean_ic / std_ic
    
    def calculate_weights(self) -> Dict[str, float]:
        """
        计算动态权重
        
        Returns:
            {model_name: weight}
        """
        if not self.ic_history:
            return {}
        
        weights = {}
        
        for model_name, ic_series in self.ic_history.items():
            if len(ic_series) == 0:
                weights[model_name] = self.min_weight
                continue
            
            # 指数衰减权重（近期权重更高）
            n = len(ic_series)
            decay_weights = np.exp(-np.log(2) / self.half_life * np.arange(n))
            decay_weights = decay_weights[::-1]  # 近期权重更高
            decay_weights = decay_weights / decay_weights.sum()
            
            # 加权IC和ICIR
            ic_arr = np.array(ic_series)
            weighted_ic = np.nansum(ic_arr * decay_weights)
            
            # 计算加权标准差
            weighted_var = np.nansum(decay_weights * (ic_arr - weighted_ic) ** 2)
            weighted_std = np.sqrt(weighted_var)
            
            icir = weighted_ic / (weighted_std + 1e-8)
            
            # 处理负ICIR
            if self.clip_negative and icir < 0:
                weights[model_name] = self.min_weight
            else:
                weights[model_name] = max(icir, 0)
        
        # 归一化权重
        if self.use_softmax:
            # Softmax平滑
            weights_arr = np.array(list(weights.values()))
            exp_weights = np.exp(weights_arr / self.temperature)
            softmax_weights = exp_weights / exp_weights.sum()
            
            for i, model_name in enumerate(weights.keys()):
                weights[model_name] = float(softmax_weights[i])
        else:
            # 简单归一化
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}
        
        # 裁剪权重到[min_weight, max_weight]
        for model_name in weights:
            weights[model_name] = np.clip(weights[model_name], self.min_weight, self.max_weight)
        
        # 再次归一化
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def aggregate(
        self,
        preds: Dict[str, pd.Series],
        labels: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        聚合多个模型的预测
        
        Args:
            preds: {model_name: prediction_series}
            labels: 真实标签（可选，用于更新IC历史）
        
        Returns:
            ensemble_pred: 集成后的预测
        """
        if not preds:
            raise ValueError("无可融合的预测结果")
        
        # 如果提供了标签，更新IC历史
        if labels is not None:
            for model_name, pred in preds.items():
                ic = self.calculate_ic(pred, labels)
                if not np.isnan(ic):
                    self.update_ic(model_name, ic)
        
        # 计算动态权重
        weights = self.calculate_weights()
        
        if not weights:
            # 如果没有历史IC，使用简单平均
            logger.warning("无历史IC数据，使用简单平均")
            weights = {name: 1.0 / len(preds) for name in preds.keys()}
        
        # 加权平均
        ensemble_pred = None
        for model_name, pred in preds.items():
            weight = weights.get(model_name, self.min_weight)
            
            if ensemble_pred is None:
                ensemble_pred = pred * weight
            else:
                ensemble_pred = ensemble_pred + pred * weight
        
        # 记录权重信息
        logger.info("动态权重: %s", {k: f"{v:.3f}" for k, v in weights.items()})
        
        return ensemble_pred.rename("dynamic_ensemble")
    
    def get_ic_summary(self) -> pd.DataFrame:
        """获取IC汇总统计"""
        if not self.ic_history:
            return pd.DataFrame()
        
        summary = []
        for model_name, ic_series in self.ic_history.items():
            ic_arr = np.array(ic_series)
            
            summary.append({
                "model": model_name,
                "ic_mean": np.nanmean(ic_arr),
                "ic_std": np.nanstd(ic_arr),
                "icir": self.calculate_icir(ic_series),
                "ic_positive_ratio": np.nanmean(ic_arr > 0),
                "n_samples": len(ic_series),
            })
        
        return pd.DataFrame(summary).sort_values("icir", ascending=False)


class AdaptiveEnsemble:
    """
    自适应集成器
    根据市场状态动态调整集成策略
    """
    
    def __init__(self, volatility_window: int = 20):
        """
        Args:
            volatility_window: 波动率计算窗口
        """
        self.volatility_window = volatility_window
        self.market_state = "normal"  # normal, high_vol, low_vol
        
        logger.info("自适应集成器初始化: volatility_window=%d", volatility_window)
    
    def detect_market_state(self, returns: pd.Series) -> str:
        """
        检测市场状态
        
        Args:
            returns: 市场收益率序列
        
        Returns:
            market_state: "normal", "high_vol", "low_vol"
        """
        if len(returns) < self.volatility_window:
            return "normal"
        
        # 计算近期波动率
        recent_vol = returns.tail(self.volatility_window).std()
        
        # 计算历史波动率分位数
        rolling_vol = returns.rolling(self.volatility_window).std()
        vol_percentile = (rolling_vol <= recent_vol).mean()
        
        # 判断市场状态
        if vol_percentile > 0.8:
            return "high_vol"
        elif vol_percentile < 0.2:
            return "low_vol"
        else:
            return "normal"
    
    def aggregate(
        self,
        preds: Dict[str, pd.Series],
        market_returns: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        根据市场状态自适应集成
        
        Args:
            preds: {model_name: prediction_series}
            market_returns: 市场收益率（用于检测市场状态）
        
        Returns:
            ensemble_pred: 集成后的预测
        """
        if not preds:
            raise ValueError("无可融合的预测结果")
        
        # 检测市场状态
        if market_returns is not None:
            self.market_state = self.detect_market_state(market_returns)
        
        # 根据市场状态调整权重
        if self.market_state == "high_vol":
            # 高波动市场：降低时序模型权重，增加树模型权重
            weights = {
                "lgb": 0.6,
                "gru": 0.3,
                "mlp": 0.1,
            }
        elif self.market_state == "low_vol":
            # 低波动市场：增加时序模型权重
            weights = {
                "lgb": 0.4,
                "gru": 0.5,
                "mlp": 0.1,
            }
        else:
            # 正常市场：均衡权重
            weights = {
                "lgb": 0.5,
                "gru": 0.4,
                "mlp": 0.1,
            }
        
        # 过滤不存在的模型
        weights = {k: v for k, v in weights.items() if k in preds}
        
        # 归一化
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        # 加权平均
        ensemble_pred = None
        for model_name, pred in preds.items():
            weight = weights.get(model_name, 1.0 / len(preds))
            
            if ensemble_pred is None:
                ensemble_pred = pred * weight
            else:
                ensemble_pred = ensemble_pred + pred * weight
        
        logger.info(
            "自适应集成: market_state=%s, weights=%s",
            self.market_state,
            {k: f"{v:.3f}" for k, v in weights.items()},
        )
        
        return ensemble_pred.rename("adaptive_ensemble")
