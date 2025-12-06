"""
加权集成聚合器：基于验证集 ICIR 的加权平均和 Meta-Learner
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from predictor.weight_dynamic import RankICDynamicWeighter

logger = logging.getLogger(__name__)


class ICIRWeightedAverageAdapter:
    """
    基于验证集 ICIR 的加权平均聚合器。
    
    在训练时计算各模型在验证集上的 IC-IR，然后使用这些 IC-IR 作为权重进行加权平均。
    """

    def __init__(self):
        self.weights: Optional[Dict[str, float]] = None
        self.ic_weighter = RankICDynamicWeighter(
            window=1,  # 只使用当前验证集的 IC
            half_life=1,
            min_weight=0.1,
            max_weight=0.8,
            clip_negative=True,
        )

    def fit(self, valid_preds: Dict[str, pd.Series], valid_label: pd.Series):
        """
        在验证集上计算各模型的 IC-IR，并生成权重。
        
        参数:
            valid_preds: {模型名: 验证集预测值}
            valid_label: 验证集标签
        """
        if not valid_preds or valid_label.empty:
            logger.warning("验证集为空，无法计算 IC-IR 权重，回退为等权")
            self.weights = {name: 1.0 / len(valid_preds) for name in valid_preds.keys()}
            return

        # 计算每个模型在验证集上的 IC
        ic_values = {}
        for name, pred in valid_preds.items():
            if pred.empty:
                continue
            aligned_pred, aligned_label = pred.align(valid_label, join="inner")
            if aligned_pred.empty:
                continue
            ic = aligned_pred.rank().corr(aligned_label, method="spearman")
            if not np.isnan(ic):
                ic_values[name] = ic

        if not ic_values:
            logger.warning("无法计算任何模型的 IC，回退为等权")
            self.weights = {name: 1.0 / len(valid_preds) for name in valid_preds.keys()}
            return

        # 将单期 IC 转换为 Series（用于 IC-IR 计算）
        # 由于只有一期，IC-IR 就是 IC 的绝对值（或直接使用 IC）
        ic_series = {name: pd.Series([ic]) for name, ic in ic_values.items()}

        # 使用 RankICDynamicWeighter 计算权重（它会处理归一化、clip 等）
        self.weights = self.ic_weighter.get_weights(ic_series)

        # 确保所有模型都有权重（即使 IC 为 NaN）
        for name in valid_preds.keys():
            if name not in self.weights:
                self.weights[name] = 0.0

        # 归一化
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
        else:
            # 回退为等权
            self.weights = {name: 1.0 / len(valid_preds) for name in valid_preds.keys()}

        logger.info(f"IC-IR 权重: {self.weights}")

    def __call__(self, preds: Dict[str, pd.Series]) -> pd.Series:
        """
        使用训练时计算的权重进行加权平均。
        
        参数:
            preds: {模型名: 预测值}
        
        返回:
            加权平均后的预测值
        """
        if not preds:
            raise ValueError("无可融合的预测结果")

        if len(preds) == 1:
            return next(iter(preds.values())).rename("weighted_ensemble")

        if self.weights is None:
            logger.warning("权重未初始化，使用等权平均")
            self.weights = {name: 1.0 / len(preds) for name in preds.keys()}

        # 确保所有预测都有对应的权重
        weights = {name: self.weights.get(name, 0.0) for name in preds.keys()}

        # 加权平均
        combined = None
        for name, series in preds.items():
            w = weights.get(name, 0.0)
            if w == 0.0:
                continue
            contrib = series * w
            combined = contrib if combined is None else combined.add(contrib, fill_value=0.0)

        if combined is None:
            # 如果所有权重都是 0，回退为等权
            logger.warning("所有权重为 0，回退为等权平均")
            combined = sum(preds.values()) / len(preds)

        return combined.rename("weighted_ensemble")


class MetaLearnerAdapter:
    """
    Meta-Learner 聚合器：使用线性回归或 Ridge 回归学习如何组合各模型的预测。
    
    在训练时，Meta-Learner 学习：label = f(pred_lgb, pred_mlp, ...)
    在预测时，使用训练好的 Meta-Learner 进行融合。
    """

    def __init__(self, model_type: str = "ridge", alpha: float = 1.0):
        """
        参数:
            model_type: "linear" 或 "ridge"
            alpha: Ridge 回归的正则化系数（仅当 model_type="ridge" 时使用）
        """
        self.model_type = model_type.lower()
        if self.model_type == "linear":
            self.model = LinearRegression()
        elif self.model_type == "ridge":
            self.model = Ridge(alpha=alpha)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}，支持 'linear' 或 'ridge'")

        self.scaler = StandardScaler()
        self.model_names: Optional[list] = None
        self.is_fitted = False

    def fit(self, valid_preds: Dict[str, pd.Series], valid_label: pd.Series):
        """
        在验证集上训练 Meta-Learner。
        
        参数:
            valid_preds: {模型名: 验证集预测值}
            valid_label: 验证集标签
        """
        if not valid_preds or valid_label.empty:
            logger.warning("验证集为空，无法训练 Meta-Learner，回退为等权")
            self.is_fitted = False
            return

        # 对齐所有预测和标签
        aligned_preds = {}
        for name, pred in valid_preds.items():
            if pred.empty:
                continue
            aligned_pred, aligned_label = pred.align(valid_label, join="inner")
            if not aligned_pred.empty:
                aligned_preds[name] = aligned_pred

        if not aligned_preds:
            logger.warning("无法对齐任何预测，无法训练 Meta-Learner")
            self.is_fitted = False
            return

        # 构建特征矩阵：每列是一个模型的预测
        self.model_names = list(aligned_preds.keys())
        X = pd.DataFrame(aligned_preds)
        y = valid_label.reindex(X.index)

        # 移除缺失值
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]

        if len(X) < 2:
            logger.warning(f"有效样本数太少 ({len(X)})，无法训练 Meta-Learner")
            self.is_fitted = False
            return

        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        y_array = y.values

        # 训练模型
        self.model.fit(X_scaled, y_array)
        self.is_fitted = True

        # 打印系数（权重）
        coef = self.model.coef_
        intercept = self.model.intercept_
        logger.info(f"Meta-Learner ({self.model_type}) 训练完成:")
        logger.info(f"  截距: {intercept:.6f}")
        for name, c in zip(self.model_names, coef):
            logger.info(f"  {name}: {c:.6f}")

    def __call__(self, preds: Dict[str, pd.Series]) -> pd.Series:
        """
        使用训练好的 Meta-Learner 进行预测融合。
        
        参数:
            preds: {模型名: 预测值}
        
        返回:
            Meta-Learner 融合后的预测值
        """
        if not preds:
            raise ValueError("无可融合的预测结果")

        if len(preds) == 1:
            return next(iter(preds.values())).rename("meta_ensemble")

        if not self.is_fitted:
            logger.warning("Meta-Learner 未训练，使用等权平均")
            return sum(preds.values()) / len(preds)

        # 确保使用训练时的模型顺序
        if self.model_names is None:
            self.model_names = list(preds.keys())

        # 构建特征矩阵
        X = pd.DataFrame({name: preds.get(name, pd.Series()) for name in self.model_names})

        # 标准化
        try:
            X_scaled = self.scaler.transform(X)
        except Exception as e:
            logger.warning(f"标准化失败: {e}，使用原始值")
            X_scaled = X.values

        # 预测
        y_pred = self.model.predict(X_scaled)

        # 转换为 Series
        result = pd.Series(y_pred, index=X.index, name="meta_ensemble")
        return result

