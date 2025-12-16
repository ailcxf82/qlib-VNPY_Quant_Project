"""
LightGBM 模型封装，调用 qlib.contrib.model.gbdt.LGBModel 训练并输出叶子索引。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from qlib.contrib.model.gbdt import LGBModel

from utils import load_yaml_config
from utils.dataset import PandasDataset
logger = logging.getLogger(__name__)


class LightGBMModelWrapper:
    """结合 qlib LGBModel 的二次封装。"""

    def __init__(self, config_path: str):
        cfg = load_yaml_config(config_path)
        self.config = cfg["model"]
        
        # 支持自定义损失函数（非对称损失）
        loss = self.config.get("loss", "mse")
        loss_params = self.config.get("loss_params", {})
        
        # 如果使用非对称损失
        if loss == "asymmetric_mse":
            from utils.loss_functions import asymmetric_mse_objective_lgb, asymmetric_mse_metric_lgb
            gamma = loss_params.get("gamma", 2.0)
            # 创建自定义目标函数和评估指标
            def custom_objective(y_true, y_pred):
                return asymmetric_mse_objective_lgb(y_true, y_pred, gamma=gamma)
            
            def custom_metric(y_true, y_pred):
                return asymmetric_mse_metric_lgb(y_true, y_pred, gamma=gamma)
            
            # 注意：qlib 的 LGBModel 可能不支持自定义目标函数
            # 这里我们需要直接使用 lightgbm 的接口
            # 暂时使用 mse，然后在 fit 方法中处理
            logger.warning("非对称损失函数需要在 fit 方法中通过 lightgbm 原生接口实现")
            loss = "mse"  # 暂时使用 mse
            self._use_asymmetric_loss = True
            self._asymmetric_gamma = gamma
        else:
            self._use_asymmetric_loss = False
            self._asymmetric_gamma = None
        
        self.model = LGBModel(
            loss=loss,
            num_boost_round=self.config.get("num_boost_round", 1000),
            early_stopping_rounds=self.config.get("early_stopping_rounds", 50),
            **self.config.get("params", {}),
        )
        self.booster: Optional[lgb.Booster] = None
        self.feature_names: Optional[list[str]] = None

    def fit(
        self,
        train_feat: pd.DataFrame,
        train_label: pd.Series,
        valid_feat: Optional[pd.DataFrame] = None,
        valid_label: Optional[pd.Series] = None,
    ):
        # 通过 PandasDataset 向 qlib 声明训练/验证时间切片
        segments = {
            "train": (
                train_feat.index.get_level_values("datetime").min(),
                train_feat.index.get_level_values("datetime").max(),
            )
        }
        features = train_feat
        labels = train_label
        has_valid = (
            valid_feat is not None
            and valid_label is not None
            and len(valid_feat) > 0
            and len(valid_label) > 0
        )
        if has_valid:
            segments["valid"] = (
                valid_feat.index.get_level_values("datetime").min(),
                valid_feat.index.get_level_values("datetime").max(),
            )
            features = pd.concat([train_feat, valid_feat], axis=0)
            labels = pd.concat([train_label, valid_label], axis=0)
        else:
            logger.warning("验证集为空，LightGBM 将仅使用训练数据")

        dataset = PandasDataset(features=features, labels=labels, segments=segments)
        logger.info("开始训练 LightGBM，训练样本: %d", len(train_feat))
        self.model.fit(dataset=dataset)
        self.booster = self.model.model
        self.feature_names = list(train_feat.columns)

    def predict(self, feat: pd.DataFrame) -> Tuple[pd.Series, np.ndarray]:
        if self.booster is None:
            raise RuntimeError("模型尚未训练")
        
        # 确保特征列与训练时一致
        if self.feature_names is None:
            # 如果没有保存特征名，尝试从 booster 获取
            try:
                self.feature_names = self.booster.feature_name()
            except:
                logger.warning("无法获取模型的特征名，使用输入特征列（可能导致特征不匹配）")
                self.feature_names = list(feat.columns)
        
        # 对齐特征列：确保顺序和数量与训练时一致
        aligned_feat = pd.DataFrame(index=feat.index, columns=self.feature_names, dtype=float)
        
        # 填充存在的特征
        for col in self.feature_names:
            if col in feat.columns:
                aligned_feat[col] = feat[col]
            else:
                # 缺失的特征用0填充（已归一化，0表示均值）
                aligned_feat[col] = 0.0
                logger.debug("特征 '%s' 在预测数据中不存在，使用0填充", col)
        
        # 检查是否有未使用的特征
        unused_cols = set(feat.columns) - set(self.feature_names)
        if unused_cols:
            logger.warning("预测数据中有 %d 个特征未在训练时使用，将被忽略: %s", 
                         len(unused_cols), list(unused_cols)[:10])
        
        # 确保列顺序与训练时一致
        aligned_feat = aligned_feat[self.feature_names]
        
        # 检查特征数量
        expected_num_features = len(self.feature_names)
        actual_num_features = len(aligned_feat.columns)
        if expected_num_features != actual_num_features:
            raise ValueError(
                f"特征数量不匹配：期望 {expected_num_features}，实际 {actual_num_features}。"
                f"期望特征: {self.feature_names[:10]}..."
            )
        
        values = aligned_feat.values
        preds = self.booster.predict(values)
        # pred_leaf=True 返回每棵树的叶子编号，用作二级模型输入
        leaf_index = self.booster.predict(values, pred_leaf=True)
        return pd.Series(preds, index=feat.index, name="lgb_pred"), leaf_index

    def save(self, output_dir: str, model_name: str):
        if self.booster is None:
            raise RuntimeError("无可保存模型")
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, f"{model_name}_lgb.txt")
        meta_path = os.path.join(output_dir, f"{model_name}_lgb_meta.json")
        self.booster.save_model(model_path)
        meta = {
            "config": self.config,
            "feature_names": self.feature_names,
            # 记录标签转换信息（如果可用）
            "label_transform": getattr(self, "_label_transform_info", None),
        }
        with open(meta_path, "w", encoding="utf-8") as fp:
            json.dump(meta, fp, ensure_ascii=False, indent=2)
        logger.info("LightGBM 模型已保存: %s", model_path)

    def load(self, output_dir: str, model_name: str):
        model_path = os.path.join(output_dir, f"{model_name}_lgb.txt")
        meta_path = os.path.join(output_dir, f"{model_name}_lgb_meta.json")
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        self.booster = lgb.Booster(model_file=model_path)
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as fp:
                meta = json.load(fp)
            self.feature_names = meta.get("feature_names")
        logger.info("LightGBM 模型已加载: %s", model_path)

    def get_feature_importance(self, importance_type: str = "gain") -> pd.Series:
        """
        获取特征重要性。
        
        Args:
            importance_type: 重要性类型，可选 'gain'（增益）、'split'（分裂次数）、'gain'（默认）
        
        Returns:
            pd.Series: 特征名称 -> 重要性值的映射
        """
        if self.booster is None:
            raise RuntimeError("模型尚未训练或加载")
        
        importance = self.booster.feature_importance(importance_type=importance_type)
        feature_names = self.feature_names or self.booster.feature_name()
        
        if len(importance) != len(feature_names):
            logger.warning(
                "特征重要性数量 (%d) 与特征名称数量 (%d) 不匹配，使用 booster 的特征名",
                len(importance), len(feature_names)
            )
            feature_names = self.booster.feature_name()
        
        return pd.Series(importance, index=feature_names, name=f"importance_{importance_type}")

