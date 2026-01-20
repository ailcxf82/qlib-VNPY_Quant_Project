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
        # 记录并打印训练时使用的特征列（便于排查“按模型分配特征集合”是否生效）
        feat_cols = list(train_feat.columns)
        logger.info("LGB 训练特征列数=%d，示例=%s", len(feat_cols), feat_cols[:30])
        # 列数不大时，直接打印完整列表；避免过长时刷屏
        if len(feat_cols) <= 60:
            logger.info("LGB 训练特征完整列表=%s", feat_cols)

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
        self.feature_names = feat_cols

    def predict(self, feat: pd.DataFrame) -> Tuple[pd.Series, np.ndarray]:
        if self.booster is None:
            raise RuntimeError("模型尚未训练")

        # 空输入短路：LightGBM 在 nrow==0 时会在内部触发 ZeroDivisionError
        if feat is None or len(feat) == 0:
            try:
                n_trees = int(self.booster.num_trees())
            except Exception:
                n_trees = 0
            empty_pred = pd.Series([], index=getattr(feat, "index", None), dtype=float, name="lgb_pred")
            empty_leaf = np.empty((0, n_trees), dtype=np.int32)
            return empty_pred, empty_leaf
        
        # 确保特征列与训练时一致
        if self.feature_names is None:
            # 如果没有保存特征名，尝试从 booster 获取
            try:
                self.feature_names = self.booster.feature_name()
            except:
                logger.warning("无法获取模型的特征名，使用输入特征列（可能导致特征不匹配）")
                self.feature_names = list(feat.columns)
        
        # 强制按训练时的 feature_names 对齐：多余特征忽略，缺失特征补 0
        missing_cols = [c for c in self.feature_names if c not in feat.columns]
        if missing_cols:
            logger.warning(
                "预测数据缺失 %d 个训练特征，将用 0 填充（示例: %s）",
                len(missing_cols),
                missing_cols[:10],
            )
        unused_cols = [c for c in feat.columns if c not in set(self.feature_names)]
        if unused_cols:
            logger.warning(
                "预测数据包含 %d 个未参与训练的特征，将被忽略（示例: %s）",
                len(unused_cols),
                unused_cols[:10],
            )
        aligned_feat = feat.reindex(columns=self.feature_names).fillna(0.0)
        
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

