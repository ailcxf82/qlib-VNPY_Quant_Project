"""
Stack 模型：使用 LightGBM 叶子索引的 One-Hot 编码作为输入，训练二级 MLP。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import OneHotEncoder

from models.mlp_model import MLPRegressor
from utils import load_yaml_config

logger = logging.getLogger(__name__)


class LeafStackModel:
    """叶子编码 + MLP 的二级模型，支持 OneHot 或哈希压缩编码。"""

    def __init__(self, config_path: str):
        cfg = load_yaml_config(config_path)
        self.config = cfg.get("stack", cfg)
        self.alpha = self.config.get("alpha", 0.5)
        self.encoding = self.config.get("encoding", "embedding").lower()  # 默认使用 embedding
        self.hash_dim = self.config.get("hash_dim", 1024)
        self.encoder: Optional[OneHotEncoder] = None
        self.hasher: Optional[FeatureHasher] = None
        if self.encoding == "onehot":
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        elif self.encoding == "hashing":
            self.hasher = FeatureHasher(
                n_features=self.hash_dim,
                input_type="dict",
                dtype=np.float32,
            )
        elif self.encoding == "embedding":
            # 使用 EmbeddingBag，不需要编码器，直接传递叶子索引
            pass
        else:
            raise ValueError(f"不支持的叶子编码方式: {self.encoding}，支持: onehot, hashing, embedding")
        exclude_keys = {"alpha", "encoding", "hash_dim"}
        self.feature_names: Optional[list[str]] = None
        self.mlp = MLPRegressor({"model": {k: v for k, v in self.config.items() if k not in exclude_keys}})

    def _to_frame(self, encoded: np.ndarray, index: pd.Index, prefix: str = "leaf") -> pd.DataFrame:
        if self.feature_names is None or len(self.feature_names) != encoded.shape[1]:
            self.feature_names = [f"{prefix}_{i}" for i in range(encoded.shape[1])]
        return pd.DataFrame(encoded, index=index, columns=self.feature_names)

    def _hash_leaf(self, leaf: np.ndarray, index: pd.Index) -> pd.DataFrame:
        if self.hasher is None:
            raise RuntimeError("哈希编码器未初始化")
        rows = ({f"t{j}_{leaf_val}": 1 for j, leaf_val in enumerate(sample)} for sample in leaf)
        encoded_sparse = self.hasher.transform(rows)
        # 保持稀疏格式，不转换为稠密矩阵
        # 注意：这里仍然返回 DataFrame，但 MLP 需要特殊处理稀疏输入
        encoded = encoded_sparse.toarray()  # 暂时保留，后续 MLP 会改为直接处理稀疏矩阵
        return self._to_frame(encoded, index, prefix="hash")
    
    def _prepare_leaf_for_embedding(self, leaf: np.ndarray, index: pd.Index) -> np.ndarray:
        """为 EmbeddingBag 准备叶子索引数据（直接返回原始叶子索引）。"""
        # 直接返回叶子索引，不进行编码
        # leaf 形状: (n_samples, n_trees)
        return leaf

    def fit(
        self,
        train_leaf: np.ndarray,
        train_residual: pd.Series,
        valid_leaf: Optional[np.ndarray] = None,
        valid_residual: Optional[pd.Series] = None,
    ):
        logger.info("训练 Stack 模型，样本: %d，原始叶子维度: %d，编码方式: %s", 
                   train_leaf.shape[0], train_leaf.shape[1], self.encoding)
        if self.encoding == "embedding":
            # 直接传递叶子索引给 MLP，使用 EmbeddingBag 处理
            train_leaf_data = self._prepare_leaf_for_embedding(train_leaf, train_residual.index)
            valid_leaf_data = None
            if (
                valid_leaf is not None
                and valid_residual is not None
                and len(valid_leaf) > 0
                and len(valid_residual) > 0
            ):
                valid_leaf_data = self._prepare_leaf_for_embedding(valid_leaf, valid_residual.index)
            self.mlp.fit_with_leaf_index(
                train_leaf_data, train_residual, 
                valid_leaf_data, valid_residual,
                num_trees=train_leaf.shape[1]
            )
        elif self.encoding == "onehot":
            if self.encoder is None:
                raise RuntimeError("OneHotEncoder 未初始化")
            train_encoded = self.encoder.fit_transform(train_leaf)
            train_df = self._to_frame(train_encoded, train_residual.index, prefix="leaf")
            valid_df = None
            if (
                valid_leaf is not None
                and valid_residual is not None
                and len(valid_leaf) > 0
                and len(valid_residual) > 0
            ):
                valid_encoded = self.encoder.transform(valid_leaf)
                valid_df = self._to_frame(valid_encoded, valid_residual.index, prefix="leaf")
            self.mlp.fit(train_df, train_residual, valid_df, valid_residual)
        else:  # hashing
            train_df = self._hash_leaf(train_leaf, train_residual.index)
            valid_df = None
            if (
                valid_leaf is not None
                and valid_residual is not None
                and len(valid_leaf) > 0
                and len(valid_residual) > 0
            ):
                valid_df = self._hash_leaf(valid_leaf, valid_residual.index)
            self.mlp.fit(train_df, train_residual, valid_df, valid_residual)

    def predict_residual(self, leaf: np.ndarray, index: pd.Index) -> pd.Series:
        # 线上阶段只需 transform（不再 fit）
        if self.encoding == "embedding":
            leaf_data = self._prepare_leaf_for_embedding(leaf, index)
            return self.mlp.predict_with_leaf_index(leaf_data, index)
        elif self.encoding == "onehot":
            if self.encoder is None:
                raise RuntimeError("OneHotEncoder 未初始化")
            encoded = self.encoder.transform(leaf)
            feat_df = self._to_frame(encoded, index, prefix="leaf")
            return self.mlp.predict(feat_df)
        else:  # hashing
            feat_df = self._hash_leaf(leaf, index)
            return self.mlp.predict(feat_df)

    def fuse(self, lgb_pred: pd.Series, residual_pred: pd.Series) -> pd.Series:
        # 通过 residual 学习补充 LGB 的结构化短板
        return lgb_pred + residual_pred * self.alpha

    def save(self, output_dir: str, model_name: str):
        os.makedirs(output_dir, exist_ok=True)
        enc_path = os.path.join(output_dir, f"{model_name}_stack_encoder.json")
        with open(enc_path, "w", encoding="utf-8") as fp:
            meta = {
                "feature_names": self.feature_names,
                "alpha": self.alpha,
                "encoding": self.encoding,
                "hash_dim": self.hash_dim,
            }
            if self.encoding == "onehot" and self.encoder is not None:
                meta["categories"] = [cat.tolist() for cat in self.encoder.categories_]
            # embedding 模式不需要保存编码器信息
            json.dump(meta, fp, ensure_ascii=False, indent=2)
        self.mlp.save(output_dir, f"{model_name}_stack")
        logger.info("Stack 模型保存完成: %s", output_dir)

    def load(self, output_dir: str, model_name: str):
        enc_path = os.path.join(output_dir, f"{model_name}_stack_encoder.json")
        if not os.path.exists(enc_path):
            raise FileNotFoundError(enc_path)
        with open(enc_path, "r", encoding="utf-8") as fp:
            meta = json.load(fp)
        self.feature_names = meta.get("feature_names")
        self.alpha = meta.get("alpha", self.alpha)
        self.encoding = meta.get("encoding", self.encoding)
        self.hash_dim = meta.get("hash_dim", self.hash_dim)
        if self.encoding == "onehot":
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
            categories = meta.get("categories")
            if categories is None:
                raise ValueError("缺少 OneHot categories")
            self.encoder.categories_ = [np.array(cat) for cat in categories]
            self.encoder.n_features_in_ = len(self.encoder.categories_)
            # sklearn OneHotEncoder 在推理阶段需要补回 feature_names_in_ 等属性
            self.encoder.feature_names_in_ = np.array([f"tree_{i}" for i in range(self.encoder.n_features_in_)])
            self.encoder.drop_idx_ = None
            self.encoder.sparse_output_ = False
            self.hasher = None
        else:
            self.encoder = None
            self.hasher = FeatureHasher(
                n_features=self.hash_dim,
                input_type="dict",
                dtype=np.float32,
            )
        self.mlp.load(output_dir, f"{model_name}_stack")

