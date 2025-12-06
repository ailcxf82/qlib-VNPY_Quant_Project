"""
PyTorch MLP 回归模型，实现基础训练、验证与持久化。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils import load_yaml_config

logger = logging.getLogger(__name__)


def _build_mlp(input_dim: int, hidden_dims: List[int], dropout: float, activation: str) -> nn.Module:
    """根据配置构造 MLP。"""
    act_cls = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
    }.get(activation.lower(), nn.ReLU)
    layers = []
    prev_dim = input_dim
    for dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, dim))
        layers.append(act_cls())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = dim
    layers.append(nn.Linear(prev_dim, 1))
    return nn.Sequential(*layers)


def _build_mlp_with_embedding(
    num_trees: int,
    max_leaf_per_tree: int,
    embedding_dim: int,
    hidden_dims: List[int],
    dropout: float,
    activation: str,
) -> nn.Module:
    """
    使用 Embedding 构造 MLP，用于处理稀疏的叶子索引输入。
    
    参数:
        num_trees: 树的数量
        max_leaf_per_tree: 每棵树的最大叶子数（用于确定 embedding 的 num_embeddings）
        embedding_dim: embedding 维度
        hidden_dims: 隐藏层维度列表
        dropout: dropout 率
        activation: 激活函数名称
    """
    act_cls = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
    }.get(activation.lower(), nn.ReLU)
    
    class MLPWithEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            # 为每棵树创建一个 Embedding 层
            # 使用一个大的 Embedding 层，通过偏移量区分不同树的叶子索引
            # 或者为每棵树创建独立的 Embedding（更灵活）
            self.embeddings = nn.ModuleList([
                nn.Embedding(
                    num_embeddings=max_leaf_per_tree,
                    embedding_dim=embedding_dim,
                    sparse=False  # 设置为 False 以获得更好的性能
                )
                for _ in range(num_trees)
            ])
            
            # 拼接所有树的 embedding
            # 输入维度 = num_trees * embedding_dim
            input_dim = num_trees * embedding_dim
            
            # 构建后续的 MLP 层
            layers = []
            prev_dim = input_dim
            for dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, dim))
                layers.append(act_cls())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                prev_dim = dim
            layers.append(nn.Linear(prev_dim, 1))
            self.mlp = nn.Sequential(*layers)
        
        def forward(self, leaf_indices: torch.Tensor):
            """
            前向传播。
            
            参数:
                leaf_indices: 形状为 (batch_size, num_trees) 的整数张量，包含每棵树的叶子索引
            """
            # 为每棵树提取 embedding
            embeddings = []
            for i, emb in enumerate(self.embeddings):
                # 提取第 i 棵树的叶子索引
                tree_leaves = leaf_indices[:, i]  # (batch_size,)
                # 使用 Embedding 获取 embedding 向量
                emb_out = emb(tree_leaves)  # (batch_size, embedding_dim)
                embeddings.append(emb_out)
            
            # 拼接所有树的 embedding
            combined = torch.cat(embeddings, dim=1)  # (batch_size, num_trees * embedding_dim)
            
            # 通过 MLP
            output = self.mlp(combined)  # (batch_size, 1)
            return output
    
    return MLPWithEmbedding()


class MLPRegressor:
    """MLP 模型封装。"""

    def __init__(self, config: Union[str, Dict[str, Any]]):
        if isinstance(config, str):
            raw = load_yaml_config(config)
        else:
            raw = config
        self.config = raw.get("model", raw)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None
        self._input_dim: Optional[int] = None
        self._use_embedding: bool = False
        self._num_trees: Optional[int] = None
        self._max_leaf_per_tree: Optional[int] = None

    def _tensorize(self, feat: pd.DataFrame, label: Optional[pd.Series] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # 直接将 pandas 数据转换为 tensor，便于组建 TensorDataset
        x = torch.tensor(feat.values, dtype=torch.float32)
        y = torch.tensor(label.values, dtype=torch.float32).unsqueeze(-1) if label is not None else None
        return x, y

    def fit(
        self,
        train_feat: pd.DataFrame,
        train_label: pd.Series,
        valid_feat: Optional[pd.DataFrame] = None,
        valid_label: Optional[pd.Series] = None,
    ):
        input_dim = train_feat.shape[1]
        self._input_dim = train_feat.shape[1]
        self.model = _build_mlp(
            input_dim=self._input_dim,
            hidden_dims=self.config.get("hidden_dims", [128, 64]),
            dropout=self.config.get("dropout", 0.1),
            activation=self.config.get("activation", "relu"),
        ).to(self.device)

        # 支持自定义损失函数
        loss_type = self.config.get("loss", "mse")
        loss_params = self.config.get("loss_params", {})
        
        if loss_type == "asymmetric_mse":
            from utils.loss_functions import AsymmetricMSELoss
            gamma = loss_params.get("gamma", 2.0)
            criterion = AsymmetricMSELoss(gamma=gamma)
            logger.info("使用非对称 MSE 损失函数，gamma=%.2f", gamma)
        elif loss_type == "weighted_mse":
            from utils.loss_functions import WeightedMSELoss
            w_positive = loss_params.get("w_positive", 2.0)
            w_negative = loss_params.get("w_negative", 0.5)
            criterion = WeightedMSELoss(w_positive=w_positive, w_negative=w_negative)
            logger.info("使用加权 MSE 损失函数，正向权重=%.2f，负向权重=%.2f", w_positive, w_negative)
        else:
            criterion = nn.MSELoss()
            logger.info("使用标准 MSE 损失函数")
        # optimizer = torch.optim.Adam(
        #     self.model.parameters(),
        #     lr=self.config.get("lr", 1e-3),
        #     weight_decay=self.config.get("weight_decay", 0.0),
        # )
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            # 强制将 lr 转换为 float 类型
            lr=float(self.config.get("lr", 1e-3)),
            # 强制将 weight_decay 转换为 float 类型
            weight_decay=float(self.config.get("weight_decay", 0.0)),
        )

        batch_size = self.config.get("batch_size", 1024)
        max_epochs = self.config.get("max_epochs", 20)
        patience = self.config.get("patience", 5)
        # 提前停止监控
        best_loss = float("inf")
        wait = 0
        best_state = None

        train_loader = DataLoader(
            TensorDataset(*self._tensorize(train_feat, train_label)),
            batch_size=batch_size,
            shuffle=True,
        )
        valid_loader = None
        if valid_feat is not None and valid_label is not None and len(valid_feat) > 0 and len(valid_label) > 0:
            valid_loader = DataLoader(
                TensorDataset(*self._tensorize(valid_feat, valid_label)),
                batch_size=batch_size,
                shuffle=False,
            )
        else:
            valid_loader = None

        for epoch in range(max_epochs):
            self.model.train()
            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch_x)
            epoch_loss /= len(train_loader.dataset)

            val_loss = epoch_loss
            if valid_loader is not None:
                self.model.eval()
                total = 0.0
                with torch.no_grad():
                    for vx, vy in valid_loader:
                        vx = vx.to(self.device)
                        vy = vy.to(self.device)
                        pred = self.model(vx)
                        total += criterion(pred, vy).item() * len(vx)
                val_loss = total / len(valid_loader.dataset)

            logger.info("MLP epoch %d train_loss=%.6f valid_loss=%.6f", epoch, epoch_loss, val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                wait = 0
                best_state = self.model.state_dict()
            else:
                wait += 1
                if wait >= patience:
                    logger.info("早停触发，最佳验证损失 %.6f", best_loss)
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict(self, feat: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("MLP 模型尚未训练")
        self.model.eval()
        with torch.no_grad():
            preds = self.model(torch.tensor(feat.values, dtype=torch.float32).to(self.device)).cpu().numpy().flatten()
        return pd.Series(preds, index=feat.index, name="mlp_pred")
    
    def fit_with_leaf_index(
        self,
        train_leaf: np.ndarray,
        train_label: pd.Series,
        valid_leaf: Optional[np.ndarray] = None,
        valid_label: Optional[pd.Series] = None,
        num_trees: Optional[int] = None,
    ):
        """
        使用叶子索引训练 MLP，使用 Embedding 处理稀疏输入。
        
        参数:
            train_leaf: 训练集叶子索引，形状为 (n_samples, num_trees)
            train_label: 训练集标签
            valid_leaf: 验证集叶子索引（可选）
            valid_label: 验证集标签（可选）
            num_trees: 树的数量（如果为 None，从 train_leaf 推断）
        """
        if num_trees is None:
            num_trees = train_leaf.shape[1]
        self._num_trees = num_trees
        
        # 计算每棵树的最大叶子数
        max_leaf_per_tree = int(train_leaf.max() + 1)  # +1 因为索引从 0 开始
        if valid_leaf is not None and len(valid_leaf) > 0:
            max_leaf_per_tree = max(max_leaf_per_tree, int(valid_leaf.max() + 1))
        self._max_leaf_per_tree = max_leaf_per_tree
        
        embedding_dim = self.config.get("embedding_dim", 32)
        self._use_embedding = True
        
        self.model = _build_mlp_with_embedding(
            num_trees=num_trees,
            max_leaf_per_tree=max_leaf_per_tree,
            embedding_dim=embedding_dim,
            hidden_dims=self.config.get("hidden_dims", [128, 64]),
            dropout=self.config.get("dropout", 0.1),
            activation=self.config.get("activation", "relu"),
        ).to(self.device)
        
        # 支持自定义损失函数
        loss_type = self.config.get("loss", "mse")
        loss_params = self.config.get("loss_params", {})
        
        if loss_type == "asymmetric_mse":
            from utils.loss_functions import AsymmetricMSELoss
            gamma = loss_params.get("gamma", 2.0)
            criterion = AsymmetricMSELoss(gamma=gamma)
            logger.info("使用非对称 MSE 损失函数，gamma=%.2f", gamma)
        elif loss_type == "weighted_mse":
            from utils.loss_functions import WeightedMSELoss
            w_positive = loss_params.get("w_positive", 2.0)
            w_negative = loss_params.get("w_negative", 0.5)
            criterion = WeightedMSELoss(w_positive=w_positive, w_negative=w_negative)
            logger.info("使用加权 MSE 损失函数，正向权重=%.2f，负向权重=%.2f", w_positive, w_negative)
        else:
            criterion = nn.MSELoss()
            logger.info("使用标准 MSE 损失函数")
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.config.get("lr", 1e-3)),
            weight_decay=float(self.config.get("weight_decay", 0.0)),
        )
        
        batch_size = self.config.get("batch_size", 1024)
        max_epochs = self.config.get("max_epochs", 20)
        patience = self.config.get("patience", 5)
        best_loss = float("inf")
        wait = 0
        best_state = None
        
        # 准备数据
        train_leaf_tensor = torch.tensor(train_leaf, dtype=torch.long)
        train_label_tensor = torch.tensor(train_label.values, dtype=torch.float32).unsqueeze(-1)
        
        train_loader = DataLoader(
            TensorDataset(train_leaf_tensor, train_label_tensor),
            batch_size=batch_size,
            shuffle=True,
        )
        
        valid_loader = None
        if valid_leaf is not None and valid_label is not None and len(valid_leaf) > 0 and len(valid_label) > 0:
            valid_leaf_tensor = torch.tensor(valid_leaf, dtype=torch.long)
            valid_label_tensor = torch.tensor(valid_label.values, dtype=torch.float32).unsqueeze(-1)
            valid_loader = DataLoader(
                TensorDataset(valid_leaf_tensor, valid_label_tensor),
                batch_size=batch_size,
                shuffle=False,
            )
        
        for epoch in range(max_epochs):
            self.model.train()
            epoch_loss = 0.0
            for batch_leaf, batch_label in train_loader:
                batch_leaf = batch_leaf.to(self.device)
                batch_label = batch_label.to(self.device)
                optimizer.zero_grad()
                pred = self.model(batch_leaf)
                loss = criterion(pred, batch_label)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch_leaf)
            epoch_loss /= len(train_loader.dataset)
            
            val_loss = epoch_loss
            if valid_loader is not None:
                self.model.eval()
                total = 0.0
                with torch.no_grad():
                    for v_leaf, v_label in valid_loader:
                        v_leaf = v_leaf.to(self.device)
                        v_label = v_label.to(self.device)
                        pred = self.model(v_leaf)
                        total += criterion(pred, v_label).item() * len(v_leaf)
                val_loss = total / len(valid_loader.dataset)
            
            logger.info("MLP (Embedding) epoch %d train_loss=%.6f valid_loss=%.6f", epoch, epoch_loss, val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                wait = 0
                best_state = self.model.state_dict()
            else:
                wait += 1
                if wait >= patience:
                    logger.info("早停触发，最佳验证损失 %.6f", best_loss)
                    break
        
        if best_state is not None:
            self.model.load_state_dict(best_state)
    
    def predict_with_leaf_index(self, leaf: np.ndarray, index: pd.Index) -> pd.Series:
        """使用叶子索引进行预测。"""
        if self.model is None:
            raise RuntimeError("MLP 模型尚未训练")
        self.model.eval()
        with torch.no_grad():
            leaf_tensor = torch.tensor(leaf, dtype=torch.long).to(self.device)
            preds = self.model(leaf_tensor).cpu().numpy().flatten()
        return pd.Series(preds, index=index, name="mlp_pred")

    def save(self, output_dir: str, model_name: str):
        if self.model is None:
            raise RuntimeError("无可保存模型")
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{model_name}_mlp.pt")
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": self.config,
            },
            path,
        )
        meta = {
            "input_dim": self._input_dim,
            "use_embedding": self._use_embedding,
            "num_trees": self._num_trees,
            "max_leaf_per_tree": self._max_leaf_per_tree,
        }
        with open(os.path.join(output_dir, f"{model_name}_mlp_meta.json"), "w", encoding="utf-8") as fp:
            json.dump(meta, fp, ensure_ascii=False, indent=2)
        logger.info("MLP 模型已保存: %s", path)

    def load(self, output_dir: str, model_name: str, input_dim: Optional[int] = None):
        path = os.path.join(output_dir, f"{model_name}_mlp.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        meta_path = os.path.join(output_dir, f"{model_name}_mlp_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(meta_path)
        with open(meta_path, "r", encoding="utf-8") as fp:
            meta = json.load(fp)
        
        self._use_embedding = meta.get("use_embedding", False)
        
        if self._use_embedding:
            # 加载 embedding 模式的模型
            self._num_trees = meta.get("num_trees")
            self._max_leaf_per_tree = meta.get("max_leaf_per_tree")
            embedding_dim = self.config.get("embedding_dim", 32)
            if self._num_trees is None or self._max_leaf_per_tree is None:
                raise ValueError("加载 embedding 模型需要 num_trees 和 max_leaf_per_tree")
            self.model = _build_mlp_with_embedding(
                num_trees=self._num_trees,
                max_leaf_per_tree=self._max_leaf_per_tree,
                embedding_dim=embedding_dim,
                hidden_dims=self.config.get("hidden_dims", [128, 64]),
                dropout=self.config.get("dropout", 0.1),
                activation=self.config.get("activation", "relu"),
            ).to(self.device)
        else:
            # 加载普通 MLP 模型
            if input_dim is None:
                input_dim = meta.get("input_dim")
                if input_dim is None:
                    raise ValueError("加载普通 MLP 模型需要 input_dim")
            self._input_dim = input_dim
            self.model = _build_mlp(
                input_dim=input_dim,
                hidden_dims=self.config.get("hidden_dims", [128, 64]),
                dropout=self.config.get("dropout", 0.1),
                activation=self.config.get("activation", "relu"),
            ).to(self.device)
        
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        logger.info("MLP 模型已加载: %s", path)

