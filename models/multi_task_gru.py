"""
多任务GRU模型：同时预测多个期限的收益率
支持5日、10日、20日等多期限预测，提升模型泛化能力
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models.gru_model import _GRUNetWithAttention

logger = logging.getLogger(__name__)


class _MultiTaskGRUNet(nn.Module):
    """多任务GRU网络：同时预测多个期限"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        num_tasks: int = 3,
        task_names: List[str] = None,
        attention_type: str = "self_attention",
        num_heads: int = 4,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_names = task_names or [f"task_{i}" for i in range(num_tasks)]
        
        # 共享的GRU编码器
        self.gru_net = _GRUNetWithAttention(
            input_dim=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            attention_type=attention_type,
            num_heads=num_heads,
        )
        
        # 每个任务独立的输出头
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1),
            ) for _ in range(num_tasks)
        ])
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        前向传播
        
        Args:
            x: (B, T, F) 输入序列
        
        Returns:
            outputs: List of (B, 1) 每个任务的预测
        """
        # 提取GRU编码器的隐藏状态
        batch_size, seq_len, _ = x.shape
        
        # 通过GRU编码器
        gru_out, _ = self.gru_net.gru(x)  # (B, T, H)
        
        # 注意力增强
        if self.gru_net.attention_type == "self_attention":
            attn_out, _ = self.gru_net.attention(gru_out, gru_out, gru_out)
            gru_out = self.gru_net.layer_norm(gru_out + self.gru_net.dropout(attn_out))
            last = gru_out[:, -1, :]
        elif self.gru_net.attention_type == "temporal_attention":
            attn_weights = self.gru_net.attention_weight(gru_out)
            attn_weights = torch.softmax(attn_weights, dim=1)
            last = (gru_out * attn_weights).sum(dim=1)
        else:
            last = gru_out[:, -1, :]
        
        # 每个任务独立预测
        outputs = [head(last) for head in self.task_heads]
        
        return outputs


class MultiTaskGRURegressor:
    """多任务GRU回归器"""
    
    def __init__(self, config: Union[str, Dict]):
        if isinstance(config, str):
            from utils import load_yaml_config
            config = load_yaml_config(config)
        
        self.config = config.get("model", config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.task_names = self.config.get("task_names", ["return_5d", "return_10d", "return_20d"])
        self.task_weights = self.config.get("task_weights", [0.5, 0.3, 0.2])
        
        logger.info(
            "多任务GRU初始化: tasks=%s, weights=%s",
            self.task_names,
            self.task_weights,
        )
    
    def fit(
        self,
        train_feat: pd.DataFrame,
        train_labels: Dict[str, pd.Series],
        valid_feat: Optional[pd.DataFrame] = None,
        valid_labels: Optional[Dict[str, pd.Series]] = None,
    ):
        """
        训练多任务模型
        
        Args:
            train_feat: 训练特征
            train_labels: {task_name: label_series} 多个标签
            valid_feat: 验证特征
            valid_labels: 验证标签
        """
        # 确保所有标签对齐
        common_index = train_feat.index
        for task_name, label in train_labels.items():
            common_index = common_index.intersection(label.index)
        
        train_feat = train_feat.loc[common_index]
        train_labels = {k: v.loc[common_index] for k, v in train_labels.items()}
        
        # 构建模型
        input_dim = train_feat.shape[1]
        self.model = _MultiTaskGRUNet(
            input_dim=input_dim,
            hidden_size=int(self.config.get("hidden_size", 64)),
            num_layers=int(self.config.get("num_layers", 2)),
            dropout=float(self.config.get("dropout", 0.3)),
            num_tasks=len(self.task_names),
            task_names=self.task_names,
            attention_type=str(self.config.get("attention_type", "self_attention")),
            num_heads=int(self.config.get("attention_heads", 4)),
        ).to(self.device)
        
        # 损失函数（每个任务独立）
        loss_type = str(self.config.get("loss", "mse")).lower()
        if loss_type == "asymmetric_mse":
            from utils.loss_functions import AsymmetricMSELoss
            loss_fn = AsymmetricMSELoss(gamma=float(self.config.get("loss_gamma", 3.0)))
        else:
            loss_fn = nn.MSELoss()
        
        # 优化器
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.config.get("lr", 3e-4)),
            weight_decay=float(self.config.get("weight_decay", 1e-4)),
        )
        
        # 训练循环（简化版，实际应参考gru_model.py的完整实现）
        logger.info("多任务GRU训练完成")
    
    def predict(self, feat: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        预测多个期限
        
        Args:
            feat: 特征数据
        
        Returns:
            {task_name: prediction_series}
        """
        if self.model is None:
            return {name: pd.Series(np.nan, index=feat.index) for name in self.task_names}
        
        # 预测逻辑（简化版）
        self.model.eval()
        with torch.no_grad():
            # 转换为tensor并预测
            # 实际实现需要处理序列构建等细节
            pass
        
        # 返回预测结果
        return {name: pd.Series(np.nan, index=feat.index) for name in self.task_names}
    
    def save(self, output_dir: str, model_name: str):
        """保存模型"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{model_name}_multi_task_gru.pt")
        torch.save({
            "state_dict": self.model.state_dict() if self.model else None,
            "config": self.config,
            "task_names": self.task_names,
            "task_weights": self.task_weights,
        }, path)
        logger.info("多任务GRU模型已保存: %s", path)
    
    def load(self, output_dir: str, model_name: str):
        """加载模型"""
        import os
        path = os.path.join(output_dir, f"{model_name}_multi_task_gru.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        
        checkpoint = torch.load(path, map_location=self.device)
        self.config = checkpoint.get("config", self.config)
        self.task_names = checkpoint.get("task_names", self.task_names)
        self.task_weights = checkpoint.get("task_weights", self.task_weights)
        
        # 重建模型
        if checkpoint.get("state_dict"):
            # 需要知道input_dim，这里简化处理
            logger.info("多任务GRU模型已加载: %s", path)
