"""
行业轮动预测专用 GRU 模型，带 Feature Attention 机制。

模型结构：
1. Feature Attention 层：自动学习特征重要性权重
2. GRU 层：处理时序信息
3. 全连接层：输出预测分数

支持排序损失函数（Ranking Loss）和标准 MSE 损失。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class FeatureAttention(nn.Module):
    """特征注意力层：为每个特征学习权重（带残差连接）。"""
    
    def __init__(self, num_features: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.attention_net = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_features),
            nn.Sigmoid()  # 输出 0-1 之间的权重
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播（门控 + 残差连接）。
        
        使用时间聚合（时间均值）生成权重，降低末端噪声影响。
        
        参数:
            x: 输入张量，形状为 (batch_size, sequence_length, num_features)
        
        返回:
            加权后的特征，形状与输入相同
        """
        # 使用时间聚合（时间均值）生成权重，而不是只用最后一步
        # 这样可以降低末端噪声的影响，使权重更稳定
        pooled = x.mean(dim=1)  # (batch_size, num_features) 时间平均
        
        # 生成注意力权重
        attention_weights = self.attention_net(pooled)  # (batch_size, num_features)
        
        # 将权重应用到所有时间步
        attention_weights = attention_weights.unsqueeze(1)  # (batch_size, 1, num_features)
        x_weighted = x * attention_weights
        
        # 残差连接：x * w + x，避免"把信息乘没了"
        return x_weighted + x


class IndustryGRU(nn.Module):
    """
    行业轮动预测 GRU 模型。
    
    模型结构：
    - Input: (batch_size, sequence_length=60, num_features)
    - Feature Attention: 学习特征权重
    - GRU: 单层 GRU，hidden_size=64
    - Output: 标量分数
    """
    
    def __init__(
        self,
        num_features: int,
        sequence_length: int = 60,
        hidden_size: int = 64,
        num_layers: int = 1,
        attention_hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_features = num_features
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Feature Attention 层
        self.feature_attention = FeatureAttention(num_features, attention_hidden_dim, dropout)
        
        # GRU 层
        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        参数:
            x: 输入张量，形状为 (batch_size, sequence_length, num_features)
        
        返回:
            预测分数，形状为 (batch_size, 1)
        """
        # Feature Attention
        x = self.feature_attention(x)
        
        # GRU
        gru_out, _ = self.gru(x)  # gru_out: (batch_size, sequence_length, hidden_size)
        
        # 取最后一个时间步的输出
        last_hidden = gru_out[:, -1, :]  # (batch_size, hidden_size)
        
        # 全连接层
        output = self.fc(last_hidden)  # (batch_size, 1)
        
        return output


class RankingLoss(nn.Module):
    """
    Pointwise Ranking Loss。
    
    对于排序任务，我们希望预测值能够正确反映标签的排序关系。
    这个损失函数结合了 MSE 和排序损失。
    """
    
    def __init__(self, alpha: float = 0.5):
        """
        参数:
            alpha: 排序损失的权重（0-1之间），alpha=0 时退化为纯 MSE
        """
        super().__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        计算损失。
        
        参数:
            pred: 预测值，形状为 (batch_size, 1)
            label: 标签，形状为 (batch_size, 1)
        
        返回:
            损失值
        """
        # MSE 损失
        mse = self.mse_loss(pred, label)
        
        if self.alpha > 0:
            # 排序损失：鼓励预测值的排序与标签的排序一致
            # 使用 pairwise ranking loss
            pred_flat = pred.squeeze(-1)  # (batch_size,)
            label_flat = label.squeeze(-1)  # (batch_size,)
            
            # 计算所有样本对的排序损失
            # 对于标签 label[i] > label[j]，我们希望 pred[i] > pred[j]
            n = pred_flat.size(0)
            if n > 1:
                # 生成所有样本对
                pred_i = pred_flat.unsqueeze(1).expand(n, n)  # (n, n)
                pred_j = pred_flat.unsqueeze(0).expand(n, n)  # (n, n)
                label_i = label_flat.unsqueeze(1).expand(n, n)  # (n, n)
                label_j = label_flat.unsqueeze(0).expand(n, n)  # (n, n)
                
                # 只考虑 label_i > label_j 的样本对
                mask = (label_i > label_j).float()
                
                # 排序损失：如果 label_i > label_j，则希望 pred_i > pred_j
                # 使用 hinge loss: max(0, margin - (pred_i - pred_j))
                margin = 0.1
                ranking_loss = F.relu(margin - (pred_i - pred_j)) * mask
                
                # 平均排序损失
                ranking_loss = ranking_loss.sum() / (mask.sum() + 1e-8)
            else:
                ranking_loss = torch.tensor(0.0, device=pred.device)
            
            total_loss = (1 - self.alpha) * mse + self.alpha * ranking_loss
        else:
            total_loss = mse
        
        return total_loss


class IndustryGRUWrapper:
    """
    IndustryGRU 模型的封装类，提供与 Qlib 工作流兼容的接口。
    支持集成学习（Ensemble）模式，通过训练多个模型并取平均预测来降低方差。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化模型。
        
        参数:
            config: 模型配置字典，包含：
                - num_features: 特征数量
                - sequence_length: 时序长度（默认60）
                - hidden_size: GRU 隐藏层大小（默认64）
                - num_layers: GRU 层数（默认1）
                - attention_hidden_dim: Attention 隐藏层维度（默认64）
                - dropout: Dropout 率（默认0.1）
                - loss: 损失函数类型，'mse' 或 'ranking'（默认'mse'）
                - ranking_alpha: Ranking Loss 的权重（默认0.5）
                - batch_size: 批处理大小（默认32）
                - lr: 学习率（默认0.001）
                - weight_decay: 权重衰减（默认1e-4）
                - max_epochs: 最大训练轮数（默认50）
                - patience: 早停耐心值（默认10）
                - num_ensemble: 集成模型数量（默认5，如果 > 1 则启用集成模式）
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 集成学习配置
        self.num_ensemble = config.get("num_ensemble", 5)
        self.is_ensemble = self.num_ensemble > 1
        
        # 模型存储：如果是集成模式，使用列表；否则使用单个模型
        if self.is_ensemble:
            self.models: Optional[list] = None  # 存储多个 IndustryGRU 实例
            self.model = None  # 保持兼容性，但实际不使用
        else:
            self.model: Optional[IndustryGRU] = None
            self.models = None
        
        self._num_features: Optional[int] = None
        self._sequence_length: Optional[int] = None
        
        # 模型超参数
        self.sequence_length = config.get("sequence_length", 60)
        self.hidden_size = config.get("hidden_size", 64)
        self.num_layers = config.get("num_layers", 1)
        self.attention_hidden_dim = config.get("attention_hidden_dim", 64)
        self.dropout = config.get("dropout", 0.1)
        
        # 训练超参数
        self.batch_size = config.get("batch_size", 32)
        self.lr = float(config.get("lr", 0.001))
        self.weight_decay = float(config.get("weight_decay", 1e-4))
        self.max_epochs = config.get("max_epochs", 50)
        self.patience = config.get("patience", 10)
        
        # 梯度裁剪配置
        self.grad_clip = config.get("grad_clip", None)  # None 表示不裁剪，数值表示最大范数
        
        # 学习率调度器配置
        self.lr_scheduler_type = config.get("lr_scheduler", None)  # None, "plateau", "step", "cosine" 等
        
        # 损失函数配置
        self.loss_type = config.get("loss", "mse")
        self.ranking_alpha = config.get("ranking_alpha", 0.5)
        
        if self.is_ensemble:
            logger.info("启用集成学习模式，集成模型数量: %d", self.num_ensemble)
    
    def _prepare_sequences(
        self, 
        feat: pd.DataFrame, 
        label: Optional[pd.Series] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        将特征数据转换为时序序列。
        
        参数:
            feat: 特征 DataFrame，索引为 MultiIndex (datetime, instrument)
            label: 标签 Series（可选），索引为 MultiIndex (datetime, instrument)
        
        返回:
            x: 时序特征张量，形状为 (n_samples, sequence_length, num_features)
            y: 标签张量，形状为 (n_samples, 1) 或 None
        """
        # 按 instrument 分组，为每个 instrument 构建时序序列
        sequences = []
        labels = []
        sample_indices = []  # 保存样本的索引，用于后续构建预测结果的索引
        
        if isinstance(feat.index, pd.MultiIndex):
            # MultiIndex: (datetime, instrument)
            # 确保索引名称正确
            if feat.index.nlevels == 2:
                # 获取所有 instrument
                instruments = feat.index.get_level_values(1).unique()
                
                for instrument in instruments:
                    try:
                        inst_data = feat.xs(instrument, level=1, drop_level=False)
                        # 如果 xs 后仍然是 MultiIndex，需要进一步处理
                        if isinstance(inst_data.index, pd.MultiIndex):
                            # 只保留 datetime 级别
                            inst_data = inst_data.reset_index(level=1, drop=True)
                        inst_data = inst_data.sort_index()
                        
                        # 构建滑动窗口序列
                        for i in range(self.sequence_length, len(inst_data)):
                            seq = inst_data.iloc[i - self.sequence_length:i].values
                            sequences.append(seq)
                            
                            # 保存样本索引
                            datetime_idx = inst_data.index[i]
                            sample_indices.append((datetime_idx, instrument))
                            
                            if label is not None:
                                # 获取对应的标签（使用 MultiIndex 定位）
                                try:
                                    if isinstance(label.index, pd.MultiIndex):
                                        label_val = label.loc[(datetime_idx, instrument)]
                                    else:
                                        # 如果 label 是单层索引，尝试直接使用 datetime
                                        label_val = label.loc[datetime_idx]
                                    labels.append(label_val)
                                except (KeyError, IndexError):
                                    labels.append(np.nan)
                    except KeyError:
                        # 某些 instrument 可能没有足够的数据
                        logger.debug("Instrument %s 数据不足，跳过", instrument)
                        continue
            else:
                raise ValueError(f"不支持的 MultiIndex 层数: {feat.index.nlevels}")
        else:
            # 单层索引：假设是 datetime（适用于单个 instrument 的情况）
            inst_data = feat.sort_index()
            for i in range(self.sequence_length, len(inst_data)):
                seq = inst_data.iloc[i - self.sequence_length:i].values
                sequences.append(seq)
                
                datetime_idx = inst_data.index[i]
                sample_indices.append(datetime_idx)
                
                if label is not None:
                    try:
                        label_val = label.loc[datetime_idx]
                        labels.append(label_val)
                    except (KeyError, IndexError):
                        labels.append(np.nan)
        
        if len(sequences) == 0:
            raise ValueError("无法构建时序序列，请检查数据格式和 sequence_length 设置")
        
        x = torch.tensor(np.array(sequences), dtype=torch.float32)
        
        # 检查 x 中是否有 NaN 或 inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.warning("特征数据中包含 NaN 或 inf，进行清理")
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 保存样本索引，用于预测时构建结果索引
        self._last_sample_indices = sample_indices
        
        if label is not None and len(labels) > 0:
            y = torch.tensor(np.array(labels), dtype=torch.float32).unsqueeze(-1)
            # 处理 NaN 和 inf
            y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            y = None
        
        return x, y
    
    def _prepare_sequences_with_history(
        self,
        combined_feat: pd.DataFrame,
        combined_label: pd.Series,
        valid_feat: pd.DataFrame,
        valid_label: pd.Series,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        使用训练集历史数据补充验证集，构建完整的时序序列。
        
        参数:
            combined_feat: 合并后的特征（训练集+验证集）
            combined_label: 合并后的标签（训练集+验证集）
            valid_feat: 验证集特征（用于确定需要预测的时间点）
            valid_label: 验证集标签
        
        返回:
            x: 时序特征张量
            y: 标签张量
        """
        sequences = []
        labels = []
        sample_indices = []
        
        if not isinstance(combined_feat.index, pd.MultiIndex):
            raise ValueError("特征索引应为 MultiIndex")
        
        # 获取验证集的时间范围
        valid_datetimes = valid_feat.index.get_level_values("datetime").unique()
        valid_instruments = valid_feat.index.get_level_values("instrument").unique()
        
        for instrument in valid_instruments:
            try:
                # 获取该 instrument 的完整数据（包括训练集和验证集）
                inst_data = combined_feat.xs(instrument, level=1, drop_level=False)
                if isinstance(inst_data.index, pd.MultiIndex):
                    inst_data = inst_data.reset_index(level=1, drop=True)
                inst_data = inst_data.sort_index()
                
                # 只对验证集的时间点构建序列
                for datetime_idx in valid_datetimes:
                    try:
                        # 找到该时间点在数据中的位置
                        if datetime_idx not in inst_data.index:
                            continue
                        
                        pos = inst_data.index.get_loc(datetime_idx)
                        
                        # 检查是否有足够的历史数据（需要 sequence_length 天的历史）
                        if pos < self.sequence_length:
                            # 数据不足，跳过
                            continue
                        
                        # 构建序列（使用历史数据）
                        seq = inst_data.iloc[pos - self.sequence_length:pos].values
                        sequences.append(seq)
                        sample_indices.append((datetime_idx, instrument))
                        
                        # 获取对应的标签
                        if isinstance(valid_label.index, pd.MultiIndex):
                            try:
                                label_val = valid_label.loc[(datetime_idx, instrument)]
                                labels.append(label_val)
                            except (KeyError, IndexError):
                                labels.append(np.nan)
                        else:
                            labels.append(np.nan)
                    except (KeyError, IndexError):
                        continue
            except KeyError:
                logger.debug("Instrument %s 数据不足，跳过", instrument)
                continue
        
        if len(sequences) == 0:
            raise ValueError("无法构建时序序列，即使使用训练集历史数据补充")
        
        x = torch.tensor(np.array(sequences), dtype=torch.float32)
        
        # 检查 x 中是否有 NaN 或 inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.warning("验证集特征数据中包含 NaN 或 inf，进行清理")
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 保存样本索引
        self._last_sample_indices = sample_indices
        
        if len(labels) > 0:
            y = torch.tensor(np.array(labels), dtype=torch.float32).unsqueeze(-1)
            # 处理 NaN 和 inf
            y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            y = None
        
        return x, y
    
    def _prepare_sequences_for_prediction(
        self,
        combined_feat: pd.DataFrame,
        target_feat: pd.DataFrame,
    ) -> Tuple[torch.Tensor, None]:
        """
        为预测构建时序序列，使用历史数据补充。
        
        参数:
            combined_feat: 合并后的特征（历史+目标）
            target_feat: 目标特征（需要预测的时间点）
        
        返回:
            x: 时序特征张量
        """
        sequences = []
        sample_indices = []
        
        if not isinstance(combined_feat.index, pd.MultiIndex):
            raise ValueError("特征索引应为 MultiIndex")
        
        # 获取目标数据的时间范围
        target_datetimes = target_feat.index.get_level_values("datetime").unique()
        target_instruments = target_feat.index.get_level_values("instrument").unique()
        
        for instrument in target_instruments:
            try:
                # 获取该 instrument 的完整数据（包括历史和目标）
                inst_data = combined_feat.xs(instrument, level=1, drop_level=False)
                if isinstance(inst_data.index, pd.MultiIndex):
                    inst_data = inst_data.reset_index(level=1, drop=True)
                inst_data = inst_data.sort_index()
                
                # 只对目标时间点构建序列
                for datetime_idx in target_datetimes:
                    try:
                        # 找到该时间点在数据中的位置
                        if datetime_idx not in inst_data.index:
                            continue
                        
                        pos = inst_data.index.get_loc(datetime_idx)
                        
                        # 检查是否有足够的历史数据
                        if pos < self.sequence_length:
                            # 数据不足，跳过
                            continue
                        
                        # 构建序列（使用历史数据）
                        seq = inst_data.iloc[pos - self.sequence_length:pos].values
                        sequences.append(seq)
                        sample_indices.append((datetime_idx, instrument))
                    except (KeyError, IndexError):
                        continue
            except KeyError:
                logger.debug("Instrument %s 数据不足，跳过", instrument)
                continue
        
        if len(sequences) == 0:
            raise ValueError("无法构建时序序列，即使使用历史数据补充")
        
        x = torch.tensor(np.array(sequences), dtype=torch.float32)
        
        # 检查 x 中是否有 NaN 或 inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.warning("预测特征数据中包含 NaN 或 inf，进行清理")
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 保存样本索引
        self._last_sample_indices = sample_indices
        
        return x, None
    
    def fit(
        self,
        train_feat: pd.DataFrame,
        train_label: pd.Series,
        valid_feat: Optional[pd.DataFrame] = None,
        valid_label: Optional[pd.Series] = None,
    ):
        """
        训练模型。
        
        参数:
            train_feat: 训练特征
            train_label: 训练标签
            valid_feat: 验证特征（可选）
            valid_label: 验证标签（可选）
        """
        # 准备时序数据（所有模型共享相同的数据准备）
        train_x, train_y = self._prepare_sequences(train_feat, train_label)
        
        if train_y is None:
            raise ValueError("训练标签不能为空")
        
        # 确定特征数量
        self._num_features = train_x.shape[2]
        self._sequence_length = train_x.shape[1]
        
        # 准备验证集数据（如果提供）
        valid_x, valid_y = None, None
        if valid_feat is not None and valid_label is not None:
            try:
                # 先尝试只用验证集数据
                valid_x, valid_y = self._prepare_sequences(valid_feat, valid_label)
            except ValueError:
                # 如果验证集数据不足，使用训练集的历史数据补充
                logger.info("验证集时间窗口较短，使用训练集历史数据补充以构建完整序列（正常行为，无数据泄露）")
                combined_feat = pd.concat([train_feat, valid_feat]).sort_index()
                combined_label = pd.concat([train_label, valid_label]).sort_index()
                valid_x, valid_y = self._prepare_sequences_with_history(
                    combined_feat, combined_label, valid_feat, valid_label
                )
        
        # 创建损失函数（所有模型共享）
        if self.loss_type == "ranking":
            criterion = RankingLoss(alpha=self.ranking_alpha)
            logger.info("使用 Ranking Loss，alpha=%.2f", self.ranking_alpha)
        else:
            criterion = nn.MSELoss()
            logger.info("使用 MSE Loss")
        
        # 集成学习模式：训练多个模型
        if self.is_ensemble:
            self.models = []
            base_seed = 42  # 基础随机种子
            
            for i in range(self.num_ensemble):
                logger.info("=" * 60)
                logger.info("[Ensemble] 训练模型 %d/%d...", i + 1, self.num_ensemble)
                logger.info("=" * 60)
                
                # 设置随机种子（关键：确保每个模型初始化不同）
                seed = base_seed + i
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                np.random.seed(seed)
                
                # 创建模型
                model = IndustryGRU(
                    num_features=self._num_features,
                    sequence_length=self._sequence_length,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    attention_hidden_dim=self.attention_hidden_dim,
                    dropout=self.dropout,
                ).to(self.device)
                
                # 优化器
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                )
                
                # 学习率调度器
                scheduler = None
                if self.lr_scheduler_type == "plateau":
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode="min",
                        factor=0.5,
                        patience=5,
                    )
                elif self.lr_scheduler_type is not None:
                    logger.warning("不支持的学习率调度器类型: %s，将不使用调度器", self.lr_scheduler_type)
                
                # 数据加载器（每次使用不同的随机种子，shuffle 顺序不同）
                train_dataset = TensorDataset(train_x, train_y)
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,  # shuffle=True 配合不同的随机种子，确保数据顺序不同
                )
                
                valid_loader = None
                if valid_x is not None and valid_y is not None and len(valid_x) > 0:
                    valid_dataset = TensorDataset(valid_x, valid_y)
                    valid_loader = DataLoader(
                        valid_dataset,
                        batch_size=self.batch_size,
                        shuffle=False,
                    )
                
                # 训练单个模型
                best_loss = float("inf")
                wait = 0
                best_state = None
                
                for epoch in range(self.max_epochs):
                    # 训练阶段
                    model.train()
                    train_loss = 0.0
                    for batch_x, batch_y in train_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        optimizer.zero_grad()
                        pred = model(batch_x)
                        loss = criterion(pred, batch_y)
                        
                        if torch.isnan(loss) or torch.isinf(loss):
                            logger.warning("训练损失为 NaN 或 inf，跳过该批次")
                            continue
                        
                        loss.backward()
                        
                        if self.grad_clip is not None:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.grad_clip)
                        
                        optimizer.step()
                        train_loss += loss.item() * len(batch_x)
                    
                    train_loss /= len(train_loader.dataset)
                    
                    # 验证阶段
                    val_loss = train_loss
                    if valid_loader is not None:
                        model.eval()
                        total_loss = 0.0
                        with torch.no_grad():
                            for vx, vy in valid_loader:
                                vx = vx.to(self.device)
                                vy = vy.to(self.device)
                                pred = model(vx)
                                loss = criterion(pred, vy)
                                
                                if torch.isnan(loss) or torch.isinf(loss):
                                    logger.warning("验证损失为 NaN 或 inf，跳过该批次")
                                    continue
                                
                                total_loss += loss.item() * len(vx)
                        val_loss = total_loss / len(valid_loader.dataset)
                    
                    # 更新学习率调度器
                    if scheduler is not None:
                        monitor_loss = val_loss if valid_loader is not None else train_loss
                        scheduler.step(monitor_loss)
                    
                    if (epoch + 1) % 10 == 0 or epoch == 0:  # 每10个epoch打印一次，减少日志
                        logger.info(
                            "[Ensemble Model %d/%d] Epoch %d/%d: train_loss=%.6f, valid_loss=%.6f",
                            i + 1, self.num_ensemble, epoch + 1, self.max_epochs, train_loss, val_loss
                        )
                    
                    # 早停
                    if val_loss < best_loss:
                        best_loss = val_loss
                        wait = 0
                        best_state = model.state_dict().copy()
                    else:
                        wait += 1
                        if wait >= self.patience:
                            logger.info("[Ensemble Model %d/%d] 早停触发，最佳验证损失: %.6f", i + 1, self.num_ensemble, best_loss)
                            break
                
                # 加载最佳模型
                if best_state is not None:
                    model.load_state_dict(best_state)
                
                self.models.append(model)
                logger.info("[Ensemble] 模型 %d/%d 训练完成，最佳验证损失: %.6f", i + 1, self.num_ensemble, best_loss)
            
            logger.info("=" * 60)
            logger.info("[Ensemble] 所有 %d 个模型训练完成", self.num_ensemble)
            logger.info("=" * 60)
        
        else:
            # 单模型模式（原有逻辑）
            self.model = IndustryGRU(
                num_features=self._num_features,
                sequence_length=self._sequence_length,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                attention_hidden_dim=self.attention_hidden_dim,
                dropout=self.dropout,
            ).to(self.device)
            
            # 优化器
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
            
            # 学习率调度器
            scheduler = None
            if self.lr_scheduler_type == "plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=0.5,
                    patience=5,
                )
                logger.info("使用 ReduceLROnPlateau 学习率调度器（factor=0.5, patience=5）")
            elif self.lr_scheduler_type is not None:
                logger.warning("不支持的学习率调度器类型: %s，将不使用调度器", self.lr_scheduler_type)
            
            # 梯度裁剪配置日志
            if self.grad_clip is not None:
                logger.info("启用梯度裁剪，最大范数: %.2f", self.grad_clip)
            else:
                logger.info("未启用梯度裁剪")
            
            # 数据加载器
            train_dataset = TensorDataset(train_x, train_y)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
            )
            
            valid_loader = None
            if valid_x is not None and valid_y is not None and len(valid_x) > 0:
                valid_dataset = TensorDataset(valid_x, valid_y)
                valid_loader = DataLoader(
                    valid_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                )
            else:
                logger.warning("验证集序列构建失败，将只使用训练集进行训练")
            
            # 训练循环
            best_loss = float("inf")
            wait = 0
            best_state = None
            
            for epoch in range(self.max_epochs):
                # 训练阶段
                self.model.train()
                train_loss = 0.0
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    pred = self.model(batch_x)
                    loss = criterion(pred, batch_y)
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning("训练损失为 NaN 或 inf，跳过该批次")
                        continue
                    
                    loss.backward()
                    
                    if self.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                    
                    optimizer.step()
                    train_loss += loss.item() * len(batch_x)
                
                train_loss /= len(train_loader.dataset)
                
                # 验证阶段
                val_loss = train_loss
                if valid_loader is not None:
                    self.model.eval()
                    total_loss = 0.0
                    with torch.no_grad():
                        for vx, vy in valid_loader:
                            vx = vx.to(self.device)
                            vy = vy.to(self.device)
                            pred = self.model(vx)
                            loss = criterion(pred, vy)
                            
                            if torch.isnan(loss) or torch.isinf(loss):
                                logger.warning("验证损失为 NaN 或 inf，跳过该批次")
                                continue
                            
                            total_loss += loss.item() * len(vx)
                    val_loss = total_loss / len(valid_loader.dataset)
                
                # 更新学习率调度器
                current_lr = optimizer.param_groups[0]["lr"]
                if scheduler is not None:
                    monitor_loss = val_loss if valid_loader is not None else train_loss
                    scheduler.step(monitor_loss)
                    new_lr = optimizer.param_groups[0]["lr"]
                    if new_lr < current_lr:
                        logger.info("学习率已降低: %.6f -> %.6f (监控损失: %.6f)", current_lr, new_lr, monitor_loss)
                
                logger.info(
                    "IndustryGRU epoch %d/%d: train_loss=%.6f, valid_loss=%.6f, lr=%.6f",
                    epoch + 1, self.max_epochs, train_loss, val_loss, optimizer.param_groups[0]["lr"]
                )
                
                # 早停
                if val_loss < best_loss:
                    best_loss = val_loss
                    wait = 0
                    best_state = self.model.state_dict().copy()
                else:
                    wait += 1
                    if wait >= self.patience:
                        logger.info("早停触发，最佳验证损失: %.6f", best_loss)
                        break
            
            # 加载最佳模型
            if best_state is not None:
                self.model.load_state_dict(best_state)
    
    def predict(self, feat: pd.DataFrame, history_feat: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        预测。
        
        参数:
            feat: 特征 DataFrame（需要预测的数据）
            history_feat: 历史特征 DataFrame（可选，用于补充构建完整序列）
        
        返回:
            预测 Series（集成模式下为所有模型的平均预测）
        """
        # 检查模型是否已训练
        if self.is_ensemble:
            if self.models is None or len(self.models) == 0:
                raise RuntimeError("集成模型尚未训练")
        else:
            if self.model is None:
                raise RuntimeError("模型尚未训练")
        
        # 准备输入数据（所有模型共享相同的数据准备）
        if history_feat is not None:
            try:
                # 先尝试只用当前数据（更快）
                x, _ = self._prepare_sequences(feat, None)
                logger.debug("使用当前数据构建序列，样本数: %d", len(x))
            except ValueError:
                # 如果数据不足，使用历史数据补充
                logger.info("预测数据时间窗口较短，使用历史数据补充以构建完整序列（正常行为，无数据泄露）")
                import time
                merge_start = time.time()
                combined_feat = pd.concat([history_feat, feat]).sort_index()
                merge_time = time.time() - merge_start
                logger.debug("合并历史数据耗时: %.2f 秒（训练集样本: %d, 验证集样本: %d）", 
                            merge_time, len(history_feat), len(feat))
                # 只对 feat 的时间点进行预测
                seq_start = time.time()
                x, _ = self._prepare_sequences_for_prediction(combined_feat, feat)
                seq_time = time.time() - seq_start
                logger.debug("构建序列耗时: %.2f 秒，序列数: %d", seq_time, len(x))
        else:
            try:
                x, _ = self._prepare_sequences(feat, None)
            except ValueError as e:
                logger.error("无法构建时序序列，且未提供历史数据: %s", e)
                raise
        
        # 集成学习模式：对所有模型预测并取平均
        if self.is_ensemble:
            all_model_preds = []
            
            for i, model in enumerate(self.models):
                model.eval()
                model_preds = []
                
                with torch.no_grad():
                    for j in range(0, len(x), self.batch_size):
                        batch_x = x[j:j + self.batch_size].to(self.device)
                        batch_pred = model(batch_x)
                        model_preds.append(batch_pred.cpu().numpy())
                
                model_pred = np.concatenate(model_preds).flatten()
                all_model_preds.append(model_pred)
            
            # 对所有模型的预测取平均
            all_model_preds = np.array(all_model_preds)  # shape: (num_ensemble, n_samples)
            preds = np.mean(all_model_preds, axis=0)  # shape: (n_samples,)
            
            logger.debug("集成预测完成，使用 %d 个模型的平均预测", self.num_ensemble)
        
        else:
            # 单模型模式
            self.model.eval()
            all_preds = []
            
            with torch.no_grad():
                for i in range(0, len(x), self.batch_size):
                    batch_x = x[i:i + self.batch_size].to(self.device)
                    batch_pred = self.model(batch_x)
                    all_preds.append(batch_pred.cpu().numpy())
            
            preds = np.concatenate(all_preds).flatten()
        
        # 构建索引（使用保存的样本索引）
        if hasattr(self, '_last_sample_indices') and self._last_sample_indices:
            if isinstance(self._last_sample_indices[0], tuple):
                # MultiIndex
                pred_index = pd.MultiIndex.from_tuples(
                    self._last_sample_indices,
                    names=feat.index.names if isinstance(feat.index, pd.MultiIndex) else ['datetime', 'instrument']
                )
            else:
                # 单层索引
                pred_index = pd.Index(self._last_sample_indices, name=feat.index.name if hasattr(feat.index, 'name') else 'datetime')
        else:
            # 回退方案：从原始特征构建索引
            logger.warning("无法使用保存的样本索引，使用回退方案构建索引")
            if isinstance(feat.index, pd.MultiIndex):
                pred_index = []
                for instrument in feat.index.get_level_values(1).unique():
                    try:
                        inst_data = feat.xs(instrument, level=1, drop_level=False)
                        if isinstance(inst_data.index, pd.MultiIndex):
                            inst_data = inst_data.reset_index(level=1, drop=True)
                        inst_data = inst_data.sort_index()
                        for idx in inst_data.index[self.sequence_length:]:
                            pred_index.append((idx, instrument))
                    except KeyError:
                        continue
                if pred_index:
                    pred_index = pd.MultiIndex.from_tuples(pred_index, names=feat.index.names)
                else:
                    pred_index = pd.RangeIndex(len(preds))
            else:
                pred_index = feat.index[self.sequence_length:] if len(feat) > self.sequence_length else pd.RangeIndex(len(preds))
        
        return pd.Series(preds, index=pred_index, name="industry_gru_pred")
    
    def save(self, output_dir: str, model_name: str):
        """
        保存模型。
        
        参数:
            output_dir: 输出目录
            model_name: 模型名称（不含扩展名）
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if self.is_ensemble:
            # 集成模式：保存所有模型的状态字典
            if self.models is None or len(self.models) == 0:
                raise RuntimeError("集成模型尚未训练，无可保存的模型")
            
            ensemble_state_dicts = {}
            for i, model in enumerate(self.models):
                ensemble_state_dicts[f"model_{i}"] = model.state_dict()
            
            path = os.path.join(output_dir, f"{model_name}_industry_gru_ensemble.pt")
            
            torch.save(
                {
                    "ensemble_state_dicts": ensemble_state_dicts,
                    "num_ensemble": self.num_ensemble,
                    "config": self.config,
                    "num_features": self._num_features,
                    "sequence_length": self._sequence_length,
                },
                path,
            )
            
            logger.info("IndustryGRU 集成模型已保存: %s（包含 %d 个子模型）", path, self.num_ensemble)
        
        else:
            # 单模型模式
            if self.model is None:
                raise RuntimeError("模型尚未训练，无可保存的模型")
            
            path = os.path.join(output_dir, f"{model_name}_industry_gru.pt")
            
            torch.save(
                {
                    "state_dict": self.model.state_dict(),
                    "config": self.config,
                    "num_features": self._num_features,
                    "sequence_length": self._sequence_length,
                },
                path,
            )
            
            logger.info("IndustryGRU 模型已保存: %s", path)
    
    def load(self, output_dir: str, model_name: str):
        """
        加载模型。
        
        参数:
            output_dir: 模型目录
            model_name: 模型名称（不含扩展名）
        """
        # 先尝试加载集成模型
        ensemble_path = os.path.join(output_dir, f"{model_name}_industry_gru_ensemble.pt")
        single_path = os.path.join(output_dir, f"{model_name}_industry_gru.pt")
        
        if os.path.exists(ensemble_path):
            # 加载集成模型
            ckpt = torch.load(ensemble_path, map_location=self.device)
            
            self._num_features = ckpt["num_features"]
            self._sequence_length = ckpt["sequence_length"]
            num_ensemble = ckpt.get("num_ensemble", len(ckpt["ensemble_state_dicts"]))
            
            # 更新配置以匹配保存的模型
            if num_ensemble != self.num_ensemble:
                logger.warning(
                    "配置中的 num_ensemble (%d) 与保存的模型数量 (%d) 不一致，使用保存的数量",
                    self.num_ensemble, num_ensemble
                )
                self.num_ensemble = num_ensemble
                self.is_ensemble = num_ensemble > 1
            
            # 重建所有模型
            self.models = []
            ensemble_state_dicts = ckpt["ensemble_state_dicts"]
            
            for i in range(num_ensemble):
                model = IndustryGRU(
                    num_features=self._num_features,
                    sequence_length=self._sequence_length,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    attention_hidden_dim=self.attention_hidden_dim,
                    dropout=self.dropout,
                ).to(self.device)
                
                model_key = f"model_{i}"
                if model_key in ensemble_state_dicts:
                    model.load_state_dict(ensemble_state_dicts[model_key])
                else:
                    logger.warning("未找到模型 %d 的状态字典，跳过", i)
                    continue
                
                self.models.append(model)
            
            logger.info("IndustryGRU 集成模型已加载: %s（包含 %d 个子模型）", ensemble_path, len(self.models))
        
        elif os.path.exists(single_path):
            # 加载单模型
            ckpt = torch.load(single_path, map_location=self.device)
            
            self._num_features = ckpt["num_features"]
            self._sequence_length = ckpt["sequence_length"]
            
            # 如果当前配置是集成模式，但加载的是单模型，需要调整
            if self.is_ensemble:
                logger.warning("配置为集成模式，但加载的是单模型，切换到单模型模式")
                self.is_ensemble = False
                self.models = None
            
            # 重建模型
            self.model = IndustryGRU(
                num_features=self._num_features,
                sequence_length=self._sequence_length,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                attention_hidden_dim=self.attention_hidden_dim,
                dropout=self.dropout,
            ).to(self.device)
            
            self.model.load_state_dict(ckpt["state_dict"])
            logger.info("IndustryGRU 模型已加载: %s", single_path)
        
        else:
            raise FileNotFoundError(
                f"未找到模型文件。尝试了以下路径：\n  - {ensemble_path}\n  - {single_path}"
            )

