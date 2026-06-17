"""
自定义损失函数：支持非对称损失（惩罚低估正向收益）。
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AsymmetricMSELoss(nn.Module):
    """
    非对称 MSE 损失函数。
    
    当实际值为正且预测值低于实际值时（低估正向收益），给予更重的惩罚。
    
    公式：
        L(y_hat, y) = {
            (y_hat - y)^2 * gamma  if y > 0 and y_hat < y  (乐观不足)
            (y_hat - y)^2          otherwise
        }
    
    Args:
        gamma: 惩罚系数，> 1 表示对低估正向收益的惩罚倍数
    """
    
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        logger.info(f"初始化非对称 MSE 损失，惩罚系数 gamma={gamma}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算损失。
        
        Args:
            pred: 预测值，形状 (batch_size,)
            target: 真实值，形状 (batch_size,)
        
        Returns:
            损失值（标量）
        """
        # 基础 MSE
        mse = (pred - target) ** 2
        
        # 识别"乐观不足"的情况：实际值为正，但预测值低于实际值
        optimistic_insufficient = (target > 0) & (pred < target)
        
        # 应用惩罚
        loss = torch.where(
            optimistic_insufficient,
            mse * self.gamma,
            mse
        )
        
        return loss.mean()


class WeightedMSELoss(nn.Module):
    """
    加权 MSE 损失函数。
    
    对正向收益给予更高权重，对负向收益给予较低权重。
    
    公式：
        L(y_hat, y) = w(y) * (y_hat - y)^2
        其中 w(y) = {
            w_positive  if y > 0
            w_negative  if y <= 0
        }
    
    Args:
        w_positive: 正向收益的权重（通常 > 1）
        w_negative: 负向收益的权重（通常 < 1）
    """
    
    def __init__(self, w_positive: float = 2.0, w_negative: float = 0.5):
        super().__init__()
        self.w_positive = w_positive
        self.w_negative = w_negative
        logger.info(f"初始化加权 MSE 损失，正向权重={w_positive}，负向权重={w_negative}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算损失。
        
        Args:
            pred: 预测值，形状 (batch_size,)
            target: 真实值，形状 (batch_size,)
        
        Returns:
            损失值（标量）
        """
        mse = (pred - target) ** 2
        
        # 根据目标值的正负分配权重
        weights = torch.where(
            target > 0,
            torch.full_like(target, self.w_positive),
            torch.full_like(target, self.w_negative)
        )
        
        loss = mse * weights
        return loss.mean()


class ListMLELoss(nn.Module):
    """ListMLE：Listwise 排序极大似然损失。

    Phase 1 P1-5：GRU 任务的最终目标是 RankIC，但 MSE 类损失在 logits 与
    label 之间插入了 L2 距离假设，跟分位/秩无关。ListMLE 直接优化"按真实
    label 排序的 permutation 在模型 logits 上的概率"，与 RankIC 的优化方向
    一致。

    定义（Plackett-Luce 似然取负对数，参考 Xia et al. 2008）：
        给定一组分数 s_1, ..., s_n（pred）和真实排名 \pi（按 target 降序），
        loss = - sum_{i=1..n} log( exp(s_{\pi(i)}) / sum_{j>=i} exp(s_{\pi(j)}) )

    实现细节：
    - 输入 ``pred`` 和 ``target`` 形状均为 (B,) 或 (B, 1)，整 batch 视为
      一个 group（GRU 训练时一个 mini-batch 通常包含同日多只股票，近似
      当日截面）。
    - 数值稳定：先对 ``pred[\pi]`` 减最大值再做 logcumsumexp。
    - 退化情况：batch 内 target 全相等时 loss=0（无序列对比信息）。

    Args:
        eps: 无实际作用，保留参数以便未来扩展。
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        logger.info("初始化 ListMLE 损失（按 batch 截面排序优化 RankIC）")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.dim() == 0:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        pred = pred.flatten()
        target = target.flatten()
        if pred.numel() < 2:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        # 若 target 全相等（如全 0），ListMLE 退化为 0；直接退回 MSE 兜底
        if torch.all(target == target[0]):
            return torch.mean((pred - target) ** 2) * 0.0  # 保证可反传，但梯度=0
        # 按 target 降序得到 permutation
        sorted_idx = torch.argsort(target, descending=True)
        scores = pred[sorted_idx]
        # 数值稳定：减最大值
        max_score = scores.max().detach()
        scores = scores - max_score
        # log( sum_{j>=i} exp(s_j) )：从尾向头累加 → flip + cumsum + flip
        logcumsumexp_rev = torch.logcumsumexp(scores.flip(0), dim=0).flip(0)
        loss = -(scores - logcumsumexp_rev).mean()
        return loss


class WeightedRankMSELoss(nn.Module):
    """按截面 rank 加权的 MSE：兼顾点估计与排序惩罚。

    Phase 1 P1-5：作为 ListMLE 的兜底实现。在 batch 内：
    - 用 target 的标准化秩 r_i ∈ [0, 1] 作为"排序信息"
    - 用 pred 的同样标准化秩 r̂_i
    - 对 (r̂_i - r_i)^2 做 sample weight 加权（极端排名样本权重更高），再
      与原始 MSE 做加权和

    最终 loss = alpha * MSE(pred, target) + (1-alpha) * weighted_rank_mse

    其中 weighted_rank_mse 主要驱动模型在排名两端的学习，而 alpha 项保留
    点估计的尺度信息。
    """

    def __init__(self, alpha: float = 0.3, eps: float = 1e-8):
        super().__init__()
        self.alpha = float(alpha)
        self.eps = eps
        logger.info(
            "初始化 WeightedRankMSE 损失，alpha=%s（alpha*MSE + (1-alpha)*rank_mse）",
            self.alpha,
        )

    @staticmethod
    def _to_rank01(x: torch.Tensor) -> torch.Tensor:
        # argsort 两次得到 0..n-1 的秩（同分按出现顺序）；线性映射到 [0, 1]
        order = torch.argsort(x, dim=-1)
        ranks = torch.argsort(order, dim=-1).to(x.dtype)
        n = x.numel()
        return ranks / max(1.0, float(n - 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_f = pred.flatten()
        target_f = target.flatten()
        if pred_f.numel() < 2:
            return torch.mean((pred_f - target_f) ** 2)
        mse = torch.mean((pred_f - target_f) ** 2)
        # 排序部分（rank01 不需要梯度通过 argsort，因此对 pred_rank
        # detach；改用 soft rank 也可，但实现复杂度更高）
        with torch.no_grad():
            target_rank = self._to_rank01(target_f)
            pred_rank = self._to_rank01(pred_f)
            # 距离两端越远，权重越大（聚焦头/尾极端）
            sample_w = (target_rank - 0.5).abs() * 2.0 + self.eps
            sample_w = sample_w / sample_w.mean()
        # 用 (pred - target) 在排名空间做近似 MSE：pred 的"近似秩"用其标准化
        # 后的 sigmoid 概率（保留梯度）。这里采用一个简化但可微的版本：
        #   rank_target = target_rank（detach），rank_pred ≈ sigmoid(pred 标准化)
        pred_std = pred_f.std() + self.eps
        pred_z = (pred_f - pred_f.mean()) / pred_std
        rank_pred_soft = torch.sigmoid(pred_z)
        rank_loss = ((rank_pred_soft - target_rank) ** 2 * sample_w).mean()
        return self.alpha * mse + (1.0 - self.alpha) * rank_loss


def asymmetric_mse_objective_lgb(y_true: np.ndarray, y_pred: np.ndarray, gamma: float = 2.0) -> tuple:
    """
    LightGBM 自定义目标函数：非对称 MSE。
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        gamma: 惩罚系数
    
    Returns:
        (gradient, hessian) 用于 LightGBM
    """
    # 计算残差
    residual = y_pred - y_true
    
    # 识别"乐观不足"的情况
    optimistic_insufficient = (y_true > 0) & (y_pred < y_true)
    
    # 梯度
    grad = np.where(optimistic_insufficient, 2 * residual * gamma, 2 * residual)
    
    # 二阶导数（Hessian）
    hess = np.where(optimistic_insufficient, 2 * gamma, 2.0)
    
    return grad, hess


def asymmetric_mse_metric_lgb(y_true: np.ndarray, y_pred: np.ndarray, gamma: float = 2.0) -> tuple:
    """
    LightGBM 自定义评估指标：非对称 MSE。
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        gamma: 惩罚系数
    
    Returns:
        (metric_name, metric_value, is_higher_better)
    """
    # 基础 MSE
    mse = (y_pred - y_true) ** 2
    
    # 识别"乐观不足"的情况
    optimistic_insufficient = (y_true > 0) & (y_pred < y_true)
    
    # 应用惩罚
    loss = np.where(optimistic_insufficient, mse * gamma, mse)
    
    metric_value = loss.mean()
    return "asymmetric_mse", metric_value, False  # False 表示越小越好


