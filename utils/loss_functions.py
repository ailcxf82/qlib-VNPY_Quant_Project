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


