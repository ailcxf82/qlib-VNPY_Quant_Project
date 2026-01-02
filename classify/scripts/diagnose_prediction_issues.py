"""
行业轮动预测问题诊断脚本

按优先级从高到低进行诊断测试：
1. 完美预测自测：令 pred=label，IC 应接近 1（否则 IC/对齐有 bug）
2. 取负号测试：用 pred=-pred 重算 IC（判断方向问题）
3. 单特征基线：只用 ret20（或 pct_change 的滚动动量）做线性回归/排序，看验证 IC 是否为正
4. 截面 rank 特征 vs 原始特征：只改标准化方式，看 IC 是否显著变化
5. 按日截面 batch + ranking loss：不改模型结构，只改训练组织
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 添加项目根目录到路径
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from feature.qlib_feature_pipeline import QlibFeaturePipeline
from utils import load_yaml_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def rank_ic(pred: pd.Series, label: pd.Series) -> float:
    """计算 Rank IC（Spearman 相关系数）"""
    pred, label = pred.align(label, join="inner")
    if pred.empty or len(pred) < 2:
        return float("nan")
    try:
        return spearmanr(pred.values, label.values)[0]
    except Exception as e:
        logger.warning("计算 Rank IC 失败: %s", e)
        return float("nan")


def test_1_perfect_prediction(train_label: pd.Series, valid_label: pd.Series) -> Tuple[float, float]:
    """
    测试 1：完美预测自测
    令 pred=label，IC 应接近 1（否则 IC/对齐有 bug）
    """
    logger.info("=" * 80)
    logger.info("测试 1：完美预测自测")
    logger.info("=" * 80)
    logger.info("原理：如果 pred=label，IC 应该接近 1.0")
    logger.info("如果 IC 不接近 1，说明 IC 计算或数据对齐有问题")
    logger.info("")
    
    # 训练集测试
    train_ic = rank_ic(train_label, train_label)
    logger.info("训练集 IC (pred=label): %.6f", train_ic)
    
    # 验证集测试
    valid_ic = rank_ic(valid_label, valid_label)
    logger.info("验证集 IC (pred=label): %.6f", valid_ic)
    logger.info("")
    
    if abs(train_ic - 1.0) > 0.01 or abs(valid_ic - 1.0) > 0.01:
        logger.error("❌ IC 计算或数据对齐有问题！")
        logger.error("   训练集 IC 偏离 1.0: %.6f", abs(train_ic - 1.0))
        logger.error("   验证集 IC 偏离 1.0: %.6f", abs(valid_ic - 1.0))
        return train_ic, valid_ic
    else:
        logger.info("✅ IC 计算正常")
        return train_ic, valid_ic


def test_2_negative_prediction(pred: pd.Series, label: pd.Series) -> Tuple[float, float]:
    """
    测试 2：取负号测试
    用 pred=-pred 重算 IC（判断方向问题）
    """
    logger.info("=" * 80)
    logger.info("测试 2：取负号测试")
    logger.info("=" * 80)
    logger.info("原理：如果模型预测方向相反，IC 应该为负")
    logger.info("通过取负号测试，判断是否存在方向问题")
    logger.info("")
    
    # 原始 IC
    original_ic = rank_ic(pred, label)
    logger.info("原始 IC: %.6f", original_ic)
    
    # 取负号后的 IC
    negative_ic = rank_ic(-pred, label)
    logger.info("取负号后 IC: %.6f", negative_ic)
    logger.info("")
    
    if abs(negative_ic) > abs(original_ic):
        logger.warning("⚠️  取负号后 IC 更高，可能存在方向问题！")
        logger.warning("   建议检查：模型是否预测了相反的方向？")
    else:
        logger.info("✅ 方向正确（原始 IC 更高）")
    
    return original_ic, negative_ic


def test_3_single_feature_baseline(
    train_feat: pd.DataFrame,
    train_label: pd.Series,
    valid_feat: pd.DataFrame,
    valid_label: pd.Series,
    feature_name: str = None,
) -> Tuple[float, float]:
    """
    测试 3：单特征基线
    只用 ret20（或 pct_change 的滚动动量）做线性回归/排序，看验证 IC 是否为正
    """
    logger.info("=" * 80)
    logger.info("测试 3：单特征基线")
    logger.info("=" * 80)
    logger.info("原理：如果单个特征（如 ret20）都无法产生正 IC，说明数据本身可能没有信号")
    logger.info("")
    
    # 查找 ret20 或 pct_change 特征
    if feature_name is None:
        # 尝试找到 ret20 或类似的特征
        possible_names = [
            "ret_20",
            "ret20",
            "$close / Ref($close, 20) - 1",
            "pct_change",
            "$pct_change",
        ]
        
        feature_name = None
        for name in possible_names:
            if name in train_feat.columns:
                feature_name = name
                break
        
        # 如果没找到，使用第一列
        if feature_name is None:
            feature_name = train_feat.columns[0]
            logger.warning("未找到 ret20 特征，使用第一列: %s", feature_name)
    
    if feature_name not in train_feat.columns:
        logger.error("特征 %s 不存在，可用特征: %s", feature_name, list(train_feat.columns[:10]))
        return float("nan"), float("nan")
    
    logger.info("使用特征: %s", feature_name)
    
    # 提取特征
    train_feat_single = train_feat[feature_name].dropna()
    valid_feat_single = valid_feat[feature_name].dropna()
    
    # 对齐标签
    train_feat_single, train_label_aligned = train_feat_single.align(train_label, join="inner")
    valid_feat_single, valid_label_aligned = valid_feat_single.align(valid_label, join="inner")
    
    # 方法 1：直接使用特征值作为预测（排序）
    train_ic_direct = rank_ic(train_feat_single, train_label_aligned)
    valid_ic_direct = rank_ic(valid_feat_single, valid_label_aligned)
    
    logger.info("直接使用特征值（排序）:")
    logger.info("  训练集 IC: %.6f", train_ic_direct)
    logger.info("  验证集 IC: %.6f", valid_ic_direct)
    logger.info("")
    
    # 方法 2：线性回归
    if len(train_feat_single) > 10 and len(valid_feat_single) > 0:
        try:
            # 训练线性回归
            X_train = train_feat_single.values.reshape(-1, 1)
            y_train = train_label_aligned.values
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # 预测
            X_valid = valid_feat_single.values.reshape(-1, 1)
            valid_pred = model.predict(X_valid)
            valid_pred_series = pd.Series(valid_pred, index=valid_feat_single.index)
            
            valid_ic_lr = rank_ic(valid_pred_series, valid_label_aligned)
            logger.info("线性回归:")
            logger.info("  验证集 IC: %.6f", valid_ic_lr)
            logger.info("")
            
            if valid_ic_direct > 0 or valid_ic_lr > 0:
                logger.info("✅ 单特征基线有正 IC，数据本身有信号")
            else:
                logger.warning("⚠️  单特征基线 IC 为负或接近 0，数据可能没有基本信号")
        except Exception as e:
            logger.warning("线性回归失败: %s", e)
            valid_ic_lr = float("nan")
    else:
        valid_ic_lr = float("nan")
    
    return valid_ic_direct, valid_ic_lr


def test_4_cross_sectional_rank_vs_raw(
    train_feat: pd.DataFrame,
    train_label: pd.Series,
    valid_feat: pd.DataFrame,
    valid_label: pd.Series,
) -> Tuple[float, float, float, float]:
    """
    测试 4：截面 rank 特征 vs 原始特征
    只改标准化方式，看 IC 是否显著变化
    """
    logger.info("=" * 80)
    logger.info("测试 4：截面 rank 特征 vs 原始特征")
    logger.info("=" * 80)
    logger.info("原理：如果截面标准化方式对 IC 影响很大，说明尺度问题很重要")
    logger.info("")
    
    # 使用第一个特征进行测试
    feature_name = train_feat.columns[0]
    logger.info("使用特征: %s", feature_name)
    
    # 原始特征
    train_feat_raw = train_feat[feature_name].dropna()
    valid_feat_raw = valid_feat[feature_name].dropna()
    
    train_feat_raw, train_label_aligned = train_feat_raw.align(train_label, join="inner")
    valid_feat_raw, valid_label_aligned = valid_feat_raw.align(valid_label, join="inner")
    
    train_ic_raw = rank_ic(train_feat_raw, train_label_aligned)
    valid_ic_raw = rank_ic(valid_feat_raw, valid_label_aligned)
    
    logger.info("原始特征:")
    logger.info("  训练集 IC: %.6f", train_ic_raw)
    logger.info("  验证集 IC: %.6f", valid_ic_raw)
    logger.info("")
    
    # 截面 rank 特征（按日期分组进行排名）
    if isinstance(train_feat_raw.index, pd.MultiIndex) and "datetime" in train_feat_raw.index.names:
        # 按日期分组进行排名
        train_feat_rank = train_feat_raw.groupby(level="datetime").transform(lambda x: x.rank(pct=True))
        valid_feat_rank = valid_feat_raw.groupby(level="datetime").transform(lambda x: x.rank(pct=True))
        
        train_ic_rank = rank_ic(train_feat_rank, train_label_aligned)
        valid_ic_rank = rank_ic(valid_feat_rank, valid_label_aligned)
        
        logger.info("截面 rank 特征（按日期排名）:")
        logger.info("  训练集 IC: %.6f", train_ic_rank)
        logger.info("  验证集 IC: %.6f", valid_ic_rank)
        logger.info("")
        
        ic_diff = abs(valid_ic_rank - valid_ic_raw)
        if ic_diff > 0.05:
            logger.warning("⚠️  标准化方式对 IC 影响很大 (差异: %.6f)，说明尺度问题很重要", ic_diff)
        else:
            logger.info("✅ 标准化方式对 IC 影响较小")
    else:
        logger.warning("无法进行截面 rank 测试（索引不是 MultiIndex 或没有 datetime 层级）")
        train_ic_rank = float("nan")
        valid_ic_rank = float("nan")
    
    return train_ic_raw, valid_ic_raw, train_ic_rank, valid_ic_rank


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="行业轮动预测问题诊断")
    parser.add_argument(
        "--config",
        type=str,
        default="classify/config_industry_rotation.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=0,
        help="使用第几个训练窗口（默认 0，即第一个窗口）",
    )
    args = parser.parse_args()
    
    logger.info("开始诊断预测问题...")
    logger.info("配置文件: %s", args.config)
    logger.info("")
    
    # 加载配置
    cfg = load_yaml_config(args.config)
    data_cfg = load_yaml_config(cfg["data_config"])
    
    # 创建特征管道
    pipeline = QlibFeaturePipeline(cfg["data_config"])
    pipeline.build()
    
    # 获取特征和标签
    features, labels = pipeline.get_all()
    
    if features.empty or labels.empty:
        logger.error("无法获取特征或标签数据")
        return
    
    # 获取第一个训练窗口的数据（用于测试）
    # 这里简化处理，使用前 80% 作为训练集，后 20% 作为验证集
    dates = features.index.get_level_values("datetime").unique().sort_values()
    split_idx = int(len(dates) * 0.8)
    train_dates = dates[:split_idx]
    valid_dates = dates[split_idx:]
    
    train_mask = features.index.get_level_values("datetime").isin(train_dates)
    valid_mask = features.index.get_level_values("datetime").isin(valid_dates)
    
    train_feat = features.loc[train_mask]
    train_label = labels.loc[train_mask]
    valid_feat = features.loc[valid_mask]
    valid_label = labels.loc[valid_mask]
    
    logger.info("数据划分:")
    logger.info("  训练集: %d 样本，日期范围: %s 至 %s", 
                len(train_feat), train_dates.min(), train_dates.max())
    logger.info("  验证集: %d 样本，日期范围: %s 至 %s", 
                len(valid_feat), valid_dates.min(), valid_dates.max())
    logger.info("")
    
    # 测试 1：完美预测自测
    test_1_perfect_prediction(train_label, valid_label)
    logger.info("")
    
    # 测试 2：取负号测试（需要先有一个预测结果）
    # 这里使用特征值作为预测（简化测试）
    if len(train_feat.columns) > 0:
        test_feature = train_feat.columns[0]
        train_pred = train_feat[test_feature].dropna()
        train_pred, train_label_aligned = train_pred.align(train_label, join="inner")
        test_2_negative_prediction(train_pred, train_label_aligned)
        logger.info("")
    
    # 测试 3：单特征基线
    test_3_single_feature_baseline(train_feat, train_label, valid_feat, valid_label)
    logger.info("")
    
    # 测试 4：截面 rank 特征 vs 原始特征
    test_4_cross_sectional_rank_vs_raw(train_feat, train_label, valid_feat, valid_label)
    logger.info("")
    
    logger.info("=" * 80)
    logger.info("诊断完成")
    logger.info("=" * 80)
    logger.info("")
    logger.info("建议：")
    logger.info("  1. 如果测试 1 失败，检查 IC 计算或数据对齐")
    logger.info("  2. 如果测试 2 显示方向问题，检查模型预测方向")
    logger.info("  3. 如果测试 3 IC 为负，说明数据本身可能没有信号")
    logger.info("  4. 如果测试 4 显示标准化影响大，考虑使用截面标准化")
    logger.info("  5. 测试 5（按日截面 batch）需要在训练代码中实现")


if __name__ == "__main__":
    main()

