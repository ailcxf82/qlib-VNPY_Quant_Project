"""
预测问题诊断工具函数

可以在训练代码中直接调用，快速诊断预测问题。
"""

import logging
from typing import Tuple

import pandas as pd
from scipy.stats import spearmanr

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


def quick_diagnosis(
    train_label: pd.Series,
    valid_label: pd.Series,
    valid_pred: pd.Series = None,
    train_feat: pd.DataFrame = None,
    valid_feat: pd.DataFrame = None,
) -> dict:
    """
    快速诊断预测问题
    
    参数:
        train_label: 训练集标签
        valid_label: 验证集标签
        valid_pred: 验证集预测（可选）
        train_feat: 训练集特征（可选，用于单特征基线测试）
        valid_feat: 验证集特征（可选，用于单特征基线测试）
    
    返回:
        诊断结果字典
    """
    results = {}
    
    # 测试 1：完美预测自测
    logger.info("=" * 60)
    logger.info("快速诊断：测试 1 - 完美预测自测")
    logger.info("=" * 60)
    
    train_ic_perfect = rank_ic(train_label, train_label)
    valid_ic_perfect = rank_ic(valid_label, valid_label)
    
    results["test1_train_ic"] = train_ic_perfect
    results["test1_valid_ic"] = valid_ic_perfect
    results["test1_pass"] = abs(train_ic_perfect - 1.0) < 0.01 and abs(valid_ic_perfect - 1.0) < 0.01
    
    logger.info("训练集 IC (pred=label): %.6f", train_ic_perfect)
    logger.info("验证集 IC (pred=label): %.6f", valid_ic_perfect)
    
    if results["test1_pass"]:
        logger.info("✅ 测试 1 通过：IC 计算正常")
    else:
        logger.error("❌ 测试 1 失败：IC 计算或数据对齐有问题！")
        logger.error("   训练集 IC 偏离 1.0: %.6f", abs(train_ic_perfect - 1.0))
        logger.error("   验证集 IC 偏离 1.0: %.6f", abs(valid_ic_perfect - 1.0))
    
    logger.info("")
    
    # 测试 2：取负号测试（如果有预测结果）
    if valid_pred is not None:
        logger.info("=" * 60)
        logger.info("快速诊断：测试 2 - 取负号测试")
        logger.info("=" * 60)
        
        original_ic = rank_ic(valid_pred, valid_label)
        negative_ic = rank_ic(-valid_pred, valid_label)
        
        results["test2_original_ic"] = original_ic
        results["test2_negative_ic"] = negative_ic
        results["test2_direction_correct"] = abs(original_ic) >= abs(negative_ic)
        
        logger.info("原始 IC: %.6f", original_ic)
        logger.info("取负号后 IC: %.6f", negative_ic)
        
        if results["test2_direction_correct"]:
            logger.info("✅ 测试 2 通过：方向正确")
        else:
            logger.warning("⚠️  测试 2 警告：取负号后 IC 更高，可能存在方向问题！")
        
        logger.info("")
    
    # 测试 3：单特征基线（如果有特征数据）
    if train_feat is not None and valid_feat is not None and len(train_feat.columns) > 0:
        logger.info("=" * 60)
        logger.info("快速诊断：测试 3 - 单特征基线")
        logger.info("=" * 60)
        
        # 尝试找到 ret20 或类似特征
        feature_name = None
        for name in ["ret_20", "ret20", "$close / Ref($close, 20) - 1", "pct_change", "$pct_change"]:
            if name in train_feat.columns:
                feature_name = name
                break
        
        if feature_name is None:
            feature_name = train_feat.columns[0]
        
        logger.info("使用特征: %s", feature_name)
        
        train_feat_single = train_feat[feature_name].dropna()
        valid_feat_single = valid_feat[feature_name].dropna()
        
        train_feat_single, train_label_aligned = train_feat_single.align(train_label, join="inner")
        valid_feat_single, valid_label_aligned = valid_feat_single.align(valid_label, join="inner")
        
        valid_ic_baseline = rank_ic(valid_feat_single, valid_label_aligned)
        
        results["test3_feature_name"] = feature_name
        results["test3_valid_ic"] = valid_ic_baseline
        results["test3_has_signal"] = valid_ic_baseline > 0
        
        logger.info("验证集 IC (单特征基线): %.6f", valid_ic_baseline)
        
        if results["test3_has_signal"]:
            logger.info("✅ 测试 3 通过：数据有基本信号")
        else:
            logger.warning("⚠️  测试 3 警告：单特征基线 IC 为负或接近 0，数据可能没有基本信号")
        
        logger.info("")
    
    # 总结
    logger.info("=" * 60)
    logger.info("诊断总结")
    logger.info("=" * 60)
    
    all_passed = results.get("test1_pass", False)
    if valid_pred is not None:
        all_passed = all_passed and results.get("test2_direction_correct", False)
    if train_feat is not None and valid_feat is not None:
        all_passed = all_passed and results.get("test3_has_signal", False)
    
    if all_passed:
        logger.info("✅ 所有测试通过")
    else:
        logger.warning("⚠️  部分测试未通过，请查看上述详细信息")
    
    logger.info("")
    
    return results

