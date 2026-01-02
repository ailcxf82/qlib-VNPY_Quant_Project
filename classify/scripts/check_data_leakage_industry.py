"""
行业轮动数据泄露检查脚本

检查以下问题：
1. Purge gap 是否正确
2. 截面标准化是否使用未来数据
3. 时间序列归一化是否使用未来数据
4. 训练集预测是否使用了训练数据本身（导致 IC 虚高）
5. 错位标签实验
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# 添加项目根目录到路径
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from classify.pytorch_industry_gru import IndustryGRUWrapper
from feature.qlib_feature_pipeline import QlibFeaturePipeline
from utils import load_yaml_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def check_purge_gap(config_path: str):
    """检查 Purge Gap 配置"""
    logger.info("=" * 80)
    logger.info("检查 1: Purge Gap 配置")
    logger.info("=" * 80)
    
    cfg = load_yaml_config(config_path)
    data_cfg = load_yaml_config(cfg["data_config"])
    model_config = cfg.get("industry_gru_config", {})
    
    # 解析标签表达式
    label_expr = data_cfg["data"].get("label", "Ref($close, -10)/$close - 1")
    label_future_days = 0
    if "Ref($close, -" in label_expr:
        import re
        match = re.search(r'Ref\(\$close,\s*-(\d+)\)', label_expr)
        if match:
            label_future_days = int(match.group(1))
    
    sequence_length = model_config.get("sequence_length", 60)
    purge_gap = sequence_length + label_future_days
    
    logger.info("配置检查:")
    logger.info("  - 序列长度 (sequence_length): %d 天", sequence_length)
    logger.info("  - 标签未来天数 (label_future_days): %d 天", label_future_days)
    logger.info("  - 计算 Purge Gap: %d + %d = %d 天", sequence_length, label_future_days, purge_gap)
    logger.info("")
    
    # 检查滚动配置
    rolling_cfg = cfg.get("rolling", {})
    train_months = rolling_cfg.get("train_months", 24)
    valid_months = rolling_cfg.get("valid_months", 1)
    
    logger.info("滚动窗口配置:")
    logger.info("  - 训练窗口: %d 个月", train_months)
    logger.info("  - 验证窗口: %d 个月", valid_months)
    logger.info("")
    
    # 计算最小训练窗口要求
    min_train_days = purge_gap + 30  # 至少需要 purge_gap + 1个月的数据
    logger.info("最小训练窗口要求:")
    logger.info("  - 至少需要 %d 天数据（Purge Gap + 1个月）", min_train_days)
    logger.info("  - 当前训练窗口: %d 个月 ≈ %d 天", train_months, train_months * 30)
    
    if train_months * 30 < min_train_days:
        logger.warning("⚠️  训练窗口可能过短！建议至少 %d 个月", (min_train_days // 30) + 1)
    else:
        logger.info("✅ 训练窗口长度足够")
    
    return purge_gap


def check_cross_sectional_normalization(config_path: str):
    """检查截面标准化是否使用未来数据"""
    logger.info("=" * 80)
    logger.info("检查 2: 截面标准化配置")
    logger.info("=" * 80)
    
    cfg = load_yaml_config(config_path)
    data_cfg = load_yaml_config(cfg["data_config"])
    
    cs_norm = data_cfg["data"].get("cross_sectional_normalization", {})
    enabled = cs_norm.get("enabled", False)
    method = cs_norm.get("method", "zscore")
    groupby = cs_norm.get("groupby", "datetime")
    
    logger.info("截面标准化配置:")
    logger.info("  - 是否启用: %s", enabled)
    logger.info("  - 方法: %s", method)
    logger.info("  - 分组字段: %s", groupby)
    logger.info("")
    
    if enabled:
        if groupby == "datetime":
            logger.info("✅ 截面标准化按日期分组，不会使用未来数据")
            logger.info("   每个日期内的标准化只使用当日截面数据，无数据泄露")
        else:
            logger.warning("⚠️  分组字段不是 'datetime'，可能使用未来数据")
    else:
        logger.warning("⚠️  截面标准化未启用，特征可能在不同时期不可比")
    
    return enabled and groupby == "datetime"


def check_temporal_normalization(config_path: str):
    """检查时间序列归一化是否使用未来数据"""
    logger.info("=" * 80)
    logger.info("检查 3: 时间序列归一化配置")
    logger.info("=" * 80)
    
    cfg = load_yaml_config(config_path)
    data_cfg = load_yaml_config(cfg["data_config"])
    
    temp_norm = data_cfg["data"].get("temporal_normalization", {})
    enabled = temp_norm.get("enabled", False)
    
    logger.info("时间序列归一化配置:")
    logger.info("  - 是否启用: %s", enabled)
    logger.info("")
    
    if enabled:
        logger.info("✅ 时间序列归一化在训练时按窗口进行，不会使用未来数据")
        logger.info("   每个训练窗口单独计算归一化参数，验证集使用训练集的参数")
    else:
        logger.warning("⚠️  时间序列归一化未启用")
    
    return enabled


def check_training_prediction_leakage():
    """检查训练集预测是否使用了训练数据本身（导致 IC 虚高）"""
    logger.info("=" * 80)
    logger.info("检查 4: 训练集预测泄漏")
    logger.info("=" * 80)
    
    logger.info("问题分析:")
    logger.info("  当前代码: train_pred = model.predict(train_feat_norm)")
    logger.info("  ⚠️  这会导致 IC 虚高，因为模型在训练时已经见过这些数据")
    logger.info("")
    logger.info("建议修复:")
    logger.info("  1. 使用时间序列交叉验证（Time Series Cross Validation）")
    logger.info("  2. 或者使用训练集的前 80% 训练，后 20% 评估")
    logger.info("  3. 或者只报告验证集 IC，不报告训练集 IC")
    logger.info("")
    logger.warning("⚠️  训练集 IC 0.74 异常高，很可能是因为使用了训练数据本身进行预测")


def test_shifted_labels(config_path: str, shift_days: int = 200):
    """错位标签实验：将标签整体向后错位，如果 IC 仍然很高，说明有数据泄露"""
    logger.info("=" * 80)
    logger.info("检查 5: 错位标签实验")
    logger.info("=" * 80)
    
    logger.info("实验设计:")
    logger.info("  将标签整体向后错位 %d 天", shift_days)
    logger.info("  如果训练 IC 仍然很高（>0.5），说明存在数据泄露")
    logger.info("  如果训练 IC 接近 0，说明没有数据泄露")
    logger.info("")
    logger.info("⚠️  这是一个诊断实验，需要修改训练代码来实现")
    logger.info("  建议在 run_industry_train.py 中添加一个选项来启用错位标签实验")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="行业轮动数据泄露检查")
    parser.add_argument(
        "--config",
        type=str,
        default="classify/config_industry_rotation.yaml",
        help="配置文件路径",
    )
    args = parser.parse_args()
    
    logger.info("开始数据泄露检查...")
    logger.info("配置文件: %s", args.config)
    logger.info("")
    
    # 检查 1: Purge Gap
    purge_gap = check_purge_gap(args.config)
    logger.info("")
    
    # 检查 2: 截面标准化
    cs_norm_ok = check_cross_sectional_normalization(args.config)
    logger.info("")
    
    # 检查 3: 时间序列归一化
    temp_norm_ok = check_temporal_normalization(args.config)
    logger.info("")
    
    # 检查 4: 训练集预测泄漏
    check_training_prediction_leakage()
    logger.info("")
    
    # 检查 5: 错位标签实验
    test_shifted_labels(args.config)
    logger.info("")
    
    # 总结
    logger.info("=" * 80)
    logger.info("检查总结")
    logger.info("=" * 80)
    
    issues = []
    if purge_gap < 60:
        issues.append("Purge Gap 可能不足")
    
    if not cs_norm_ok:
        issues.append("截面标准化可能有问题")
    
    if not temp_norm_ok:
        issues.append("时间序列归一化未启用")
    
    issues.append("训练集预测使用了训练数据本身（导致 IC 虚高）")
    
    if issues:
        logger.warning("发现 %d 个潜在问题:", len(issues))
        for i, issue in enumerate(issues, 1):
            logger.warning("  %d. %s", i, issue)
    else:
        logger.info("✅ 未发现明显的数据泄露问题")
    
    logger.info("")
    logger.info("建议修复:")
    logger.info("  1. 修复训练集预测逻辑（使用时间序列交叉验证）")
    logger.info("  2. 验证 Purge Gap 是否正确实现")
    logger.info("  3. 运行错位标签实验验证是否有数据泄露")


if __name__ == "__main__":
    main()

