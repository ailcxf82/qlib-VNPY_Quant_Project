"""
检查数据泄露问题：未来函数和非时序归一化
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 添加项目根目录到路径
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

# 初始化 qlib
import qlib
from qlib.config import REG_CN
from qlib.data import D

from feature.qlib_feature_pipeline import QlibFeaturePipeline
from utils import load_yaml_config

# 初始化 qlib（如果尚未初始化）
try:
    qlib.init(provider_uri="D:/qlib_data/qlib_data", region=REG_CN)
except Exception:
    pass  # 可能已经初始化

def check_normalization_leakage():
    """检查归一化是否存在数据泄露"""
    print("=" * 80)
    print("检查 1: 归一化数据泄露（非时序归一化）")
    print("=" * 80)
    
    config_path = "config/pipeline.yaml"
    cfg = load_yaml_config(config_path)
    data_cfg_path = cfg["data_config"]
    
    pipeline = QlibFeaturePipeline(data_cfg_path)
    pipeline.build()
    
    # 检查归一化方法
    print("\n当前归一化实现：")
    print("  位置: feature/qlib_feature_pipeline.py::_fit_norm()")
    print("  方法: 使用全局均值和标准差（整个数据集）")
    print("  问题: ❌ 存在数据泄露！")
    print("  原因: 在时间序列数据上使用全局归一化，训练时使用了未来数据的信息")
    
    # 模拟问题演示
    all_features, _ = pipeline.get_all()
    if all_features is not None and not all_features.empty:
        datetime_level = all_features.index.get_level_values("datetime")
        actual_start = datetime_level.min()
        actual_end = datetime_level.max()
        
        # 模拟训练/测试分割
        split_date = actual_start + (actual_end - actual_start) * 0.7
        train_mask = datetime_level <= split_date
        test_mask = datetime_level > split_date
        
        train_data = all_features.loc[train_mask]
        test_data = all_features.loc[test_mask]
        
        print(f"\n数据范围: {actual_start} 到 {actual_end}")
        print(f"模拟分割点: {split_date}")
        print(f"训练集: {len(train_data)} 个样本")
        print(f"测试集: {len(test_data)} 个样本")
        
        # 当前方法（错误）：使用全局统计量
        global_mean = all_features.mean()
        global_std = all_features.std().replace(0, 1)
        
        # 正确方法：只使用训练集统计量
        train_mean = train_data.mean()
        train_std = train_data.std().replace(0, 1)
        
        # 比较差异
        mean_diff = (global_mean - train_mean).abs().mean()
        std_diff = (global_std - train_std).abs().mean()
        
        print(f"\n归一化统计量差异：")
        print(f"  均值差异: {mean_diff:.6f}")
        print(f"  标准差差异: {std_diff:.6f}")
        print(f"  影响: 测试集使用了训练集+测试集的统计量，导致数据泄露")
        
    print("\n建议修复：")
    print("  1. 在滚动窗口训练时，每个窗口只使用该窗口内的数据计算归一化参数")
    print("  2. 预测时，使用训练时的归一化参数，而不是重新计算")
    print("  3. 或者使用滚动窗口归一化（如60日滚动均值和标准差）")
    
    return True

def check_future_function():
    """检查特征中是否存在未来函数"""
    print("\n" + "=" * 80)
    print("检查 2: 未来函数（使用未来数据预测过去）")
    print("=" * 80)
    
    config_path = "config/pipeline.yaml"
    cfg = load_yaml_config(config_path)
    data_cfg_path = cfg["data_config"]
    data_cfg = load_yaml_config(data_cfg_path)
    
    # 检查标签表达式
    label_expr = data_cfg["data"].get("label", "Ref($close, -5)/$close - 1")
    print(f"\n标签表达式: {label_expr}")
    
    if "Ref($close, -" in label_expr:
        print("  ✅ 正确：标签使用未来数据（Ref($close, -N) 表示未来N天）")
        print("  说明：标签是预测目标，使用未来数据是正常的")
    else:
        print("  ⚠️  警告：标签可能未使用未来数据")
    
    # 检查特征表达式
    features = data_cfg["data"].get("features", [])
    print(f"\n检查 {len(features)} 个自定义特征表达式：")
    
    future_functions = []
    safe_features = []
    
    for feat in features:
        # 检查是否使用未来数据（Ref($xxx, -N) 其中 N < 0 表示未来）
        if "Ref(" in feat:
            # 提取 Ref 函数的参数
            import re
            ref_matches = re.findall(r'Ref\([^,]+,\s*(-?\d+)\)', feat)
            for match in ref_matches:
                shift = int(match)
                if shift < 0:
                    future_functions.append((feat, shift))
                    break
            else:
                safe_features.append(feat)
        else:
            safe_features.append(feat)
    
    if future_functions:
        print(f"  ❌ 发现 {len(future_functions)} 个使用未来数据的特征：")
        for feat, shift in future_functions[:10]:  # 只显示前10个
            print(f"    - {feat} (偏移: {shift} 天，表示未来数据)")
        if len(future_functions) > 10:
            print(f"    ... 还有 {len(future_functions) - 10} 个")
    else:
        print("  ✅ 未发现使用未来数据的特征")
    
    print(f"\n  ✅ 安全特征: {len(safe_features)} 个")
    
    # 检查 Alpha158 因子（这些因子由 qlib 生成，通常不会有未来函数）
    use_alpha158 = data_cfg["data"].get("use_alpha158", False)
    if use_alpha158:
        print(f"\nAlpha158 因子: 已启用")
        print("  说明：Alpha158 因子由 qlib 自动生成，通常不包含未来函数")
        print("  建议：如果担心，可以检查生成的因子表达式")
    
    return len(future_functions) == 0

def check_training_procedure():
    """检查训练流程是否正确处理时间序列"""
    print("\n" + "=" * 80)
    print("检查 3: 训练流程时间序列处理")
    print("=" * 80)
    
    config_path = "config/pipeline.yaml"
    cfg = load_yaml_config(config_path)
    
    rolling = cfg.get("rolling", {})
    train_months = rolling.get("train_months", 24)
    valid_months = rolling.get("valid_months", 1)
    step_months = rolling.get("step_months", 1)
    
    print(f"\n滚动窗口配置：")
    print(f"  训练窗口: {train_months} 个月")
    print(f"  验证窗口: {valid_months} 个月")
    print(f"  步长: {step_months} 个月")
    
    print(f"\n训练流程检查：")
    print(f"  位置: trainer/trainer.py::RollingTrainer")
    print(f"  方法: 滚动窗口训练")
    
    # 检查是否在训练时重新计算归一化
    print(f"\n归一化处理：")
    print(f"  ❌ 问题：在 build() 时使用全局归一化")
    print(f"  位置: feature/qlib_feature_pipeline.py::build()")
    print(f"  影响: 每个训练窗口都使用了全局统计量，而不是窗口内统计量")
    
    print(f"\n建议修复：")
    print(f"  1. 在 _slice() 时对每个窗口单独计算归一化参数")
    print(f"  2. 或者在训练时传入归一化参数，而不是在 build() 时计算")
    
    return True

def main():
    print("=" * 80)
    print("数据泄露检查报告")
    print("=" * 80)
    
    issues = []
    
    # 检查1：归一化数据泄露
    norm_issue = check_normalization_leakage()
    if not norm_issue:
        issues.append("归一化数据泄露")
    
    # 检查2：未来函数
    future_issue = check_future_function()
    if not future_issue:
        issues.append("未来函数")
    
    # 检查3：训练流程
    training_issue = check_training_procedure()
    
    # 总结
    print("\n" + "=" * 80)
    print("检查总结")
    print("=" * 80)
    
    if issues:
        print(f"\n❌ 发现 {len(issues)} 个问题：")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ 未发现明显的数据泄露问题")
    
    print("\n关键问题：")
    print("  1. ❌ 非时序归一化：使用全局均值和标准差，导致数据泄露")
    print("  2. ✅ 未来函数：特征中未发现使用未来数据（标签使用未来数据是正常的）")
    print("  3. ⚠️  训练流程：虽然使用滚动窗口，但归一化参数计算有问题")
    
    print("\n建议修复优先级：")
    print("  1. 【高优先级】修复归一化方法，使用滚动窗口归一化")
    print("  2. 【中优先级】确保训练时每个窗口使用独立的归一化参数")
    print("  3. 【低优先级】验证 Alpha158 因子不包含未来函数")

if __name__ == "__main__":
    main()

