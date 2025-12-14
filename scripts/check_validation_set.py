"""
检查验证集为空的问题

诊断：
1. 检查数据时间范围
2. 检查标签计算需要多少未来数据
3. 检查滚动窗口配置
4. 计算实际可用的验证集范围
"""

import os
import sys
from pathlib import Path
import pandas as pd

# 添加项目根目录到路径
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

from utils import load_yaml_config


def check_validation_set_issue():
    """检查验证集为空的问题"""
    print("=" * 80)
    print("验证集为空问题诊断")
    print("=" * 80)
    
    # 1. 读取配置
    pipeline_config = "config/pipeline.yaml"
    data_config = "config/data.yaml"
    
    pipeline_cfg = load_yaml_config(pipeline_config)
    data_cfg = load_yaml_config(data_config)
    
    # 2. 获取配置信息
    rolling = pipeline_cfg.get("rolling", {})
    train_months = rolling.get("train_months", 24)
    valid_months = rolling.get("valid_months", 1)
    step_months = rolling.get("step_months", 1)
    
    data_info = data_cfg.get("data", {})
    start_time = data_info.get("start_time", "2022-08-01")
    end_time = data_info.get("end_time", "2025-10-31")
    label_expr = data_info.get("label", "Ref($close, -5)/$close - 1")
    
    # 3. 解析标签表达式，获取需要的未来天数
    import re
    label_future_days = 0
    if "Ref($close, -" in label_expr:
        match = re.search(r'Ref\(\$close,\s*-(\d+)\)', label_expr)
        if match:
            label_future_days = int(match.group(1))
    
    print("\n【配置信息】")
    print(f"数据时间范围: {start_time} 至 {end_time}")
    print(f"训练窗口: {train_months} 个月")
    print(f"验证窗口: {valid_months} 个月")
    print(f"步长: {step_months} 个月")
    print(f"标签表达式: {label_expr}")
    print(f"标签需要未来数据: {label_future_days} 天")
    
    # 4. 计算实际可用的数据范围（考虑标签需要未来数据）
    start_ts = pd.Timestamp(start_time)
    end_ts = pd.Timestamp(end_time)
    
    # 由于标签需要未来N天，最后N天的数据没有标签
    if label_future_days > 0:
        available_end = end_ts - pd.Timedelta(days=label_future_days)
        print(f"\n【数据可用性分析】")
        print(f"数据结束时间: {end_ts.strftime('%Y-%m-%d')}")
        print(f"标签需要未来 {label_future_days} 天数据")
        print(f"实际可用数据结束时间: {available_end.strftime('%Y-%m-%d')}")
        print(f"最后 {label_future_days} 天的数据没有标签（无法用于训练/验证）")
    
    # 5. 模拟窗口生成
    print(f"\n【滚动窗口分析】")
    train_offset = pd.DateOffset(months=train_months)
    valid_offset = pd.DateOffset(months=valid_months)
    step = pd.DateOffset(months=step_months)
    
    cursor = start_ts + train_offset
    window_count = 0
    valid_windows = []
    invalid_windows = []
    
    while cursor + valid_offset <= end_ts:
        train_start = cursor - train_offset
        train_end = cursor - pd.Timedelta(days=1)
        valid_start = cursor
        valid_end = cursor + valid_offset - pd.Timedelta(days=1)
        
        # 检查验证集是否有标签数据
        # 验证集需要未来 label_future_days 天来计算标签
        valid_end_with_label = valid_end - pd.Timedelta(days=label_future_days) if label_future_days > 0 else valid_end
        
        if valid_end_with_label >= valid_start:
            # 验证集有部分数据可用
            actual_valid_end = min(valid_end_with_label, available_end if label_future_days > 0 else valid_end)
            valid_windows.append({
                "window": window_count,
                "train": f"{train_start.strftime('%Y-%m-%d')} 至 {train_end.strftime('%Y-%m-%d')}",
                "valid": f"{valid_start.strftime('%Y-%m-%d')} 至 {actual_valid_end.strftime('%Y-%m-%d')}",
                "valid_original": f"{valid_start.strftime('%Y-%m-%d')} 至 {valid_end.strftime('%Y-%m-%d')}",
            })
        else:
            # 验证集完全没有标签数据
            invalid_windows.append({
                "window": window_count,
                "train": f"{train_start.strftime('%Y-%m-%d')} 至 {train_end.strftime('%Y-%m-%d')}",
                "valid": f"{valid_start.strftime('%Y-%m-%d')} 至 {valid_end.strftime('%Y-%m-%d')}",
                "reason": f"验证集结束日期 {valid_end.strftime('%Y-%m-%d')} 需要未来 {label_future_days} 天数据，但数据只到 {end_ts.strftime('%Y-%m-%d')}",
            })
        
        window_count += 1
        cursor += step
    
    print(f"总窗口数: {window_count}")
    print(f"有效验证集窗口: {len(valid_windows)}")
    print(f"无效验证集窗口: {len(invalid_windows)}")
    
    if invalid_windows:
        print(f"\n【问题窗口详情】")
        for win in invalid_windows[:5]:  # 只显示前5个
            print(f"  窗口 {win['window']}:")
            print(f"    训练: {win['train']}")
            print(f"    验证: {win['valid']}")
            print(f"    原因: {win['reason']}")
        if len(invalid_windows) > 5:
            print(f"  ... 还有 {len(invalid_windows) - 5} 个问题窗口")
    
    if valid_windows:
        print(f"\n【有效窗口示例】")
        for win in valid_windows[:3]:  # 只显示前3个
            print(f"  窗口 {win['window']}:")
            print(f"    训练: {win['train']}")
            print(f"    验证: {win['valid']}")
            if win['valid'] != win['valid_original']:
                print(f"    注意: 原始验证范围 {win['valid_original']} 被截断（缺少未来数据）")
    
    # 6. 诊断和建议
    print(f"\n【问题诊断】")
    if label_future_days > 0:
        print(f"✓ 标签需要未来 {label_future_days} 天数据")
        print(f"✓ 数据结束时间: {end_ts.strftime('%Y-%m-%d')}")
        print(f"✓ 实际可用数据结束时间: {available_end.strftime('%Y-%m-%d')}")
        print(f"✓ 最后 {label_future_days} 天的数据无法计算标签")
    
    if invalid_windows:
        print(f"\n⚠ 发现 {len(invalid_windows)} 个窗口的验证集为空")
        print(f"  原因: 验证集需要未来 {label_future_days} 天数据来计算标签，但数据不足")
    
    print(f"\n【解决方案】")
    print(f"方案 1: 调整数据结束时间（推荐）")
    print(f"  将 end_time 提前 {label_future_days} 天，确保所有数据都有标签")
    print(f"  建议: end_time: \"{available_end.strftime('%Y-%m-%d')}\"")
    
    print(f"\n方案 2: 修改 _slice 方法，自动处理标签缺失")
    print(f"  在切片时，验证集自动排除最后 {label_future_days} 天")
    print(f"  位置: trainer/trainer.py::_slice()")
    
    print(f"\n方案 3: 调整滚动窗口配置")
    print(f"  减少 valid_months 或调整窗口生成逻辑")
    print(f"  确保验证集结束日期 <= 数据结束日期 - {label_future_days} 天")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    check_validation_set_issue()

