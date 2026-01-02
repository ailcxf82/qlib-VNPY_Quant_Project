"""
核实验证集的定义和数据情况
"""

import sys
from pathlib import Path
import pandas as pd

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils import load_yaml_config
from feature.qlib_feature_pipeline import QlibFeaturePipeline
import re

def analyze_validation_set():
    """分析验证集的实际数据情况"""
    
    # 加载配置
    cfg = load_yaml_config("classify/config_industry_rotation.yaml")
    data_cfg = load_yaml_config(cfg["data_config"])
    
    # 解析标签表达式
    label_expr = data_cfg["data"].get("label", "Ref($close, -10)/$close - 1")
    label_future_days = 0
    if "Ref($close, -" in label_expr:
        match = re.search(r'Ref\(\$close,\s*-(\d+)\)', label_expr)
        if match:
            label_future_days = int(match.group(1))
    
    print("=" * 80)
    print("验证集定义核验")
    print("=" * 80)
    print(f"\n标签表达式: {label_expr}")
    print(f"标签需要未来天数: {label_future_days}")
    
    # 构建数据
    pipeline = QlibFeaturePipeline(cfg["data_config"])
    pipeline.build()
    features, labels = pipeline.get_all()
    
    # 获取滚动配置
    rolling_cfg = cfg.get("rolling", {})
    train_months = rolling_cfg.get("train_months", 24)
    valid_months = rolling_cfg.get("valid_months", 1)
    
    # 生成第一个窗口进行分析
    start_time = data_cfg["data"]["start_time"]
    end_time = data_cfg["data"]["end_time"]
    
    start = pd.Timestamp(start_time)
    train_offset = pd.DateOffset(months=train_months)
    valid_offset = pd.DateOffset(months=valid_months)
    
    cursor = start + train_offset
    valid_start = cursor
    valid_end = cursor + valid_offset - pd.Timedelta(days=1)
    
    print(f"\n窗口 0 配置:")
    print(f"  验证窗口（原始）: {valid_start.strftime('%Y-%m-%d')} 到 {valid_end.strftime('%Y-%m-%d')}")
    print(f"  验证窗口（调整后，考虑标签未来数据）: {valid_start.strftime('%Y-%m-%d')} 到 {(valid_end - pd.Timedelta(days=label_future_days)).strftime('%Y-%m-%d')}")
    
    # 切片验证集
    valid_start_ts = valid_start
    valid_end_ts = valid_end - pd.Timedelta(days=label_future_days)
    
    datetime_level = features.index.get_level_values("datetime")
    mask = (datetime_level >= valid_start_ts) & (datetime_level <= valid_end_ts)
    valid_feat = features.loc[mask]
    
    if not valid_feat.empty:
        actual_start = valid_feat.index.get_level_values("datetime").min()
        actual_end = valid_feat.index.get_level_values("datetime").max()
        unique_dates = valid_feat.index.get_level_values("datetime").unique()
        unique_instruments = valid_feat.index.get_level_values("instrument").unique()
        
        print(f"\n验证集实际数据:")
        print(f"  实际日期范围: {actual_start} 到 {actual_end}")
        print(f"  唯一交易日数: {len(unique_dates)} 天")
        print(f"  唯一行业数: {len(unique_instruments)} 个")
        print(f"  总样本数: {len(valid_feat)} 条")
        print(f"  平均每天样本数: {len(valid_feat) / len(unique_dates):.1f} 条/天")
        
        # 检查每个行业的数据情况
        print(f"\n各行业数据情况（前10个）:")
        for inst in unique_instruments[:10]:
            inst_data = valid_feat.xs(inst, level=1, drop_level=False)
            if isinstance(inst_data.index, pd.MultiIndex):
                inst_data = inst_data.reset_index(level=1, drop=True)
            inst_dates = inst_data.index.unique()
            print(f"  {inst}: {len(inst_dates)} 个交易日")
        
        # 检查序列构建需求
        sequence_length = cfg["industry_gru_config"].get("sequence_length", 60)
        print(f"\n序列构建需求:")
        print(f"  模型需要: {sequence_length} 天历史数据")
        print(f"  验证集提供: {len(unique_dates)} 个交易日")
        print(f"  缺口: {sequence_length - len(unique_dates)} 天")
        
        # 检查是否可以使用训练集数据补充
        train_end = cursor - pd.Timedelta(days=1)
        train_end_ts = train_end - pd.Timedelta(days=label_future_days)
        
        # 合并训练集和验证集
        train_mask = (datetime_level >= start) & (datetime_level <= train_end_ts)
        train_feat = features.loc[train_mask]
        
        combined_feat = pd.concat([train_feat, valid_feat]).sort_index()
        
        print(f"\n使用训练集历史数据补充:")
        print(f"  训练集结束: {train_end_ts.strftime('%Y-%m-%d')}")
        print(f"  验证集开始: {valid_start_ts.strftime('%Y-%m-%d')}")
        print(f"  合并后数据范围: {combined_feat.index.get_level_values('datetime').min()} 到 {combined_feat.index.get_level_values('datetime').max()}")
        
        # 检查每个行业在合并后是否有足够数据
        sufficient_count = 0
        insufficient_count = 0
        for inst in unique_instruments:
            try:
                inst_data = combined_feat.xs(inst, level=1, drop_level=False)
                if isinstance(inst_data.index, pd.MultiIndex):
                    inst_data = inst_data.reset_index(level=1, drop=True)
                inst_data = inst_data.sort_index()
                
                # 检查验证集的第一个时间点是否有足够历史
                first_valid_date = valid_feat.xs(inst, level=1, drop_level=False)
                if isinstance(first_valid_date.index, pd.MultiIndex):
                    first_valid_date = first_valid_date.reset_index(level=1, drop=True)
                if not first_valid_date.empty:
                    first_date = first_valid_date.index.min()
                    if first_date in inst_data.index:
                        pos = inst_data.index.get_loc(first_date)
                        if pos >= sequence_length:
                            sufficient_count += 1
                        else:
                            insufficient_count += 1
            except:
                insufficient_count += 1
        
        print(f"\n行业数据充足性检查:")
        print(f"  数据充足的行业: {sufficient_count} 个")
        print(f"  数据不足的行业: {insufficient_count} 个")
        print(f"  充足率: {sufficient_count / len(unique_instruments) * 100:.1f}%")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    analyze_validation_set()


