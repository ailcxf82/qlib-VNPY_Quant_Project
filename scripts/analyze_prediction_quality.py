"""分析预测质量：计算预测值与实际收益的IC值"""
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

from feature.qlib_feature_pipeline import QlibFeaturePipeline
from utils import load_yaml_config

def compute_ic(pred: pd.Series, label: pd.Series) -> float:
    """计算Spearman秩相关系数（IC）"""
    aligned = pred.align(label, join="inner")
    if aligned[0].empty:
        return np.nan
    return aligned[0].rank().corr(aligned[1].rank(), method="spearman")

def analyze_prediction_quality(prediction_file: str, config_file: str = "config/pipeline.yaml"):
    """分析预测质量"""
    print("=" * 80)
    print("预测质量分析")
    print("=" * 80)
    
    # 1. 加载预测文件
    print(f"\n1. 加载预测文件: {prediction_file}")
    df_pred = pd.read_csv(prediction_file, index_col=[0, 1])
    # 确保日期是Timestamp格式
    if isinstance(df_pred.index, pd.MultiIndex):
        dates = df_pred.index.get_level_values(0)
        if not isinstance(dates[0], pd.Timestamp):
            # 如果是字符串，转换为Timestamp
            dates_converted = pd.to_datetime(dates)
            codes = df_pred.index.get_level_values(1)
            df_pred.index = pd.MultiIndex.from_arrays([dates_converted, codes], names=df_pred.index.names)
    print(f"   预测文件行数: {len(df_pred)}")
    print(f"   日期范围: {df_pred.index.get_level_values(0).min()} 到 {df_pred.index.get_level_values(0).max()}")
    print(f"   预测值统计:")
    print(f"     {df_pred['final'].describe()}")
    
    # 2. 加载实际标签
    print(f"\n2. 加载实际标签")
    cfg = load_yaml_config(config_file)
    pipeline = QlibFeaturePipeline(cfg["data_config"])
    pipeline.build()
    _, labels = pipeline.get_all()
    
    # 检查索引格式
    print(f"   预测文件索引格式: {type(df_pred.index)}")
    if isinstance(df_pred.index, pd.MultiIndex):
        print(f"   预测文件索引层级: {df_pred.index.names}")
        pred_dates = df_pred.index.get_level_values(0).unique()[:5]
        pred_instruments = df_pred.index.get_level_values(1).unique()[:5]
        print(f"   预测文件日期示例: {list(pred_dates)}")
        print(f"   预测文件股票代码示例: {list(pred_instruments)}")
    
    print(f"   标签数据索引格式: {type(labels.index)}")
    if isinstance(labels.index, pd.MultiIndex):
        print(f"   标签数据索引层级: {labels.index.names}")
        label_dates = labels.index.get_level_values(0).unique()[:5]
        label_instruments = labels.index.get_level_values(1).unique()[:5]
        print(f"   标签数据日期示例: {list(label_dates)}")
        print(f"   标签数据股票代码示例: {list(label_instruments)}")
    
    # 对齐预测和标签
    # 检查索引格式并转换
    print(f"   检查索引格式...")
    pred_index = df_pred.index
    label_index = labels.index
    
    if isinstance(pred_index, pd.MultiIndex) and isinstance(label_index, pd.MultiIndex):
        pred_dates = pred_index.get_level_values(0)
        pred_codes = pred_index.get_level_values(1)
        label_dates = label_index.get_level_values(0)
        label_codes = label_index.get_level_values(1)
        
        # 检查代码格式
        label_code_sample = str(label_codes[0]) if len(label_codes) > 0 else ""
        pred_code_sample = str(pred_codes[0]) if len(pred_codes) > 0 else ""
        
        print(f"   标签代码示例: {label_code_sample}")
        print(f"   预测代码示例: {pred_code_sample}")
        
        # 转换预测代码格式以匹配标签
        # 情况1: 标签代码是6位字符串（如'000001'），预测代码是整数（如1）
        if label_code_sample.isdigit() and len(label_code_sample) == 6:
            # 检查预测代码是否需要转换（可能是整数或短字符串）
            pred_code_str = str(pred_codes[0]) if len(pred_codes) > 0 else ""
            if not pred_code_str.isdigit() or len(pred_code_str) != 6:
                print(f"   将预测代码转换为6位字符串格式...")
                def format_code(code):
                    return str(code).zfill(6)
                
                pred_codes_converted = pd.Series(pred_codes).apply(format_code)
                pred_index_new = pd.MultiIndex.from_arrays([pred_dates, pred_codes_converted], names=pred_index.names)
                df_pred.index = pred_index_new
                # 更新pred_codes和pred_code_sample用于后续判断
                pred_codes = pred_codes_converted
                pred_code_sample = str(pred_codes[0]) if len(pred_codes) > 0 else ""
                print(f"   转换后预测代码示例: {pred_code_sample}")
        
        # 情况2: 标签代码包含 SH/SZ 前缀，预测代码没有
        if label_code_sample.startswith(("SH", "SZ")) and not pred_code_sample.startswith(("SH", "SZ")):
            print(f"   为预测代码添加 SH/SZ 前缀...")
            def add_prefix(code):
                code_str = str(code).zfill(6)  # 确保6位数字
                if code_str.startswith(("600", "601", "603", "688", "5")):
                    return f"SH{code_str}"
                elif code_str.startswith(("000", "001", "002", "300", "159")):
                    return f"SZ{code_str}"
                else:
                    return code_str
            
            pred_codes_converted = pd.Series(pred_codes).apply(add_prefix)
            pred_index_new = pd.MultiIndex.from_arrays([pred_dates, pred_codes_converted], names=pred_index.names)
            df_pred.index = pred_index_new
        
        # 情况3: 预测代码包含 SH/SZ 前缀，标签代码没有
        elif pred_code_sample.startswith(("SH", "SZ")) and not label_code_sample.startswith(("SH", "SZ")):
            print(f"   移除预测代码的 SH/SZ 前缀...")
            def remove_prefix(code):
                code_str = str(code)
                if code_str.startswith("SH"):
                    return code_str[2:]
                elif code_str.startswith("SZ"):
                    return code_str[2:]
                else:
                    return code_str
            
            pred_codes_converted = pd.Series(pred_codes).apply(remove_prefix)
            pred_index_new = pd.MultiIndex.from_arrays([pred_dates, pred_codes_converted], names=pred_index.names)
            df_pred.index = pred_index_new
    
    # 对齐
    # 添加调试信息
    print(f"   对齐前检查:")
    if isinstance(df_pred.index, pd.MultiIndex) and isinstance(labels.index, pd.MultiIndex):
        pred_dates = df_pred.index.get_level_values(0)
        pred_codes = df_pred.index.get_level_values(1)
        label_dates = labels.index.get_level_values(0)
        label_codes = labels.index.get_level_values(1)
        
        print(f"     预测日期类型: {type(pred_dates[0])}, 示例: {pred_dates[0]}")
        print(f"     标签日期类型: {type(label_dates[0])}, 示例: {label_dates[0]}")
        print(f"     预测代码类型: {type(pred_codes[0])}, 示例: {pred_codes[0]}")
        print(f"     标签代码类型: {type(label_codes[0])}, 示例: {label_codes[0]}")
        
        # 检查日期范围重叠
        pred_date_range = (pred_dates.min(), pred_dates.max())
        label_date_range = (label_dates.min(), label_dates.max())
        print(f"     预测日期范围: {pred_date_range[0]} 到 {pred_date_range[1]}")
        print(f"     标签日期范围: {label_date_range[0]} 到 {label_date_range[1]}")
        
        # 检查代码重叠
        pred_codes_set = set(str(c) for c in pred_codes.unique()[:10])
        label_codes_set = set(str(c) for c in label_codes.unique()[:10])
        print(f"     预测代码示例: {pred_codes_set}")
        print(f"     标签代码示例: {label_codes_set}")
        print(f"     代码示例重叠: {pred_codes_set & label_codes_set}")
    
    common_index = df_pred.index.intersection(labels.index)
    pred_aligned = df_pred.loc[common_index, "final"] if len(common_index) > 0 else pd.Series(dtype=float)
    label_aligned = labels.loc[common_index] if len(common_index) > 0 else pd.Series(dtype=float)
    
    print(f"   标签数据行数: {len(labels)}")
    print(f"   对齐后数据行数: {len(pred_aligned)}")
    if len(pred_aligned) > 0:
        print(f"   标签值统计:")
        print(f"     {label_aligned.describe()}")
    else:
        print(f"   ❌ 对齐失败，无法计算IC值")
        print(f"   请检查预测文件和标签数据的股票代码格式是否一致")
    
    # 3. 计算整体IC
    print(f"\n3. 整体IC分析")
    overall_ic = compute_ic(pred_aligned, label_aligned)
    print(f"   整体IC: {overall_ic:.4f}")
    if overall_ic > 0.05:
        print("   ✅ IC > 0.05，预测方向正确")
    elif overall_ic > 0:
        print("   ⚠️  IC > 0 但 < 0.05，预测能力较弱")
    elif overall_ic < 0:
        print("   ❌ IC < 0，预测方向错误")
    else:
        print("   ⚠️  IC ≈ 0，预测无方向性")
    
    # 4. 按日期分组计算IC
    print(f"\n4. 按日期分组的IC分析")
    dates = pred_aligned.index.get_level_values(0).unique()
    ic_by_date = []
    
    for dt in sorted(dates):
        try:
            pred_dt = pred_aligned.xs(dt, level=0)
            label_dt = label_aligned.xs(dt, level=0)
            ic = compute_ic(pred_dt, label_dt)
            if not np.isnan(ic):
                ic_by_date.append({"date": dt, "ic": ic})
        except Exception as e:
            continue
    
    if ic_by_date:
        df_ic = pd.DataFrame(ic_by_date)
        print(f"   有效IC日期数: {len(df_ic)}")
        print(f"   IC统计:")
        print(f"     {df_ic['ic'].describe()}")
        print(f"   平均IC: {df_ic['ic'].mean():.4f}")
        print(f"   IC标准差: {df_ic['ic'].std():.4f}")
        print(f"   IC信息比率 (IR): {df_ic['ic'].mean() / (df_ic['ic'].std() + 1e-8):.4f}")
        
        # IC为正的比例
        positive_ic_ratio = (df_ic['ic'] > 0).sum() / len(df_ic)
        print(f"   IC为正的比例: {positive_ic_ratio:.2%}")
        
        # 最近30天的IC
        if len(df_ic) >= 30:
            recent_ic = df_ic.tail(30)['ic']
            print(f"\n   最近30天IC:")
            print(f"     平均IC: {recent_ic.mean():.4f}")
            print(f"     IC标准差: {recent_ic.std():.4f}")
            print(f"     IC为正的比例: {(recent_ic > 0).sum() / len(recent_ic):.2%}")
    
    # 5. 按预测值分组的收益分析
    print(f"\n5. 按预测值分组的收益分析")
    # 将预测值分为5组
    pred_quantiles = pd.qcut(pred_aligned, q=5, labels=False, duplicates='drop')
    label_by_quantile = []
    for q in range(5):
        mask = pred_quantiles == q
        if mask.sum() > 0:
            label_q = label_aligned[mask]
            label_by_quantile.append({
                "quantile": q,
                "count": len(label_q),
                "mean_return": label_q.mean(),
                "median_return": label_q.median(),
            })
    
    if label_by_quantile:
        df_quantile = pd.DataFrame(label_by_quantile)
        print(f"   分组收益:")
        print(f"     {df_quantile.to_string(index=False)}")
        
        # 检查单调性
        if len(df_quantile) == 5:
            returns = df_quantile['mean_return'].values
            is_monotonic = all(returns[i] <= returns[i+1] for i in range(4))
            if is_monotonic:
                print(f"   ✅ 收益随预测值单调递增，预测有效")
            else:
                print(f"   ⚠️  收益未随预测值单调递增，预测可能无效")
    
    print("\n" + "=" * 80)
    print("分析完成")
    print("=" * 80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="分析预测质量")
    parser.add_argument("--prediction", type=str, default=None, help="预测文件路径")
    parser.add_argument("--config", type=str, default="config/pipeline.yaml", help="配置文件路径")
    args = parser.parse_args()
    
    if args.prediction is None:
        # 自动查找最新预测文件
        prediction_dir = _project_root / "data" / "predictions"
        pred_files = list(prediction_dir.glob("pred_*.csv"))
        if not pred_files:
            print("错误：未找到预测文件")
            sys.exit(1)
        prediction_file = max(pred_files, key=lambda p: p.stat().st_mtime)
        print(f"自动选择最新预测文件: {prediction_file}")
    else:
        prediction_file = args.prediction
    
    analyze_prediction_quality(str(prediction_file), args.config)

