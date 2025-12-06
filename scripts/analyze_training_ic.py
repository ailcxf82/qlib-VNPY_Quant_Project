"""分析训练日志中的IC值"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from pathlib import Path

# 设置UTF-8编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 添加项目根目录到路径
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

def analyze_training_ic(log_path: str = "data/logs/training_metrics.csv"):
    """分析训练日志中的IC值"""
    print("=" * 80)
    print("训练日志IC值分析")
    print("=" * 80)
    
    if not os.path.exists(log_path):
        print(f"❌ 训练日志文件不存在: {log_path}")
        return
    
    # 读取训练日志
    print(f"\n1. 加载训练日志: {log_path}")
    df = pd.read_csv(log_path)
    print(f"   日志行数: {len(df)}")
    print(f"   列名: {list(df.columns)}")
    
    # 检查IC列
    ic_columns = [col for col in df.columns if col.startswith('ic_')]
    if not ic_columns:
        print("   ❌ 未找到IC列")
        return
    
    print(f"   IC列: {ic_columns}")
    
    # 解析日期
    if 'valid_end' in df.columns:
        df['valid_end'] = pd.to_datetime(df['valid_end'])
        df = df.sort_values('valid_end')
    
    # 2. 整体IC统计
    print(f"\n2. 整体IC统计")
    for col in ic_columns:
        ic_values = df[col].dropna()
        if len(ic_values) > 0:
            print(f"\n   {col}:")
            print(f"     平均IC: {ic_values.mean():.4f}")
            print(f"     IC标准差: {ic_values.std():.4f}")
            print(f"     IC信息比率 (IR): {ic_values.mean() / (ic_values.std() + 1e-8):.4f}")
            print(f"     最小值: {ic_values.min():.4f}")
            print(f"     最大值: {ic_values.max():.4f}")
            print(f"     IC > 0 的比例: {(ic_values > 0).sum() / len(ic_values):.2%}")
            print(f"     IC > 0.05 的比例: {(ic_values > 0.05).sum() / len(ic_values):.2%}")
    
    # 3. IC时间序列分析
    print(f"\n3. IC时间序列分析")
    if 'valid_end' in df.columns:
        print(f"   日期范围: {df['valid_end'].min()} 到 {df['valid_end'].max()}")
        
        # 最近30个窗口的IC
        if len(df) >= 30:
            recent_df = df.tail(30)
            print(f"\n   最近30个训练窗口的IC:")
            for col in ic_columns:
                ic_values = recent_df[col].dropna()
                if len(ic_values) > 0:
                    print(f"     {col}: 平均={ic_values.mean():.4f}, 标准差={ic_values.std():.4f}, IR={ic_values.mean() / (ic_values.std() + 1e-8):.4f}")
        
        # 按年份分组统计
        if len(df) > 0:
            df['year'] = df['valid_end'].dt.year
            print(f"\n   按年份分组的IC统计:")
            for year in sorted(df['year'].unique()):
                year_df = df[df['year'] == year]
                print(f"\n     {year}年:")
                for col in ic_columns:
                    ic_values = year_df[col].dropna()
                    if len(ic_values) > 0:
                        print(f"       {col}: 平均={ic_values.mean():.4f}, 样本数={len(ic_values)}")
    
    # 4. 模型对比
    print(f"\n4. 模型IC对比")
    if len(ic_columns) > 1:
        model_ic_summary = []
        for col in ic_columns:
            ic_values = df[col].dropna()
            if len(ic_values) > 0:
                model_ic_summary.append({
                    'model': col.replace('ic_', ''),
                    'mean_ic': ic_values.mean(),
                    'std_ic': ic_values.std(),
                    'ir': ic_values.mean() / (ic_values.std() + 1e-8),
                    'positive_ratio': (ic_values > 0).sum() / len(ic_values),
                })
        
        if model_ic_summary:
            summary_df = pd.DataFrame(model_ic_summary)
            summary_df = summary_df.sort_values('mean_ic', ascending=False)
            print(f"\n   模型IC排名:")
            print(f"   {summary_df.to_string(index=False)}")
    
    # 5. IC趋势分析
    print(f"\n5. IC趋势分析")
    if 'valid_end' in df.columns and len(df) > 10:
        # 计算滚动平均IC
        window = min(20, len(df) // 2)
        for col in ic_columns:
            ic_values = df[col].dropna()
            if len(ic_values) >= window:
                rolling_mean = ic_values.rolling(window=window, min_periods=1).mean()
                recent_trend = rolling_mean.tail(10).mean() - rolling_mean.head(10).mean()
                print(f"   {col}:")
                print(f"     早期平均IC: {rolling_mean.head(10).mean():.4f}")
                print(f"     近期平均IC: {rolling_mean.tail(10).mean():.4f}")
                print(f"     趋势变化: {recent_trend:+.4f} ({'改善' if recent_trend > 0 else '恶化' if recent_trend < 0 else '稳定'})")
    
    # 6. 保存IC时间序列图
    if 'valid_end' in df.columns:
        print(f"\n6. 生成IC时间序列图")
        fig, axes = plt.subplots(len(ic_columns), 1, figsize=(12, 4 * len(ic_columns)))
        if len(ic_columns) == 1:
            axes = [axes]
        
        for idx, col in enumerate(ic_columns):
            ax = axes[idx]
            ic_values = df[col].dropna()
            dates = df.loc[ic_values.index, 'valid_end'] if 'valid_end' in df.columns else None
            
            if dates is not None and len(ic_values) > 0:
                ax.plot(dates, ic_values.values, label=col, alpha=0.7)
                # 添加滚动平均线
                window = min(20, len(ic_values) // 2)
                if len(ic_values) >= window:
                    rolling_mean = ic_values.rolling(window=window, min_periods=1).mean()
                    ax.plot(dates, rolling_mean.values, label=f'{col} (滚动平均)', linestyle='--', linewidth=2)
                # 添加零线
                ax.axhline(y=0, color='r', linestyle=':', alpha=0.5)
                ax.axhline(y=0.05, color='g', linestyle=':', alpha=0.5, label='IC=0.05阈值')
                ax.set_title(f'{col} 时间序列')
                ax.set_xlabel('日期')
                ax.set_ylabel('IC值')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = _project_root / "data" / "logs" / "ic_timeseries.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ IC时间序列图已保存: {output_path}")
        plt.close()
    
    print("\n" + "=" * 80)
    print("分析完成")
    print("=" * 80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="分析训练日志中的IC值")
    parser.add_argument("--log", type=str, default="data/logs/training_metrics.csv", help="训练日志文件路径")
    args = parser.parse_args()
    
    analyze_training_ic(args.log)


