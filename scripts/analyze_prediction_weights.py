"""
分析预测结果中各模型的权重分配和贡献
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 设置UTF-8编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 添加项目根目录到路径
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

def analyze_prediction_weights(pred_file: str = None, log_file: str = "data/logs/training_metrics.csv"):
    """分析预测结果中的权重分配"""
    print("=" * 80)
    print("预测结果权重分析")
    print("=" * 80)
    
    # 1. 查找最新的预测文件
    if pred_file is None:
        pred_dir = Path("data/predictions")
        if not pred_dir.exists():
            print(f"❌ 预测目录不存在: {pred_dir}")
            return
        
        pred_files = list(pred_dir.glob("pred_*.csv"))
        if not pred_files:
            print(f"❌ 未找到预测文件")
            return
        
        # 选择最新的预测文件（排除 rqalpha_ 前缀的）
        pred_files = [f for f in pred_files if not f.name.startswith("rqalpha_")]
        if not pred_files:
            print(f"❌ 未找到标准预测文件")
            return
        
        pred_file = max(pred_files, key=os.path.getmtime)
        print(f"\n使用预测文件: {pred_file.name}")
    else:
        pred_file = Path(pred_file)
    
    # 2. 读取预测结果
    print(f"\n1. 读取预测结果: {pred_file}")
    try:
        df = pd.read_csv(pred_file, index_col=[0, 1])
        df.index.names = ['datetime', 'instrument']
    except Exception as e:
        print(f"   ❌ 读取失败: {e}")
        return
    
    print(f"   总样本数: {len(df):,}")
    print(f"   日期范围: {df.index.get_level_values('datetime').min()} 到 {df.index.get_level_values('datetime').max()}")
    print(f"   股票数量: {df.index.get_level_values('instrument').nunique()}")
    print(f"   列名: {list(df.columns)}")
    
    # 3. 分析各模型的预测值
    print(f"\n2. 各模型预测值统计")
    model_cols = [col for col in df.columns if col != 'final']
    if not model_cols:
        print("   ❌ 未找到模型预测列（只有 final 列）")
        return
    
    for col in model_cols + ['final']:
        values = df[col].dropna()
        if len(values) > 0:
            print(f"\n   {col}:")
            print(f"     样本数: {len(values):,}")
            print(f"     均值: {values.mean():.6f}")
            print(f"     标准差: {values.std():.6f}")
            print(f"     最小值: {values.min():.6f}")
            print(f"     最大值: {values.max():.6f}")
            print(f"     唯一值: {values.nunique():,} / {len(values):,} ({values.nunique()/len(values)*100:.2f}%)")
    
    # 4. 分析模型相关性
    print(f"\n3. 模型预测相关性")
    corr_matrix = df[model_cols + ['final']].corr()
    print("\n   相关系数矩阵:")
    print(corr_matrix.round(4))
    
    # 5. 分析权重分配（通过相关性推断）
    print(f"\n4. 权重分配分析（基于相关性推断）")
    if 'final' in df.columns:
        final = df['final']
        for col in model_cols:
            model_pred = df[col]
            # 计算与 final 的相关性
            corr = final.corr(model_pred)
            # 计算平均差异
            diff = (final - model_pred).abs().mean()
            # 计算贡献度（通过回归系数近似）
            try:
                from sklearn.linear_model import LinearRegression
                X = model_pred.values.reshape(-1, 1)
                y = final.values
                mask = ~(np.isnan(X.flatten()) | np.isnan(y))
                if mask.sum() > 10:
                    reg = LinearRegression().fit(X[mask], y[mask])
                    coef = reg.coef_[0]
                    intercept = reg.intercept_
                    print(f"\n   {col}:")
                    print(f"     与 final 的相关系数: {corr:.4f}")
                    print(f"     回归系数（近似权重）: {coef:.4f}")
                    print(f"     截距: {intercept:.4f}")
                    print(f"     平均绝对差异: {diff:.6f}")
                else:
                    print(f"\n   {col}: 样本不足，无法计算回归系数")
            except Exception as e:
                print(f"\n   {col}: 计算失败 - {e}")
    
    # 6. 读取训练日志，分析IC历史
    print(f"\n5. 训练日志IC分析")
    if os.path.exists(log_file):
        try:
            log_df = pd.read_csv(log_file)
            if 'valid_end' in log_df.columns:
                log_df['valid_end'] = pd.to_datetime(log_df['valid_end'])
                log_df = log_df.sort_values('valid_end')
            
            ic_cols = [col for col in log_df.columns if col.startswith('ic_')]
            if ic_cols:
                print(f"\n   训练窗口数: {len(log_df)}")
                print(f"\n   各模型IC统计（最近 {min(20, len(log_df))} 个窗口）:")
                recent_df = log_df.tail(20)
                for col in ic_cols:
                    ic_values = recent_df[col].dropna()
                    if len(ic_values) > 0:
                        model_name = col.replace('ic_', '')
                        print(f"\n     {model_name}:")
                        print(f"       平均IC: {ic_values.mean():.4f}")
                        print(f"       标准差: {ic_values.std():.4f}")
                        print(f"       最大IC: {ic_values.max():.4f}")
                        print(f"       最小IC: {ic_values.min():.4f}")
                        print(f"       正IC比例: {(ic_values > 0).sum() / len(ic_values) * 100:.1f}%")
                
                # 计算IC-IR（信息比率）
                print(f"\n   IC-IR（信息比率，最近 {min(20, len(log_df))} 个窗口）:")
                for col in ic_cols:
                    ic_values = recent_df[col].dropna()
                    if len(ic_values) > 1:
                        model_name = col.replace('ic_', '')
                        mean_ic = ic_values.mean()
                        std_ic = ic_values.std()
                        ic_ir = mean_ic / std_ic if std_ic > 0 else 0
                        print(f"     {model_name}: {ic_ir:.4f} (均值={mean_ic:.4f}, 标准差={std_ic:.4f})")
            else:
                print("   ❌ 未找到IC列")
        except Exception as e:
            print(f"   ❌ 读取训练日志失败: {e}")
    else:
        print(f"   ⚠️  训练日志不存在: {log_file}")
    
    # 7. 分析MLP的贡献
    print(f"\n6. MLP模型贡献分析")
    if 'mlp' in [col.lower() for col in model_cols]:
        mlp_col = [col for col in model_cols if 'mlp' in col.lower()][0]
        mlp_pred = df[mlp_col]
        final_pred = df['final']
        
        # 计算MLP与final的差异
        diff = (final_pred - mlp_pred).abs()
        print(f"\n   MLP预测与最终预测的差异:")
        print(f"     平均绝对差异: {diff.mean():.6f}")
        print(f"     中位数差异: {diff.median():.6f}")
        print(f"     最大差异: {diff.max():.6f}")
        
        # 计算MLP的贡献度（如果final是加权平均）
        if 'lgb' in [col.lower() for col in model_cols]:
            lgb_col = [col for col in model_cols if 'lgb' in col.lower()][0]
            lgb_pred = df[lgb_col]
            
            # 尝试推断权重
            try:
                from sklearn.linear_model import LinearRegression
                X = pd.DataFrame({
                    'lgb': lgb_pred,
                    'mlp': mlp_pred
                })
                y = final_pred
                mask = ~(X.isna().any(axis=1) | y.isna())
                if mask.sum() > 10:
                    reg = LinearRegression(fit_intercept=False).fit(X[mask], y[mask])
                    weights = reg.coef_
                    print(f"\n   推断的权重（通过回归）:")
                    print(f"     LGB权重: {weights[0]:.4f}")
                    print(f"     MLP权重: {weights[1]:.4f}")
                    print(f"     权重和: {weights.sum():.4f}")
                    
                    if abs(weights[1]) < 0.1:
                        print(f"\n   ⚠️  MLP权重很小（{weights[1]:.4f}），贡献度较低")
                    elif abs(weights[1]) > 0.4:
                        print(f"\n   ✅ MLP权重较大（{weights[1]:.4f}），贡献度较高")
                    else:
                        print(f"\n   ℹ️  MLP权重中等（{weights[1]:.4f}），有一定贡献")
            except Exception as e:
                print(f"   推断权重失败: {e}")
    
    # 8. 总结
    print(f"\n7. 总结")
    print(f"\n   模型列表: {model_cols}")
    print(f"\n   建议:")
    if 'mlp' in [col.lower() for col in model_cols]:
        print(f"     - 检查MLP模型的IC历史表现")
        print(f"     - 如果MLP的IC较低或波动大，考虑调整MLP模型配置")
        print(f"     - 如果MLP的IC-IR较低，考虑降低MLP的权重或移除MLP")
    print(f"     - 查看训练日志中的IC-IR，了解各模型的稳定性")
    print(f"     - 根据IC-IR调整权重配置（config/pipeline.yaml 中的 ic_logging）")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="分析预测结果中的权重分配")
    parser.add_argument("--pred", type=str, default=None, help="预测结果文件路径（默认自动查找最新）")
    parser.add_argument("--log", type=str, default="data/logs/training_metrics.csv", help="训练日志路径")
    args = parser.parse_args()
    
    analyze_prediction_weights(args.pred, args.log)


