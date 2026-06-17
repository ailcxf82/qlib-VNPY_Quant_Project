"""分析回测结果，检查收益率是否异常"""
import json
import pandas as pd
from pathlib import Path

# 读取回测报告
report_path = Path("data/backtest/rqalpha/csi300/report.json")
try:
    with open(report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)
except UnicodeDecodeError:
    # 尝试其他编码
    with open(report_path, 'r', encoding='gbk') as f:
        report = json.load(f)

print("=" * 80)
print("CSI300 回测结果分析")
print("=" * 80)

print("\n=== 关键指标 ===")
print(f"总收益率: {report.get('total_returns', 'N/A')}")
print(f"年化收益率: {report.get('annualized_returns', 'N/A')}")
print(f"最大回撤: {report.get('max_drawdown', 'N/A')}")
print(f"夏普比率: {report.get('sharpe', 'N/A')}")
print(f"胜率: {report.get('win_rate', 'N/A')}")

print("\n=== 详细指标 ===")
for k, v in report.items():
    print(f"{k}: {v}")

# 读取交易记录
trades_path = Path("data/backtest/rqalpha/csi300/trades_detail.csv")
if trades_path.exists():
    trades = pd.read_csv(trades_path)
    print("\n=== 交易记录分析 ===")
    print(f"总交易次数: {len(trades)}")
    print(f"买入次数: {len(trades[trades['side'] == 'BUY'])}")
    print(f"卖出次数: {len(trades[trades['side'] == 'SELL'])}")
    
    # 计算平均持仓时间
    if 'holding_days' in trades.columns:
        print(f"平均持仓天数: {trades['holding_days'].mean():.2f}")
    
    # 计算盈亏分布
    if 'pnl' in trades.columns:
        profitable = len(trades[trades['pnl'] > 0])
        total_closed = len(trades[trades['side'] == 'SELL'])
        if total_closed > 0:
            win_rate = profitable / total_closed * 100
            print(f"实际胜率: {win_rate:.2f}%")
            print(f"平均盈利: {trades[trades['pnl'] > 0]['pnl'].mean():.2f}")
            print(f"平均亏损: {trades[trades['pnl'] < 0]['pnl'].mean():.2f}")

# 读取预测文件
pred_path = Path("data/predictions/pred_csi300.csv")
if pred_path.exists():
    pred = pd.read_csv(pred_path)
    print("\n=== 预测值分析 ===")
    print(f"预测日期范围: {pred['datetime'].min()} 到 {pred['datetime'].max()}")
    print(f"预测样本数: {len(pred)}")
    print(f"预测值均值: {pred['final'].mean():.4f}")
    print(f"预测值标准差: {pred['final'].std():.4f}")
    print(f"预测值范围: [{pred['final'].min():.4f}, {pred['final'].max():.4f}]")
    
    # 检查预测值分布
    print("\n预测值分位数:")
    print(pred['final'].quantile([0.1, 0.25, 0.5, 0.75, 0.9]))

print("\n" + "=" * 80)
print("分析完成")
print("=" * 80)
