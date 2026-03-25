import pandas as pd
import os

# 检查预测文件的最新日期
for pool in ['csi300', 'csi101']:
    pred_file = os.path.join('data', 'predictions', f'pred_{pool}.csv')
    if os.path.exists(pred_file):
        df = pd.read_csv(pred_file)
        df['datetime'] = pd.to_datetime(df['datetime'])
        latest_date = df['datetime'].max()
        earliest_date = df['datetime'].min()
        unique_dates = df['datetime'].nunique()
        total_rows = len(df)
        
        print(f"=== {pool} 预测文件信息 ===")
        print(f"最新预测日期: {latest_date.date()}")
        print(f"最早预测日期: {earliest_date.date()}")
        print(f"预测日期数量: {unique_dates}")
        print(f"总数据条数: {total_rows}")
        print(f"文件路径: {pred_file}")
        print()
    else:
        print(f"=== {pool} 预测文件信息 ===")
        print(f"预测文件不存在: {pred_file}")
        print()

# 检查今天的日期
import datetime
today = datetime.datetime.now().date()
print(f"=== 当前日期 ===")
print(f"今天: {today}")
print()

# 检查近一周的日期范围
one_week_ago = today - datetime.timedelta(days=7)
print(f"=== 近一周范围 ===")
print(f"一周前: {one_week_ago}")
print(f"今天: {today}")
print()
