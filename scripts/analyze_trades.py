"""
分析回测交易明细，检查潜在问题和漏洞。
"""

import pandas as pd
import numpy as np
from collections import defaultdict

def analyze_trades(filepath):
    df = pd.read_csv(filepath)
    
    print("=" * 80)
    print("交易明细分析报告")
    print("=" * 80)
    
    print(f"\n总交易记录数: {len(df)}")
    print(f"买入次数: {len(df[df['side'] == 'BUY'])}")
    print(f"卖出次数: {len(df[df['side'] == 'SELL'])}")
    
    print(f"\n日期范围: {df['datetime'].min()} 到 {df['datetime'].max()}")
    
    unique_dates = df['datetime'].unique()
    print(f"交易日数量: {len(unique_dates)}")
    
    print("\n" + "=" * 80)
    print("检查 1: 买入卖出配对分析")
    print("=" * 80)
    
    positions = defaultdict(list)
    for _, row in df.iterrows():
        symbol = row['symbol']
        side = row['side']
        qty = row['last_quantity']
        date = row['datetime']
        
        if side == 'BUY':
            positions[symbol].append({'date': date, 'qty': qty, 'type': 'BUY'})
        elif side == 'SELL':
            positions[symbol].append({'date': date, 'qty': qty, 'type': 'SELL'})
    
    unpaired_buys = []
    unpaired_sells = []
    for symbol, trades in positions.items():
        buy_qty = sum(t['qty'] for t in trades if t['type'] == 'BUY')
        sell_qty = sum(t['qty'] for t in trades if t['type'] == 'SELL')
        if buy_qty != sell_qty:
            print(f"  ⚠️ {symbol}: 买入={buy_qty}, 卖出={sell_qty}, 差异={buy_qty - sell_qty}")
            if buy_qty > sell_qty:
                unpaired_buys.append((symbol, buy_qty - sell_qty))
            else:
                unpaired_sells.append((symbol, sell_qty - buy_qty))
    
    if not unpaired_buys and not unpaired_sells:
        print("  ✅ 所有股票买卖数量匹配")
    
    print("\n" + "=" * 80)
    print("检查 2: 预测值分布分析")
    print("=" * 80)
    
    pred_values = df['prediction_value'].values
    print(f"预测值范围: [{pred_values.min():.4f}, {pred_values.max():.4f}]")
    print(f"预测值均值: {pred_values.mean():.4f}")
    print(f"预测值中位数: {np.median(pred_values):.4f}")
    print(f"预测值标准差: {pred_values.std():.4f}")
    
    buy_preds = df[df['side'] == 'BUY']['prediction_value']
    sell_preds = df[df['side'] == 'SELL']['prediction_value']
    
    print(f"\n买入时预测值均值: {buy_preds.mean():.4f}")
    print(f"卖出时预测值均值: {sell_preds.mean():.4f}")
    
    if sell_preds.mean() > buy_preds.mean():
        print("  ⚠️ 异常: 卖出时预测值高于买入时，可能存在反向操作")
    else:
        print("  ✅ 正常: 买入时预测值高于卖出时")
    
    print("\n" + "=" * 80)
    print("检查 3: 持仓周期分析")
    print("=" * 80)
    
    holding_periods = []
    for symbol, trades in positions.items():
        buy_trades = [t for t in trades if t['type'] == 'BUY']
        sell_trades = [t for t in trades if t['type'] == 'SELL']
        
        for buy in buy_trades:
            for sell in sell_trades:
                if sell['qty'] == buy['qty']:
                    buy_date = pd.to_datetime(buy['date'])
                    sell_date = pd.to_datetime(sell['date'])
                    days = (sell_date - buy_date).days
                    if days > 0:
                        holding_periods.append(days)
                    break
    
    if holding_periods:
        print(f"平均持仓天数: {np.mean(holding_periods):.1f}")
        print(f"最短持仓天数: {min(holding_periods)}")
        print(f"最长持仓天数: {max(holding_periods)}")
        
        short_holds = [h for h in holding_periods if h <= 5]
        print(f"持仓<=5天的交易占比: {len(short_holds)/len(holding_periods)*100:.1f}%")
    
    print("\n" + "=" * 80)
    print("检查 4: 交易成本分析")
    print("=" * 80)
    
    total_commission = df['commission'].sum()
    total_tax = df['tax'].sum()
    total_cost = df['transaction_cost'].sum()
    
    print(f"总手续费: {total_commission:,.2f}")
    print(f"总印花税: {total_tax:,.2f}")
    print(f"总交易成本: {total_cost:,.2f}")
    
    sell_df = df[df['side'] == 'SELL']
    if len(sell_df) > 0:
        total_pnl = sell_df['pnl'].sum()
        print(f"总盈亏: {total_pnl:,.2f}")
        print(f"交易成本占盈亏比例: {abs(total_cost/total_pnl)*100:.1f}%" if total_pnl != 0 else "N/A")
    
    print("\n" + "=" * 80)
    print("检查 5: 单日交易集中度")
    print("=" * 80)
    
    daily_trades = df.groupby('datetime').size()
    print(f"单日最大交易次数: {daily_trades.max()}")
    print(f"单日最小交易次数: {daily_trades.min()}")
    print(f"平均每日交易次数: {daily_trades.mean():.1f}")
    
    high_trade_days = daily_trades[daily_trades > 20]
    if len(high_trade_days) > 0:
        print(f"\n高交易量日期 (>20笔):")
        for date, count in high_trade_days.items():
            print(f"  {date}: {count} 笔")
    
    print("\n" + "=" * 80)
    print("检查 6: 股票覆盖分析")
    print("=" * 80)
    
    unique_stocks = df['symbol'].unique()
    print(f"交易股票数量: {len(unique_stocks)}")
    
    stock_trades = df.groupby('symbol').size().sort_values(ascending=False)
    print(f"\n交易最频繁的10只股票:")
    for symbol, count in stock_trades.head(10).items():
        print(f"  {symbol}: {count} 次")
    
    print("\n" + "=" * 80)
    print("检查 7: 盈亏分布分析")
    print("=" * 80)
    
    if len(sell_df) > 0:
        profits = sell_df['pnl'].dropna()
        positive = (profits > 0).sum()
        negative = (profits < 0).sum()
        zero = (profits == 0).sum()
        
        print(f"盈利交易: {positive} 次 ({positive/len(profits)*100:.1f}%)")
        print(f"亏损交易: {negative} 次 ({negative/len(profits)*100:.1f}%)")
        print(f"持平交易: {zero} 次")
        
        print(f"\n平均盈利: {profits[profits > 0].mean():,.2f}" if positive > 0 else "N/A")
        print(f"平均亏损: {profits[profits < 0].mean():,.2f}" if negative > 0 else "N/A")
        
        total_profit = profits[profits > 0].sum()
        total_loss = abs(profits[profits < 0].sum())
        print(f"\n总盈利: {total_profit:,.2f}")
        print(f"总亏损: {total_loss:,.2f}")
        print(f"盈亏比: {total_profit/total_loss:.2f}" if total_loss > 0 else "N/A")
    
    print("\n" + "=" * 80)
    print("检查 8: 预测值异常分析")
    print("=" * 80)
    
    low_pred_buys = df[(df['side'] == 'BUY') & (df['prediction_value'] < 0.5)]
    high_pred_sells = df[(df['side'] == 'SELL') & (df['prediction_value'] > 0.5)]
    
    if len(low_pred_buys) > 0:
        print(f"⚠️ 低预测值买入 (<0.5): {len(low_pred_buys)} 次")
        print(f"   示例:")
        for _, row in low_pred_buys.head(5).iterrows():
            print(f"     {row['datetime']}: {row['symbol']} @ {row['prediction_value']:.4f}")
    
    if len(high_pred_sells) > 0:
        print(f"\n⚠️ 高预测值卖出 (>0.5): {len(high_pred_sells)} 次")
        print(f"   示例:")
        for _, row in high_pred_sells.head(5).iterrows():
            print(f"     {row['datetime']}: {row['symbol']} @ {row['prediction_value']:.4f}")
    
    print("\n" + "=" * 80)
    print("检查 9: 时间间隔分析")
    print("=" * 80)
    
    dates = pd.to_datetime(df['datetime'].unique())
    dates = dates.sort_values()
    gaps = np.diff(dates).astype('timedelta64[D]').astype(int)
    
    print(f"交易日间隔统计:")
    print(f"  最小间隔: {gaps.min()} 天")
    print(f"  最大间隔: {gaps.max()} 天")
    print(f"  平均间隔: {gaps.mean():.1f} 天")
    
    long_gaps = gaps[gaps > 7]
    if len(long_gaps) > 0:
        print(f"\n⚠️ 长间隔 (>7天): {len(long_gaps)} 次")
        for i, gap in enumerate(long_gaps[:5]):
            idx = np.where(gaps == gap)[0][0]
            print(f"   {dates[idx]} -> {dates[idx+1]}: {gap} 天")
    
    print("\n" + "=" * 80)
    print("检查 10: 滑点和成交价分析")
    print("=" * 80)
    
    buy_df = df[df['side'] == 'BUY']
    if len(buy_df) > 0:
        avg_buy_cost = (buy_df['cost'] / buy_df['last_quantity']).mean()
        avg_buy_price = buy_df['last_price'].mean()
        print(f"买入平均成本/股: {avg_buy_cost:.4f}")
        print(f"买入平均价格: {avg_buy_price:.4f}")
    
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    
    issues = []
    
    if len(unpaired_buys) > 0 or len(unpaired_sells) > 0:
        issues.append("存在未配对的买卖交易")
    
    if sell_preds.mean() > buy_preds.mean():
        issues.append("卖出时预测值高于买入时（可能反向操作）")
    
    if len(low_pred_buys) > len(buy_df) * 0.1:
        issues.append(f"低预测值买入占比过高 ({len(low_pred_buys)/len(buy_df)*100:.1f}%)")
    
    if len(high_pred_sells) > len(sell_df) * 0.1:
        issues.append(f"高预测值卖出占比过高 ({len(high_pred_sells)/len(sell_df)*100:.1f}%)")
    
    if len(issues) == 0:
        print("✅ 未发现明显问题")
    else:
        print("发现以下潜在问题:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")

if __name__ == "__main__":
    analyze_trades("data/backtest/rqalpha/csi300/trades_detail.csv")
