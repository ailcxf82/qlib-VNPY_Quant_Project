"""
分析 RQAlpha 回测结果，打印投资明细、盈亏状态和效率指标
"""
import os
import sys
import json
import pandas as pd
from pathlib import Path

# 添加项目根目录到路径
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))


def analyze_backtest_results(output_dir: str):
    """
    分析回测结果并打印详细信息
    
    参数:
        output_dir: 回测结果输出目录
    """
    print("=" * 80)
    print("RQAlpha 回测结果分析")
    print("=" * 80)
    
    if not os.path.exists(output_dir):
        print(f"错误: 输出目录不存在: {output_dir}")
        return
    
    # 1. 读取 report.json
    report_path = os.path.join(output_dir, "report.json")
    if os.path.exists(report_path):
        print(f"\n1. 读取回测报告: {report_path}")
        try:
            # 尝试多种编码方式
            encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin-1"]
            report = None
            for encoding in encodings:
                try:
                    with open(report_path, "r", encoding=encoding) as f:
                        report = json.load(f)
                    print(f"   成功使用 {encoding} 编码读取文件")
                    break
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
            
            if report is None:
                print(f"   警告: 无法读取 report.json，尝试使用二进制模式")
                # 如果所有编码都失败，尝试二进制模式
                with open(report_path, "rb") as f:
                    content = f.read()
                    # 尝试检测编码
                    try:
                        import chardet
                        detected = chardet.detect(content)
                        encoding = detected.get("encoding", "utf-8")
                        print(f"   检测到编码: {encoding}")
                        report = json.loads(content.decode(encoding))
                    except ImportError:
                        # 如果没有 chardet，尝试 utf-8 并忽略错误
                        report = json.loads(content.decode("utf-8", errors="ignore"))
                    except Exception as e:
                        print(f"   错误: 无法读取 report.json: {e}")
                        report = None
        except Exception as e:
            print(f"   错误: 读取 report.json 失败: {e}")
            report = None
        
        if report is None:
            print(f"   跳过 report.json 的读取")
            report = {}
        
        # 打印摘要信息
        if "summary" in report:
            print("\n" + "=" * 80)
            print("回测摘要:")
            print("=" * 80)
            summary = report["summary"]
            for key, value in summary.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        
        # 打印交易统计
        if "trades" in report:
            print("\n" + "=" * 80)
            print("交易统计:")
            print("=" * 80)
            trades_info = report["trades"]
            if isinstance(trades_info, dict):
                for key, value in trades_info.items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
    else:
        print(f"警告: 未找到回测报告: {report_path}")
    
    # 2. 读取持仓明细
    positions_path = os.path.join(output_dir, "positions_detail.csv")
    if os.path.exists(positions_path):
        print(f"\n2. 读取持仓明细: {positions_path}")
        try:
            # 尝试多种编码方式
            encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312"]
            positions_df = None
            for encoding in encodings:
                try:
                    positions_df = pd.read_csv(positions_path, encoding=encoding)
                    print(f"   成功使用 {encoding} 编码读取文件")
                    break
                except (UnicodeDecodeError, pd.errors.ParserError):
                    continue
            
            if positions_df is None:
                print(f"   警告: 无法读取持仓明细文件，尝试使用错误处理")
                positions_df = pd.read_csv(positions_path, encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"   错误: 读取持仓明细失败: {e}")
            positions_df = None
        
        if positions_df is not None:
            print(f"   共 {len(positions_df)} 条持仓记录")
        
        if positions_df is not None and len(positions_df) > 0:
            print("\n" + "=" * 80)
            print("持仓明细（前20条）:")
            print("=" * 80)
            print(positions_df.head(20).to_string(index=False))
            
            # 统计持仓信息
            if "order_book_id" in positions_df.columns:
                unique_stocks = positions_df["order_book_id"].nunique()
                print(f"\n   持仓股票数量: {unique_stocks} 只")
            
            if "market_value" in positions_df.columns:
                total_market_value = positions_df["market_value"].sum()
                print(f"   总持仓市值: {total_market_value:,.2f} 元")
    else:
        print(f"警告: 未找到持仓明细文件: {positions_path}")
    
    # 3. 读取交易明细
    trades_path = os.path.join(output_dir, "trades_detail.csv")
    if os.path.exists(trades_path):
        print(f"\n3. 读取交易明细: {trades_path}")
        try:
            # 尝试多种编码方式
            encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312"]
            trades_df = None
            for encoding in encodings:
                try:
                    trades_df = pd.read_csv(trades_path, encoding=encoding)
                    print(f"   成功使用 {encoding} 编码读取文件")
                    break
                except (UnicodeDecodeError, pd.errors.ParserError):
                    continue
            
            if trades_df is None:
                print(f"   警告: 无法读取交易明细文件，尝试使用错误处理")
                trades_df = pd.read_csv(trades_path, encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"   错误: 读取交易明细失败: {e}")
            trades_df = None
        
        if trades_df is not None:
            print(f"   共 {len(trades_df)} 条交易记录")
        
        if trades_df is not None and len(trades_df) > 0:
            print("\n" + "=" * 80)
            print("交易明细（前20条）:")
            print("=" * 80)
            print(trades_df.head(20).to_string(index=False))
            
            # 统计交易信息
            if "side" in trades_df.columns:
                buy_count = (trades_df["side"] == "BUY").sum()
                sell_count = (trades_df["side"] == "SELL").sum()
                print(f"\n   买入次数: {buy_count} 次")
                print(f"   卖出次数: {sell_count} 次")
            
            if "amount" in trades_df.columns:
                total_amount = trades_df["amount"].sum()
                print(f"   总交易金额: {total_amount:,.2f} 元")
            
            if "commission" in trades_df.columns:
                total_commission = trades_df["commission"].sum()
                print(f"   总手续费: {total_commission:,.2f} 元")
            
            # 统计预测值和盈亏信息
            if "prediction_value" in trades_df.columns:
                buy_trades = trades_df[trades_df["side"] == "BUY"]
                if len(buy_trades) > 0:
                    valid_pred = buy_trades["prediction_value"].dropna()
                    if len(valid_pred) > 0:
                        print(f"\n   买入时预测值统计:")
                        print(f"     平均预测值: {valid_pred.mean():.6f}")
                        print(f"     预测值范围: [{valid_pred.min():.6f}, {valid_pred.max():.6f}]")
                        print(f"     有预测值的买入次数: {len(valid_pred)} / {len(buy_trades)}")
            
            if "cost" in trades_df.columns:
                buy_trades = trades_df[trades_df["side"] == "BUY"]
                if len(buy_trades) > 0:
                    valid_cost = buy_trades["cost"].dropna()
                    if len(valid_cost) > 0:
                        total_cost = valid_cost.sum()
                        print(f"\n   买入成本统计:")
                        print(f"     总买入成本: {total_cost:,.2f} 元")
                        print(f"     平均单次买入成本: {valid_cost.mean():,.2f} 元")
            
            if "profit" in trades_df.columns or "pnl" in trades_df.columns:
                sell_trades = trades_df[trades_df["side"] == "SELL"]
                if len(sell_trades) > 0:
                    if "pnl" in trades_df.columns:
                        valid_pnl = sell_trades["pnl"].dropna()
                        if len(valid_pnl) > 0:
                            total_pnl = valid_pnl.sum()
                            win_count = (valid_pnl > 0).sum()
                            loss_count = (valid_pnl < 0).sum()
                            print(f"\n   卖出盈亏统计:")
                            print(f"     总盈亏: {total_pnl:,.2f} 元")
                            print(f"     盈利次数: {win_count} 次")
                            print(f"     亏损次数: {loss_count} 次")
                            print(f"     胜率: {win_count / len(valid_pnl) * 100:.2f}%")
                            if win_count > 0:
                                avg_win = valid_pnl[valid_pnl > 0].mean()
                                print(f"     平均盈利: {avg_win:,.2f} 元")
                            if loss_count > 0:
                                avg_loss = valid_pnl[valid_pnl < 0].mean()
                                print(f"     平均亏损: {avg_loss:,.2f} 元")
                    elif "profit" in trades_df.columns:
                        valid_profit = sell_trades["profit"].dropna()
                        if len(valid_profit) > 0:
                            total_profit = valid_profit.sum()
                            print(f"\n   卖出收益统计:")
                            print(f"     总卖出收益: {total_profit:,.2f} 元")
                            print(f"     平均单次卖出收益: {valid_profit.mean():,.2f} 元")
    else:
        print(f"警告: 未找到交易明细文件: {trades_path}")
    
    # 4. 读取详细结果
    detailed_path = os.path.join(output_dir, "detailed_results.json")
    if os.path.exists(detailed_path):
        print(f"\n4. 读取详细结果: {detailed_path}")
        try:
            # 尝试多种编码方式
            encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin-1"]
            detailed_results = None
            for encoding in encodings:
                try:
                    with open(detailed_path, "r", encoding=encoding) as f:
                        detailed_results = json.load(f)
                    print(f"   成功使用 {encoding} 编码读取文件")
                    break
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
            
            if detailed_results is None:
                print(f"   警告: 无法读取 detailed_results.json，尝试使用二进制模式")
                with open(detailed_path, "rb") as f:
                    content = f.read()
                    try:
                        import chardet
                        detected = chardet.detect(content)
                        encoding = detected.get("encoding", "utf-8")
                        print(f"   检测到编码: {encoding}")
                        detailed_results = json.loads(content.decode(encoding))
                    except ImportError:
                        detailed_results = json.loads(content.decode("utf-8", errors="ignore"))
                    except Exception as e:
                        print(f"   错误: 无法读取 detailed_results.json: {e}")
                        detailed_results = {}
        except Exception as e:
            print(f"   错误: 读取详细结果失败: {e}")
            detailed_results = {}
        
        # 打印盈亏状态
        if "盈亏状态" in detailed_results:
            print("\n" + "=" * 80)
            print("盈亏状态:")
            print("=" * 80)
            profit_loss = detailed_results["盈亏状态"]
            for key, value in profit_loss.items():
                if isinstance(value, (int, float)):
                    if "率" in key or "收益" in key:
                        print(f"  {key}: {value:.2f}%")
                    else:
                        print(f"  {key}: {value:,.2f} 元")
                else:
                    print(f"  {key}: {value}")
        
        # 打印效率指标
        if "效率指标" in detailed_results:
            print("\n" + "=" * 80)
            print("效率指标:")
            print("=" * 80)
            metrics = detailed_results["效率指标"]
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
    else:
        print(f"警告: 未找到详细结果文件: {detailed_path}")
    
    # 5. 读取 summary.json（如果存在）
    summary_path = os.path.join(output_dir, "summary.json")
    if os.path.exists(summary_path):
        print(f"\n5. 读取摘要文件: {summary_path}")
        try:
            # 尝试多种编码方式
            encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin-1"]
            summary = None
            for encoding in encodings:
                try:
                    with open(summary_path, "r", encoding=encoding) as f:
                        summary = json.load(f)
                    print(f"   成功使用 {encoding} 编码读取文件")
                    break
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
            
            if summary is None:
                print(f"   警告: 无法读取摘要文件，尝试使用二进制模式")
                with open(summary_path, "rb") as f:
                    content = f.read()
                    try:
                        import chardet
                        detected = chardet.detect(content)
                        encoding = detected.get("encoding", "utf-8")
                        print(f"   检测到编码: {encoding}")
                        summary = json.loads(content.decode(encoding))
                    except ImportError:
                        summary = json.loads(content.decode("utf-8", errors="ignore"))
                    except Exception as e:
                        print(f"   错误: 无法读取摘要文件: {e}")
                        summary = {}
        except Exception as e:
            print(f"   错误: 读取摘要文件失败: {e}")
            summary = {}
        
        if summary:
            print("\n" + "=" * 80)
            print("摘要信息:")
            print("=" * 80)
            for key, value in summary.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)
    print("分析完成")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="分析 RQAlpha 回测结果，打印投资明细、盈亏状态和效率指标")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/backtest/rqalpha",
        help="回测结果输出目录（默认: data/backtest/rqalpha）",
    )
    
    args = parser.parse_args()
    
    analyze_backtest_results(args.output_dir)

