"""
测试脚本：绘制中小板指数（399005.XSHE）和其他基准指数

功能：
1. 使用 RQAlpha 加载指数数据
2. 绘制指数净值曲线
3. 对比多个基准指数
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timedelta

# 添加项目根目录到路径
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

try:
    from rqalpha import run_file
    from rqalpha.api import history_bars
    RQALPHA_AVAILABLE = True
except ImportError:
    RQALPHA_AVAILABLE = False
    print("警告: RQAlpha 未安装，请运行: pip install rqalpha")


def plot_benchmark_index(benchmark_code: str, start_date: str, end_date: str, output_path: Optional[str] = None):
    """
    绘制单个基准指数的净值曲线
    
    参数:
        benchmark_code: 基准代码（如 "399005.XSHE"）
        start_date: 起始日期
        end_date: 结束日期
        output_path: 输出图片路径（可选）
    """
    if not RQALPHA_AVAILABLE:
        print("错误: RQAlpha 未安装")
        return
    
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    # 创建最小策略来获取基准数据
    strategy_code = f'''
import pandas as pd
import numpy as np

def init(context):
    pass

def handle_bar(context, bar_dict):
    pass
'''
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        strategy_path = f.name
        f.write(strategy_code)
    
    try:
        config = {
            "base": {
                "start_date": start_date,
                "end_date": end_date,
                "accounts": {
                    "stock": 100000,
                },
                "benchmark": benchmark_code,
            },
            "mod": {
                "sys_accounts": {
                    "enabled": True,
                },
                "sys_progress": {
                    "enabled": False,
                },
                "sys_analyser": {
                    "enabled": True,
                    "output_file": None,
                },
            },
        }
        
        # 运行回测
        result = run_file(strategy_path, config=config)
        
        # 提取基准数据
        benchmark_data = None
        if isinstance(result, dict) and "sys_analyser" in result:
            analyser = result["sys_analyser"]
            if "benchmark_portfolio" in analyser:
                benchmark_df = analyser["benchmark_portfolio"]
                if isinstance(benchmark_df, pd.DataFrame) and not benchmark_df.empty:
                    if "unit_net_value" in benchmark_df.columns:
                        if "date" in benchmark_df.columns:
                            benchmark_df = benchmark_df.set_index("date")
                        benchmark_data = benchmark_df["unit_net_value"]
        
        if benchmark_data is None or len(benchmark_data) == 0:
            print(f"错误: 无法获取基准数据 {benchmark_code}")
            return
        
        # 绘制图表
        plt.figure(figsize=(14, 8))
        
        # 基准代码到名称的映射
        benchmark_names = {
            "000300.XSHG": "沪深300",
            "000905.XSHG": "中证500",
            "399005.XSHE": "中小板指",
            "399101.XSHE": "中小综指",
            "399006.XSHE": "创业板指",
        }
        benchmark_name = benchmark_names.get(benchmark_code, benchmark_code)
        
        # 提取日期和净值
        dates = benchmark_data.index
        values = benchmark_data.values
        
        # 转换日期格式
        if isinstance(dates, pd.DatetimeIndex):
            dates = [d.date() for d in dates]
        else:
            dates = [pd.Timestamp(d).date() if isinstance(d, (pd.Timestamp, str)) else d for d in dates]
        
        # 绘制曲线
        plt.plot(dates, values, label=f"{benchmark_name} ({benchmark_code})", linewidth=2.5, color="#ff7f0e")
        
        # 设置图表
        plt.title(f"{benchmark_name} ({benchmark_code}) 净值曲线", fontsize=18, fontweight="bold")
        plt.xlabel("日期", fontsize=14)
        plt.ylabel("净值", fontsize=14)
        plt.legend(loc="best", fontsize=12)
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.xticks(rotation=45, ha='right')
        
        # 添加统计信息
        total_return = (values[-1] - values[0]) / values[0] * 100 if len(values) > 0 and values[0] > 0 else 0
        max_value = np.max(values)
        min_value = np.min(values)
        max_drawdown = (max_value - min_value) / max_value * 100 if max_value > 0 else 0
        
        stats_text = f"起始净值: {values[0]:.4f}\n"
        stats_text += f"结束净值: {values[-1]:.4f}\n"
        stats_text += f"总收益率: {total_return:.2f}%\n"
        stats_text += f"最大净值: {max_value:.4f}\n"
        stats_text += f"最小净值: {min_value:.4f}\n"
        stats_text += f"最大回撤: {max_drawdown:.2f}%"
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 保存或显示
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {output_path}")
        else:
            output_path = f"data/backtest/rqalpha/benchmark_{benchmark_code.replace('.', '_')}.png"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {output_path}")
        
        plt.close()
        
        print(f"\n基准指数: {benchmark_name} ({benchmark_code})")
        print(f"数据点数量: {len(benchmark_data)}")
        print(f"日期范围: {dates[0]} 至 {dates[-1]}")
        print(f"净值范围: {min_value:.4f} 至 {max_value:.4f}")
        print(f"总收益率: {total_return:.2f}%")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            os.unlink(strategy_path)
        except Exception:
            pass


def plot_multiple_benchmarks(benchmark_codes: List[str], start_date: str, end_date: str, output_path: Optional[str] = None):
    """
    绘制多个基准指数的对比图
    
    参数:
        benchmark_codes: 基准代码列表
        start_date: 起始日期
        end_date: 结束日期
        output_path: 输出图片路径（可选）
    """
    if not RQALPHA_AVAILABLE:
        print("错误: RQAlpha 未安装")
        return
    
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    # 基准代码到名称的映射
    benchmark_names = {
        "000300.XSHG": "沪深300",
        "000905.XSHG": "中证500",
        "399005.XSHE": "中小板指",
        "399101.XSHE": "中小综指",
        "399006.XSHE": "创业板指",
    }
    
    # 颜色列表
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    
    plt.figure(figsize=(16, 10))
    
    all_data = {}
    
    # 获取每个基准的数据
    for idx, benchmark_code in enumerate(benchmark_codes):
        print(f"\n正在获取 {benchmark_code} 的数据...")
        
        # 创建最小策略
        strategy_code = f'''
def init(context):
    pass

def handle_bar(context, bar_dict):
    pass
'''
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            strategy_path = f.name
            f.write(strategy_code)
        
        try:
            config = {
                "base": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "accounts": {
                        "stock": 100000,
                    },
                    "benchmark": benchmark_code,
                },
                "mod": {
                    "sys_accounts": {
                        "enabled": True,
                    },
                    "sys_progress": {
                        "enabled": False,
                    },
                    "sys_analyser": {
                        "enabled": True,
                        "output_file": None,
                    },
                },
            }
            
            result = run_file(strategy_path, config=config)
            
            # 提取基准数据
            if isinstance(result, dict) and "sys_analyser" in result:
                analyser = result["sys_analyser"]
                if "benchmark_portfolio" in analyser:
                    benchmark_df = analyser["benchmark_portfolio"]
                    if isinstance(benchmark_df, pd.DataFrame) and not benchmark_df.empty:
                        if "unit_net_value" in benchmark_df.columns:
                            if "date" in benchmark_df.columns:
                                benchmark_df = benchmark_df.set_index("date")
                            benchmark_data = benchmark_df["unit_net_value"]
                            all_data[benchmark_code] = benchmark_data
                            print(f"  ✓ 成功获取 {len(benchmark_data)} 个数据点")
                        else:
                            print(f"  ✗ 未找到 unit_net_value 列")
                    else:
                        print(f"  ✗ 基准数据为空")
                else:
                    print(f"  ✗ 未找到 benchmark_portfolio")
            else:
                print(f"  ✗ 无法获取回测结果")
        except Exception as e:
            print(f"  ✗ 错误: {e}")
        finally:
            try:
                os.unlink(strategy_path)
            except Exception:
                pass
    
    # 绘制所有基准
    if not all_data:
        print("\n错误: 未获取到任何基准数据")
        return
    
    for idx, (benchmark_code, benchmark_data) in enumerate(all_data.items()):
        benchmark_name = benchmark_names.get(benchmark_code, benchmark_code)
        color = colors[idx % len(colors)]
        
        # 提取日期和净值
        dates = benchmark_data.index
        values = benchmark_data.values
        
        # 转换日期格式
        if isinstance(dates, pd.DatetimeIndex):
            dates = [d.date() for d in dates]
        else:
            dates = [pd.Timestamp(d).date() if isinstance(d, (pd.Timestamp, str)) else d for d in dates]
        
        # 绘制曲线
        linestyle = "--" if "399" in benchmark_code else "-"  # 中小指数用虚线
        plt.plot(dates, values, label=f"{benchmark_name} ({benchmark_code})", 
                linewidth=2, color=color, linestyle=linestyle)
    
    # 设置图表
    plt.title("基准指数净值对比", fontsize=18, fontweight="bold")
    plt.xlabel("日期", fontsize=14)
    plt.ylabel("净值", fontsize=14)
    plt.legend(loc="best", fontsize=11, ncol=2)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.xticks(rotation=45, ha='right')
    
    # 保存或显示
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n对比图表已保存到: {output_path}")
    else:
        output_path = f"data/backtest/rqalpha/benchmark_comparison.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n对比图表已保存到: {output_path}")
    
    plt.close()
    
    # 打印统计信息
    print("\n" + "="*80)
    print("基准指数统计信息")
    print("="*80)
    for benchmark_code, benchmark_data in all_data.items():
        benchmark_name = benchmark_names.get(benchmark_code, benchmark_code)
        values = benchmark_data.values
        total_return = (values[-1] - values[0]) / values[0] * 100 if len(values) > 0 and values[0] > 0 else 0
        print(f"\n{benchmark_name} ({benchmark_code}):")
        print(f"  数据点: {len(benchmark_data)}")
        print(f"  起始净值: {values[0]:.4f}")
        print(f"  结束净值: {values[-1]:.4f}")
        print(f"  总收益率: {total_return:.2f}%")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="绘制基准指数净值曲线")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="399005.XSHE",
        help="基准代码（默认: 399005.XSHE 中小板指）",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2023-10-01",
        help="起始日期（格式: YYYY-MM-DD）",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2025-10-01",
        help="结束日期（格式: YYYY-MM-DD）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出图片路径（可选）",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="对比多个基准指数（中小板指、中小综指、创业板指、沪深300、中证500）",
    )
    
    args = parser.parse_args()
    
    if not RQALPHA_AVAILABLE:
        print("错误: RQAlpha 未安装")
        print("请运行: pip install rqalpha")
        sys.exit(1)
    
    print("="*80)
    print("基准指数绘图测试")
    print("="*80)
    print(f"起始日期: {args.start_date}")
    print(f"结束日期: {args.end_date}")
    print()
    
    if args.compare:
        # 对比多个基准
        benchmark_codes = [
            "399005.XSHE",  # 中小板指
            "399101.XSHE",  # 中小综指
            "399006.XSHE",  # 创业板指
            "000300.XSHG",  # 沪深300
            "000905.XSHG",  # 中证500
        ]
        print(f"对比基准: {', '.join(benchmark_codes)}")
        plot_multiple_benchmarks(benchmark_codes, args.start_date, args.end_date, args.output)
    else:
        # 绘制单个基准
        print(f"基准代码: {args.benchmark}")
        plot_benchmark_index(args.benchmark, args.start_date, args.end_date, args.output)
    
    print("\n" + "="*80)
    print("测试完成")
    print("="*80)


if __name__ == "__main__":
    main()

