"""
检查回测结果中的基准数据，诊断为什么图表中没有显示基准曲线
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path

# 添加项目根目录到路径
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

def check_benchmark_data(output_dir: str = "data/backtest/rqalpha"):
    """检查基准数据"""
    print("=" * 80)
    print("检查基准数据诊断")
    print("=" * 80)
    
    output_dir = os.path.abspath(output_dir)
    print(f"输出目录: {output_dir}\n")
    
    # 1. 检查 detailed_results.json
    detailed_path = os.path.join(output_dir, "detailed_results.json")
    if os.path.exists(detailed_path):
        print("【1. 检查 detailed_results.json】")
        try:
            with open(detailed_path, "r", encoding="utf-8") as f:
                detailed = json.load(f)
            
            if "效率指标" in detailed:
                metrics = detailed["效率指标"]
                benchmark_info = {k: v for k, v in metrics.items() if "benchmark" in k.lower()}
                if benchmark_info:
                    print("  [OK] 找到基准相关指标:")
                    for k, v in benchmark_info.items():
                        print(f"    {k}: {v}")
                else:
                    print("  [WARN] 未找到基准相关指标")
        except Exception as e:
            print(f"  [ERROR] 读取失败: {e}")
        print()
    
    # 2. 检查 report.json（尝试读取）
    report_path = os.path.join(output_dir, "report.json")
    if os.path.exists(report_path):
        print("【2. 检查 report.json】")
        try:
            # 尝试多种编码
            report = None
            for enc in ["utf-8", "utf-8-sig", "gbk"]:
                try:
                    with open(report_path, "r", encoding=enc) as f:
                        report = json.load(f)
                    print(f"  [OK] 成功读取（编码: {enc}）")
                    break
                except:
                    continue
            
            if report:
                # 检查基准相关字段
                if "summary" in report:
                    summary = report["summary"]
                    benchmark_keys = [k for k in summary.keys() if "benchmark" in k.lower()]
                    if benchmark_keys:
                        print("  [OK] 找到基准相关字段:")
                        for k in benchmark_keys[:5]:  # 只显示前5个
                            print(f"    {k}: {summary[k]}")
                    else:
                        print("  [WARN] 未找到基准相关字段")
        except Exception as e:
            print(f"  [WARN] 读取 report.json 失败: {e}")
        print()
    
    # 3. 尝试模拟绘图数据提取过程
    print("【3. 模拟绘图数据提取】")
    print("  提示: 需要运行回测才能检查实际的 result 对象")
    print("  建议: 查看回测日志中的以下信息:")
    print("    - '从 sys_analyser.benchmark_portfolio 提取基准净值数据'")
    print("    - '从 sys_analyser.plots 提取基准净值数据'")
    print("    - '基准净值: 日期范围 ... 至 ...'")
    print("    - '绘制基准净值曲线: X 个数据点'")
    print()
    
    # 4. 检查配置文件
    print("【4. 检查配置文件】")
    config_path = "config/rqalpha_config.yaml"
    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            benchmark = config.get("base", {}).get("benchmark", "未设置")
            print(f"  [OK] 配置的基准代码: {benchmark}")
        except Exception as e:
            print(f"  [WARN] 读取配置失败: {e}")
    else:
        print("  [WARN] 配置文件不存在")
    print()
    
    # 5. 建议
    print("【诊断建议】")
    print("  如果图表中没有基准曲线，可能的原因:")
    print("  1. 基准数据提取失败 - 查看回测日志中的警告信息")
    print("  2. 基准数据和策略数据日期不匹配 - 检查日期范围")
    print("  3. 基准数据被过滤掉 - 检查是否有 NaN 或无效值")
    print("  4. 绘图时基准数据为空 - 查看日志中的 '绘制基准净值曲线' 信息")
    print()
    print("  解决方法:")
    print("  1. 重新运行回测，查看详细的日志输出")
    print("  2. 检查回测日志中是否有 '⚠ 未找到基准净值数据' 警告")
    print("  3. 确认配置的基准代码是否正确")
    print("  4. 尝试使用其他基准代码（如 000300.XSHG）进行测试")
    print("=" * 80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="检查基准数据")
    parser.add_argument("--output-dir", type=str, default="data/backtest/rqalpha", help="回测结果输出目录")
    args = parser.parse_args()
    check_benchmark_data(args.output_dir)

