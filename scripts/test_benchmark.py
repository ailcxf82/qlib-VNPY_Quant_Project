"""
测试脚本：验证 RQAlpha 回测中的基准指数配置是否生效

功能：
1. 检查配置文件中的基准设置
2. 验证回测结果中是否包含基准数据
3. 分析基准净值和策略净值的对比
4. 生成测试报告
"""

import os
import sys
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# 添加项目根目录到路径
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

from utils import load_yaml_config


def check_config_benchmark(config_path: str) -> Dict[str, Any]:
    """
    检查配置文件中的基准设置
    
    参数:
        config_path: 配置文件路径
        
    返回:
        包含基准配置信息的字典
    """
    print("=" * 80)
    print("步骤 1: 检查配置文件中的基准设置")
    print("=" * 80)
    
    if not os.path.exists(config_path):
        print(f"❌ 错误: 配置文件不存在: {config_path}")
        return {"status": "error", "message": "配置文件不存在"}
    
    try:
        config = load_yaml_config(config_path)
        base_config = config.get("base", {})
        benchmark_code = base_config.get("benchmark", "未设置")
        
        print(f"✓ 配置文件路径: {config_path}")
        print(f"✓ 基准代码: {benchmark_code}")
        print(f"✓ 回测起始日期: {base_config.get('start_date', '未设置')}")
        print(f"✓ 回测结束日期: {base_config.get('end_date', '未设置')}")
        print(f"✓ 初始资金: {base_config.get('initial_cash', '未设置')} 元")
        
        # 验证基准代码格式
        if isinstance(benchmark_code, str) and benchmark_code != "未设置":
            if "." in benchmark_code:
                code_part, exchange_part = benchmark_code.split(".", 1)
                print(f"✓ 基准代码解析: 代码={code_part}, 交易所={exchange_part}")
                
                # 常见基准代码说明
                benchmark_names = {
                    "000300.XSHG": "沪深300",
                    "000905.XSHG": "中证500",
                    "399005.XSHE": "中小板指",
                    "399101.XSHE": "中小综指",
                    "399006.XSHE": "创业板指",
                }
                
                benchmark_name = benchmark_names.get(benchmark_code, "未知指数")
                print(f"✓ 基准指数名称: {benchmark_name}")
                
                if exchange_part not in ["XSHG", "XSHE"]:
                    print(f"⚠ 警告: 交易所后缀 '{exchange_part}' 可能不正确")
            else:
                print(f"⚠ 警告: 基准代码格式可能不正确（缺少交易所后缀）")
        else:
            print(f"⚠ 警告: 基准代码未设置，将使用默认值")
        
        return {
            "status": "success",
            "benchmark_code": benchmark_code,
            "start_date": base_config.get("start_date"),
            "end_date": base_config.get("end_date"),
            "initial_cash": base_config.get("initial_cash"),
        }
    except Exception as e:
        print(f"❌ 错误: 读取配置文件失败: {e}")
        return {"status": "error", "message": str(e)}


def check_backtest_results(output_dir: str) -> Dict[str, Any]:
    """
    检查回测结果中的基准数据
    
    参数:
        output_dir: 回测结果输出目录
        
    返回:
        包含基准数据检查结果的字典
    """
    print("\n" + "=" * 80)
    print("步骤 2: 检查回测结果中的基准数据")
    print("=" * 80)
    
    if not os.path.exists(output_dir):
        print(f"❌ 错误: 输出目录不存在: {output_dir}")
        return {"status": "error", "message": "输出目录不存在"}
    
    results = {
        "status": "success",
        "report_exists": False,
        "benchmark_data_exists": False,
        "benchmark_data_count": 0,
        "strategy_data_exists": False,
        "strategy_data_count": 0,
        "benchmark_nav_range": None,
        "strategy_nav_range": None,
    }
    
    # 1. 检查 report.json
    report_path = os.path.join(output_dir, "report.json")
    if os.path.exists(report_path):
        print(f"✓ 找到回测报告: {report_path}")
        results["report_exists"] = True
        
        try:
            # 尝试多种编码读取
            report = None
            encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312"]
            for enc in encodings:
                try:
                    with open(report_path, "r", encoding=enc) as f:
                        report = json.load(f)
                    print(f"✓ 成功读取报告（编码: {enc}）")
                    break
                except Exception:
                    continue
            
            if report is None:
                print("⚠ 警告: 无法读取 report.json")
            else:
                # 检查报告中的基准信息
                if "summary" in report:
                    summary = report["summary"]
                    print("\n回测摘要信息:")
                    for key, value in summary.items():
                        if isinstance(value, (int, float)):
                            print(f"  {key}: {value:.4f}")
                        else:
                            print(f"  {key}: {value}")
        except Exception as e:
            print(f"⚠ 警告: 读取报告时出错: {e}")
    else:
        print(f"⚠ 警告: 未找到回测报告: {report_path}")
        print("  提示: 请先运行回测生成报告")
    
    # 2. 检查 detailed_results.json
    detailed_path = os.path.join(output_dir, "detailed_results.json")
    if os.path.exists(detailed_path):
        print(f"\n✓ 找到详细结果: {detailed_path}")
        try:
            with open(detailed_path, "r", encoding="utf-8") as f:
                detailed = json.load(f)
            
            if "效率指标" in detailed:
                print("\n效率指标:")
                for key, value in detailed["效率指标"].items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
        except Exception as e:
            print(f"⚠ 警告: 读取详细结果时出错: {e}")
    
    # 3. 尝试从图表数据中检查基准净值
    # 这里我们需要检查是否有其他方式获取基准数据
    # 由于 RQAlpha 的结果结构可能不同，我们尝试多种方式
    
    return results


def analyze_benchmark_data(output_dir: str) -> Dict[str, Any]:
    """
    分析基准数据（如果存在）
    
    参数:
        output_dir: 回测结果输出目录
        
    返回:
        包含基准数据分析结果的字典
    """
    print("\n" + "=" * 80)
    print("步骤 3: 分析基准数据")
    print("=" * 80)
    
    results = {
        "benchmark_nav_found": False,
        "strategy_nav_found": False,
        "benchmark_stats": None,
        "strategy_stats": None,
    }
    
    # 尝试从 report.json 中提取基准数据
    report_path = os.path.join(output_dir, "report.json")
    if os.path.exists(report_path):
        try:
            # 读取报告
            report = None
            encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312"]
            for enc in encodings:
                try:
                    with open(report_path, "r", encoding=enc) as f:
                        report = json.load(f)
                    break
                except Exception:
                    continue
            
            if report:
                # 检查是否有基准收益率信息
                if "summary" in report:
                    summary = report["summary"]
                    
                    # 查找基准相关指标
                    benchmark_keys = [k for k in summary.keys() if "benchmark" in k.lower() or "基准" in k]
                    if benchmark_keys:
                        print("✓ 找到基准相关指标:")
                        for key in benchmark_keys:
                            value = summary[key]
                            if isinstance(value, (int, float)):
                                print(f"  {key}: {value:.4f}")
                            else:
                                print(f"  {key}: {value}")
                        results["benchmark_nav_found"] = True
                    else:
                        print("⚠ 警告: 报告中未找到基准相关指标")
                        print("  可能原因:")
                        print("    1. 基准代码无效或数据源中不存在")
                        print("    2. 基准数据未正确加载")
                        print("    3. RQAlpha 版本问题")
                
                # 检查策略收益率
                strategy_keys = [k for k in summary.keys() if "return" in k.lower() or "收益" in k or "收益率" in k]
                if strategy_keys:
                    print("\n✓ 找到策略收益指标:")
                    for key in strategy_keys:
                        value = summary[key]
                        if isinstance(value, (int, float)):
                            print(f"  {key}: {value:.4f}")
                        else:
                            print(f"  {key}: {value}")
                    results["strategy_nav_found"] = True
        except Exception as e:
            print(f"⚠ 警告: 分析报告时出错: {e}")
    
    return results


def generate_test_report(config_info: Dict, backtest_info: Dict, analysis_info: Dict) -> str:
    """
    生成测试报告
    
    参数:
        config_info: 配置检查结果
        backtest_info: 回测结果检查结果
        analysis_info: 数据分析结果
        
    返回:
        报告文本
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("基准配置测试报告")
    report_lines.append("=" * 80)
    report_lines.append(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # 配置信息
    report_lines.append("【配置信息】")
    if config_info.get("status") == "success":
        report_lines.append(f"✓ 基准代码: {config_info.get('benchmark_code', '未设置')}")
        report_lines.append(f"✓ 回测日期: {config_info.get('start_date')} 至 {config_info.get('end_date')}")
    else:
        report_lines.append(f"❌ 配置检查失败: {config_info.get('message', '未知错误')}")
    report_lines.append("")
    
    # 回测结果
    report_lines.append("【回测结果检查】")
    if backtest_info.get("status") == "success":
        if backtest_info.get("report_exists"):
            report_lines.append("✓ 回测报告存在")
        else:
            report_lines.append("⚠ 回测报告不存在（请先运行回测）")
        
        if backtest_info.get("benchmark_data_exists"):
            report_lines.append(f"✓ 基准数据存在（数据点: {backtest_info.get('benchmark_data_count', 0)}）")
        else:
            report_lines.append("⚠ 基准数据不存在或为空")
    else:
        report_lines.append(f"❌ 回测结果检查失败: {backtest_info.get('message', '未知错误')}")
    report_lines.append("")
    
    # 数据分析
    report_lines.append("【数据分析】")
    if analysis_info.get("benchmark_nav_found"):
        report_lines.append("✓ 基准净值数据已找到")
    else:
        report_lines.append("⚠ 基准净值数据未找到")
        report_lines.append("  可能原因:")
        report_lines.append("    1. 基准代码无效（请检查配置文件中的基准代码）")
        report_lines.append("    2. 数据源中不包含该基准数据")
        report_lines.append("    3. 回测日期范围内无基准数据")
        report_lines.append("    4. RQAlpha 版本或配置问题")
    
    if analysis_info.get("strategy_nav_found"):
        report_lines.append("✓ 策略净值数据已找到")
    report_lines.append("")
    
    # 结论
    report_lines.append("【测试结论】")
    if config_info.get("status") == "success" and analysis_info.get("benchmark_nav_found"):
        report_lines.append("✓ 基准配置已生效！")
        report_lines.append("  基准数据已成功加载并在回测中使用。")
    elif config_info.get("status") == "success":
        report_lines.append("⚠ 基准配置可能未生效")
        report_lines.append("  建议:")
        report_lines.append("    1. 检查基准代码是否正确（常见代码：000300.XSHG=沪深300, 399005.XSHE=中小板指）")
        report_lines.append("    2. 确认数据源包含该基准数据")
        report_lines.append("    3. 检查回测日志中的基准配置信息")
        report_lines.append("    4. 尝试使用其他基准代码进行测试")
    else:
        report_lines.append("❌ 无法完成测试（配置检查失败）")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    return "\n".join(report_lines)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="测试 RQAlpha 回测中的基准配置是否生效")
    parser.add_argument(
        "--config",
        type=str,
        default="config/rqalpha_config.yaml",
        help="RQAlpha 配置文件路径",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/backtest/rqalpha",
        help="回测结果输出目录",
    )
    parser.add_argument(
        "--save-report",
        type=str,
        default=None,
        help="保存测试报告的文件路径（可选）",
    )
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    config_path = os.path.abspath(args.config) if not os.path.isabs(args.config) else args.config
    output_dir = os.path.abspath(args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    
    print("\n" + "=" * 80)
    print("RQAlpha 基准配置测试工具")
    print("=" * 80)
    print(f"配置文件: {config_path}")
    print(f"输出目录: {output_dir}")
    print("")
    
    # 步骤 1: 检查配置
    config_info = check_config_benchmark(config_path)
    
    # 步骤 2: 检查回测结果
    backtest_info = check_backtest_results(output_dir)
    
    # 步骤 3: 分析基准数据
    analysis_info = analyze_benchmark_data(output_dir)
    
    # 生成报告
    report = generate_test_report(config_info, backtest_info, analysis_info)
    
    print("\n" + report)
    
    # 保存报告
    if args.save_report:
        report_path = os.path.abspath(args.save_report) if not os.path.isabs(args.save_report) else args.save_report
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n✓ 测试报告已保存到: {report_path}")
    
    # 返回状态码
    if config_info.get("status") == "success" and analysis_info.get("benchmark_nav_found"):
        sys.exit(0)  # 成功
    else:
        sys.exit(1)  # 失败或警告


if __name__ == "__main__":
    main()

