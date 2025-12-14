"""
测试 RQAlpha 对中小指数的支持情况

根据 RQAlpha 文档：https://rqalpha.readthedocs.io/zh-cn/latest/api/base_api.html
使用 RQAlpha API 查询和测试基准指数的支持情况
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# 添加项目根目录到路径
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

try:
    from rqalpha import run_file
    from rqalpha.api import all_instruments, instruments
    RQALPHA_AVAILABLE = True
except ImportError:
    RQALPHA_AVAILABLE = False
    print("⚠ 警告: RQAlpha 未安装，部分功能无法使用")
    print("  请运行: pip install rqalpha")


def test_benchmark_codes() -> List[str]:
    """
    测试常见的中小指数代码
    
    返回:
        要测试的基准代码列表
    """
    return [
        "399005.XSHE",  # 中小板指
        "399101.XSHE",  # 中小综指
        "399006.XSHE",  # 创业板指
        "000300.XSHG",  # 沪深300（作为对照，注意是XSHG不是XSHE）
        "000905.XSHG",  # 中证500（作为对照）
    ]


def query_instrument_info_with_strategy(order_book_id: str) -> Optional[Dict[str, Any]]:
    """
    在策略环境中查询合约信息（通过最小回测）
    
    参数:
        order_book_id: 合约代码
        
    返回:
        合约信息字典，如果不存在则返回 None
    """
    if not RQALPHA_AVAILABLE:
        return None
    
    # 创建一个简单的策略来查询合约信息
    # 使用全局变量存储结果（通过文件）
    import tempfile
    result_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8')
    result_file_path = result_file.name
    result_file.close()
    
    strategy_code = f'''
import json
import os

def init(context):
    result = {{"order_book_id": "{order_book_id}", "found": False, "error": None}}
    try:
        from rqalpha.api import instruments
        ins = instruments("{order_book_id}")
        if ins:
            result["found"] = True
            result["order_book_id"] = ins.order_book_id
            result["symbol"] = getattr(ins, "symbol", "")
            result["type"] = str(getattr(ins, "type", ""))
            result["listed_date"] = str(getattr(ins, "listed_date", ""))
            result["de_listed_date"] = str(getattr(ins, "de_listed_date", ""))
        else:
            result["error"] = "合约不存在"
    except Exception as e:
        result["error"] = str(e)
    
    # 保存结果到文件
    with open(r"{result_file_path}", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)

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
                "start_date": "2023-01-01",
                "end_date": "2023-01-02",  # 最短日期范围
                "accounts": {
                    "stock": 100000,
                },
            },
            "mod": {
                "sys_accounts": {
                    "enabled": True,
                },
                "sys_progress": {
                    "enabled": False,
                },
                "sys_analyser": {
                    "enabled": False,  # 不需要分析器
                },
            },
        }
        
        # 运行回测以获取合约信息
        run_file(strategy_path, config=config)
        
        # 读取结果文件
        if os.path.exists(result_file_path):
            import json
            with open(result_file_path, "r", encoding="utf-8") as f:
                result = json.load(f)
                if result.get("found"):
                    return {
                        "order_book_id": result.get("order_book_id"),
                        "symbol": result.get("symbol", ""),
                        "type": result.get("type", ""),
                        "listed_date": result.get("listed_date"),
                        "de_listed_date": result.get("de_listed_date"),
                    }
                else:
                    return {"error": result.get("error", "合约不存在")}
        else:
            return {"error": "无法获取查询结果"}
    except Exception as e:
        return {"error": f"运行策略失败: {str(e)}"}
    finally:
        # 清理临时文件
        try:
            os.unlink(strategy_path)
            if os.path.exists(result_file_path):
                os.unlink(result_file_path)
        except Exception:
            pass
    
    return None


def test_benchmark_support_simple(benchmark_code: str) -> Dict[str, Any]:
    """
    简单测试基准代码是否可用（使用策略环境查询）
    
    参数:
        benchmark_code: 基准代码
        
    返回:
        测试结果字典
    """
    result = {
        "benchmark_code": benchmark_code,
        "instrument_exists": False,
        "instrument_info": None,
        "error": None,
    }
    
    if not RQALPHA_AVAILABLE:
        result["error"] = "RQAlpha 未安装"
        return result
    
    try:
        # 使用策略环境查询合约信息
        ins_info = query_instrument_info_with_strategy(benchmark_code)
        if ins_info:
            if "error" in ins_info:
                result["error"] = ins_info["error"]
            else:
                result["instrument_exists"] = True
                result["instrument_info"] = ins_info
        else:
            result["error"] = "合约不存在或无法查询"
    except Exception as e:
        result["error"] = str(e)
    
    return result


def test_benchmark_with_minimal_backtest(benchmark_code: str, start_date: str = "2023-01-01", end_date: str = "2023-01-10") -> Dict[str, Any]:
    """
    使用最小回测测试基准代码（仅测试配置是否有效）
    
    参数:
        benchmark_code: 基准代码
        start_date: 起始日期
        end_date: 结束日期
        
    返回:
        测试结果字典
    """
    if not RQALPHA_AVAILABLE:
        return {
            "benchmark_code": benchmark_code,
            "status": "error",
            "error": "RQAlpha 未安装",
        }
    
    # 创建最小策略脚本
    strategy_code = f'''
def init(context):
    pass

def handle_bar(context, bar_dict):
    pass
'''
    
    # 创建临时策略文件
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
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
                    "output_file": None,  # 不输出文件
                },
            },
        }
        
        # 运行回测
        result = run_file(strategy_path, config=config)
        
        # 检查结果
        benchmark_loaded = False
        if isinstance(result, dict) and "sys_analyser" in result:
            analyser = result["sys_analyser"]
            if "benchmark_portfolio" in analyser:
                benchmark_df = analyser.get("benchmark_portfolio")
                if benchmark_df is not None:
                    import pandas as pd
                    if isinstance(benchmark_df, pd.DataFrame) and not benchmark_df.empty:
                        benchmark_loaded = True
        
        return {
            "benchmark_code": benchmark_code,
            "status": "success" if benchmark_loaded else "warning",
            "benchmark_loaded": benchmark_loaded,
            "error": None if benchmark_loaded else "基准数据未加载",
        }
    except Exception as e:
        return {
            "benchmark_code": benchmark_code,
            "status": "error",
            "error": str(e),
        }
    finally:
        # 清理临时文件
        try:
            os.unlink(strategy_path)
        except Exception:
            pass


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="测试 RQAlpha 对中小指数的支持情况")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["simple", "backtest", "both"],
        default="simple",
        help="测试模式: simple=简单查询, backtest=回测测试, both=两种都测试",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="指定要测试的基准代码（默认测试所有常见中小指数）",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2023-01-01",
        help="回测起始日期（仅用于 backtest 模式）",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2023-01-10",
        help="回测结束日期（仅用于 backtest 模式）",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("RQAlpha 中小指数支持测试")
    print("=" * 80)
    print(f"测试模式: {args.mode}")
    print("")
    
    if not RQALPHA_AVAILABLE:
        print("❌ 错误: RQAlpha 未安装")
        print("  请运行: pip install rqalpha")
        sys.exit(1)
    
    # 确定要测试的基准代码
    if args.benchmark:
        benchmark_codes = [args.benchmark]
    else:
        benchmark_codes = test_benchmark_codes()
    
    print(f"要测试的基准代码: {', '.join(benchmark_codes)}")
    print("")
    
    results = []
    
    # 简单查询测试
    if args.mode in ["simple", "both"]:
        print("=" * 80)
        print("步骤 1: 简单查询测试（查询合约信息）")
        print("=" * 80)
        
        for code in benchmark_codes:
            print(f"\n测试基准代码: {code}")
            result = test_benchmark_support_simple(code)
            results.append(("simple", result))
            
            if result["instrument_exists"]:
                print(f"  ✓ 合约存在")
                if result["instrument_info"]:
                    info = result["instrument_info"]
                    print(f"    代码: {info.get('order_book_id')}")
                    print(f"    名称: {info.get('symbol')}")
                    print(f"    类型: {info.get('type')}")
                    print(f"    上市日期: {info.get('listed_date')}")
                    print(f"    退市日期: {info.get('de_listed_date', '未退市')}")
            else:
                print(f"  ⚠ 无法通过简单查询验证（可能需要回测环境）")
                if result["error"]:
                    error_msg = result['error']
                    if "stack is empty" in error_msg.lower():
                        print(f"    提示: RQAlpha API 需要在环境中调用")
                        print(f"    建议: 使用 --mode backtest 进行实际回测验证")
                    else:
                        print(f"    错误: {error_msg}")
    
    # 回测测试
    if args.mode in ["backtest", "both"]:
        print("\n" + "=" * 80)
        print("步骤 2: 回测测试（测试基准数据是否可加载）")
        print("=" * 80)
        print("注意: 此测试会运行最小回测，可能需要一些时间...")
        print("")
        
        for code in benchmark_codes:
            print(f"\n测试基准代码: {code}")
            result = test_benchmark_with_minimal_backtest(
                code,
                args.start_date,
                args.end_date,
            )
            results.append(("backtest", result))
            
            if result["status"] == "success":
                if result.get("benchmark_loaded"):
                    print(f"  ✓ 基准数据加载成功")
                else:
                    print(f"  ⚠ 回测成功但基准数据未加载")
                    if result.get("error"):
                        print(f"    原因: {result['error']}")
            elif result["status"] == "error":
                print(f"  ❌ 回测失败")
                if result.get("error"):
                    print(f"    错误: {result['error']}")
            else:
                print(f"  ⚠ 回测完成但有警告")
                if result.get("error"):
                    print(f"    警告: {result['error']}")
    
    # 生成总结报告
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    # 按模式分组结果
    simple_results = [r for mode, r in results if mode == "simple"]
    backtest_results = [r for mode, r in results if mode == "backtest"]
    
    if simple_results:
        print("\n【简单查询测试结果】")
        for result in simple_results:
            code = result["benchmark_code"]
            if result["instrument_exists"]:
                print(f"  ✓ {code}: 合约存在")
            else:
                print(f"  ❌ {code}: 合约不存在 - {result.get('error', '未知错误')}")
    
    if backtest_results:
        print("\n【回测测试结果】")
        for result in backtest_results:
            code = result["benchmark_code"]
            if result.get("status") == "success" and result.get("benchmark_loaded"):
                print(f"  ✓ {code}: 基准数据可正常加载")
            elif result.get("status") == "success":
                print(f"  ⚠ {code}: 回测成功但基准数据未加载")
            else:
                print(f"  ❌ {code}: 回测失败 - {result.get('error', '未知错误')}")
    
    # 推荐
    print("\n【推荐】")
    supported_codes = []
    # 注意：results 中的每个元素是 (mode, result_dict) 的 tuple
    for mode, result_dict in results:
        if result_dict.get("instrument_exists") or result_dict.get("benchmark_loaded"):
            code = result_dict["benchmark_code"]
            if code not in supported_codes:
                supported_codes.append(code)
    
    if supported_codes:
        print(f"  以下基准代码可用: {', '.join(supported_codes)}")
    else:
        print("  ⚠ 未找到可用的基准代码")
        print("  建议:")
        print("    1. 检查 RQAlpha 数据源是否包含所需指数数据")
        print("    2. 尝试使用常见的基准代码（如 000300.XSHG）")
        print("    3. 检查 RQAlpha 版本和文档")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

