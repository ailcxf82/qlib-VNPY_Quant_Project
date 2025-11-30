"""
RQAlpha 回测执行脚本：加载配置并运行策略。
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict

import pandas as pd

try:
    from rqalpha import run_file
    from rqalpha.utils.config import parse_config
    RQALPHA_AVAILABLE = True
except ImportError:
    RQALPHA_AVAILABLE = False
    logging.warning("RQAlpha 未安装，请运行: pip install rqalpha")

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils import load_yaml_config


def _resolve_path(path: Optional[str], *, prefer_project_root: bool = False) -> Optional[str]:
    """将输入路径转换为绝对路径，优先匹配当前工作目录，否则回退到项目根目录。"""
    if not path:
        return path
    normalized = path.replace("/", os.sep)
    if os.path.isabs(normalized):
        return normalized
    if not prefer_project_root:
        cwd_candidate = os.path.abspath(normalized)
        if os.path.exists(cwd_candidate):
            return cwd_candidate
    return os.path.join(project_root, normalized)


def convert_qlib_code_to_rqalpha(instrument) -> str:
    """
    将 Qlib 格式代码转换为 RQAlpha 格式。
    
    支持两种格式：
    1. 带交易所前缀：SH600000 -> 600000.XSHG, SZ000001 -> 000001.XSHE
    2. 纯数字代码：000001 -> 000001.XSHE, 600000 -> 600000.XSHG
    
    注意：会自动补全股票代码到6位（中国股票代码标准长度）
    """
    # 转换为字符串类型
    code_str = str(instrument).strip()
    
    # 如果已经有交易所前缀（SH/SZ），提取代码部分
    if code_str.startswith("SH"):
        code = code_str[2:]
        # 补全到6位
        code = code.zfill(6)
        return f"{code}.XSHG"
    elif code_str.startswith("SZ"):
        code = code_str[2:]
        # 补全到6位
        code = code.zfill(6)
        return f"{code}.XSHE"
    
    # 如果没有交易所前缀，先补全到6位
    code_str = code_str.zfill(6)
    
    # 根据代码前缀判断交易所
    # 先判断特殊情况（更具体的匹配）
    if code_str.startswith("688") or code_str.startswith("689"):
        # 科创板属于上海
        return f"{code_str}.XSHG"
    elif code_str.startswith("300"):
        # 创业板属于深圳
        return f"{code_str}.XSHE"
    # 再判断一般情况
    elif code_str.startswith("6") or code_str.startswith("9"):
        # 6、9开头的是上海（SH）
        return f"{code_str}.XSHG"
    elif code_str.startswith("0") or code_str.startswith("3"):
        # 0、3开头的是深圳（SZ）
        return f"{code_str}.XSHE"
    else:
        # 未知格式，返回原样（可能会报错，但至少不会崩溃）
        logging.warning(f"无法识别股票代码格式: {code_str}，返回原值")
        return code_str


def prepare_prediction_file(prediction_path: str, output_path: str):
    """
    预处理预测文件，将 Qlib 代码格式转换为 RQAlpha 格式。
    
    参数:
        prediction_path: 原始预测文件路径
        output_path: 输出文件路径
    """
    # 读取 CSV 时指定 instrument 列为字符串类型，保留前导零
    df = pd.read_csv(prediction_path, dtype={"instrument": str})
    df["datetime"] = pd.to_datetime(df["datetime"])
    
    # 转换代码格式
    df["rq_code"] = df["instrument"].apply(convert_qlib_code_to_rqalpha)
    
    # 保存转换后的文件
    df[["datetime", "rq_code", "final"]].to_csv(output_path, index=False)
    logging.info(f"预测文件已转换并保存到: {output_path}")


def load_industry_map(industry_path: Optional[str]) -> Optional[Dict[str, str]]:
    """加载行业映射文件。"""
    if not industry_path or not os.path.exists(industry_path):
        return None
    
    # 读取 CSV 时指定 instrument 列为字符串类型，保留前导零
    df = pd.read_csv(industry_path, dtype={"instrument": str})
    if "instrument" not in df.columns or "industry" not in df.columns:
        raise ValueError("行业文件需包含 instrument 与 industry 列")
    
    # 转换代码格式
    industry_map = {}
    for _, row in df.iterrows():
        rq_code = convert_qlib_code_to_rqalpha(row["instrument"])
        industry_map[rq_code] = row["industry"]
    
    return industry_map


def run_rqalpha_backtest(
    rqalpha_config_path: str,
    prediction_path: str,
    industry_path: Optional[str] = None,
    strategy_path: Optional[str] = None,
):
    """
    执行 RQAlpha 回测。
    
    参数:
        rqalpha_config_path: RQAlpha 配置文件路径
        prediction_path: 预测信号文件路径
        industry_path: 行业映射文件路径（可选）
        strategy_path: 策略脚本路径（默认使用内置策略）
    """
    # 规范化路径，兼容从项目根目录外部调用
    rqalpha_config_path = _resolve_path(rqalpha_config_path)
    prediction_path = _resolve_path(prediction_path)
    industry_path = _resolve_path(industry_path)
    if strategy_path is not None:
        strategy_path = _resolve_path(strategy_path)

    # 加载 RQAlpha 配置
    rqalpha_cfg = load_yaml_config(rqalpha_config_path)
    
    # 从预测文件读取日期范围（如果配置中未指定）
    df_pred = pd.read_csv(prediction_path)
    df_pred["datetime"] = pd.to_datetime(df_pred["datetime"])
    pred_start = df_pred["datetime"].min().strftime("%Y-%m-%d")
    pred_end = df_pred["datetime"].max().strftime("%Y-%m-%d")
    
    # 预处理预测文件
    temp_prediction_path = os.path.join(
        os.path.dirname(prediction_path),
        f"rqalpha_{os.path.basename(prediction_path)}"
    )
    prepare_prediction_file(prediction_path, temp_prediction_path)
    
    # 加载行业映射
    industry_map = load_industry_map(industry_path)
    
    # 构建 RQAlpha 配置字典
    base_config = rqalpha_cfg.get("base", {})
    commission_config = rqalpha_cfg.get("commission", {})
    slippage_config = rqalpha_cfg.get("slippage", {})
    trading_config = rqalpha_cfg.get("trading", {})
    risk_config = rqalpha_cfg.get("risk", {})
    output_config = rqalpha_cfg.get("output", {})

    # 输出目录使用项目相对路径时，转换为绝对路径
    output_dir = _resolve_path(output_config.get("output_dir", "data/backtest/rqalpha"), prefer_project_root=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用预测文件的日期范围（如果配置中未指定）
    start_date = base_config.get("start_date") or pred_start
    end_date = base_config.get("end_date") or pred_end
    
    # RQAlpha 配置
    config_dict = {
        "base": {
            "start_date": start_date,
            "end_date": end_date,
            "accounts": {
                "stock": base_config.get("initial_cash", 10000000),
            },
            "benchmark": base_config.get("benchmark", "000300.XSHG"),
            "data_bundle_path": base_config.get("data_bundle_path"),
        },
        "mod": {
            "sys_accounts": {
                "enabled": True,
            },
            "sys_progress": {
                "enabled": True,
                "show": True,
            },
            "sys_analyser": {
                "enabled": True,
                "output_file": os.path.join(output_dir, "report.json"),
                "benchmark": base_config.get("benchmark", "000300.XSHG"),  # 修复弃用警告
            },
            "sys_simulation": {
                "enabled": True,
                "matching_type": "current_bar",
                "price_type": "limit",
                "slippage": slippage_config.get("slippage_rate", 0.0001),
                "commission_multiplier": 1.0,
            },
        },
        "extra": {
            "log_level": "INFO",
        },
    }
    
    # 手续费配置（通过 mod 配置）
    if commission_config.get("commission_rate"):
        config_dict["mod"]["sys_simulation"]["commission"] = commission_config["commission_rate"]
    if commission_config.get("min_commission"):
        config_dict["mod"]["sys_simulation"]["min_commission"] = commission_config["min_commission"]
    
    # 交易限制（T+1 交易）
    if not trading_config.get("day_trade", False):
        # T+1 交易：使用 next_bar 匹配，当日下单次日成交
        config_dict["mod"]["sys_simulation"]["matching_type"] = "next_bar"
    
    # 策略参数（通过 context.config 传递）
    strategy_params = {
        "prediction_file": temp_prediction_path,
        "max_position": risk_config.get("max_position", 0.3),
        "max_stock_weight": risk_config.get("max_stock_weight", 0.05),
        "max_industry_weight": risk_config.get("max_industry_weight", 0.2),
        "top_k": risk_config.get("top_k", 50),
    }
    if industry_map:
        strategy_params["industry_map"] = industry_map
    
    config_dict["extra"]["context_vars"] = strategy_params
    
    # 策略脚本路径
    if strategy_path is None:
        strategy_path = os.path.join(
            os.path.dirname(__file__),
            "rqalpha_strategy.py"
        )
    else:
        strategy_path = _resolve_path(strategy_path)
    
    if not RQALPHA_AVAILABLE:
        raise ImportError("RQAlpha 未安装，请运行: pip install rqalpha")
    
    # 执行回测
    logging.info("开始执行 RQAlpha 回测...")
    logging.info(f"策略文件: {strategy_path}")
    logging.info(f"预测文件: {temp_prediction_path}")
    
    # RQAlpha 的回测调用方式
    try:
        result = run_file(
            strategy_path,
            config=config_dict,
        )
    except Exception as e:
        logging.error(f"RQAlpha 回测执行失败: {e}")
        raise
    
    # 保存回测结果
    # 提取回测结果
    if result and hasattr(result, "summary"):
        summary_path = os.path.join(output_dir, "summary.json")
        import json
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(result.summary, f, ensure_ascii=False, indent=2)
        logging.info(f"回测摘要已保存到: {summary_path}")
    
    # 生成并保存图表
    try:
        logging.info(f"开始生成图表...{result}")
        generate_and_save_plot(result, output_dir)
    except Exception as e:
        logging.warning(f"生成图表失败: {e}，继续执行...")
    
    return result


def generate_and_save_plot(result, output_dir: str):
    """
    从 RQAlpha 回测结果中提取绘图数据，生成图表并保存。
    
    参数:
        result: RQAlpha 回测结果对象
        output_dir: 输出目录路径
    """
    import matplotlib.pyplot as plt
    import json
    import pandas as pd
    
    plot_data = {}
    
    # 方法0: 优先从 result 字典的 sys_analyser 中提取（这是 RQAlpha 的标准结构）
    if isinstance(result, dict) and "sys_analyser" in result:
        try:
            analyser = result["sys_analyser"]
            
            # 提取策略净值数据
            if "plots" in analyser and isinstance(analyser["plots"], pd.DataFrame):
                plots_df = analyser["plots"]
                if "strategy_nav" in plots_df.columns:
                    # 将 DataFrame 转换为字典格式（日期 -> 净值）
                    strategy_nav_dict = plots_df["strategy_nav"].to_dict()
                    plot_data["strategy_nav"] = strategy_nav_dict
                    logging.info(f"从 sys_analyser.plots 提取策略净值数据，共 {len(strategy_nav_dict)} 个数据点")
            
            # 提取基准净值数据
            if "benchmark_portfolio" in analyser and isinstance(analyser["benchmark_portfolio"], pd.DataFrame):
                benchmark_df = analyser["benchmark_portfolio"]
                if "unit_net_value" in benchmark_df.columns:
                    # 将 DataFrame 转换为字典格式（日期 -> 净值）
                    benchmark_nav_dict = benchmark_df["unit_net_value"].to_dict()
                    plot_data["benchmark_nav"] = benchmark_nav_dict
                    logging.info(f"从 sys_analyser.benchmark_portfolio 提取基准净值数据，共 {len(benchmark_nav_dict)} 个数据点")
            
            # 如果 plots 中没有 strategy_nav，尝试从 portfolio 中计算
            if "strategy_nav" not in plot_data and "portfolio" in analyser:
                portfolio_df = analyser["portfolio"]
                if isinstance(portfolio_df, pd.DataFrame) and "unit_net_value" in portfolio_df.columns:
                    nav_dict = portfolio_df["unit_net_value"].to_dict()
                    plot_data["strategy_nav"] = nav_dict
                    logging.info(f"从 sys_analyser.portfolio 提取策略净值数据，共 {len(nav_dict)} 个数据点")
                    
        except Exception as e:
            logging.debug(f"从 sys_analyser 提取绘图数据失败: {e}")
    
    # 方法1: 尝试从 result 对象的环境对象中获取绘图数据
    if result:
        try:
            # RQAlpha 的 plot 数据通常存储在环境对象的 plot_store 中
            if hasattr(result, "env"):
                env = result.env
                # 尝试多种方式获取 plot_store
                plot_store = None
                if hasattr(env, "plot_store"):
                    plot_store = env.plot_store
                elif hasattr(env, "plot_data"):
                    plot_store = env.plot_data
                elif hasattr(env, "_plot_store"):
                    plot_store = env._plot_store
                
                if plot_store:
                    if hasattr(plot_store, "get_plot_data"):
                        plot_data = plot_store.get_plot_data()
                    elif hasattr(plot_store, "data"):
                        plot_data = plot_store.data
                    elif hasattr(plot_store, "_data"):
                        plot_data = plot_store._data
                    elif isinstance(plot_store, dict):
                        plot_data = plot_store
                    elif hasattr(plot_store, "__dict__"):
                        # 尝试从对象的属性中提取
                        plot_data = {k: v for k, v in plot_store.__dict__.items() if not k.startswith("_")}
        except Exception as e:
            logging.debug(f"从环境对象提取绘图数据失败: {e}")
    
    # 方法2: 尝试从 report.json 中读取绘图数据
    if not plot_data:
        report_path = os.path.join(output_dir, "report.json")
        if os.path.exists(report_path):
            try:
                with open(report_path, "r", encoding="utf-8") as f:
                    report = json.load(f)
                    # RQAlpha 的 plot 数据可能在 report 的不同字段中
                    if "plot" in report:
                        plot_data = report["plot"]
                    elif "plots" in report:
                        plot_data = report["plots"]
                    elif "plot_data" in report:
                        plot_data = report["plot_data"]
            except Exception as e:
                logging.debug(f"从 report.json 读取绘图数据失败: {e}")
    
    # 方法3: 尝试从 result 对象中直接提取
    if not plot_data and result:
        try:
            if hasattr(result, "plot_data"):
                plot_data = result.plot_data
            elif hasattr(result, "plots"):
                plot_data = result.plots
        except Exception as e:
            logging.debug(f"从 result 对象提取绘图数据失败: {e}")
    
    # 方法4: 如果仍然没有数据，尝试从 portfolio 的历史数据生成
    if not plot_data and result:
        try:
            portfolio = None
            if hasattr(result, "portfolio"):
                portfolio = result.portfolio
            elif hasattr(result, "env") and hasattr(result.env, "portfolio"):
                portfolio = result.env.portfolio
            
            if portfolio:
                # 尝试多种方式获取历史净值数据
                nav_history = None
                if hasattr(portfolio, "unit_net_value_history"):
                    nav_history = portfolio.unit_net_value_history
                elif hasattr(portfolio, "total_value_history"):
                    initial_value = getattr(portfolio, "initial_value", 10000000)
                    if hasattr(portfolio, "total_value_history") and portfolio.total_value_history:
                        nav_history = [v / initial_value for v in portfolio.total_value_history]
                elif hasattr(portfolio, "_unit_net_value_history"):
                    nav_history = portfolio._unit_net_value_history
                
                if nav_history:
                    plot_data["strategy_nav"] = nav_history
                    logging.info(f"从 portfolio 历史数据生成策略净值曲线，共 {len(nav_history)} 个数据点")
        except Exception as e:
            logging.debug(f"从 portfolio 历史数据生成绘图数据失败: {e}")
    
    # 如果还是没有数据，记录警告并返回
    if not plot_data:
        logging.warning("未找到绘图数据，无法生成图表。请确保策略中正确调用了 plot() 函数。")
        logging.info("提示：plot() 函数的数据会被 RQAlpha 自动收集，如果仍然无法生成图表，")
        logging.info("     可能需要检查 RQAlpha 版本或使用 RQAlpha 的 Web 界面查看报告。")
        return
    
    # 创建图表
    plt.figure(figsize=(12, 6))
    
    # 绘制策略净值曲线
    if "strategy_nav" in plot_data:
        strategy_data = plot_data["strategy_nav"]
        if isinstance(strategy_data, dict):
            # 如果是字典格式，提取日期和净值
            dates = list(strategy_data.keys())
            values = list(strategy_data.values())
            # 转换日期格式以便绘图
            if dates and isinstance(dates[0], (pd.Timestamp, pd.DatetimeIndex)):
                dates = [pd.Timestamp(d).date() if isinstance(d, (pd.Timestamp, pd.DatetimeIndex)) else d for d in dates]
            plt.plot(dates, values, label="策略净值", linewidth=2, color="#1f77b4")
        elif isinstance(strategy_data, list):
            # 如果是列表格式
            plt.plot(range(len(strategy_data)), strategy_data, label="策略净值", linewidth=2, color="#1f77b4")
        elif isinstance(strategy_data, pd.Series):
            # 如果是 pandas Series
            dates = strategy_data.index.tolist()
            values = strategy_data.values.tolist()
            if dates and isinstance(dates[0], (pd.Timestamp, pd.DatetimeIndex)):
                dates = [pd.Timestamp(d).date() if isinstance(d, (pd.Timestamp, pd.DatetimeIndex)) else d for d in dates]
            plt.plot(dates, values, label="策略净值", linewidth=2, color="#1f77b4")
    
    # 绘制基准净值曲线
    if "benchmark_nav" in plot_data:
        benchmark_data = plot_data["benchmark_nav"]
        if isinstance(benchmark_data, dict):
            dates = list(benchmark_data.keys())
            values = list(benchmark_data.values())
            # 转换日期格式以便绘图
            if dates and isinstance(dates[0], (pd.Timestamp, pd.DatetimeIndex)):
                dates = [pd.Timestamp(d).date() if isinstance(d, (pd.Timestamp, pd.DatetimeIndex)) else d for d in dates]
            plt.plot(dates, values, label="基准净值", linewidth=2, color="#ff7f0e", linestyle="--")
        elif isinstance(benchmark_data, list):
            plt.plot(range(len(benchmark_data)), benchmark_data, label="基准净值", linewidth=2, color="#ff7f0e", linestyle="--")
        elif isinstance(benchmark_data, pd.Series):
            # 如果是 pandas Series
            dates = benchmark_data.index.tolist()
            values = benchmark_data.values.tolist()
            if dates and isinstance(dates[0], (pd.Timestamp, pd.DatetimeIndex)):
                dates = [pd.Timestamp(d).date() if isinstance(d, (pd.Timestamp, pd.DatetimeIndex)) else d for d in dates]
            plt.plot(dates, values, label="基准净值", linewidth=2, color="#ff7f0e", linestyle="--")
    
    # 设置图表标题和标签
    plt.title("策略净值曲线对比", fontsize=16, fontweight="bold")
    plt.xlabel("交易日", fontsize=12)
    plt.ylabel("净值", fontsize=12)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 优化日期显示（如果有日期数据）
    if plot_data:
        try:
            from matplotlib.dates import DateFormatter
            ax = plt.gca()
            # 如果 x 轴是日期类型，格式化显示
            if hasattr(ax, 'xaxis') and len(ax.get_xticklabels()) > 0:
                # 尝试自动格式化日期
                plt.xticks(rotation=45, ha='right')
        except Exception:
            pass
    
    # 保存图片（必须在 show() 之前调用）
    plot_path = os.path.join(output_dir, "rqalpha_strategy_plot.png")
    plt.savefig(
        plot_path,
        dpi=300,                      # 分辨率（越高越清晰）
        bbox_inches='tight'           # 避免图表边缘被截断
    )
    logging.info(f"图表已保存到: {plot_path}")
    
    # 显示图表（可选：如需仅保存不显示，可注释此行）
    try:
        plt.show()
    except Exception as e:
        # 在某些环境中（如无 GUI），show() 可能失败，这是正常的
        logging.debug(f"显示图表失败（可能无 GUI 环境）: {e}")
    
    # 关闭图形以释放资源
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RQAlpha 回测执行")
    parser.add_argument("--rqalpha-config", type=str, required=True, help="RQAlpha 配置文件路径")
    parser.add_argument("--prediction", type=str, required=True, help="预测信号文件路径")
    parser.add_argument("--industry", type=str, default=None, help="行业映射文件路径")
    parser.add_argument("--strategy", type=str, default=None, help="策略脚本路径（可选）")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    
    run_rqalpha_backtest(
        args.rqalpha_config,
        args.prediction,
        args.industry,
        args.strategy,
    )

