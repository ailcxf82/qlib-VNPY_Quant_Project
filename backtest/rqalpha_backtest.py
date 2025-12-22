"""
RQAlpha 回测执行脚本：加载配置并运行策略。
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict

import pandas as pd
import numpy as np
import pathlib

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
    # elif code_str.startswith("1"):
    #     return f"{code_str}.XSHG"
    # elif code_str.startswith("5"):
    #     return f"{code_str}.XSHE"
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
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.normalize()

    # 如果预测文件的 datetime 已经被“对齐到下一交易日”（trade_date），则回测侧需要反向还原到 signal_date，
    # 否则在 RQAlpha 的 next_bar（T+1 成交）撮合下会变成隐性 T+2。
    # 约定：predictor.save_predictions 会写入 _meta_shifted_next_day=1
    shifted_flag = False
    if "_meta_shifted_next_day" in df.columns:
        try:
            shifted_flag = int(pd.to_numeric(df["_meta_shifted_next_day"], errors="coerce").fillna(0).max()) == 1
        except Exception:
            shifted_flag = False

    if shifted_flag:
        import bisect
        # 构造交易日历（尽量用 qlib，否则退化到工作日）
        min_dt = pd.Timestamp(df["datetime"].min()).normalize()
        max_dt = pd.Timestamp(df["datetime"].max()).normalize()
        start = min_dt - pd.Timedelta(days=60)
        end = max_dt + pd.Timedelta(days=5)
        cal = None
        try:
            from qlib.data import D  # type: ignore
            cal = D.calendar(start_time=start, end_time=end, freq="day")
            cal = pd.to_datetime(list(cal)).normalize()
        except Exception:
            cal = pd.bdate_range(start=start, end=end).normalize()
        cal = pd.Index(cal).sort_values().unique()

        cal_list = list(cal)
        unique_dts = pd.Index(df["datetime"].unique()).sort_values()
        mapping_prev = {}
        for d in unique_dts:
            # prev trading day（严格左侧）
            i = bisect.bisect_left(cal_list, pd.Timestamp(d))
            if i <= 0:
                mapping_prev[pd.Timestamp(d)] = None
            else:
                mapping_prev[pd.Timestamp(d)] = pd.Timestamp(cal_list[i - 1])

        prev_dt = df["datetime"].map(lambda x: mapping_prev.get(pd.Timestamp(x), None))
        before_rows = len(df)
        df = df.assign(datetime=pd.to_datetime(prev_dt))
        # 丢弃没有前一交易日的样本（通常只会影响最早一天）
        df = df.dropna(subset=["datetime"])
        after_rows = len(df)
        logging.info(
            "检测到预测文件已 shift 到 trade_date（_meta_shifted_next_day=1），回测侧反向还原到 signal_date：%s~%s，丢弃=%d",
            df["datetime"].min(),
            df["datetime"].max(),
            before_rows - after_rows,
        )
    
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
    *,
    full_invested: bool = False,
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
    
    # 构建 RQAlpha 配置字典
    base_config = rqalpha_cfg.get("base", {})
    commission_config = rqalpha_cfg.get("commission", {})
    slippage_config = rqalpha_cfg.get("slippage", {})
    trading_config = rqalpha_cfg.get("trading", {})
    risk_config = rqalpha_cfg.get("risk", {})
    output_config = rqalpha_cfg.get("output", {})
    extra_config = rqalpha_cfg.get("extra", {})

    # 输出目录使用项目相对路径时，转换为绝对路径
    output_dir = _resolve_path(output_config.get("output_dir", "data/backtest/rqalpha"), prefer_project_root=True)
    os.makedirs(output_dir, exist_ok=True)

    # 诊断：检查 RQAlpha 数据 bundle 是否覆盖到配置 end_date（很多“只跑到某天”的根因在这里）
    try:
        # 推断 bundle 路径：优先使用配置；否则使用默认 ~/.rqalpha/bundle
        bundle_path = base_config.get("data_bundle_path") if isinstance(base_config, dict) else None
        if not bundle_path:
            bundle_path = str(pathlib.Path.home() / ".rqalpha" / "bundle")
        bundle_path = str(bundle_path).replace("/", os.sep)
        if os.path.exists(bundle_path):
            # 读取 stocks.h5 的样本尾部 datetime，估算行情数据覆盖的最后一天
            import h5py

            stocks_h5 = os.path.join(bundle_path, "stocks.h5")
            if os.path.exists(stocks_h5):
                with h5py.File(stocks_h5, "r") as f:
                    keys = list(f.keys())
                    # 采样若干只股票估算（全量扫描会非常慢）
                    sample = keys[:50] if len(keys) >= 50 else keys
                    last_dt = None
                    for k in sample:
                        ds = f[k]
                        if ds.shape[0] <= 0:
                            continue
                        # datetime 字段为 int64: yyyymmddHHMMSS
                        v = ds[-1]["datetime"] if "datetime" in ds.dtype.names else None
                        if v is None:
                            continue
                        v = int(v)
                        if last_dt is None or v > last_dt:
                            last_dt = v
                    if last_dt is not None:
                        last_day = pd.to_datetime(str(last_dt)[:8], format="%Y%m%d")
                        logging.info("检测到 RQAlpha bundle=%s（stocks.h5 采样估算最后行情日=%s）", bundle_path, last_day.date())
                        # 若配置 end_date 晚于 bundle 覆盖，强提示用户
                        cfg_end = base_config.get("end_date") if isinstance(base_config, dict) else None
                        if cfg_end:
                            try:
                                cfg_end_ts = pd.Timestamp(cfg_end).normalize()
                                if cfg_end_ts > last_day.normalize():
                                    logging.warning(
                                        "⚠ 配置 end_date=%s 晚于 RQAlpha bundle 行情数据最后覆盖日=%s，回测会被自动截断到 %s。",
                                        cfg_end_ts.date(),
                                        last_day.date(),
                                        last_day.date(),
                                    )
                            except Exception:
                                pass
    except Exception:
        pass
    
    # 保存原始预测文件路径到输出目录，供后续分析使用（移到output_dir定义之后）
    import shutil
    try:
        # 将预测文件复制到输出目录，方便后续分析
        prediction_copy_path = os.path.join(output_dir, f"rqalpha_{os.path.basename(prediction_path)}")
        if os.path.exists(temp_prediction_path):
            shutil.copy2(temp_prediction_path, prediction_copy_path)
            logging.info(f"预测文件已复制到输出目录: {prediction_copy_path}")
    except Exception as e:
        logging.warning(f"复制预测文件到输出目录失败: {e}")
    
    # 加载行业映射
    industry_map = load_industry_map(industry_path)
    
    # 使用预测文件的日期范围（如果配置中未指定）
    cfg_start = base_config.get("start_date")
    cfg_end = base_config.get("end_date")
    start_date = cfg_start or pred_start
    end_date = cfg_end or pred_end
    logging.info(
        "回测周期选择：config(base)=%s~%s，pred(file)=%s~%s => use=%s~%s",
        cfg_start,
        cfg_end,
        pred_start,
        pred_end,
        start_date,
        end_date,
    )
    
    # 读取基准配置并输出日志
    benchmark_code = base_config.get("benchmark", "000300.XSHG")
    logging.info("="*80)
    logging.info("基准配置信息:")
    logging.info(f"  配置文件中的基准代码: {base_config.get('benchmark', '未设置（使用默认值）')}")
    logging.info(f"  实际使用的基准代码: {benchmark_code}")
    
    # 验证基准代码格式
    if benchmark_code:
        if not isinstance(benchmark_code, str):
            logging.warning(f"基准代码格式错误: 应为字符串，实际为 {type(benchmark_code)}")
        elif "." not in benchmark_code:
            logging.warning(f"基准代码格式可能错误: 缺少交易所后缀（应为 '代码.交易所' 格式，如 '000300.XSHG'）")
        else:
            code_part, exchange_part = benchmark_code.split(".", 1)
            logging.info(f"  基准代码解析: 代码={code_part}, 交易所={exchange_part}")
            if exchange_part not in ["XSHG", "XSHE"]:
                logging.warning(f"  交易所后缀 '{exchange_part}' 可能不正确，标准格式应为 XSHG（上海）或 XSHE（深圳）")
    logging.info("="*80)
    
    # RQAlpha 配置
    # 注意：base.benchmark 已被弃用，只使用 mod.sys_analyser.benchmark
    config_dict = {
        "base": {
            "start_date": start_date,
            "end_date": end_date,
            "accounts": {
                "stock": base_config.get("initial_cash", 10000000),
            },
            # "benchmark": benchmark_code,  # 已弃用，改用 mod.sys_analyser.benchmark
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
                "benchmark": benchmark_code,  # 使用新的配置方式（避免弃用警告）
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
            "log_level": extra_config.get("log_level", "INFO"),
            # 透传 context_vars（如 rebalance_interval）
            "context_vars": extra_config.get("context_vars", {}),
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
    # 先获取原有的 context_vars（可能包含 rebalance_interval 等配置）
    existing_context_vars = config_dict["extra"].get("context_vars", {})
    
    strategy_params = {
        "prediction_file": temp_prediction_path,
        "max_position": risk_config.get("max_position", 0.3),
        "max_stock_weight": risk_config.get("max_stock_weight", 0.05),
        "max_industry_weight": risk_config.get("max_industry_weight", 0.2),
        "top_k": risk_config.get("top_k", 50),
        "full_invested": bool(full_invested),
    }
    if industry_map:
        strategy_params["industry_map"] = industry_map
    
    # 合并原有配置（如 rebalance_interval）和新参数，避免覆盖
    config_dict["extra"]["context_vars"] = {**existing_context_vars, **strategy_params}
    
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
    logging.info(f"回测日期范围: {start_date} 至 {end_date}")
    logging.info(f"基准代码: {benchmark_code}")
    logging.info(f"初始资金: {base_config.get('initial_cash', 10000000)} 元")
    
    # 打印完整配置（用于调试）
    logging.debug("RQAlpha 完整配置:")
    import json
    logging.debug(json.dumps(config_dict, indent=2, default=str, ensure_ascii=False))
    
    # RQAlpha 的回测调用方式
    try:
        result = run_file(
            strategy_path,
            config=config_dict,
        )
    except Exception as e:
        logging.error(f"RQAlpha 回测执行失败: {e}")
        raise

    # 诊断：检查“实际跑到的周期”是否被数据源截断（常见原因：本地 bundle 数据只到某个日期）
    try:
        actual_start = None
        actual_end = None
        if isinstance(result, dict) and "sys_analyser" in result:
            analyser = result["sys_analyser"]
            if isinstance(analyser, dict) and "summary" in analyser and isinstance(analyser["summary"], dict):
                actual_start = analyser["summary"].get("start_date")
                actual_end = analyser["summary"].get("end_date")
        if actual_start and actual_end:
            cfg_start = config_dict.get("base", {}).get("start_date")
            cfg_end = config_dict.get("base", {}).get("end_date")
            if cfg_start and cfg_end:
                if str(actual_start) != str(cfg_start) or str(actual_end) != str(cfg_end):
                    logging.warning(
                        "⚠ 实际回测周期与配置不一致：cfg=%s~%s, actual=%s~%s。"
                        "若 actual_end 显著早于 cfg_end，通常是数据源/bundle 缺少后续交易日数据导致被截断。",
                        cfg_start,
                        cfg_end,
                        actual_start,
                        actual_end,
                    )
    except Exception:
        pass
    
    # 保存回测结果
    # 提取回测结果
    if result and hasattr(result, "summary"):
        summary_path = os.path.join(output_dir, "summary.json")
        import json
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(result.summary, f, ensure_ascii=False, indent=2)
        logging.info(f"回测摘要已保存到: {summary_path}")
    
    # 验证基准数据是否被正确加载
    try:
        if isinstance(result, dict) and "sys_analyser" in result:
            analyser = result["sys_analyser"]
            if "benchmark_portfolio" in analyser:
                benchmark_df = analyser.get("benchmark_portfolio")
                if benchmark_df is not None and not (isinstance(benchmark_df, pd.DataFrame) and benchmark_df.empty):
                    logging.info(f"✓ 基准数据已成功加载，数据点数量: {len(benchmark_df) if isinstance(benchmark_df, pd.DataFrame) else 'N/A'}")
                else:
                    logging.warning(f"⚠ 基准数据为空或未加载，基准代码 '{benchmark_code}' 可能无效或数据源中不存在")
                    logging.warning(f"  请检查：1) 基准代码是否正确 2) 数据源是否包含该基准数据 3) 回测日期范围内是否有数据")
            else:
                logging.warning(f"⚠ 回测结果中未找到基准数据，基准代码 '{benchmark_code}' 可能无效")
    except Exception as e:
        logging.debug(f"检查基准数据时出错: {e}")
    
    # 提取并保存详细的回测信息（投资明细、盈亏状态、效率指标）
    try:
        logging.info("开始提取详细的回测信息...")
        # 传递预测文件路径，用于增强交易明细
        extract_and_save_detailed_results(result, output_dir, temp_prediction_path)
    except Exception as e:
        logging.warning(f"提取详细回测信息失败: {e}，继续执行...")
        import traceback
        traceback.print_exc()
    
    # 生成并保存图表
    try:
        logging.info(f"开始生成图表...{result}")
        generate_and_save_plot(result, output_dir, benchmark_code)
    except Exception as e:
        logging.warning(f"生成图表失败: {e}，继续执行...")
    
    return result


def enhance_trades_with_prediction_and_pnl(trades_df: pd.DataFrame, output_dir: str, prediction_file_path: Optional[str] = None) -> pd.DataFrame:
    """
    增强交易明细：添加预测值、成本和收益信息
    
    参数:
        trades_df: 原始交易明细 DataFrame
        output_dir: 输出目录（用于查找预测文件）
    
    返回:
        增强后的交易明细 DataFrame
    """
    import glob
    
    # 创建副本，避免修改原数据
    enhanced_df = trades_df.copy()
    
    # 0. 处理索引问题：如果 datetime 既是索引又是列，安全地重置索引
    def _ensure_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
        # 如果已有 datetime 列且索引也包含 datetime，先丢弃索引中的该层，避免重复列
        if isinstance(df.index, pd.MultiIndex) and "datetime" in df.index.names:
            if "datetime" in df.columns:
                df = df.reset_index(level="datetime", drop=True)
            else:
                df = df.reset_index()
        elif df.index.name == "datetime":
            if "datetime" in df.columns:
                df = df.reset_index(drop=True)
            else:
                df = df.reset_index()
        
        # 如果仍不存在 datetime 列，尝试从索引或其他列补充
        if "datetime" not in df.columns:
            if isinstance(df.index, pd.MultiIndex) and "datetime" in df.index.names:
                df = df.reset_index(level="datetime")
            elif df.index.name == "datetime":
                df = df.reset_index()
            elif "trading_datetime" in df.columns:
                df["datetime"] = df["trading_datetime"]
        return df
    
    enhanced_df = _ensure_datetime_column(enhanced_df)
    
    # 1. 尝试从预测文件中读取预测值
    prediction_file = prediction_file_path
    prediction_data = None
    
    # 如果未提供预测文件路径，尝试自动查找
    if prediction_file is None or not os.path.exists(prediction_file):
        # 查找预测文件（可能在 output_dir 的父目录或当前目录）
        search_dirs = [
            output_dir,
            os.path.dirname(output_dir),
            os.path.join(os.path.dirname(output_dir), "predictions"),
        ]
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                # 查找 rqalpha_ 开头的预测文件
                pattern = os.path.join(search_dir, "rqalpha_*.csv")
                files = glob.glob(pattern)
                if files:
                    # 使用最新的文件
                    prediction_file = max(files, key=os.path.getmtime)
                    break
        
        # 如果没找到 rqalpha_ 开头的文件，尝试找 pred_ 开头的
        if prediction_file is None or not os.path.exists(prediction_file):
            for search_dir in search_dirs:
                if os.path.exists(search_dir):
                    pattern = os.path.join(search_dir, "pred_*.csv")
                    files = glob.glob(pattern)
                    if files:
                        prediction_file = max(files, key=os.path.getmtime)
                        break
    
    if prediction_file and os.path.exists(prediction_file):
        try:
            # 读取预测文件
            pred_df = pd.read_csv(prediction_file)
            if "datetime" in pred_df.columns:
                pred_df["datetime"] = pd.to_datetime(pred_df["datetime"])
                # 确定代码列名
                code_col = None
                for col in ["rq_code", "instrument", "order_book_id"]:
                    if col in pred_df.columns:
                        code_col = col
                        break
                
                if code_col and "final" in pred_df.columns:
                    # 创建预测值字典：{(日期, 代码): 预测值}
                    pred_df["date"] = pred_df["datetime"].dt.date
                    prediction_data = {}
                    for _, row in pred_df.iterrows():
                        date = row["date"]
                        code = str(row[code_col])
                        pred_value = row["final"]
                        prediction_data[(date, code)] = pred_value
                    
                    logging.info(f"成功加载预测文件: {prediction_file}，共 {len(prediction_data)} 条预测记录")
        except Exception as e:
            logging.warning(f"读取预测文件失败: {e}")
            prediction_data = None
    else:
        logging.warning(f"未找到预测文件，无法添加预测值信息")
    
    # 2. 添加预测值列
    enhanced_df["prediction_value"] = None
    if prediction_data and "datetime" in enhanced_df.columns and "order_book_id" in enhanced_df.columns:
        enhanced_df["datetime"] = pd.to_datetime(enhanced_df["datetime"])
        enhanced_df["date"] = enhanced_df["datetime"].dt.date
        
        for idx, row in enhanced_df.iterrows():
            date = row["date"]
            code = str(row["order_book_id"])
            key = (date, code)
            if key in prediction_data:
                enhanced_df.at[idx, "prediction_value"] = prediction_data[key]
        
        # 删除临时日期列
        enhanced_df = enhanced_df.drop(columns=["date"], errors="ignore")
    
    # 3. 计算成本和收益
    enhanced_df["cost"] = None  # 买入成本（包含手续费）
    enhanced_df["profit"] = None  # 卖出收益（扣除手续费）
    enhanced_df["pnl"] = None  # 盈亏（profit - cost，对于卖出交易）
    
    # 按股票代码分组，计算累计成本和收益
    if "order_book_id" in enhanced_df.columns and "side" in enhanced_df.columns:
        # 初始化列
        enhanced_df["cumulative_cost"] = 0.0
        enhanced_df["cumulative_quantity"] = 0
        enhanced_df["avg_cost"] = 0.0
        
        # 按股票代码分组处理
        for code in enhanced_df["order_book_id"].unique():
            code_mask = enhanced_df["order_book_id"] == code
            code_trades = enhanced_df[code_mask].copy()
            
            # 确保 datetime 列存在后再排序
            if "datetime" in code_trades.columns:
                code_trades = code_trades.sort_values("datetime")
            elif "trading_datetime" in code_trades.columns:
                code_trades = code_trades.sort_values("trading_datetime")
            
            cumulative_cost = 0.0
            cumulative_quantity = 0
            avg_cost = 0.0
            
            for idx, row in code_trades.iterrows():
                side = row.get("side", "")
                quantity = row.get("last_quantity", 0)
                price = row.get("last_price", 0.0)
                commission = row.get("commission", 0.0)
                tax = row.get("tax", 0.0)
                
                if side == "BUY":
                    # 买入：计算成本
                    trade_amount = quantity * price
                    total_cost = trade_amount + commission + tax
                    enhanced_df.at[idx, "cost"] = total_cost
                    
                    # 更新累计成本和数量（用于计算平均成本）
                    cumulative_cost += total_cost
                    cumulative_quantity += quantity
                    if cumulative_quantity > 0:
                        avg_cost = cumulative_cost / cumulative_quantity
                    
                    enhanced_df.at[idx, "cumulative_cost"] = cumulative_cost
                    enhanced_df.at[idx, "cumulative_quantity"] = cumulative_quantity
                    enhanced_df.at[idx, "avg_cost"] = avg_cost
                    
                elif side == "SELL":
                    # 卖出：计算收益
                    trade_amount = quantity * price
                    total_cost_fees = commission + tax
                    net_amount = trade_amount - total_cost_fees
                    
                    # 收益 = 卖出金额 - 手续费 - 对应持仓的成本
                    if cumulative_quantity > 0 and avg_cost > 0:
                        # 使用平均成本法计算收益
                        cost_basis = quantity * avg_cost
                        profit = net_amount - cost_basis
                        enhanced_df.at[idx, "profit"] = net_amount
                        enhanced_df.at[idx, "pnl"] = profit
                        
                        # 更新累计成本和数量
                        cumulative_cost -= cost_basis
                        cumulative_quantity -= quantity
                        if cumulative_quantity > 0:
                            avg_cost = cumulative_cost / cumulative_quantity
                        else:
                            avg_cost = 0.0
                        
                        enhanced_df.at[idx, "cumulative_cost"] = cumulative_cost
                        enhanced_df.at[idx, "cumulative_quantity"] = cumulative_quantity
                        enhanced_df.at[idx, "avg_cost"] = avg_cost
                    else:
                        # 如果没有持仓成本信息，只计算净卖出金额
                        enhanced_df.at[idx, "profit"] = net_amount
                        enhanced_df.at[idx, "pnl"] = net_amount
        
        # 删除临时列
        enhanced_df = enhanced_df.drop(columns=["cumulative_cost", "cumulative_quantity", "avg_cost"], errors="ignore")
    
    return enhanced_df


def extract_and_save_detailed_results(result, output_dir: str, prediction_file_path: Optional[str] = None):
    """
    从 RQAlpha 回测结果中提取详细的投资明细、盈亏状态和效率指标。
    
    参数:
        result: RQAlpha 回测结果对象
        output_dir: 输出目录路径
        prediction_file_path: 预测文件路径（可选，用于增强交易明细）
    """
    import json
    
    detailed_results = {
        "投资明细": [],
        "盈亏状态": {},
        "效率指标": {},
    }
    
    # 1. 提取投资明细（持仓记录）
    try:
        if isinstance(result, dict) and "sys_analyser" in result:
            analyser = result["sys_analyser"]
            
            # 提取持仓记录
            if "positions" in analyser and isinstance(analyser["positions"], pd.DataFrame):
                positions_df = analyser["positions"]
                if not positions_df.empty:
                    # 保存持仓明细
                    positions_path = os.path.join(output_dir, "positions_detail.csv")
                    positions_df.to_csv(positions_path, index=False, encoding="utf-8-sig")
                    logging.info(f"持仓明细已保存到: {positions_path}")
                    logging.info(f"持仓明细共 {len(positions_df)} 条记录")
                    
                    # 打印部分持仓明细
                    if len(positions_df) > 0:
                        logging.info("\n" + "="*80)
                        logging.info("持仓明细（前10条）:")
                        logging.info("="*80)
                        print_df = positions_df.head(10)
                        for col in print_df.columns:
                            logging.info(f"  {col}: {print_df[col].tolist()}")
            
            # 提取交易记录
            if "trades" in analyser and isinstance(analyser["trades"], pd.DataFrame):
                trades_df = analyser["trades"]
                if not trades_df.empty:
                    # 增强交易明细：添加预测值、成本和收益
                    logging.info("开始增强交易明细：添加预测值、成本和收益信息...")
                    trades_df = enhance_trades_with_prediction_and_pnl(trades_df, output_dir, prediction_file_path)
                    
                    # 保存交易明细
                    trades_path = os.path.join(output_dir, "trades_detail.csv")
                    trades_df.to_csv(trades_path, index=False, encoding="utf-8-sig")
                    logging.info(f"交易明细已保存到: {trades_path}")
                    logging.info(f"交易明细共 {len(trades_df)} 条记录（已增强：包含预测值、成本、收益）")
                    
                    # 打印部分交易明细
                    if len(trades_df) > 0:
                        logging.info("\n" + "="*80)
                        logging.info("交易明细（前10条，包含预测值、成本、收益）:")
                        logging.info("="*80)
                        # 选择关键列显示
                        key_cols = ["datetime", "order_book_id", "side", "last_price", "last_quantity", 
                                   "prediction_value", "cost", "profit", "commission"]
                        available_cols = [col for col in key_cols if col in trades_df.columns]
                        if available_cols:
                            print_df = trades_df[available_cols].head(10)
                            logging.info(print_df.to_string(index=False))
                        else:
                            print_df = trades_df.head(10)
                            for col in print_df.columns:
                                logging.info(f"  {col}: {print_df[col].tolist()}")
            
            # 提取盈亏状态
            if "portfolio" in analyser and isinstance(analyser["portfolio"], pd.DataFrame):
                portfolio_df = analyser["portfolio"]
                if not portfolio_df.empty:
                    # 计算盈亏状态
                    if "total_value" in portfolio_df.columns:
                        initial_value = portfolio_df["total_value"].iloc[0] if len(portfolio_df) > 0 else 0
                        final_value = portfolio_df["total_value"].iloc[-1] if len(portfolio_df) > 0 else 0
                        total_return = (final_value - initial_value) / initial_value if initial_value > 0 else 0
                        
                        detailed_results["盈亏状态"] = {
                            "初始资金": float(initial_value),
                            "最终资金": float(final_value),
                            "总收益": float(final_value - initial_value),
                            "总收益率": float(total_return * 100),
                        }
                        
                        # 打印盈亏状态
                        logging.info("\n" + "="*80)
                        logging.info("盈亏状态:")
                        logging.info("="*80)
                        logging.info(f"  初始资金: {initial_value:,.2f} 元")
                        logging.info(f"  最终资金: {final_value:,.2f} 元")
                        logging.info(f"  总收益: {final_value - initial_value:,.2f} 元")
                        logging.info(f"  总收益率: {total_return * 100:.2f}%")
            
            # 提取效率指标
            if "summary" in analyser:
                summary = analyser["summary"]
                if isinstance(summary, dict):
                    detailed_results["效率指标"] = summary
                    
                    # 打印效率指标
                    logging.info("\n" + "="*80)
                    logging.info("效率指标:")
                    logging.info("="*80)
                    for key, value in summary.items():
                        if isinstance(value, (int, float)):
                            logging.info(f"  {key}: {value:.4f}")
                        else:
                            logging.info(f"  {key}: {value}")
        
        # 2. 从 report.json 中提取更多信息
        report_path = os.path.join(output_dir, "report.json")
        if os.path.exists(report_path):
            # 尝试多种编码读取 report.json，避免编码报错
            report = None
            encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin-1"]
            for enc in encodings:
                try:
                    with open(report_path, "r", encoding=enc) as f:
                        report = json.load(f)
                    logging.info(f"成功使用 {enc} 编码读取 report.json")
                    break
                except Exception:
                    continue
            if report is None:
                try:
                    with open(report_path, "rb") as f:
                        content = f.read()
                    # 可选依赖：chardet。没有就退化到 utf-8 ignore。
                    try:
                        import chardet  # type: ignore
                        detected = chardet.detect(content)
                        enc = detected.get("encoding", "utf-8")
                        logging.info(f"chardet 检测 report.json 编码: {enc}")
                        report = json.loads(content.decode(enc))
                    except Exception:
                        report = json.loads(content.decode("utf-8", errors="ignore"))
                except Exception as e:
                    logging.warning(f"读取 report.json 失败: {e}")
                    report = None
            
            # 提取更多效率指标（需要在 report 不为 None 时执行）
            if report is not None:
                # 提取更多效率指标
                if "summary" in report:
                    summary = report["summary"]
                    if isinstance(summary, dict):
                        # 合并效率指标
                        for key, value in summary.items():
                            if key not in detailed_results["效率指标"]:
                                detailed_results["效率指标"][key] = value
                        
                        # 打印更多效率指标
                        logging.info("\n" + "="*80)
                        logging.info("更多效率指标（从 report.json）:")
                        logging.info("="*80)
                        for key, value in summary.items():
                            if isinstance(value, (int, float)):
                                logging.info(f"  {key}: {value:.4f}")
                            else:
                                logging.info(f"  {key}: {value}")
                
                # 提取交易统计
                if "trades" in report:
                    trades_info = report["trades"]
                    if isinstance(trades_info, dict):
                        logging.info("\n" + "="*80)
                        logging.info("交易统计:")
                        logging.info("="*80)
                        for key, value in trades_info.items():
                            if isinstance(value, (int, float)):
                                logging.info(f"  {key}: {value:.4f}")
                            else:
                                logging.info(f"  {key}: {value}")
            else:
                logging.warning("report.json 读取失败，无法提取额外信息")
    
    except Exception as e:
        logging.error(f"提取详细回测信息时出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 保存详细结果到 JSON
    detailed_path = os.path.join(output_dir, "detailed_results.json")
    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2, default=str)
    logging.info(f"详细回测结果已保存到: {detailed_path}")


def generate_and_save_plot(result, output_dir: str, benchmark_code: Optional[str] = None):
    """
    从 RQAlpha 回测结果中提取绘图数据，生成图表并保存。
    
    参数:
        result: RQAlpha 回测结果对象
        output_dir: 输出目录路径
        benchmark_code: 基准代码（用于在图表中显示）
    """
    import matplotlib.pyplot as plt
    import json
    import pandas as pd
    import matplotlib
    from matplotlib import font_manager
    from matplotlib.font_manager import FontProperties
    from typing import Tuple
    
    plot_data = {}

    # 注意：plt.style.use(...) 可能会覆盖字体 rcParams，所以字体设置必须放在 style.use 之后再做。
    # 这里先定义工具函数，稍后在创建 figure 前调用。
    def _setup_cn_font() -> Tuple[str, Optional[FontProperties]]:
        """
        尽可能确保中文可见：
        - 先按 Windows 字体文件路径强制指定（最稳）
        - 再按字体名称设置 rcParams（次稳）
        """
        fp: Optional[FontProperties] = None
        font_name = ""
        try:
            # 1) Windows 常见字体文件（最稳）
            win_dir = os.environ.get("WINDIR", r"C:\Windows")
            font_files = [
                os.path.join(win_dir, "Fonts", "msyh.ttc"),    # 微软雅黑
                os.path.join(win_dir, "Fonts", "msyhbd.ttc"),
                os.path.join(win_dir, "Fonts", "simhei.ttf"),  # 黑体
                os.path.join(win_dir, "Fonts", "simsun.ttc"),  # 宋体
            ]
            for p in font_files:
                if os.path.exists(p):
                    fp = FontProperties(fname=p)
                    try:
                        font_name = fp.get_name()
                    except Exception:
                        font_name = os.path.basename(p)
                    # rcParams 也同步设置，影响全局文本/legend
                    matplotlib.rcParams["font.family"] = "sans-serif"
                    matplotlib.rcParams["font.sans-serif"] = [font_name]
                    matplotlib.rcParams["axes.unicode_minus"] = False
                    return font_name, fp

            # 2) 按字体名称（次稳）
            candidates = [
                "Microsoft YaHei",  # 微软雅黑
                "SimHei",           # 黑体
                "SimSun",
                "PingFang SC",
                "Noto Sans CJK SC",
                "Source Han Sans SC",
                "Arial Unicode MS",
            ]
            available = {f.name for f in font_manager.fontManager.ttflist}
            for name in candidates:
                if name in available:
                    matplotlib.rcParams["font.family"] = "sans-serif"
                    matplotlib.rcParams["font.sans-serif"] = [name]
                    matplotlib.rcParams["axes.unicode_minus"] = False
                    return name, None

            # 兜底：至少保证负号正常
            matplotlib.rcParams["axes.unicode_minus"] = False
        except Exception:
            pass
        return "", None

    font_used: str = ""
    font_prop: Optional[FontProperties] = None
    
    # 方法0: 优先从 result 字典的 sys_analyser 中提取（这是 RQAlpha 的标准结构）
    if isinstance(result, dict) and "sys_analyser" in result:
        try:
            analyser = result["sys_analyser"]
            
            # 优先从 portfolio 中提取策略净值（这是 RQAlpha 的标准方式，更可靠）
            if "portfolio" in analyser:
                portfolio_df = analyser["portfolio"]
                if isinstance(portfolio_df, pd.DataFrame) and "unit_net_value" in portfolio_df.columns:
                    nav_series = portfolio_df["unit_net_value"]
                    # 检查是否有 date 列或索引是日期
                    if "date" in portfolio_df.columns:
                        portfolio_df_with_date = portfolio_df.set_index("date")
                        nav_series = portfolio_df_with_date["unit_net_value"]
                    elif isinstance(portfolio_df.index, pd.DatetimeIndex):
                        # 确保 Series 使用 DataFrame 的日期索引
                        nav_series = pd.Series(nav_series.values, index=portfolio_df.index)
                    
                    plot_data["strategy_nav"] = nav_series
                    logging.info(f"从 sys_analyser.portfolio 提取策略净值数据，共 {len(nav_series)} 个数据点，索引类型: {type(nav_series.index)}")
                    if len(nav_series) > 0:
                        sample_values = nav_series.head(5).tolist()
                        logging.info(f"策略净值（portfolio）前5个值: {sample_values}")
                        logging.info(f"策略净值范围: {nav_series.min():.4f} 至 {nav_series.max():.4f}")
            
            # 如果 portfolio 中没有数据，尝试从 plots 中提取
            if "strategy_nav" not in plot_data and "plots" in analyser and isinstance(analyser["plots"], pd.DataFrame):
                plots_df = analyser["plots"]
                logging.info(f"plots DataFrame 列: {plots_df.columns.tolist()}, 索引: {plots_df.index.name if plots_df.index.name else '无名称'}, 形状: {plots_df.shape}")
                
                if "strategy_nav" in plots_df.columns:
                    # 检查是否有 date 列或索引是日期
                    if "date" in plots_df.columns:
                        # 使用 date 列作为索引
                        plots_df_with_date = plots_df.set_index("date")
                        strategy_nav_series = plots_df_with_date["strategy_nav"]
                        plot_data["strategy_nav"] = strategy_nav_series
                        logging.info(f"从 sys_analyser.plots 提取策略净值数据（使用date列），共 {len(strategy_nav_series)} 个数据点，索引类型: {type(strategy_nav_series.index)}")
                    elif isinstance(plots_df.index, pd.DatetimeIndex):
                        # 索引已经是日期
                        strategy_nav_series = plots_df["strategy_nav"]
                        plot_data["strategy_nav"] = strategy_nav_series
                        logging.info(f"从 sys_analyser.plots 提取策略净值数据（使用DatetimeIndex），共 {len(strategy_nav_series)} 个数据点")
                    else:
                        # 使用现有索引
                        strategy_nav_series = plots_df["strategy_nav"]
                        plot_data["strategy_nav"] = strategy_nav_series
                        logging.info(f"从 sys_analyser.plots 提取策略净值数据（使用现有索引），共 {len(strategy_nav_series)} 个数据点，索引类型: {type(strategy_nav_series.index)}")
                    
                    # 验证数据
                    if len(plot_data["strategy_nav"]) > 0:
                        sample_values = plot_data["strategy_nav"].head(5).tolist()
                        logging.info(f"策略净值（plots）前5个值: {sample_values}")
                else:
                    logging.warning(f"plots DataFrame 中没有 'strategy_nav' 列，可用列: {plots_df.columns.tolist()}")
            
            # 提取基准净值数据
            # 方法1: 优先从 benchmark_portfolio 中提取（RQAlpha 自动生成的基准数据）
            if "benchmark_portfolio" in analyser:
                benchmark_df = analyser["benchmark_portfolio"]
                if benchmark_df is not None and isinstance(benchmark_df, pd.DataFrame) and not benchmark_df.empty:
                    logging.info(f"benchmark_portfolio DataFrame 列: {benchmark_df.columns.tolist()}, 索引: {benchmark_df.index.name if benchmark_df.index.name else '无名称'}, 形状: {benchmark_df.shape}")
                    
                    if "unit_net_value" in benchmark_df.columns:
                        # 检查是否有 date 列或索引是日期
                        if "date" in benchmark_df.columns:
                            # 使用 date 列作为索引
                            benchmark_df_with_date = benchmark_df.set_index("date")
                            benchmark_nav_series = benchmark_df_with_date["unit_net_value"]
                            plot_data["benchmark_nav"] = benchmark_nav_series
                            logging.info(f"从 sys_analyser.benchmark_portfolio 提取基准净值数据（使用date列），共 {len(benchmark_nav_series)} 个数据点，索引类型: {type(benchmark_nav_series.index)}")
                        elif isinstance(benchmark_df.index, pd.DatetimeIndex):
                            # 索引已经是日期
                            benchmark_nav_series = benchmark_df["unit_net_value"]
                            plot_data["benchmark_nav"] = benchmark_nav_series
                            logging.info(f"从 sys_analyser.benchmark_portfolio 提取基准净值数据（使用DatetimeIndex），共 {len(benchmark_nav_series)} 个数据点")
                        else:
                            # 使用现有索引
                            benchmark_nav_series = benchmark_df["unit_net_value"]
                            plot_data["benchmark_nav"] = benchmark_nav_series
                            logging.info(f"从 sys_analyser.benchmark_portfolio 提取基准净值数据（使用现有索引），共 {len(benchmark_nav_series)} 个数据点，索引类型: {type(benchmark_nav_series.index)}")
                        
                        # 验证数据
                        if len(plot_data["benchmark_nav"]) > 0:
                            sample_values = plot_data["benchmark_nav"].head(5).tolist()
                            logging.info(f"基准净值前5个值: {sample_values}")
                    else:
                        logging.warning(f"benchmark_portfolio DataFrame 中没有 'unit_net_value' 列，可用列: {benchmark_df.columns.tolist()}")
                else:
                    logging.debug(f"benchmark_portfolio 为空或不是 DataFrame: {type(benchmark_df)}")
            
            # 方法2: 如果 benchmark_portfolio 中没有数据，尝试从 plots 中提取（策略中 plot() 函数保存的数据）
            if "benchmark_nav" not in plot_data and "plots" in analyser and isinstance(analyser["plots"], pd.DataFrame):
                plots_df = analyser["plots"]
                logging.info(f"尝试从 plots 中提取基准净值，plots DataFrame 列: {plots_df.columns.tolist()}")
                
                if "benchmark_nav" in plots_df.columns:
                    # 检查是否有 date 列或索引是日期
                    if "date" in plots_df.columns:
                        # 使用 date 列作为索引
                        plots_df_with_date = plots_df.set_index("date")
                        benchmark_nav_series = plots_df_with_date["benchmark_nav"]
                        plot_data["benchmark_nav"] = benchmark_nav_series
                        logging.info(f"从 sys_analyser.plots 提取基准净值数据（使用date列），共 {len(benchmark_nav_series)} 个数据点，索引类型: {type(benchmark_nav_series.index)}")
                    elif isinstance(plots_df.index, pd.DatetimeIndex):
                        # 索引已经是日期
                        benchmark_nav_series = plots_df["benchmark_nav"]
                        plot_data["benchmark_nav"] = benchmark_nav_series
                        logging.info(f"从 sys_analyser.plots 提取基准净值数据（使用DatetimeIndex），共 {len(benchmark_nav_series)} 个数据点")
                    else:
                        # 使用现有索引
                        benchmark_nav_series = plots_df["benchmark_nav"]
                        plot_data["benchmark_nav"] = benchmark_nav_series
                        logging.info(f"从 sys_analyser.plots 提取基准净值数据（使用现有索引），共 {len(benchmark_nav_series)} 个数据点，索引类型: {type(benchmark_nav_series.index)}")
                    
                    # 验证数据
                    if len(plot_data["benchmark_nav"]) > 0:
                        sample_values = plot_data["benchmark_nav"].head(5).tolist()
                        logging.info(f"基准净值（plots）前5个值: {sample_values}")
                else:
                    logging.debug(f"plots DataFrame 中没有 'benchmark_nav' 列，可用列: {plots_df.columns.tolist()}")
            
            # 如果仍然没有基准数据，记录警告
            if "benchmark_nav" not in plot_data:
                logging.warning("⚠ 未找到基准净值数据")
                logging.warning("  可能原因:")
                logging.warning("    1. 基准代码配置可能无效")
                logging.warning("    2. RQAlpha 未正确生成基准数据")
                logging.warning("    3. 策略中未正确调用 plot('benchmark_nav', ...)")
                logging.warning(f"  请检查配置的基准代码是否正确，以及回测日志中的基准配置信息")
                # 输出 analyser 中可用的键，帮助调试
                if isinstance(analyser, dict):
                    logging.warning(f"  analyser 中可用的键: {list(analyser.keys())}")
            else:
                # 成功提取基准数据，输出详细信息
                benchmark_nav = plot_data["benchmark_nav"]
                logging.info(f"✓ 成功提取基准净值数据")
                logging.info(f"  数据类型: {type(benchmark_nav)}")
                if isinstance(benchmark_nav, pd.Series):
                    logging.info(f"  数据长度: {len(benchmark_nav)}")
                    logging.info(f"  索引类型: {type(benchmark_nav.index)}")
                    logging.info(f"  前5个值: {benchmark_nav.head().tolist()}")
                    logging.info(f"  后5个值: {benchmark_nav.tail().tolist()}")
                    logging.info(f"  净值范围: {benchmark_nav.min():.4f} 至 {benchmark_nav.max():.4f}")
            
                    
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
    
    # 创建图表（更接近 RQAlpha 教程风格：多条曲线 + 多个子图）
    try:
        plt.style.use("seaborn-v0_8")
    except Exception:
        pass
    # === 字体：解决中文乱码（放在 style.use 之后，避免被覆盖） ===
    try:
        font_used, font_prop = _setup_cn_font()
    except Exception:
        font_used, font_prop = "", None
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.2, 1.4]},
    )
    
    # 辅助函数：从数据中提取日期和净值
    def extract_dates_and_values(data, data_name="数据"):
        """从各种格式的数据中提取日期和净值"""
        if isinstance(data, pd.Series):
            dates = data.index
            values = data.values
            # 转换日期格式
            if isinstance(dates, pd.DatetimeIndex):
                dates = [d.date() for d in dates]
            elif len(dates) > 0:
                dates = [pd.Timestamp(d).date() if isinstance(d, (pd.Timestamp, pd.DatetimeIndex)) else (d.date() if hasattr(d, 'date') else d) for d in dates]
            # 确保 dates 和 values 长度一致
            if len(dates) != len(values):
                logging.warning(f"{data_name}: 日期和净值数量不一致！日期: {len(dates)}, 净值: {len(values)}")
                min_len = min(len(dates), len(values))
                dates = dates[:min_len]
                values = values[:min_len]
            logging.info(f"{data_name}: Series格式，日期数量: {len(dates)}, 净值数量: {len(values)}, 前3个净值: {values[:3] if len(values) >= 3 else values}")
            return dates, values
        elif isinstance(data, dict):
            dates = list(data.keys())
            values = list(data.values())
            # 转换日期格式
            if dates and isinstance(dates[0], (pd.Timestamp, pd.DatetimeIndex)):
                dates = [pd.Timestamp(d).date() if isinstance(d, (pd.Timestamp, pd.DatetimeIndex)) else d for d in dates]
            elif dates and hasattr(dates[0], 'date'):
                dates = [d.date() if hasattr(d, 'date') else d for d in dates]
            logging.debug(f"{data_name}: 字典格式，日期数量: {len(dates)}, 净值数量: {len(values)}")
            return dates, values
        elif isinstance(data, list):
            logging.debug(f"{data_name}: 列表格式，数量: {len(data)}")
            return None, data
        else:
            logging.warning(f"{data_name}: 未知格式 {type(data)}")
            return None, None
    
    # 提取策略和基准数据
    strategy_dates = None
    strategy_values = None
    benchmark_dates = None
    benchmark_values = None
    
    if "strategy_nav" in plot_data:
        strategy_dates, strategy_values = extract_dates_and_values(plot_data["strategy_nav"], "策略净值")
        if strategy_dates is not None and strategy_values is not None:
            logging.info(f"策略净值: 日期范围 {strategy_dates[0] if strategy_dates else 'N/A'} 至 {strategy_dates[-1] if strategy_dates else 'N/A'}, 净值范围 {min(strategy_values):.4f} 至 {max(strategy_values):.4f}")
    
    if "benchmark_nav" in plot_data:
        benchmark_dates, benchmark_values = extract_dates_and_values(plot_data["benchmark_nav"], "基准净值")
        if benchmark_dates is not None and benchmark_values is not None:
            logging.info(f"基准净值: 日期范围 {benchmark_dates[0] if benchmark_dates else 'N/A'} 至 {benchmark_dates[-1] if benchmark_dates else 'N/A'}, 净值范围 {min(benchmark_values):.4f} 至 {max(benchmark_values):.4f}")
    
    # 过滤掉 None 值
    if strategy_dates is not None and strategy_values is not None:
        # 过滤掉 None 值
        valid_pairs = [(d, v) for d, v in zip(strategy_dates, strategy_values) if v is not None and not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))]
        if valid_pairs:
            strategy_dates, strategy_values = zip(*valid_pairs)
            strategy_dates = list(strategy_dates)
            strategy_values = list(strategy_values)
        else:
            strategy_dates = None
            strategy_values = None
            logging.warning("策略净值数据全部为 None 或无效值")
    
    if benchmark_dates is not None and benchmark_values is not None:
        # 过滤掉 None 值
        valid_pairs = [(d, v) for d, v in zip(benchmark_dates, benchmark_values) if v is not None and not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))]
        if valid_pairs:
            benchmark_dates, benchmark_values = zip(*valid_pairs)
            benchmark_dates = list(benchmark_dates)
            benchmark_values = list(benchmark_values)
        else:
            benchmark_dates = None
            benchmark_values = None
            logging.warning("基准净值数据全部为 None 或无效值")
    
    ax0 = axes[0]
    # 绘制策略净值曲线
    if strategy_dates is not None and strategy_values is not None and len(strategy_dates) > 0 and len(strategy_values) > 0:
        logging.info(f"绘制策略净值曲线: {len(strategy_dates)} 个数据点")
        ax0.plot(strategy_dates, strategy_values, label="策略净值", linewidth=2.2, color="#1f77b4")
    elif strategy_values is not None and len(strategy_values) > 0:
        logging.info(f"绘制策略净值曲线（无日期）: {len(strategy_values)} 个数据点")
        ax0.plot(range(len(strategy_values)), strategy_values, label="策略净值", linewidth=2.2, color="#1f77b4")
    else:
        logging.warning("策略净值数据为空，无法绘制")
    
    # 绘制基准净值曲线
    # 确定基准标签（包含基准代码）
    benchmark_label = "基准净值"
    if benchmark_code:
        # 基准代码到名称的映射
        benchmark_names = {
            "000300.XSHG": "沪深300",
            "000905.XSHG": "中证500",
            "399005.XSHE": "中小板指",
            "399101.XSHE": "中小综指",
            "399006.XSHE": "创业板指",
        }
        benchmark_name = benchmark_names.get(benchmark_code, benchmark_code)
        benchmark_label = f"基准净值 ({benchmark_name})"
        logging.info(f"基准指数代码: {benchmark_code} ({benchmark_name})")
    
    if benchmark_dates is not None and benchmark_values is not None and len(benchmark_dates) > 0 and len(benchmark_values) > 0:
        logging.info(f"绘制基准净值曲线: {len(benchmark_dates)} 个数据点")
        ax0.plot(benchmark_dates, benchmark_values, label=benchmark_label, linewidth=2.0, color="#ff7f0e", linestyle="--")
    elif benchmark_values is not None and len(benchmark_values) > 0:
        logging.info(f"绘制基准净值曲线（无日期）: {len(benchmark_values)} 个数据点")
        ax0.plot(range(len(benchmark_values)), benchmark_values, label=benchmark_label, linewidth=2.0, color="#ff7f0e", linestyle="--")
    else:
        logging.warning("⚠ 基准净值数据为空，无法绘制")
        logging.warning(f"  基准指数代码: {benchmark_code if benchmark_code else '未指定'}")
        logging.warning(f"  benchmark_dates: {benchmark_dates}")
        logging.warning(f"  benchmark_values: {benchmark_values}")
        logging.warning(f"  plot_data 中的键: {list(plot_data.keys())}")
        if "benchmark_nav" in plot_data:
            logging.warning(f"  benchmark_nav 数据类型: {type(plot_data['benchmark_nav'])}")
            if isinstance(plot_data["benchmark_nav"], pd.Series):
                logging.warning(f"  benchmark_nav Series 长度: {len(plot_data['benchmark_nav'])}")
                logging.warning(f"  benchmark_nav 前5个值: {plot_data['benchmark_nav'].head().tolist()}")
    
    # === 子图1：净值 ===
    title = "MSA 策略净值 vs 基准"
    if benchmark_code:
        benchmark_names = {
            "000300.XSHG": "沪深300",
            "000905.XSHG": "中证500",
            "399005.XSHE": "中小板指",
            "399101.XSHE": "中小综指",
            "399006.XSHE": "创业板指",
        }
        benchmark_name = benchmark_names.get(benchmark_code, benchmark_code)
        title = f"策略净值曲线对比 (基准: {benchmark_name})"
        logging.info(f"图表标题: {title}")
    
    ax0.set_title(title, fontsize=15, fontweight="bold")
    ax0.set_ylabel("净值", fontsize=11)
    ax0.legend(loc="best", fontsize=10)
    ax0.grid(True, alpha=0.25)

    # === 指标信息框：夏普/年化/回撤等（优先读取 RQAlpha summary，缺失则自行计算） ===
    summary = {}
    try:
        if isinstance(result, dict) and "sys_analyser" in result:
            analyser = result.get("sys_analyser") or {}
            if isinstance(analyser, dict) and isinstance(analyser.get("summary"), dict):
                summary = analyser.get("summary") or {}
    except Exception:
        summary = {}

    # 自行计算（基于净值）
    def _metrics_from_nav(vals: Optional[list]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        try:
            if not vals or len(vals) < 3:
                return out
            s = pd.Series([float(x) for x in vals]).replace([np.inf, -np.inf], np.nan).dropna()
            if len(s) < 3:
                return out
            total_ret = float(s.iloc[-1] / s.iloc[0] - 1.0) if float(s.iloc[0]) != 0 else 0.0
            rets = s.pct_change().dropna()
            vol = float(rets.std(ddof=1) * np.sqrt(252)) if len(rets) > 1 else 0.0
            sharpe = float(rets.mean() / (rets.std(ddof=1) + 1e-12) * np.sqrt(252)) if len(rets) > 1 else 0.0
            # 年化收益：用复利按日频估算
            ann = float((1.0 + total_ret) ** (252.0 / max(1.0, float(len(rets)))) - 1.0)
            # 最大回撤
            peak = s.cummax()
            dd = s / peak - 1.0
            mdd = float(dd.min())
            out.update(
                {
                    "total_return": total_ret,
                    "annual_return": ann,
                    "volatility": vol,
                    "sharpe": sharpe,
                    "max_drawdown": mdd,
                }
            )
        except Exception:
            pass
        return out

    calc = _metrics_from_nav(strategy_values if isinstance(strategy_values, list) else None)
    def _get_first(keys: list, default=None):
        for k in keys:
            if k in summary and summary[k] is not None:
                return summary[k]
        return default

    sharpe_v = _get_first(["sharpe", "sharpe_ratio"], calc.get("sharpe", None))
    ann_v = _get_first(["annualized_returns", "annualized_return", "annual_return"], calc.get("annual_return", None))
    mdd_v = _get_first(["max_drawdown"], calc.get("max_drawdown", None))
    vol_v = _get_first(["volatility", "annual_volatility"], calc.get("volatility", None))
    tot_v = _get_first(["total_returns", "total_return"], calc.get("total_return", None))

    # 取回测周期（如果有日期）
    date_start = None
    date_end = None
    try:
        if strategy_dates:
            date_start = str(strategy_dates[0])
            date_end = str(strategy_dates[-1])
    except Exception:
        pass

    lines = []
    if benchmark_code:
        lines.append(f"基准: {benchmark_code}")
    if date_start and date_end:
        lines.append(f"区间: {date_start} ~ {date_end}")
    if tot_v is not None:
        try:
            lines.append(f"总收益: {float(tot_v):.2%}")
        except Exception:
            pass
    if ann_v is not None:
        try:
            lines.append(f"年化: {float(ann_v):.2%}")
        except Exception:
            pass
    if vol_v is not None:
        try:
            lines.append(f"波动: {float(vol_v):.2%}")
        except Exception:
            pass
    if sharpe_v is not None:
        try:
            lines.append(f"夏普: {float(sharpe_v):.2f}")
        except Exception:
            pass
    if mdd_v is not None:
        try:
            lines.append(f"最大回撤: {float(mdd_v):.2%}")
        except Exception:
            pass
    if font_used:
        lines.append(f"字体: {font_used}")

    if lines:
        ax0.text(
            0.01,
            0.99,
            "\n".join(lines),
            transform=ax0.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            fontproperties=font_prop,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#999999", alpha=0.80),
        )

    # === 子图2：回撤（由净值计算） ===
    ax1 = axes[1]
    def _drawdown(vals: list) -> Optional[list]:
        try:
            if not vals:
                return None
            peak = -1e18
            out = []
            for v in vals:
                vv = float(v)
                peak = max(peak, vv)
                out.append(vv / peak - 1.0 if peak > 0 else 0.0)
            return out
        except Exception:
            return None

    dd_s = _drawdown(strategy_values) if strategy_values is not None else None
    if dd_s is not None:
        if strategy_dates is not None and len(strategy_dates) == len(dd_s):
            ax1.plot(strategy_dates, dd_s, color="#d62728", linewidth=1.6, label="策略回撤")
        else:
            ax1.plot(range(len(dd_s)), dd_s, color="#d62728", linewidth=1.6, label="策略回撤")
    dd_b = _drawdown(benchmark_values) if benchmark_values is not None else None
    if dd_b is not None:
        if benchmark_dates is not None and len(benchmark_dates) == len(dd_b):
            ax1.plot(benchmark_dates, dd_b, color="#9467bd", linewidth=1.2, linestyle="--", label="基准回撤")
        else:
            ax1.plot(range(len(dd_b)), dd_b, color="#9467bd", linewidth=1.2, linestyle="--", label="基准回撤")
    ax1.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax1.set_ylabel("回撤", fontsize=11)
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.25)

    # === 子图3：运行指标（来自策略 plot()） ===
    ax2 = axes[2]
    plots_df = None
    if isinstance(result, dict) and "sys_analyser" in result:
        try:
            analyser = result["sys_analyser"]
            if isinstance(analyser, dict) and "plots" in analyser and isinstance(analyser["plots"], pd.DataFrame):
                plots_df = analyser["plots"].copy()
        except Exception:
            plots_df = None
    if plots_df is not None and not plots_df.empty:
        # 标准化 index 为日期
        if "date" in plots_df.columns:
            try:
                plots_df["date"] = pd.to_datetime(plots_df["date"])
                plots_df = plots_df.set_index("date")
            except Exception:
                pass
        if not isinstance(plots_df.index, pd.DatetimeIndex):
            try:
                plots_df.index = pd.to_datetime(plots_df.index)
            except Exception:
                pass
        plots_df = plots_df.sort_index()

        # 兼容 key：rqalpha plot 名称里带斜杠会原样成为列名
        candidates = {
            "msa/exposure": ("仓位(市值/总资产)", "#2ca02c"),
            "msa/cash_ratio": ("现金占比", "#17becf"),
            "msa/turnover": ("换手(0.5*Σ|Δw|)", "#8c564b"),
            "msa/holdings": ("持仓数", "#7f7f7f"),
            "msa/risk_sells": ("止损卖出数", "#bcbd22"),
        }
        left_plotted = False
        right_ax = ax2.twinx()
        right_plotted = False

        for col, (label, color) in candidates.items():
            if col not in plots_df.columns:
                continue
            s = pd.to_numeric(plots_df[col], errors="coerce").fillna(0.0)
            # exposure/cash 用左轴（0~1），其余用右轴
            if col in {"msa/exposure", "msa/cash_ratio"}:
                ax2.plot(s.index, s.values, label=label, color=color, linewidth=1.6)
                left_plotted = True
            else:
                right_ax.plot(s.index, s.values, label=label, color=color, linewidth=1.2, alpha=0.9)
                right_plotted = True

        ax2.set_ylabel("比例", fontsize=11)
        right_ax.set_ylabel("计数/换手", fontsize=11)
        ax2.grid(True, alpha=0.25)
        # 合并 legend
        handles = []
        labels = []
        if left_plotted:
            h, l = ax2.get_legend_handles_labels()
            handles += h
            labels += l
        if right_plotted:
            h, l = right_ax.get_legend_handles_labels()
            handles += h
            labels += l
        if handles:
            ax2.legend(handles, labels, loc="best", fontsize=9)
    else:
        ax2.text(0.01, 0.6, "未检测到策略 plot() 指标数据\n（可正常，仅影响第三子图）", transform=ax2.transAxes)
        ax2.grid(True, alpha=0.15)
        ax2.set_ylabel("指标", fontsize=11)

    axes[-1].set_xlabel("交易日", fontsize=11)
    
    # 优化日期显示
    try:
        for ax in axes:
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha("right")
    except Exception:
        pass
    
    # 保存图片（必须在 show() 之前调用）
    plot_path = os.path.join(output_dir, "rqalpha_strategy_plot.png")
    fig.tight_layout()
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
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RQAlpha 回测执行")
    parser.add_argument("--rqalpha-config", type=str, required=True, help="RQAlpha 配置文件路径")
    parser.add_argument("--prediction", type=str, required=True, help="预测信号文件路径")
    parser.add_argument("--industry", type=str, default=None, help="行业映射文件路径")
    parser.add_argument("--strategy", type=str, default=None, help="策略脚本路径（可选）")
    parser.add_argument("--full-invested", action="store_true", help="满仓回测：忽略仓位/单股/行业限制，权重归一化为100%")
    
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
        full_invested=args.full_invested,
    )

