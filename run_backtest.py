"""
回测脚本：支持两种模式
1. 简化回测（默认）：快速验证信号有效性
2. RQAlpha 回测（--use-rqalpha）：真实 T+1 交易，包含手续费、滑点等成本
"""

import argparse
import logging
import os
import subprocess
import sys
from typing import Optional, Union

import pandas as pd

from feature.qlib_feature_pipeline import QlibFeaturePipeline
from portfolio.portfolio_builder import PortfolioBuilder
from utils import load_yaml_config


def parse_args():
    parser = argparse.ArgumentParser(description="基于预测结果的回测")
    parser.add_argument(
        "--config",
        type=str,
        default="config/pipeline.yaml",
        help="pipeline 配置文件路径",
    )
    parser.add_argument(
        "--prediction",
        type=str,
        default=None,
        help="预测结果 CSV 路径（需包含 final 列），默认自动查找最新预测文件",
    )
    parser.add_argument(
        "--industry",
        type=str,
        default=None,
        help="行业映射 CSV，包含 instrument, industry 列",
    )
    parser.add_argument(
        "--use-rqalpha",
        action="store_true",
        help="使用 RQAlpha 框架进行真实 T+1 回测（包含手续费、滑点）",
    )
    parser.add_argument(
        "--rqalpha-config",
        type=str,
        default="config/rqalpha_config.yaml",
        help="RQAlpha 配置文件路径（仅在 --use-rqalpha 时使用）",
    )
    parser.add_argument(
        "--full-invested",
        action="store_true",
        help="满仓回测：忽略仓位/单股/行业限制，权重归一化为100%（简化回测与RQAlpha都生效）",
    )
    return parser.parse_args()


def load_predictions(path: str) -> pd.Series:
    # 关键：instrument 必须按字符串读取，否则如 "2001" 会被读成 int 2001，
    # 与标签索引里的字符串 instrument 无法精确对齐，导致“没有重叠”的误判。
    df = pd.read_csv(path, dtype={"instrument": str})
    if "datetime" not in df.columns or "instrument" not in df.columns:
        raise ValueError(f"预测文件缺少必要列 datetime/instrument: {path}")
    if "final" not in df.columns:
        raise ValueError(f"预测文件缺少必要列 final: {path}")
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.normalize()
    df["instrument"] = df["instrument"].astype(str).str.strip()
    # 兼容形如 "000001.SZ"/"600000.SH" 的情况：对齐到纯代码
    df["instrument"] = df["instrument"].str.split(".", n=1).str[0]
    df.set_index(["datetime", "instrument"], inplace=True)
    return df["final"].sort_index()


def load_industry(path: Optional[str]) -> Optional[pd.Series]:
    if path is None:
        return None
    df = pd.read_csv(path, dtype={"instrument": str})
    if "instrument" not in df.columns or "industry" not in df.columns:
        raise ValueError("行业文件需包含 instrument 与 industry 列")
    df["instrument"] = df["instrument"].astype(str).str.strip()
    df["instrument"] = df["instrument"].str.split(".", n=1).str[0]
    return df.set_index("instrument")["industry"]


def _normalize_dt_inst_index(data: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
    """统一 (datetime, instrument) 索引的类型/格式，避免 int vs str、带后缀等导致无法对齐。"""
    if not isinstance(data.index, pd.MultiIndex) or data.index.nlevels < 2:
        return data
    if "datetime" not in data.index.names or "instrument" not in data.index.names:
        return data

    if isinstance(data, pd.Series):
        name = data.name or "value"
        df = data.reset_index(name=name)
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.normalize()
        df["instrument"] = df["instrument"].astype(str).str.strip().str.split(".", n=1).str[0]
        out = df.set_index(["datetime", "instrument"])[name].sort_index()
        out.name = data.name
        return out

    # DataFrame
    df = data.reset_index()
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.normalize()
    df["instrument"] = df["instrument"].astype(str).str.strip().str.split(".", n=1).str[0]
    out = df.set_index(["datetime", "instrument"]).sort_index()
    return out


def _find_latest_prediction(prediction_dir: str, pool_name: str = None) -> str:
    """自动查找最新的预测文件。
    
    参数:
        prediction_dir: 预测文件目录
        pool_name: 股票池名称，如果指定则只查找包含该名称的文件
    """
    import glob
    if pool_name:
        # 查找包含股票池名称的预测文件
        pattern = os.path.join(prediction_dir, f"*{pool_name}*pred*.csv")
        files = glob.glob(pattern)
        if not files:
            # 也尝试查找 pred_* 格式的文件
            pattern = os.path.join(prediction_dir, f"pred_*{pool_name}*.csv")
            files = glob.glob(pattern)
    else:
        # 查找所有预测文件
        pattern = os.path.join(prediction_dir, "*pred*.csv")
        files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"在 {prediction_dir} 中未找到预测文件（股票池: {pool_name}），请先运行 run_predict.py 或显式指定 --prediction")
    # 按修改时间排序，返回最新的
    latest = max(files, key=os.path.getmtime)
    logging.info("自动选择最新预测文件: %s", latest)
    return latest


def run_rqalpha_backtest(
    rqalpha_config_path: str,
    prediction_path: str,
    industry_path: Optional[str],
    pipeline_cfg: dict,
    pool_name: Optional[str] = None,
    *,
    full_invested: bool = False,
):
    """调用 RQAlpha 回测脚本。
    
    参数:
        rqalpha_config_path: RQAlpha 配置文件路径
        prediction_path: 预测文件路径
        industry_path: 行业映射文件路径
        pipeline_cfg: Pipeline 配置
        pool_name: 股票池名称（用于修改输出目录）
    """
    import tempfile
    import yaml
    
    strategy_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "backtest",
        "rqalpha_backtest.py"
    )
    
    if not os.path.exists(strategy_path):
        raise FileNotFoundError(f"RQAlpha 回测脚本不存在: {strategy_path}")
    
    # 如果指定了股票池名称，修改输出目录
    rqalpha_config_to_use = rqalpha_config_path
    if pool_name:
        rqalpha_cfg = load_yaml_config(rqalpha_config_path)
        original_output_dir = rqalpha_cfg.get("output", {}).get("output_dir", "data/backtest/rqalpha")
        # 在输出目录中添加股票池名称
        new_output_dir = os.path.join(original_output_dir, pool_name)
        rqalpha_cfg.setdefault("output", {})["output_dir"] = new_output_dir

        # 根据股票池自动选择基准指数（用于收益/绘图更准确）
        # 优先读取 config/rqalpha_config.yaml 中 base.benchmark300 / base.benchmark101
        base_cfg = rqalpha_cfg.setdefault("base", {})
        cfg_benchmark300 = base_cfg.get("benchmark300")
        cfg_benchmark101 = base_cfg.get("benchmark101")
        fallback_map = {
            "csi300": "000300.XSHG",
            "csi101": "399005.XSHE",
        }
        if pool_name == "csi300":
            benchmark = cfg_benchmark300 or fallback_map["csi300"]
            base_cfg["benchmark"] = benchmark
            logging.info("股票池 %s 基准指数设置为: %s", pool_name, benchmark)
        elif pool_name == "csi101":
            benchmark = cfg_benchmark101 or fallback_map["csi101"]
            base_cfg["benchmark"] = benchmark
            logging.info("股票池 %s 基准指数设置为: %s", pool_name, benchmark)
        
        # 创建临时配置文件
        temp_config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8')
        yaml.dump(rqalpha_cfg, temp_config_file, allow_unicode=True, default_flow_style=False)
        temp_config_file.close()
        rqalpha_config_to_use = temp_config_file.name
    
    try:
        # 构建命令
        cmd = [
            sys.executable,
            strategy_path,
            "--rqalpha-config", rqalpha_config_to_use,
            "--prediction", prediction_path,
        ]
        
        if full_invested:
            cmd.append("--full-invested")
        
        if industry_path:
            cmd.extend(["--industry", industry_path])
        
        logging.info(f"执行 RQAlpha 回测命令: {' '.join(cmd)}")
        
        # 执行回测
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        
        if result.returncode != 0:
            raise RuntimeError(f"RQAlpha 回测执行失败，返回码: {result.returncode}")
        
        logging.info("RQAlpha 回测完成")
    finally:
        # 清理临时配置文件
        if pool_name and rqalpha_config_to_use != rqalpha_config_path:
            os.unlink(rqalpha_config_to_use)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_yaml_config(args.config)
    data_cfg = load_yaml_config(cfg["data_config"])
    
    # 解析股票池列表
    instruments_config = data_cfg["data"]["instruments"]
    instrument_pools = QlibFeaturePipeline._parse_instrument_pools(instruments_config)
    
    logger = logging.getLogger(__name__)
    
    # 如果指定了预测文件，只回测该文件
    if args.prediction:
        logger.info("使用指定的预测文件: %s", args.prediction)
        # 尝试从预测文件名中提取股票池名称
        import re
        pool_name_match = re.search(r'([a-zA-Z0-9]+)_(?:pred|rqalpha_pred)', os.path.basename(args.prediction))
        pool_name = pool_name_match.group(1) if pool_name_match else None
        
        args.use_rqalpha = True
        if args.use_rqalpha:
            logger.info("使用 RQAlpha 框架进行真实 T+1 回测" + (f"（股票池: {pool_name}）" if pool_name else ""))
            run_rqalpha_backtest(
                args.rqalpha_config,
                args.prediction,
                args.industry,
                cfg,
                pool_name=pool_name,
                full_invested=args.full_invested,
            )
        return
    
    # 如果未指定预测文件，为每个股票池分别回测
    logger.info("检测到 %d 个股票池: %s", len(instrument_pools), instrument_pools)
    
    # 默认使用 RQAlpha 回测（如果没有通过参数指定）
    if not hasattr(args, 'use_rqalpha') or args.use_rqalpha is None:
        args.use_rqalpha = True
    
    for pool_name in instrument_pools:
        logger.info("=" * 80)
        logger.info("开始回测股票池: %s", pool_name)
        logger.info("=" * 80)
        
        # 自动查找该股票池的最新预测文件
        prediction_dir = cfg["paths"]["prediction_dir"]
        try:
            prediction_path = _find_latest_prediction(prediction_dir, pool_name)
        except FileNotFoundError as e:
            logger.warning("未找到股票池 %s 的预测文件，跳过: %s", pool_name, e)
            continue
        
        # 如果使用 RQAlpha 回测，直接调用 RQAlpha 脚本
        if args.use_rqalpha:
            logger.info("使用 RQAlpha 框架进行真实 T+1 回测（股票池: %s）", pool_name)
            run_rqalpha_backtest(
                args.rqalpha_config,
                prediction_path,
                args.industry,
                cfg,
                pool_name=pool_name,
                full_invested=args.full_invested,
            )
            logger.info("股票池 %s 回测完成", pool_name)
        else:
            # 简化回测模式（无交易成本，使用标签收益）
            logger.info("使用简化回测模式（无交易成本，使用标签收益）")
            
            preds = load_predictions(prediction_path)
            preds = _normalize_dt_inst_index(preds)
            instruments = preds.index.get_level_values("instrument").unique()
            logger.info("预测数据：共 %d 个日期，%d 只股票", 
                       len(preds.index.get_level_values("datetime").unique()),
                       len(instruments))

            # 为当前股票池创建临时配置文件
            import tempfile
            import yaml
            import copy
            
            temp_data_config = copy.deepcopy(data_cfg)
            temp_data_config["data"]["instruments"] = pool_name
            
            temp_data_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8')
            yaml.dump(temp_data_config, temp_data_file, allow_unicode=True, default_flow_style=False)
            temp_data_file.close()
            
            try:
                # 使用当前股票池的配置构建特征管道
                pipeline = QlibFeaturePipeline(temp_data_file.name)
                pipeline.build()
                _, labels = pipeline.get_all()
                labels = _normalize_dt_inst_index(labels)
                
                logger.info("标签数据：共 %d 条记录，日期范围: %s 到 %s",
                           len(labels),
                           labels.index.get_level_values("datetime").min() if len(labels) > 0 else "N/A",
                           labels.index.get_level_values("datetime").max() if len(labels) > 0 else "N/A")
                
                # 对齐预测和标签的索引
                labels = labels.loc[labels.index.isin(preds.index)]
                logger.info("对齐后的标签数据：共 %d 条记录", len(labels))
                
                if len(labels) == 0:
                    logger.warning("股票池 %s 的预测数据和标签数据没有重叠，无法进行回测", pool_name)
                    logger.warning("预测数据日期范围: %s 到 %s",
                                 preds.index.get_level_values("datetime").min() if len(preds) > 0 else "N/A",
                                 preds.index.get_level_values("datetime").max() if len(preds) > 0 else "N/A")
                    logger.warning("预测数据股票: %s", list(instruments)[:10])
                    continue
            finally:
                if os.path.exists(temp_data_file.name):
                    os.unlink(temp_data_file.name)
            
            industry_map = load_industry(args.industry)
            portfolio_cfg = cfg.get("portfolio", {})
            builder = PortfolioBuilder(
                max_position=portfolio_cfg.get("max_position", 0.3),
                max_stock_weight=portfolio_cfg.get("max_stock_weight", 0.05),
                max_industry_weight=portfolio_cfg.get("max_industry_weight", 0.2),
            )

            results = []
            detail_frames = []
            pred_dates = sorted(preds.index.get_level_values("datetime").unique())
            logger.info("开始回测，共 %d 个交易日", len(pred_dates))
            
            for dt in pred_dates:
                score = preds.xs(dt)
                try:
                    label_slice = labels.xs(dt)
                except KeyError:
                    logger.debug("日期 %s 在标签数据中不存在，跳过", dt)
                    continue
                
                if len(label_slice) == 0:
                    logger.debug("日期 %s 的标签数据为空，跳过", dt)
                    continue
                
                industry_slice = None
                if industry_map is not None:
                    industry_slice = industry_map.reindex(score.index)
                weights = builder.build(
                    score,
                    industry_slice,
                    top_k=portfolio_cfg.get("top_k", 50),
                    full_invested=args.full_invested,
                )
                
                if len(weights) == 0:
                    logger.debug("日期 %s 无法构建投资组合（权重为空），跳过", dt)
                    continue
                
                realized = label_slice.reindex(weights.index).dropna()
                if realized.empty:
                    logger.debug("日期 %s 的已实现收益为空（选中的股票没有标签数据），跳过", dt)
                    continue
                ret = (weights.loc[realized.index] * realized).sum()
                results.append({"date": dt, "return": ret})
                detail = pd.DataFrame(
                    {
                        "date": dt,
                        "instrument": realized.index,
                        "signal": score.reindex(realized.index).values,
                        "weight": weights.reindex(realized.index).values,
                        "label": realized.values,
                    }
                )
                detail["contribution"] = detail["weight"] * detail["label"]
                detail_frames.append(detail)

            if not results:
                logger.warning("股票池 %s 未生成任何回测记录", pool_name)
                continue

            df = pd.DataFrame(results)
            df.sort_values("date", inplace=True)
            df["cum_return"] = (1 + df["return"]).cumprod() - 1
            stats = {
                "total_return": df["cum_return"].iloc[-1],
                "avg_return": df["return"].mean(),
                "volatility": df["return"].std(),
                "sharpe": df["return"].mean() / (df["return"].std() + 1e-8) * (252 ** 0.5),
            }

            # 为每个股票池创建单独的回测结果目录
            backtest_dir = os.path.join(cfg["paths"]["backtest_dir"], pool_name)
            os.makedirs(backtest_dir, exist_ok=True)
            out_path = os.path.join(backtest_dir, "backtest_result.csv")
            df.to_csv(out_path, index=False)
            logger.info("股票池 %s 回测完成，结果写入 %s", pool_name, out_path)
            if detail_frames:
                detail_df = pd.concat(detail_frames, ignore_index=True)
                detail_path = os.path.join(backtest_dir, "backtest_detail.csv")
                detail_df.to_csv(detail_path, index=False)
                logger.info("股票池 %s 投资明细写入 %s", pool_name, detail_path)
            logger.info("股票池 %s 统计指标: %s", pool_name, stats)


if __name__ == "__main__":
    main()

