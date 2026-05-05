"""
基于 chanSignal CSV 文件的模型预测入口脚本。

功能：
1. 从 data/chanSignal 文件夹读取 CSV 文件（文件名格式：YYYYMMDD_chan.csv）
2. 从文件名提取日期（如 20250101_chan.csv -> 20250101）
3. 读取 CSV 文件内容作为股票代码列表
4. 在 data/models/chanModels/YYYYMMDD/ 目录下找到对应日期的模型
5. 使用该模型进行预测
6. 输出预测结果到 data/predictions/pred_chen.csv

用法示例：
python run_predict_chan.py
python run_predict_chan.py --chan-signal-file data/chanSignal/20250106_chan.csv
python run_predict_chan.py --start 2025-01-10 --end 2025-01-20
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import glob
from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd
import yaml
import tempfile
import copy
import json

from feature.qlib_feature_pipeline import QlibFeaturePipeline
from predictor.predictor import PredictorEngine
from utils import load_yaml_config


def extract_date_from_filename(filename: str) -> Optional[str]:
    """
    从文件名中提取日期。
    
    支持格式：
    - 20250101_chan.csv -> 20250101
    - 2025-01-01_chan.csv -> 20250101
    - chan_20250101.csv -> 20250101
    """
    # 提取 8 位数字日期（YYYYMMDD）
    match = re.search(r'(\d{8})', filename)
    if match:
        return match.group(1)
    
    # 尝试提取 YYYY-MM-DD 格式并转换
    match = re.search(r'(\d{4})-(\d{2})-(\d{2})', filename)
    if match:
        year, month, day = match.groups()
        return f"{year}{month}{day}"
    
    return None


def find_chan_signal_files() -> List[str]:
    """查找 data/chanSignal 文件夹中的所有 CSV 文件"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    chan_signal_dir = os.path.join(project_root, "data", "chanSignal")
    
    if not os.path.exists(chan_signal_dir):
        return []
    
    pattern = os.path.join(chan_signal_dir, "*_chan.csv")
    files = glob.glob(pattern)
    return sorted(files)


def read_stock_codes_from_csv(csv_path: str) -> List[str]:
    """
    从 CSV 文件读取股票代码列表。
    
    支持格式：
    - 无表头，每行一个代码
    - 有表头，包含 code/instrument/stock 等列名
    """
    df = pd.read_csv(csv_path, dtype=str, header=None, nrows=1)
    
    # 尝试读取表头
    df_with_header = pd.read_csv(csv_path, dtype=str, nrows=0)
    has_header = len(df_with_header.columns) > 0 and any(
        keyword in col.lower() for col in df_with_header.columns 
        for keyword in ["instrument", "code", "stock"]
    )
    
    if has_header:
        # 有表头，查找包含 code/instrument/stock 的列
        df = pd.read_csv(csv_path, dtype=str)
        code_col = None
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ["instrument", "code", "stock"]):
                code_col = col
                break
        
        if code_col:
            codes = df[code_col].astype(str).str.strip().dropna().tolist()
        else:
            # 使用第一列
            codes = df.iloc[:, 0].astype(str).str.strip().dropna().tolist()
    else:
        # 无表头，每行一个代码
        codes = pd.read_csv(csv_path, dtype=str, header=None).iloc[:, 0].astype(str).str.strip().dropna().tolist()
    
    # 清理代码格式（移除 .SH, .SZ 后缀，确保是纯数字）
    cleaned_codes = []
    for code in codes:
        code = str(code).strip()
        if not code or code.startswith("#"):
            continue
        # 移除后缀
        if '.' in code:
            code = code.split('.')[0]
        # 确保是6位数字
        if code.isdigit() and len(code) == 6:
            cleaned_codes.append(code)
    
    return cleaned_codes


def _load_ic_histories(log_path: str) -> Dict[str, pd.Series]:
    """加载历史 IC 数据（用于动态权重计算）

    注意 (Phase 1 P1-1)：必须显式包含 `gru` 键，否则下游
    `RankICDynamicWeighter.blend(...)` 会以 ``weights.get("gru", 0.0)``
    把 GRU 预测乘 0，等同于 GRU 不参与最终融合。"""
    if not os.path.exists(log_path):
        today = pd.Timestamp.today()
        base = pd.Series([0.1], index=[today])
        return {
            "lgb": base,
            "gru": base,
            "mlp": base,
            "stack": base,
            "qlib_ensemble": base,
        }
    df = pd.read_csv(log_path, parse_dates=["valid_end"])
    histories = {
        "lgb": pd.Series(df["ic_lgb"].values, index=df["valid_end"]),
        "mlp": pd.Series(df["ic_mlp"].values, index=df["valid_end"]),
        "stack": pd.Series(df["ic_stack"].values, index=df["valid_end"]),
    }
    if "ic_gru" in df.columns:
        histories["gru"] = pd.Series(df["ic_gru"].values, index=df["valid_end"])
    else:
        histories["gru"] = histories["lgb"]
    if "ic_qlib_ensemble" in df.columns:
        histories["qlib_ensemble"] = pd.Series(df["ic_qlib_ensemble"].values, index=df["valid_end"])
    else:
        histories["qlib_ensemble"] = histories["lgb"]
    return histories


def find_model_tag(model_dir: str, date_str: str) -> Optional[str]:
    """
    在模型目录中查找指定日期的模型 tag。
    
    模型目录结构：data/models/chanModels/YYYYMMDD/YYYYMMDD_*.json
    """
    date_model_dir = os.path.join(model_dir, date_str)
    if not os.path.exists(date_model_dir):
        return None
    
    # 查找模型文件（以 date_str 开头的 _lgb.txt 或 _norm_meta.json）
    for name in os.listdir(date_model_dir):
        if name.startswith(date_str) and (name.endswith("_lgb.txt") or name.endswith("_norm_meta.json")):
            # 提取 tag（去掉后缀）
            if name.endswith("_lgb.txt"):
                return name.replace("_lgb.txt", "")
            elif name.endswith("_norm_meta.json"):
                return name.replace("_norm_meta.json", "")
    
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="基于 chanSignal 的模型预测")
    parser.add_argument(
        "--config",
        type=str,
        default="config/pipeline.yaml",
        help="Pipeline 配置文件路径",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="config/data.yaml",
        help="数据配置文件路径",
    )
    parser.add_argument(
        "--chan-signal-file",
        type=str,
        default=None,
        help="指定单个 chanSignal CSV 文件路径（不指定则处理所有文件）",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=os.environ.get("RUN_PRED_START", "2025-01-10"),
        help="预测起始日期，默认 2025-01-10",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=os.environ.get("RUN_PRED_END", "2025-01-20"),
        help="预测结束日期，默认 2025-01-20",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    logger = logging.getLogger(__name__)
    
    # 加载配置
    cfg = load_yaml_config(args.config)
    data_cfg = load_yaml_config(args.data_config)
    
    # 确定要处理的 CSV 文件列表
    if args.chan_signal_file:
        csv_files = [args.chan_signal_file]
    else:
        csv_files = find_chan_signal_files()
        if not csv_files:
            raise FileNotFoundError(
                f"未找到 chanSignal CSV 文件，请检查 data/chanSignal 目录，"
                f"或使用 --chan-signal-file 指定文件路径"
            )
    
    logger.info("找到 %d 个 CSV 文件需要处理", len(csv_files))
    
    # 预测结果保存路径
    predictions_dir = os.path.join(project_root, "data", "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    output_file = os.path.join(predictions_dir, "pred_chen.csv")
    
    # 存储所有预测结果
    all_predictions = []
    
    # 为每个 CSV 文件进行预测
    for csv_path in csv_files:
        csv_filename = os.path.basename(csv_path)
        logger.info("=" * 80)
        logger.info("处理文件: %s", csv_filename)
        logger.info("=" * 80)
        
        # 提取日期
        date_str = extract_date_from_filename(csv_filename)
        if not date_str:
            logger.warning("无法从文件名提取日期，跳过: %s", csv_filename)
            continue
        
        logger.info("提取的日期: %s", date_str)
        
        # 读取股票代码
        try:
            stock_codes = read_stock_codes_from_csv(csv_path)
            if not stock_codes:
                logger.warning("文件为空或未找到有效股票代码，跳过: %s", csv_path)
                continue
            logger.info("读取到 %d 只股票代码", len(stock_codes))
            logger.info("前10只股票: %s", stock_codes[:10])
        except Exception as e:
            logger.error("读取股票代码失败: %s", e)
            continue
        
        # 查找对应的模型
        base_model_dir = os.path.join(project_root, "data", "models", "chanModels")
        model_tag = find_model_tag(base_model_dir, date_str)
        
        if not model_tag:
            logger.warning("未找到日期 %s 对应的模型，跳过", date_str)
            logger.warning("请检查模型目录: %s", os.path.join(base_model_dir, date_str))
            continue
        
        logger.info("找到模型 tag: %s", model_tag)
        
        # 创建临时数据配置文件
        temp_data_config = data_cfg.copy()
        temp_data_config["data"]["instruments"] = stock_codes
        # 预测时间范围
        temp_data_config["data"]["start_time"] = args.start
        temp_data_config["data"]["end_time"] = args.end
        
        temp_data_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8')
        yaml.dump(temp_data_config, temp_data_file, allow_unicode=True, default_flow_style=False)
        temp_data_file.close()
        
        # 创建临时 pipeline 配置文件
        temp_pipeline_config = copy.deepcopy(cfg)
        temp_pipeline_config["data_config"] = temp_data_file.name
        
        # 设置模型目录（使用日期子目录）
        model_dir = os.path.join(base_model_dir, date_str)
        log_dir = os.path.join(project_root, "data", "logs", "chanModels", date_str)
        
        temp_pipeline_config["paths"]["model_dir"] = model_dir
        temp_pipeline_config["paths"]["log_dir"] = log_dir
        
        temp_pipeline_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8')
        yaml.dump(temp_pipeline_config, temp_pipeline_file, allow_unicode=True, default_flow_style=False)
        temp_pipeline_file.close()
        
        try:
            # 加载 IC 历史（用于动态权重计算）
            log_path = os.path.join(log_dir, "training_metrics.csv")
            ic_histories = _load_ic_histories(log_path)
            
            # 构建特征
            pipeline = QlibFeaturePipeline(temp_data_file.name)
            pipeline.build(include_label=False)
            features, _ = pipeline.get_slice(args.start, args.end)
            
            if features.empty:
                logger.warning("特征数据为空，跳过预测")
                continue
            
            # 诊断：实际返回的特征日期范围
            if isinstance(features.index, pd.MultiIndex) and "datetime" in features.index.names:
                dt = features.index.get_level_values("datetime")
                logger.info("实际用于预测的特征日期范围: %s 到 %s（样本=%d）",
                           dt.min(), dt.max(), len(features))
            
            # 加载模型并预测
            predictor = PredictorEngine(temp_pipeline_file.name)
            try:
                predictor.load_models(model_tag)
            except FileNotFoundError as e:
                logger.error("无法加载模型 (tag=%s): %s", model_tag, e)
                logger.error("请检查模型文件是否存在: %s", model_dir)
                continue
            
            final_pred, preds, weights = predictor.predict(features, ic_histories)
            
            logger.info("预测完成，IC 动态权重: %s", weights)
            
            # 将预测结果添加到列表（后续合并）
            # 使用与 run_predict.py 相同的格式：MultiIndex (datetime, instrument)
            if isinstance(final_pred.index, pd.MultiIndex):
                # MultiIndex: (datetime, instrument)
                pred_df = pd.DataFrame({"final": final_pred})
                # 添加其他模型的预测
                for name, series in preds.items():
                    pred_df[name] = series
                # 确保索引顺序正确
                pred_df = pred_df.reorder_levels(["datetime", "instrument"]).sort_index()
                # 重置索引以便后续合并
                pred_df = pred_df.reset_index()
                pred_df["source_date"] = date_str  # 标记来源日期
                all_predictions.append(pred_df)
            else:
                logger.warning("预测结果索引格式不是 MultiIndex，跳过该文件")
            
        except Exception as e:
            logger.error("预测失败: %s", e, exc_info=True)
            continue
        finally:
            # 清理临时文件
            try:
                os.unlink(temp_data_file.name)
                os.unlink(temp_pipeline_file.name)
            except Exception:
                pass
    
    # 合并所有预测结果并保存
    if all_predictions:
        logger.info("=" * 80)
        logger.info("合并预测结果并保存")
        logger.info("=" * 80)
        
        combined_df = pd.concat(all_predictions, ignore_index=True)
        
        # 如果有多行相同日期和股票的预测（来自不同源文件），取平均值
        if "source_date" in combined_df.columns:
            # 按日期和股票分组，对数值列取平均值
            numeric_cols = ["final"] + [col for col in combined_df.columns 
                                       if col not in ["datetime", "instrument", "source_date"]]
            agg_dict = {col: "mean" for col in numeric_cols if col in combined_df.columns}
            combined_df = combined_df.groupby(["datetime", "instrument"]).agg(agg_dict).reset_index()
        
        # 设置 MultiIndex 并保存（与 run_predict.py 格式一致）
        if "datetime" in combined_df.columns and "instrument" in combined_df.columns:
            combined_df = combined_df.set_index(["datetime", "instrument"])
            # 确保索引顺序正确
            if isinstance(combined_df.index, pd.MultiIndex):
                combined_df = combined_df.reorder_levels(["datetime", "instrument"]).sort_index()
            
            # 保存为 CSV（格式：datetime, instrument, final, lgb, mlp, stack, ...）
            combined_df.to_csv(output_file, index_label=["datetime", "instrument"], encoding="utf-8")
            logger.info("预测结果已保存到: %s", output_file)
            logger.info("共 %d 条预测记录", len(combined_df))
            if isinstance(combined_df.index, pd.MultiIndex):
                dt_level = combined_df.index.get_level_values("datetime")
                logger.info("日期范围: %s 到 %s", dt_level.min(), dt_level.max())
        else:
            logger.error("合并后的数据缺少 datetime 或 instrument 列，无法保存")
    else:
        logger.warning("没有生成任何预测结果")


if __name__ == "__main__":
    main()

