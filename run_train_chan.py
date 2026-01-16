"""
基于 chanSignal CSV 文件的模型训练入口脚本。

功能：
1. 从 data/chanSignal 文件夹读取 CSV 文件（文件名格式：YYYYMMDD_chan.csv）
2. 从文件名提取日期（如 20250101_chan.csv -> 20250101）
3. 读取 CSV 文件内容作为股票代码列表
4. 根据文件名日期动态设置训练时间窗口
5. 训练结果保存到 data/models/chanModels

用法示例：
python run_train_chan.py
python run_train_chan.py --chan-signal-file data/chanSignal/20250101_chan.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import glob
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
import tempfile
import copy

from feature.qlib_feature_pipeline import QlibFeaturePipeline
from trainer.trainer import RollingTrainer
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


def parse_date_string(date_str: str) -> pd.Timestamp:
    """将日期字符串（YYYYMMDD）转换为 pd.Timestamp"""
    if len(date_str) == 8:
        return pd.Timestamp(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}")
    raise ValueError(f"不支持的日期格式: {date_str}")


def read_stock_codes_from_csv(csv_path: str) -> list[str]:
    """
    从 CSV 文件读取股票代码列表。
    
    支持格式：
    - 每行一个代码（无表头）
    - 包含 instrument 或 code 列的 CSV
    """
    df = pd.read_csv(csv_path, dtype=str, header=None)
    
    # 尝试读取第一行作为表头
    first_row = df.iloc[0, 0] if len(df) > 0 else ""
    
    # 如果第一行看起来像表头（包含 "instrument", "code" 等关键词），重新读取
    if isinstance(first_row, str) and any(keyword in first_row.lower() for keyword in ["instrument", "code", "stock"]):
        df = pd.read_csv(csv_path, dtype=str)
        # 查找包含代码的列
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
        codes = df.iloc[:, 0].astype(str).str.strip().dropna().tolist()
    
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
        elif code.isdigit():
            # 补零到6位
            cleaned_codes.append(code.zfill(6))
    
    return cleaned_codes


def calculate_time_window(end_date_str: str, train_days: int, valid_days: int) -> dict:
    """
    根据结束日期和训练配置计算时间窗口（单窗口训练，不滚动）。
    
    参数:
        end_date_str: 结束日期（YYYYMMDD 格式，CSV文件中的日期）
        train_days: 训练窗口天数
        valid_days: 验证窗口天数
    
    返回:
        包含 start_time, end_time, train_end, valid_start, valid_end 的字典
    """
    end_date = parse_date_string(end_date_str)
    
    # 计算训练结束日期：结束日期往前推 valid_days 天（为验证窗口留出空间）
    train_end_date = end_date - pd.Timedelta(days=valid_days)
    
    # 计算训练开始日期：训练结束日期往前推 train_days 天
    train_start_date = train_end_date - pd.Timedelta(days=train_days)
    
    # 验证窗口：从训练结束日期后一天开始，到结束日期
    valid_start_date = train_end_date + pd.Timedelta(days=1)
    valid_end_date = end_date
    
    return {
        "start_time": train_start_date.strftime("%Y-%m-%d"),  # 数据开始时间
        "end_time": end_date.strftime("%Y-%m-%d"),  # 数据结束时间
        "train_start": train_start_date.strftime("%Y-%m-%d"),
        "train_end": train_end_date.strftime("%Y-%m-%d"),
        "valid_start": valid_start_date.strftime("%Y-%m-%d"),
        "valid_end": valid_end_date.strftime("%Y-%m-%d"),
    }


def find_chan_signal_files(signal_dir: str = "data/chanSignal") -> list[str]:
    """查找所有 chanSignal CSV 文件"""
    signal_path = Path(signal_dir)
    if not signal_path.exists():
        return []
    
    pattern = str(signal_path / "*_chan.csv")
    files = glob.glob(pattern)
    return sorted(files)


def parse_args():
    parser = argparse.ArgumentParser(description="基于 chanSignal CSV 文件的模型训练")
    parser.add_argument(
        "--chan-signal-file",
        type=str,
        default=None,
        help="chanSignal CSV 文件路径（不指定则自动查找 data/chanSignal 下所有文件）",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/pipeline.yaml",
        help="pipeline 配置文件路径",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="config/data.yaml",
        help="数据配置文件路径（用于获取特征和标签配置）",
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
    
    # 获取训练窗口配置（单窗口训练，不需要 step_days）
    rolling_cfg = cfg.get("rolling", {})
    train_days = rolling_cfg.get("train_days", 720)
    valid_days = rolling_cfg.get("valid_days", 30)
    
    logger.info("训练窗口配置: train_days=%d, valid_days=%d（单窗口训练，不滚动）", 
                train_days, valid_days)
    
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
        logger.info("自动找到 %d 个 chanSignal 文件", len(csv_files))
    
    # 为每个 CSV 文件进行训练
    for csv_file in csv_files:
        csv_path = os.path.abspath(csv_file) if not os.path.isabs(csv_file) else csv_file
        
        if not os.path.exists(csv_path):
            logger.warning("文件不存在，跳过: %s", csv_path)
            continue
        
        # 从文件名提取日期
        filename = os.path.basename(csv_path)
        date_str = extract_date_from_filename(filename)
        
        if not date_str:
            logger.warning("无法从文件名提取日期，跳过: %s", filename)
            continue
        
        logger.info("=" * 80)
        logger.info("处理文件: %s", filename)
        logger.info("提取日期: %s", date_str)
        logger.info("=" * 80)
        
        # 读取股票代码列表
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
        
        # 计算时间窗口（单窗口训练，不滚动）
        try:
            time_window = calculate_time_window(date_str, train_days, valid_days)
            logger.info("训练窗口: [%s, %s]", time_window["train_start"], time_window["train_end"])
            logger.info("验证窗口: [%s, %s]", time_window["valid_start"], time_window["valid_end"])
            logger.info("数据时间范围: [%s, %s]", time_window["start_time"], time_window["end_time"])
        except Exception as e:
            logger.error("计算时间窗口失败: %s", e)
            continue
        
        # 创建临时数据配置文件
        temp_data_config = data_cfg.copy()
        # 设置股票代码列表（直接使用列表格式，_parse_instruments 会正确处理）
        temp_data_config["data"]["instruments"] = stock_codes
        # 设置时间窗口（需要包含训练和验证窗口）
        temp_data_config["data"]["start_time"] = time_window["start_time"]
        temp_data_config["data"]["end_time"] = time_window["end_time"]
        
        temp_data_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8')
        yaml.dump(temp_data_config, temp_data_file, allow_unicode=True, default_flow_style=False)
        temp_data_file.close()
        
        # 创建临时 pipeline 配置文件
        temp_pipeline_config = copy.deepcopy(cfg)
        temp_pipeline_config["data_config"] = temp_data_file.name
        
        # 设置模型保存路径
        base_model_dir = os.path.join(project_root, "data", "models", "chanModels")
        base_log_dir = os.path.join(project_root, "data", "logs", "chanModels")
        
        # 为每个日期创建子目录
        model_dir = os.path.join(base_model_dir, date_str)
        log_dir = os.path.join(base_log_dir, date_str)
        
        temp_pipeline_config["paths"]["model_dir"] = model_dir
        temp_pipeline_config["paths"]["log_dir"] = log_dir
        
        # 设置 step_days 为一个很大的值，确保只生成一个窗口（不滚动）
        temp_pipeline_config["rolling"]["step_days"] = 99999
        
        temp_pipeline_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8')
        yaml.dump(temp_pipeline_config, temp_pipeline_file, allow_unicode=True, default_flow_style=False)
        temp_pipeline_file.close()
        
        try:
            # 执行单窗口训练
            logger.info("开始训练单窗口模型（股票池: %d 只股票，日期: %s）", len(stock_codes), date_str)
            trainer = RollingTrainer(temp_pipeline_file.name)
            
            # 重写 _generate_windows 方法，只生成一个窗口
            from trainer.trainer import Window
            
            def single_window_generator():
                """只生成一个训练窗口（不滚动）"""
                yield Window(
                    train_start=time_window["train_start"],
                    train_end=time_window["train_end"],
                    valid_start=time_window["valid_start"],
                    valid_end=time_window["valid_end"],
                )
            
            trainer._generate_windows = single_window_generator
            
            # 重写训练方法，实现单窗口训练并保存为 CSV 文件日期
            def train_single_window():
                """训练并保存模型，使用 CSV 文件日期作为文件名"""
                trainer.pipeline.build()
                features, labels = trainer.pipeline.get_all()
                
                os.makedirs(trainer.paths["model_dir"], exist_ok=True)
                os.makedirs(trainer.paths["log_dir"], exist_ok=True)
                
                # 只处理第一个（也是唯一的）窗口
                for idx, window in enumerate(trainer._generate_windows()):
                    logger.info("==== 训练窗口: 训练 [%s, %s] 验证 [%s, %s] ====", 
                               window.train_start, window.train_end, window.valid_start, window.valid_end)
                    
                    train_feat, train_lbl = trainer._slice(features, labels, window.train_start, window.train_end, is_validation=False)
                    valid_feat, valid_lbl = trainer._slice(features, labels, window.valid_start, window.valid_end, is_validation=True)
                    
                    if len(train_feat) < trainer.cfg["rolling"].get("min_samples", 1000):
                        logger.warning("训练样本不足 (%d < %d)，跳过", 
                                     len(train_feat), trainer.cfg["rolling"].get("min_samples", 1000))
                        return
                    
                    has_valid = valid_feat is not None and not valid_feat.empty and valid_lbl is not None and not valid_lbl.empty
                    if not has_valid:
                        logger.warning("验证集为空，退化为仅训练")
                        valid_feat = None
                        valid_lbl = None
                    
                    # 计算归一化参数
                    logger.info("计算训练窗口归一化参数（仅使用训练集数据）")
                    train_feat_norm, norm_mean, norm_std = trainer.pipeline.normalize_features(train_feat)
                    
                    if has_valid:
                        valid_feat_norm = (valid_feat - norm_mean) / norm_std
                        valid_feat_norm = valid_feat_norm.clip(-5, 5)
                    else:
                        valid_feat_norm = None
                    
                    # 训练集成模型
                    trainer.ensemble.fit(
                        train_feat_norm,
                        train_lbl,
                        valid_feat_norm,
                        valid_lbl,
                        history_feat=train_feat_norm,
                    )
                    
                    # 获取 LightGBM 预测和叶子索引（用于 stack 模型）
                    train_blend, train_preds, train_aux = trainer.ensemble.predict(train_feat_norm)
                    lgb_train_pred = train_preds.get("lgb")
                    lgb_train_leaf = train_aux.get("lgb")
                    if lgb_train_pred is None or lgb_train_leaf is None:
                        raise RuntimeError("LeafStackModel 需要 LightGBM 输出，请在 ensemble.models 中包含 `lgb`")
                    
                    valid_pred = valid_leaf = None
                    if has_valid:
                        valid_blend, valid_preds, valid_aux = trainer.ensemble.predict(valid_feat_norm)
                        if valid_preds is not None:
                            valid_pred = valid_preds.get("lgb")
                        if valid_aux is not None:
                            valid_leaf = valid_aux.get("lgb")
                    
                    # 训练 stack 模型（学习残差）
                    train_leaf = lgb_train_leaf
                    train_residual = train_lbl - lgb_train_pred
                    valid_residual = None if (not has_valid or valid_pred is None) else valid_lbl - valid_pred
                    trainer.stack.fit(train_leaf, train_residual, valid_leaf, valid_residual)
                    
                    # 保存模型（使用 CSV 文件日期）
                    model_tag = date_str
                    trainer.ensemble.save(trainer.paths["model_dir"], model_tag)
                    trainer.stack.save(trainer.paths["model_dir"], model_tag)
                    
                    # 保存归一化参数
                    import json
                    norm_meta_path = os.path.join(trainer.paths["model_dir"], f"{model_tag}_norm_meta.json")
                    norm_meta = {
                        "feature_mean": norm_mean.to_dict(),
                        "feature_std": norm_std.to_dict(),
                        "train_start": window.train_start,
                        "train_end": window.train_end,
                        "valid_start": window.valid_start,
                        "valid_end": window.valid_end,
                    }
                    with open(norm_meta_path, "w", encoding="utf-8") as fp:
                        json.dump(norm_meta, fp, ensure_ascii=False, indent=2, default=str)
                    logger.info("归一化参数已保存: %s", norm_meta_path)
                    
                    logger.info("单窗口模型训练完成，模型文件: %s", model_tag)
                    break  # 只训练一个窗口
            
            trainer.train = train_single_window
            trainer.train()
            
            logger.info("训练完成，模型保存到: %s", model_dir)
        except Exception as e:
            logger.error("训练失败: %s", e)
            import traceback
            traceback.print_exc()
        finally:
            # 清理临时文件（安全删除，避免文件不存在时报错）
            try:
                if os.path.exists(temp_data_file.name):
                    os.unlink(temp_data_file.name)
            except Exception as e:
                logger.debug(f"删除临时数据配置文件失败: {e}")
            
            try:
                if os.path.exists(temp_pipeline_file.name):
                    os.unlink(temp_pipeline_file.name)
            except Exception as e:
                logger.debug(f"删除临时pipeline配置文件失败: {e}")


if __name__ == "__main__":
    main()

