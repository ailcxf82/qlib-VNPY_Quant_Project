"""
运行预测流程：加载最新模型 -> 计算多模型预测 -> 输出融合结果。
"""

import argparse
import logging
import os
from typing import Dict

import pandas as pd

from feature.qlib_feature_pipeline import QlibFeaturePipeline
from predictor.predictor import PredictorEngine
from utils import load_yaml_config


def parse_args():
    parser = argparse.ArgumentParser(description="Qlib 因子模型预测")
    parser.add_argument("--config", type=str, default="config/pipeline.yaml")
    # parser.add_argument("--start", type=str, required=True, help="预测起始日期")
    # parser.add_argument("--end", type=str, required=True, help="预测结束日期")
    # parser.add_argument("--tag", type=str, default="auto", help="模型标识，auto 则使用最新窗口")
    
    parser.add_argument(
        "--start",
        type=str,
        default=os.environ.get("RUN_PRED_START", "2025-11-01"),
        help="预测起始日期，默认 2023-10-01，可通过环境变量 RUN_PRED_START 覆盖",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=os.environ.get("RUN_PRED_END", "2025-12-22"),
        help="预测结束日期，默认 2025-10-01，可通过环境变量 RUN_PRED_END 覆盖",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=os.environ.get("RUN_PRED_TAG", "auto"),
        help="模型标识，默认 auto，可通过环境变量 RUN_PRED_TAG 覆盖",
    )
    return parser.parse_args()


def _load_ic_histories(log_path: str) -> Dict[str, pd.Series]:
    if not os.path.exists(log_path):
        today = pd.Timestamp.today()
        base = pd.Series([0.1], index=[today])
        return {"lgb": base, "mlp": base, "stack": base, "qlib_ensemble": base}
    df = pd.read_csv(log_path, parse_dates=["valid_end"])
    histories = {
        "lgb": pd.Series(df["ic_lgb"].values, index=df["valid_end"]),
        "mlp": pd.Series(df["ic_mlp"].values, index=df["valid_end"]),
        "stack": pd.Series(df["ic_stack"].values, index=df["valid_end"]),
    }
    if "ic_qlib_ensemble" in df.columns:
        histories["qlib_ensemble"] = pd.Series(df["ic_qlib_ensemble"].values, index=df["valid_end"])
    else:
        histories["qlib_ensemble"] = histories["lgb"]
    return histories


def _infer_latest_from_models(model_dir: str) -> str:
    """当日志缺失时，尝试在模型目录中推断最新 tag。"""
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"模型目录不存在: {model_dir}，请先运行训练或指定 --tag")
    candidates = []
    try:
        for name in os.listdir(model_dir):
            if not name.endswith("_lgb.txt"):
                continue
            tag = name.replace("_lgb.txt", "")
            candidates.append(tag)
    except OSError as e:
        raise FileNotFoundError(f"无法读取模型目录 {model_dir}: {e}")
    if not candidates:
        raise FileNotFoundError(f"模型目录 {model_dir} 中未找到 *_lgb.txt 文件，无法推断 tag。请先运行训练或指定 --tag")
    return sorted(candidates)[-1]


def _latest_tag(log_path: str, model_dir: str) -> str:
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        latest = df.iloc[-1]["valid_end"].replace("-", "")
        return latest
    logging.warning("未找到训练日志 %s，将根据模型目录推断最新 tag", log_path)
    return _infer_latest_from_models(model_dir)


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
    logger.info("检测到 %d 个股票池: %s", len(instrument_pools), instrument_pools)
    logger.info("预测请求日期范围: %s 到 %s（RUN_PRED_START/RUN_PRED_END 或命令行参数）", args.start, args.end)
    
    # 在循环开始前，保存原始的基础路径（避免在循环中被修改）
    import copy
    original_paths = copy.deepcopy(cfg["paths"])
    base_model_dir = original_paths["model_dir"]
    base_log_dir = original_paths["log_dir"]
    
    # 确保基础路径是绝对路径或相对于项目根目录的路径
    if not os.path.isabs(base_model_dir):
        base_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), base_model_dir)
    if not os.path.isabs(base_log_dir):
        base_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), base_log_dir)
    
    # 为每个股票池分别预测
    for pool_name in instrument_pools:
        logger.info("=" * 80)
        logger.info("开始预测股票池: %s", pool_name)
        logger.info("=" * 80)
        
        # 创建临时配置文件
        import tempfile
        import yaml
        
        # 创建临时数据配置文件
        temp_data_config = data_cfg.copy()
        temp_data_config["data"]["instruments"] = pool_name
        
        temp_data_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8')
        yaml.dump(temp_data_config, temp_data_file, allow_unicode=True, default_flow_style=False)
        temp_data_file.close()
        
        # 创建临时pipeline配置文件
        temp_pipeline_config = copy.deepcopy(cfg)
        temp_pipeline_config["data_config"] = temp_data_file.name
        
        # 使用循环开始前保存的原始基础路径，拼接股票池特定的路径
        temp_pipeline_config["paths"]["model_dir"] = os.path.join(base_model_dir, f"{pool_name}_models")
        temp_pipeline_config["paths"]["log_dir"] = os.path.join(base_log_dir, f"{pool_name}_logs")
        # 预测文件夹保持统一，但文件名会包含股票池信息
        
        temp_pipeline_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8')
        yaml.dump(temp_pipeline_config, temp_pipeline_file, allow_unicode=True, default_flow_style=False)
        temp_pipeline_file.close()
        
        try:
            # 加载对应的日志和模型
            log_path = os.path.join(temp_pipeline_config["paths"]["log_dir"], "training_metrics.csv")
            model_dir = temp_pipeline_config["paths"]["model_dir"]
            
            # 检查模型目录是否存在
            if not os.path.exists(model_dir):
                logger.warning("股票池 %s 的模型目录不存在: %s，跳过该股票池", pool_name, model_dir)
                logger.warning("请先运行训练为股票池 %s 生成模型，或检查配置是否正确", pool_name)
                continue
            
            try:
                tag = _latest_tag(log_path, model_dir) if args.tag == "auto" else args.tag
            except FileNotFoundError as e:
                logger.error("股票池 %s 无法找到模型或日志: %s", pool_name, e)
                logger.error("请先运行训练为股票池 %s 生成模型", pool_name)
                continue
            
            ic_histories = _load_ic_histories(log_path)
            
            # 构建特征
            pipeline = QlibFeaturePipeline(temp_data_file.name)
            # 预测模式：只构建特征，不依赖 label，避免因为 label 的未来窗口/缺失导致预测日期被截断
            pipeline.build(include_label=False)
            features, _ = pipeline.get_slice(args.start, args.end)
            # 诊断：实际返回的特征日期范围（决定预测文件起点）
            try:
                if len(features) > 0 and isinstance(features.index, pd.MultiIndex) and "datetime" in features.index.names:
                    dt = features.index.get_level_values("datetime")
                    logger.info("股票池 %s 实际用于预测的特征日期范围: %s 到 %s（样本=%d）",
                                pool_name, dt.min(), dt.max(), len(features))
                    if pd.Timestamp(args.start) < dt.min():
                        logger.warning("股票池 %s：请求 start=%s 早于可用特征起点=%s，因此预测文件会从 %s 开始",
                                       pool_name, args.start, dt.min(), dt.min())
                else:
                    logger.warning("股票池 %s：get_slice 返回空特征（可能该范围无数据或清理后为空）", pool_name)
            except Exception as e:
                logger.debug("打印预测特征日期范围失败(可忽略): %s", e)
            
            # 预测
            predictor = PredictorEngine(temp_pipeline_file.name)
            try:
                predictor.load_models(tag)
            except FileNotFoundError as e:
                logger.error("股票池 %s 无法加载模型 (tag=%s): %s", pool_name, tag, e)
                logger.error("请检查模型文件是否存在: %s", model_dir)
                continue
            
            final_pred, preds, weights = predictor.predict(features, ic_histories)
            
            # 保存预测结果，文件名包含股票池信息
            # 新规则：预测文件名固定为 pred_{pool}.csv（不携带日期），便于传输/覆盖更新更稳定
            # 例如：data/predictions/pred_csi101.csv
            predictor.save_predictions(final_pred, preds, pool_name)
            logger.info("股票池 %s 预测完成，IC 动态权重: %s", pool_name, weights)
        except Exception as e:
            logger.error("股票池 %s 预测失败: %s", pool_name, e, exc_info=True)
            continue
        finally:
            # 清理临时文件
            if os.path.exists(temp_data_file.name):
                os.unlink(temp_data_file.name)
            if os.path.exists(temp_pipeline_file.name):
                os.unlink(temp_pipeline_file.name)


if __name__ == "__main__":
    main()


