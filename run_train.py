"""
运行训练流程：加载配置 -> 滚动训练 -> 保存模型。
支持多个股票池，为每个股票池分别训练并保存到不同的模型文件夹。
"""

import argparse
import copy
import logging
import os

from feature.qlib_feature_pipeline import QlibFeaturePipeline
from trainer.trainer import RollingTrainer
from utils import load_yaml_config


def parse_args():
    parser = argparse.ArgumentParser(description="Qlib 因子模型训练")
    parser.add_argument(
        "--config",
        type=str,
        default="config/pipeline.yaml",
        help="pipeline 配置文件路径",
    )
    parser.add_argument(
        "--gru_only",
        action="store_true",
        help="仅训练 GRU：自动设置 base_models=['gru']，并关闭 stack 与 oof_stacking（不改动原配置文件）",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default=None,
        help="覆盖 pipeline 中的 data 配置文件路径（用于因子消融/临时配置，不修改磁盘上的 pipeline 文件）",
    )
    parser.add_argument(
        "--active-feature-sets",
        type=str,
        default=None,
        help='覆盖 data.active_feature_sets，逗号分隔集合名，例如 "gru_ohlcv,lgb_fundamental"',
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="覆盖 data.label（qlib 标签表达式）；不传则使用配置文件中的标签",
    )
    return parser.parse_args()


def _log_torch_env():
    """一次性打印 torch/CUDA/显卡摘要，方便一眼确认训练设备。"""
    logger = logging.getLogger(__name__)
    try:
        import torch
        msg = [f"torch={torch.__version__}", f"cuda_available={torch.cuda.is_available()}"]
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            total = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
            msg.append(f"device=cuda:{idx}({name})")
            msg.append(f"total_mem={total:.2f}GB")
            msg.append(f"cuda_ver={torch.version.cuda}")
            try:
                msg.append(f"cudnn_ver={torch.backends.cudnn.version()}")
            except Exception:
                pass
        else:
            msg.append("device=cpu")
        logger.info("[env] %s", " | ".join(msg))
    except Exception as e:
        logger.warning("打印 torch 环境摘要失败（忽略继续）：%s", e)


def main():
    args = parse_args()
    logging.basicConfig( 
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    _log_torch_env()
    
    # 加载配置
    cfg = load_yaml_config(args.config)
    if args.gru_only:
        # 只训练 GRU：不依赖 lgb/stack
        cfg["base_models"] = ["gru"]
        cfg.setdefault("ensemble", {})
        # 让模型列表从 base_models 自动推导
        cfg["ensemble"]["models"] = []
        # 单模型不需要融合器训练
        cfg["ensemble"].setdefault("aggregator", "average")
        cfg.setdefault("stack", {})
        cfg["stack"]["enabled"] = False
        cfg.setdefault("oof_stacking", {})
        cfg["oof_stacking"]["enabled"] = False
    data_config_path = args.data_config if args.data_config else cfg["data_config"]
    data_cfg = load_yaml_config(data_config_path)
    if args.active_feature_sets:
        names = [s.strip() for s in args.active_feature_sets.split(",") if s.strip()]
        if names:
            data_cfg.setdefault("data", {})["active_feature_sets"] = names
    if args.label is not None:
        data_cfg.setdefault("data", {})["label"] = args.label

    # 解析股票池列表
    instruments_config = data_cfg["data"]["instruments"]
    instrument_pools = QlibFeaturePipeline._parse_instrument_pools(instruments_config)
    
    logger = logging.getLogger(__name__)
    logger.info("检测到 %d 个股票池: %s", len(instrument_pools), instrument_pools)
    afs = (data_cfg.get("data") or {}).get("active_feature_sets")
    logger.info(
        "数据配置: data_config=%s | active_feature_sets=%s",
        data_config_path,
        afs,
    )
    
    # 在循环开始前，保存原始的基础路径（避免在循环中被修改）
    original_paths = copy.deepcopy(cfg["paths"])
    base_model_dir = original_paths["model_dir"]
    base_log_dir = original_paths["log_dir"]
    
    # 确保基础路径是绝对路径或相对于项目根目录的路径
    if not os.path.isabs(base_model_dir):
        base_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), base_model_dir)
    if not os.path.isabs(base_log_dir):
        base_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), base_log_dir)
    
    # 为每个股票池分别训练
    for pool_name in instrument_pools:
        logger.info("=" * 80)
        logger.info("开始训练股票池: %s", pool_name)
        logger.info("=" * 80)
        
        # 创建临时配置文件，只包含当前股票池
        import tempfile
        import yaml

        # 创建临时数据配置文件（深拷贝，避免多股票池时污染共用嵌套字典）
        temp_data_config = copy.deepcopy(data_cfg)
        temp_data_config["data"]["instruments"] = pool_name
        
        temp_data_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8')
        yaml.dump(temp_data_config, temp_data_file, allow_unicode=True, default_flow_style=False)
        temp_data_file.close()
        
        # 创建临时pipeline配置文件，修改路径和data_config
        temp_pipeline_config = copy.deepcopy(cfg)
        temp_pipeline_config["data_config"] = temp_data_file.name
        
        # 使用循环开始前保存的原始基础路径，拼接股票池特定的路径
        temp_pipeline_config["paths"]["model_dir"] = os.path.join(base_model_dir, f"{pool_name}_models")
        temp_pipeline_config["paths"]["log_dir"] = os.path.join(base_log_dir, f"{pool_name}_logs")
        
        temp_pipeline_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8')
        yaml.dump(temp_pipeline_config, temp_pipeline_file, allow_unicode=True, default_flow_style=False)
        temp_pipeline_file.close()
        
        try:
            # 使用临时配置文件进行训练
            trainer = RollingTrainer(temp_pipeline_file.name)
            trainer.train()
            logger.info("股票池 %s 训练完成", pool_name)
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


