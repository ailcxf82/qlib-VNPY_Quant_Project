"""
运行训练流程：加载配置 -> 滚动训练 -> 保存模型。
支持多个股票池，为每个股票池分别训练并保存到不同的模型文件夹。
"""

import argparse
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
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig( 
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 加载配置
    cfg = load_yaml_config(args.config)
    data_cfg = load_yaml_config(cfg["data_config"])
    
    # 解析股票池列表
    instruments_config = data_cfg["data"]["instruments"]
    instrument_pools = QlibFeaturePipeline._parse_instrument_pools(instruments_config)
    
    logger = logging.getLogger(__name__)
    logger.info("检测到 %d 个股票池: %s", len(instrument_pools), instrument_pools)
    
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
    
    # 为每个股票池分别训练
    for pool_name in instrument_pools:
        logger.info("=" * 80)
        logger.info("开始训练股票池: %s", pool_name)
        logger.info("=" * 80)
        
        # 创建临时配置文件，只包含当前股票池
        import tempfile
        import yaml
        import shutil
        
        # 创建临时数据配置文件
        temp_data_config = data_cfg.copy()
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
            # 清理临时文件
            os.unlink(temp_data_file.name)
            os.unlink(temp_pipeline_file.name)


if __name__ == "__main__":
    main()


