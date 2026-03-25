"""
训练CSI101专用模型
针对中小盘股票的特点进行优化
"""

import argparse
import logging
import os
import sys

from feature.qlib_feature_pipeline import QlibFeaturePipeline
from trainer.trainer import RollingTrainer
from utils import load_yaml_config


def parse_args():
    parser = argparse.ArgumentParser(description="训练CSI101专用模型")
    parser.add_argument(
        "--config",
        type=str,
        default="config/pipeline_csi101.yaml",
        help="CSI101专用pipeline配置文件路径",
    )
    parser.add_argument(
        "--gru_only",
        action="store_true",
        help="仅训练 GRU",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("开始训练CSI101专用模型")
    logger.info("=" * 80)
    
    cfg = load_yaml_config(args.config)
    
    if args.gru_only:
        cfg["base_models"] = ["gru"]
        cfg.setdefault("ensemble", {})
        cfg["ensemble"]["models"] = []
        cfg["ensemble"].setdefault("aggregator", "average")
        cfg.setdefault("stack", {})
        cfg["stack"]["enabled"] = False
        cfg.setdefault("oof_stacking", {})
        cfg["oof_stacking"]["enabled"] = False
    
    logger.info("\n【CSI101专用模型配置】")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"数据配置: {cfg['data_config']}")
    logger.info(f"基模型: {cfg['base_models']}")
    logger.info(f"模型保存路径: {cfg['paths']['model_dir']}")
    
    logger.info("\n【关键优化点】")
    logger.info("1. 训练窗口：LGB=360天，GRU=180天（适应中小盘快速变化）")
    logger.info("2. 验证窗口：20天（更快响应市场变化）")
    logger.info("3. 滚动步长：5天（更频繁更新）")
    logger.info("4. LGB参数：更浅的树(max_depth=5)，更强正则化")
    logger.info("5. GRU参数：更大的隐藏层(hidden_size=24)，更多训练轮数")
    logger.info("6. 特征工程：增加波动率、流动性、市值因子")
    
    logger.info("\n【开始训练】")
    
    try:
        trainer = RollingTrainer(args.config)
        trainer.train()
        
        logger.info("\n" + "=" * 80)
        logger.info("CSI101专用模型训练完成！")
        logger.info("=" * 80)
        
        logger.info("\n【下一步】")
        logger.info("1. 运行预测：python run_predict.py --config config/pipeline_csi101.yaml")
        logger.info("2. 运行回测：python backtest/rqalpha_backtest.py --pool csi101")
        logger.info("3. 对比结果：对比新模型与原模型的回测表现")
        
    except Exception as e:
        logger.error(f"训练失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
