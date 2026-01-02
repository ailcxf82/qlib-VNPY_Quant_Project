"""
检查行业轮动数据源，验证数据获取是否正常。
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import qlib
from qlib.data import D

# 添加项目根目录到路径
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from feature.qlib_feature_pipeline import QlibFeaturePipeline
from utils import load_yaml_config

logger = logging.getLogger(__name__)


def check_industry_index_file(industry_path: str) -> list[str]:
    """检查行业指数文件是否存在，并读取行业指数列表"""
    logger.info("=" * 80)
    logger.info("步骤 1: 检查行业指数文件")
    logger.info("=" * 80)
    
    if not industry_path:
        logger.error("❌ industry_index_path 未配置")
        return []
    
    # 检查是否为文件路径
    if os.path.sep in industry_path or "/" in industry_path or "\\" in industry_path or industry_path.endswith(".txt"):
        if not os.path.exists(industry_path):
            logger.error("❌ 行业指数文件不存在: %s", industry_path)
            return []
        
        logger.info("✅ 文件存在: %s", industry_path)
        
        # 读取文件
        industry_list = []
        with open(industry_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith("#"):
                    # 处理制表符分隔的格式（代码\t开始日期\t结束日期）
                    # 或逗号分隔的格式
                    # 处理制表符分隔的格式（代码\t开始日期\t结束日期）
                    if "\t" in line:
                        # 制表符分隔：取第一列作为代码
                        parts = line.split("\t")
                        code = parts[0].strip()
                        if code:
                            industry_list.append(code)
                    elif "," in line:
                        # 逗号分隔
                        codes = [c.strip() for c in line.split(",") if c.strip()]
                        industry_list.extend(codes)
                    else:
                        # 单行单个代码
                        if line:
                            industry_list.append(line)
                    if line_num <= 5:  # 显示前5行
                        logger.info("  第 %d 行: %s", line_num, line)
        
        if not industry_list:
            logger.error("❌ 文件为空或格式不正确")
            return []
        
        logger.info("✅ 成功读取 %d 个行业指数", len(industry_list))
        logger.info("   前10个: %s", industry_list[:10])
        return industry_list
    else:
        # 逗号分隔的字符串
        industry_list = [code.strip() for code in industry_path.split(",") if code.strip()]
        logger.info("✅ 从配置字符串解析到 %d 个行业指数", len(industry_list))
        logger.info("   前10个: %s", industry_list[:10])
        return industry_list


def check_qlib_initialization(data_cfg: dict) -> bool:
    """检查 qlib 配置"""
    logger.info("=" * 80)
    logger.info("步骤 2: 检查 qlib 配置")
    logger.info("=" * 80)
    
    qlib_cfg = data_cfg.get("qlib", {})
    provider_uri = qlib_cfg.get("provider_uri")
    region = qlib_cfg.get("region", "cn")
    
    if not provider_uri:
        logger.error("❌ provider_uri 未配置")
        return False
    
    if not os.path.exists(provider_uri):
        logger.error("❌ qlib 数据目录不存在: %s", provider_uri)
        return False
    
    logger.info("✅ qlib 数据目录存在: %s", provider_uri)
    logger.info("   区域: %s", region)
    logger.info("   注意: qlib 将在特征提取时自动初始化")
    return True


def check_instruments_data(industry_list: list[str], start_time: str, end_time: str, data_cfg_path: str) -> bool:
    """检查行业指数数据是否可用"""
    logger.info("=" * 80)
    logger.info("步骤 3: 检查行业指数数据可用性")
    logger.info("=" * 80)
    
    if not industry_list:
        logger.error("❌ 行业指数列表为空")
        return False
    
    logger.info("测试时间范围: %s 至 %s", start_time, end_time)
    logger.info("测试行业指数数量: %d", len(industry_list))
    
    # 通过 QlibFeaturePipeline 来测试数据获取（会自动初始化 qlib）
    logger.info("通过 QlibFeaturePipeline 测试数据获取...")
    try:
        pipeline = QlibFeaturePipeline(data_cfg_path)
        # 这会自动初始化 qlib 并测试数据获取
        logger.info("✅ QlibFeaturePipeline 创建成功，qlib 已初始化")
    except Exception as e:
        logger.error("❌ QlibFeaturePipeline 创建失败: %s", e, exc_info=True)
        return False
    
    # 测试前5个行业指数
    test_instruments = industry_list[:5]
    logger.info("测试前 %d 个行业指数: %s", len(test_instruments), test_instruments)
    
    success_count = 0
    failed_instruments = []
    
    for inst in test_instruments:
        try:
            # 尝试获取基础数据
            data = D.features(
                instruments=[inst],
                fields=["$close", "$open", "$high", "$low", "$volume"],
                start_time=start_time,
                end_time=end_time,
                freq="day"
            )
            
            if data.empty:
                logger.warning("  ⚠️  %s: 数据为空", inst)
                failed_instruments.append(inst)
            else:
                data_start = data.index.get_level_values("datetime").min()
                data_end = data.index.get_level_values("datetime").max()
                logger.info("  ✅ %s: 数据可用，时间范围 %s 至 %s，共 %d 条记录",
                          inst, data_start, data_end, len(data))
                success_count += 1
        except Exception as e:
            logger.error("  ❌ %s: 数据获取失败 - %s", inst, str(e))
            failed_instruments.append(inst)
    
    if success_count == 0:
        logger.error("❌ 所有测试行业指数都无法获取数据")
        return False
    elif success_count < len(test_instruments):
        logger.warning("⚠️  部分行业指数无法获取数据: %s", failed_instruments)
        return True
    else:
        logger.info("✅ 所有测试行业指数数据可用")
        return True


def check_feature_extraction(data_cfg_path: str) -> bool:
    """检查特征提取是否正常"""
    logger.info("=" * 80)
    logger.info("步骤 4: 检查特征提取")
    logger.info("=" * 80)
    
    try:
        pipeline = QlibFeaturePipeline(data_cfg_path)
        logger.info("✅ 特征管道创建成功")
        
        logger.info("开始构建特征...")
        pipeline.build()
        logger.info("✅ 特征构建成功")
        
        features, labels = pipeline.get_all()
        
        if features is None or features.empty:
            logger.error("❌ 特征数据为空")
            return False
        
        logger.info("✅ 特征数据获取成功")
        logger.info("   特征形状: %s", features.shape)
        logger.info("   特征列数: %d", len(features.columns))
        logger.info("   特征列: %s", list(features.columns[:10]))  # 显示前10个特征
        
        if isinstance(features.index, pd.MultiIndex):
            datetime_level = features.index.get_level_values("datetime")
            instrument_level = features.index.get_level_values("instrument")
            logger.info("   时间范围: %s 至 %s", datetime_level.min(), datetime_level.max())
            logger.info("   行业数量: %d", len(instrument_level.unique()))
            logger.info("   总记录数: %d", len(features))
        
        # 检查缺失值
        missing_stats = features.isnull().sum()
        if missing_stats.sum() > 0:
            logger.warning("⚠️  存在缺失值:")
            for col, count in missing_stats[missing_stats > 0].head(10).items():
                logger.warning("   %s: %d 个缺失值 (%.2f%%)", col, count, count / len(features) * 100)
        else:
            logger.info("✅ 无缺失值")
        
        if labels is not None and not labels.empty:
            logger.info("✅ 标签数据获取成功")
            logger.info("   标签形状: %s", labels.shape)
            logger.info("   标签统计: min=%.6f, max=%.6f, mean=%.6f",
                      labels.min(), labels.max(), labels.mean())
        else:
            logger.warning("⚠️  标签数据为空")
        
        return True
        
    except Exception as e:
        logger.error("❌ 特征提取失败: %s", e, exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="检查行业轮动数据源")
    parser.add_argument(
        "--config",
        type=str,
        default="classify/config_industry_rotation.yaml",
        help="配置文件路径",
    )
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    
    os.chdir(_project_root)
    
    logger.info("=" * 80)
    logger.info("行业轮动数据源检查")
    logger.info("=" * 80)
    
    # 加载配置
    cfg = load_yaml_config(args.config)
    data_cfg = load_yaml_config(cfg["data_config"])
    
    logger.info("配置文件: %s", args.config)
    logger.info("数据配置: %s", cfg["data_config"])
    
    # 步骤 1: 检查行业指数文件
    industry_path = data_cfg["data"].get("industry_index_path", "")
    industry_list = check_industry_index_file(industry_path)
    
    if not industry_list:
        logger.error("=" * 80)
        logger.error("检查失败：无法读取行业指数列表")
        logger.error("=" * 80)
        return
    
    # 步骤 2: 检查 qlib 配置
    if not check_qlib_initialization(data_cfg):
        logger.error("=" * 80)
        logger.error("检查失败：qlib 配置错误")
        logger.error("=" * 80)
        return
    
    # 步骤 3: 检查行业指数数据
    start_time = data_cfg["data"].get("start_time", "2015-01-01")
    end_time = data_cfg["data"].get("end_time", "2024-12-31")
    
    if not check_instruments_data(industry_list, start_time, end_time, cfg["data_config"]):
        logger.error("=" * 80)
        logger.error("检查失败：行业指数数据不可用")
        logger.error("=" * 80)
        return
    
    # 步骤 4: 检查特征提取
    if not check_feature_extraction(cfg["data_config"]):
        logger.error("=" * 80)
        logger.error("检查失败：特征提取失败")
        logger.error("=" * 80)
        return
    
    logger.info("=" * 80)
    logger.info("✅ 所有检查通过！数据源配置正确，可以正常获取数据")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

