"""
特征重要性分析脚本：从训练好的 LightGBM 模型中提取特征重要性，筛选前 N 个重要因子。

使用方法:
    python scripts/analyze_feature_importance.py --config config/pipeline.yaml --top_k 75
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径，确保可以导入项目模块
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root))

import pandas as pd

from models.lightgbm_model import LightGBMModelWrapper
from utils import load_yaml_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def find_model_files(model_dir: str) -> list[str]:
    """查找所有 *_lgb.txt 模型文件，返回对应的 tag 列表。"""
    model_dir_path = Path(model_dir)
    if not model_dir_path.exists():
        logger.warning("模型目录不存在: %s", model_dir)
        return []
    
    model_files = list(model_dir_path.glob("*_lgb.txt"))
    tags = [f.stem.replace("_lgb", "") for f in model_files]
    logger.info("找到 %d 个模型文件", len(tags))
    return sorted(tags)


def aggregate_feature_importance(
    model_dir: str,
    tags: list[str],
    importance_type: str = "gain",
) -> pd.DataFrame:
    """
    聚合多个模型的特征重要性。
    
    Returns:
        pd.DataFrame: 列包括 feature_name, importance_mean, importance_std, importance_sum, model_count
    """
    all_importances = []
    
    for tag in tags:
        try:
            lgb_model = LightGBMModelWrapper("config/model_lgb.yaml")
            lgb_model.load(model_dir, tag)
            importance = lgb_model.get_feature_importance(importance_type=importance_type)
            importance.name = tag  # 使用 tag 作为列名
            all_importances.append(importance)
            logger.debug("已加载模型 %s 的特征重要性，共 %d 个特征", tag, len(importance))
        except Exception as e:
            logger.warning("加载模型 %s 失败: %s", tag, e)
            continue
    
    if not all_importances:
        raise ValueError("未成功加载任何模型的特征重要性")
    
    # 合并所有模型的重要性，对齐特征名
    df = pd.DataFrame(all_importances).T
    df.columns = [f"model_{i}" for i in range(len(df.columns))]
    
    # 计算统计量
    result = pd.DataFrame({
        "feature_name": df.index,
        "importance_mean": df.mean(axis=1),
        "importance_std": df.std(axis=1),
        "importance_sum": df.sum(axis=1),
        "importance_max": df.max(axis=1),
        "importance_min": df.min(axis=1),
        "model_count": df.count(axis=1),  # 非空模型数量
    })
    
    # 按平均重要性排序
    result = result.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    
    logger.info("聚合了 %d 个模型的特征重要性，共 %d 个特征", len(all_importances), len(result))
    return result


def filter_top_features(
    importance_df: pd.DataFrame,
    top_k: int = 75,
    min_importance: float = 0.0,
) -> pd.DataFrame:
    """
    筛选前 top_k 个重要特征。
    
    Args:
        importance_df: 特征重要性 DataFrame
        top_k: 保留前 k 个特征
        min_importance: 最小重要性阈值
    
    Returns:
        筛选后的 DataFrame
    """
    # 先按最小重要性过滤
    filtered = importance_df[importance_df["importance_mean"] >= min_importance].copy()
    
    # 再取前 top_k 个
    if len(filtered) > top_k:
        filtered = filtered.head(top_k)
    
    logger.info("筛选出 %d 个重要特征（top_k=%d, min_importance=%.4f）", 
                len(filtered), top_k, min_importance)
    return filtered


def categorize_features(feature_names: list[str]) -> dict[str, list[str]]:
    """
    将特征按类型分类（Alpha158 因子分类）。
    
    Returns:
        dict: {category: [feature_names]}
    """
    categories = {
        "kbar": [],      # K线特征
        "price": [],     # 价格特征
        "rolling": [],   # 滚动特征
        "custom": [],    # 自定义特征
    }
    
    for feat in feature_names:
        feat_lower = feat.lower()
        if any(k in feat_lower for k in ["kmid", "klen", "kup", "klow", "ksft"]):
            categories["kbar"].append(feat)
        elif any(k in feat_lower for k in ["open0", "high0", "low0", "vwap0"]):
            categories["price"].append(feat)
        elif any(k in feat_lower for k in ["roc", "ma", "std", "beta", "max", "min", 
                                           "rank", "rsv", "corr", "cord", "sump", "vma",
                                           "resi", "qtlu", "qtld", "imax", "imin", "cntd", "vstd"]):
            categories["rolling"].append(feat)
        else:
            categories["custom"].append(feat)
    
    return categories


def main():
    parser = argparse.ArgumentParser(description="分析 LightGBM 特征重要性并筛选重要因子")
    parser.add_argument(
        "--config",
        type=str,
        default="config/pipeline.yaml",
        help="pipeline 配置文件路径",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=75,
        help="保留前 k 个重要特征（默认 75）",
    )
    parser.add_argument(
        "--min_importance",
        type=float,
        default=0.0,
        help="最小重要性阈值（默认 0.0）",
    )
    parser.add_argument(
        "--importance_type",
        type=str,
        default="gain",
        choices=["gain", "split"],
        help="重要性类型：gain（增益）或 split（分裂次数），默认 gain",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/logs/feature_importance.csv",
        help="输出文件路径（CSV 格式）",
    )
    parser.add_argument(
        "--output_top",
        type=str,
        default="data/logs/top_features.txt",
        help="输出前 N 个特征名称列表（每行一个）",
    )
    args = parser.parse_args()
    
    # 切换到项目根目录
    os.chdir(_project_root)
    
    # 加载配置
    cfg = load_yaml_config(args.config)
    model_dir = cfg["paths"]["model_dir"]
    log_dir = cfg["paths"]["log_dir"]
    
    # 确保输出目录存在
    os.makedirs(log_dir, exist_ok=True)
    
    # 查找所有模型文件
    tags = find_model_files(model_dir)
    if not tags:
        logger.error("未找到任何模型文件，请先运行训练")
        return
    
    # 聚合特征重要性
    logger.info("开始聚合 %d 个模型的特征重要性...", len(tags))
    importance_df = aggregate_feature_importance(model_dir, tags, args.importance_type)
    
    # 筛选重要特征
    top_features_df = filter_top_features(
        importance_df,
        top_k=args.top_k,
        min_importance=args.min_importance,
    )
    
    # 分类统计
    categories = categorize_features(top_features_df["feature_name"].tolist())
    logger.info("特征分类统计:")
    for cat, feats in categories.items():
        logger.info("  %s: %d 个", cat, len(feats))
    
    # 保存完整的重要性数据
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("完整特征重要性已保存到: %s", output_path)
    
    # 保存前 N 个特征名称列表
    output_top_path = Path(args.output_top)
    output_top_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_top_path, "w", encoding="utf-8") as f:
        for feat in top_features_df["feature_name"]:
            f.write(f"{feat}\n")
    logger.info("前 %d 个特征名称已保存到: %s", len(top_features_df), output_top_path)
    
    # 打印前 20 个重要特征
    logger.info("\n前 20 个重要特征:")
    for idx, row in top_features_df.head(20).iterrows():
        logger.info(
            "  %2d. %-40s 重要性: %.4f (std: %.4f, 模型数: %d)",
            idx + 1,
            row["feature_name"],
            row["importance_mean"],
            row["importance_std"],
            row["model_count"],
        )
    
    # 保存筛选后的特征重要性（仅前 N 个）
    top_output_path = output_path.parent / f"top_{args.top_k}_features.csv"
    top_features_df.to_csv(top_output_path, index=False, encoding="utf-8-sig")
    logger.info("前 %d 个特征重要性已保存到: %s", args.top_k, top_output_path)


if __name__ == "__main__":
    main()

