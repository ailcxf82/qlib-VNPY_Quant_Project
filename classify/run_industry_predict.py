"""
行业轮动预测系统预测脚本。

使用训练好的 IndustryGRU 模型进行行业轮动预测，输出排名靠前的行业。
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

# 添加项目根目录到路径
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from feature.qlib_feature_pipeline import QlibFeaturePipeline
from utils import load_yaml_config

# 导入 IndustryGRU 模型
from classify.pytorch_industry_gru import IndustryGRUWrapper

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="行业轮动预测系统预测")
    parser.add_argument(
        "--config",
        type=str,
        default="classify/config_industry_rotation.yaml",
        help="行业轮动配置文件路径",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="预测日期（格式：YYYY-MM-DD），默认为最新交易日",
    )
    parser.add_argument(
        "--model_tag",
        type=str,
        default=None,
        help="模型标签（用于加载特定版本的模型）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    
    # 切换到项目根目录
    os.chdir(_project_root)
    
    logger.info("=" * 80)
    logger.info("开始行业轮动预测")
    logger.info("=" * 80)
    
    # 加载配置
    cfg = load_yaml_config(args.config)
    data_cfg = load_yaml_config(cfg["data_config"])
    
    # 确定预测日期
    predict_date = args.date
    if predict_date is None:
        # 使用最新交易日
        predict_date = datetime.now().strftime("%Y-%m-%d")
        logger.info("未指定预测日期，使用: %s", predict_date)
    
    # 创建特征管道
    pipeline = QlibFeaturePipeline(cfg["data_config"])
    
    # 加载模型
    model_config = cfg.get("industry_gru_config", {})
    model = IndustryGRUWrapper(model_config)
    
    # 模型路径
    model_dir = cfg["paths"]["model_dir"]
    model_tag = args.model_tag or "latest"
    
    try:
        model.load(model_dir, model_tag)
        logger.info("模型加载成功: %s/%s", model_dir, model_tag)
    except FileNotFoundError as e:
        logger.error("模型加载失败: %s", e)
        logger.error("请先运行训练脚本生成模型")
        return
    
    # 获取行业列表
    industry_path = data_cfg["data"].get("industry_index_path", "")
    if not industry_path:
        logger.error("行业指数路径未配置！请在配置文件中设置 industry_index_path")
        return
    
    # 读取行业列表
    industry_list = []
    if os.path.exists(industry_path):
        with open(industry_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "\t" in line:
                        code = line.split("\t")[0].strip()
                    elif "," in line:
                        codes = [c.strip() for c in line.split(",") if c.strip()]
                        industry_list.extend(codes)
                        continue
                    else:
                        code = line
                    if code:
                        industry_list.append(code)
    else:
        # 尝试作为逗号分隔的字符串解析
        industry_list = [code.strip() for code in industry_path.split(",") if code.strip()]
    
    if not industry_list:
        logger.error("无法获取行业列表，请检查 industry_index_path 配置")
        return
    
    logger.info("共 %d 个行业指数", len(industry_list))
    
    # 获取模型配置中的序列长度
    sequence_length = model_config.get("sequence_length", 60)
    
    # 计算需要的历史数据范围
    # 需要 sequence_length 天的历史数据来构建序列
    predict_date_ts = pd.Timestamp(predict_date)
    history_start = (predict_date_ts - pd.Timedelta(days=sequence_length + 60)).strftime("%Y-%m-%d")  # 多取60天作为缓冲
    
    # 更新数据配置的时间范围，确保能获取到足够的历史数据
    original_start = data_cfg["data"]["start_time"]
    original_end = data_cfg["data"]["end_time"]
    
    # 如果预测日期晚于配置的结束日期，需要扩展
    if predict_date_ts > pd.Timestamp(original_end):
        data_cfg["data"]["end_time"] = (predict_date_ts + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        logger.info("预测日期晚于配置的结束日期，已扩展数据时间范围至: %s", data_cfg["data"]["end_time"])
    
    # 如果历史开始日期早于配置的开始日期，需要扩展
    if pd.Timestamp(history_start) < pd.Timestamp(original_start):
        data_cfg["data"]["start_time"] = history_start
        logger.info("历史数据需求早于配置的开始日期，已扩展数据时间范围至: %s", history_start)
    
    logger.info("提取预测期特征...")
    logger.info("  预测日期: %s", predict_date)
    logger.info("  数据时间范围: %s 至 %s", data_cfg["data"]["start_time"], data_cfg["data"]["end_time"])
    
    # 重新创建特征管道（使用更新后的时间范围）
    pipeline = QlibFeaturePipeline(cfg["data_config"])
    
    # 构建特征和标签（用于获取特征）
    # 注意：预测时不需要标签，但需要构建特征管道
    # 使用 include_label=False 避免因为标签缺失而截断数据
    pipeline.build(include_label=False)
    
    # 获取所有特征数据
    all_features, _ = pipeline.get_all()
    
    # 提取预测日期及之前的数据（用于构建序列）
    predict_date_ts = pd.Timestamp(predict_date)
    mask = all_features.index.get_level_values("datetime") <= predict_date_ts
    history_features = all_features.loc[mask]
    
    if history_features.empty:
        logger.error("无法获取历史特征数据，请检查数据时间范围")
        return
    
    # 获取预测日期的特征（每个行业）
    predict_mask = history_features.index.get_level_values("datetime") == predict_date_ts
    predict_features = history_features.loc[predict_mask]
    
    if predict_features.empty:
        logger.warning("预测日期 %s 没有数据，尝试使用最近一个交易日", predict_date)
        # 使用最近一个交易日
        last_date = history_features.index.get_level_values("datetime").max()
        predict_mask = history_features.index.get_level_values("datetime") == last_date
        predict_features = history_features.loc[predict_mask]
        predict_date_actual = last_date.strftime("%Y-%m-%d")
        logger.info("使用最近交易日: %s", predict_date_actual)
    else:
        predict_date_actual = predict_date
    
    if predict_features.empty:
        logger.error("无法获取预测特征数据")
        return
    
    logger.info("提取到 %d 个行业的特征数据", len(predict_features))
    
    # 加载归一化参数
    norm_meta_path = os.path.join(model_dir, f"{model_tag}_norm_meta.json")
    norm_mean = None
    norm_std = None
    if os.path.exists(norm_meta_path):
        import json
        with open(norm_meta_path, "r", encoding="utf-8") as fp:
            norm_meta = json.load(fp)
        norm_mean = pd.Series(norm_meta.get("feature_mean", {}))
        norm_std = pd.Series(norm_meta.get("feature_std", {}))
        logger.info("已加载归一化参数")
    else:
        logger.warning("未找到归一化参数文件，将使用特征管道的归一化方法")
        # 使用特征管道的归一化方法
        predict_features_norm, _, _ = QlibFeaturePipeline.normalize_features(predict_features)
    
    # 如果加载了归一化参数，使用它们进行归一化
    if norm_mean is not None and norm_std is not None:
        predict_features_norm = (predict_features - norm_mean) / norm_std
        predict_features_norm = predict_features_norm.clip(-5, 5)
    
    # 进行预测
    logger.info("开始预测...")
    try:
        # 使用历史数据作为补充（用于构建完整序列）
        predictions = model.predict(predict_features_norm, history_feat=history_features)
        logger.info("预测完成，共 %d 个行业的预测结果", len(predictions))
    except Exception as e:
        logger.error("预测失败: %s", e, exc_info=True)
        return
    
    # 构建预测结果 DataFrame
    pred_df = pd.DataFrame({
        "industry_code": predictions.index.get_level_values("instrument") if isinstance(predictions.index, pd.MultiIndex) else predictions.index,
        "prediction_score": predictions.values,
    })
    
    # 按预测分数排序（降序，分数越高排名越靠前）
    pred_df = pred_df.sort_values("prediction_score", ascending=False)
    pred_df["rank"] = range(1, len(pred_df) + 1)
    
    # 获取配置中的 top_k
    top_k = cfg.get("portfolio", {}).get("top_k", 10)
    
    # 显示排名前 K 的行业
    logger.info("=" * 80)
    logger.info("行业预测排名（预测日期: %s）", predict_date_actual)
    logger.info("=" * 80)
    logger.info("排名前 %d 的行业:", top_k)
    logger.info("")
    
    top_industries = pred_df.head(top_k)
    for idx, row in top_industries.iterrows():
        logger.info("  %2d. %s (预测分数: %.6f)", row["rank"], row["industry_code"], row["prediction_score"])
    
    logger.info("")
    logger.info("完整排名:")
    for idx, row in pred_df.iterrows():
        logger.info("  %2d. %s (预测分数: %.6f)", row["rank"], row["industry_code"], row["prediction_score"])
    
    # 保存预测结果
    output_dir = cfg["paths"]["prediction_dir"]
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"industry_pred_{predict_date_actual.replace('-', '')}.csv")
    
    # 重新组织 DataFrame，包含更多信息
    output_df = pred_df[["rank", "industry_code", "prediction_score"]].copy()
    output_df.columns = ["排名", "行业代码", "预测分数"]
    output_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    logger.info("")
    logger.info("预测完成！")
    logger.info("  预测日期: %s", predict_date_actual)
    logger.info("  结果已保存: %s", output_path)
    logger.info("  推荐关注排名前 %d 的行业", top_k)


if __name__ == "__main__":
    main()







