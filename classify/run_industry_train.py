"""
行业轮动预测系统训练脚本。

使用 IndustryGRU 模型进行行业轮动预测训练。
"""

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

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


@dataclass
class Window:
    """滚动窗口数据结构"""
    train_start: str
    train_end: str
    valid_start: str
    valid_end: str


def _rank_ic(pred: pd.Series, label: pd.Series) -> float:
    """计算 Rank IC（Spearman 相关系数）"""
    pred, label = pred.align(label, join="inner")
    if pred.empty:
        return float("nan")
    return pred.rank().corr(label, method="spearman")


def parse_args():
    parser = argparse.ArgumentParser(description="行业轮动预测系统训练")
    parser.add_argument(
        "--config",
        type=str,
        default="classify/config_industry_rotation.yaml",
        help="行业轮动配置文件路径",
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
    logger.info("开始行业轮动预测系统训练")
    logger.info("=" * 80)
    
    # 加载配置
    cfg = load_yaml_config(args.config)
    data_cfg = load_yaml_config(cfg["data_config"])
    
    logger.info("配置文件: %s", args.config)
    logger.info("数据配置: %s", cfg["data_config"])
    
    # 检查行业指数路径配置
    industry_path = data_cfg["data"].get("industry_index_path", "")
    if not industry_path:
        logger.warning(
            "行业指数路径未配置！请在 config_data_industry.yaml 中设置 industry_index_path。"
        )
        logger.info("示例格式: industry_index_path: 'sw_l1_801010,sw_l1_801020,...'")
    
    # 创建特征管道
    pipeline = QlibFeaturePipeline(cfg["data_config"])
    
    # 创建模型（使用 IndustryGRU）
    model_config = cfg.get("industry_gru_config", {})
    model = IndustryGRUWrapper(model_config)
    
    logger.info("开始滚动训练...")
    
    # 获取数据时间范围
    start_time = data_cfg["data"]["start_time"]
    end_time = data_cfg["data"]["end_time"]
    
    # 获取滚动配置
    rolling_cfg = cfg.get("rolling", {})
    train_months = rolling_cfg.get("train_months", 24)
    valid_months = rolling_cfg.get("valid_months", 1)
    test_months = rolling_cfg.get("test_months", 1)
    step_months = rolling_cfg.get("step_months", 1)
    
    # 训练集 IC 计算配置
    train_ic_config = cfg.get("train_ic", {})
    compute_train_ic = train_ic_config.get("enabled", True)  # 是否计算训练集 IC
    train_ic_method = train_ic_config.get("method", "ts_cv")  # "ts_cv" 或 "none"
    logger.info("训练集 IC 配置: enabled=%s, method=%s", compute_train_ic, train_ic_method)
    
    logger.info("训练配置:")
    logger.info("  - 时间范围: %s 至 %s", start_time, end_time)
    logger.info("  - 训练窗口: %d 个月", train_months)
    logger.info("  - 验证窗口: %d 个月", valid_months)
    logger.info("  - 测试窗口: %d 个月", test_months)
    logger.info("  - 滚动步长: %d 个月", step_months)
    
    # 解析标签表达式，获取需要的未来天数
    label_expr = data_cfg["data"].get("label", "Ref($close, -10)/$close - 1")
    label_future_days = 0
    if "Ref($close, -" in label_expr:
        match = re.search(r'Ref\(\$close,\s*-(\d+)\)', label_expr)
        if match:
            label_future_days = int(match.group(1))
            logger.info(
                "标签需要未来 %d 天数据来计算",
                label_future_days,
            )
    
    # 获取序列长度（用于计算 purge gap）
    sequence_length = model_config.get("sequence_length", 60)
    
    # 计算 purge gap：序列长度 + 标签未来天数
    # 这是为了防止数据泄露：训练集结束日期需要提前 purge_gap 天
    # 因为模型需要 sequence_length 天的历史数据，而标签需要 label_future_days 天的未来数据
    purge_gap = sequence_length + label_future_days
    logger.info(
        "Purge Gap 配置: 序列长度=%d, 标签未来天数=%d, Purge Gap=%d 天",
        sequence_length,
        label_future_days,
        purge_gap,
    )
    logger.info(
        "训练集结束日期将提前 %d 天（Purge Gap），验证集开始日期从训练集原始结束日期+1天开始",
        purge_gap,
    )
    
    # 构建特征和标签
    logger.info("构建特征和标签...")
    pipeline.build()
    features, labels = pipeline.get_all()
    
    # 记录数据时间范围
    if len(features) > 0:
        data_start = features.index.get_level_values("datetime").min()
        data_end = features.index.get_level_values("datetime").max()
        logger.info("特征数据时间范围: %s 至 %s，共 %d 条记录", data_start, data_end, len(features))
    else:
        logger.error("特征数据为空，无法继续训练")
        return
    
    # 创建输出目录
    paths = cfg.get("paths", {})
    model_dir = paths.get("model_dir", "data/models/industry_rotation")
    log_dir = paths.get("log_dir", "data/logs/industry_rotation")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 确定窗口生成使用的结束日期
    # 使用实际数据的日期范围，而不是配置的日期范围
    # 因为 qlib 数据可能未更新到配置的结束日期，或者标签需要未来数据导致最后几天被过滤
    actual_end = pd.Timestamp(data_end)
    config_end = pd.Timestamp(end_time)
    
    if actual_end < config_end:
        logger.warning(
            "实际数据结束日期 (%s) 早于配置的结束日期 (%s)，将使用实际数据结束日期来生成窗口",
            actual_end.strftime("%Y-%m-%d"),
            config_end.strftime("%Y-%m-%d")
        )
        logger.warning("可能原因：1) qlib 数据未更新到该日期；2) 标签计算需要未来数据导致最后几天被过滤")
    
    logger.info("窗口生成配置:")
    logger.info("  - 配置的结束日期: %s", config_end.strftime("%Y-%m-%d"))
    logger.info("  - 实际数据结束日期: %s", actual_end.strftime("%Y-%m-%d"))
    logger.info("  - 窗口生成将使用实际数据结束日期: %s", actual_end.strftime("%Y-%m-%d"))
    
    # 生成滚动窗口（考虑 purge gap）
    def generate_windows() -> Iterable[Window]:
        """生成滚动训练窗口（考虑 purge gap 防止数据泄露）"""
        start = pd.Timestamp(start_time)
        end = actual_end  # 使用实际数据的结束日期
        train_offset = pd.DateOffset(months=train_months)
        valid_offset = pd.DateOffset(months=valid_months)
        step = pd.DateOffset(months=step_months)
        
        # cursor 指向验证起点，前推 train_offset 即训练区间
        # 但需要考虑 purge gap：训练集实际可用结束日期 = cursor - purge_gap
        cursor = start + train_offset
        window_count = 0
        while cursor + valid_offset <= end:
            # 训练集原始结束日期（用于确定验证集开始日期）
            train_end_original = cursor - pd.Timedelta(days=1)
            # 训练集实际可用结束日期（提前 purge_gap 天，防止数据泄露）
            train_end_adjusted = train_end_original - pd.Timedelta(days=purge_gap)
            # 训练集开始日期
            train_start = cursor - train_offset
            
            # 验证集开始日期：从训练集原始结束日期+1天开始
            valid_start = train_end_original + pd.Timedelta(days=1)
            # 验证集结束日期
            valid_end = cursor + valid_offset - pd.Timedelta(days=1)
            
            # 检查训练集是否有效（开始日期 < 结束日期）
            if train_end_adjusted < train_start:
                logger.warning(
                    "窗口 %d: 训练集调整后结束日期 %s 早于开始日期 %s（Purge Gap=%d天），跳过该窗口",
                    window_count,
                    train_end_adjusted.strftime("%Y-%m-%d"),
                    train_start.strftime("%Y-%m-%d"),
                    purge_gap,
                )
                cursor += step
                continue
            
            yield Window(
                train_start=train_start.strftime("%Y-%m-%d"),
                train_end=train_end_adjusted.strftime("%Y-%m-%d"),  # 使用调整后的结束日期
                valid_start=valid_start.strftime("%Y-%m-%d"),
                valid_end=valid_end.strftime("%Y-%m-%d"),
            )
            window_count += 1
            cursor += step
    
    def slice_data(
        feat: pd.DataFrame,
        lbl: pd.Series,
        start: str,
        end: str,
        is_validation: bool = False,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """按时间范围切片特征和标签"""
        idx = feat.index
        if not isinstance(idx, pd.MultiIndex):
            raise ValueError(f"特征索引应为 MultiIndex，实际为 {type(idx)}")
        
        if "datetime" not in idx.names:
            raise ValueError(f"索引层级中未找到 'datetime'，当前层级: {idx.names}")
        
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        
        # 标签需要未来数据，需要提前结束日期
        # 注意：训练集的结束日期在 generate_windows 中已经提前了 purge_gap
        # 这里只需要再提前 label_future_days（用于标签计算）
        orig_end_ts = end_ts
        if label_future_days > 0:
            end_ts = end_ts - pd.Timedelta(days=label_future_days)
            if end_ts < start_ts:
                seg = "验证集" if is_validation else "训练集"
                logger.warning(
                    "%s [%s, %s] 需要未来 %d 天数据，调整后结束日期 %s 早于开始日期，返回空集",
                    seg,
                    start,
                    end,
                    label_future_days,
                    end_ts.strftime("%Y-%m-%d"),
                )
                return pd.DataFrame(), pd.Series(dtype=float)
        
        datetime_level = idx.get_level_values("datetime")
        mask = (datetime_level >= start_ts) & (datetime_level <= end_ts)
        
        feat_slice = feat.loc[mask]
        lbl_slice = lbl.loc[mask]
        
        # 过滤掉标签为 NaN 的数据
        if not lbl_slice.empty:
            valid_mask = ~lbl_slice.isna()
            feat_slice = feat_slice.loc[valid_mask]
            lbl_slice = lbl_slice.loc[valid_mask]
        
        return feat_slice, lbl_slice
    
    # 开始滚动训练
    metrics = []
    min_samples = rolling_cfg.get("min_samples", 100)
    
    for idx, window in enumerate(generate_windows()):
        logger.info("=" * 80)
        logger.info("滚动窗口 %d: 训练 [%s, %s] 验证 [%s, %s]",
                   idx, window.train_start, window.train_end, window.valid_start, window.valid_end)
        
        # 切片训练集和验证集
        train_feat, train_lbl = slice_data(features, labels, window.train_start, window.train_end, is_validation=False)
        valid_feat, valid_lbl = slice_data(features, labels, window.valid_start, window.valid_end, is_validation=True)
        
        # 记录实际日期范围
        if not train_feat.empty:
            tmin = train_feat.index.get_level_values("datetime").min()
            tmax = train_feat.index.get_level_values("datetime").max()
            logger.info("窗口 %d 实际训练集日期范围: %s ~ %s（样本=%d）", idx, tmin, tmax, len(train_feat))
        if valid_feat is not None and not valid_feat.empty:
            vmin = valid_feat.index.get_level_values("datetime").min()
            vmax = valid_feat.index.get_level_values("datetime").max()
            logger.info("窗口 %d 实际验证集日期范围: %s ~ %s（样本=%d）", idx, vmin, vmax, len(valid_feat))
        
        # 检查训练样本数
        if len(train_feat) < min_samples:
            logger.warning("训练样本不足 (%d < %d)，跳过该窗口", len(train_feat), min_samples)
            continue
        
        has_valid = valid_feat is not None and not valid_feat.empty and valid_lbl is not None and not valid_lbl.empty
        if not has_valid:
            logger.warning("窗口 %d 验证集为空，退化为仅训练", idx)
            valid_feat = None
            valid_lbl = None
        else:
            logger.info("窗口 %d: 训练样本 %d，验证样本 %d", idx, len(train_feat), len(valid_feat))
        
        # 对每个训练窗口单独计算归一化参数，避免数据泄露
        logger.info("窗口 %d: 计算训练窗口归一化参数（仅使用训练集数据）", idx)
        train_feat_norm, norm_mean, norm_std = QlibFeaturePipeline.normalize_features(train_feat)
        
        # 验证集使用训练集的归一化参数
        if has_valid:
            valid_feat_norm = (valid_feat - norm_mean) / norm_std
            valid_feat_norm = valid_feat_norm.clip(-5, 5)
        else:
            valid_feat_norm = None
        
        # 训练模型
        logger.info("窗口 %d: 开始训练 IndustryGRU 模型...", idx)
        try:
            model.fit(train_feat_norm, train_lbl, valid_feat_norm, valid_lbl)
        except Exception as e:
            logger.error("窗口 %d 训练失败: %s", idx, str(e), exc_info=True)
            continue
        
        # 评估模型
        logger.info("窗口 %d: 评估模型性能...", idx)
        
        # 修复：训练集 IC 不应该使用训练数据本身进行预测（会导致 IC 虚高）
        # 使用时间序列交叉验证：使用训练集的前 80% 训练，后 20% 评估
        train_ic = float("nan")
        if compute_train_ic and train_ic_method == "ts_cv":
            try:
                if len(train_feat_norm) > 100:  # 只有样本数足够时才计算训练集 IC
                    # 将训练集分为两部分：前 80% 用于训练，后 20% 用于评估
                    split_idx = int(len(train_feat_norm) * 0.8)
                    train_feat_train = train_feat_norm.iloc[:split_idx]
                    train_lbl_train = train_lbl.iloc[:split_idx]
                    train_feat_eval = train_feat_norm.iloc[split_idx:]
                    train_lbl_eval = train_lbl.iloc[split_idx:]
                    
                    # 使用前 80% 的数据重新训练一个临时模型用于评估
                    # 注意：这里不保存模型，只用于评估
                    logger.info("窗口 %d: 使用时间序列交叉验证计算训练集 IC（前80%%训练，后20%%评估）", idx)
                    temp_model = IndustryGRUWrapper(model_config)
                    temp_model.fit(train_feat_train, train_lbl_train, None, None)
                    
                    # 使用训练集的历史数据补充评估集
                    train_pred_eval = temp_model.predict(train_feat_eval, history_feat=train_feat_train)
                    train_ic = _rank_ic(train_pred_eval, train_lbl_eval)
                    logger.info("窗口 %d: 训练集 IC（时间序列交叉验证）=%.4f", idx, train_ic)
                else:
                    logger.warning("窗口 %d: 训练样本数不足（%d < 100），跳过训练集 IC 计算", idx, len(train_feat_norm))
            except Exception as e:
                logger.error("窗口 %d: 训练集预测失败: %s", idx, e)
                train_ic = float("nan")
        elif not compute_train_ic:
            logger.info("窗口 %d: 训练集 IC 计算已禁用（避免使用训练数据本身进行预测导致 IC 虚高）", idx)
        
        metric = {
            "window": idx,
            "train_start": window.train_start,
            "train_end": window.train_end,
            "valid_start": window.valid_start,
            "valid_end": window.valid_end,
            "train_samples": len(train_feat),
            "valid_samples": len(valid_feat) if has_valid else 0,
            "train_ic": float(train_ic),
            "valid_ic": float("nan"),
        }
        
        if has_valid:
            try:
                # 对于验证集，使用训练集数据作为历史数据补充
                logger.info("窗口 %d: 开始验证集预测（可能需要一些时间构建序列）...", idx)
                import time
                predict_start = time.time()
                valid_pred = model.predict(valid_feat_norm, history_feat=train_feat_norm)
                predict_time = time.time() - predict_start
                logger.info("窗口 %d: 验证集预测完成，耗时 %.2f 秒", idx, predict_time)
                
                valid_ic = _rank_ic(valid_pred, valid_lbl)
                metric["valid_ic"] = float(valid_ic)
                logger.info("窗口 %d: 训练集 IC=%.4f, 验证集 IC=%.4f", idx, train_ic, valid_ic)
                
                # 可选：快速诊断（如果启用）
                enable_diagnosis = cfg.get("diagnosis", {}).get("enabled", False)
                if enable_diagnosis:
                    try:
                        from classify.utils.diagnosis_utils import quick_diagnosis
                        logger.info("窗口 %d: 运行快速诊断...", idx)
                        diagnosis_results = quick_diagnosis(
                            train_label=train_lbl,
                            valid_label=valid_lbl,
                            valid_pred=valid_pred,
                            train_feat=train_feat,
                            valid_feat=valid_feat,
                        )
                        # 保存诊断结果到 metric
                        metric["diagnosis"] = diagnosis_results
                    except Exception as e:
                        logger.warning("窗口 %d: 快速诊断失败: %s", idx, e)
            except Exception as e:
                logger.error("窗口 %d: 验证集预测失败: %s", idx, e)
                logger.info("窗口 %d: 训练集 IC=%.4f", idx, train_ic)
        else:
            logger.info("窗口 %d: 训练集 IC=%.4f", idx, train_ic)
        
        metrics.append(metric)
        
        # 保存模型
        tag = window.valid_end.replace("-", "")
        model.save(model_dir, tag)
        logger.info("窗口 %d: 模型已保存 (tag=%s)", idx, tag)
        
        # 保存归一化参数
        norm_meta_path = os.path.join(model_dir, f"{tag}_norm_meta.json")
        norm_meta = {
            "feature_mean": norm_mean.to_dict(),
            "feature_std": norm_std.to_dict(),
            "train_start": window.train_start,
            "train_end": window.train_end,
            "valid_end": window.valid_end,
        }
        with open(norm_meta_path, "w", encoding="utf-8") as fp:
            json.dump(norm_meta, fp, ensure_ascii=False, indent=2, default=str)
        logger.info("窗口 %d: 归一化参数已保存", idx)
    
    # 保存训练指标
    if metrics:
        df = pd.DataFrame(metrics)
        metrics_path = os.path.join(log_dir, "training_metrics.csv")
        df.to_csv(metrics_path, index=False)
        logger.info("训练指标已保存: %s，共 %d 条记录", metrics_path, len(df))
        
        # 打印汇总统计
        logger.info("=" * 80)
        logger.info("训练汇总统计:")
        logger.info("  - 总窗口数: %d", len(metrics))
        if df["valid_ic"].notna().any():
            logger.info("  - 平均验证集 IC: %.4f", df["valid_ic"].mean())
            logger.info("  - 验证集 IC 标准差: %.4f", df["valid_ic"].std())
        else:
            logger.info("  - 平均验证集 IC: N/A（无有效数据）")
            logger.info("  - 验证集 IC 标准差: N/A（无有效数据）")
        if df["train_ic"].notna().any():
            logger.info("  - 平均训练集 IC: %.4f", df["train_ic"].mean())
            logger.info("  - 训练集 IC 标准差: %.4f", df["train_ic"].std())
        else:
            logger.info("  - 平均训练集 IC: N/A（训练集 IC 计算已禁用，见 train_ic.enabled 配置）")
            logger.info("  - 训练集 IC 标准差: N/A（训练集 IC 计算已禁用）")
    else:
        logger.warning("未产出任何训练窗口指标")
    
    logger.info("=" * 80)
    logger.info("训练流程完成")


if __name__ == "__main__":
    main()





