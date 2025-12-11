"""
预测模块：负责加载已训练模型、计算单模型预测并执行 IC 动态加权。
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Tuple

import pandas as pd

from models.ensemble_manager import EnsembleModelManager
from models.stack_model import LeafStackModel
from predictor.weight_dynamic import RankICDynamicWeighter
from utils import load_yaml_config

logger = logging.getLogger(__name__)


class PredictorEngine:
    """预测管线封装。"""

    def __init__(self, pipeline_config: str):
        cfg = load_yaml_config(pipeline_config)
        self.cfg = cfg
        self.paths = cfg["paths"]
        self.ensemble = EnsembleModelManager(cfg, cfg.get("ensemble"))
        self.stack = LeafStackModel(cfg["stack_config"])
        ic_cfg = cfg.get("ic_logging", {})
        self.weighter = RankICDynamicWeighter(
            window=ic_cfg.get("window", 60),
            half_life=ic_cfg.get("half_life", 20),
            min_weight=ic_cfg.get("min_weight", 0.05),
            max_weight=ic_cfg.get("max_weight", 0.7),
            clip_negative=ic_cfg.get("clip_negative", True),
        )
        # 加载数据配置以检查 label_transform
        data_config_path = cfg.get("data_config", "config/data.yaml")
        if isinstance(data_config_path, str):
            self.data_cfg = load_yaml_config(data_config_path)
        else:
            self.data_cfg = data_config_path
        # 归一化参数（在 load_models 时加载）
        self._norm_mean = None
        self._norm_std = None

    def load_models(self, tag: str):
        model_dir = self.paths["model_dir"]
        logger.info("加载模型，标识: %s", tag)
        self.ensemble.load(model_dir, tag)
        self.stack.load(model_dir, tag)
        
        # 加载归一化参数
        import json
        norm_meta_path = os.path.join(model_dir, f"{tag}_norm_meta.json")
        if os.path.exists(norm_meta_path):
            with open(norm_meta_path, "r", encoding="utf-8") as fp:
                norm_meta = json.load(fp)
            self._norm_mean = pd.Series(norm_meta["feature_mean"])
            self._norm_std = pd.Series(norm_meta["feature_std"])
            logger.info("归一化参数已加载: 训练窗口 [%s, %s]", 
                       norm_meta.get("train_start"), norm_meta.get("train_end"))
        else:
            logger.warning("未找到归一化参数文件: %s，将使用特征本身的统计量（不推荐）", norm_meta_path)
            self._norm_mean = None
            self._norm_std = None

    def predict(
        self,
        features: pd.DataFrame,
        ic_histories: Dict[str, pd.Series],
    ) -> Tuple[pd.Series, Dict[str, pd.Series], Dict[str, float]]:
        """返回融合预测、各模型预测以及权重。"""
        # 修复：使用训练时的归一化参数对特征进行归一化
        if self._norm_mean is not None and self._norm_std is not None:
            logger.info("使用训练时的归一化参数对特征进行归一化")
            # 确保特征列顺序与归一化参数一致
            feature_cols = [col for col in features.columns if col in self._norm_mean.index]
            missing_cols = set(features.columns) - set(feature_cols)
            if missing_cols:
                logger.warning("部分特征在归一化参数中不存在，将用0填充: %s", list(missing_cols)[:10])
            
            # 对存在的特征进行归一化
            if feature_cols:
                features_norm = (features[feature_cols] - self._norm_mean[feature_cols]) / self._norm_std[feature_cols]
                features_norm = features_norm.clip(-5, 5)
            else:
                logger.error("没有匹配的特征列，无法进行归一化")
                features_norm = features.copy()
            
            # 填充缺失的特征（用0填充，因为已经归一化）
            for col in missing_cols:
                features_norm[col] = 0.0
            
            # 确保列顺序与原始特征一致
            features_norm = features_norm[features.columns]
            features = features_norm
        else:
            logger.warning("未加载归一化参数，使用原始特征（可能导致预测不准确）")
        
        blend_pred, base_preds, aux = self.ensemble.predict(features)
        lgb_pred = base_preds.get("lgb")
        lgb_leaf = aux.get("lgb")
        if lgb_pred is None or lgb_leaf is None:
            raise RuntimeError("需要 LightGBM 预测以驱动 Stack 模型，请检查 ensemble 配置")
        residual_pred = self.stack.predict_residual(lgb_leaf, features.index)
        stack_pred = self.stack.fuse(lgb_pred, residual_pred)
        preds = dict(base_preds)
        preds["stack"] = stack_pred
        if blend_pred is not None:
            preds["qlib_ensemble"] = blend_pred
        # 根据历史 IC 计算动态权重，兼顾稳定性
        weights = self.weighter.get_weights(ic_histories)
        final_pred = self.weighter.blend(preds, weights)
        
        # 如果训练时使用了 Rank 转换，对预测值也进行截面排名转换
        label_transform = self.data_cfg.get("data", {}).get("label_transform", {})
        if label_transform.get("enabled", False):
            from utils.label_transform import transform_to_rank
            method = label_transform.get("method", "percentile")
            groupby = label_transform.get("groupby", "datetime")
            logger.info("训练使用了 Rank 转换，对预测值进行截面排名转换（方法: %s, 分组: %s）", method, groupby)
            
            # 诊断：检查原始预测值的分布
            unique_before = final_pred.nunique()
            total_before = len(final_pred)
            logger.info("Rank 转换前诊断：唯一值 %d / %d (%.2f%%)", 
                       unique_before, total_before, unique_before/total_before*100)
            logger.info("Rank 转换前统计：最小值=%.6f, 最大值=%.6f, 均值=%.6f, 标准差=%.6f",
                       final_pred.min(), final_pred.max(), final_pred.mean(), final_pred.std())
            
            # 按日期检查原始预测值的唯一性（只检查前5个日期作为示例）
            if isinstance(final_pred.index, pd.MultiIndex):
                date_groups = final_pred.groupby(level="datetime")
                sample_dates = list(date_groups)[:5]
                for date, group in sample_dates:
                    logger.info("  日期 %s: 唯一值 %d / %d (%.2f%%), 范围 [%.6f, %.6f]",
                               date, group.nunique(), len(group), 
                               group.nunique()/len(group)*100 if len(group) > 0 else 0,
                               group.min(), group.max())
            
            # 对最终预测值进行截面排名转换
            final_pred = transform_to_rank(final_pred, method=method, groupby=groupby)
            
            # 诊断：检查转换后的分布
            unique_after = final_pred.nunique()
            total_after = len(final_pred)
            logger.info("Rank 转换后诊断：唯一值 %d / %d (%.2f%%)", 
                       unique_after, total_after, unique_after/total_after*100)
            
            # 检查是否有大量重复值
            if unique_after < total_after * 0.1:
                logger.warning("Rank 转换后唯一值过少（%.2f%%），可能存在以下问题：", unique_after/total_after*100)
                logger.warning("  1. 原始预测值本身就有很多重复（模型预测过于保守）")
                logger.warning("  2. Rank 转换导致不同日期之间的排名百分位相同（这是正常现象）")
                logger.warning("  3. 建议检查原始预测值的分布，确认模型预测质量")
            
            # 对各个模型的预测值也进行转换（可选，用于一致性）
            for name in preds:
                preds[name] = transform_to_rank(preds[name], method=method, groupby=groupby)
            logger.info("预测值已转换为排名百分位，范围应在 [0, 1] 之间")
        
        return final_pred, preds, weights

    def save_predictions(
        self,
        final_pred: pd.Series,
        preds: Dict[str, pd.Series],
        tag: str,
        extra_cols: Dict[str, pd.Series] = None,
    ):
        os.makedirs(self.paths["prediction_dir"], exist_ok=True)
        out_path = os.path.join(self.paths["prediction_dir"], f"pred_{tag}.csv")
        df = pd.DataFrame({"final": final_pred})
        for name, series in preds.items():
            df[name] = series
        # 追加附加列（如波动率等衍生指标），索引需与预测值对齐
        if extra_cols:
            for name, series in extra_cols.items():
                if series is None:
                    continue
                try:
                    df[name] = series.reindex(df.index)
                except Exception as e:
                    logger.warning("附加列 %s 对齐失败: %s", name, e)
        # 统一索引顺序为 (datetime, instrument)，便于后续回测与查看
        if df.index.nlevels == 2:
            try:
                df = df.reorder_levels(["datetime", "instrument"]).sort_index()
            except KeyError:
                # 如果 MultiIndex 未命名，则按照位置强制调换
                df = df.reorder_levels([1, 0]).sort_index()
        # MultiIndex 直接写入 csv，便于后续回测按日期/证券读取
        df.to_csv(out_path, index_label=["datetime", "instrument"])
        logger.info("预测结果已保存: %s", out_path)

