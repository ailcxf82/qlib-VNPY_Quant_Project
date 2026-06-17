"""
预测模块：负责加载已训练模型、计算单模型预测并执行 IC 动态加权。
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Tuple

import pandas as pd

from models.ensemble_manager import EnsembleModelManager
from models.meta_stacker import MetaStacker
from predictor.weight_dynamic import RankICDynamicWeighter
from utils import load_yaml_config
import json
import numpy as np
from utils.meta_ridge import predict_meta_ridge
from utils.normalize import normalize_by_date

logger = logging.getLogger(__name__)


class PredictorEngine:
    """预测管线封装。"""

    def __init__(self, pipeline_config: str):
        cfg = load_yaml_config(pipeline_config)
        self.cfg = cfg
        self.paths = cfg["paths"]
        self.ensemble = EnsembleModelManager(cfg, cfg.get("ensemble"))
        # stack（LeafStackModel）可选：其内部依赖 torch/MLP，用于学习 LGB leaf 的残差
        stack_cfg = (cfg.get("stack") or {})
        self.stack_enabled: bool = bool(stack_cfg.get("enabled", True))
        self.stack = None
        if self.stack_enabled:
            try:
                from models.stack_model import LeafStackModel  # 延迟导入：避免无 torch 时阻断纯树模型预测
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "启用 stack 需要 torch（LeafStackModel 依赖 MLP/torch）。"
                    "若你只使用基础模型，请在 pipeline.yaml 中设置 stack.enabled=false。"
                ) from e
            self.stack = LeafStackModel(cfg["stack_config"])
        ic_cfg = cfg.get("ic_logging", {})
        self.weighter = RankICDynamicWeighter(
            window=ic_cfg.get("window", 60),
            half_life=ic_cfg.get("half_life", 20),
            min_weight=ic_cfg.get("min_weight", 0.05),
            max_weight=ic_cfg.get("max_weight", 0.5),  # 默认降低到 0.5，防止单一模型权重过大
            clip_negative=ic_cfg.get("clip_negative", True),
            use_softmax=ic_cfg.get("use_softmax", False),  # 可选：使用 softmax 平滑权重
            temperature=ic_cfg.get("temperature", 1.0),  # softmax 温度参数
        )
        # 加载数据配置以检查 label_transform
        data_config_path = cfg.get("data_config", "config/data.yaml")
        if isinstance(data_config_path, str):
            self.data_cfg = load_yaml_config(data_config_path)
        else:
            self.data_cfg = data_config_path
        # 归一化参数（在 load_models 时加载）
        self._norm_meta = None
        self._meta: MetaStacker | None = None
        enabled = (cfg.get("oof_stacking") or {}).get("enabled", False)
        if isinstance(enabled, str) and enabled.strip().lower() == "auto":
            base = [str(x).strip().lower() for x in (cfg.get("base_models") or [])]
            enabled = ("gru" in base)
        self._use_meta_stacking: bool = bool(enabled)
        self._meta_dir = cfg["paths"].get("meta_dir", "data/meta")

    def load_models(self, tag: str):
        model_dir = self.paths["model_dir"]
        logger.info("加载模型，标识: %s", tag)
        self.ensemble.load(model_dir, tag)
        if self.stack_enabled and self.stack is not None:
            self.stack.load(model_dir, tag)

        # 可选：加载 MetaStacker（若启用）
        self._meta = None
        if self._use_meta_stacking:
            try:
                meta = MetaStacker((self.cfg.get("oof_stacking") or {}).get("meta_model", {}) or {})
                meta.load(model_dir, tag)
                self._meta = meta
                logger.info("MetaStacker 已加载: %s", tag)
            except FileNotFoundError:
                logger.warning("启用了 oof_stacking 但未找到 meta 模型文件（%s），将回退到原 final 逻辑", tag)
                self._meta = None
        
        # 加载归一化参数
        norm_meta_path = os.path.join(model_dir, f"{tag}_norm_meta.json")
        if os.path.exists(norm_meta_path):
            with open(norm_meta_path, "r", encoding="utf-8") as fp:
                norm_meta = json.load(fp)
            self._norm_meta = norm_meta
            # 归一化由 EnsembleModelManager 负责（支持 per-model 与旧 global）
            self.ensemble.set_norm_meta(norm_meta)
            logger.info(
                "归一化参数已加载: mode=%s, 训练窗口 [%s, %s]",
                norm_meta.get("mode", "unknown"),
                norm_meta.get("train_start"),
                norm_meta.get("train_end"),
            )
        else:
            logger.warning("未找到归一化参数文件: %s，将使用特征本身的统计量（不推荐）", norm_meta_path)
            self._norm_meta = None

    def predict(
        self,
        features: pd.DataFrame,
        ic_histories: Dict[str, pd.Series],
        history_feat: pd.DataFrame = None,
    ) -> Tuple[pd.Series, Dict[str, pd.Series], Dict[str, float]]:
        """返回融合预测、各模型预测以及权重。
        
        参数:
            features: 预测特征（目标日期范围）
            ic_histories: 各模型的IC历史，用于动态加权
            history_feat: 历史特征（用于GRU等序列模型构造序列）
        """
        # 空特征短路：常见于某日期股票池为空、或因缺因子/清洗过滤导致切片无样本
        if features is None or len(features) == 0:
            logger.warning("predict 输入特征为空，将返回空预测结果（跳过模型推理与加权）")
            empty_idx = getattr(features, "index", pd.MultiIndex.from_arrays([[], []], names=["datetime", "instrument"]))
            empty_final = pd.Series([], index=empty_idx, dtype=float, name="final")
            return empty_final, {}, {}

        # 归一化由 EnsembleModelManager 内部处理（按模型 per-model 或兼容旧 global）
        # 传递 history_feat 给 GRU 等序列模型
        blend_pred, base_preds, aux = self.ensemble.predict(features, history_feat=history_feat)
        preds = dict(base_preds)
        if self.stack_enabled:
            lgb_pred = base_preds.get("lgb")
            lgb_leaf = aux.get("lgb")
            if lgb_pred is None or lgb_leaf is None:
                raise RuntimeError("启用 stack 需要 LightGBM 输出 leaf，请检查 base_models/ensemble.models 是否包含 lgb")
            if self.stack is None:
                raise RuntimeError("stack_enabled=True 但 stack 未初始化")
            residual_pred = self.stack.predict_residual(lgb_leaf, features.index)
            stack_pred = self.stack.fuse(lgb_pred, residual_pred)
            preds["stack"] = stack_pred
        if blend_pred is not None:
            preds["qlib_ensemble"] = blend_pred

        # 1) 若配置 meta_model=ridge 且文件存在，则使用 Ridge 参数融合（方案A）
        meta_cfg = (self.cfg.get("oof_stacking") or {}).get("meta_model") or {}
        if isinstance(meta_cfg, str):
            meta_cfg = {"model_type": meta_cfg}
        meta_type = str(meta_cfg.get("model_type", "")).lower()
        ridge_path = os.path.join(self._meta_dir, f"{self.cfg['paths'].get('tag','')}_meta_ridge.json")
        use_ridge = meta_type == "ridge" and os.path.exists(ridge_path)
        if use_ridge:
            try:
                # 构造 DataFrame 以 (date, code) join
                df = pd.DataFrame({"pred_lgb": base_preds.get("lgb"), "pred_gru": base_preds.get("gru")})
                if not isinstance(df.index, pd.MultiIndex) or "datetime" not in df.index.names or "instrument" not in df.index.names:
                    raise ValueError("Ridge meta 需要预测索引包含 datetime/instrument")
                df = df.reset_index().rename(columns={"datetime": "date", "instrument": "code"})
                df = normalize_by_date(
                    df,
                    cols=["pred_lgb", "pred_gru"],
                    date_col="date",
                    mode=meta_cfg.get("normalize_mode", "zscore"),
                    eps=float(meta_cfg.get("normalize_eps", 1e-6)),
                )
                coef = np.array(json.load(open(ridge_path, "r", encoding="utf-8"))["coef"], dtype=float)
                intercept = float(json.load(open(ridge_path, "r", encoding="utf-8"))["intercept"])
                X = df[["pred_lgb", "pred_gru"]].values
                df["final_score"] = X.dot(coef) + intercept
                final_pred = df.set_index(["date", "code"])["final_score"]
                # 对齐回原索引
                final_pred.index = final_pred.index.set_names(["datetime", "instrument"])
                final_pred = final_pred.reindex(base_preds["lgb"].index)
                weights = {}
            except Exception as e:
                logger.error("Ridge meta 预测失败，回退到动态加权: %s", e, exc_info=True)
                weights = self.weighter.get_weights(ic_histories)
                final_pred = self.weighter.blend(preds, weights)
        elif self._meta is not None:
            # 2) 若 meta stacking 启用且已加载：final 走同一元模型
            X_meta = pd.DataFrame({k: v for k, v in base_preds.items()})
            meta_pred = self._meta.predict(X_meta)
            final_pred = meta_pred.rename("final")
            weights = {}
        else:
            # 3) 否则沿用原 IC 动态加权 final
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

    def save_predictions(self, final_pred: pd.Series, preds: Dict[str, pd.Series], tag: str):
        def _shift_to_next_trading_day(df: pd.DataFrame) -> pd.DataFrame:
            """
            将信号的 datetime 整体后移到下一个交易日，用于避免“用当日收盘价/最高价等信息做当日交易”的隐性未来信息。
            规则：
            - 优先使用 qlib 交易日历（若可用）
            - 否则退化为工作日（Mon-Fri）
            - 默认允许 **最多延伸 1 个交易日**（即允许输出“下一交易日”的买入建议），
              超过该范围的样本会被丢弃。
            """
            if not isinstance(df.index, pd.MultiIndex) or "datetime" not in df.index.names:
                return df

            # 空数据短路：避免对空索引求 min/max
            if len(df) == 0:
                return df

            orig_max_dt = pd.Timestamp(df.index.get_level_values("datetime").max()).normalize()
            orig_min_dt = pd.Timestamp(df.index.get_level_values("datetime").min()).normalize()
            orig_rows = len(df)

            dt_vals = pd.to_datetime(df.index.get_level_values("datetime")).normalize()
            unique_dts = pd.Index(dt_vals.unique()).sort_values()
            if unique_dts.empty:
                return df

            # 构造交易日历
            cal = None
            start = pd.Timestamp(unique_dts.min()).normalize()
            end = pd.Timestamp(unique_dts.max()).normalize() + pd.Timedelta(days=30)
            try:
                from qlib.data import D  # type: ignore
                # 兼容不同版本 qlib 的接口，尽量取到交易日序列
                cal = D.calendar(start_time=start, end_time=end, freq="day")
                cal = pd.to_datetime(list(cal)).normalize()
            except Exception:
                cal = pd.bdate_range(start=start, end=end).normalize()

            cal = pd.Index(cal).sort_values().unique()
            if cal.empty:
                return df

            # 映射：dt -> 下一个交易日（严格右侧）
            import bisect
            cal_list = list(cal)
            mapping = {}
            for d in unique_dts:
                i = bisect.bisect_right(cal_list, pd.Timestamp(d))
                if i >= len(cal_list):
                    # 兜底：直接加一天（后面会被 orig_max_dt 过滤掉）
                    mapping[pd.Timestamp(d)] = pd.Timestamp(d) + pd.Timedelta(days=1)
                else:
                    mapping[pd.Timestamp(d)] = pd.Timestamp(cal_list[i])

            # 生成新索引
            df2 = df.copy()
            dt_new = dt_vals.map(lambda x: mapping.get(pd.Timestamp(x), pd.Timestamp(x)))
            df2 = df2.reset_index()
            df2["datetime"] = pd.to_datetime(dt_new).normalize()
            df2 = df2.set_index(df.index.names)
            df2 = df2.sort_index()

            # 丢弃越界（仅允许最多延伸 1 个交易日：即 orig_max_dt 的下一交易日）
            # 这能满足“用最新数据给下一交易日建议”，同时避免输出无限外推的未来日期。
            max_allowed_dt = mapping.get(pd.Timestamp(orig_max_dt))
            if max_allowed_dt is None:
                max_allowed_dt = orig_max_dt
            df2 = df2.loc[df2.index.get_level_values("datetime") <= pd.Timestamp(max_allowed_dt).normalize()]
            new_min_dt = pd.Timestamp(df2.index.get_level_values("datetime").min()).normalize() if len(df2) > 0 else None
            new_max_dt = pd.Timestamp(df2.index.get_level_values("datetime").max()).normalize() if len(df2) > 0 else None
            dropped = orig_rows - len(df2)
            logger.info(
                "预测日期对齐(next trading day): %s~%s -> %s~%s，丢弃越界样本=%d",
                orig_min_dt,
                orig_max_dt,
                new_min_dt,
                new_max_dt,
                dropped,
            )
            return df2

        os.makedirs(self.paths["prediction_dir"], exist_ok=True)
        out_path = os.path.join(self.paths["prediction_dir"], f"pred_{tag}.csv")
        df = pd.DataFrame({"final": final_pred})
        for name, series in preds.items():
            df[name] = series
        # 统一索引顺序为 (datetime, instrument)，便于后续回测与查看
        if df.index.nlevels == 2:
            try:
                df = df.reorder_levels(["datetime", "instrument"]).sort_index()
            except KeyError:
                # 如果 MultiIndex 未命名，则按照位置强制调换
                df = df.reorder_levels([1, 0]).sort_index()

        # 关键：信号日期对齐到下一个交易日（可通过环境变量关闭）
        # 修复：由于特征已经使用 T-1 日数据（Ref($xxx, 1)），预测文件的日期应该直接使用 signal_date
        # 不再偏移到下一个交易日，避免数据泄露
        shift_flag = str(os.environ.get("SHIFT_PRED_TO_NEXT_DAY", "0")).strip().lower()
        # 写入元信息，供回测端判断是否需要反向对齐（避免回测侧出现隐性 T+2）
        df["_meta_shifted_next_day"] = 0
        if shift_flag not in {"0", "false", "no"}:
            df = _shift_to_next_trading_day(df)
            df["_meta_shifted_next_day"] = 1
            logger.warning("预测日期已偏移到下一个交易日，这可能导致数据泄露！建议设置 SHIFT_PRED_TO_NEXT_DAY=0")

        # MultiIndex 直接写入 csv，便于后续回测按日期/证券读取
        df.to_csv(out_path, index_label=["datetime", "instrument"])
        logger.info("预测结果已保存: %s", out_path)

