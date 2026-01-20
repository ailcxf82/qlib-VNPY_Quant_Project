"""
滚动训练器：串联特征、模型，输出多模型权重与指标。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from feature.qlib_feature_pipeline import QlibFeaturePipeline
from models.ensemble_manager import EnsembleModelManager
from models.meta_stacker import MetaStacker
from trainer.oof_manager import OOFManager, TimeSeriesFoldSplitter
from utils import load_yaml_config
from utils.meta_oof_builder import build_meta_oof
from utils.meta_ridge import train_meta_ridge

logger = logging.getLogger(__name__)


@dataclass
class Window:
    train_start: str
    train_end: str
    valid_start: str
    valid_end: str


def _rank_ic(pred: pd.Series, label: pd.Series) -> float:
    pred, label = pred.align(label, join="inner")
    if pred.empty:
        return float("nan")
    return pred.rank().corr(label, method="spearman")


class RollingTrainer:
    """核心训练流程。"""

    def __init__(self, pipeline_config: str):
        self.cfg = load_yaml_config(pipeline_config)
        self.paths = self.cfg["paths"]
        self.data_cfg_path = self.cfg["data_config"]

        # 预加载 data.yaml（后续窗口生成/标签解析/动态时间窗口都会用到）
        self.data_cfg = load_yaml_config(self.data_cfg_path)
        self.data_section = self.data_cfg.get("data", {}) or {}
        # aligned 模式下：data.start_time 作为“验证起点锚点”，但取数 start_time 需要更早（包含训练历史）
        self.anchor_start_time = self.data_section.get("start_time")

        # ===== 动态训练窗口对齐（按模型 train_days，但窗口的 valid_start 对齐）=====
        # 需求：不同模型（lgb/gru）可能使用不同 train_days，但验证窗口必须从同一个日期开始，
        # 否则各模型产出的 tag（valid_end）会错位，无法做 ensemble/stack/OOF 对齐。
        rolling = self.cfg.get("rolling", {}) or {}
        model_train_days = rolling.get("model_train_days") or rolling.get("train_days_by_model") or {}
        if isinstance(model_train_days, dict) and model_train_days:
            # 若配置了按模型天数，默认启用 aligned 窗口模式（以 data.start_time 作为 valid_start 锚点）
            rolling.setdefault("window_mode", "aligned")
            self.cfg["rolling"] = rolling

        window_mode = str(rolling.get("window_mode", "classic")).strip().lower()
        if window_mode == "aligned":
            if not self.anchor_start_time:
                raise ValueError("aligned 窗口模式需要 data.start_time 作为验证起点锚点")

            # 训练窗口长度取：rolling.train_days 与 model_train_days 的最大值
            max_days = int(rolling.get("train_days", 0) or 0)
            if isinstance(model_train_days, dict):
                for _, v in model_train_days.items():
                    try:
                        max_days = max(max_days, int(v))
                    except Exception:
                        continue
            if max_days <= 0:
                raise ValueError(
                    "aligned 窗口模式需要 rolling.train_days 或 rolling.model_train_days 提供正整数训练天数"
                )

            # 为保证最长训练窗口可用，把 D.features 的 start_time 向前扩展 max_days
            fetch_start = (pd.Timestamp(self.anchor_start_time) - pd.Timedelta(days=max_days)).strftime("%Y-%m-%d")
            if str(fetch_start) != str(self.data_section.get("start_time")):
                logger.info(
                    "aligned 窗口模式：将 data.start_time 从 %s 向前扩展为 %s（保证最长训练窗口可用）",
                    self.data_section.get("start_time"),
                    fetch_start,
                )
                self.data_section["start_time"] = fetch_start
                self.data_cfg["data"] = self.data_section

        # 用（可能被扩展过的）data_cfg 初始化特征管线
        self.pipeline = QlibFeaturePipeline(self.data_cfg)
        self.ensemble = EnsembleModelManager(self.cfg, self.cfg.get("ensemble"))
        # 诊断：打印特征集合配置，便于核对模型是否按集合取特征
        try:
            feature_sets = self.data_cfg.get("data", {}).get("feature_sets", {}) or {}
            active_sets = self.data_cfg.get("data", {}).get("active_feature_sets", []) or []
            model_features = self.cfg.get("model_features", {}) or {}
            logger.info("配置检查: active_feature_sets=%s", active_sets)
            logger.info("配置检查: feature_sets_keys=%s", list(feature_sets.keys()))
            logger.info("配置检查: model_features=%s", model_features)
        except Exception as e:
            logger.warning("配置检查失败（可忽略）: %s", e)
        stack_cfg = (self.cfg.get("stack") or {})
        self.stack_enabled = bool(stack_cfg.get("enabled", True))
        if self.stack_enabled:
            try:
                from models.stack_model import LeafStackModel  # 延迟导入：避免无 torch 时阻断纯树模型流程
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "启用 stack 需要 torch（LeafStackModel 依赖 MLP/torch）。"
                    "若你只训练基础模型，请在 pipeline.yaml 中设置 stack.enabled=false。"
                ) from e
            self.stack = LeafStackModel(self.cfg["stack_config"])
        else:
            self.stack = None
        if not self.stack_enabled:
            logger.info("已关闭 LeafStackModel（stack.enabled=false）：本次训练仅训练/评估基础模型，不训练 stack。")
        
        # 解析标签表达式，获取需要的未来天数
        label_expr = self.data_section.get("label", "Ref($close, -5)/$close - 1")
        import re
        self.label_future_days = 0
        if "Ref($close, -" in label_expr:
            match = re.search(r'Ref\(\$close,\s*-(\d+)\)', label_expr)
            if match:
                self.label_future_days = int(match.group(1))
                logger.info(
                    "标签需要未来 %d 天数据来计算，训练/验证切片的结束日期将自动提前 %d 天（避免跨窗口使用未来价格形成标签）",
                    self.label_future_days,
                    self.label_future_days,
                )

    def _generate_windows(self) -> Iterable[Window]:
        rolling = self.cfg["rolling"]
        data_cfg = self.data_section
        start = pd.Timestamp(data_cfg["start_time"])
        end = pd.Timestamp(data_cfg["end_time"])
        window_mode = str(rolling.get("window_mode", "classic")).strip().lower()

        # 训练窗口长度取：rolling.train_days 与 model_train_days 的最大值（用于窗口生成 train_start）
        model_train_days = rolling.get("model_train_days") or rolling.get("train_days_by_model") or {}
        max_train_days = int(rolling.get("train_days", 0) or 0)
        if isinstance(model_train_days, dict):
            for _, v in model_train_days.items():
                try:
                    max_train_days = max(max_train_days, int(v))
                except Exception:
                    continue
        # 支持按日训练：优先使用 train_days/valid_days/step_days，如果没有则回退到按月（兼容旧配置）
        if "train_days" in rolling:
            train_offset = pd.Timedelta(days=max_train_days)
            valid_offset = pd.Timedelta(days=rolling["valid_days"])
            step = pd.Timedelta(days=rolling["step_days"])
        else:
            # 兼容旧配置：按月训练
            train_offset = pd.DateOffset(months=rolling["train_months"])
            valid_offset = pd.DateOffset(months=rolling["valid_months"])
            step = pd.DateOffset(months=rolling["step_months"])

        # classic：start_time 是训练起点（原逻辑）
        # aligned：start_time 是验证起点锚点（新逻辑，用于对齐多模型）
        if window_mode == "aligned":
            # 注意：start 可能已被向前扩展用于取数；aligned 的窗口锚点必须使用原始 start_time
            cursor = pd.Timestamp(self.anchor_start_time)
        else:
            # cursor 指向验证起点，前推 train_offset 即训练区间
            cursor = start + train_offset

        while cursor + valid_offset <= end:
            train_start = cursor - train_offset
            train_end = cursor - pd.Timedelta(days=1)
            valid_start = cursor
            valid_end = cursor + valid_offset - pd.Timedelta(days=1)
            yield Window(
                train_start=train_start.strftime("%Y-%m-%d"),
                train_end=train_end.strftime("%Y-%m-%d"),
                valid_start=valid_start.strftime("%Y-%m-%d"),
                valid_end=valid_end.strftime("%Y-%m-%d"),
            )
            cursor += step

    def _slice(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        start: str,
        end: str,
        is_validation: bool = False,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        按时间范围切片特征和标签。
        
        参数:
            features: 特征数据
            labels: 标签数据
            start: 起始日期
            end: 结束日期
            is_validation: 是否为验证集（如果是，需要考虑标签需要未来数据）
        """
        idx = features.index
        if not isinstance(idx, pd.MultiIndex):
            raise ValueError(f"特征索引应为 MultiIndex，实际为 {type(idx)}")
        
        # 确保 datetime 层级存在
        if "datetime" not in idx.names:
            raise ValueError(f"索引层级中未找到 'datetime'，当前层级: {idx.names}")
        
        # 转换为 Timestamp 以确保正确比较
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)

        # 标签是未来收益（如 Ref($close, -20)/$close - 1）。
        # 为避免“训练集的标签使用了验证期的价格/验证集标签使用了测试期的价格”，
        # 训练/验证切片都需要将结束日期提前 N 天，形成 gap。
        orig_end_ts = end_ts
        if self.label_future_days > 0:
            end_ts = end_ts - pd.Timedelta(days=self.label_future_days)
            if end_ts < start_ts:
                seg = "验证集" if is_validation else "训练集"
                logger.warning(
                    "%s [%s, %s] 需要未来 %d 天数据，调整后结束日期 %s 早于开始日期，返回空集",
                    seg,
                    start,
                    end,
                    self.label_future_days,
                    end_ts.strftime("%Y-%m-%d"),
                )
                return pd.DataFrame(), pd.Series(dtype=float)
            logger.debug(
                "%s结束日期从 %s 调整为 %s（标签需要未来 %d 天数据）",
                "验证集" if is_validation else "训练集",
                orig_end_ts.strftime("%Y-%m-%d"),
                end_ts.strftime("%Y-%m-%d"),
                self.label_future_days,
            )
        
        datetime_level = idx.get_level_values("datetime")
        mask = (datetime_level >= start_ts) & (datetime_level <= end_ts)
        
        feat = features.loc[mask]
        lbl = labels.loc[mask]
        
        # 过滤掉标签为 NaN 的数据（这些数据没有标签，无法用于训练/验证）
        if not lbl.empty:
            valid_mask = ~lbl.isna()
            feat = feat.loc[valid_mask]
            lbl = lbl.loc[valid_mask]
        
        logger.debug(
            "切片 [%s, %s]: 特征样本 %d，标签样本 %d（过滤NaN后）",
            start,
            end_ts.strftime("%Y-%m-%d") if self.label_future_days > 0 else end,
            len(feat), len(lbl)
        )
        
        return feat, lbl

    def _slice_features_only(
        self,
        features: pd.DataFrame,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """仅按时间切片特征，不做 label_future_days 缩尾，不做 label NaN 过滤。"""
        if start > end:
            return pd.DataFrame(index=features.index[:0], columns=features.columns)
        idx = features.index
        if not isinstance(idx, pd.MultiIndex):
            raise ValueError(f"特征索引应为 MultiIndex，实际为 {type(idx)}")
        datetime_level = idx.get_level_values("datetime")
        mask = (datetime_level >= start) & (datetime_level <= end)
        return features.loc[mask]

    def train(self):
        self.pipeline.build()
        features, labels = self.pipeline.get_all()
        
        # 检查标签转换是否生效
        label_is_rank = getattr(self.pipeline, "_label_is_rank", False)
        if label_is_rank:
            logger.info("训练使用 Rank 转换后的标签（范围应在 [0, 1] 之间）")
            logger.info("标签值统计: min=%.6f, max=%.6f, mean=%.6f", 
                       labels.min(), labels.max(), labels.mean())
            if labels.min() < 0 or labels.max() > 1:
                logger.warning("标签值不在 [0, 1] 范围内！可能转换未生效")
        else:
            logger.info("训练使用原始标签（未进行 Rank 转换）")
            logger.info("标签值统计: min=%.6f, max=%.6f, mean=%.6f", 
                       labels.min(), labels.max(), labels.mean())
        
        os.makedirs(self.paths["model_dir"], exist_ok=True)
        os.makedirs(self.paths["log_dir"], exist_ok=True)
        metrics: List[Dict] = []

        # 记录数据时间范围，便于诊断
        if len(features) > 0:
            data_start = features.index.get_level_values("datetime").min()
            data_end = features.index.get_level_values("datetime").max()
            logger.info("特征数据时间范围: %s 至 %s，共 %d 条记录", data_start, data_end, len(features))
        
        for idx, window in enumerate(self._generate_windows()):
            logger.info("==== 滚动窗口 %d: 训练 [%s, %s] 验证 [%s, %s] ====", 
                       idx, window.train_start, window.train_end, window.valid_start, window.valid_end)
            train_feat, train_lbl = self._slice(features, labels, window.train_start, window.train_end, is_validation=False)
            valid_feat, valid_lbl = self._slice(features, labels, window.valid_start, window.valid_end, is_validation=True)
            # 审计日志：显示切片后实际日期范围（已包含 label_future_days 的 gap 调整）
            if not train_feat.empty:
                tmin = train_feat.index.get_level_values("datetime").min()
                tmax = train_feat.index.get_level_values("datetime").max()
                logger.info("窗口 %d 实际训练集日期范围: %s ~ %s（样本=%d）", idx, tmin, tmax, len(train_feat))
            if valid_feat is not None and not valid_feat.empty:
                vmin = valid_feat.index.get_level_values("datetime").min()
                vmax = valid_feat.index.get_level_values("datetime").max()
                logger.info("窗口 %d 实际验证集日期范围: %s ~ %s（样本=%d）", idx, vmin, vmax, len(valid_feat))
            
            if len(train_feat) < self.cfg["rolling"].get("min_samples", 1000):
                logger.warning("训练样本不足 (%d < %d)，跳过该窗口", 
                             len(train_feat), self.cfg["rolling"].get("min_samples", 1000))
                continue
            
            has_valid = valid_feat is not None and not valid_feat.empty and valid_lbl is not None and not valid_lbl.empty
            if not has_valid:
                logger.warning("窗口 %d 验证集为空 (特征: %d, 标签: %d)，退化为仅训练", 
                             idx, len(valid_feat) if valid_feat is not None else 0, 
                             len(valid_lbl) if valid_lbl is not None else 0)
                # 诊断：检查验证时间范围是否在数据范围内
                if len(features) > 0:
                    data_start = features.index.get_level_values("datetime").min()
                    data_end = features.index.get_level_values("datetime").max()
                    valid_start_ts = pd.Timestamp(window.valid_start)
                    valid_end_ts = pd.Timestamp(window.valid_end)
                    if valid_start_ts < data_start or valid_end_ts > data_end:
                        logger.warning("验证时间范围 [%s, %s] 超出数据范围 [%s, %s]", 
                                     window.valid_start, window.valid_end, data_start, data_end)
                valid_feat = None
                valid_lbl = None
            else:
                logger.info("窗口 %d: 训练样本 %d，验证样本 %d", idx, len(train_feat), len(valid_feat))

            # 构造 GRU 等序列模型的历史特征（补齐 label_future_days 造成的间隙）
            # 注意：特征归一化在 EnsembleModelManager 内按模型各自处理（方案B），这里保持 raw 特征。
            history_feat_raw = train_feat
            if has_valid and self.label_future_days > 0:
                try:
                    train_end_actual = train_feat.index.get_level_values("datetime").max()
                    valid_start_actual = valid_feat.index.get_level_values("datetime").min()
                    gap_start = pd.Timestamp(train_end_actual) + pd.Timedelta(days=1)
                    gap_end = pd.Timestamp(valid_start_actual) - pd.Timedelta(days=1)
                    if gap_start <= gap_end:
                        gap_feat = self._slice_features_only(features, gap_start, gap_end)
                        if len(gap_feat) > 0:
                            history_feat_raw = pd.concat([train_feat, gap_feat], axis=0).sort_index()
                            logger.info(
                                "为序列模型补齐历史间隙: %s ~ %s (rows=%d)",
                                gap_start.strftime("%Y-%m-%d"),
                                gap_end.strftime("%Y-%m-%d"),
                                len(gap_feat),
                            )
                except Exception as e:
                    logger.warning("构造序列历史特征失败（忽略继续）：%s", e)

            # 统一训练多模型（归一化在 EnsembleModelManager 内部按模型进行）
            self.ensemble.fit(
                train_feat,
                train_lbl,
                valid_feat if has_valid else None,
                valid_lbl,
                history_feat=history_feat_raw,
            )

            train_blend, train_preds, train_aux = self.ensemble.predict(train_feat, history_feat=history_feat_raw)
            lgb_train_pred = train_preds.get("lgb")
            lgb_train_leaf = train_aux.get("lgb")
            valid_blend = valid_preds = valid_aux = None
            if has_valid:
                # 对序列模型（如 GRU）需要提供训练历史 + 间隙补齐，避免验证集前期缺历史导致预测为空
                valid_blend, valid_preds, valid_aux = self.ensemble.predict(
                    valid_feat,
                    history_feat=history_feat_raw,
                )

            valid_pred = valid_leaf = None
            if valid_preds is not None:
                valid_pred = valid_preds.get("lgb")
            if valid_aux is not None:
                valid_leaf = valid_aux.get("lgb")

            # residual = label - lgb，用于二级学习（可选：仅在启用 stack 且存在 lgb 输出时）
            if self.stack_enabled:
                if lgb_train_pred is None or lgb_train_leaf is None:
                    raise RuntimeError("启用 LeafStackModel 需要 LightGBM 输出，请在 base_models/ensemble.models 中包含 `lgb`")
                train_leaf = lgb_train_leaf
                train_residual = train_lbl - lgb_train_pred
                valid_residual = None if (not has_valid or valid_pred is None) else valid_lbl - valid_pred
                if self.stack is None:
                    raise RuntimeError("stack_enabled=True 但 self.stack 未初始化")
                self.stack.fit(train_leaf, train_residual, valid_leaf, valid_residual)
            else:
                train_leaf = None

            # 可选：OOF + Meta-Stacking（按日期 walk-forward folds 生成 OOF，并训练二层模型）
            oof_cfg = (self.cfg.get("oof_stacking") or {})
            enabled = oof_cfg.get("enabled", False)
            if isinstance(enabled, str) and enabled.strip().lower() == "auto":
                base = [str(x).strip().lower() for x in (self.cfg.get("base_models") or [])]
                # 方案A（lgb+gru+ridge）至少需要 lgb+gru 同时存在
                enabled = ("gru" in base) and ("lgb" in base)
            if bool(enabled):
                try:
                    tag_tmp = window.valid_end.replace("-", "")
                    oof_dir = self.paths.get("oof_dir", os.path.join("data", "oof"))
                    meta_dir = self.paths.get("meta_dir", os.path.join("data", "meta"))
                    splitter = TimeSeriesFoldSplitter(
                        n_splits=int(oof_cfg.get("n_splits", 5)),
                        valid_days=oof_cfg.get("valid_days_per_fold"),
                        min_train_days=int(oof_cfg.get("min_train_days", 60)),
                        gap_days=int(oof_cfg.get("gap_days", 0)),
                    )
                    oof_mgr = OOFManager(oof_dir=oof_dir)
                    model_specs = getattr(self.ensemble, "specs", None) or []
                    X_meta, y_meta, oof_detail = oof_mgr.generate_oof(
                        tag=tag_tmp,
                        model_specs=model_specs,
                        pipeline_cfg=self.cfg,
                        train_feat=train_feat_norm,
                        train_label=train_lbl,
                        splitter=splitter,
                        use_cache=bool(oof_cfg.get("use_cache", True)),
                    )
                    # 落 OOF parquet（方便后续复用/调试）
                    os.makedirs(meta_dir, exist_ok=True)
                    meta_oof_path = os.path.join(meta_dir, f"{tag_tmp}_meta_oof.parquet")
                    if len(oof_detail) > 0:
                        # 将 MultiIndex 拆成列
                        if isinstance(oof_detail.index, pd.MultiIndex):
                            oof_detail = oof_detail.reset_index()
                        # 重命名
                        oof_detail = oof_detail.rename(columns={"datetime": "date", "instrument": "code"})
                        # 仅保留当前基模型列 + y + fold
                        # 统一约定：基模型预测列名使用 pred_{model_name}（与 utils/meta_ridge.py 一致）
                        model_names = [str(spec["name"]).strip().lower() for spec in model_specs]
                        rename_map = {name: f"pred_{name}" for name in model_names}
                        oof_detail = oof_detail.rename(columns=rename_map)
                        keep_cols = ["date", "code", "fold", "y"] + [f"pred_{name}" for name in model_names]
                        missing_cols = [c for c in keep_cols if c not in oof_detail.columns]
                        if missing_cols:
                            raise ValueError(f"meta_oof 缺少列: {missing_cols}")
                        oof_detail = oof_detail[keep_cols]
                        # Ridge 不接受 NaN：在落盘前就过滤掉含 NaN 的行（常见：GRU 序列不足导致 pred_gru 为 NaN）
                        pred_keep = [c for c in oof_detail.columns if c.startswith("pred_")]
                        before = len(oof_detail)
                        oof_detail = oof_detail.dropna(subset=(pred_keep + ["y"]))
                        after = len(oof_detail)
                        if after < before:
                            logger.warning(
                                "meta_oof 落盘前过滤 NaN: %d -> %d (dropped=%d). 可能原因：GRU 序列不足/预测为空或标签缺失。",
                                before,
                                after,
                                before - after,
                            )
                        oof_detail.to_parquet(meta_oof_path, index=False)
                        logger.info("meta_oof 已保存: %s (rows=%d)", meta_oof_path, len(oof_detail))
                    else:
                        logger.warning("meta_oof 未生成（oof_detail 为空）")

                    # 训练 Ridge Meta（方案A）
                    meta_model_cfg = (oof_cfg.get("meta_model") or {})
                    if str(meta_model_cfg.get("model_type", "")).lower() == "ridge":
                        norm_mode = oof_cfg.get("normalize_mode", "zscore")
                        norm_eps = oof_cfg.get("normalize_eps", 1e-6)
                        alpha = float(meta_model_cfg.get("alpha", 1.0))
                        meta_json = os.path.join(meta_dir, f"{tag_tmp}_meta_ridge.json")
                        train_meta_ridge(
                            meta_oof_path=meta_oof_path,
                            out_json=meta_json,
                            alpha=alpha,
                            grid=None,
                            norm_mode=norm_mode,
                            norm_eps=norm_eps,
                        )
                    else:
                        # 仍保留原 MetaStacker 路径（可选）
                        meta = MetaStacker(meta_model_cfg)
                        meta.fit(X_meta, y_meta)
                        meta.save(self.paths["model_dir"], tag_tmp)
                        logger.info("OOF+MetaStacking 已训练并保存: %s (samples=%d, cols=%s)",
                                    tag_tmp, len(X_meta), list(X_meta.columns))
                except Exception as e:
                    logger.error("OOF+MetaStacking 训练失败（不影响主流程）：%s", e, exc_info=True)

            metric = {
                "window": idx,
                "train_start": window.train_start,
                "train_end": window.train_end,
                "valid_start": window.valid_start,
                "valid_end": window.valid_end,
                "segment": "valid" if has_valid else "train",
                "ic_lgb": float("nan"),
                "ic_mlp": float("nan"),
                "ic_gru": float("nan"),
                "ic_stack": float("nan"),
                "ic_qlib_ensemble": float("nan"),
            }

            if has_valid:
                mlp_valid_pred = valid_preds.get("mlp") if valid_preds is not None else None
                gru_valid_pred = valid_preds.get("gru") if valid_preds is not None else None
                stack_valid_pred = None
                if self.stack_enabled and self.stack is not None and valid_leaf is not None and valid_pred is not None:
                    stack_residual = self.stack.predict_residual(valid_leaf, valid_feat.index)
                    stack_valid_pred = self.stack.fuse(valid_pred, stack_residual)
                if valid_pred is not None:
                    metric["ic_lgb"] = _rank_ic(valid_pred, valid_lbl)
                if mlp_valid_pred is not None:
                    metric["ic_mlp"] = _rank_ic(mlp_valid_pred, valid_lbl)
                if gru_valid_pred is not None:
                    metric["ic_gru"] = _rank_ic(gru_valid_pred, valid_lbl)
                if stack_valid_pred is not None:
                    metric["ic_stack"] = _rank_ic(stack_valid_pred, valid_lbl)
                if valid_blend is not None:
                    metric["ic_qlib_ensemble"] = _rank_ic(valid_blend, valid_lbl)
            else:
                # 退化为训练集指标，至少保证输出文件存在，便于预测阶段读取
                mlp_train_pred = train_preds.get("mlp")
                gru_train_pred = train_preds.get("gru")
                stack_train_pred = None
                if self.stack_enabled and self.stack is not None and train_leaf is not None and lgb_train_pred is not None:
                    stack_train_residual = self.stack.predict_residual(train_leaf, train_feat.index)
                    stack_train_pred = self.stack.fuse(lgb_train_pred, stack_train_residual)
                if lgb_train_pred is not None:
                    metric["ic_lgb"] = _rank_ic(lgb_train_pred, train_lbl)
                if mlp_train_pred is not None:
                    metric["ic_mlp"] = _rank_ic(mlp_train_pred, train_lbl)
                if gru_train_pred is not None:
                    metric["ic_gru"] = _rank_ic(gru_train_pred, train_lbl)
                if stack_train_pred is not None:
                    metric["ic_stack"] = _rank_ic(stack_train_pred, train_lbl)
                if train_blend is not None:
                    metric["ic_qlib_ensemble"] = _rank_ic(train_blend, train_lbl)
            metrics.append(metric)

            # 以验证区间结束日作为模型文件名，方便按日期加载
            tag = window.valid_end.replace("-", "")
            self.ensemble.save(self.paths["model_dir"], tag)
            if self.stack_enabled and self.stack is not None:
                self.stack.save(self.paths["model_dir"], tag)
            
            # 保存归一化参数（用于预测时使用；支持 per-model 与旧 global 格式）
            import json
            norm_meta_path = os.path.join(self.paths["model_dir"], f"{tag}_norm_meta.json")
            norm_meta = self.ensemble.get_norm_meta()
            norm_meta.update(
                {
                    "train_start": window.train_start,
                    "train_end": window.train_end,
                    "valid_end": window.valid_end,
                }
            )
            with open(norm_meta_path, "w", encoding="utf-8") as fp:
                json.dump(norm_meta, fp, ensure_ascii=False, indent=2, default=str)
            logger.info("归一化参数已保存: %s", norm_meta_path)

        if metrics:
            # 记录所有窗口的 IC，用于后续动态加权或评估
            df = pd.DataFrame(metrics)
            df.to_csv(os.path.join(self.paths["log_dir"], "training_metrics.csv"), index=False)
            logger.info("训练指标已保存，共 %d 条记录", len(df))
            # 训练结束：输出 GRU 的 IC/ICIR 汇总（优先使用 valid 段）
            try:
                import numpy as np
                seg = "valid" if (df.get("segment") == "valid").any() else None
                df_eval = df[df["segment"] == "valid"] if seg == "valid" else df
                if "ic_gru" in df_eval.columns:
                    s = pd.to_numeric(df_eval["ic_gru"], errors="coerce").dropna()
                    if len(s) >= 2:
                        mean_ic = float(s.mean())
                        std_ic = float(s.std(ddof=0))
                        icir = float(mean_ic / (std_ic + 1e-12))
                        logger.info("GRU IC 汇总(%s): n=%d mean=%.6f std=%.6f ICIR=%.6f",
                                    "valid" if seg == "valid" else "all", len(s), mean_ic, std_ic, icir)
                    elif len(s) == 1:
                        logger.info("GRU IC 汇总(%s): n=1 ic=%.6f（窗口数不足，无法计算 ICIR）",
                                    "valid" if seg == "valid" else "all", float(s.iloc[0]))
                    else:
                        logger.warning("GRU IC 汇总：没有有效 ic_gru（可能 GRU 预测为空/NaN 或验证集为空）")
            except Exception as e:
                logger.warning("输出 GRU ICIR 汇总失败：%s", e)
        else:
            logger.warning("未产出任何训练窗口指标")

