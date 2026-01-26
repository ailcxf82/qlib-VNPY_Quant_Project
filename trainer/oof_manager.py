"""
OOF（Out-Of-Fold）生成与缓存管理：
- 统一的 time-series / walk-forward folds（按日期切分，避免未来信息泄漏）
- 对多个基模型复用同一套 folds，生成并缓存 OOF 预测
- 输出用于 stacking 的 X_meta（按 MultiIndex 对齐）
"""

from __future__ import annotations

import os
import pickle
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import json

from models.model_registry import create_model
from utils import load_yaml_config

logger = logging.getLogger(__name__)


@dataclass
class Fold:
    fold: int
    train_dates: pd.Index  # DatetimeIndex / Index of timestamps
    valid_dates: pd.Index


class TimeSeriesFoldSplitter:
    """
    按日期生成 walk-forward folds（面向 panel 数据，按 datetime 切分）。

    设计目标：
    - 所有模型共享同一套 folds
    - 仅按时间切分，不混用未来日期
    """

    def __init__(
        self,
        n_splits: int = 5,
        valid_days: Optional[int] = None,
        min_train_days: int = 60,
        gap_days: int = 0,
    ):
        self.n_splits = int(n_splits)
        self.valid_days = int(valid_days) if valid_days is not None else None
        self.min_train_days = int(min_train_days)
        self.gap_days = int(gap_days)

    def split(self, dt_index: pd.Index) -> List[Fold]:
        if len(dt_index) == 0:
            return []
        dts = pd.Index(pd.to_datetime(dt_index).unique()).sort_values()
        if len(dts) < self.min_train_days + 1:
            return []

        # 如果指定 valid_days，则每折使用固定长度；否则采用等分的 TimeSeriesSplit 风格
        if self.valid_days is not None:
            v = self.valid_days
            total_needed = self.min_train_days + self.gap_days + self.n_splits * v
            if len(dts) < total_needed:
                # 尽量缩小有效折数（不报错，便于小数据跑通）
                max_splits = max(1, (len(dts) - self.min_train_days - self.gap_days) // max(1, v))
                n_splits = min(self.n_splits, max_splits)
            else:
                n_splits = self.n_splits

            folds: List[Fold] = []
            for i in range(n_splits):
                train_end = self.min_train_days + i * v
                valid_start = train_end + self.gap_days
                valid_end = valid_start + v
                if valid_end > len(dts):
                    break
                train_dates = dts[:train_end]
                valid_dates = dts[valid_start:valid_end]
                folds.append(Fold(fold=i, train_dates=train_dates, valid_dates=valid_dates))
            return folds

        # 等分方式：参考 sklearn TimeSeriesSplit 的 fold_size 逻辑
        n_dates = len(dts)
        test_size = max(1, n_dates // (self.n_splits + 1))
        folds = []
        for i in range(self.n_splits):
            train_end = (i + 1) * test_size
            valid_start = train_end + self.gap_days
            valid_end = valid_start + test_size
            if train_end < self.min_train_days:
                continue
            if valid_end > n_dates:
                break
            folds.append(Fold(fold=i, train_dates=dts[:train_end], valid_dates=dts[valid_start:valid_end]))
        return folds


class OOFManager:
    """
    生成并缓存 OOF 预测，输出 stacking 的 X_meta。

    缓存约定（可通过 paths.oof_dir 配置根目录）：
      {oof_dir}/{tag}/{model_name}_{fold}.npy
      {oof_dir}/{tag}/y_{fold}.npy
      {oof_dir}/{tag}/index_{fold}.pkl
    """

    def __init__(self, oof_dir: str):
        self.oof_dir = oof_dir

    def _tag_dir(self, tag: str) -> str:
        d = os.path.join(self.oof_dir, str(tag))
        os.makedirs(d, exist_ok=True)
        return d

    @staticmethod
    def _mask_by_dates(idx: pd.MultiIndex, dates: pd.Index) -> np.ndarray:
        dt = pd.to_datetime(idx.get_level_values("datetime"))
        date_set = set(pd.to_datetime(dates))
        return np.array([x in date_set for x in dt], dtype=bool)

    def generate_oof(
        self,
        *,
        tag: str,
        model_specs: List[Dict],
        pipeline_cfg: Dict,
        train_feat: pd.DataFrame,
        train_label: pd.Series,
        splitter: TimeSeriesFoldSplitter,
        use_cache: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        返回：
        - X_meta: DataFrame(index=MultiIndex[datetime,instrument], columns=[model_name...])
        - y_meta: Series(label) 与 X_meta 对齐
        - oof_detail: DataFrame[datetime, instrument, fold, preds..., y]（便于落 parquet）
        """
        if not isinstance(train_feat.index, pd.MultiIndex) or "datetime" not in train_feat.index.names:
            raise ValueError("OOF 需要 train_feat 为 MultiIndex 且包含 datetime level")

        # 解析按模型特征集合
        raw_model_features = pipeline_cfg.get("model_features", {}) or {}
        model_features = {str(k).strip().lower(): v for k, v in raw_model_features.items()}
        feature_sets: Dict[str, List[str]] = {}
        data_cfg_path = pipeline_cfg.get("data_config")
        if isinstance(data_cfg_path, str):
            try:
                data_cfg = load_yaml_config(data_cfg_path)
                feature_sets = data_cfg.get("data", {}).get("feature_sets", {}) or {}
            except Exception as e:
                logger.warning(
                    "OOF: 读取 data_config 失败，将忽略 data.feature_sets（data_config=%s, cwd=%s, err=%s）",
                    data_cfg_path,
                    os.getcwd(),
                    e,
                )
                feature_sets = {}

        feature_log_done: set[str] = set()

        def _resolve_feature_cols(model_name: str, all_cols: List[str]) -> Optional[List[str]]:
            if not model_features:
                return None
            spec = model_features.get(model_name)
            if spec is None:
                raise ValueError(
                    f"OOF: model_features 已配置但未包含模型 {model_name}。"
                    "请在 pipeline.yaml 的 model_features 中为该模型指定特征集合。"
                )
            if isinstance(spec, str):
                key = spec.strip()
                if key in {"*", "all", "ALL"}:
                    return None
                if key in feature_sets:
                    cols = list(feature_sets[key] or [])
                else:
                    avail = []
                    try:
                        if isinstance(feature_sets, dict):
                            avail = sorted([str(k) for k in feature_sets.keys()])[:30]
                    except Exception:
                        avail = []
                    raise ValueError(
                        f"OOF: model_features[{model_name}]={key} 未在 data.feature_sets 中定义"
                        + (f"（可用 keys 示例: {avail}）" if avail else "")
                    )
            elif isinstance(spec, list):
                cols = list(spec)
            else:
                raise ValueError(f"OOF: model_features[{model_name}] 仅支持 str 或 list")
            if not cols:
                return None
            missing = [c for c in cols if c not in all_cols]
            if missing:
                raise ValueError(f"OOF: 模型 {model_name} 特征缺失: {missing[:10]}")
            if model_name not in feature_log_done:
                logger.info(
                    "OOF: 模型 %s 使用特征集合=%s，列数=%d（示例: %s）",
                    model_name,
                    spec,
                    len(cols),
                    cols[:8],
                )
                feature_log_done.add(model_name)
            return cols

        def _select_features(model_name: str, df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
            if df is None or df.empty:
                return df
            cols = _resolve_feature_cols(model_name, list(df.columns))
            if not cols:
                return df
            return df.loc[:, cols]

        tag_dir = self._tag_dir(tag)
        # 缓存版本：避免“配置/归一化逻辑变了但 index 没变”导致语义错用旧 .npy
        cache_info_path = os.path.join(tag_dir, "cache_info.json")
        cache_version = "v2_per_model_norm"
        if use_cache and os.path.exists(cache_info_path):
            try:
                info = json.load(open(cache_info_path, "r", encoding="utf-8"))
                if info.get("version") != cache_version:
                    logger.warning("OOF cache 版本变化：将忽略旧缓存并重算（%s -> %s）", info.get("version"), cache_version)
                    use_cache = False
            except Exception:
                use_cache = False
        if not use_cache:
            try:
                json.dump({"version": cache_version}, open(cache_info_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            except Exception:
                pass
        folds = splitter.split(train_feat.index.get_level_values("datetime"))
        if not folds:
            raise RuntimeError("无法生成 folds：训练数据日期过少或 min_train_days 设置过大")

        # 收集各模型的 OOF 预测（按 fold 追加，最后 concat）
        per_model_parts: Dict[str, List[pd.Series]] = {spec["name"]: [] for spec in model_specs}
        y_parts: List[pd.Series] = []
        detail_rows: List[pd.DataFrame] = []

        for fold in folds:
            idx_fold_path = os.path.join(tag_dir, f"index_{fold.fold}.pkl")
            y_fold_path = os.path.join(tag_dir, f"y_{fold.fold}.npy")
            # 先计算 valid 的 index（用于所有模型复用/对齐）
            valid_mask = self._mask_by_dates(train_feat.index, fold.valid_dates)
            valid_index = train_feat.index[valid_mask]
            if len(valid_index) == 0:
                continue

            # 标签缓存（标签与 index 一一对应）
            y_valid = train_label.reindex(valid_index)
            y_parts.append(y_valid)

            # 缓存校验：OOF 缓存必须与“本次 fold 的 valid_index”严格一致，否则长度会错配
            # 典型触发场景：修改了数据时间范围、rolling 配置、特征缺失清理策略、模型训练窗口等，导致 valid_index 变化，
            # 但 tag 相同、目录相同，旧的 .npy 预测仍在。
            cached_index = None
            if use_cache and os.path.exists(idx_fold_path):
                try:
                    with open(idx_fold_path, "rb") as fp:
                        cached_index = pickle.load(fp)
                    if not isinstance(cached_index, pd.MultiIndex):
                        cached_index = None
                except Exception:
                    cached_index = None

            index_ok = (cached_index is not None) and (len(cached_index) == len(valid_index)) and cached_index.equals(valid_index)
            if use_cache and index_ok and os.path.exists(y_fold_path):
                # y 缓存也要校验长度
                try:
                    y_arr = np.load(y_fold_path)
                    if len(y_arr) != len(valid_index):
                        index_ok = False
                except Exception:
                    index_ok = False

            if not (use_cache and index_ok):
                # index/y 缓存无效：重写 index/y，后续模型预测缓存也将按长度校验决定是否重算
                with open(idx_fold_path, "wb") as fp:
                    pickle.dump(valid_index, fp)
                np.save(y_fold_path, y_valid.values.astype(np.float32), allow_pickle=False)

            # 训练集 mask（可选 gap）
            train_mask = self._mask_by_dates(train_feat.index, fold.train_dates)
            fold_train_feat = train_feat.loc[train_mask]
            fold_train_lbl = train_label.loc[train_mask]
            fold_valid_feat = train_feat.loc[valid_mask]

            for spec in model_specs:
                mname = str(spec["name"]).strip().lower()
                out_path = os.path.join(tag_dir, f"{mname}_{fold.fold}.npy")
                if use_cache and os.path.exists(out_path):
                    try:
                        pred_arr = np.load(out_path)
                        if len(pred_arr) != len(valid_index):
                            logger.warning(
                                "OOF cache 长度不匹配，已自动失效并重算: tag=%s fold=%s model=%s pred_len=%d index_len=%d",
                                tag,
                                fold.fold,
                                mname,
                                len(pred_arr),
                                len(valid_index),
                            )
                        else:
                            per_model_parts[mname].append(pd.Series(pred_arr, index=valid_index, name=mname))
                            continue
                    except Exception as e:
                        logger.warning(
                            "OOF cache 读取失败，已自动失效并重算: tag=%s fold=%s model=%s err=%s",
                            tag,
                            fold.fold,
                            mname,
                            e,
                        )

                # 每 fold 单独实例化模型，避免 state 污染
                cfg = spec.get("config")
                if cfg is None:
                    # config_key 解析逻辑与 EnsembleModelManager 一致
                    key = spec.get("config_key")
                    if key and key in pipeline_cfg:
                        cfg = pipeline_cfg[key]
                if cfg is None:
                    raise ValueError(f"OOF: 模型 {mname} 缺少 config/config_key")

                model = create_model(spec["type"], cfg)
                # 按模型选择特征列
                fold_train_feat_sel = _select_features(mname, fold_train_feat)
                fold_valid_feat_sel = _select_features(mname, fold_valid_feat)
                # ===== per-model normalization（每 fold 单独拟合；与主训练一致）=====
                mean = fold_train_feat_sel.mean()
                std = fold_train_feat_sel.std().replace(0, 1)
                # 对齐列：缺失列用 mean 填充（归一化后为 0）
                def _apply_norm(df: pd.DataFrame) -> pd.DataFrame:
                    expected = list(mean.index)
                    aligned = pd.DataFrame(index=df.index, columns=expected, dtype=float)
                    for c in expected:
                        aligned[c] = df[c] if c in df.columns else float(mean[c])
                    z = (aligned - mean) / std
                    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                    return z.clip(-5, 5)
                fold_train_feat_sel = _apply_norm(fold_train_feat_sel)
                fold_valid_feat_sel = _apply_norm(fold_valid_feat_sel)
                # 注意：OOF 仅基于训练折训练；valid 折仅用于预测
                model.fit(fold_train_feat_sel, fold_train_lbl, None, None)
                # 对序列模型（如 GRU）传入历史特征，避免 valid 前期缺历史导致 OOF 为空
                try:
                    pred_out = model.predict(fold_valid_feat_sel, history_feat=fold_train_feat_sel)
                except TypeError:
                    pred_out = model.predict(fold_valid_feat_sel)
                if isinstance(pred_out, tuple):
                    pred_series = pred_out[0]
                else:
                    pred_series = pred_out
                pred_series = pred_series.reindex(valid_index)
                np.save(out_path, pred_series.values.astype(np.float32), allow_pickle=False)
                per_model_parts[mname].append(pred_series.rename(mname))

            # 构建当前 fold 的明细行，便于后续一次性落 parquet
            detail = pd.DataFrame(index=valid_index)
            detail["fold"] = fold.fold
            for spec in model_specs:
                name = spec["name"]
                detail[name] = per_model_parts[name][-1].reindex(valid_index)
            detail["y"] = y_valid
            detail_rows.append(detail)

        # 拼接所有 folds 的 OOF，并按共同索引对齐
        oof_cols = {}
        for mname, parts in per_model_parts.items():
            if not parts:
                continue
            s = pd.concat(parts).sort_index()
            oof_cols[mname] = s
        if not oof_cols:
            raise RuntimeError("OOF 生成失败：未得到任何模型的 OOF 预测")

        y_all = pd.concat(y_parts).sort_index()
        X_meta = pd.DataFrame(oof_cols)

        # 对齐：取所有列与 y 的交集索引，避免错位/NaN
        common = X_meta.dropna().index
        common = common.intersection(y_all.dropna().index)
        X_meta = X_meta.loc[common].sort_index()
        y_all = y_all.loc[common].sort_index()
        # 明细
        if detail_rows:
            detail_df = pd.concat(detail_rows).sort_index()
        else:
            detail_df = pd.DataFrame(columns=["datetime", "instrument", "fold", "y"])
        return X_meta, y_all, detail_df


