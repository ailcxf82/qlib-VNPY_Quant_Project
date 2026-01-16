"""
OOF（Out-Of-Fold）生成与缓存管理：
- 统一的 time-series / walk-forward folds（按日期切分，避免未来信息泄漏）
- 对多个基模型复用同一套 folds，生成并缓存 OOF 预测
- 输出用于 stacking 的 X_meta（按 MultiIndex 对齐）
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from models.model_registry import create_model


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

        tag_dir = self._tag_dir(tag)
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

            if use_cache and os.path.exists(idx_fold_path) and os.path.exists(y_fold_path):
                # index/y 缓存存在就复用；模型预测各自单独判断
                pass
            else:
                with open(idx_fold_path, "wb") as fp:
                    pickle.dump(valid_index, fp)
                np.save(y_fold_path, y_valid.values.astype(np.float32), allow_pickle=False)

            # 训练集 mask（可选 gap）
            train_mask = self._mask_by_dates(train_feat.index, fold.train_dates)
            fold_train_feat = train_feat.loc[train_mask]
            fold_train_lbl = train_label.loc[train_mask]
            fold_valid_feat = train_feat.loc[valid_mask]

            for spec in model_specs:
                mname = spec["name"]
                out_path = os.path.join(tag_dir, f"{mname}_{fold.fold}.npy")
                if use_cache and os.path.exists(out_path):
                    pred_arr = np.load(out_path)
                    per_model_parts[mname].append(pd.Series(pred_arr, index=valid_index, name=mname))
                    continue

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
                # 注意：OOF 仅基于训练折训练；valid 折仅用于预测
                model.fit(fold_train_feat, fold_train_lbl, None, None)
                # 对序列模型（如 GRU）传入历史特征，避免 valid 前期缺历史导致 OOF 为空
                try:
                    pred_out = model.predict(fold_valid_feat, history_feat=fold_train_feat)
                except TypeError:
                    pred_out = model.predict(fold_valid_feat)
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


