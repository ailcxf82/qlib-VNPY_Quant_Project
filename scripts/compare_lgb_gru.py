"""
对比脚本：在同一 walk-forward 窗口/切分下，比较 LightGBM 与 GRU 的验证表现（RankIC/MSE）。

支持“两个模型用不同的特征库”：
- 可以通过 --lgb-data-config / --gru-data-config 指定不同的 data.yaml（不同因子/字段）
- 或者通过 --lgb-cols / --gru-cols 选择不同的特征列子集

用法示例：
  python scripts/compare_lgb_gru.py --pipeline-config config/pipeline.yaml --tag 20250119
  python scripts/compare_lgb_gru.py --lgb-data-config config/data.yaml --gru-data-config config/data_gru.yaml
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, List

import numpy as np
import pandas as pd

from feature.qlib_feature_pipeline import QlibFeaturePipeline
from models.lightgbm_model import LightGBMModelWrapper
from models.gru_model import GRURegressor
from trainer.oof_manager import TimeSeriesFoldSplitter
from utils import load_yaml_config


def _rank_ic(pred: pd.Series, y: pd.Series) -> float:
    pred, y = pred.align(y, join="inner")
    if pred.empty:
        return float("nan")
    return pred.rank().corr(y, method="spearman")


def _parse_cols(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    cols = [c.strip() for c in s.split(",") if c.strip()]
    return cols or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline-config", type=str, default="config/pipeline.yaml")
    ap.add_argument("--lgb-data-config", type=str, default=None)
    ap.add_argument("--gru-data-config", type=str, default=None)
    ap.add_argument("--lgb-cols", type=str, default=None, help="逗号分隔的列名子集（可选）")
    ap.add_argument("--gru-cols", type=str, default=None, help="逗号分隔的列名子集（可选）")
    ap.add_argument("--folds", type=int, default=1, help="对比前 N 折（默认 1）")
    args = ap.parse_args()

    cfg = load_yaml_config(args.pipeline_config)
    data_cfg_path = cfg["data_config"]
    lgb_data_cfg = args.lgb_data_config or data_cfg_path
    gru_data_cfg = args.gru_data_config or data_cfg_path

    # 1) 构建特征（允许两套 data_config）
    pipe_lgb = QlibFeaturePipeline(lgb_data_cfg)
    pipe_lgb.build()
    feat_lgb, y_lgb = pipe_lgb.get_all()

    pipe_gru = QlibFeaturePipeline(gru_data_cfg)
    pipe_gru.build()
    feat_gru, y_gru = pipe_gru.get_all()

    # 2) 对齐标签（以 date+code 交集为准）
    common_idx = feat_lgb.index.intersection(feat_gru.index)
    common_idx = common_idx.intersection(y_lgb.index).intersection(y_gru.index)
    feat_lgb = feat_lgb.loc[common_idx]
    feat_gru = feat_gru.loc[common_idx]
    y = y_lgb.loc[common_idx]  # label 定义应一致，直接用 lgb 这一份

    # 3) 可选：不同特征列子集
    lgb_cols = _parse_cols(args.lgb_cols)
    gru_cols = _parse_cols(args.gru_cols)
    if lgb_cols is not None:
        feat_lgb = feat_lgb[lgb_cols]
    if gru_cols is not None:
        feat_gru = feat_gru[gru_cols]

    # 4) 构造 folds（按日期）
    rolling = cfg.get("rolling", {})
    splitter = TimeSeriesFoldSplitter(
        n_splits=max(1, int(args.folds)),
        valid_days=int(rolling.get("valid_days", 30)),
        min_train_days=int(rolling.get("train_days", 720)),
        gap_days=0,
    )
    folds = splitter.split(common_idx.get_level_values("datetime"))
    if not folds:
        raise RuntimeError("无法生成 folds：请检查 rolling.train_days/valid_days 或数据日期范围")

    # 5) 只对比前 N 折
    folds = folds[: int(args.folds)]
    print(f"[compare] total folds={len(folds)}  (lgb_data={lgb_data_cfg}, gru_data={gru_data_cfg})")

    # 6) 模型配置
    lgb_model = LightGBMModelWrapper(cfg["lightgbm_config"])
    gru_cfg = {"model": cfg.get("model_gru", load_yaml_config(cfg.get("gru_config", "config/model_gru.yaml")).get("model", {}))}
    gru_model = GRURegressor(gru_cfg)

    for f in folds:
        dt = common_idx.get_level_values("datetime")
        train_mask = dt.isin(pd.to_datetime(f.train_dates))
        valid_mask = dt.isin(pd.to_datetime(f.valid_dates))

        tr_lgb, va_lgb = feat_lgb.loc[train_mask], feat_lgb.loc[valid_mask]
        tr_gru, va_gru = feat_gru.loc[train_mask], feat_gru.loc[valid_mask]
        y_tr, y_va = y.loc[train_mask], y.loc[valid_mask]

        # 归一化（各自用训练集统计量；避免泄露）
        tr_lgb_n, mean_lgb, std_lgb = pipe_lgb.normalize_features(tr_lgb)
        va_lgb_n = ((va_lgb - mean_lgb) / std_lgb).clip(-5, 5)

        tr_gru_n, mean_gru, std_gru = pipe_gru.normalize_features(tr_gru)
        va_gru_n = ((va_gru - mean_gru) / std_gru).clip(-5, 5)

        # 训练/预测：LGB
        lgb_model.fit(tr_lgb_n, y_tr, None, None)
        lgb_pred, _ = lgb_model.predict(va_lgb_n)
        ic_lgb = _rank_ic(lgb_pred, y_va)
        mse_lgb = float(np.mean((lgb_pred.align(y_va, join="inner")[0].values - y_va.align(lgb_pred, join="inner")[0].values) ** 2))

        # 训练/预测：GRU（关键：valid 预测需要 train 历史）
        gru_model.fit(tr_gru_n, y_tr, va_gru_n, y_va)
        gru_pred = gru_model.predict(va_gru_n, history_feat=tr_gru_n)
        ic_gru = _rank_ic(gru_pred, y_va)
        aligned_p, aligned_y = gru_pred.align(y_va, join="inner")
        mse_gru = float(np.mean((aligned_p.values - aligned_y.values) ** 2)) if len(aligned_p) else float("nan")

        print(
            f"[fold={f.fold}] "
            f"LGB ic={ic_lgb:.6f} mse={mse_lgb:.6f} | "
            f"GRU ic={ic_gru:.6f} mse={mse_gru:.6f} "
            f"(train_dates={len(f.train_dates)}, valid_dates={len(f.valid_dates)})"
        )


if __name__ == "__main__":
    main()



