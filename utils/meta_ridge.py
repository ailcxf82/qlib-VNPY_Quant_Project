"""
方案 A 的 Meta 模型：Ridge 回归（线性 + L2），输入为按日标准化后的 [pred_lgb, pred_gru]，输出 final_score。

提供两个主要函数：
  1) train_meta_ridge(meta_oof_path, out_json, alpha=1.0, norm_mode="zscore", norm_eps=1e-6, grid=None)
     - 使用严格 OOF（meta_oof）训练 Ridge
     - 支持固定 alpha 或在 OOF 上做简单网格（可选）
     - 打印/保存 coef_、intercept_、OOF 指标（RankIC、MSE）
     - 将参数、归一化配置、列顺序写入 json（out_json）
  2) predict_meta_ridge(lgb_pred_path, gru_pred_path, ridge_json, out_path="meta_pred.parquet")
     - 读取 LGB/GRU 的预测文件（date, code, pred_lgb/pred_gru）
     - (date, code) join 后按日标准化（与训练一致），使用 Ridge 参数输出 final_score
     - 落盘最终信号（包含 date, code, final_score）

依赖：
  - scikit-learn（Ridge）
  - utils.normalize.normalize_by_date
"""

from __future__ import annotations

import json
import os
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from utils.normalize import normalize_by_date

REQUIRED_COLS_OOF_BASE = {"date", "code", "fold", "y"}
REQUIRED_COLS_PRED_BASE = {"date", "code"}


def _read_df(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _rank_ic(pred: pd.Series, y: pd.Series) -> float:
    pred, y = pred.align(y, join="inner")
    if len(pred) == 0:
        return float("nan")
    return pred.rank().corr(y, method="spearman")


def _assert_no_dupe(df: pd.DataFrame, cols: Iterable[str], msg: str):
    if df.duplicated(subset=list(cols)).any():
        dup = df[df.duplicated(subset=list(cols), keep=False)]
        raise ValueError(f"{msg} 发现重复 key，样本数={len(dup)}")


def train_meta_ridge(
    meta_oof_path: str,
    out_json: str = "outputs/meta_ridge.json",
    *,
    alpha: float = 1.0,
    grid: Optional[Iterable[float]] = None,
    norm_mode: str = "zscore",
    norm_eps: float = 1e-6,
):
    """
    用严格 OOF 的 meta_oof 训练 Ridge。
    meta_oof 需包含: [date, code, fold, y, pred_lgb, pred_gru]
    """
    df = _read_df(meta_oof_path)
    missing = REQUIRED_COLS_OOF_BASE - set(df.columns)
    if missing:
        raise ValueError(f"meta_oof 缺少必要列: {missing}")

    # 兼容历史文件：有的版本使用列名 lgb/gru（无 pred_ 前缀）
    if "pred_lgb" not in df.columns and "lgb" in df.columns:
        df = df.rename(columns={"lgb": "pred_lgb"})
    if "pred_gru" not in df.columns and "gru" in df.columns:
        df = df.rename(columns={"gru": "pred_gru"})

    # 自动识别预测列：
    # - 优先使用 pred_* 列
    # - 若不存在，则退回到除 base 列之外的所有列
    pred_cols = [c for c in df.columns if str(c).startswith("pred_")]
    if not pred_cols:
        pred_cols = [c for c in df.columns if c not in REQUIRED_COLS_OOF_BASE]
    # 常见方案 A：固定顺序 pred_lgb, pred_gru（若存在）
    preferred = [c for c in ["pred_lgb", "pred_gru"] if c in pred_cols]
    if preferred:
        # 如果还有其它 pred_ 列，也保留在后面，保证可扩展到更多基模型
        rest = [c for c in pred_cols if c not in preferred]
        pred_cols = preferred + sorted(rest)
    else:
        pred_cols = sorted(pred_cols)
    if not pred_cols:
        raise ValueError("meta_oof 未找到任何预测列（期望 pred_* 或至少存在除 [date,code,fold,y] 外的列）")

    # Phase 1 P1-3：处理新版 meta_oof（含 gru_coverage 列，pred_gru 允许 NaN）。
    # 旧行为：``df.dropna(subset=pred_cols + ["y"])`` 把 GRU 缺失行整体剔除，
    # 直接导致 Ridge 在 LGB 占优分布上拟合 → GRU 系数被压低。
    # 新行为：仅过滤 y 缺失或 LGB 缺失的行；对 GRU 等"高 NaN 率"列在
    # normalize 之后用 0 填充（z-score 后 0 = 当日均值 = 无信号）。
    has_coverage_col = "gru_coverage" in df.columns
    required_cols = ["y"]
    if "pred_lgb" in pred_cols:
        required_cols.append("pred_lgb")
    before = len(df)
    df = df.dropna(subset=required_cols)
    after = len(df)
    if after < before:
        print(f"[meta_ridge] dropna y/pred_lgb: {before} -> {after} (dropped={before-after})")
    if len(df) == 0:
        raise ValueError("[meta_ridge] 过滤 y/pred_lgb 缺失后样本为 0：请检查 OOF 是否完整")
    # 报告 GRU 覆盖率（仅记录，不剔除）
    if has_coverage_col:
        cov = float(df["gru_coverage"].mean())
        print(f"[meta_ridge] gru_coverage = {cov:.2%}（缺失行的 pred_gru 在归一化后置 0 = 当日均值）")
    elif "pred_gru" in pred_cols:
        gru_nan_ratio = float(df["pred_gru"].isna().mean())
        print(f"[meta_ridge] pred_gru NaN ratio = {gru_nan_ratio:.2%}（旧版 OOF；将在归一化后置 0）")

    # 按日标准化（仅用当日截面；pandas mean/std 自动跳过 NaN）
    df = normalize_by_date(df, cols=pred_cols, date_col="date", mode=norm_mode, eps=norm_eps)

    # 标准化后剩余 NaN/Inf（GRU 序列不足等原因）：用 0 填充表示"无信号"
    df = df.replace([np.inf, -np.inf], np.nan)
    before2 = len(df)
    df = df.dropna(subset=["y"])  # y 仍必须非 NaN
    after2 = len(df)
    if after2 < before2:
        print(f"[meta_ridge] dropna y after normalize: {before2} -> {after2} (dropped={before2-after2})")
    nan_pre = df[pred_cols].isna().sum()
    if nan_pre.sum() > 0:
        for c in pred_cols:
            n = int(nan_pre[c])
            if n > 0:
                print(f"[meta_ridge] fillna(0) {c}: {n} 条（z-score 后 0 = 当日均值 = 无信号）")
        df[pred_cols] = df[pred_cols].fillna(0.0)
    if len(df) == 0:
        raise ValueError("[meta_ridge] 标准化后样本为 0：请检查 pred 列按日是否全缺失/全常量")

    X = df[pred_cols].values
    y = df["y"].values

    def fit_score(a: float):
        model = Ridge(alpha=float(a))
        model.fit(X, y)
        pred = model.predict(X)
        rank_ic = _rank_ic(pd.Series(pred, index=df.index), pd.Series(y, index=df.index))
        mse = float(np.mean((pred - y) ** 2))
        return model, rank_ic, mse

    best_alpha = float(alpha)
    best_model, best_ic, best_mse = fit_score(alpha)
    if grid:
        for a in grid:
            m, ic, mse = fit_score(a)
            # 以 RankIC 最大为准，若并列则取较小 MSE
            if (not np.isnan(ic)) and (ic > best_ic or (ic == best_ic and mse < best_mse)):
                best_alpha, best_model, best_ic, best_mse = float(a), m, ic, mse

    coef = best_model.coef_.tolist()
    intercept = float(best_model.intercept_)

    print(f"[meta_ridge] alpha={best_alpha} coef={coef} intercept={intercept}")
    print(f"[meta_ridge] OOF RankIC={best_ic:.6f} MSE={best_mse:.6f}")

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    payload = {
        "alpha": best_alpha,
        "coef": coef,
        "intercept": intercept,
        "cols": pred_cols,
        "norm_mode": norm_mode,
        "norm_eps": norm_eps,
        "metrics": {"oof_rank_ic": best_ic, "oof_mse": best_mse},
        "meta_oof_path": meta_oof_path,
    }
    with open(out_json, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    print(f"[meta_ridge] saved params -> {out_json}")
    return payload


def predict_meta_ridge(
    lgb_pred_path: str,
    gru_pred_path: str,
    ridge_json: str,
    out_path: str = "meta_pred.parquet",
):
    """
    推理阶段：读取 LGB/GRU 预测，(date, code) join，按日标准化后用 Ridge 参数输出 final_score。
    需要：
      - lgb_pred_path: 包含 [date, code, pred_lgb]
      - gru_pred_path: 包含 [date, code, pred_gru]
      - ridge_json: 训练保存的参数文件
    """
    with open(ridge_json, "r", encoding="utf-8") as fp:
        cfg = json.load(fp)

    lgb = _read_df(lgb_pred_path)
    gru = _read_df(gru_pred_path)
    # 兼容历史文件：允许列名为 lgb/gru
    if "pred_lgb" not in lgb.columns and "lgb" in lgb.columns:
        lgb = lgb.rename(columns={"lgb": "pred_lgb"})
    if "pred_gru" not in gru.columns and "gru" in gru.columns:
        gru = gru.rename(columns={"gru": "pred_gru"})

    for req_col, df, name in [
        ("pred_lgb", lgb, "lgb_pred_path"),
        ("pred_gru", gru, "gru_pred_path"),
    ]:
        missing_base = REQUIRED_COLS_PRED_BASE - set(df.columns)
        if missing_base:
            raise ValueError(f"{name} 缺少必要列: {missing_base}")
        if req_col not in df.columns:
            raise ValueError(f"{name} 缺少必要列: {{{req_col}}}")

    merged = pd.merge(lgb[["date", "code", "pred_lgb"]], gru[["date", "code", "pred_gru"]], on=["date", "code"], how="inner")
    _assert_no_dupe(merged, ["date", "code"], "predict_meta_ridge")
    merged = normalize_by_date(
        merged,
        cols=["pred_lgb", "pred_gru"],
        date_col="date",
        mode=cfg.get("norm_mode", "zscore"),
        eps=float(cfg.get("norm_eps", 1e-6)),
    )

    coef = np.array(cfg["coef"], dtype=float)
    intercept = float(cfg["intercept"])
    X = merged[["pred_lgb", "pred_gru"]].values
    merged["final_score"] = X.dot(coef) + intercept

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if out_path.lower().endswith(".parquet"):
        merged[["date", "code", "final_score"]].to_parquet(out_path, index=False)
    else:
        merged[["date", "code", "final_score"]].to_csv(out_path, index=False)

    print(f"[meta_ridge] saved final_score -> {out_path}, rows={len(merged)}")
    print(merged.head(5))
    return merged



