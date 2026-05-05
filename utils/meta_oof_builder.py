"""
构建用于 Stacking 的 OOF 数据（方案 A：LGBM + GRU + Ridge）。

功能：
  - 读取 lgb/gru 的 OOF 预测（含 date, code, fold, pred_xxx）
  - 读取标签（y），与 OOF 用 (date, code) 且校验 fold 一致的方式严格对齐
  - 过滤掉任一模型预测为 NaN 的样本（典型场景：GRU 序列不足 T=60）
  - 断言无重复 key，fold 一致
  - 输出统一的 meta_oof.parquet/csv，列为 [date, code, fold, y, pred_lgb, pred_gru]

使用示例：
```python
from utils.meta_oof_builder import build_meta_oof
build_meta_oof(
    lgb_oof_path="data/oof/20250131/oof_lgb.parquet",
    gru_oof_path="data/oof/20250131/oof_gru.parquet",
    y_path="data/oof/20250131/y.parquet",   # 需包含 date, code, y
    out_path="data/oof/20250131/meta_oof.parquet",
)
```
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from utils.normalize import normalize_by_date

REQUIRED_COLS_LGB = {"date", "code", "fold", "pred_lgb"}
REQUIRED_COLS_GRU = {"date", "code", "fold", "pred_gru"}
REQUIRED_COLS_Y = {"date", "code", "y"}


def _read_oof(path: str, required_cols: set[str]) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path} 缺少必要列: {missing}")
    return df[list(required_cols)].copy()


def build_meta_oof(
    lgb_oof_path: str,
    gru_oof_path: str,
    y_path: Optional[str],
    out_path: str = "meta_oof.parquet",
    *,
    norm_mode: str = "zscore",
    norm_eps: float = 1e-6,
):
    """
    构建 Meta OOF 数据，输出包含 [date, code, fold, y, pred_lgb, pred_gru] 的文件。
    """
    lgb = _read_oof(lgb_oof_path, REQUIRED_COLS_LGB).rename(columns={"pred_lgb": "pred_lgb"})
    gru = _read_oof(gru_oof_path, REQUIRED_COLS_GRU).rename(columns={"pred_gru": "pred_gru"})

    # 1) 用 (date, code) join；禁止假设数组顺序一致
    on_cols = ["date", "code"]
    merged = pd.merge(lgb, gru, on=on_cols, how="inner", suffixes=("_lgb", "_gru"))

    # 2) 断言无重复 key
    if merged.duplicated(subset=on_cols).any():
        dup = merged[merged.duplicated(subset=on_cols, keep=False)]
        raise ValueError(f"发现重复 key (date,code)，样本数={len(dup)}，请检查 OOF 生成过程")

    # 3) 断言 fold 一致
    if not (merged["fold_lgb"] == merged["fold_gru"]).all():
        mismatch = merged[merged["fold_lgb"] != merged["fold_gru"]]
        raise ValueError(f"fold 不一致的样本数={len(mismatch)}，请检查 OOF 生成过程")
    merged["fold"] = merged["fold_lgb"]
    merged = merged.drop(columns=["fold_lgb", "fold_gru"])

    # 4) Phase 1 P1-3：不再整行 dropna。
    # 旧行为：``merged.dropna(subset=["pred_lgb", "pred_gru"])`` 把 GRU
    # 序列不足导致 ``pred_gru=NaN`` 的样本整行剔除。这等价于"凡是 GRU
    # 看不到的样本，LGB 也看不到了"，导致下游 Ridge 训练样本严重偏向
    # LGB 高覆盖区，并把 GRU 系数压低。
    # 新行为：仅过滤 ``pred_lgb`` 缺失的行（LGB 是 100% 覆盖基线模型，
    # 缺它则该行没有任何信号），保留 ``pred_gru`` 缺失行；同时新增
    # ``gru_coverage`` 列（0/1）记录覆盖情况，供 meta_ridge 训练阶段做
    # 必要的后处理（fillna(0) → 等价于"无 GRU 信号"）。
    before = len(merged)
    merged = merged.dropna(subset=["pred_lgb"])
    after = len(merged)
    if after < before:
        print(f"[build_meta_oof] 过滤 pred_lgb 缺失样本 {before - after} 条")
    merged["gru_coverage"] = merged["pred_gru"].notna().astype(np.int8)
    gru_missing = int((merged["gru_coverage"] == 0).sum())
    if gru_missing > 0:
        ratio = gru_missing / max(1, len(merged))
        print(
            f"[build_meta_oof] pred_gru 缺失样本 {gru_missing} 条 "
            f"(占比 {ratio:.2%})；保留行并打 gru_coverage=0 标记"
        )

    # 5) 加载标签
    if y_path is not None:
        y_df = _read_oof(y_path, REQUIRED_COLS_Y)
        merged = pd.merge(merged, y_df, on=on_cols, how="inner")
        # 再次断言无重复
        if merged.duplicated(subset=on_cols).any():
            dup = merged[merged.duplicated(subset=on_cols, keep=False)]
            raise ValueError(f"合并标签后发现重复 key (date,code)，样本数={len(dup)}")
    else:
        if "y" not in merged.columns:
            raise ValueError("未提供 y_path，且合并结果中不存在 y 列，无法构建 meta OOF")

    # 6) 最终列顺序（在 pred 列之后追加 gru_coverage）
    merged = merged[["date", "code", "fold", "y", "pred_lgb", "pred_gru", "gru_coverage"]]

    # 7) 按日横截面标准化（仅作用于预测列；NaN 自动被 pandas mean/std 跳过）
    merged = normalize_by_date(merged, cols=["pred_lgb", "pred_gru"], date_col="date", mode=norm_mode, eps=norm_eps)

    # 8) 输出 parquet/csv
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if out_path.lower().endswith(".parquet"):
        merged.to_parquet(out_path, index=False)
    else:
        merged.to_csv(out_path, index=False)

    # 打印前5行示例
    print("[build_meta_oof] saved:", out_path, "rows:", len(merged))
    print(merged.head(5))

    return merged


