"""``utils.meta_oof_builder.build_meta_oof`` Phase 1 P1-3 回归测试。

覆盖的不变量：
- GRU 预测有 NaN 时，**不**再整行 dropna；输出行数应等于 LGB 行数减去仅
  pred_lgb 缺失的行（典型场景：0）。
- 输出 schema 含 ``gru_coverage`` 列（0/1）。
- pred_gru 缺失行的 ``gru_coverage`` = 0；其余 = 1。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.meta_oof_builder import build_meta_oof


def _write_oof(path, df: pd.DataFrame):
    df.to_parquet(str(path), index=False)


def _make_oof(tmp_path, gru_nan_ratio: float = 0.3):
    """构造一份最小 OOF 三元组（lgb / gru / y），按要求让部分 pred_gru = NaN。"""
    rng = np.random.default_rng(0)
    n = 200
    n_dates = 5
    dates = pd.to_datetime([f"2024-01-{i+1:02d}" for i in range(n_dates)])
    rows = []
    for d in dates:
        for k in range(n // n_dates):
            rows.append((d, f"S{k:03d}"))
    df_idx = pd.DataFrame(rows, columns=["date", "code"])
    df_idx["fold"] = 0

    lgb = df_idx.copy()
    lgb["pred_lgb"] = rng.normal(size=len(lgb)).astype(np.float32)

    gru = df_idx.copy()
    gru["pred_gru"] = rng.normal(size=len(gru)).astype(np.float32)
    nan_mask = rng.random(len(gru)) < gru_nan_ratio
    gru.loc[nan_mask, "pred_gru"] = np.nan

    y = df_idx[["date", "code"]].copy()
    y["y"] = rng.normal(size=len(y)).astype(np.float32)

    p_lgb = tmp_path / "oof_lgb.parquet"
    p_gru = tmp_path / "oof_gru.parquet"
    p_y = tmp_path / "y.parquet"
    _write_oof(p_lgb, lgb)
    _write_oof(p_gru, gru)
    _write_oof(p_y, y)
    return str(p_lgb), str(p_gru), str(p_y), int(nan_mask.sum()), len(df_idx)


def test_no_dropna_when_gru_missing(tmp_path):
    """GRU 30% NaN 时，输出行数应等于 LGB 行数（不再被砍）。"""
    p_lgb, p_gru, p_y, n_nan, n_total = _make_oof(tmp_path, gru_nan_ratio=0.3)
    out = tmp_path / "meta_oof.parquet"
    df = build_meta_oof(p_lgb, p_gru, p_y, str(out))
    assert len(df) == n_total, (
        f"P1-3 回归：build_meta_oof 行数 {len(df)} 应等于原始 {n_total}，不再因 GRU NaN 而剔除"
    )
    assert n_nan > 0, "fixture 未制造任何 GRU NaN，测试不可信"


def test_gru_coverage_column_present(tmp_path):
    """schema 必须含 gru_coverage；GRU NaN 行 coverage=0，其余=1。"""
    p_lgb, p_gru, p_y, n_nan, n_total = _make_oof(tmp_path, gru_nan_ratio=0.4)
    out = tmp_path / "meta_oof.parquet"
    df = build_meta_oof(p_lgb, p_gru, p_y, str(out))
    assert "gru_coverage" in df.columns, "缺少 gru_coverage 列"
    # gru_coverage 必须只有 0/1
    assert set(df["gru_coverage"].unique()).issubset({0, 1}), df["gru_coverage"].unique()
    # gru_coverage=0 当且仅当 pred_gru 归一化前/后为 NaN（归一化后 NaN 也保留）
    cov0 = (df["gru_coverage"] == 0).sum()
    assert cov0 == n_nan, f"gru_coverage=0 行数 {cov0} 应等于原始 NaN 行数 {n_nan}"
