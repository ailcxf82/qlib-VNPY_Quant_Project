"""单测：``scripts.validate.prepare_baseline_prediction``。

覆盖：
1. MultiIndex schema + 自动匹配 pred_ensemble 列 → 正常输出
2. Flat schema + 自动匹配 date/instrument/pred 列 → 正常输出
3. 显式 --prediction-col / --date-col / --instrument-col 覆盖
4. 切窗（--start/--end）
5. 重复 (datetime, instrument) 自动去重（保留最后）
6. 空预测列 → raise
7. synthetic-zero 模式：从 label parquet 生成全零 baseline
8. 切窗后为空 → raise
9. 缺 src 且没 synthetic → raise
10. CLI: main() 端到端
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.validate import prepare_baseline_prediction as mod


# ---------------------------------------------------------------- fixtures


def _mk_flat(
    tmp_path: Path,
    pred_col: str = "pred_ensemble",
    date_col: str = "datetime",
    inst_col: str = "instrument",
    n_days: int = 40,
    n_inst: int = 10,
    seed: int = 0,
) -> Path:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-02", periods=n_days)
    instruments = [f"S{i:03d}" for i in range(n_inst)]
    rows = []
    for d in dates:
        for s in instruments:
            rows.append((d, s, float(rng.normal())))
    df = pd.DataFrame(rows, columns=[date_col, inst_col, pred_col])
    p = tmp_path / f"flat_{pred_col}.parquet"
    df.to_parquet(p)
    return p


def _mk_multiindex(
    tmp_path: Path,
    pred_col: str = "pred_ensemble",
    n_days: int = 40,
    n_inst: int = 10,
    seed: int = 0,
) -> Path:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-02", periods=n_days)
    instruments = [f"S{i:03d}" for i in range(n_inst)]
    idx = pd.MultiIndex.from_product(
        [dates, instruments], names=["datetime", "instrument"]
    )
    df = pd.DataFrame(
        {pred_col: rng.normal(size=len(idx))}, index=idx
    ).astype("float64")
    p = tmp_path / f"mi_{pred_col}.parquet"
    df.to_parquet(p)
    return p


# ---------------------------------------------------------------- tests


def test_multiindex_autodetect_prediction(tmp_path: Path) -> None:
    src = _mk_multiindex(tmp_path, "pred_ensemble", n_days=40, n_inst=10, seed=1)
    out_path = tmp_path / "out.parquet"
    mod.prepare_baseline(
        src=src,
        out_path=out_path,
        prediction_col=None,
        date_col=None,
        instrument_col=None,
        start=None,
        end=None,
    )
    df = pd.read_parquet(out_path)
    assert list(df.columns) == ["prediction"]
    assert df.index.names == ["datetime", "instrument"]
    assert len(df) == 400
    assert df["prediction"].dtype == np.float64


def test_flat_autodetect(tmp_path: Path) -> None:
    src = _mk_flat(tmp_path, "final", "date", "code", n_days=30, n_inst=8)
    out = tmp_path / "flat_out.parquet"
    mod.prepare_baseline(
        src=src,
        out_path=out,
        prediction_col=None,
        date_col=None,
        instrument_col=None,
        start=None,
        end=None,
    )
    df = pd.read_parquet(out)
    assert list(df.columns) == ["prediction"]
    assert df.index.names == ["datetime", "instrument"]
    assert len(df) == 240


def test_explicit_col_override(tmp_path: Path) -> None:
    # 用不在候选列表里的列名，必须显式指定
    src = _mk_flat(
        tmp_path, "my_custom_pred", "trade_dt", "stock_id", n_days=20, n_inst=5
    )
    out = tmp_path / "o.parquet"
    mod.prepare_baseline(
        src=src,
        out_path=out,
        prediction_col="my_custom_pred",
        date_col="trade_dt",
        instrument_col="stock_id",
        start=None,
        end=None,
    )
    df = pd.read_parquet(out)
    assert len(df) == 100
    assert list(df.columns) == ["prediction"]


def test_window_slicing(tmp_path: Path) -> None:
    src = _mk_multiindex(tmp_path, "prediction", n_days=60, n_inst=5)
    out = tmp_path / "o.parquet"
    mod.prepare_baseline(
        src=src,
        out_path=out,
        prediction_col=None,
        date_col=None,
        instrument_col=None,
        start="2025-02-01",
        end="2025-02-28",
    )
    df = pd.read_parquet(out)
    dt = df.index.get_level_values("datetime")
    assert dt.min() >= pd.Timestamp("2025-02-01")
    assert dt.max() <= pd.Timestamp("2025-02-28") + pd.Timedelta(days=1)


def test_duplicate_rows_deduped(tmp_path: Path) -> None:
    dates = pd.bdate_range("2025-01-02", periods=5)
    instruments = ["S001", "S002"]
    idx = pd.MultiIndex.from_product(
        [dates, instruments], names=["datetime", "instrument"]
    )
    df = pd.DataFrame({"pred_ensemble": np.arange(10, dtype=float)}, index=idx)
    # append same 5 rows with different values (simulate fold overlap)
    extra = df.iloc[:5].copy()
    extra["pred_ensemble"] = extra["pred_ensemble"] * 10 + 999
    combined = pd.concat([df, extra])
    src = tmp_path / "dup.parquet"
    combined.to_parquet(src)

    out = tmp_path / "o.parquet"
    mod.prepare_baseline(
        src=src,
        out_path=out,
        prediction_col=None,
        date_col=None,
        instrument_col=None,
        start=None,
        end=None,
    )
    result = pd.read_parquet(out)
    # 应只剩 10 行（去重保留 last）
    assert len(result) == 10
    # 验证"保留最后"：前 5 行 value 应是 extra 的（带 999）
    for i in range(5):
        assert result.iloc[i]["prediction"] >= 999


def test_missing_prediction_col_raises(tmp_path: Path) -> None:
    dates = pd.bdate_range("2025-01-02", periods=5)
    instruments = ["S001"]
    idx = pd.MultiIndex.from_product(
        [dates, instruments], names=["datetime", "instrument"]
    )
    df = pd.DataFrame({"some_other": np.arange(5, dtype=float)}, index=idx)
    src = tmp_path / "nopred.parquet"
    df.to_parquet(src)
    with pytest.raises(ValueError, match="prediction"):
        mod.prepare_baseline(
            src=src,
            out_path=tmp_path / "o.parquet",
            prediction_col=None,
            date_col=None,
            instrument_col=None,
            start=None,
            end=None,
        )


def test_synthetic_zero_mode(tmp_path: Path) -> None:
    dates = pd.bdate_range("2025-01-02", periods=30)
    instruments = ["S001", "S002", "S003"]
    idx = pd.MultiIndex.from_product(
        [dates, instruments], names=["datetime", "instrument"]
    )
    label = pd.DataFrame(
        {"label": np.random.default_rng(0).normal(size=len(idx))}, index=idx
    ).astype({"label": "float64"})
    lp = tmp_path / "labels.parquet"
    label.to_parquet(lp)

    out = tmp_path / "zero_baseline.parquet"
    mod.prepare_baseline(
        src=None,
        out_path=out,
        prediction_col=None,
        date_col=None,
        instrument_col=None,
        start=None,
        end=None,
        synthetic_zero_from=lp,
    )
    df = pd.read_parquet(out)
    assert list(df.columns) == ["prediction"]
    assert (df["prediction"] == 0.0).all()
    assert len(df) == len(idx)


def test_empty_after_window_raises(tmp_path: Path) -> None:
    src = _mk_multiindex(tmp_path, "pred_ensemble", n_days=5, n_inst=3)
    with pytest.raises(RuntimeError, match="为空"):
        mod.prepare_baseline(
            src=src,
            out_path=tmp_path / "o.parquet",
            prediction_col=None,
            date_col=None,
            instrument_col=None,
            start="2030-01-01",
            end="2030-12-31",
        )


def test_missing_src_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--src"):
        mod.prepare_baseline(
            src=None,
            out_path=tmp_path / "o.parquet",
            prediction_col=None,
            date_col=None,
            instrument_col=None,
            start=None,
            end=None,
        )


def test_cli_main(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = _mk_multiindex(tmp_path, "pred_ensemble", n_days=20, n_inst=5)
    out = tmp_path / "cli_out.parquet"
    ret = mod.main(
        [
            "--src",
            str(src),
            "--out",
            str(out),
            "--start",
            "2025-01-02",
            "--end",
            "2025-01-20",
        ]
    )
    assert ret == 0
    printed = capsys.readouterr().out.strip()
    assert str(out) in printed
    df = pd.read_parquet(out)
    assert list(df.columns) == ["prediction"]
    assert not df.empty
