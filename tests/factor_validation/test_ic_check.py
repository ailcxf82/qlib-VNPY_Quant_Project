"""单测：``factor_validation.checks.ic_check``。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from factor_lab.exporters.schema import CandidateFactorPackage
from factor_validation.checks.base import CheckContext
from factor_validation.checks.ic_check import IcCheck


_HEX_TAG_DEFAULT = "abcdef12"


def _stub_candidate(
    tmp_path: Path, name: str, df: pd.DataFrame, factor_id: str | None = None
) -> CandidateFactorPackage:
    tag = (factor_id or _HEX_TAG_DEFAULT).lower()
    hex_tag = "".join(c for c in tag if c in "0123456789abcdef")
    if len(hex_tag) < 8:
        hex_tag = (hex_tag + _HEX_TAG_DEFAULT)[:8]
    full_id = f"manual_{name}_{hex_tag}"
    vp = tmp_path / f"{full_id}.parquet"
    cp = tmp_path / f"{full_id}.py"
    df[[name]].astype("float64").to_parquet(vp)
    cp.write_text(
        '"""stub"""\n\ndef calculate(df):\n    return df\n', encoding="utf-8"
    )
    return CandidateFactorPackage(
        factor_id=full_id,
        name=name,
        source="manual",
        hypothesis="t",
        formulation="t",
        code_path=cp,
        values_path=vp,
        universe="csi300",
        date_range=(date(2025, 1, 2), date(2025, 6, 30)),
        lab_metrics={},
        parent_loop=None,
        created_at=pd.Timestamp("2026-04-19", tz="UTC").to_pydatetime(),
        lab_run_id="lab-ic-run-00001",
    )


def _mk_series(arr: np.ndarray, instruments: list[str], name: str) -> pd.DataFrame:
    T, N = arr.shape
    dates = pd.bdate_range("2025-01-02", periods=T)
    idx = pd.MultiIndex.from_product(
        [dates, instruments], names=["datetime", "instrument"]
    )
    return pd.DataFrame({name: arr.reshape(-1)}, index=idx).astype("float64")


def _ctx(label_path: Path) -> CheckContext:
    return CheckContext(
        universe="csi300",
        oos_window=(date(2025, 1, 2), date(2025, 12, 31)),
        benchmark="SH000300",
        project_root=Path("."),
        label_parquet=label_path,
    )


# ---------------------------------------------------------- 正 IC → PASS


def test_positive_ic_passes(tmp_path: Path) -> None:
    T, N = 60, 30
    rng = np.random.default_rng(0)
    factor = rng.standard_normal(size=(T, N))
    # label = factor + 噪声 → rank IC 接近 1
    label = factor + rng.normal(0, 0.3, size=(T, N))
    instruments = [f"S{i:03d}" for i in range(N)]

    cand_df = _mk_series(factor, instruments, "f")
    cand = _stub_candidate(tmp_path, "f", cand_df, "pos_ic")

    label_df = _mk_series(label, instruments, "label")
    label_path = tmp_path / "labels.parquet"
    label_df.to_parquet(label_path)

    result = IcCheck().run(
        cand,
        _ctx(label_path),
        {"min_rank_ic": 0.03, "min_ic_ir": 0.3},
    )
    assert result.passed, result.detail
    assert result.detail["rank_ic"] > 0.5
    assert result.detail["ic_ir"] > 1.0
    assert 0.0 <= result.score <= 1.0


# ---------------------------------------------------------- 零 IC → FAIL


def test_zero_ic_fails(tmp_path: Path) -> None:
    T, N = 80, 30
    rng = np.random.default_rng(1)
    factor = rng.standard_normal(size=(T, N))
    label = rng.standard_normal(size=(T, N))
    instruments = [f"S{i:03d}" for i in range(N)]

    cand_df = _mk_series(factor, instruments, "f")
    cand = _stub_candidate(tmp_path, "f", cand_df, "zero_ic")

    label_df = _mk_series(label, instruments, "label")
    label_path = tmp_path / "labels.parquet"
    label_df.to_parquet(label_path)

    result = IcCheck().run(
        cand,
        _ctx(label_path),
        {"min_rank_ic": 0.03, "min_ic_ir": 0.3},
    )
    assert not result.passed
    assert abs(result.detail["rank_ic"]) < 0.10


# ---------------------------------------------------------- 负 IC + allow_negative → PASS


def test_negative_ic_passes_when_allowed(tmp_path: Path) -> None:
    T, N = 60, 30
    rng = np.random.default_rng(2)
    factor = rng.standard_normal(size=(T, N))
    label = -factor + rng.normal(0, 0.3, size=(T, N))
    instruments = [f"S{i:03d}" for i in range(N)]

    cand_df = _mk_series(factor, instruments, "f")
    cand = _stub_candidate(tmp_path, "f", cand_df, "neg_ic_allowed")

    label_df = _mk_series(label, instruments, "label")
    label_path = tmp_path / "labels.parquet"
    label_df.to_parquet(label_path)

    # 不允许负向 → FAIL
    r1 = IcCheck().run(
        cand,
        _ctx(label_path),
        {"min_rank_ic": 0.03, "min_ic_ir": 0.3, "allow_negative": False},
    )
    assert not r1.passed
    assert r1.detail["rank_ic"] < -0.5

    # 允许负向 → PASS
    r2 = IcCheck().run(
        cand,
        _ctx(label_path),
        {"min_rank_ic": 0.03, "min_ic_ir": 0.3, "allow_negative": True},
    )
    assert r2.passed


# ---------------------------------------------------------- 缺配置 / 缺 context


def test_missing_label_context_raises(tmp_path: Path) -> None:
    T, N = 30, 10
    factor = np.random.default_rng(0).standard_normal(size=(T, N))
    instruments = [f"S{i:03d}" for i in range(N)]
    cand_df = _mk_series(factor, instruments, "f")
    cand = _stub_candidate(tmp_path, "f", cand_df, "miss_ctx")
    ctx = CheckContext(
        universe="csi300",
        oos_window=(date(2025, 1, 2), date(2025, 12, 31)),
        benchmark="SH000300",
        project_root=Path("."),
    )
    with pytest.raises(ValueError, match="label_parquet"):
        IcCheck().run(cand, ctx, {"min_rank_ic": 0.03, "min_ic_ir": 0.3})


def test_missing_config_fields(tmp_path: Path) -> None:
    T, N = 30, 10
    factor = np.random.default_rng(0).standard_normal(size=(T, N))
    instruments = [f"S{i:03d}" for i in range(N)]
    cand_df = _mk_series(factor, instruments, "f")
    cand = _stub_candidate(tmp_path, "f", cand_df, "miss_cfg")
    label_df = _mk_series(
        np.random.default_rng(1).standard_normal(size=(T, N)), instruments, "label"
    )
    lp = tmp_path / "l.parquet"
    label_df.to_parquet(lp)

    with pytest.raises(ValueError, match="min_rank_ic"):
        IcCheck().run(cand, _ctx(lp), {"min_ic_ir": 0.3})
    with pytest.raises(ValueError, match="min_ic_ir"):
        IcCheck().run(cand, _ctx(lp), {"min_rank_ic": 0.03})


# ---------------------------------------------------------- OOS 样本过短 → raise


def test_insufficient_days_raises(tmp_path: Path) -> None:
    T, N = 5, 10  # < _MIN_VALID_DAYS=20
    rng = np.random.default_rng(3)
    factor = rng.standard_normal(size=(T, N))
    label = factor + rng.normal(0, 0.3, size=(T, N))
    instruments = [f"S{i:03d}" for i in range(N)]
    cand_df = _mk_series(factor, instruments, "f")
    cand = _stub_candidate(tmp_path, "f", cand_df, "short_oos")
    label_df = _mk_series(label, instruments, "label")
    lp = tmp_path / "l.parquet"
    label_df.to_parquet(lp)
    with pytest.raises(ValueError, match="有效日数不足"):
        IcCheck().run(cand, _ctx(lp), {"min_rank_ic": 0.03, "min_ic_ir": 0.3})
