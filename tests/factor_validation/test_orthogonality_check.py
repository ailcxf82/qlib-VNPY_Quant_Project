"""单测：``factor_validation.checks.orthogonality_check``。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from factor_lab.exporters.schema import CandidateFactorPackage
from factor_validation.checks.base import CheckContext
from factor_validation.checks.orthogonality_check import OrthogonalityCheck


_HEX_TAG_DEFAULT = "abcdef12"


def _stub_candidate(
    tmp_path: Path, name: str, df: pd.DataFrame, factor_id: str | None = None
) -> CandidateFactorPackage:
    tag = (factor_id or _HEX_TAG_DEFAULT).lower()
    hex_tag = "".join(c for c in tag if c in "0123456789abcdef")
    if len(hex_tag) < 8:
        hex_tag = (hex_tag + _HEX_TAG_DEFAULT)[:8]
    full_id = f"manual_{name}_{hex_tag}"
    values_path = tmp_path / f"{full_id}.parquet"
    code_path = tmp_path / f"{full_id}.py"
    df[[name]].astype("float64").to_parquet(values_path)
    code_path.write_text(
        '"""stub"""\n\ndef calculate(df):\n    return df\n', encoding="utf-8"
    )
    return CandidateFactorPackage(
        factor_id=full_id,
        name=name,
        source="manual",
        hypothesis="test",
        formulation="test",
        code_path=code_path,
        values_path=values_path,
        universe="csi300",
        date_range=(date(2025, 1, 2), date(2025, 3, 31)),
        lab_metrics={},
        parent_loop=None,
        created_at=pd.Timestamp("2026-04-19", tz="UTC").to_pydatetime(),
        lab_run_id="lab-ortho-run-0001",
    )


def _mk_df(arr: np.ndarray, col: str, instruments: list[str]) -> pd.DataFrame:
    T, N = arr.shape
    dates = pd.bdate_range("2025-01-02", periods=T)
    idx = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    return pd.DataFrame({col: arr.reshape(-1)}, index=idx).astype("float64")


def _ctx_with_ref(ref_path: Path) -> CheckContext:
    return CheckContext(
        universe="csi300",
        oos_window=(date(2025, 1, 2), date(2025, 12, 31)),
        benchmark="SH000300",
        project_root=Path("."),
        reference_factors_parquet=ref_path,
    )


# ---------------------------------------------------------- 高相关 → FAIL


def test_high_correlation_fails(tmp_path: Path) -> None:
    T, N = 40, 20
    rng = np.random.default_rng(0)
    a = rng.standard_normal(size=(T, N))
    # b = a + 小噪声 → rank corr 近 1
    b = a + rng.normal(0, 0.01, size=(T, N))
    instruments = [f"S{i:03d}" for i in range(N)]

    cand_df = _mk_df(a, "factor_a", instruments)
    cand = _stub_candidate(tmp_path, "factor_a", cand_df, "corr_test_cand")

    ref_df = _mk_df(b, "factor_b", instruments)
    ref_path = tmp_path / "reference.parquet"
    ref_df.to_parquet(ref_path)

    result = OrthogonalityCheck().run(
        cand,
        _ctx_with_ref(ref_path),
        {"max_abs_corr": 0.50},
    )
    assert not result.passed
    assert result.detail["max_abs_corr"] > 0.95
    assert result.detail["n_reference_cols"] == 1


# ---------------------------------------------------------- 独立 → PASS


def test_orthogonal_passes(tmp_path: Path) -> None:
    T, N = 60, 30
    rng = np.random.default_rng(42)
    a = rng.standard_normal(size=(T, N))
    b = rng.standard_normal(size=(T, N))
    instruments = [f"S{i:03d}" for i in range(N)]

    cand_df = _mk_df(a, "factor_a", instruments)
    cand = _stub_candidate(tmp_path, "factor_a", cand_df, "indep_test_cand")

    ref_df = _mk_df(b, "factor_b", instruments)
    ref_path = tmp_path / "reference.parquet"
    ref_df.to_parquet(ref_path)

    result = OrthogonalityCheck().run(
        cand,
        _ctx_with_ref(ref_path),
        {"max_abs_corr": 0.50},
    )
    assert result.passed, result.detail
    assert result.detail["max_abs_corr"] < 0.30


# ---------------------------------------------------------- 参照集空（first-factor）


def test_empty_reference_passes(tmp_path: Path) -> None:
    T, N = 20, 10
    a = np.random.default_rng(0).standard_normal(size=(T, N))
    instruments = [f"S{i:03d}" for i in range(N)]
    cand_df = _mk_df(a, "factor_a", instruments)
    cand = _stub_candidate(tmp_path, "factor_a", cand_df, "empty_ref_cand")

    # 空 DataFrame 但有索引（用 factor_a 列先写，再删列）
    ref_df = _mk_df(a, "factor_a", instruments).iloc[:, :0]
    ref_path = tmp_path / "reference.parquet"
    ref_df.to_parquet(ref_path)

    result = OrthogonalityCheck().run(
        cand,
        _ctx_with_ref(ref_path),
        {"max_abs_corr": 0.50},
    )
    assert result.passed
    assert result.score == 1.0
    assert result.detail["n_reference_cols"] == 0


# ---------------------------------------------------------- 参照集含候选自己


def test_reference_containing_self_is_skipped(tmp_path: Path) -> None:
    T, N = 40, 15
    rng = np.random.default_rng(1)
    a = rng.standard_normal(size=(T, N))
    instruments = [f"S{i:03d}" for i in range(N)]
    cand_df = _mk_df(a, "factor_a", instruments)
    cand = _stub_candidate(tmp_path, "factor_a", cand_df, "self_ref_cand")

    # 参照只有 factor_a，应被 drop，等效于空参照集
    ref_df = _mk_df(a, "factor_a", instruments)
    ref_path = tmp_path / "reference.parquet"
    ref_df.to_parquet(ref_path)

    result = OrthogonalityCheck().run(
        cand,
        _ctx_with_ref(ref_path),
        {"max_abs_corr": 0.50},
    )
    assert result.detail["n_reference_cols"] == 0
    assert result.passed


# ---------------------------------------------------------- context 未设 → raise


def test_missing_reference_context_raises(tmp_path: Path) -> None:
    T, N = 20, 10
    a = np.random.default_rng(0).standard_normal(size=(T, N))
    instruments = [f"S{i:03d}" for i in range(N)]
    cand_df = _mk_df(a, "factor_a", instruments)
    cand = _stub_candidate(tmp_path, "factor_a", cand_df, "miss_ctx_cand")

    ctx = CheckContext(
        universe="csi300",
        oos_window=(date(2025, 1, 2), date(2025, 12, 31)),
        benchmark="SH000300",
        project_root=Path("."),
    )
    with pytest.raises(ValueError, match="reference_factors_parquet"):
        OrthogonalityCheck().run(cand, ctx, {"max_abs_corr": 0.50})


# ---------------------------------------------------------- 多参照列：最大值决策


def test_max_abs_corr_picked_across_refs(tmp_path: Path) -> None:
    T, N = 40, 20
    rng = np.random.default_rng(7)
    a = rng.standard_normal(size=(T, N))
    instruments = [f"S{i:03d}" for i in range(N)]

    cand_df = _mk_df(a, "factor_a", instruments)
    cand = _stub_candidate(tmp_path, "factor_a", cand_df, "multi_ref_cand")

    # 参照矩阵同时包含：(1) a 的近似，(2) 独立噪声
    b_similar = a + rng.normal(0, 0.01, size=(T, N))
    b_indep = rng.standard_normal(size=(T, N))
    dates = pd.bdate_range("2025-01-02", periods=T)
    idx = pd.MultiIndex.from_product(
        [dates, instruments], names=["datetime", "instrument"]
    )
    ref_df = pd.DataFrame(
        {
            "b_similar": b_similar.reshape(-1),
            "b_indep": b_indep.reshape(-1),
        },
        index=idx,
    ).astype("float64")
    ref_path = tmp_path / "reference.parquet"
    ref_df.to_parquet(ref_path)

    result = OrthogonalityCheck().run(
        cand,
        _ctx_with_ref(ref_path),
        {"max_abs_corr": 0.50},
    )
    # max_abs_corr 应由 b_similar 决定，非 b_indep
    assert not result.passed
    per = result.detail["per_ref_corr"]
    assert set(per) == {"b_similar", "b_indep"}
    assert abs(per["b_similar"]) > abs(per["b_indep"])
