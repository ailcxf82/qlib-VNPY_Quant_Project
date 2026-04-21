"""单测：``factor_validation.checks.marginal_check``（残差 IC 法）。

覆盖场景：
1. 独立新信号 + baseline 与 label 部分相关 → residual IC 高 → PASS
2. 候选因子 == baseline 预测（完全冗余）→ residual IC ≈ 0 → FAIL
3. 零信号（candidate 与 label/baseline 独立）→ FAIL
4. 负向候选 + allow_negative 切换行为
5. 缺 baseline_prediction_parquet → raise
6. 缺 label_parquet → raise
7. 缺必填配置 → raise
8. 有效日数不足 → raise
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from factor_lab.exporters.schema import CandidateFactorPackage
from factor_validation.checks.base import CheckContext
from factor_validation.checks.marginal_check import MarginalCheck


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
        date_range=(date(2025, 1, 2), date(2025, 12, 31)),
        lab_metrics={},
        parent_loop=None,
        created_at=pd.Timestamp("2026-04-19", tz="UTC").to_pydatetime(),
        lab_run_id="lab-marg-run-00001",
    )


def _mk_panel(arr: np.ndarray, instruments: list[str], name: str) -> pd.DataFrame:
    T, _ = arr.shape
    dates = pd.bdate_range("2025-01-02", periods=T)
    idx = pd.MultiIndex.from_product(
        [dates, instruments], names=["datetime", "instrument"]
    )
    return pd.DataFrame({name: arr.reshape(-1)}, index=idx).astype("float64")


def _ctx(label_path: Path, baseline_path: Path | None) -> CheckContext:
    return CheckContext(
        universe="csi300",
        oos_window=(date(2025, 1, 2), date(2025, 12, 31)),
        benchmark="SH000300",
        project_root=Path("."),
        label_parquet=label_path,
        baseline_prediction_parquet=baseline_path,
    )


def _default_config(**overrides) -> dict:
    cfg = {
        "min_residual_rank_ic": 0.02,
        "min_residual_ic_ir": 0.2,
    }
    cfg.update(overrides)
    return cfg


def _write_triplet(
    tmp_path: Path,
    factor: np.ndarray,
    label: np.ndarray,
    baseline: np.ndarray,
    instruments: list[str],
    tag: str,
) -> tuple[CandidateFactorPackage, Path, Path]:
    cand_df = _mk_panel(factor, instruments, "f")
    cand = _stub_candidate(tmp_path, "f", cand_df, tag)
    lp = tmp_path / f"label_{tag}.parquet"
    _mk_panel(label, instruments, "label").to_parquet(lp)
    bp = tmp_path / f"baseline_{tag}.parquet"
    _mk_panel(baseline, instruments, "prediction").to_parquet(bp)
    return cand, lp, bp


# --------------------------------------------------- 独立新信号 → PASS


def test_independent_signal_passes(tmp_path: Path) -> None:
    T, N = 120, 60
    rng = np.random.default_rng(42)
    # baseline 只解释 label 的一部分，candidate 正交且与 label 强相关
    cand = rng.standard_normal(size=(T, N))
    other = rng.standard_normal(size=(T, N))
    noise = rng.normal(0, 0.01, size=(T, N))
    label = 0.015 * cand + 0.015 * other + noise
    baseline = 0.015 * other + rng.normal(0, 0.005, size=(T, N))
    instruments = [f"S{i:03d}" for i in range(N)]

    c, lp, bp = _write_triplet(tmp_path, cand, label, baseline, instruments, "indep")
    result = MarginalCheck().run(c, _ctx(lp, bp), _default_config())
    assert result.passed, result.detail
    assert result.detail["residual_rank_ic"] > 0.02
    assert result.detail["residual_ic_ir"] > 0.2
    # baseline 与 candidate 不重叠
    assert abs(result.detail["baseline_overlap_ic"]) < 0.1
    assert result.detail["n_valid_days"] >= 80
    assert 0.0 < result.score <= 1.0


# --------------------------------------------------- 冗余信号 (cand = baseline) → FAIL


def test_redundant_signal_fails(tmp_path: Path) -> None:
    T, N = 120, 60
    rng = np.random.default_rng(7)
    base = rng.standard_normal(size=(T, N))
    noise = rng.normal(0, 0.01, size=(T, N))
    label = 0.02 * base + noise
    baseline = base.copy()  # baseline == candidate（完全冗余）
    cand = base + rng.normal(0, 0.0005, size=(T, N))  # 极小扰动，保证几乎相同 rank
    instruments = [f"S{i:03d}" for i in range(N)]

    c, lp, bp = _write_triplet(tmp_path, cand, label, baseline, instruments, "redun")
    result = MarginalCheck().run(c, _ctx(lp, bp), _default_config())
    assert not result.passed
    # label_rank_ic 应远高于 residual_rank_ic
    assert result.detail["label_rank_ic"] > 0.1
    assert abs(result.detail["residual_rank_ic"]) < 0.05
    # overlap 应非常高
    assert result.detail["baseline_overlap_ic"] > 0.9


# --------------------------------------------------- 零信号 → FAIL


def test_zero_signal_fails(tmp_path: Path) -> None:
    T, N = 120, 60
    rng = np.random.default_rng(3)
    cand = rng.standard_normal(size=(T, N))
    label = rng.standard_normal(size=(T, N)) * 0.01
    baseline = rng.standard_normal(size=(T, N)) * 0.01
    instruments = [f"S{i:03d}" for i in range(N)]

    c, lp, bp = _write_triplet(tmp_path, cand, label, baseline, instruments, "zero")
    result = MarginalCheck().run(c, _ctx(lp, bp), _default_config())
    assert not result.passed
    assert abs(result.detail["residual_rank_ic"]) < 0.05


# --------------------------------------------------- 负向 + allow_negative


def test_negative_factor_with_allow_negative(tmp_path: Path) -> None:
    T, N = 120, 60
    rng = np.random.default_rng(11)
    cand = rng.standard_normal(size=(T, N))
    other = rng.standard_normal(size=(T, N))
    noise = rng.normal(0, 0.01, size=(T, N))
    # candidate 与 label 反向相关；baseline 解释 other
    label = -0.015 * cand + 0.015 * other + noise
    baseline = 0.015 * other + rng.normal(0, 0.005, size=(T, N))
    instruments = [f"S{i:03d}" for i in range(N)]

    c, lp, bp = _write_triplet(tmp_path, cand, label, baseline, instruments, "neg")

    r1 = MarginalCheck().run(
        c, _ctx(lp, bp), _default_config(allow_negative=False)
    )
    assert not r1.passed
    assert r1.detail["residual_rank_ic"] < 0

    r2 = MarginalCheck().run(
        c, _ctx(lp, bp), _default_config(allow_negative=True)
    )
    assert r2.passed, r2.detail
    # raw 仍为负
    assert r2.detail["residual_rank_ic"] < 0


# --------------------------------------------------- 缺 baseline → raise


def test_missing_baseline_raises(tmp_path: Path) -> None:
    T, N = 30, 10
    rng = np.random.default_rng(0)
    cand = rng.standard_normal(size=(T, N))
    instruments = [f"S{i:03d}" for i in range(N)]
    cand_df = _mk_panel(cand, instruments, "f")
    c = _stub_candidate(tmp_path, "f", cand_df, "mnb")
    lp = tmp_path / "l.parquet"
    _mk_panel(
        rng.standard_normal(size=(T, N)) * 0.01, instruments, "label"
    ).to_parquet(lp)

    with pytest.raises(ValueError, match="baseline_prediction_parquet"):
        MarginalCheck().run(c, _ctx(lp, None), _default_config())


# --------------------------------------------------- 缺 label → raise


def test_missing_label_raises(tmp_path: Path) -> None:
    T, N = 30, 10
    rng = np.random.default_rng(0)
    cand = rng.standard_normal(size=(T, N))
    instruments = [f"S{i:03d}" for i in range(N)]
    cand_df = _mk_panel(cand, instruments, "f")
    c = _stub_candidate(tmp_path, "f", cand_df, "mnl")
    bp = tmp_path / "b.parquet"
    _mk_panel(
        rng.standard_normal(size=(T, N)) * 0.01, instruments, "prediction"
    ).to_parquet(bp)

    ctx = CheckContext(
        universe="csi300",
        oos_window=(date(2025, 1, 2), date(2025, 12, 31)),
        benchmark="SH000300",
        project_root=Path("."),
        label_parquet=None,
        baseline_prediction_parquet=bp,
    )
    with pytest.raises(ValueError, match="label_parquet"):
        MarginalCheck().run(c, ctx, _default_config())


# --------------------------------------------------- 缺必填配置 → raise


def test_missing_required_config_raises(tmp_path: Path) -> None:
    T, N = 30, 10
    rng = np.random.default_rng(0)
    cand = rng.standard_normal(size=(T, N))
    label = rng.standard_normal(size=(T, N)) * 0.01
    baseline = rng.standard_normal(size=(T, N)) * 0.01
    instruments = [f"S{i:03d}" for i in range(N)]
    c, lp, bp = _write_triplet(
        tmp_path, cand, label, baseline, instruments, "mcfg"
    )

    with pytest.raises(ValueError, match="min_residual_rank_ic"):
        MarginalCheck().run(c, _ctx(lp, bp), {"min_residual_ic_ir": 0.2})
    with pytest.raises(ValueError, match="min_residual_ic_ir"):
        MarginalCheck().run(c, _ctx(lp, bp), {"min_residual_rank_ic": 0.02})


# --------------------------------------------------- 有效日数不足 → raise


def test_insufficient_days_raises(tmp_path: Path) -> None:
    T, N = 10, 40
    rng = np.random.default_rng(1)
    cand = rng.standard_normal(size=(T, N))
    label = 0.02 * cand + rng.normal(0, 0.01, size=(T, N))
    baseline = rng.standard_normal(size=(T, N)) * 0.005
    instruments = [f"S{i:03d}" for i in range(N)]
    c, lp, bp = _write_triplet(
        tmp_path, cand, label, baseline, instruments, "short"
    )

    with pytest.raises(ValueError, match="有效日数不足"):
        MarginalCheck().run(c, _ctx(lp, bp), _default_config())
