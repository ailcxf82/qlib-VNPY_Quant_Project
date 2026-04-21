"""单测：``factor_validation.checks.backtest_check``。

覆盖：
1. 强信号 → long-short 组合高 Sharpe，PASS
2. 零信号 → FAIL
3. 负向因子 + allow_negative 切换行为
4. 缺 label_parquet → raise
5. 缺必填配置 → raise
6. OOS 样本过短 → raise
7. quantile 越界 → raise
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from factor_lab.exporters.schema import CandidateFactorPackage
from factor_validation.checks.backtest_check import BacktestCheck
from factor_validation.checks.base import CheckContext


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
        lab_run_id="lab-bt-run-00001",
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


def _default_config(**overrides) -> dict:
    cfg = {
        "quantile": 0.2,
        "min_long_short_sharpe": 1.0,
        "max_drawdown": 0.50,
        "min_annual_return": 0.0,
        "min_win_ratio": 0.0,
    }
    cfg.update(overrides)
    return cfg


# ----------------------------------------------------------- 强信号 → PASS


def test_strong_signal_passes(tmp_path: Path) -> None:
    T, N = 120, 60
    rng = np.random.default_rng(42)
    factor = rng.standard_normal(size=(T, N))
    # label = factor * 0.02 + 小噪声：强 rank 相关 + 稳定 long-short spread
    label = 0.02 * factor + rng.normal(0, 0.01, size=(T, N))
    instruments = [f"S{i:03d}" for i in range(N)]

    cand_df = _mk_series(factor, instruments, "f")
    cand = _stub_candidate(tmp_path, "f", cand_df, "bt_strong")

    label_df = _mk_series(label, instruments, "label")
    lp = tmp_path / "labels.parquet"
    label_df.to_parquet(lp)

    result = BacktestCheck().run(cand, _ctx(lp), _default_config())
    assert result.passed, result.detail
    assert result.detail["long_short_sharpe"] > 1.0
    assert result.detail["annual_return"] > 0.0
    assert result.detail["max_drawdown"] < 0.50
    assert result.detail["direction"] == 1
    assert 0.0 < result.score <= 1.0
    assert result.detail["n_valid_days"] >= 80


# ----------------------------------------------------------- 零信号 → FAIL


def test_zero_signal_fails(tmp_path: Path) -> None:
    T, N = 120, 60
    rng = np.random.default_rng(7)
    factor = rng.standard_normal(size=(T, N))
    label = rng.standard_normal(size=(T, N)) * 0.01  # factor 与 label 独立
    instruments = [f"S{i:03d}" for i in range(N)]

    cand_df = _mk_series(factor, instruments, "f")
    cand = _stub_candidate(tmp_path, "f", cand_df, "bt_zero")

    label_df = _mk_series(label, instruments, "label")
    lp = tmp_path / "labels.parquet"
    label_df.to_parquet(lp)

    result = BacktestCheck().run(cand, _ctx(lp), _default_config())
    assert not result.passed
    assert abs(result.detail["long_short_sharpe"]) < 1.0


# ----------------------------------------------------------- 负向因子 + allow_negative


def test_negative_factor_with_allow_negative(tmp_path: Path) -> None:
    T, N = 120, 60
    rng = np.random.default_rng(3)
    factor = rng.standard_normal(size=(T, N))
    label = -0.02 * factor + rng.normal(0, 0.01, size=(T, N))
    instruments = [f"S{i:03d}" for i in range(N)]

    cand_df = _mk_series(factor, instruments, "f")
    cand = _stub_candidate(tmp_path, "f", cand_df, "bt_neg")
    label_df = _mk_series(label, instruments, "label")
    lp = tmp_path / "labels.parquet"
    label_df.to_parquet(lp)

    # 不允许负向 → FAIL (Sharpe 为负)
    r1 = BacktestCheck().run(
        cand, _ctx(lp), _default_config(allow_negative=False)
    )
    assert not r1.passed
    assert r1.detail["long_short_sharpe"] < -1.0
    assert r1.detail["direction"] == 1

    # 允许负向 → PASS，direction 翻转
    r2 = BacktestCheck().run(
        cand, _ctx(lp), _default_config(allow_negative=True)
    )
    assert r2.passed, r2.detail
    assert r2.detail["direction"] == -1
    # raw sharpe 仍为负（detail 里保留原始指标）
    assert r2.detail["long_short_sharpe"] < -1.0


# ----------------------------------------------------------- 缺 label_parquet → raise


def test_missing_label_context_raises(tmp_path: Path) -> None:
    T, N = 30, 10
    factor = np.random.default_rng(0).standard_normal(size=(T, N))
    instruments = [f"S{i:03d}" for i in range(N)]
    cand_df = _mk_series(factor, instruments, "f")
    cand = _stub_candidate(tmp_path, "f", cand_df, "bt_nolbl")
    ctx = CheckContext(
        universe="csi300",
        oos_window=(date(2025, 1, 2), date(2025, 12, 31)),
        benchmark="SH000300",
        project_root=Path("."),
    )
    with pytest.raises(ValueError, match="label_parquet"):
        BacktestCheck().run(cand, ctx, _default_config())


# ----------------------------------------------------------- 缺必填配置 → raise


def test_missing_required_config_raises(tmp_path: Path) -> None:
    T, N = 30, 10
    rng = np.random.default_rng(0)
    factor = rng.standard_normal(size=(T, N))
    instruments = [f"S{i:03d}" for i in range(N)]
    cand_df = _mk_series(factor, instruments, "f")
    cand = _stub_candidate(tmp_path, "f", cand_df, "bt_miscfg")
    label_df = _mk_series(
        rng.standard_normal(size=(T, N)) * 0.01, instruments, "label"
    )
    lp = tmp_path / "l.parquet"
    label_df.to_parquet(lp)

    # 缺 quantile
    with pytest.raises(ValueError, match="quantile"):
        BacktestCheck().run(
            cand,
            _ctx(lp),
            {"min_long_short_sharpe": 1.0, "max_drawdown": 0.5},
        )
    # 缺 min_long_short_sharpe
    with pytest.raises(ValueError, match="min_long_short_sharpe"):
        BacktestCheck().run(
            cand,
            _ctx(lp),
            {"quantile": 0.2, "max_drawdown": 0.5},
        )
    # 缺 max_drawdown
    with pytest.raises(ValueError, match="max_drawdown"):
        BacktestCheck().run(
            cand,
            _ctx(lp),
            {"quantile": 0.2, "min_long_short_sharpe": 1.0},
        )


# ----------------------------------------------------------- OOS 样本过短 → raise


def test_insufficient_days_raises(tmp_path: Path) -> None:
    T, N = 10, 40  # 有效日 < min_valid_days=20
    rng = np.random.default_rng(1)
    factor = rng.standard_normal(size=(T, N))
    label = 0.02 * factor + rng.normal(0, 0.01, size=(T, N))
    instruments = [f"S{i:03d}" for i in range(N)]
    cand_df = _mk_series(factor, instruments, "f")
    cand = _stub_candidate(tmp_path, "f", cand_df, "bt_short")
    label_df = _mk_series(label, instruments, "label")
    lp = tmp_path / "l.parquet"
    label_df.to_parquet(lp)

    with pytest.raises(ValueError, match="有效交易日不足"):
        BacktestCheck().run(cand, _ctx(lp), _default_config())


# ----------------------------------------------------------- 参数越界 → raise


def test_bad_quantile_raises(tmp_path: Path) -> None:
    T, N = 30, 20
    rng = np.random.default_rng(2)
    factor = rng.standard_normal(size=(T, N))
    instruments = [f"S{i:03d}" for i in range(N)]
    cand_df = _mk_series(factor, instruments, "f")
    cand = _stub_candidate(tmp_path, "f", cand_df, "bt_bq")
    label_df = _mk_series(
        rng.standard_normal(size=(T, N)) * 0.01, instruments, "label"
    )
    lp = tmp_path / "l.parquet"
    label_df.to_parquet(lp)

    # quantile=0 → 越界（lo=1e-6）
    with pytest.raises(ValueError, match="quantile"):
        BacktestCheck().run(cand, _ctx(lp), _default_config(quantile=0.0))

    # quantile=0.8 → 越界（hi=0.5）
    with pytest.raises(ValueError, match="quantile"):
        BacktestCheck().run(cand, _ctx(lp), _default_config(quantile=0.8))

    # max_drawdown=1.5 → 越界
    with pytest.raises(ValueError, match="max_drawdown"):
        BacktestCheck().run(cand, _ctx(lp), _default_config(max_drawdown=1.5))
