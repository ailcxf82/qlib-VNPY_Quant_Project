"""单测：``factor_validation.checks.turnover_check``。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from factor_lab.exporters.schema import CandidateFactorPackage
from factor_validation.checks.base import CheckContext
from factor_validation.checks.turnover_check import TurnoverCheck


# ---------------------------------------------------------- fixtures


_HEX_TAG_DEFAULT = "abcdef12"


def _make_candidate(
    tmp_path: Path, name: str, df: pd.DataFrame, factor_id: str | None = None
) -> CandidateFactorPackage:
    # 接收外部传入的"短 id tag"（兼容旧 call site：它们把这个当标签用，不必是完整 factor_id）
    tag = (factor_id or _HEX_TAG_DEFAULT).lower()
    # 只保留 hex 字符，再 pad 到 8 位
    hex_tag = "".join(c for c in tag if c in "0123456789abcdef")
    if len(hex_tag) < 8:
        hex_tag = (hex_tag + _HEX_TAG_DEFAULT)[:8]
    factor_id = f"manual_{name}_{hex_tag}"  # <source>_<name>_<hex8-16>
    values_path = tmp_path / f"{factor_id}.parquet"
    code_path = tmp_path / f"{factor_id}.py"
    df[[name]].astype("float64").to_parquet(values_path)
    code_path.write_text(
        '"""stub"""\n\ndef calculate(df):\n    return df\n', encoding="utf-8"
    )
    return CandidateFactorPackage(
        factor_id=factor_id,
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
        lab_run_id="lab-test-run-0001",
    )


def _make_df(values: np.ndarray, instruments: list[str]) -> pd.DataFrame:
    """values: (T, N) → MultiIndex(datetime, instrument)."""
    dates = pd.bdate_range("2025-01-02", periods=values.shape[0])
    idx = pd.MultiIndex.from_product(
        [dates, instruments], names=["datetime", "instrument"]
    )
    flat = values.reshape(-1)
    return pd.DataFrame({"factor": flat}, index=idx)


def _ctx() -> CheckContext:
    # OOS 窗要覆盖测试造出来的全部日期（最多 120 bdays ≈ 6 个月）
    return CheckContext(
        universe="csi300",
        oos_window=(date(2025, 1, 2), date(2025, 12, 31)),
        benchmark="SH000300",
        project_root=Path("."),
    )


# ---------------------------------------------------------- slow factor (低换手)
#
# 新算法语义（C.6 升级）：
#   daily_rank_turnover = (1 - cross_sectional_spearman(today, prev_day)) / 2
#
#   * 完全稳定  ρ=+1 → 0.0
#   * 随机重排  ρ=0  → 0.5
#   * 完全反转  ρ=-1 → 1.0
#
# 因此：慢因子应 ≪ 0.5；随机因子应 ≈ 0.5。


def test_slow_factor_passes(tmp_path: Path) -> None:
    """慢变因子：每股**单调线性漂移** + 小噪声 → rank 缓慢单调重排。"""
    T, N = 100, 30
    rng = np.random.default_rng(42)
    t_axis = np.arange(T)
    # 初始位置彼此拉开 → rank 大体稳定；漂移速率小 → rank 偶尔交换
    init_pos = rng.uniform(-5.0, 5.0, N)
    drift_rate = rng.uniform(-0.05, 0.05, N)
    noise = rng.normal(0, 0.05, size=(T, N))
    values = init_pos[None, :] + drift_rate[None, :] * t_axis[:, None] + noise
    df = _make_df(values, [f"S{i:03d}" for i in range(N)])
    cand = _make_candidate(tmp_path, "factor", df, factor_id="slow_test")

    result = TurnoverCheck().run(
        cand,
        _ctx(),
        {"max_daily_rank_turnover": 0.15, "min_rank_autocorr": 0.60},
    )
    assert result.passed, result.detail
    # 新语义下慢因子应远低于 random walk 水平（0.5），这里 < 0.15
    assert result.detail["daily_rank_turnover"] < 0.15
    assert result.detail["rank_autocorr"] > 0.60
    assert 0.0 <= result.score <= 1.0
    assert result.detail["algorithm"].startswith(
        "(1 - cross_sectional_spearman)"
    )


# ---------------------------------------------------------- fast factor (高换手)


def test_fast_factor_fails_turnover(tmp_path: Path) -> None:
    """完全随机的因子值 → 每日独立截面 → spearman ≈ 0 → turnover ≈ 0.5 → FAIL。"""
    T, N = 120, 50
    values = np.random.default_rng(1).standard_normal(size=(T, N))
    df = _make_df(values, [f"S{i:03d}" for i in range(N)])
    cand = _make_candidate(tmp_path, "factor", df, factor_id="fast_test")

    result = TurnoverCheck().run(
        cand,
        _ctx(),
        {"max_daily_rank_turnover": 0.35, "min_rank_autocorr": 0.60},
    )
    assert not result.passed, result.detail
    # 随机因子：spearman ≈ 0 ⇒ turnover ≈ 0.5，容忍 ±0.1 波动
    assert 0.40 <= result.detail["daily_rank_turnover"] <= 0.60
    # 随机因子的 rank autocorr 应该接近 0
    assert abs(result.detail["rank_autocorr"]) < 0.2


# ---------------------------------------------------------- reversed factor（完全反转）


def test_reversed_factor_has_turnover_near_one(tmp_path: Path) -> None:
    """每天把因子值 negate 一次（结合底层同为单调漂移），构造 rank 反转 → turnover ≈ 1。"""
    T, N = 40, 20
    # 用单调漂移制造基础 rank；然后逐日交替 negate → 每日 rank 完全反转
    rng = np.random.default_rng(7)
    base = rng.uniform(-3.0, 3.0, N)[None, :] + 0.1 * np.arange(T)[:, None]
    flip = np.where(np.arange(T)[:, None] % 2 == 0, 1.0, -1.0)
    values = base * flip
    df = _make_df(values, [f"S{i:03d}" for i in range(N)])
    cand = _make_candidate(tmp_path, "factor", df, factor_id="reversed")

    result = TurnoverCheck().run(
        cand,
        _ctx(),
        {"max_daily_rank_turnover": 0.35, "min_rank_autocorr": 0.0},
    )
    assert not result.passed
    # 完全反转 ⇒ spearman ≈ -1 ⇒ turnover ≈ 1
    assert result.detail["daily_rank_turnover"] > 0.90


# ---------------------------------------------------------- invalid config


def test_invalid_threshold_out_of_range(tmp_path: Path) -> None:
    df = _make_df(
        np.ones((5, 5)),
        [f"S{i:03d}" for i in range(5)],
    )
    cand = _make_candidate(tmp_path, "factor", df, factor_id="bad_cfg")

    with pytest.raises(ValueError):
        TurnoverCheck().run(
            cand,
            _ctx(),
            {"max_daily_rank_turnover": 1.5, "min_rank_autocorr": 0.6},
        )


def test_missing_config_field_raises(tmp_path: Path) -> None:
    df = _make_df(np.ones((5, 5)), [f"S{i:03d}" for i in range(5)])
    cand = _make_candidate(tmp_path, "factor", df, factor_id="miss_cfg")
    with pytest.raises(ValueError, match="max_daily_rank_turnover"):
        TurnoverCheck().run(cand, _ctx(), {"min_rank_autocorr": 0.6})


# ---------------------------------------------------------- empty OOS window


def test_empty_oos_raises(tmp_path: Path) -> None:
    T, N = 20, 10
    values = np.random.default_rng(0).standard_normal(size=(T, N))
    df = _make_df(values, [f"S{i:03d}" for i in range(N)])
    cand = _make_candidate(tmp_path, "factor", df, factor_id="empty_oos")
    # OOS window 放到因子数据范围之外
    ctx = CheckContext(
        universe="csi300",
        oos_window=(date(2020, 1, 1), date(2020, 12, 31)),
        benchmark="SH000300",
        project_root=Path("."),
    )
    with pytest.raises(ValueError, match="OOS"):
        TurnoverCheck().run(
            cand,
            ctx,
            {"max_daily_rank_turnover": 0.40, "min_rank_autocorr": 0.60},
        )
