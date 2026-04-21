"""单测：``factor_validation.checks.marginal_training_check``。

覆盖：
1. 候选携带 baseline 无法提供的新信号 → uplift 显著 → PASS
2. 候选与 baseline 完全冗余（等于某一 baseline 列的噪声扰动）→ uplift ≈ 0 → FAIL
3. 候选是纯噪声 → uplift ≈ 0 → FAIL
4. 负向新信号 + allow_negative 切换
5. 缺 label_parquet → raise
6. 缺 reference_factors_parquet → raise
7. 缺必填配置 → raise
8. 有效交易日不足 → raise
9. 参照因子集为空 → raise

为控制耗时，测试统一用小规模 (T≈80, N≈40, features≈3, num_boost_round=60)。
整个文件总 runtime 预计 < 60 秒。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from factor_lab.exporters.schema import CandidateFactorPackage
from factor_validation.checks.base import CheckContext
from factor_validation.checks.marginal_training_check import MarginalTrainingCheck


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
        lab_run_id="lab-mtrain-00001",
    )


def _panel(arr: np.ndarray, instruments: list[str], name: str) -> pd.DataFrame:
    T, _ = arr.shape
    dates = pd.bdate_range("2025-01-02", periods=T)
    idx = pd.MultiIndex.from_product(
        [dates, instruments], names=["datetime", "instrument"]
    )
    return pd.DataFrame({name: arr.reshape(-1)}, index=idx).astype("float64")


def _write_ref(tmp_path: Path, cols: dict[str, np.ndarray], instruments: list[str]) -> Path:
    dfs = []
    for name, arr in cols.items():
        dfs.append(_panel(arr, instruments, name))
    merged = pd.concat(dfs, axis=1)
    p = tmp_path / "ref.parquet"
    merged.to_parquet(p)
    return p


def _ctx(
    label_path: Path | None,
    ref_path: Path | None,
    oos: tuple[date, date] = (date(2025, 1, 2), date(2025, 12, 31)),
) -> CheckContext:
    return CheckContext(
        universe="csi300",
        oos_window=oos,
        benchmark="SH000300",
        project_root=Path("."),
        label_parquet=label_path,
        reference_factors_parquet=ref_path,
    )


def _default_config(**overrides) -> dict:
    cfg = {
        "min_ic_uplift": 0.01,
        "train_ratio": 0.6,
        "num_boost_round": 60,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_data_in_leaf": 20,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "random_seed": 42,
        "min_test_days": 10,
    }
    cfg.update(overrides)
    return cfg


def _setup_common(
    tmp_path: Path,
    T: int,
    N: int,
    seed: int,
    *,
    cand_signal: str,  # "new" | "redundant" | "noise" | "negative"
    noise_scale: float = 0.01,
) -> tuple[CandidateFactorPackage, Path, Path]:
    rng = np.random.default_rng(seed)
    instruments = [f"S{i:03d}" for i in range(N)]
    # 3 个 baseline 因子
    f1 = rng.standard_normal(size=(T, N))
    f2 = rng.standard_normal(size=(T, N))
    f3 = rng.standard_normal(size=(T, N))

    if cand_signal == "new":
        # candidate 是独立的新信号，label 由 f1/f2 + candidate 共同决定
        cand = rng.standard_normal(size=(T, N))
        label = (
            0.02 * f1
            + 0.015 * f2
            + 0.02 * cand
            + rng.normal(0, noise_scale, size=(T, N))
        )
    elif cand_signal == "negative":
        # candidate 反向相关
        cand = rng.standard_normal(size=(T, N))
        label = (
            0.02 * f1
            + 0.015 * f2
            - 0.02 * cand
            + rng.normal(0, noise_scale, size=(T, N))
        )
    elif cand_signal == "redundant":
        # candidate ≈ f1 + 少量噪声（信息已在 baseline 里）
        cand = f1 + rng.normal(0, 0.0005, size=(T, N))
        label = (
            0.02 * f1
            + 0.015 * f2
            + rng.normal(0, noise_scale, size=(T, N))
        )
    elif cand_signal == "noise":
        # candidate 和 label 毫无关系
        cand = rng.standard_normal(size=(T, N))
        label = (
            0.02 * f1
            + 0.015 * f2
            + rng.normal(0, noise_scale, size=(T, N))
        )
    else:
        raise ValueError(cand_signal)

    cand_df = _panel(cand, instruments, "f")
    cand_pkg = _stub_candidate(tmp_path, "f", cand_df, factor_id=f"mt_{cand_signal[:4]}")

    lp = tmp_path / f"label_{cand_signal}.parquet"
    _panel(label, instruments, "label").to_parquet(lp)

    ref_path = _write_ref(
        tmp_path, {"f1": f1, "f2": f2, "f3": f3}, instruments
    )
    return cand_pkg, lp, ref_path


# ------------------------------------------------------------ 新信号 → PASS


def test_new_signal_uplift_passes(tmp_path: Path) -> None:
    c, lp, rp = _setup_common(tmp_path, T=80, N=40, seed=42, cand_signal="new")
    result = MarginalTrainingCheck().run(c, _ctx(lp, rp), _default_config())
    assert result.passed, result.detail
    assert result.detail["ic_uplift"] >= 0.01
    # B 模型中 candidate importance 应不为 0
    assert result.detail["candidate_importance_ratio"] > 0.05
    assert 0.0 < result.score <= 1.0
    assert result.detail["n_train_days"] + result.detail["n_test_days"] == 80


# ------------------------------------------------------------ 冗余 → FAIL


def test_redundant_candidate_fails(tmp_path: Path) -> None:
    c, lp, rp = _setup_common(
        tmp_path, T=80, N=40, seed=11, cand_signal="redundant"
    )
    result = MarginalTrainingCheck().run(c, _ctx(lp, rp), _default_config())
    # 与 baseline 完全冗余 → uplift 应远低于阈值
    assert not result.passed
    assert result.detail["ic_uplift"] < 0.01


# ------------------------------------------------------------ 噪声 → FAIL


def test_noise_candidate_fails(tmp_path: Path) -> None:
    c, lp, rp = _setup_common(tmp_path, T=80, N=40, seed=7, cand_signal="noise")
    result = MarginalTrainingCheck().run(c, _ctx(lp, rp), _default_config())
    assert not result.passed
    assert abs(result.detail["ic_uplift"]) < 0.01


# ------------------------------------------------------------ 负向 + allow_negative


def test_negative_signal_with_allow_negative(tmp_path: Path) -> None:
    c, lp, rp = _setup_common(
        tmp_path, T=80, N=40, seed=23, cand_signal="negative"
    )
    # 未开 allow_negative：uplift 为正（LGB 可以自动学到反向），通常仍然 PASS；
    # 但如果 uplift 实际上接近 0/负，则 FAIL —— 这里只验证开/关行为不抛错
    res_open = MarginalTrainingCheck().run(
        c, _ctx(lp, rp), _default_config(allow_negative=True)
    )
    # 负向独立信号 LGB 能学，uplift 应为正
    assert res_open.detail["ic_uplift"] > 0
    assert res_open.passed, res_open.detail


# ------------------------------------------------------------ 缺 label_parquet


def test_missing_label_raises(tmp_path: Path) -> None:
    c, _, rp = _setup_common(tmp_path, T=30, N=20, seed=0, cand_signal="new")
    with pytest.raises(ValueError, match="label_parquet"):
        MarginalTrainingCheck().run(c, _ctx(None, rp), _default_config())


# ------------------------------------------------------------ 缺 reference_factors


def test_missing_reference_raises(tmp_path: Path) -> None:
    c, lp, _ = _setup_common(tmp_path, T=30, N=20, seed=0, cand_signal="new")
    with pytest.raises(ValueError, match="reference_factors_parquet"):
        MarginalTrainingCheck().run(c, _ctx(lp, None), _default_config())


# ------------------------------------------------------------ 缺必填配置


def test_missing_required_config_raises(tmp_path: Path) -> None:
    c, lp, rp = _setup_common(tmp_path, T=30, N=20, seed=0, cand_signal="new")
    with pytest.raises(ValueError, match="min_ic_uplift"):
        MarginalTrainingCheck().run(c, _ctx(lp, rp), {"train_ratio": 0.6})


# ------------------------------------------------------------ 天数不足


def test_insufficient_days_raises(tmp_path: Path) -> None:
    c, lp, rp = _setup_common(tmp_path, T=12, N=15, seed=0, cand_signal="new")
    # train_ratio=0.6 → train=7, test=5 < min_test_days=10
    with pytest.raises(ValueError, match="有效交易日不足"):
        MarginalTrainingCheck().run(c, _ctx(lp, rp), _default_config())


# ------------------------------------------------------------ 参照空


def test_empty_reference_raises(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    T, N = 30, 15
    instruments = [f"S{i:03d}" for i in range(N)]
    cand = rng.standard_normal(size=(T, N))
    cand_df = _panel(cand, instruments, "f")
    c = _stub_candidate(tmp_path, "f", cand_df, "mtempty")

    label = rng.standard_normal(size=(T, N)) * 0.01
    lp = tmp_path / "l.parquet"
    _panel(label, instruments, "label").to_parquet(lp)

    # 参照只有候选同名列 → 会被 drop → 0 列
    rp = tmp_path / "r.parquet"
    _panel(rng.standard_normal(size=(T, N)), instruments, "f").to_parquet(rp)

    with pytest.raises(ValueError, match="参照因子集为空"):
        MarginalTrainingCheck().run(c, _ctx(lp, rp), _default_config())
