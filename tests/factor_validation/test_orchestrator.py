"""单元测试：``factor_validation.orchestrator`` + ``checks.coverage_check``。"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
import yaml

from factor_lab.exporters.schema import CandidateFactorPackage
from factor_validation.checks import get_check
from factor_validation.checks.coverage_check import CoverageCheck
from factor_validation.orchestrator import (
    ProfileError,
    VALIDATOR_VERSION,
    ValidationProfile,
    validate_candidate,
)
from factor_validation.schema import Decision


# ---------------------------------------------------------- profile loading

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_GF_PROFILE = _PROJECT_ROOT / "factor_validation" / "profiles" / "manual_grandfathered.yaml"


def _write_profile(path: Path, overrides: dict | None = None) -> Path:
    base = {
        "profile_name": path.stem,
        "description": "test",
        "universe": "csi300",
        "oos_window": ["2025-01-01", "2025-10-31"],
        "benchmark": "SH000300",
        "aggregation": {
            "method": "weighted_sum",
            "pass_threshold": 0.6,
            "fail_threshold": 0.3,
        },
        "checks": {
            "coverage": {
                "enabled": True,
                "weight": 1.0,
                "min_non_null_ratio": 0.90,
            }
        },
    }
    if overrides:
        _deep_merge(base, overrides)
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    return path


def _deep_merge(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if k in dst and isinstance(dst[k], dict) and isinstance(v, dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v


def test_manual_grandfathered_profile_loads_ok() -> None:
    profile = ValidationProfile(_GF_PROFILE)
    assert profile.name == "manual_grandfathered"
    assert profile.universe == "csi300"
    assert profile.benchmark == "SH000300"
    assert set(profile.enabled_checks()) == {"coverage"}
    assert profile.weight_of("coverage") == 1.0
    assert len(profile.profile_hash) == 64


def test_profile_rejects_name_mismatch(tmp_path: Path) -> None:
    p = tmp_path / "foo.yaml"
    _write_profile(p, {"profile_name": "bar"})
    with pytest.raises(ProfileError, match="profile_name"):
        ValidationProfile(p)


def test_profile_rejects_weight_sum_mismatch(tmp_path: Path) -> None:
    p = tmp_path / "bad_w.yaml"
    _write_profile(
        p,
        {
            "checks": {
                "coverage": {"enabled": True, "weight": 0.5, "min_non_null_ratio": 0.9},
            }
        },
    )
    with pytest.raises(ProfileError, match="weight 总和"):
        ValidationProfile(p)


def test_profile_rejects_swapped_thresholds(tmp_path: Path) -> None:
    p = tmp_path / "bad_thr.yaml"
    _write_profile(
        p,
        {"aggregation": {"pass_threshold": 0.3, "fail_threshold": 0.6}},
    )
    with pytest.raises(ProfileError, match="pass_threshold"):
        ValidationProfile(p)


def test_profile_rejects_unknown_aggregation(tmp_path: Path) -> None:
    p = tmp_path / "bad_agg.yaml"
    _write_profile(p, {"aggregation": {"method": "arithmetic"}})
    with pytest.raises(ProfileError, match="aggregation.method"):
        ValidationProfile(p)


def test_profile_rejects_no_enabled(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    _write_profile(
        p,
        {
            "checks": {
                "coverage": {
                    "enabled": False,
                    "min_non_null_ratio": 0.9,
                }
            }
        },
    )
    with pytest.raises(ProfileError, match="enabled"):
        ValidationProfile(p)


def test_profile_rejects_missing_required(tmp_path: Path) -> None:
    p = tmp_path / "nobench.yaml"
    raw = {
        "profile_name": "nobench",
        "description": "test",
        "universe": "csi300",
        "oos_window": ["2025-01-01", "2025-10-31"],
        "aggregation": {
            "method": "weighted_sum",
            "pass_threshold": 0.6,
            "fail_threshold": 0.3,
        },
        "checks": {"coverage": {"enabled": True, "weight": 1.0}},
    }
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ProfileError, match="benchmark"):
        ValidationProfile(p)


# ---------------------------------------------------------- coverage check


def test_coverage_check_registered() -> None:
    assert get_check("coverage") is CoverageCheck


def test_coverage_check_passes(tmp_path: Path, minimal_candidate) -> None:
    from factor_validation.checks.base import CheckContext

    ctx = CheckContext(
        universe="csi300",
        oos_window=(date(2025, 1, 2), date(2025, 6, 30)),
        benchmark="SH000300",
        project_root=tmp_path,
    )
    result = CoverageCheck().run(minimal_candidate, ctx, {"min_non_null_ratio": 0.90})
    assert result.name == "coverage"
    assert result.passed is True
    assert result.score == 1.0
    assert result.detail["n_instruments"] == 1


def test_coverage_check_detects_holes(tmp_path: Path, minimal_candidate) -> None:
    """把 parquet 改成 50% NaN → score=0.5 → passed=False。"""
    from factor_validation.checks.base import CheckContext

    # 重写 values.parquet：2 行中 1 行 NaN
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2025-01-02"), "SH600000"),
            (pd.Timestamp("2025-06-30"), "SH600000"),
        ],
        names=["datetime", "instrument"],
    )
    df = pd.DataFrame(
        {minimal_candidate.name: [0.1, float("nan")]},
        index=idx,
        dtype="float64",
    )
    df.to_parquet(minimal_candidate.values_path)

    ctx = CheckContext(
        universe="csi300",
        oos_window=(date(2025, 1, 2), date(2025, 6, 30)),
        benchmark="SH000300",
        project_root=tmp_path,
    )
    result = CoverageCheck().run(minimal_candidate, ctx, {"min_non_null_ratio": 0.90})
    assert result.passed is False
    assert result.score == 0.5


def test_coverage_check_rejects_bad_threshold(tmp_path: Path, minimal_candidate) -> None:
    from factor_validation.checks.base import CheckContext

    ctx = CheckContext(
        universe="csi300",
        oos_window=(date(2025, 1, 2), date(2025, 6, 30)),
        benchmark="SH000300",
        project_root=tmp_path,
    )
    with pytest.raises(ValueError, match="min_non_null_ratio"):
        CoverageCheck().run(minimal_candidate, ctx, {"min_non_null_ratio": 2.0})


# ---------------------------------------------------------- validate_candidate


def test_validate_candidate_pass_on_grandfathered(
    tmp_path: Path, minimal_candidate
) -> None:
    cert = validate_candidate(
        minimal_candidate,
        _GF_PROFILE,
        project_root=tmp_path,
        now=datetime(2026, 4, 19, 13, 0, tzinfo=timezone.utc),
        notes="migration",
    )
    assert cert.decision == Decision.PASS
    assert cert.profile_name == "manual_grandfathered"
    assert cert.overall_score == 1.0
    assert cert.notes == "migration"
    assert cert.validator_version == VALIDATOR_VERSION
    assert len(cert.check_results) == 1
    assert cert.check_results[0].passed is True


def test_validate_candidate_fail_on_bad_coverage(
    tmp_path: Path, minimal_candidate
) -> None:
    # 污染 parquet 让 coverage < 0.9
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2025-01-02"), "SH600000"),
            (pd.Timestamp("2025-06-30"), "SH600000"),
        ],
        names=["datetime", "instrument"],
    )
    pd.DataFrame(
        {minimal_candidate.name: [0.1, float("nan")]}, index=idx, dtype="float64"
    ).to_parquet(minimal_candidate.values_path)

    cert = validate_candidate(
        minimal_candidate, _GF_PROFILE, project_root=tmp_path
    )
    assert cert.decision == Decision.FAIL
    assert cert.check_results[0].passed is False


def test_validate_candidate_exception_wrapped(
    tmp_path: Path, minimal_candidate
) -> None:
    """人为损坏 parquet → CoverageCheck 抛错 → 被 orchestrator 捕获为 FAIL。"""
    # 清空 parquet → CoverageCheck 里 df.empty 抛 ValueError
    pd.DataFrame(
        {minimal_candidate.name: []},
        index=pd.MultiIndex.from_tuples(
            [], names=["datetime", "instrument"]
        ),
        dtype="float64",
    ).to_parquet(minimal_candidate.values_path)

    cert = validate_candidate(minimal_candidate, _GF_PROFILE, project_root=tmp_path)
    assert cert.decision == Decision.FAIL
    r = cert.check_results[0]
    assert r.passed is False
    assert "error" in r.detail


def test_validate_candidate_hold(tmp_path: Path, minimal_candidate) -> None:
    """构造 profile：pass_thr=0.99 → 完全覆盖 parquet 也会 HOLD。"""
    # minimal_candidate coverage = 1.0，但 pass_threshold=1.0 + fail=0.0，
    # 若再把 pass_threshold 设为 1.0000001 就 > 1，不合理；改用非满分场景：
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2025-01-02"), "SH600000"),
            (pd.Timestamp("2025-06-30"), "SH600000"),
        ],
        names=["datetime", "instrument"],
    )
    # 非空率 0.5 → 通过 min_non_null_ratio=0.10 → passed=True
    pd.DataFrame(
        {minimal_candidate.name: [0.1, float("nan")]},
        index=idx,
        dtype="float64",
    ).to_parquet(minimal_candidate.values_path)

    p = tmp_path / "hold_profile.yaml"
    _write_profile(
        p,
        {
            "aggregation": {"pass_threshold": 0.9, "fail_threshold": 0.1},
            "checks": {
                "coverage": {
                    "enabled": True,
                    "weight": 1.0,
                    "min_non_null_ratio": 0.10,  # 通过 → passed=True
                }
            },
        },
    )
    cert = validate_candidate(minimal_candidate, p, project_root=tmp_path)
    # all_passed=True, overall=0.5, 0.1 <= 0.5 < 0.9 → HOLD
    assert cert.decision == Decision.HOLD
    assert cert.overall_score == 0.5


def test_profile_hash_sensitive_to_content(tmp_path: Path) -> None:
    p1 = tmp_path / "p1.yaml"
    p2 = tmp_path / "p2.yaml"
    _write_profile(p1, {"profile_name": "p1"})
    _write_profile(p2, {"profile_name": "p2", "description": "changed"})
    h1 = ValidationProfile(p1).profile_hash
    h2 = ValidationProfile(p2).profile_hash
    assert h1 != h2


def test_profile_hash_stable_across_reloads(tmp_path: Path) -> None:
    p = tmp_path / "p1.yaml"
    _write_profile(p)
    h1 = ValidationProfile(p).profile_hash
    h2 = ValidationProfile(p).profile_hash
    assert h1 == h2
