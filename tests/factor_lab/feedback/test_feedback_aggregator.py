"""E.2 单测：聚合器 build_feedback_bundle / classify_family / IO。"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from factor_lab.feedback.aggregator import (
    build_feedback_bundle,
    classify_family,
    load_latest_feedback_bundle,
    write_feedback_bundle,
)


def _ts(y: int = 2026, m: int = 4, d: int = 20) -> datetime:
    return datetime(y, m, d, 0, 0, 0, tzinfo=timezone.utc)


# ---------------------- fixture 构造器 ----------------------


def _write_cycle_report(reports_dir: Path, cycle_id: str, mtime_offset: float) -> Path:
    """写一个最小合法的 lab_cycle_*.json。mtime_offset 让我们控制"谁更新"。"""
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"lab_cycle_{cycle_id}.json"
    payload = {
        "cycle_id": cycle_id,
        "started_at": _ts().isoformat(),
        "finished_at": _ts().isoformat(),
        "totals": {"exploratory_pass": 0, "default_pass": 0, "promoted": 0, "failed": 1},
        "candidates": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    # 设置 mtime 以控制排序
    when = time.time() + mtime_offset
    import os
    os.utime(path, (when, when))
    return path


def _write_cert(
    cert_cycle_dir: Path,
    *,
    factor_id: str,
    name: str,
    stage: str,
    decision: str,
    check_results: list[dict[str, Any]],
) -> Path:
    cert_cycle_dir.mkdir(parents=True, exist_ok=True)
    path = cert_cycle_dir / f"{factor_id}.{stage}.json"
    payload = {
        "factor_id": factor_id,
        "candidate": {
            "factor_id": factor_id,
            "name": name,
            "source": "rdagent",
            "hypothesis": "h",
            "formulation": "f",
            "code_path": "placeholder.py",
            "values_path": "placeholder.parquet",
            "universe": "csi300",
            "date_range": ["2025-01-02", "2025-01-03"],
            "lab_metrics": {},
            "parent_loop": 0,
            "created_at": _ts().isoformat(),
            "lab_run_id": "run-xxxxxxxx",
        },
        "profile_name": "default",
        "profile_hash": "a" * 64,
        "decision": decision,
        "overall_score": 0.4,
        "check_results": check_results,
        "backtest_metrics": {},
        "validated_at": _ts().isoformat(),
        "validator_version": "0.1.0",
        "notes": None,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_manifest(registry_data_dir: Path, factors: list[dict[str, Any]]) -> Path:
    registry_data_dir.mkdir(parents=True, exist_ok=True)
    path = registry_data_dir / "manifest.json"
    payload = {
        "schema_version": 1,
        "last_updated_at": _ts().isoformat(),
        "factors": factors,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _active_record(*, factor_id: str, name: str, tags: list[str] | None = None) -> dict:
    return {
        "factor_id": factor_id,
        "name": name,
        "status": "active",
        "parquet_version": 1,
        "parquet_column": name,
        "certificate_path": f"certified/{factor_id}.json",
        "registered_at": _ts().isoformat(),
        "tags": tags or [],
    }


def _retired_record(*, factor_id: str, name: str, reason: str) -> dict:
    return {
        "factor_id": factor_id,
        "name": name,
        "status": "retired",
        "parquet_version": 1,
        "parquet_column": name,
        "certificate_path": f"retired/{factor_id}.json",
        "registered_at": _ts(2026, 3, 1).isoformat(),
        "retired_at": _ts(2026, 4, 10).isoformat(),
        "retire_reason": reason,
        "tags": ["legacy"],
    }


# ---------------------- family classification ----------------------


def test_classify_family_matches_reversal_family() -> None:
    assert classify_family("VolRet_5D") == "volume_price_reversal"
    assert classify_family("RangeRatio_10D") == "volume_price_reversal"
    assert classify_family("VolumeTrend_10D") == "volume_price_reversal"


def test_classify_family_matches_quality_persist() -> None:
    assert classify_family("QualPersist_60D") == "quality_persist"
    assert classify_family("ROEPersist_4Q") == "quality_persist"


def test_classify_family_unknown_returns_unknown() -> None:
    assert classify_family("SomeNovelThing_30D") == "unknown"


# ---------------------- build_feedback_bundle ----------------------


def test_build_feedback_bundle_happy_path(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    certs = tmp_path / "certs"
    reg = tmp_path / "reg"

    _write_cycle_report(reports, "cycle-001", mtime_offset=-10)
    _write_cycle_report(reports, "cycle-002", mtime_offset=0)

    # cycle-001：VolRev_10D 在 default 阶段 FAIL（ic + orthogonality）
    _write_cert(
        certs / "cycle-001",
        factor_id="rdagent_VolRev_10D_aaaaaaaa",
        name="VolRev_10D",
        stage="default",
        decision="FAIL",
        check_results=[
            {
                "name": "ic",
                "passed": False,
                "score": 0.2,
                "threshold": 0.012,
                "detail": {"rank_ic": 0.004, "ic_ir": 0.05},
                "elapsed_ms": 1,
            },
            {
                "name": "orthogonality",
                "passed": False,
                "score": 0.1,
                "threshold": 0.6,
                "detail": {"max_abs_corr": 0.72},
                "elapsed_ms": 1,
            },
        ],
    )
    # cycle-002：同类因子 VolTwist_20D 仅在 exploratory 阶段 FAIL（ic）
    _write_cert(
        certs / "cycle-002",
        factor_id="rdagent_VolTwist_20D_bbbbbbbb",
        name="VolTwist_20D",
        stage="exploratory",
        decision="FAIL",
        check_results=[
            {
                "name": "ic",
                "passed": False,
                "score": 0.1,
                "threshold": 0.005,
                "detail": {"rank_ic": 0.0015},
                "elapsed_ms": 1,
            }
        ],
    )

    _write_manifest(
        reg,
        [
            _active_record(
                factor_id="manual_QualPersist_60D_aabbccdd11",
                name="QualPersist_60D",
                tags=["production"],
            ),
            _retired_record(
                factor_id="manual_VolRet_5D_11d70180fb",
                name="VolRet_5D",
                reason="oracle default FAIL",
            ),
        ],
    )

    bundle = build_feedback_bundle(
        reports_dir=reports,
        cert_dir=certs,
        registry_data_dir=reg,
        max_cycles=8,
        min_fail_count_for_discouraged=2,
        now=_ts(),
    )

    assert bundle.cycles_included == ("cycle-001", "cycle-002")
    assert bundle.window_max_cycles == 8

    names = sorted(f.name for f in bundle.recent_fails)
    assert names == ["VolRev_10D", "VolTwist_20D"]
    for ff in bundle.recent_fails:
        assert ff.family == "volume_price_reversal"

    assert bundle.failure_family_counts.get("volume_price_reversal") == 2
    assert "volume_price_reversal" in bundle.discouraged_families

    assert len(bundle.active_factors) == 1
    assert bundle.active_factors[0].name == "QualPersist_60D"
    assert len(bundle.retired_factors) == 1
    assert bundle.retired_factors[0].name == "VolRet_5D"


def test_build_feedback_bundle_max_cycles_truncates(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    certs = tmp_path / "certs"
    reg = tmp_path / "reg"

    # 5 个 cycle，但只收最近 2 个
    for i, cid in enumerate(["c-a", "c-b", "c-c", "c-d", "c-e"]):
        _write_cycle_report(reports, cid, mtime_offset=-10 + i)

    bundle = build_feedback_bundle(
        reports_dir=reports,
        cert_dir=certs,
        registry_data_dir=reg,
        max_cycles=2,
        now=_ts(),
    )
    assert bundle.cycles_included == ("c-d", "c-e")


def test_build_feedback_bundle_prefers_deepest_stage(tmp_path: Path) -> None:
    """同一因子出现在 exploratory + default，只保留 default 那条。"""
    reports = tmp_path / "reports"
    certs = tmp_path / "certs"
    reg = tmp_path / "reg"

    _write_cycle_report(reports, "cycle-dup", mtime_offset=0)
    cycle_dir = certs / "cycle-dup"
    # exploratory HOLD
    _write_cert(
        cycle_dir,
        factor_id="rdagent_VolRev_10D_aaaaaaaa",
        name="VolRev_10D",
        stage="exploratory",
        decision="HOLD",
        check_results=[
            {
                "name": "ic",
                "passed": False,
                "score": 0.1,
                "threshold": 0.005,
                "detail": {"rank_ic": 0.003},
                "elapsed_ms": 1,
            }
        ],
    )
    # default FAIL（更深）
    _write_cert(
        cycle_dir,
        factor_id="rdagent_VolRev_10D_aaaaaaaa",
        name="VolRev_10D",
        stage="default",
        decision="FAIL",
        check_results=[
            {
                "name": "marginal",
                "passed": False,
                "score": 0.1,
                "threshold": 0.008,
                "detail": {"residual_rank_ic": 0.002},
                "elapsed_ms": 1,
            }
        ],
    )

    bundle = build_feedback_bundle(
        reports_dir=reports,
        cert_dir=certs,
        registry_data_dir=reg,
        max_cycles=8,
        now=_ts(),
    )
    assert len(bundle.recent_fails) == 1
    assert bundle.recent_fails[0].stage == "default"
    assert bundle.recent_fails[0].decision == "FAIL"
    assert bundle.recent_fails[0].failure_modes == ("marginal",)


def test_build_feedback_bundle_retired_family_always_discouraged(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    certs = tmp_path / "certs"
    reg = tmp_path / "reg"
    _write_cycle_report(reports, "cycle-x", mtime_offset=0)

    _write_manifest(
        reg,
        [
            _retired_record(
                factor_id="manual_MarginTrend_20D_1234567890",
                name="MarginTrend_20D",
                reason="poor IC",
            )
        ],
    )
    bundle = build_feedback_bundle(
        reports_dir=reports,
        cert_dir=certs,
        registry_data_dir=reg,
        max_cycles=8,
        min_fail_count_for_discouraged=100,
        now=_ts(),
    )
    assert "margin_trend" in bundle.discouraged_families


def test_build_feedback_bundle_skips_malformed_cert(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    certs = tmp_path / "certs"
    reg = tmp_path / "reg"

    _write_cycle_report(reports, "cycle-bad", mtime_offset=0)
    bad_dir = certs / "cycle-bad"
    bad_dir.mkdir(parents=True)
    (bad_dir / "fid.default.json").write_text("{bad-json", encoding="utf-8")
    # 同目录里放一个合法的
    _write_cert(
        bad_dir,
        factor_id="rdagent_ValueMR_60D_ccccccc1",
        name="ValueMR_60D",
        stage="default",
        decision="FAIL",
        check_results=[
            {
                "name": "ic",
                "passed": False,
                "score": 0.1,
                "threshold": 0.012,
                "detail": {"rank_ic": 0.001},
                "elapsed_ms": 1,
            }
        ],
    )

    bundle = build_feedback_bundle(
        reports_dir=reports,
        cert_dir=certs,
        registry_data_dir=reg,
        max_cycles=8,
        now=_ts(),
    )
    names = [f.name for f in bundle.recent_fails]
    assert names == ["ValueMR_60D"]


def test_build_feedback_bundle_empty_tree(tmp_path: Path) -> None:
    bundle = build_feedback_bundle(
        reports_dir=tmp_path / "none",
        cert_dir=tmp_path / "nope",
        registry_data_dir=tmp_path / "nil",
        max_cycles=8,
        now=_ts(),
    )
    assert bundle.cycles_included == ()
    assert bundle.recent_fails == ()
    assert bundle.active_factors == ()
    assert bundle.retired_factors == ()
    assert bundle.discouraged_families == ()


def test_build_feedback_bundle_rejects_zero_max_cycles(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        build_feedback_bundle(
            reports_dir=tmp_path,
            cert_dir=tmp_path,
            registry_data_dir=tmp_path,
            max_cycles=0,
        )


def test_build_feedback_bundle_skips_pass_decision(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    certs = tmp_path / "certs"
    reg = tmp_path / "reg"

    _write_cycle_report(reports, "cycle-pass", mtime_offset=0)
    _write_cert(
        certs / "cycle-pass",
        factor_id="rdagent_GoodAlpha_1D_ffff0000",
        name="GoodAlpha_1D",
        stage="default",
        decision="PASS",
        check_results=[
            {
                "name": "ic",
                "passed": True,
                "score": 0.9,
                "threshold": 0.012,
                "detail": {"rank_ic": 0.03},
                "elapsed_ms": 1,
            }
        ],
    )
    bundle = build_feedback_bundle(
        reports_dir=reports,
        cert_dir=certs,
        registry_data_dir=reg,
        max_cycles=8,
        now=_ts(),
    )
    assert bundle.recent_fails == ()


# ---------------------- write / load ----------------------


def test_write_and_load_latest_roundtrip(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    certs = tmp_path / "certs"
    reg = tmp_path / "reg"
    feedback = tmp_path / "feedback"

    _write_cycle_report(reports, "cycle-z", mtime_offset=0)
    bundle = build_feedback_bundle(
        reports_dir=reports,
        cert_dir=certs,
        registry_data_dir=reg,
        max_cycles=8,
        now=_ts(),
    )
    latest_json, latest_md = write_feedback_bundle(
        bundle, workspace_feedback_dir=feedback
    )
    assert latest_json.exists()
    assert latest_md.exists()
    assert (feedback / "history" / "cycle-z.json").exists()

    md = latest_md.read_text(encoding="utf-8")
    assert "Interpretation rules for LLM" in md
    assert "`n" not in md  # 防止 PowerShell 反引号换行回潮

    loaded = load_latest_feedback_bundle(feedback)
    assert loaded is not None
    assert loaded.cycles_included == ("cycle-z",)


def test_load_latest_missing_returns_none(tmp_path: Path) -> None:
    assert load_latest_feedback_bundle(tmp_path / "nope") is None


def test_load_latest_corrupt_returns_none(tmp_path: Path) -> None:
    d = tmp_path / "feedback"
    d.mkdir()
    (d / "latest.json").write_text("{not json", encoding="utf-8")
    assert load_latest_feedback_bundle(d) is None


def test_write_feedback_bundle_history_snap_when_no_cycles(tmp_path: Path) -> None:
    bundle = build_feedback_bundle(
        reports_dir=tmp_path / "none",
        cert_dir=tmp_path / "none",
        registry_data_dir=tmp_path / "none",
        max_cycles=8,
        now=_ts(),
    )
    feedback = tmp_path / "feedback"
    write_feedback_bundle(bundle, workspace_feedback_dir=feedback)
    hist_files = list((feedback / "history").glob("snap-*.json"))
    assert len(hist_files) == 1
