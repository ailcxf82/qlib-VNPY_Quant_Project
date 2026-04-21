"""阶段 F.2：FeedbackBundle 三个 summary 的 universe 字段 + aggregator 填充 + markdown 渲染。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from factor_lab.feedback import (
    ActiveFactorSummary,
    FailedCandidateSummary,
    FeedbackBundle,
    RetiredFactorSummary,
)
from factor_lab.feedback.aggregator import build_feedback_bundle


# ------------------------------------------------------------------ schema level


def test_active_summary_accepts_universe() -> None:
    s = ActiveFactorSummary(
        factor_id="prod_alpha_abcdef01",
        name="Quality_ROE_5d",
        family="quality_persist",
        universe="csi300",
        parquet_version=1,
    )
    assert s.universe == "csi300"


def test_active_summary_default_universe_is_none() -> None:
    """老 JSON（没有 universe 字段）必须仍可加载，对应 universe=None。"""
    s = ActiveFactorSummary(
        factor_id="prod_alpha_abcdef01",
        name="Alpha_1",
        parquet_version=1,
    )
    assert s.universe is None


def test_active_summary_rejects_bad_universe() -> None:
    with pytest.raises(Exception) as excinfo:
        ActiveFactorSummary(
            factor_id="prod_alpha_abcdef01",
            name="Alpha_1",
            parquet_version=1,
            universe="has space",
        )
    assert "universe" in str(excinfo.value)


@pytest.mark.parametrize("bad", ["", " ", "has space", "a" * 65, "has/slash", "uni\tverse"])
def test_universe_pattern_rejections(bad: str) -> None:
    with pytest.raises(Exception):
        FailedCandidateSummary(
            name="VolRev_5d",
            family="volume_price_reversal",
            universe=bad,
            cycle_id="c1",
            stage="default",
            decision="FAIL",
        )


def test_retired_summary_accepts_universe() -> None:
    s = RetiredFactorSummary(
        factor_id="prod_alpha_abcdef02",
        name="Old_VolRev",
        family="volume_price_reversal",
        universe="csi500",
        reason="rolling IC degraded",
    )
    assert s.universe == "csi500"


def test_failed_summary_accepts_universe() -> None:
    s = FailedCandidateSummary(
        name="VolRev_5d",
        family="volume_price_reversal",
        universe="csi300",
        cycle_id="c1",
        stage="default",
        decision="FAIL",
    )
    assert s.universe == "csi300"


def test_old_bundle_json_loads_without_universe_fields() -> None:
    """backward-compat：老 bundle JSON 里没有 universe 字段，仍能 model_validate_json。"""
    payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cycles_included": ["c1"],
        "window_max_cycles": 4,
        "active_factors": [
            {
                "factor_id": "prod_alpha_abcdef01",
                "name": "Alpha_1",
                "family": "quality_persist",
                "parquet_version": 1,
                "tags": [],
            }
        ],
        "retired_factors": [],
        "recent_fails": [
            {
                "name": "VolRev_5d",
                "family": "volume_price_reversal",
                "cycle_id": "c1",
                "stage": "default",
                "decision": "FAIL",
                "failure_modes": ["ic"],
                "metrics": {"rank_ic": 0.004},
            }
        ],
        "failure_family_counts": {"volume_price_reversal": 1},
        "discouraged_families": [],
    }
    bundle = FeedbackBundle.model_validate(payload)
    assert bundle.active_factors[0].universe is None
    assert bundle.recent_fails[0].universe is None


# --------------------------------------------------------------- markdown render


def _make_bundle_with_universe() -> FeedbackBundle:
    return FeedbackBundle(
        generated_at=datetime(2026, 4, 18, tzinfo=timezone.utc),
        cycles_included=("c1",),
        window_max_cycles=4,
        active_factors=(
            ActiveFactorSummary(
                factor_id="prod_alpha_abcdef01",
                name="Quality_ROE_5d",
                family="quality_persist",
                universe="csi300",
                parquet_version=1,
                tags=("liqud",),
            ),
        ),
        retired_factors=(
            RetiredFactorSummary(
                factor_id="prod_alpha_abcdef02",
                name="Old_VolRev",
                family="volume_price_reversal",
                universe="csi500",
                retired_at=datetime(2026, 3, 10, tzinfo=timezone.utc),
                reason="rolling IC degraded",
            ),
        ),
        recent_fails=(
            FailedCandidateSummary(
                name="VolRev_5d",
                family="volume_price_reversal",
                universe="csi300",
                cycle_id="c1",
                stage="default",
                decision="FAIL",
                failure_modes=("ic",),
            ),
        ),
        failure_family_counts={"volume_price_reversal": 1},
        discouraged_families=("volume_price_reversal",),
    )


def test_markdown_includes_universe_tag_per_entry() -> None:
    md = _make_bundle_with_universe().to_markdown()
    # 每一条都应该带 [universe=...]
    assert "[universe=csi300]" in md
    assert "[universe=csi500]" in md
    # active / retired / fails 三个分区里都能找到
    assert md.count("[universe=") == 3


def test_markdown_without_universe_stays_backward_compat() -> None:
    """universe=None 时，不该冒出空的 [universe=] 文本。"""
    bundle = FeedbackBundle(
        generated_at=datetime(2026, 4, 18, tzinfo=timezone.utc),
        cycles_included=("c1",),
        window_max_cycles=4,
        active_factors=(
            ActiveFactorSummary(
                factor_id="prod_alpha_abcdef01",
                name="Alpha_1",
                parquet_version=1,
            ),
        ),
    )
    md = bundle.to_markdown()
    assert "[universe=" not in md


# ------------------------------------------------------- aggregator integration


def _seed_cycle(tmp_path: Path) -> tuple[Path, Path, Path]:
    reports_dir = tmp_path / "reports"
    cert_dir = tmp_path / "certs"
    registry_dir = tmp_path / "registry"
    reports_dir.mkdir()
    cert_dir.mkdir()
    registry_dir.mkdir()

    # cycle report
    (reports_dir / "lab_cycle_cyc1.json").write_text(
        json.dumps({"cycle_id": "cyc1"}), encoding="utf-8"
    )

    # cert: candidate.universe=csi300
    cyc_dir = cert_dir / "cyc1"
    cyc_dir.mkdir()
    cert_payload = {
        "decision": "FAIL",
        "candidate": {
            "name": "VolRev_5d",
            "universe": "csi300",
        },
        "check_results": [
            {"name": "ic", "passed": False, "detail": {"rank_ic": 0.004}},
        ],
    }
    (cyc_dir / "fid.default.json").write_text(json.dumps(cert_payload), encoding="utf-8")

    # manifest: active with universe; retired with universe
    manifest = {
        "factors": [
            {
                "factor_id": "prod_alpha_abcdef01",
                "name": "Quality_ROE_5d",
                "status": "active",
                "parquet_version": 1,
                "universe": "csi300",
            },
            {
                "factor_id": "prod_alpha_abcdef02",
                "name": "Old_VolRev",
                "status": "retired",
                "retired_at": "2026-03-10T00:00:00+00:00",
                "retire_reason": "rolling IC degraded",
                "universe": "csi500",
            },
        ]
    }
    (registry_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    return reports_dir, cert_dir, registry_dir


def test_aggregator_fills_universe_on_all_three_summaries(tmp_path: Path) -> None:
    r, c, reg = _seed_cycle(tmp_path)
    bundle = build_feedback_bundle(
        reports_dir=r, cert_dir=c, registry_data_dir=reg, max_cycles=4
    )
    assert bundle.active_factors[0].universe == "csi300"
    assert bundle.retired_factors[0].universe == "csi500"
    assert bundle.recent_fails[0].universe == "csi300"


def test_aggregator_tolerates_missing_universe_in_manifest(tmp_path: Path) -> None:
    r, c, reg = _seed_cycle(tmp_path)
    # 覆盖 manifest，去掉 universe 字段
    manifest = {
        "factors": [
            {
                "factor_id": "prod_alpha_abcdef01",
                "name": "Alpha_1",
                "status": "active",
                "parquet_version": 1,
            },
        ]
    }
    (reg / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    bundle = build_feedback_bundle(
        reports_dir=r, cert_dir=c, registry_data_dir=reg, max_cycles=4
    )
    assert bundle.active_factors[0].universe is None


def test_aggregator_tolerates_missing_universe_in_candidate(tmp_path: Path) -> None:
    r, c, reg = _seed_cycle(tmp_path)
    cert_payload = {
        "decision": "FAIL",
        "candidate": {"name": "VolRev_5d"},  # 无 universe
        "check_results": [{"name": "ic", "passed": False, "detail": {}}],
    }
    (c / "cyc1" / "fid.default.json").write_text(json.dumps(cert_payload), encoding="utf-8")
    bundle = build_feedback_bundle(
        reports_dir=r, cert_dir=c, registry_data_dir=reg, max_cycles=4
    )
    assert bundle.recent_fails[0].universe is None


def test_markdown_end_to_end_from_aggregator(tmp_path: Path) -> None:
    r, c, reg = _seed_cycle(tmp_path)
    bundle = build_feedback_bundle(
        reports_dir=r, cert_dir=c, registry_data_dir=reg, max_cycles=4
    )
    md = bundle.to_markdown()
    assert "[universe=csi300]" in md
    assert "[universe=csi500]" in md
