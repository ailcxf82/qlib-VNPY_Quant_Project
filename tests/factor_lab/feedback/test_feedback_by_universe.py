"""阶段 G.1：FeedbackBundle 的 by_universe 子结构 + aggregator 构造 + 渲染 + 落盘。

覆盖点：

1. ``UniverseSubBundle`` 基本字段/校验（key 与 sub.universe 的双重 validation）。
2. ``FeedbackBundle`` 的 by_universe 字段：默认空 dict、老 JSON 向后兼容、
   key/value 一致性、``cycle_id ⊆ cycles_included`` 校验。
3. ``build_feedback_bundle`` 从 manifest + cert 推导出正确的 per-universe 子桶。
4. ``write_feedback_bundle`` 按要求落 ``latest_<universe>.json`` 附加文件。
5. ``to_markdown()`` 渲染分 universe 视图；universe 未知的条目**不**进任何桶。
"""

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
    UniverseSubBundle,
)
from factor_lab.feedback.aggregator import (
    _derive_by_universe,
    build_feedback_bundle,
    write_feedback_bundle,
)


# -------------------------------------------------------------- UniverseSubBundle


def test_sub_bundle_minimal_construction() -> None:
    sub = UniverseSubBundle(universe="csi300")
    assert sub.universe == "csi300"
    assert sub.active_factors == ()
    assert sub.retired_factors == ()
    assert sub.recent_fails == ()
    assert sub.failure_family_counts == {}
    assert sub.discouraged_families == ()


def test_sub_bundle_rejects_cross_universe_content() -> None:
    with pytest.raises(Exception):
        UniverseSubBundle(
            universe="csi300",
            recent_fails=(
                FailedCandidateSummary(
                    name="VolRev_5d",
                    family="volume_price_reversal",
                    universe="csi500",
                    cycle_id="c1",
                    stage="default",
                    decision="FAIL",
                ),
            ),
        )


def test_sub_bundle_allows_none_universe_inner_entries() -> None:
    """子桶内部条目 universe=None 时视为与本桶兼容（聚合器不会这样构造，但手搓时要允许）。"""
    sub = UniverseSubBundle(
        universe="csi300",
        recent_fails=(
            FailedCandidateSummary(
                name="VolRev_5d",
                family="volume_price_reversal",
                universe=None,
                cycle_id="c1",
                stage="default",
                decision="FAIL",
            ),
        ),
    )
    assert sub.universe == "csi300"


# ------------------------------------------------------- FeedbackBundle.by_universe


def _bundle_with(**kwargs) -> FeedbackBundle:
    base = dict(
        generated_at=datetime(2026, 4, 18, tzinfo=timezone.utc),
        cycles_included=("c1",),
        window_max_cycles=4,
    )
    base.update(kwargs)
    return FeedbackBundle(**base)


def test_by_universe_defaults_to_empty_dict() -> None:
    b = _bundle_with()
    assert b.by_universe == {}


def test_old_json_without_by_universe_loads_fine() -> None:
    payload = {
        "schema_version": "1.0",
        "generated_at": datetime(2026, 4, 18, tzinfo=timezone.utc).isoformat(),
        "cycles_included": ["c1"],
        "window_max_cycles": 4,
        "active_factors": [],
        "retired_factors": [],
        "recent_fails": [],
        "failure_family_counts": {},
        "discouraged_families": [],
    }
    bundle = FeedbackBundle.model_validate(payload)
    assert bundle.by_universe == {}


def test_by_universe_key_must_equal_sub_universe() -> None:
    with pytest.raises(Exception) as exc:
        _bundle_with(
            by_universe={
                "csi300": UniverseSubBundle(universe="csi500"),
            }
        )
    assert "键值必须与子桶 universe 一致" in str(exc.value) or "universe" in str(exc.value)


def test_by_universe_cycle_id_must_be_in_cycles_included() -> None:
    with pytest.raises(Exception):
        _bundle_with(
            cycles_included=("c1",),
            by_universe={
                "csi300": UniverseSubBundle(
                    universe="csi300",
                    recent_fails=(
                        FailedCandidateSummary(
                            name="VolRev_5d",
                            family="volume_price_reversal",
                            universe="csi300",
                            cycle_id="c-other",
                            stage="default",
                            decision="FAIL",
                        ),
                    ),
                )
            },
        )


# ------------------------------------------------ aggregator builds by_universe


def test_derive_by_universe_pure_logic() -> None:
    active = [
        ActiveFactorSummary(
            factor_id="prod_alpha_abcdef01",
            name="Quality_ROE_5d",
            family="quality_persist",
            universe="csi300",
            parquet_version=1,
        ),
        ActiveFactorSummary(
            factor_id="prod_alpha_abcdef02",
            name="LiqStab_20D",
            family="liquidity_stability",
            universe=None,  # 不进任何桶
            parquet_version=1,
        ),
    ]
    retired = [
        RetiredFactorSummary(
            factor_id="prod_alpha_abcdef03",
            name="Old_VolRev",
            family="volume_price_reversal",
            universe="csi500",
            reason="bad IR",
        ),
    ]
    fails = [
        FailedCandidateSummary(
            name="VolRev_5d",
            family="volume_price_reversal",
            universe="csi300",
            cycle_id="c1",
            stage="default",
            decision="FAIL",
            failure_modes=("ic",),
        ),
        FailedCandidateSummary(
            name="VolRev_10d",
            family="volume_price_reversal",
            universe="csi300",
            cycle_id="c1",
            stage="default",
            decision="FAIL",
            failure_modes=("ic",),
        ),
    ]

    sub = _derive_by_universe(active, retired, fails, min_fail_count=2)
    assert set(sub.keys()) == {"csi300", "csi500"}
    # csi300: 2 次 volume_price_reversal fails >= 阈值 2 → discouraged
    assert sub["csi300"].failure_family_counts == {"volume_price_reversal": 2}
    assert sub["csi300"].discouraged_families == ("volume_price_reversal",)
    assert len(sub["csi300"].active_factors) == 1
    # csi500: retired 出现 volume_price_reversal → 无条件 discouraged
    assert sub["csi500"].discouraged_families == ("volume_price_reversal",)
    assert len(sub["csi500"].retired_factors) == 1


def _seed_cycle(tmp_path: Path) -> tuple[Path, Path, Path]:
    reports = tmp_path / "reports"
    certs = tmp_path / "certs"
    reg = tmp_path / "registry"
    reports.mkdir()
    certs.mkdir()
    reg.mkdir()

    (reports / "lab_cycle_cyc1.json").write_text(json.dumps({"cycle_id": "cyc1"}), encoding="utf-8")

    cyc = certs / "cyc1"
    cyc.mkdir()
    for i, u in enumerate(["csi300", "csi300"]):
        (cyc / f"fid{i}.default.json").write_text(
            json.dumps(
                {
                    "decision": "FAIL",
                    "candidate": {"name": f"VolRev_{5 + i*5}d", "universe": u},
                    "check_results": [{"name": "ic", "passed": False, "detail": {"rank_ic": 0.003}}],
                }
            ),
            encoding="utf-8",
        )

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
                "retire_reason": "bad IR",
                "universe": "csi500",
            },
        ]
    }
    (reg / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return reports, certs, reg


def test_build_feedback_bundle_populates_by_universe(tmp_path: Path) -> None:
    r, c, reg = _seed_cycle(tmp_path)
    bundle = build_feedback_bundle(
        reports_dir=r, cert_dir=c, registry_data_dir=reg, max_cycles=4
    )
    assert "csi300" in bundle.by_universe
    assert "csi500" in bundle.by_universe
    # csi300 下有 2 次 volume_price_reversal 失败
    assert bundle.by_universe["csi300"].failure_family_counts == {"volume_price_reversal": 2}
    # csi500 有一个 retired，discouraged 应包含 volume_price_reversal
    assert "volume_price_reversal" in bundle.by_universe["csi500"].discouraged_families


# ------------------------------------------------------ write_feedback_bundle I/O


def test_write_feedback_bundle_writes_per_universe_files(tmp_path: Path) -> None:
    r, c, reg = _seed_cycle(tmp_path)
    bundle = build_feedback_bundle(
        reports_dir=r, cert_dir=c, registry_data_dir=reg, max_cycles=4
    )
    out_dir = tmp_path / "fb"
    latest_json, _latest_md = write_feedback_bundle(bundle, workspace_feedback_dir=out_dir)
    assert latest_json.exists()
    assert (out_dir / "latest_csi300.json").exists()
    assert (out_dir / "latest_csi500.json").exists()
    # 文件内容能反序列化回 UniverseSubBundle
    data = json.loads((out_dir / "latest_csi300.json").read_text(encoding="utf-8"))
    assert data["universe"] == "csi300"
    assert UniverseSubBundle.model_validate(data).universe == "csi300"


def test_write_feedback_bundle_can_skip_per_universe(tmp_path: Path) -> None:
    r, c, reg = _seed_cycle(tmp_path)
    bundle = build_feedback_bundle(
        reports_dir=r, cert_dir=c, registry_data_dir=reg, max_cycles=4
    )
    out_dir = tmp_path / "fb"
    write_feedback_bundle(
        bundle, workspace_feedback_dir=out_dir, also_write_per_universe=False
    )
    assert not (out_dir / "latest_csi300.json").exists()


# ---------------------------------------------- markdown rendering of by_universe


def test_markdown_renders_per_universe_section() -> None:
    bundle = FeedbackBundle(
        generated_at=datetime(2026, 4, 18, tzinfo=timezone.utc),
        cycles_included=("c1",),
        window_max_cycles=4,
        by_universe={
            "csi300": UniverseSubBundle(
                universe="csi300",
                failure_family_counts={"volume_price_reversal": 3},
                discouraged_families=("volume_price_reversal",),
            ),
        },
    )
    md = bundle.to_markdown()
    assert "Per-universe view" in md
    assert "universe=csi300" in md
    assert "volume_price_reversal:3" in md
    assert "discouraged (local)" in md


def test_markdown_omits_per_universe_when_empty() -> None:
    bundle = FeedbackBundle(
        generated_at=datetime(2026, 4, 18, tzinfo=timezone.utc),
        cycles_included=("c1",),
        window_max_cycles=4,
    )
    md = bundle.to_markdown()
    assert "Per-universe view" not in md
