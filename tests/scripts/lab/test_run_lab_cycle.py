from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from factor_lab.exporters.schema import CandidateFactorPackage
from factor_validation.schema import CertifiedFactorRecord, CheckResult, Decision
from scripts.lab import run_lab_cycle as mod


def _make_candidate_json(tmp_path: Path, *, factor_id: str = "rdagent_Alpha_aaaaaaaa") -> Path:
    ws = tmp_path / "cand" / factor_id
    ws.mkdir(parents=True, exist_ok=True)
    code_path = ws / "factor.py"
    code_path.write_text('"""stub"""\n', encoding="utf-8")

    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2025-01-02"), "SH600000"), (pd.Timestamp("2025-01-03"), "SH600000")],
        names=["datetime", "instrument"],
    )
    values_path = ws / "values.parquet"
    pd.DataFrame({"Alpha": [0.1, 0.2]}, index=idx, dtype="float64").to_parquet(values_path)

    cand = CandidateFactorPackage(
        factor_id=factor_id,
        name="Alpha",
        source="rdagent",
        hypothesis="h",
        formulation="f",
        code_path=code_path,
        values_path=values_path,
        universe="csi300",
        date_range=(date(2025, 1, 2), date(2025, 1, 3)),
        lab_metrics={"x": 1.0},
        parent_loop=0,
        created_at=datetime(2026, 4, 20, tzinfo=timezone.utc),
        lab_run_id="2026-04-20_01-19-42-374811",
    )
    c1 = ws / "c1.json"
    c1.write_text(cand.model_dump_json(indent=2), encoding="utf-8")
    return c1


def _make_cert(cand: CandidateFactorPackage, decision: Decision) -> CertifiedFactorRecord:
    passed = decision == Decision.PASS
    return CertifiedFactorRecord(
        factor_id=cand.factor_id,
        candidate=cand,
        profile_name="default",
        profile_hash="a" * 64,
        decision=decision,
        overall_score=0.9 if passed else 0.4,
        check_results=[
            CheckResult(
                name="coverage",
                passed=passed,
                score=0.9 if passed else 0.4,
                threshold=0.8,
                detail={},
                elapsed_ms=1,
            )
        ],
        backtest_metrics={},
        validated_at=datetime(2026, 4, 20, tzinfo=timezone.utc),
        validator_version="0.1.0",
        notes=None,
    )


def test_run_cycle_happy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    c1 = _make_candidate_json(tmp_path)
    export_summary = SimpleNamespace(
        exports=[SimpleNamespace(package_path=c1)],
        skipped=[],
    )
    monkeypatch.setattr(mod, "export_rdagent_log_tree", lambda **_: [export_summary])

    calls = {"validate": 0, "promote": 0}

    def _validate(cand, profile_path, **kwargs):
        calls["validate"] += 1
        return _make_cert(cand, Decision.PASS)

    def _promote(**kwargs):
        calls["promote"] += 1
        return 0, "ok"

    monkeypatch.setattr(mod, "validate_candidate", _validate)
    monkeypatch.setattr(mod, "promote", _promote)

    summary = mod.run_cycle(
        log_root=tmp_path / "log",
        log_run_dir=None,
        run_filter=None,
        workspace_candidates_dir=tmp_path / "cands",
        rdagent_workspace_root=tmp_path / "ws",
        exploratory_profile=tmp_path / "exploratory.yaml",
        default_profile=tmp_path / "default.yaml",
        report_dir=tmp_path / "reports",
        cert_dir=tmp_path / "certs",
        registry_data_dir=tmp_path / "registry_data",
        registry_parquet_dir=tmp_path / "registry_parquet",
        registry_parquet_version=1,
        universe="csi300",
        overwrite_export=False,
        promote_tags=["production"],
        allow_promote_overwrite=False,
        cycle_id="cycle-001",
        feedback_dir=tmp_path / "feedback",
    )

    assert summary.export_candidates == 1
    assert summary.exploratory_pass == 1
    assert summary.default_pass == 1
    assert summary.promoted == 1
    assert calls["validate"] == 2
    assert calls["promote"] == 1
    assert (tmp_path / "reports" / "lab_cycle_cycle-001.json").exists()
    md_file = tmp_path / "reports" / "lab_cycle_cycle-001.md"
    assert md_file.exists()
    md_text = md_file.read_text(encoding="utf-8")
    assert "`n" not in md_text, "markdown report must use real newlines, not PowerShell backtick-n"
    assert md_text.startswith("# Lab Cycle Report")
    assert "\n- started_at:" in md_text
    # E.5：feedback bundle 应自动重建
    assert summary.feedback_bundle_path is not None
    assert summary.feedback_error is None
    assert (tmp_path / "feedback" / "latest.json").exists()
    assert (tmp_path / "feedback" / "latest.md").exists()


def test_exploratory_blocks_default_and_promote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    c1 = _make_candidate_json(tmp_path)
    export_summary = SimpleNamespace(exports=[SimpleNamespace(package_path=c1)], skipped=[])
    monkeypatch.setattr(mod, "export_rdagent_log_tree", lambda **_: [export_summary])

    def _validate(cand, profile_path, **kwargs):
        return _make_cert(cand, Decision.HOLD)

    monkeypatch.setattr(mod, "validate_candidate", _validate)
    monkeypatch.setattr(mod, "promote", lambda **_: (_ for _ in ()).throw(RuntimeError("should not call")))

    summary = mod.run_cycle(
        log_root=tmp_path / "log",
        log_run_dir=None,
        run_filter=None,
        workspace_candidates_dir=tmp_path / "cands",
        rdagent_workspace_root=tmp_path / "ws",
        exploratory_profile=tmp_path / "exploratory.yaml",
        default_profile=tmp_path / "default.yaml",
        report_dir=tmp_path / "reports",
        cert_dir=tmp_path / "certs",
        registry_data_dir=tmp_path / "registry_data",
        registry_parquet_dir=tmp_path / "registry_parquet",
        registry_parquet_version=1,
        universe="csi300",
        overwrite_export=False,
        promote_tags=["production"],
        allow_promote_overwrite=False,
        cycle_id="cycle-002",
        feedback_dir=tmp_path / "feedback",
    )

    assert summary.exploratory_pass == 0
    assert summary.default_pass == 0
    assert summary.promoted == 0
    assert summary.failed == 1


def test_invalid_candidate_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{bad", encoding="utf-8")
    export_summary = SimpleNamespace(exports=[SimpleNamespace(package_path=bad)], skipped=[])
    monkeypatch.setattr(mod, "export_rdagent_log_tree", lambda **_: [export_summary])
    monkeypatch.setattr(mod, "validate_candidate", lambda *a, **k: None)
    monkeypatch.setattr(mod, "promote", lambda **_: (0, "ok"))

    summary = mod.run_cycle(
        log_root=tmp_path / "log",
        log_run_dir=None,
        run_filter=None,
        workspace_candidates_dir=tmp_path / "cands",
        rdagent_workspace_root=tmp_path / "ws",
        exploratory_profile=tmp_path / "exploratory.yaml",
        default_profile=tmp_path / "default.yaml",
        report_dir=tmp_path / "reports",
        cert_dir=tmp_path / "certs",
        registry_data_dir=tmp_path / "registry_data",
        registry_parquet_dir=tmp_path / "registry_parquet",
        registry_parquet_version=1,
        universe="csi300",
        overwrite_export=False,
        promote_tags=["production"],
        allow_promote_overwrite=False,
        cycle_id="cycle-003",
        feedback_dir=tmp_path / "feedback",
    )

    assert summary.export_candidates == 1
    assert summary.failed == 1
    assert summary.candidates[0].factor_id == "<invalid>"


def test_main_wires_to_run_cycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "factor_lab.yaml"
    cfg.write_text(
        json.dumps(
            {
                "validation": {
                    "profile_dir": "factor_validation/profiles",
                    "report_dir": "factor_validation/reports",
                },
                "registry": {
                    "data_dir": "factor_registry/data",
                    "parquet_dir": "factor_registry/parquet",
                    "current_parquet_version": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    called = {}

    def _run_cycle(**kwargs):
        called.update(kwargs)
        return mod.CycleSummary(
            cycle_id="x",
            started_at="a",
            finished_at="b",
            export_runs=0,
            export_candidates=0,
            export_skipped=0,
            exploratory_pass=0,
            default_pass=0,
            promoted=0,
            failed=0,
            candidates=[],
        )

    monkeypatch.setattr(mod, "run_cycle", _run_cycle)
    rc = mod.main(["--config", str(cfg), "--cycle-id", "cycle-main", "--log-level", "WARNING"])
    assert rc == 0
    assert called["cycle_id"] == "cycle-main"
