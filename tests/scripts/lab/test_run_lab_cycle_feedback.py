"""
E.5 单测：run_lab_cycle 末尾 feedback bundle 重建钩子。

覆盖场景：

1. 默认开钩子：cycle 结束后 latest.json + latest.md 存在，summary.feedback_* 字段正确。
2. ``--skip-feedback-rebuild``：跳过钩子，summary.feedback_bundle_path is None，不写文件。
3. 钩子抛异常：不影响 cycle 返回，summary.feedback_error 记录错误串。
4. ``--feedback-max-cycles`` CLI 参数传导到 run_cycle。
"""

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
        lab_metrics={},
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


def _monkey_export_and_validate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, decision: Decision):
    c1 = _make_candidate_json(tmp_path)
    export_summary = SimpleNamespace(exports=[SimpleNamespace(package_path=c1)], skipped=[])
    monkeypatch.setattr(mod, "export_rdagent_log_tree", lambda **_: [export_summary])
    monkeypatch.setattr(
        mod, "validate_candidate", lambda cand, profile_path, **kw: _make_cert(cand, decision)
    )
    monkeypatch.setattr(mod, "promote", lambda **_: (0, "ok"))


# -------------------------------------------------------------- 正常钩子


def test_feedback_hook_writes_latest_and_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _monkey_export_and_validate(tmp_path, monkeypatch, Decision.PASS)
    feedback_dir = tmp_path / "feedback"

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
        cycle_id="cycle-feedback-ok",
        feedback_dir=feedback_dir,
    )

    assert summary.feedback_bundle_path is not None
    assert summary.feedback_error is None
    # cycle-feedback-ok 是本测试中唯一一个 cycle 报告 → bundle 应收进
    assert summary.feedback_cycles_included == 1
    assert (feedback_dir / "latest.json").exists()
    assert (feedback_dir / "latest.md").exists()
    hist = list((feedback_dir / "history").glob("cycle-feedback-ok.json"))
    assert hist, "history 目录应按 cycle_id 命名"

    # JSON 报告里也要含 feedback 分区
    rep = json.loads(
        (tmp_path / "reports" / "lab_cycle_cycle-feedback-ok.json").read_text(encoding="utf-8")
    )
    assert rep["feedback"]["bundle_path"] == str(feedback_dir / "latest.json")
    assert rep["feedback"]["cycles_included"] == 1
    assert rep["feedback"]["error"] is None


# -------------------------------------------------------------- 跳过钩子


def test_skip_feedback_rebuild_disables_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _monkey_export_and_validate(tmp_path, monkeypatch, Decision.PASS)
    feedback_dir = tmp_path / "feedback"

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
        cycle_id="cycle-skip",
        feedback_dir=feedback_dir,
        skip_feedback_rebuild=True,
    )

    assert summary.feedback_bundle_path is None
    assert summary.feedback_cycles_included == 0
    assert summary.feedback_error is None
    assert not feedback_dir.exists()  # 压根没建


# -------------------------------------------------------------- 钩子失败不污染 cycle


def test_feedback_failure_does_not_break_cycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _monkey_export_and_validate(tmp_path, monkeypatch, Decision.PASS)
    feedback_dir = tmp_path / "feedback"

    def _boom(**_: object):
        raise RuntimeError("synthetic aggregator failure")

    monkeypatch.setattr(mod, "build_feedback_bundle", _boom)

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
        cycle_id="cycle-boom",
        feedback_dir=feedback_dir,
    )

    assert summary.promoted == 1  # 主流程正常跑完
    assert summary.feedback_bundle_path is None
    assert summary.feedback_error is not None
    assert "synthetic aggregator failure" in summary.feedback_error


# -------------------------------------------------------------- CLI flag 传导


def test_cli_passes_feedback_flags_to_run_cycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "factor_lab.yaml"
    cfg.write_text("{}", encoding="utf-8")

    received = {}

    def _fake_run_cycle(**kwargs):
        received.update(kwargs)
        return mod.CycleSummary(
            cycle_id=kwargs.get("cycle_id") or "x",
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

    monkeypatch.setattr(mod, "run_cycle", _fake_run_cycle)

    rc = mod.main(
        [
            "--config",
            str(cfg),
            "--cycle-id",
            "cli-feedback",
            "--feedback-dir",
            str(tmp_path / "feedback"),
            "--feedback-max-cycles",
            "3",
            "--skip-feedback-rebuild",
            "--log-level",
            "WARNING",
        ]
    )
    assert rc == 0
    assert received["feedback_dir"] == tmp_path / "feedback"
    assert received["feedback_max_cycles"] == 3
    assert received["skip_feedback_rebuild"] is True
