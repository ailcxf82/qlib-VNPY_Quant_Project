"""单元测试：``scripts.promote.validate_candidate``。

策略：
* 使用 ``manual_grandfathered.yaml``（只启用 coverage，无外部数据依赖），保证测试不
  需要 OOS label / 参照因子 parquet。
* tmp_path 下造一份单列因子 parquet 并搭 candidate JSON；调 run() 后检查证书落盘、
  exit_code、解析结果一致。
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from factor_lab.exporters.schema import CandidateFactorPackage
from factor_validation.schema import CertifiedFactorRecord, Decision
from scripts.promote import validate_candidate as vc


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROFILE = (
    _PROJECT_ROOT / "factor_validation" / "profiles" / "manual_grandfathered.yaml"
)


# ---------------------------------------------------------------- fixtures


def _make_candidate_artifacts(
    tmp_path: Path,
    *,
    name: str = "SlowAlpha",
    nan_ratio: float = 0.0,
) -> tuple[Path, Path]:
    """返回 (code_path, values_path)。"""
    ws = tmp_path / "candidates" / "test_cand"
    ws.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range("2024-06-03", periods=80, freq="B")
    stocks = [f"SH{600000 + i:06d}" for i in range(12)]
    idx = pd.MultiIndex.from_product(
        [dates, stocks], names=["datetime", "instrument"]
    )
    rng = np.random.default_rng(0)
    vals = rng.uniform(-1.0, 1.0, size=len(idx))
    if nan_ratio > 0:
        mask = rng.random(len(idx)) < nan_ratio
        vals = vals.copy()
        vals[mask] = np.nan
    df = pd.DataFrame({name: vals.astype("float64")}, index=idx)
    values_path = ws / "values.parquet"
    df.to_parquet(values_path)

    code_path = ws / "factor.py"
    code_path.write_text(
        '"""stub"""\n\ndef calculate(df):\n    return df\n', encoding="utf-8"
    )
    return code_path, values_path


def _make_candidate_obj(tmp_path: Path, *, name: str, nan_ratio: float) -> CandidateFactorPackage:
    code_path, values_path = _make_candidate_artifacts(
        tmp_path, name=name, nan_ratio=nan_ratio
    )
    return CandidateFactorPackage(
        factor_id=f"manual_{name}_abcd1234ef",
        name=name,
        source="manual",
        hypothesis="test hypothesis",
        formulation="test formulation",
        code_path=code_path,
        values_path=values_path,
        universe="csi300",
        date_range=(date(2024, 6, 3), date(2024, 10, 1)),
        lab_metrics={},
        parent_loop=None,
        created_at=datetime(2026, 4, 19, tzinfo=timezone.utc),
        lab_run_id="cli-test-run-00000001",
    )


def _write_candidate_json(obj: CandidateFactorPackage, path: Path) -> None:
    path.write_text(obj.model_dump_json(indent=2), encoding="utf-8")


# ---------------------------------------------------------------- happy path


def test_validate_from_json_pass(tmp_path: Path) -> None:
    cand = _make_candidate_obj(tmp_path, name="SlowAlpha", nan_ratio=0.0)
    cand_json = tmp_path / "candidate.json"
    _write_candidate_json(cand, cand_json)

    out_cert = tmp_path / "cert.json"
    code = vc.main(
        [
            "--profile",
            str(_PROFILE),
            "--candidate-json",
            str(cand_json),
            "--out-cert",
            str(out_cert),
        ]
    )
    assert code == vc.EXIT_PASS
    assert out_cert.exists()
    cert = CertifiedFactorRecord.model_validate_json(out_cert.read_text("utf-8"))
    assert cert.decision == Decision.PASS
    assert cert.factor_id == cand.factor_id
    # coverage check 必须存在
    assert any(r.name == "coverage" for r in cert.check_results)


def test_validate_from_json_fail_low_coverage(tmp_path: Path) -> None:
    """NaN 覆盖率过低 → coverage FAIL → decision=FAIL → EXIT_FAIL。"""
    cand = _make_candidate_obj(tmp_path, name="SparseAlpha", nan_ratio=0.5)
    cand_json = tmp_path / "candidate.json"
    _write_candidate_json(cand, cand_json)

    out_cert = tmp_path / "cert.json"
    code = vc.main(
        [
            "--profile",
            str(_PROFILE),
            "--candidate-json",
            str(cand_json),
            "--out-cert",
            str(out_cert),
        ]
    )
    assert code == vc.EXIT_FAIL
    assert out_cert.exists()
    cert = CertifiedFactorRecord.model_validate_json(out_cert.read_text("utf-8"))
    assert cert.decision == Decision.FAIL


def test_validate_ad_hoc_cli_mode(tmp_path: Path) -> None:
    """不给 --candidate-json，完全从 CLI 参数拼装 candidate。"""
    code_path, values_path = _make_candidate_artifacts(
        tmp_path, name="MyFactor", nan_ratio=0.0
    )
    out_cert = tmp_path / "cert.json"
    code = vc.main(
        [
            "--profile",
            str(_PROFILE),
            "--out-cert",
            str(out_cert),
            "--factor-id",
            "manual_MyFactor_12345678",
            "--name",
            "MyFactor",
            "--source",
            "manual",
            "--hypothesis",
            "just for test",
            "--formulation",
            "x = 1",
            "--code-path",
            str(code_path),
            "--values-path",
            str(values_path),
            "--universe",
            "csi300",
            "--date-range",
            "2024-06-03",
            "2024-10-01",
        ]
    )
    assert code == vc.EXIT_PASS
    cert = CertifiedFactorRecord.model_validate_json(out_cert.read_text("utf-8"))
    assert cert.decision == Decision.PASS
    assert cert.factor_id == "manual_MyFactor_12345678"


def test_validate_ad_hoc_missing_fields_errors(tmp_path: Path) -> None:
    """即席模式下缺必填字段 → SystemExit。"""
    code_path, values_path = _make_candidate_artifacts(tmp_path, name="F", nan_ratio=0.0)
    with pytest.raises(SystemExit):
        vc.main(
            [
                "--profile",
                str(_PROFILE),
                "--code-path",
                str(code_path),
                "--values-path",
                str(values_path),
                # 故意不给 factor-id / name / source 等
            ]
        )


def test_validate_bad_profile_path(tmp_path: Path) -> None:
    cand = _make_candidate_obj(tmp_path, name="Foo", nan_ratio=0.0)
    cand_json = tmp_path / "candidate.json"
    _write_candidate_json(cand, cand_json)

    code = vc.main(
        [
            "--profile",
            str(tmp_path / "no_such_profile.yaml"),
            "--candidate-json",
            str(cand_json),
        ]
    )
    assert code == vc.EXIT_BAD_ARGS


def test_summarize_contains_key_fields(tmp_path: Path) -> None:
    cand = _make_candidate_obj(tmp_path, name="Beta", nan_ratio=0.0)
    cand_json = tmp_path / "candidate.json"
    _write_candidate_json(cand, cand_json)

    import argparse

    ns = argparse.Namespace(
        profile=_PROFILE,
        candidate_json=cand_json,
        out_cert=None,
        factor_id=None,
        name=None,
        source=None,
        hypothesis=None,
        formulation=None,
        code_path=None,
        values_path=None,
        universe=None,
        date_range=None,
        parent_loop=None,
        lab_run_id=None,
        created_at=None,
        notes=None,
        log_level="INFO",
    )
    cert, code = vc.run(ns)
    s = vc.summarize(cert)
    assert cand.factor_id in s
    assert "coverage" in s
    assert "PASS" in s
    assert code == vc.EXIT_PASS
