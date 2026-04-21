"""单元测试：``scripts.promote.retire_factor``。"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from factor_lab.exporters.schema import CandidateFactorPackage
from factor_registry.registry import FactorRegistry
from factor_registry.schema import ProductionStatus
from factor_registry.store import ParquetStore
from factor_validation.schema import (
    CertifiedFactorRecord,
    CheckResult,
    Decision,
)
from scripts.promote import retire_factor as rf


# -------------------------------------------------------------- helpers


def _prepare_registry_with_factor(
    tmp_path: Path, *, factor_id: str, name: str
) -> tuple[Path, Path]:
    """在 tmp_path 下搭一个含单个 active 因子的 registry。返回 (config_path, data_dir)。"""
    data_dir = tmp_path / "registry" / "data"
    parquet_dir = tmp_path / "registry" / "parquet"
    data_dir.mkdir(parents=True)
    parquet_dir.mkdir(parents=True)

    # 小 parquet
    dates = pd.date_range("2024-06-03", periods=10, freq="B")
    stocks = [f"SH{600000 + i:06d}" for i in range(3)]
    idx = pd.MultiIndex.from_product(
        [dates, stocks], names=["datetime", "instrument"]
    )
    df = pd.DataFrame(
        {name: np.linspace(0.1, 0.9, len(idx)).astype("float64")}, index=idx
    )
    ParquetStore(parquet_dir).write_version(df, 1)

    # candidate 物料
    cand_ws = tmp_path / "cand" / factor_id
    cand_ws.mkdir(parents=True)
    values_path = cand_ws / "values.parquet"
    df[[name]].astype("float64").to_parquet(values_path)
    code_path = cand_ws / "factor.py"
    code_path.write_text('"""stub"""\n', encoding="utf-8")

    cand = CandidateFactorPackage(
        factor_id=factor_id,
        name=name,
        source="manual",
        hypothesis="h",
        formulation="f",
        code_path=code_path,
        values_path=values_path,
        universe="csi300",
        date_range=(date(2024, 6, 3), date(2024, 6, 17)),
        lab_metrics={},
        parent_loop=None,
        created_at=datetime(2026, 4, 19, tzinfo=timezone.utc),
        lab_run_id="retire-test-run-0001",
    )
    cert = CertifiedFactorRecord(
        factor_id=factor_id,
        candidate=cand,
        profile_name="manual_grandfathered",
        profile_hash="0" * 64,
        decision=Decision.PASS,
        overall_score=1.0,
        check_results=[
            CheckResult(
                name="coverage",
                passed=True,
                score=1.0,
                threshold=0.85,
                detail={},
                elapsed_ms=1,
            )
        ],
        validated_at=datetime(2026, 4, 19, tzinfo=timezone.utc),
        validator_version="0.1.0",
    )
    reg = FactorRegistry(data_dir)
    reg.register(cert, parquet_version=1, tags=["legacy"])

    config_path = tmp_path / "factor_lab.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "registry": {
                    "data_dir": str(data_dir),
                    "parquet_dir": str(parquet_dir),
                    "current_parquet_version": 1,
                }
            }
        ),
        encoding="utf-8",
    )
    return config_path, data_dir


# -------------------------------------------------------------- tests


def test_retire_single_factor(tmp_path: Path) -> None:
    fid = "manual_AlphaOne_deadbeef12"
    cfg, data_dir = _prepare_registry_with_factor(tmp_path, factor_id=fid, name="AlphaOne")

    code, results = rf.retire_many(
        factor_ids=[fid],
        reason="oracle FAIL",
        registry_data_dir=data_dir,
    )
    assert code == rf.EXIT_OK
    assert len(results) == 1
    assert results[0]["ok"] is True

    reg = FactorRegistry(data_dir)
    rec = reg.get(fid)
    assert rec.status == ProductionStatus.RETIRED
    # certificate 应搬到 retired/
    assert rec.certificate_path.parts[0] == "retired"
    # 物理 parquet 完全没动（文件仍在）
    assert (tmp_path / "registry" / "parquet" / "factors_v1.parquet").exists()


def test_retire_unknown_factor_returns_partial(tmp_path: Path) -> None:
    fid = "manual_Existing_deadbeef12"
    cfg, data_dir = _prepare_registry_with_factor(tmp_path, factor_id=fid, name="Existing")

    code, results = rf.retire_many(
        factor_ids=[fid, "manual_Ghost_ffffffff12"],
        reason="cleanup",
        registry_data_dir=data_dir,
    )
    assert code == rf.EXIT_PARTIAL
    ok_map = {r["factor_id"]: r for r in results}
    assert ok_map[fid]["ok"] is True
    assert ok_map["manual_Ghost_ffffffff12"]["ok"] is False
    assert "不存在" in ok_map["manual_Ghost_ffffffff12"]["error"]


def test_retire_already_retired_is_skipped(tmp_path: Path) -> None:
    fid = "manual_OldBeta_deadbeef12"
    cfg, data_dir = _prepare_registry_with_factor(tmp_path, factor_id=fid, name="OldBeta")
    # 先手动 retire 一次
    FactorRegistry(data_dir).retire(fid, reason="first")

    code, results = rf.retire_many(
        factor_ids=[fid],
        reason="second",
        registry_data_dir=data_dir,
    )
    assert code == rf.EXIT_OK
    assert results[0]["ok"] is True
    assert results[0].get("skipped") == "already retired"


def test_retire_empty_reason_rejected(tmp_path: Path) -> None:
    code, results = rf.retire_many(
        factor_ids=["manual_foo_deadbeef12"],
        reason="   ",
        registry_data_dir=tmp_path / "r",
    )
    assert code == rf.EXIT_BAD_ARGS


def test_retire_dry_run_does_not_modify(tmp_path: Path) -> None:
    fid = "manual_Gamma_deadbeef12"
    cfg, data_dir = _prepare_registry_with_factor(tmp_path, factor_id=fid, name="Gamma")

    code, results = rf.retire_many(
        factor_ids=[fid],
        reason="maybe",
        registry_data_dir=data_dir,
        dry_run=True,
    )
    assert code == rf.EXIT_OK
    assert results[0].get("dry_run") is True
    # status 应仍为 ACTIVE
    reg = FactorRegistry(data_dir)
    assert reg.get(fid).status == ProductionStatus.ACTIVE


def test_retire_cli_ids_file(tmp_path: Path) -> None:
    fid1 = "manual_Delta_deadbeef12"
    fid2 = "manual_Epsilon_deadbeef13"
    cfg, data_dir = _prepare_registry_with_factor(tmp_path, factor_id=fid1, name="Delta")
    # 注册第二个
    reg = FactorRegistry(data_dir)
    # 复用 _prepare_registry_with_factor 不方便，直接手动构一份 cert
    from factor_lab.exporters.schema import CandidateFactorPackage

    v_path = (tmp_path / "cand" / fid1 / "values.parquet")
    c_path = (tmp_path / "cand" / fid1 / "factor.py")
    cand2 = CandidateFactorPackage(
        factor_id=fid2,
        name="Epsilon",
        source="manual",
        hypothesis="h",
        formulation="f",
        code_path=c_path,
        values_path=v_path,
        universe="csi300",
        date_range=(date(2024, 6, 3), date(2024, 6, 17)),
        lab_metrics={},
        parent_loop=None,
        created_at=datetime(2026, 4, 19, tzinfo=timezone.utc),
        lab_run_id="retire-test-run-0002",
    )
    cert2 = CertifiedFactorRecord(
        factor_id=fid2,
        candidate=cand2,
        profile_name="manual_grandfathered",
        profile_hash="0" * 64,
        decision=Decision.PASS,
        overall_score=1.0,
        check_results=[
            CheckResult(
                name="coverage", passed=True, score=1.0, threshold=0.85,
                detail={}, elapsed_ms=1,
            )
        ],
        validated_at=datetime(2026, 4, 19, tzinfo=timezone.utc),
        validator_version="0.1.0",
    )
    reg.register(cert2, parquet_version=1)

    ids_file = tmp_path / "to_retire.txt"
    ids_file.write_text(
        f"# batch retire\n{fid1}\n\n{fid2}\n", encoding="utf-8"
    )

    rc = rf.main(
        [
            "--ids-file",
            str(ids_file),
            "--reason",
            "batch oracle FAIL",
            "--config",
            str(cfg),
        ]
    )
    assert rc == rf.EXIT_OK
    reg2 = FactorRegistry(data_dir)
    assert reg2.get(fid1).status == ProductionStatus.RETIRED
    assert reg2.get(fid2).status == ProductionStatus.RETIRED
