"""单元测试：``scripts.promote.promote_certified``。

策略：
* tmp_path 下自建 registry/data + registry/parquet + factors_v1.parquet + 一份 PASS 证书。
* 模拟 config.yaml 的 registry 段落（指向 tmp_path 内的目录）。
* 验证各种分支：OK / not-PASS / parquet 缺列 / registry 已 active。
"""

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
from scripts.promote import promote_certified as pc


# --------------------------------------------------------------- fixtures


def _make_cert(
    *,
    factor_id: str,
    name: str,
    decision: Decision,
    code_path: Path,
    values_path: Path,
) -> CertifiedFactorRecord:
    cand = CandidateFactorPackage(
        factor_id=factor_id,
        name=name,
        source="manual",
        hypothesis="h",
        formulation="f",
        code_path=code_path,
        values_path=values_path,
        universe="csi300",
        date_range=(date(2024, 6, 3), date(2024, 10, 1)),
        lab_metrics={},
        parent_loop=None,
        created_at=datetime(2026, 4, 19, tzinfo=timezone.utc),
        lab_run_id="promote-test-run-0001",
    )
    check = CheckResult(
        name="coverage",
        passed=(decision == Decision.PASS),
        score=1.0 if decision == Decision.PASS else 0.1,
        threshold=0.85,
        detail={},
        elapsed_ms=1,
    )
    return CertifiedFactorRecord(
        factor_id=factor_id,
        candidate=cand,
        profile_name="manual_grandfathered",
        profile_hash="0" * 64,
        decision=decision,
        overall_score=0.99 if decision == Decision.PASS else 0.1,
        check_results=[check],
        validated_at=datetime(2026, 4, 19, tzinfo=timezone.utc),
        validator_version="0.1.0",
    )


def _setup_registry(tmp_path: Path, *, name: str) -> tuple[Path, Path, Path, Path, Path]:
    """
    返回 (config_path, data_dir, parquet_dir, code_path, values_path)。
    parquet_dir 内已经有 factors_v1.parquet，包含 name 列。
    """
    data_dir = tmp_path / "registry" / "data"
    parquet_dir = tmp_path / "registry" / "parquet"
    data_dir.mkdir(parents=True)
    parquet_dir.mkdir(parents=True)

    # 构造小型 parquet
    dates = pd.date_range("2024-06-03", periods=20, freq="B")
    stocks = [f"SH{600000 + i:06d}" for i in range(4)]
    idx = pd.MultiIndex.from_product([dates, stocks], names=["datetime", "instrument"])
    df = pd.DataFrame(
        {name: np.linspace(0.1, 0.9, len(idx)).astype("float64")}, index=idx
    )
    store = ParquetStore(parquet_dir)
    store.write_version(df, 1)

    # 造一份候选物料（证书.candidate 里引用的 code_path/values_path 需要存在）
    cand_ws = tmp_path / "cand" / f"manual_{name}_deadbeef12"
    cand_ws.mkdir(parents=True)
    values_path = cand_ws / "values.parquet"
    df[[name]].astype("float64").to_parquet(values_path)
    code_path = cand_ws / "factor.py"
    code_path.write_text('"""stub"""\n', encoding="utf-8")

    # 写 config
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
    return config_path, data_dir, parquet_dir, code_path, values_path


# -------------------------------------------------------------------- tests


def test_promote_happy_path(tmp_path: Path) -> None:
    name = "AlphaOne"
    cfg, data_dir, parquet_dir, code_p, val_p = _setup_registry(tmp_path, name=name)
    cert = _make_cert(
        factor_id="manual_AlphaOne_deadbeef12",
        name=name,
        decision=Decision.PASS,
        code_path=code_p,
        values_path=val_p,
    )
    cert_path = tmp_path / "cert.json"
    cert_path.write_text(cert.model_dump_json(indent=2), encoding="utf-8")

    code, msg = pc.promote(
        cert_path=cert_path,
        parquet_version=1,
        registry_data_dir=data_dir,
        parquet_store_dir=parquet_dir,
        tags=["test"],
    )
    assert code == pc.EXIT_OK, msg
    reg = FactorRegistry(data_dir)
    rec = reg.get(cert.factor_id)
    assert rec.status == ProductionStatus.ACTIVE
    assert rec.parquet_version == 1
    assert rec.tags == ["test"]


def test_promote_rejects_non_pass(tmp_path: Path) -> None:
    name = "HoldAlpha"
    cfg, data_dir, parquet_dir, code_p, val_p = _setup_registry(tmp_path, name=name)
    cert = _make_cert(
        factor_id="manual_HoldAlpha_deadbeef12",
        name=name,
        decision=Decision.HOLD,
        code_path=code_p,
        values_path=val_p,
    )
    cert_path = tmp_path / "cert.json"
    cert_path.write_text(cert.model_dump_json(indent=2), encoding="utf-8")

    code, msg = pc.promote(
        cert_path=cert_path,
        parquet_version=1,
        registry_data_dir=data_dir,
        parquet_store_dir=parquet_dir,
        tags=[],
    )
    assert code == pc.EXIT_NOT_PASS, msg
    # 注册表应保持空
    reg = FactorRegistry(data_dir)
    assert cert.factor_id not in reg


def test_promote_parquet_missing_column(tmp_path: Path) -> None:
    name_in_parquet = "AlphaA"
    cfg, data_dir, parquet_dir, code_p, val_p = _setup_registry(
        tmp_path, name=name_in_parquet
    )
    # 造一个证书，name 不在 parquet 里（用 AlphaB）
    alt_ws = tmp_path / "altcand"
    alt_ws.mkdir()
    dates = pd.date_range("2024-06-03", periods=20, freq="B")
    stocks = [f"SH{600000 + i:06d}" for i in range(4)]
    idx = pd.MultiIndex.from_product(
        [dates, stocks], names=["datetime", "instrument"]
    )
    df = pd.DataFrame(
        {"AlphaB": np.linspace(0.1, 0.9, len(idx)).astype("float64")}, index=idx
    )
    alt_values = alt_ws / "values.parquet"
    df.to_parquet(alt_values)
    alt_code = alt_ws / "factor.py"
    alt_code.write_text('"""stub"""\n', encoding="utf-8")

    cert = _make_cert(
        factor_id="manual_AlphaB_deadbeef12",
        name="AlphaB",
        decision=Decision.PASS,
        code_path=alt_code,
        values_path=alt_values,
    )
    cert_path = tmp_path / "cert.json"
    cert_path.write_text(cert.model_dump_json(indent=2), encoding="utf-8")

    code, msg = pc.promote(
        cert_path=cert_path,
        parquet_version=1,
        registry_data_dir=data_dir,
        parquet_store_dir=parquet_dir,
        tags=[],
    )
    assert code == pc.EXIT_PARQUET_MISSING, msg


def test_promote_parquet_version_not_exists(tmp_path: Path) -> None:
    name = "AlphaOne"
    cfg, data_dir, parquet_dir, code_p, val_p = _setup_registry(tmp_path, name=name)
    cert = _make_cert(
        factor_id="manual_AlphaOne_deadbeef12",
        name=name,
        decision=Decision.PASS,
        code_path=code_p,
        values_path=val_p,
    )
    cert_path = tmp_path / "cert.json"
    cert_path.write_text(cert.model_dump_json(indent=2), encoding="utf-8")

    # 版本 99 不存在
    code, msg = pc.promote(
        cert_path=cert_path,
        parquet_version=99,
        registry_data_dir=data_dir,
        parquet_store_dir=parquet_dir,
        tags=[],
    )
    assert code == pc.EXIT_PARQUET_MISSING, msg


def test_promote_double_register_requires_overwrite(tmp_path: Path) -> None:
    name = "AlphaOne"
    cfg, data_dir, parquet_dir, code_p, val_p = _setup_registry(tmp_path, name=name)
    cert = _make_cert(
        factor_id="manual_AlphaOne_deadbeef12",
        name=name,
        decision=Decision.PASS,
        code_path=code_p,
        values_path=val_p,
    )
    cert_path = tmp_path / "cert.json"
    cert_path.write_text(cert.model_dump_json(indent=2), encoding="utf-8")

    code, _ = pc.promote(
        cert_path=cert_path,
        parquet_version=1,
        registry_data_dir=data_dir,
        parquet_store_dir=parquet_dir,
        tags=[],
    )
    assert code == pc.EXIT_OK

    # 第二次不允许覆盖 → REGISTRY_ERROR
    code2, _ = pc.promote(
        cert_path=cert_path,
        parquet_version=1,
        registry_data_dir=data_dir,
        parquet_store_dir=parquet_dir,
        tags=[],
    )
    assert code2 == pc.EXIT_REGISTRY_ERROR

    # 加 overwrite 再来一次 → OK
    code3, _ = pc.promote(
        cert_path=cert_path,
        parquet_version=1,
        registry_data_dir=data_dir,
        parquet_store_dir=parquet_dir,
        tags=["refreshed"],
        allow_overwrite=True,
    )
    assert code3 == pc.EXIT_OK
    reg = FactorRegistry(data_dir)
    assert reg.get(cert.factor_id).tags == ["refreshed"]


def test_promote_bad_cert_json_path(tmp_path: Path) -> None:
    code, msg = pc.promote(
        cert_path=tmp_path / "no_such.json",
        parquet_version=1,
        registry_data_dir=tmp_path / "r",
        parquet_store_dir=tmp_path / "p",
        tags=[],
    )
    assert code == pc.EXIT_BAD_ARGS


def test_promote_cli_main(tmp_path: Path) -> None:
    name = "AlphaOne"
    cfg, data_dir, parquet_dir, code_p, val_p = _setup_registry(tmp_path, name=name)
    cert = _make_cert(
        factor_id="manual_AlphaOne_deadbeef12",
        name=name,
        decision=Decision.PASS,
        code_path=code_p,
        values_path=val_p,
    )
    cert_path = tmp_path / "cert.json"
    cert_path.write_text(cert.model_dump_json(indent=2), encoding="utf-8")

    rc = pc.main(
        [
            "--cert",
            str(cert_path),
            "--config",
            str(cfg),
            "--tags",
            "production,legacy",
        ]
    )
    assert rc == pc.EXIT_OK
    reg = FactorRegistry(data_dir)
    assert reg.get(cert.factor_id).tags == ["production", "legacy"]
