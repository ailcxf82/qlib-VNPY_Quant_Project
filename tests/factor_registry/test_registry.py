"""单元测试：``factor_registry.registry.FactorRegistry``。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from factor_registry.registry import FactorRegistry, RegistryError
from factor_registry.schema import ProductionStatus
from factor_validation.schema import CheckResult, Decision


# ---------------------------------------------------------------- bootstrap


def test_init_creates_empty_manifest(tmp_path: Path) -> None:
    reg = FactorRegistry(tmp_path)
    assert reg.manifest_path.is_file()
    assert (tmp_path / "certified").is_dir()
    assert (tmp_path / "retired").is_dir()
    assert reg.list_all() == []
    assert reg.last_updated_at is None


def test_init_accepts_existing_manifest(tmp_path: Path) -> None:
    FactorRegistry(tmp_path).save()
    reg2 = FactorRegistry(tmp_path)
    assert reg2.list_all() == []


def test_init_rejects_wrong_schema_version(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema_version": 99, "factors": []}),
        encoding="utf-8",
    )
    with pytest.raises(RegistryError, match="schema_version"):
        FactorRegistry(tmp_path)


def test_init_rejects_non_dict_root(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text("[]", encoding="utf-8")
    with pytest.raises(RegistryError, match="根节点"):
        FactorRegistry(tmp_path)


def test_init_rejects_factors_not_list(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "factors": {}}),
        encoding="utf-8",
    )
    with pytest.raises(RegistryError, match="factors"):
        FactorRegistry(tmp_path)


# ---------------------------------------------------------------- register


def test_register_happy_path(tmp_path: Path, minimal_certified) -> None:
    reg = FactorRegistry(tmp_path)
    rec = reg.register(minimal_certified, parquet_version=1)
    assert rec.factor_id == minimal_certified.factor_id
    assert rec.name == minimal_certified.candidate.name
    assert rec.parquet_column == minimal_certified.candidate.name
    assert rec.status == ProductionStatus.ACTIVE
    assert rec.parquet_version == 1
    # 证书副本已落盘
    cert_file = tmp_path / rec.certificate_path
    assert cert_file.is_file()
    # manifest 已更新 + last_updated_at 写入
    payload = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["last_updated_at"] is not None
    assert len(payload["factors"]) == 1
    assert payload["factors"][0]["factor_id"] == minimal_certified.factor_id


def test_register_persists_across_instances(tmp_path: Path, minimal_certified) -> None:
    reg = FactorRegistry(tmp_path)
    reg.register(minimal_certified, parquet_version=1)
    reg2 = FactorRegistry(tmp_path)
    assert minimal_certified.factor_id in reg2
    assert len(reg2) == 1


def test_register_rejects_non_pass_decision(
    tmp_path: Path, minimal_candidate
) -> None:
    from factor_validation.schema import CertifiedFactorRecord

    check = CheckResult(
        name="ic", passed=False, score=0.1, threshold=0.4, detail={}, elapsed_ms=5
    )
    held = CertifiedFactorRecord(
        factor_id=minimal_candidate.factor_id,
        candidate=minimal_candidate,
        profile_name="default",
        profile_hash="b" * 64,
        decision=Decision.HOLD,
        overall_score=0.3,
        check_results=[check],
        validated_at=datetime(2026, 4, 19, tzinfo=timezone.utc),
        validator_version="0.1.0",
    )
    reg = FactorRegistry(tmp_path)
    with pytest.raises(RegistryError, match="decision=PASS"):
        reg.register(held, parquet_version=1)


def test_register_rejects_wrong_type(tmp_path: Path) -> None:
    reg = FactorRegistry(tmp_path)
    with pytest.raises(RegistryError, match="CertifiedFactorRecord"):
        reg.register({"foo": "bar"}, parquet_version=1)  # type: ignore[arg-type]


def test_register_duplicate_rejected_by_default(
    tmp_path: Path, minimal_certified
) -> None:
    reg = FactorRegistry(tmp_path)
    reg.register(minimal_certified, parquet_version=1)
    with pytest.raises(RegistryError, match="已 active"):
        reg.register(minimal_certified, parquet_version=1)


def test_register_allow_overwrite(tmp_path: Path, minimal_certified) -> None:
    reg = FactorRegistry(tmp_path)
    reg.register(minimal_certified, parquet_version=1)
    # 同一因子，版本升级
    rec = reg.register(minimal_certified, parquet_version=2, allow_overwrite=True)
    assert rec.parquet_version == 2
    assert len(reg) == 1


def test_register_writes_sorted_manifest(
    tmp_path: Path, minimal_candidate, minimal_certified
) -> None:
    """manifest.factors 按 (parquet_version, factor_id) 稳定排序。"""
    from factor_lab.exporters.schema import CandidateFactorPackage
    from factor_validation.schema import CertifiedFactorRecord

    reg = FactorRegistry(tmp_path)
    reg.register(minimal_certified, parquet_version=2)

    # 造一个 v1 的因子（name / id 故意在字典序更靠后）
    other = minimal_candidate.model_copy(
        update={
            "factor_id": "rdagent_ZooFactor_00000001",
            "name": "ZooFactor",
        }
    )
    other_cert = CertifiedFactorRecord(
        factor_id=other.factor_id,
        candidate=other,
        profile_name="default",
        profile_hash="c" * 64,
        decision=Decision.PASS,
        overall_score=0.8,
        check_results=[
            CheckResult(
                name="coverage",
                passed=True,
                score=0.9,
                threshold=0.8,
                detail={},
                elapsed_ms=1,
            )
        ],
        validated_at=datetime(2026, 4, 19, tzinfo=timezone.utc),
        validator_version="0.1.0",
    )
    reg.register(other_cert, parquet_version=1)

    payload = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    ids = [f["factor_id"] for f in payload["factors"]]
    # v1 的 ZooFactor 排在 v2 的前面
    assert ids == [other.factor_id, minimal_certified.factor_id]


# ---------------------------------------------------------------- retire


def test_retire_happy_path(tmp_path: Path, minimal_certified) -> None:
    reg = FactorRegistry(tmp_path)
    reg.register(minimal_certified, parquet_version=1)
    retired = reg.retire(minimal_certified.factor_id, "ic turned negative")
    assert retired.status == ProductionStatus.RETIRED
    assert retired.retire_reason == "ic turned negative"
    assert retired.retired_at is not None
    # 证书从 certified/ 搬到了 retired/
    assert not (tmp_path / "certified" / f"{minimal_certified.factor_id}.json").exists()
    assert (tmp_path / "retired" / f"{minimal_certified.factor_id}.json").is_file()
    # list_active 不再返回
    assert reg.list_active() == []
    assert len(reg.list_all()) == 1


def test_retire_persists(tmp_path: Path, minimal_certified) -> None:
    reg = FactorRegistry(tmp_path)
    reg.register(minimal_certified, parquet_version=1)
    reg.retire(minimal_certified.factor_id, "ic turned negative")
    reg2 = FactorRegistry(tmp_path)
    rec = reg2.get(minimal_certified.factor_id)
    assert rec.status == ProductionStatus.RETIRED


def test_retire_unknown_raises(tmp_path: Path) -> None:
    reg = FactorRegistry(tmp_path)
    with pytest.raises(RegistryError, match="不存在"):
        reg.retire("rdagent_NoSuch_deadbeef", "reason")


def test_retire_already_retired(tmp_path: Path, minimal_certified) -> None:
    reg = FactorRegistry(tmp_path)
    reg.register(minimal_certified, parquet_version=1)
    reg.retire(minimal_certified.factor_id, "r1")
    with pytest.raises(RegistryError, match="已 retired"):
        reg.retire(minimal_certified.factor_id, "r2")


def test_retire_empty_reason(tmp_path: Path, minimal_certified) -> None:
    reg = FactorRegistry(tmp_path)
    reg.register(minimal_certified, parquet_version=1)
    with pytest.raises(RegistryError, match="reason 不能为空"):
        reg.retire(minimal_certified.factor_id, "   ")


# ---------------------------------------------------------------- read


def test_get_and_contains(tmp_path: Path, minimal_certified) -> None:
    reg = FactorRegistry(tmp_path)
    with pytest.raises(RegistryError, match="不存在"):
        reg.get(minimal_certified.factor_id)
    assert minimal_certified.factor_id not in reg
    reg.register(minimal_certified, parquet_version=1)
    assert minimal_certified.factor_id in reg
    assert reg.get(minimal_certified.factor_id).factor_id == minimal_certified.factor_id


def test_list_by_version(
    tmp_path: Path, minimal_candidate, minimal_certified
) -> None:
    from factor_validation.schema import CertifiedFactorRecord

    reg = FactorRegistry(tmp_path)
    reg.register(minimal_certified, parquet_version=1)

    other = minimal_candidate.model_copy(
        update={
            "factor_id": "rdagent_Other_feedbeef",
            "name": "Other",
        }
    )
    other_cert = CertifiedFactorRecord(
        factor_id=other.factor_id,
        candidate=other,
        profile_name="default",
        profile_hash="d" * 64,
        decision=Decision.PASS,
        overall_score=0.8,
        check_results=[
            CheckResult(
                name="coverage",
                passed=True,
                score=0.9,
                threshold=0.8,
                detail={},
                elapsed_ms=1,
            )
        ],
        validated_at=datetime(2026, 4, 19, tzinfo=timezone.utc),
        validator_version="0.1.0",
    )
    reg.register(other_cert, parquet_version=2)

    assert [r.factor_id for r in reg.list_by_version(1)] == [
        minimal_certified.factor_id
    ]
    assert [r.factor_id for r in reg.list_by_version(2)] == [other.factor_id]
    assert reg.list_by_version(9) == []

    # retire v1 因子后 only_active 返回空，only_active=False 仍返回
    reg.retire(minimal_certified.factor_id, "test")
    assert reg.list_by_version(1, only_active=True) == []
    assert len(reg.list_by_version(1, only_active=False)) == 1


def test_last_updated_at_tracked(tmp_path: Path, minimal_certified) -> None:
    reg = FactorRegistry(tmp_path)
    assert reg.last_updated_at is None
    reg.register(minimal_certified, parquet_version=1)
    assert reg.last_updated_at is not None
    first = reg.last_updated_at
    # 手动 load 一次，last_updated_at 要在重新读取后保持一致
    reg.load()
    assert reg.last_updated_at == first
