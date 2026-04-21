"""单元测试：``feature.production_factor_loader``。

重点验证：

* 空 registry 的优雅降级
* 多因子时列顺序稳定（按 registered_at, factor_id）
* align_index 的 reindex 语义（缺失填 NaN，不报错）
* ``from_default_paths`` 正确解析 factor_lab.yaml
* 显式版本 vs registry 推断的优先级
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from factor_lab.exporters.schema import CandidateFactorPackage
from factor_registry.registry import FactorRegistry
from factor_registry.store import ParquetStore
from factor_validation.schema import (
    CertifiedFactorRecord,
    CheckResult,
    Decision,
)
from feature.production_factor_loader import (
    ProductionFactorLoader,
    _parse_factor_lab_config,
    load_active_factors,
)


# --------------------------------------------------------------- helpers


def _mk_parquet(cols: list[str]) -> pd.DataFrame:
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2025-01-02"), "SH600000"),
            (pd.Timestamp("2025-01-02"), "SZ000001"),
            (pd.Timestamp("2025-01-03"), "SH600000"),
            (pd.Timestamp("2025-01-03"), "SZ000001"),
        ],
        names=["datetime", "instrument"],
    )
    data = {c: np.linspace(0.1, 0.4, 4) for c in cols}
    return pd.DataFrame(data, index=idx, dtype="float64")


def _mk_certified(
    candidate: CandidateFactorPackage,
    *,
    name: str,
    factor_id: str,
    profile_hash_char: str = "a",
) -> CertifiedFactorRecord:
    cand = candidate.model_copy(update={"factor_id": factor_id, "name": name})
    return CertifiedFactorRecord(
        factor_id=factor_id,
        candidate=cand,
        profile_name="manual_grandfathered",
        profile_hash=profile_hash_char * 64,
        decision=Decision.PASS,
        overall_score=0.9,
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


# ---------------------------------------------------------- empty registry


def test_empty_registry_returns_empty(tmp_path: Path) -> None:
    registry = FactorRegistry(tmp_path / "data")
    store = ParquetStore(tmp_path / "parquet")
    loader = ProductionFactorLoader(registry=registry, store=store)
    assert loader.list_active_columns() == []
    df = loader.load_active()
    assert df.empty
    assert list(df.index.names) == ["datetime", "instrument"]


def test_empty_registry_with_align_index(tmp_path: Path) -> None:
    registry = FactorRegistry(tmp_path / "data")
    store = ParquetStore(tmp_path / "parquet")
    loader = ProductionFactorLoader(registry=registry, store=store)
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2025-01-02"), "SH600000")],
        names=["datetime", "instrument"],
    )
    df = loader.load_active(align_index=idx)
    assert df.shape == (1, 0)
    assert df.index.equals(idx)


# ---------------------------------------------------------- happy path


def test_load_active_returns_registered_columns(
    tmp_path: Path, minimal_candidate
) -> None:
    registry = FactorRegistry(tmp_path / "data")
    store = ParquetStore(tmp_path / "parquet")

    # 写 parquet v1：两列
    df_orig = _mk_parquet(["AlphaOne", "AlphaTwo"])
    store.write_version(df_orig, 1)

    # 注册两个因子
    c1 = _mk_certified(
        minimal_candidate,
        name="AlphaOne",
        factor_id="rdagent_AlphaOne_deadbeef",
        profile_hash_char="a",
    )
    c2 = _mk_certified(
        minimal_candidate,
        name="AlphaTwo",
        factor_id="rdagent_AlphaTwo_feedface",
        profile_hash_char="b",
    )
    registry.register(c1, parquet_version=1, now=datetime(2026, 4, 19, tzinfo=timezone.utc))
    registry.register(
        c2,
        parquet_version=1,
        now=datetime(2026, 4, 19, 0, 0, 1, tzinfo=timezone.utc),
    )

    loader = ProductionFactorLoader(registry=registry, store=store)
    assert loader.list_active_columns() == ["AlphaOne", "AlphaTwo"]
    out = loader.load_active()
    assert list(out.columns) == ["AlphaOne", "AlphaTwo"]
    pd.testing.assert_frame_equal(out, df_orig[["AlphaOne", "AlphaTwo"]])


def test_load_active_column_order_by_registered_at(
    tmp_path: Path, minimal_candidate
) -> None:
    """先注册的因子列在前；factor_id 按字典序做二次排序。"""
    registry = FactorRegistry(tmp_path / "data")
    store = ParquetStore(tmp_path / "parquet")
    store.write_version(_mk_parquet(["Bravo", "Alpha", "Charlie"]), 1)

    base = datetime(2026, 4, 19, tzinfo=timezone.utc)
    # 反序注册：Charlie → Alpha → Bravo
    registry.register(
        _mk_certified(
            minimal_candidate,
            name="Charlie",
            factor_id="rdagent_Charlie_cccccccc",
            profile_hash_char="c",
        ),
        parquet_version=1,
        now=base,
    )
    registry.register(
        _mk_certified(
            minimal_candidate,
            name="Alpha",
            factor_id="rdagent_Alpha_aaaaaaaa",
            profile_hash_char="a",
        ),
        parquet_version=1,
        now=base + timedelta(seconds=1),
    )
    registry.register(
        _mk_certified(
            minimal_candidate,
            name="Bravo",
            factor_id="rdagent_Bravo_bbbbbbbb",
            profile_hash_char="b",
        ),
        parquet_version=1,
        now=base + timedelta(seconds=2),
    )

    loader = ProductionFactorLoader(registry=registry, store=store)
    assert loader.list_active_columns() == ["Charlie", "Alpha", "Bravo"]


def test_load_active_with_align_index(tmp_path: Path, minimal_candidate) -> None:
    registry = FactorRegistry(tmp_path / "data")
    store = ParquetStore(tmp_path / "parquet")
    store.write_version(_mk_parquet(["AlphaOne"]), 1)
    registry.register(
        _mk_certified(
            minimal_candidate,
            name="AlphaOne",
            factor_id="rdagent_AlphaOne_deadbeef",
        ),
        parquet_version=1,
    )

    # align_index 包含 parquet 没有的日期 → 对应行应为 NaN
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2025-01-02"), "SH600000"),
            (pd.Timestamp("2099-12-31"), "SH600000"),
        ],
        names=["datetime", "instrument"],
    )
    loader = ProductionFactorLoader(registry=registry, store=store)
    out = loader.load_active(align_index=idx)
    assert out.shape == (2, 1)
    assert out.index.equals(idx)
    assert not pd.isna(out.iloc[0, 0])
    assert pd.isna(out.iloc[1, 0])


def test_retired_factor_excluded(tmp_path: Path, minimal_candidate) -> None:
    registry = FactorRegistry(tmp_path / "data")
    store = ParquetStore(tmp_path / "parquet")
    store.write_version(_mk_parquet(["AlphaOne", "AlphaTwo"]), 1)

    registry.register(
        _mk_certified(
            minimal_candidate,
            name="AlphaOne",
            factor_id="rdagent_AlphaOne_deadbeef",
        ),
        parquet_version=1,
    )
    registry.register(
        _mk_certified(
            minimal_candidate,
            name="AlphaTwo",
            factor_id="rdagent_AlphaTwo_feedface",
            profile_hash_char="b",
        ),
        parquet_version=1,
    )
    registry.retire("rdagent_AlphaTwo_feedface", "test retire")

    loader = ProductionFactorLoader(registry=registry, store=store)
    assert loader.list_active_columns() == ["AlphaOne"]
    out = loader.load_active()
    assert list(out.columns) == ["AlphaOne"]


# ---------------------------------------------------------- resolve_version


def test_resolve_version_prefers_explicit(tmp_path: Path) -> None:
    registry = FactorRegistry(tmp_path / "data")
    store = ParquetStore(tmp_path / "parquet")
    store.write_version(_mk_parquet(["AlphaOne"]), 1)
    store.write_version(_mk_parquet(["AlphaTwo"]), 2)

    loader = ProductionFactorLoader(registry=registry, store=store, parquet_version=2)
    assert loader.resolve_version() == 2


def test_resolve_version_explicit_missing_falls_back_none(tmp_path: Path) -> None:
    registry = FactorRegistry(tmp_path / "data")
    store = ParquetStore(tmp_path / "parquet")
    loader = ProductionFactorLoader(
        registry=registry, store=store, parquet_version=99
    )
    assert loader.resolve_version() is None


def test_resolve_version_from_registry_max(tmp_path: Path, minimal_candidate) -> None:
    registry = FactorRegistry(tmp_path / "data")
    store = ParquetStore(tmp_path / "parquet")
    store.write_version(_mk_parquet(["AlphaOne"]), 1)
    store.write_version(_mk_parquet(["AlphaTwo"]), 2)

    registry.register(
        _mk_certified(
            minimal_candidate,
            name="AlphaOne",
            factor_id="rdagent_AlphaOne_deadbeef",
        ),
        parquet_version=1,
    )
    registry.register(
        _mk_certified(
            minimal_candidate,
            name="AlphaTwo",
            factor_id="rdagent_AlphaTwo_feedface",
            profile_hash_char="b",
        ),
        parquet_version=2,
    )
    loader = ProductionFactorLoader(registry=registry, store=store)
    assert loader.resolve_version() == 2
    assert loader.list_active_columns() == ["AlphaTwo"]


# ---------------------------------------------------------- load_columns


def test_load_columns_explicit(tmp_path: Path, minimal_candidate) -> None:
    registry = FactorRegistry(tmp_path / "data")
    store = ParquetStore(tmp_path / "parquet")
    store.write_version(_mk_parquet(["AlphaOne", "AlphaTwo"]), 1)
    registry.register(
        _mk_certified(
            minimal_candidate,
            name="AlphaOne",
            factor_id="rdagent_AlphaOne_deadbeef",
        ),
        parquet_version=1,
    )
    loader = ProductionFactorLoader(registry=registry, store=store)

    got = loader.load_columns(["AlphaTwo"])
    assert list(got.columns) == ["AlphaTwo"]


def test_load_columns_empty_raises(tmp_path: Path) -> None:
    registry = FactorRegistry(tmp_path / "data")
    store = ParquetStore(tmp_path / "parquet")
    loader = ProductionFactorLoader(registry=registry, store=store)
    with pytest.raises(ValueError, match="columns 不能为空"):
        loader.load_columns([])


def test_load_columns_no_version_raises(tmp_path: Path) -> None:
    registry = FactorRegistry(tmp_path / "data")
    store = ParquetStore(tmp_path / "parquet")
    loader = ProductionFactorLoader(registry=registry, store=store)
    with pytest.raises(RuntimeError, match="parquet version"):
        loader.load_columns(["AlphaOne"])


# ---------------------------------------------------------- from_default_paths


def test_from_default_paths_custom_config(
    tmp_path: Path, minimal_candidate
) -> None:
    """自定义 factor_lab.yaml + project_root 路径可以被正确解析。"""
    # 造一套完整目录
    project = tmp_path / "proj"
    (project / "config").mkdir(parents=True)
    (project / "factor_registry" / "data").mkdir(parents=True)
    (project / "factor_registry" / "parquet").mkdir(parents=True)

    cfg = {
        "schema_version": 1,
        "registry": {
            "data_dir": "factor_registry/data",
            "parquet_dir": "factor_registry/parquet",
            "current_parquet_version": 1,
        },
    }
    cfg_path = project / "config" / "factor_lab.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    # 写 parquet + 注册因子
    store = ParquetStore(project / "factor_registry" / "parquet")
    store.write_version(_mk_parquet(["AlphaOne"]), 1)
    registry = FactorRegistry(project / "factor_registry" / "data")
    registry.register(
        _mk_certified(
            minimal_candidate,
            name="AlphaOne",
            factor_id="rdagent_AlphaOne_deadbeef",
        ),
        parquet_version=1,
    )

    loader = ProductionFactorLoader.from_default_paths(
        project_root=project, config_path=cfg_path
    )
    assert loader.resolve_version() == 1
    assert loader.list_active_columns() == ["AlphaOne"]


def test_from_default_paths_missing_config(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="factor_lab"):
        ProductionFactorLoader.from_default_paths(
            project_root=tmp_path, config_path=tmp_path / "not_exist.yaml"
        )


def test_parse_config_rejects_non_int_version(tmp_path: Path) -> None:
    cfg_path = tmp_path / "factor_lab.yaml"
    cfg_path.write_text(
        "registry:\n  current_parquet_version: abc\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="current_parquet_version"):
        _parse_factor_lab_config(cfg_path, tmp_path)


# ---------------------------------------------------------- functional API


def test_load_active_factors_function(tmp_path: Path, minimal_candidate) -> None:
    project = tmp_path / "proj"
    (project / "config").mkdir(parents=True)

    store = ParquetStore(project / "factor_registry" / "parquet")
    store.write_version(_mk_parquet(["AlphaOne"]), 1)
    registry = FactorRegistry(project / "factor_registry" / "data")
    registry.register(
        _mk_certified(
            minimal_candidate,
            name="AlphaOne",
            factor_id="rdagent_AlphaOne_deadbeef",
        ),
        parquet_version=1,
    )
    cfg_path = project / "config" / "factor_lab.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {"registry": {"current_parquet_version": 1}}
        ),
        encoding="utf-8",
    )

    # 注意：monkey-patch module-level _DEFAULT_CONFIG 不便，直接用 from_default_paths 等价路径
    df, cols = load_active_factors(project_root=project, parquet_version=1)
    assert cols == ["AlphaOne"]
    assert list(df.columns) == ["AlphaOne"]
