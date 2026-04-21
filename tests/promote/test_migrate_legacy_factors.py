"""单元测试：``scripts.promote.migrate_legacy_factors.migrate``。

使用 tmp_path 隔离；不动真实 ``git_ignore_folder/`` 与 ``factor_registry/`` 目录。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from factor_registry.registry import FactorRegistry
from factor_registry.schema import ProductionStatus
from factor_registry.store import ParquetStore
from scripts.promote import migrate_legacy_factors as m


_PROFILE = (
    Path(__file__).resolve().parents[2]
    / "factor_validation"
    / "profiles"
    / "manual_grandfathered.yaml"
)


def _make_src_parquet(path: Path, *, cols: list[str], nan_ratio: float = 0.0) -> None:
    # 50 天 x 4 票 = 200 行，每列设定 nan_ratio
    dates = pd.date_range("2024-01-02", periods=50, freq="B")
    stocks = ["SH600000", "SZ000001", "SH600519", "SZ000002"]
    idx = pd.MultiIndex.from_product([dates, stocks], names=["datetime", "instrument"])
    data = {}
    for i, c in enumerate(cols):
        import numpy as np

        arr = np.linspace(0.1, 0.9, len(idx)) + i * 0.01
        if nan_ratio > 0:
            mask = np.random.default_rng(0).random(len(idx)) < nan_ratio
            arr = arr.copy()
            arr[mask] = float("nan")
        data[c] = arr.astype("float64")
    df = pd.DataFrame(data, index=idx, dtype="float64")
    df.to_parquet(path)


# ---------------------------------------------------------- factor_id stability


def test_factor_id_stable_for_same_data(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "src.parquet"
    _make_src_parquet(src, cols=["AlphaOne"])
    df = pd.read_parquet(src).astype("float64")
    id1 = m._factor_id_for("AlphaOne", df["AlphaOne"])
    id2 = m._factor_id_for("AlphaOne", df["AlphaOne"])
    assert id1 == id2
    # 格式：manual_<name>_<10 hex>
    parts = id1.split("_")
    assert parts[0] == "manual"
    assert "_".join(parts[1:-1]) == "AlphaOne"
    assert len(parts[-1]) == 10


# ---------------------------------------------------------- migrate happy path


def test_migrate_happy_path(tmp_path: Path, monkeypatch) -> None:
    # 重定向 WORKSPACE_ROOT 与 project_root 到 tmp_path，避免污染真实工作区
    ws_root = tmp_path / "ws"
    monkeypatch.setattr(m, "WORKSPACE_ROOT", ws_root)
    monkeypatch.setattr(m, "_PROJECT_ROOT", tmp_path)

    src = tmp_path / "src.parquet"
    _make_src_parquet(src, cols=["AlphaOne", "AlphaTwo"])

    reg_dir = tmp_path / "registry" / "data"
    parquet_dir = tmp_path / "registry" / "parquet"
    now = datetime(2026, 4, 19, 13, 0, tzinfo=timezone.utc)

    results = m.migrate(
        source_parquet=src,
        profile_path=_PROFILE,
        parquet_version=1,
        registry_data_dir=reg_dir,
        parquet_store_dir=parquet_dir,
        universe="csi300",
        now=now,
    )

    # 两列都 PASS + 注册成功
    assert len(results) == 2
    for r in results:
        assert r["decision"] == "PASS"
        assert r["registered"] is True
    # parquet 存在
    store = ParquetStore(parquet_dir)
    assert store.exists(1)
    got = store.read_version(1)
    assert list(got.columns) == ["AlphaOne", "AlphaTwo"]
    # registry 两条 active 记录
    registry = FactorRegistry(reg_dir)
    actives = registry.list_active()
    assert {r.name for r in actives} == {"AlphaOne", "AlphaTwo"}
    for rec in actives:
        assert rec.status == ProductionStatus.ACTIVE
        assert rec.parquet_version == 1
        assert rec.parquet_column == rec.name
        assert "legacy" in rec.tags and "grandfathered" in rec.tags
        # 证书副本写在了 registry 目录
        assert (reg_dir / rec.certificate_path).is_file()
    # workspace 里每个因子有单列 parquet + factor.py
    for rec in actives:
        ws = ws_root / rec.factor_id
        assert (ws / "values.parquet").is_file()
        assert (ws / "factor.py").is_file()


def test_migrate_end_to_end_readable_by_loader(tmp_path: Path, monkeypatch) -> None:
    """迁移完成后，ProductionFactorLoader 能读回完全相等的列子集。"""
    from feature.production_factor_loader import ProductionFactorLoader

    monkeypatch.setattr(m, "WORKSPACE_ROOT", tmp_path / "ws")
    monkeypatch.setattr(m, "_PROJECT_ROOT", tmp_path)

    src = tmp_path / "src.parquet"
    _make_src_parquet(src, cols=["AlphaOne", "AlphaTwo", "AlphaThree"])
    orig = pd.read_parquet(src).astype("float64")

    reg_dir = tmp_path / "registry" / "data"
    parquet_dir = tmp_path / "registry" / "parquet"
    m.migrate(
        source_parquet=src,
        profile_path=_PROFILE,
        parquet_version=1,
        registry_data_dir=reg_dir,
        parquet_store_dir=parquet_dir,
        now=datetime(2026, 4, 19, tzinfo=timezone.utc),
    )

    loader = ProductionFactorLoader(
        registry=FactorRegistry(reg_dir),
        store=ParquetStore(parquet_dir),
        parquet_version=1,
    )
    loaded = loader.load_active()
    assert set(loaded.columns) == set(orig.columns)
    # 值完全一致（< 1e-12）
    pd.testing.assert_frame_equal(
        loaded.sort_index()[sorted(orig.columns)],
        orig.sort_index()[sorted(orig.columns)],
        check_exact=True,
    )


# ---------------------------------------------------------- idempotency


def test_migrate_twice_requires_overwrite(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(m, "WORKSPACE_ROOT", tmp_path / "ws")
    monkeypatch.setattr(m, "_PROJECT_ROOT", tmp_path)

    src = tmp_path / "src.parquet"
    _make_src_parquet(src, cols=["AlphaOne"])
    reg_dir = tmp_path / "registry" / "data"
    parquet_dir = tmp_path / "registry" / "parquet"

    m.migrate(
        source_parquet=src,
        profile_path=_PROFILE,
        parquet_version=1,
        registry_data_dir=reg_dir,
        parquet_store_dir=parquet_dir,
    )

    # 第二次——默认 parquet 冲突直接报错
    with pytest.raises(Exception):
        m.migrate(
            source_parquet=src,
            profile_path=_PROFILE,
            parquet_version=1,
            registry_data_dir=reg_dir,
            parquet_store_dir=parquet_dir,
        )

    # 加上 overwrite 组合参数后成功
    results = m.migrate(
        source_parquet=src,
        profile_path=_PROFILE,
        parquet_version=1,
        registry_data_dir=reg_dir,
        parquet_store_dir=parquet_dir,
        overwrite_parquet=True,
        overwrite_registry=True,
    )
    assert results[0]["registered"] is True


# ---------------------------------------------------------- failure modes


def test_migrate_rejects_bad_index(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(m, "WORKSPACE_ROOT", tmp_path / "ws")
    monkeypatch.setattr(m, "_PROJECT_ROOT", tmp_path)

    src = tmp_path / "src.parquet"
    df = pd.DataFrame({"AlphaOne": [1.0, 2.0]}, dtype="float64")
    df.to_parquet(src)
    with pytest.raises(ValueError, match="MultiIndex"):
        m.migrate(
            source_parquet=src,
            profile_path=_PROFILE,
            parquet_version=1,
            registry_data_dir=tmp_path / "r",
            parquet_store_dir=tmp_path / "p",
        )


def test_migrate_empty_source(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(m, "WORKSPACE_ROOT", tmp_path / "ws")
    monkeypatch.setattr(m, "_PROJECT_ROOT", tmp_path)

    src = tmp_path / "src.parquet"
    idx = pd.MultiIndex.from_tuples([], names=["datetime", "instrument"])
    pd.DataFrame({"AlphaOne": pd.Series([], dtype="float64")}, index=idx).to_parquet(src)
    with pytest.raises(ValueError, match="为空"):
        m.migrate(
            source_parquet=src,
            profile_path=_PROFILE,
            parquet_version=1,
            registry_data_dir=tmp_path / "r",
            parquet_store_dir=tmp_path / "p",
        )


def test_migrate_low_coverage_fails_validation(tmp_path: Path, monkeypatch) -> None:
    """把 nan_ratio 拉到 0.5 → coverage 不通过 → decision=FAIL → 不注册。"""
    monkeypatch.setattr(m, "WORKSPACE_ROOT", tmp_path / "ws")
    monkeypatch.setattr(m, "_PROJECT_ROOT", tmp_path)

    src = tmp_path / "src.parquet"
    _make_src_parquet(src, cols=["LowCov"], nan_ratio=0.5)

    reg_dir = tmp_path / "registry" / "data"
    parquet_dir = tmp_path / "registry" / "parquet"
    results = m.migrate(
        source_parquet=src,
        profile_path=_PROFILE,
        parquet_version=1,
        registry_data_dir=reg_dir,
        parquet_store_dir=parquet_dir,
    )
    assert results[0]["decision"] == "FAIL"
    assert results[0]["registered"] is False
    # 即使 validation 失败，parquet 仍然写入了（store 是无条件的物理搬运）
    assert (parquet_dir / "factors_v1.parquet").is_file()
    # registry 没有记录
    registry = FactorRegistry(reg_dir)
    assert len(registry) == 0
