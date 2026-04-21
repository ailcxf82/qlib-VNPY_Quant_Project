"""集成测试：``QlibFeaturePipeline._maybe_merge_rdagent_parquet`` 切换到 L3 loader
后对**旧行为的等价性**与**新路径的正确性**。

技巧：
* 绕过 ``QlibFeaturePipeline.__init__``（它会调用 ``qlib.init``），用
  ``object.__new__`` 手工塞 attributes。
* monkeypatch ``_project_root`` 让 ``config/factor_lab.yaml`` 指向 tmp_path。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from factor_registry.registry import FactorRegistry
from factor_registry.store import ParquetStore
from factor_validation.schema import (
    CertifiedFactorRecord,
    CheckResult,
    Decision,
)
from feature import qlib_feature_pipeline as qfp_mod
from feature.qlib_feature_pipeline import QlibFeaturePipeline


# ---------------------------------------------------------------- fixtures


def _make_pipeline(data_cfg: dict) -> QlibFeaturePipeline:
    """绕过 __init__ 造一个可调用 _maybe_merge_rdagent_parquet 的实例。"""
    pipe = object.__new__(QlibFeaturePipeline)
    pipe.config = {"data": data_cfg}
    pipe.feature_cfg = data_cfg
    pipe.rdagent_factor_columns = []
    pipe.features_df = None
    pipe.label_series = None
    pipe._feature_mean = None
    pipe._feature_std = None
    pipe._label_is_rank = False
    return pipe


def _make_feature_panel() -> pd.DataFrame:
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2025-01-02"), "SH600000"),
            (pd.Timestamp("2025-01-02"), "SZ000001"),
            (pd.Timestamp("2025-01-03"), "SH600000"),
            (pd.Timestamp("2025-01-03"), "SZ000001"),
        ],
        names=["datetime", "instrument"],
    )
    return pd.DataFrame(
        {"lgb_price": [10.0, 20.0, 11.0, 21.0], "lgb_vol": [1.0, 2.0, 1.1, 2.1]},
        index=idx,
        dtype="float64",
    )


def _setup_factor_lab(
    project_root: Path,
    parquet_factors: pd.DataFrame,
    *,
    flags_enabled: bool = True,
    min_candidate_fixture=None,
) -> None:
    """在 ``project_root`` 下建立最小 factor_lab + registry + parquet 目录。"""
    (project_root / "config").mkdir(parents=True, exist_ok=True)
    (project_root / "factor_registry" / "data").mkdir(parents=True, exist_ok=True)
    (project_root / "factor_registry" / "parquet").mkdir(parents=True, exist_ok=True)

    cfg = {
        "schema_version": 1,
        "registry": {
            "data_dir": "factor_registry/data",
            "parquet_dir": "factor_registry/parquet",
            "current_parquet_version": 1,
        },
        "flags": {"enabled": flags_enabled},
    }
    (project_root / "config" / "factor_lab.yaml").write_text(
        yaml.safe_dump(cfg), encoding="utf-8"
    )

    store = ParquetStore(project_root / "factor_registry" / "parquet")
    store.write_version(parquet_factors, 1)


def _register_all_columns(
    project_root: Path, parquet_factors: pd.DataFrame, minimal_candidate
) -> None:
    """把 parquet_factors 的每一列都以 active 状态写进 registry。"""
    registry = FactorRegistry(project_root / "factor_registry" / "data")
    for i, col in enumerate(parquet_factors.columns):
        cand = minimal_candidate.model_copy(
            update={
                "factor_id": f"manual_{col}_{'0' * 7}{i}",
                "name": col,
                "source": "manual",
                "parent_loop": None,
            }
        )
        cert = CertifiedFactorRecord(
            factor_id=cand.factor_id,
            candidate=cand,
            profile_name="manual_grandfathered",
            profile_hash="a" * 64,
            decision=Decision.PASS,
            overall_score=0.99,
            check_results=[
                CheckResult(
                    name="coverage",
                    passed=True,
                    score=0.99,
                    threshold=0.9,
                    detail={},
                    elapsed_ms=1,
                )
            ],
            validated_at=datetime(2026, 4, 19, tzinfo=timezone.utc),
            validator_version="0.1.0",
        )
        registry.register(
            cert,
            parquet_version=1,
            tags=["legacy"],
            now=datetime(2026, 4, 19, 0, i, tzinfo=timezone.utc),
        )


# ---------------------------------------------------------------- no active_feature_sets


def test_no_rdagent_exported_returns_panel_unchanged(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(qfp_mod, "_project_root", tmp_path)
    pipe = _make_pipeline({"active_feature_sets": ["lgb_short_cycle"]})
    panel = _make_feature_panel()
    out = pipe._maybe_merge_rdagent_parquet(panel)
    # 未启用 → 直接返回原 panel，且 rdagent_factor_columns = []
    assert out is panel
    assert pipe.rdagent_factor_columns == []


# ---------------------------------------------------------------- L3 path


def test_l3_loader_preferred(
    tmp_path: Path, monkeypatch, minimal_candidate
) -> None:
    monkeypatch.setattr(qfp_mod, "_project_root", tmp_path)

    # L3 parquet 含 AlphaOne / AlphaTwo
    l3_df = pd.DataFrame(
        {
            "AlphaOne": [0.1, 0.2, 0.3, 0.4],
            "AlphaTwo": [0.5, 0.6, 0.7, 0.8],
        },
        index=_make_feature_panel().index,
        dtype="float64",
    )
    _setup_factor_lab(tmp_path, l3_df, flags_enabled=True)
    _register_all_columns(tmp_path, l3_df, minimal_candidate)

    # 旧路径 parquet 故意不写 → 只可能走 L3
    pipe = _make_pipeline(
        {
            "active_feature_sets": ["rdagent_exported"],
            "feature_sets": {"rdagent_exported": []},
        }
    )
    out = pipe._maybe_merge_rdagent_parquet(_make_feature_panel())
    # L3 两列接入
    assert set(pipe.rdagent_factor_columns) == {"AlphaOne", "AlphaTwo"}
    assert "AlphaOne" in out.columns and "AlphaTwo" in out.columns
    # 值与 L3 parquet 一致
    pd.testing.assert_frame_equal(
        out[["AlphaOne", "AlphaTwo"]], l3_df[["AlphaOne", "AlphaTwo"]]
    )


def test_l3_equivalent_to_legacy_parquet(
    tmp_path: Path, monkeypatch, minimal_candidate
) -> None:
    """**核心等价性测试**：L3 与旧路径从相同 parquet 取数，结果按值完全一致。"""
    monkeypatch.setattr(qfp_mod, "_project_root", tmp_path)

    panel = _make_feature_panel()
    factors = pd.DataFrame(
        {
            "AlphaOne": [0.11, 0.22, 0.33, 0.44],
            "AlphaTwo": [0.55, 0.66, 0.77, 0.88],
        },
        index=panel.index,
        dtype="float64",
    )
    _setup_factor_lab(tmp_path, factors, flags_enabled=True)
    _register_all_columns(tmp_path, factors, minimal_candidate)

    # 同一份数据也放到旧路径位置
    legacy_parquet = tmp_path / "git_ignore_folder" / "combined_factors_df.parquet"
    legacy_parquet.parent.mkdir(parents=True, exist_ok=True)
    factors.to_parquet(legacy_parquet)

    # ---- L3 跑一次（flags.enabled=True）
    pipe_l3 = _make_pipeline(
        {
            "active_feature_sets": ["rdagent_exported"],
            "feature_sets": {"rdagent_exported": []},
            "rdagent_parquet": {"path": "git_ignore_folder/combined_factors_df.parquet"},
        }
    )
    out_l3 = pipe_l3._maybe_merge_rdagent_parquet(panel.copy())

    # ---- 关闭 flag 后强制走旧路径
    cfg_path = tmp_path / "config" / "factor_lab.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["flags"]["enabled"] = False
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    pipe_legacy = _make_pipeline(
        {
            "active_feature_sets": ["rdagent_exported"],
            "feature_sets": {"rdagent_exported": []},
            "rdagent_parquet": {"path": "git_ignore_folder/combined_factors_df.parquet"},
        }
    )
    out_legacy = pipe_legacy._maybe_merge_rdagent_parquet(panel.copy())

    # 两条路径产出的 RD-Agent 列集合相同、值相等
    assert set(pipe_l3.rdagent_factor_columns) == set(
        pipe_legacy.rdagent_factor_columns
    )
    cols = sorted(pipe_l3.rdagent_factor_columns)
    pd.testing.assert_frame_equal(
        out_l3[cols].sort_index(), out_legacy[cols].sort_index(), check_exact=True
    )


def test_l3_disabled_flag_falls_back_to_legacy(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(qfp_mod, "_project_root", tmp_path)

    panel = _make_feature_panel()
    legacy_parquet = tmp_path / "git_ignore_folder" / "combined_factors_df.parquet"
    legacy_parquet.parent.mkdir(parents=True, exist_ok=True)
    legacy_df = pd.DataFrame(
        {"LegacyA": [0.01, 0.02, 0.03, 0.04]},
        index=panel.index,
        dtype="float64",
    )
    legacy_df.to_parquet(legacy_parquet)

    # factor_lab.yaml 存在但 flag=false
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "factor_lab.yaml").write_text(
        yaml.safe_dump({"flags": {"enabled": False}}),
        encoding="utf-8",
    )

    pipe = _make_pipeline(
        {
            "active_feature_sets": ["rdagent_exported"],
            "feature_sets": {"rdagent_exported": []},
        }
    )
    out = pipe._maybe_merge_rdagent_parquet(panel)
    assert pipe.rdagent_factor_columns == ["LegacyA"]
    assert "LegacyA" in out.columns


def test_l3_empty_registry_falls_back_to_legacy(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(qfp_mod, "_project_root", tmp_path)
    panel = _make_feature_panel()
    legacy_parquet = tmp_path / "git_ignore_folder" / "combined_factors_df.parquet"
    legacy_parquet.parent.mkdir(parents=True, exist_ok=True)
    legacy_df = pd.DataFrame(
        {"LegacyA": [0.01, 0.02, 0.03, 0.04]},
        index=panel.index,
        dtype="float64",
    )
    legacy_df.to_parquet(legacy_parquet)

    # factor_lab.yaml 启用 + registry / parquet 目录已建，但 manifest 为空
    _setup_factor_lab(
        tmp_path,
        pd.DataFrame(
            {"Placeholder": [0.0, 0.0, 0.0, 0.0]},
            index=panel.index,
            dtype="float64",
        ),
        flags_enabled=True,
    )

    pipe = _make_pipeline(
        {
            "active_feature_sets": ["rdagent_exported"],
            "feature_sets": {"rdagent_exported": []},
        }
    )
    out = pipe._maybe_merge_rdagent_parquet(panel)
    # 因为 registry 里没注册 active → 回退旧路径 → 得到 LegacyA
    assert pipe.rdagent_factor_columns == ["LegacyA"]
    assert "LegacyA" in out.columns


def test_l3_loader_raises_falls_back_silently(
    tmp_path: Path, monkeypatch
) -> None:
    """L3 loader 抛任何错都应 fallback，而不是让整个训练流崩溃。"""
    monkeypatch.setattr(qfp_mod, "_project_root", tmp_path)

    panel = _make_feature_panel()
    legacy_parquet = tmp_path / "git_ignore_folder" / "combined_factors_df.parquet"
    legacy_parquet.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"LegacyA": [0.01, 0.02, 0.03, 0.04]},
        index=panel.index,
        dtype="float64",
    ).to_parquet(legacy_parquet)

    # 故意把 factor_lab.yaml 写坏
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "factor_lab.yaml").write_text(
        "this is: not: valid yaml: [unbalanced",
        encoding="utf-8",
    )

    pipe = _make_pipeline(
        {
            "active_feature_sets": ["rdagent_exported"],
            "feature_sets": {"rdagent_exported": []},
        }
    )
    out = pipe._maybe_merge_rdagent_parquet(panel)
    assert "LegacyA" in out.columns  # fallback 生效


def test_l3_column_name_overlap_with_qlib_features(
    tmp_path: Path, monkeypatch, minimal_candidate
) -> None:
    """L3 列与 qlib 特征列名重名 → 丢弃 L3 侧冲突列，保留 qlib 的。"""
    monkeypatch.setattr(qfp_mod, "_project_root", tmp_path)

    panel = _make_feature_panel()  # 列 ['lgb_price', 'lgb_vol']
    l3_df = pd.DataFrame(
        {
            "lgb_price": [99.0, 99.0, 99.0, 99.0],   # 与 panel 冲突
            "UniqueAlpha": [0.1, 0.2, 0.3, 0.4],     # 独有
        },
        index=panel.index,
        dtype="float64",
    )
    _setup_factor_lab(tmp_path, l3_df, flags_enabled=True)
    _register_all_columns(tmp_path, l3_df, minimal_candidate)

    pipe = _make_pipeline(
        {
            "active_feature_sets": ["rdagent_exported"],
            "feature_sets": {"rdagent_exported": []},
        }
    )
    out = pipe._maybe_merge_rdagent_parquet(panel)
    # lgb_price 保留 panel 原值 (10/20/11/21)
    assert list(out["lgb_price"]) == [10.0, 20.0, 11.0, 21.0]
    # UniqueAlpha 成功接入
    assert "UniqueAlpha" in out.columns
    assert pipe.rdagent_factor_columns == ["UniqueAlpha"]
