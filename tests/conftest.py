"""顶层测试夹具：跨 ``tests/factor_lab/``、``tests/factor_validation/`` 等子目录共享。

放在顶层 ``tests/conftest.py`` 而不是子目录的 conftest，是为了让 pytest 的 plugin
注册机制（按文件路径）不会与子模块 ``tests.factor_lab.conftest`` 命名冲突。
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from factor_lab import CandidateFactorPackage
from factor_validation.schema import (
    CertifiedFactorRecord,
    CheckResult,
    Decision,
)


# 阶段 F.4：注册 e2e_rdagent marker 并默认 skip；只有显式传 -m e2e_rdagent 时才跑。
_E2E_MARKER = "e2e_rdagent"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        f"{_E2E_MARKER}: 阶段 F.4 端到端冒烟测试（需要 RD-Agent + LLM，默认 skip；"
        "用 `pytest -m {_E2E_MARKER}` 显式触发）",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """默认跳过带 e2e_rdagent 标记的用例；命令行 `-m e2e_rdagent` 时放行。"""
    expr = config.getoption("-m", default="") or ""
    if _E2E_MARKER in expr:
        return
    skip_marker = pytest.mark.skip(
        reason=(
            f"默认跳过 {_E2E_MARKER} 端到端测试；用 `pytest -m {_E2E_MARKER}` 显式触发。"
        )
    )
    for item in items:
        if _E2E_MARKER in item.keywords:
            item.add_marker(skip_marker)


def _write_min_parquet(path: Path, name: str, start: date, end: date) -> None:
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp(start), "SH600000"),
            (pd.Timestamp(end), "SH600000"),
        ],
        names=["datetime", "instrument"],
    )
    df = pd.DataFrame({name: [0.1, 0.2]}, index=idx, dtype="float64")
    df.to_parquet(path)


def _write_factor_py(path: Path) -> None:
    path.write_text(
        '"""Minimal factor module for tests."""\n\n'
        "def calculate(df):\n    return df\n",
        encoding="utf-8",
    )


@pytest.fixture()
def minimal_candidate(tmp_path) -> CandidateFactorPackage:
    """构造一个能通过所有校验（含物料校验）的最小候选包。"""
    name = "QualPersist_60D"
    factor_id = f"rdagent_{name}_a1b2c3d4"
    start, end = date(2025, 1, 2), date(2025, 6, 30)

    code_path = tmp_path / "factor.py"
    values_path = tmp_path / "values.parquet"
    _write_factor_py(code_path)
    _write_min_parquet(values_path, name, start, end)

    return CandidateFactorPackage(
        factor_id=factor_id,
        name=name,
        source="rdagent",
        hypothesis="Slow-moving quality factor based on 60D rolling ROE.",
        formulation="rank(rolling_mean($roe, 60))",
        code_path=code_path,
        values_path=values_path,
        universe="csi300",
        date_range=(start, end),
        lab_metrics={"lab_rank_ic": 0.018, "lab_ic_ir": 0.42},
        parent_loop=0,
        created_at=datetime(2026, 4, 19, 12, 34, 56, tzinfo=timezone.utc),
        lab_run_id="lab-2026-04-19-deadbeef",
    )


@pytest.fixture()
def minimal_certified(minimal_candidate) -> CertifiedFactorRecord:
    """基于 ``minimal_candidate`` 构造一个 PASS 的最小认证记录。"""
    check = CheckResult(
        name="coverage",
        passed=True,
        score=0.95,
        threshold=0.9,
        detail={"min_coverage": 0.95},
        elapsed_ms=12,
    )
    return CertifiedFactorRecord(
        factor_id=minimal_candidate.factor_id,
        candidate=minimal_candidate,
        profile_name="manual_grandfathered",
        profile_hash="a" * 64,
        decision=Decision.PASS,
        overall_score=0.95,
        check_results=[check],
        backtest_metrics={},
        validated_at=datetime(2026, 4, 19, 13, 0, 0, tzinfo=timezone.utc),
        validator_version="0.1.0",
        notes="grandfathered",
    )
