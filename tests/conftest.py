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
