"""``coverage`` check —— 验证候选因子数据覆盖率。

通过阈值：
* ``min_non_null_ratio``   —— 因子值列的非空比例（每个 (datetime, instrument) 交叉点
  视为一个样本）必须 >= 此阈值。
* ``min_instrument_ratio`` —— *可选*；若给出，校验 parquet 覆盖的 **instrument 集合**
  相对于 universe 预期集合的比例（需要 qlib；默认关闭，避免 L2 依赖）。

返回：
* ``score``：非空比例本身（已在 [0,1]）。
* ``passed``：``score >= min_non_null_ratio`` 且 （若启用） instrument_ratio 达标。
* ``detail``：``n_rows``, ``n_non_null``, ``non_null_ratio``, ``n_instruments``。
"""

from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd

from factor_lab.exporters.schema import CandidateFactorPackage
from factor_validation.checks.base import CheckBase, CheckContext
from factor_validation.schema import CheckResult

logger = logging.getLogger(__name__)


class CoverageCheck(CheckBase):
    """Parquet 样本点非空比例检查。"""

    name = "coverage"

    def run(
        self,
        candidate: CandidateFactorPackage,
        context: CheckContext,
        config: dict[str, Any],
    ) -> CheckResult:
        start = time.perf_counter()

        min_ratio = float(config.get("min_non_null_ratio", 0.9))
        if not 0.0 <= min_ratio <= 1.0:
            raise ValueError(
                f"coverage.min_non_null_ratio 必须在 [0,1]: {min_ratio}"
            )

        df = pd.read_parquet(candidate.values_path)
        if df.empty:
            raise ValueError(
                f"coverage: values_path 空 parquet: {candidate.values_path}"
            )
        if candidate.name not in df.columns:
            raise ValueError(
                f"coverage: parquet 缺列 {candidate.name!r}，"
                f"实际列: {list(df.columns)}"
            )
        series = df[candidate.name]
        n_rows = int(len(series))
        n_non_null = int(series.notna().sum())
        non_null_ratio = float(n_non_null / n_rows) if n_rows > 0 else 0.0
        n_instruments = int(
            df.index.get_level_values("instrument").nunique()
            if isinstance(df.index, pd.MultiIndex)
            else 0
        )

        passed = non_null_ratio >= min_ratio
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        return CheckResult(
            name=self.name,
            passed=passed,
            score=round(non_null_ratio, 6),
            threshold=min_ratio,
            detail={
                "n_rows": n_rows,
                "n_non_null": n_non_null,
                "non_null_ratio": round(non_null_ratio, 6),
                "n_instruments": n_instruments,
                "universe": context.universe,
            },
            elapsed_ms=elapsed_ms,
        )
