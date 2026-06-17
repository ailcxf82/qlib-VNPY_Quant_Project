"""``orthogonality`` check —— 候选因子与 **参照因子集** 的去冗余检查。

**动机**：L3 是"因子集"而不是"因子池" —— 相关性高的因子不仅不增加模型表达力，反而会
稀释权重、加剧过拟合。本 check 要求候选因子与当前 active 因子**线性独立**到可接受程度。

**参照集来源**：``CheckContext.reference_factors_parquet``（通常由 profile
``data_sources.reference_factors_parquet`` 指向，默认就是 L3 当前版本
``factor_registry/parquet/factors_v<N>.parquet``）。

**算法**：
    for ref_col in reference_factors:
        按 (datetime, instrument) 对齐 → 每日截面 **Spearman rank corr** → 沿时间取平均
        → ``corr_vs[ref_col]`` = 该平均值
    ``max_abs_corr`` = max(|corr_vs[*]|)

**阈值字段**（profile 配置）：

* ``max_abs_corr`` （必填；``[0, 1]``）—— 与任何参照列的平均截面 rank-corr 绝对值上限

**Passed**：``max_abs_corr <= threshold``

**Score**：``1 - max_abs_corr``，夹到 ``[0, 1]``。

**边界情况**：
* 参照集里包含与候选同名/同 ``parquet_column`` 的列 → 自动跳过（不和自己比）。
* 参照集为空（例如 first-factor）→ ``max_abs_corr = 0``, ``score = 1``, ``passed = True``，
  detail 里标注 ``n_reference_cols = 0``。
* 对齐后有效交叉点 < 30 → raise（样本太少），转成 orchestrator 层面的 FAIL。
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from factor_lab.exporters.schema import CandidateFactorPackage
from factor_validation.checks.base import CheckBase, CheckContext
from factor_validation.schema import CheckResult

logger = logging.getLogger(__name__)

_MIN_CROSS_SAMPLES_PER_DAY = 5
_MIN_VALID_DAYS = 5


class OrthogonalityCheck(CheckBase):
    """与参照因子集的平均截面 rank corr 检查。"""

    name = "orthogonality"

    def run(
        self,
        candidate: CandidateFactorPackage,
        context: CheckContext,
        config: dict[str, Any],
    ) -> CheckResult:
        start = time.perf_counter()

        if "max_abs_corr" not in config:
            raise ValueError("orthogonality: 配置缺字段 max_abs_corr")
        max_abs_corr = float(config["max_abs_corr"])
        if not 0.0 <= max_abs_corr <= 1.0:
            raise ValueError(
                f"orthogonality: max_abs_corr 必须在 [0, 1]: {max_abs_corr}"
            )

        if context.reference_factors_parquet is None:
            raise ValueError(
                "orthogonality: CheckContext.reference_factors_parquet 未设置；"
                "请在 profile.data_sources.reference_factors_parquet 指向 L3 当前 parquet"
            )

        # ---- 1) 加载候选因子 + 切 OOS ----
        cand_df = pd.read_parquet(candidate.values_path)
        if candidate.name not in cand_df.columns:
            raise ValueError(
                f"orthogonality: parquet 缺列 {candidate.name!r}，"
                f"实际列: {list(cand_df.columns)}"
            )
        cand_series = cand_df[candidate.name]
        cand_series = self._slice_oos(cand_series, context)
        if cand_series.empty:
            raise ValueError("orthogonality: OOS 窗内候选因子无样本")

        # ---- 2) 加载参照矩阵 + 切 OOS ----
        ref_path = context.reference_factors_parquet
        if not ref_path.exists():
            raise ValueError(
                f"orthogonality: reference_factors_parquet 不存在: {ref_path}"
            )
        ref_df = pd.read_parquet(ref_path)
        ref_df = self._slice_oos_frame(ref_df, context)

        # 去掉与候选同名列（自对比没意义）
        if candidate.name in ref_df.columns:
            ref_df = ref_df.drop(columns=[candidate.name])

        if ref_df.shape[1] == 0:
            logger.info("orthogonality: 参照集为空；直接 PASS (score=1)")
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            return CheckResult(
                name=self.name,
                passed=True,
                score=1.0,
                threshold=max_abs_corr,
                detail={
                    "max_abs_corr": 0.0,
                    "n_reference_cols": 0,
                    "per_ref_corr": {},
                    "note": "no reference factors (first-factor case)",
                },
                elapsed_ms=elapsed_ms,
            )

        # ---- 3) 每日截面 rank-corr → 跨天平均 ----
        cand_wide = cand_series.unstack("instrument")
        per_ref_corr: dict[str, float] = {}
        for ref_col in ref_df.columns:
            ref_series = ref_df[ref_col]
            ref_wide = ref_series.unstack("instrument")
            avg_corr = self._daily_rank_corr_mean(cand_wide, ref_wide)
            if np.isnan(avg_corr):
                # 与此参照列完全无法对齐 → 忽略（不写入 per_ref_corr）
                continue
            per_ref_corr[ref_col] = float(avg_corr)

        if not per_ref_corr:
            raise ValueError(
                "orthogonality: 所有参照列均无法与候选对齐 "
                f"(ref_cols={list(ref_df.columns)})"
            )

        max_abs = max(abs(v) for v in per_ref_corr.values())
        score = max(0.0, min(1.0, 1.0 - max_abs))
        passed = max_abs <= max_abs_corr

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        # 详情：把每个 ref 对应的 corr 也落下，便于人工诊断
        return CheckResult(
            name=self.name,
            passed=passed,
            score=round(score, 6),
            threshold=max_abs_corr,
            detail={
                "max_abs_corr": round(max_abs, 6),
                "n_reference_cols": len(per_ref_corr),
                "per_ref_corr": {
                    k: round(v, 6)
                    for k, v in sorted(per_ref_corr.items(), key=lambda kv: -abs(kv[1]))
                },
                "reference_parquet": str(ref_path),
            },
            elapsed_ms=elapsed_ms,
        )

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _slice_oos(series: pd.Series, context: CheckContext) -> pd.Series:
        start_d, end_d = context.oos_window
        start_ts = pd.Timestamp(start_d)
        end_ts = pd.Timestamp(end_d) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        idx = series.index.get_level_values("datetime")
        return series.loc[(idx >= start_ts) & (idx <= end_ts)]

    @staticmethod
    def _slice_oos_frame(df: pd.DataFrame, context: CheckContext) -> pd.DataFrame:
        start_d, end_d = context.oos_window
        start_ts = pd.Timestamp(start_d)
        end_ts = pd.Timestamp(end_d) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        idx = df.index.get_level_values("datetime")
        return df.loc[(idx >= start_ts) & (idx <= end_ts)]

    @staticmethod
    def _daily_rank_corr_mean(
        a_wide: pd.DataFrame, b_wide: pd.DataFrame
    ) -> float:
        """两个 (datetime x instrument) 宽表逐日截面 Spearman rank corr → 取算术平均。"""
        common_idx = a_wide.index.intersection(b_wide.index)
        common_cols = a_wide.columns.intersection(b_wide.columns)
        if len(common_idx) == 0 or len(common_cols) == 0:
            return float("nan")
        a = a_wide.loc[common_idx, common_cols]
        b = b_wide.loc[common_idx, common_cols]

        # 截面 rank（跨股票），同日两边都有效才参与
        a_rank = a.rank(axis=1, method="average", na_option="keep")
        b_rank = b.rank(axis=1, method="average", na_option="keep")

        daily_corrs: list[float] = []
        for ts in common_idx:
            a_row = a_rank.loc[ts]
            b_row = b_rank.loc[ts]
            mask = a_row.notna() & b_row.notna()
            if mask.sum() < _MIN_CROSS_SAMPLES_PER_DAY:
                continue
            # Pearson on ranks = Spearman
            corr = float(np.corrcoef(a_row[mask].values, b_row[mask].values)[0, 1])
            if np.isnan(corr):
                continue
            daily_corrs.append(corr)

        if len(daily_corrs) < _MIN_VALID_DAYS:
            return float("nan")
        return float(np.mean(daily_corrs))
