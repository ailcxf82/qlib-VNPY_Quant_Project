"""``ic`` check —— 候选因子在 OOS 窗内的 Rank IC + IC_IR。

**动机**：因子价值的第一性指标。没有 OOS IC 的因子一律不能进 L3。

**数据源**：
* 候选因子：``candidate.values_path`` 的单列 parquet
* Label：``CheckContext.label_parquet``，MultiIndex ``(datetime, instrument)``，
  单列 ``label: float64``；由 ``scripts/validate/prepare_oos_labels.py`` 离线生成。

**算法**：
1. 在 OOS 窗内对齐候选因子与 label（内连接）
2. 按日截面 Spearman Rank IC（最少 ``min_cross_samples_per_day=5`` 只股票；
   有效日数 < ``_MIN_VALID_DAYS=20`` 直接抛错）
3. ``rank_ic = mean(daily_ic)``；``ic_ir = mean / (std + 1e-12)``

**阈值字段**（profile 配置）：

* ``min_rank_ic`` —— 候选因子平均日 Rank IC 下限（绝对值比较前先判方向，允许
  负向因子：若配置 ``allow_negative: true`` 则按 ``abs(rank_ic)`` 比较）
* ``min_ic_ir``   —— 候选因子 IC_IR 下限（同上，``allow_negative`` 时按绝对值）
* ``allow_negative``（可选，默认 False）—— 若 True，允许负向因子通过

**Passed**：同时满足 min_rank_ic + min_ic_ir。

**Score**：``min(rank_ic_ratio, 1) * 0.5 + min(ic_ir_ratio, 1) * 0.5``，其中
``*_ratio = max(0, value) / max(threshold, eps)``。允许负向时 value 取绝对值。
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
_MIN_VALID_DAYS = 20
_EPS = 1e-12


class IcCheck(CheckBase):
    """OOS Rank IC + IC_IR。"""

    name = "ic"

    def run(
        self,
        candidate: CandidateFactorPackage,
        context: CheckContext,
        config: dict[str, Any],
    ) -> CheckResult:
        start = time.perf_counter()

        if "min_rank_ic" not in config:
            raise ValueError("ic: 配置缺字段 min_rank_ic")
        if "min_ic_ir" not in config:
            raise ValueError("ic: 配置缺字段 min_ic_ir")
        min_rank_ic = float(config["min_rank_ic"])
        min_ic_ir = float(config["min_ic_ir"])
        allow_negative = bool(config.get("allow_negative", False))

        if context.label_parquet is None:
            raise ValueError(
                "ic: CheckContext.label_parquet 未设置；"
                "请先跑 scripts/validate/prepare_oos_labels.py 生成标签 parquet，"
                "并在 profile.data_sources.label_parquet 指向它"
            )
        if not context.label_parquet.exists():
            raise ValueError(
                f"ic: label_parquet 不存在: {context.label_parquet}"
            )

        # ---- 加载候选因子（单列） ----
        cand_df = pd.read_parquet(candidate.values_path)
        if candidate.name not in cand_df.columns:
            raise ValueError(
                f"ic: parquet 缺列 {candidate.name!r}，实际列: {list(cand_df.columns)}"
            )
        cand_series = cand_df[candidate.name]
        cand_series = self._slice_oos(cand_series, context)
        if cand_series.empty:
            raise ValueError("ic: OOS 窗内候选因子无样本")

        # ---- 加载 label ----
        label_df = pd.read_parquet(context.label_parquet)
        if "label" not in label_df.columns:
            raise ValueError(
                f"ic: label_parquet 必须包含 'label' 列，实际: {list(label_df.columns)}"
            )
        label_series = label_df["label"]
        label_series = self._slice_oos(label_series, context)

        # ---- 对齐（内连接）----
        merged = pd.concat(
            [cand_series.rename("factor"), label_series.rename("label")],
            axis=1,
            join="inner",
        ).dropna()
        if merged.empty:
            raise ValueError(
                "ic: 候选因子与 label 对齐后无样本；"
                "请确认 OOS 窗 + universe 一致"
            )

        # ---- 逐日截面 Rank IC ----
        cand_wide = merged["factor"].unstack("instrument")
        label_wide = merged["label"].unstack("instrument")
        cand_rank = cand_wide.rank(axis=1, method="average", na_option="keep")
        label_rank = label_wide.rank(axis=1, method="average", na_option="keep")

        daily_ic: list[float] = []
        for ts in cand_rank.index:
            a = cand_rank.loc[ts]
            b = label_rank.loc[ts]
            mask = a.notna() & b.notna()
            if mask.sum() < _MIN_CROSS_SAMPLES_PER_DAY:
                continue
            c = float(np.corrcoef(a[mask].values, b[mask].values)[0, 1])
            if np.isnan(c):
                continue
            daily_ic.append(c)

        if len(daily_ic) < _MIN_VALID_DAYS:
            raise ValueError(
                f"ic: 有效日数不足 {_MIN_VALID_DAYS} (实际 {len(daily_ic)})；"
                f"样本 / OOS 窗过短"
            )

        arr = np.asarray(daily_ic, dtype=float)
        rank_ic = float(arr.mean())
        std = float(arr.std(ddof=1))
        ic_ir = rank_ic / (std + _EPS)

        # ---- 阈值比较 + 评分 ----
        cmp_ic = abs(rank_ic) if allow_negative else rank_ic
        cmp_ir = abs(ic_ir) if allow_negative else ic_ir
        passed = cmp_ic >= min_rank_ic and cmp_ir >= min_ic_ir

        def _ratio(value: float, target: float) -> float:
            return max(0.0, value) / max(target, _EPS)

        ratio_ic = min(_ratio(cmp_ic, min_rank_ic), 1.0)
        ratio_ir = min(_ratio(cmp_ir, min_ic_ir), 1.0)
        score = max(0.0, min(1.0, 0.5 * ratio_ic + 0.5 * ratio_ir))

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return CheckResult(
            name=self.name,
            passed=passed,
            score=round(score, 6),
            threshold=None,  # 双阈值，主阈值写 detail
            detail={
                "rank_ic": round(rank_ic, 6),
                "ic_ir": round(ic_ir, 6),
                "n_valid_days": len(daily_ic),
                "min_rank_ic": min_rank_ic,
                "min_ic_ir": min_ic_ir,
                "allow_negative": allow_negative,
                "label_parquet": str(context.label_parquet),
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
