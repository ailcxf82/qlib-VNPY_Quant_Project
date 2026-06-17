"""``turnover`` check —— 因子自身的换手/平滑性检查（零外部依赖）。

**动机**：阶段 P0-1 历史教训 —— 日内排名几乎每天翻滚的因子会给下游模型带来高交易成本
和虚假 IC。本 check 在**不需要回测**的前提下，用纯截面 rank + 时序 rank-autocorr
快速识别这种"陷阱因子"。

**两项指标**（都在 candidate 的 OOS 时间窗内计算）：

1. ``daily_rank_turnover`` —— 对每个交易日 d，先把当日所有股票按因子值做截面
   **rank (1..N)**；然后对当日 rank 与前一交易日 rank 的**截面 Spearman 秩相关系数**
   做 ``(1 - ρ) / 2`` 变换；对 OOS 窗内全部有效日取算术平均。

   语义对照：

   * ρ = 1  （rank 完全不变） → turnover = 0
   * ρ = 0  （完全无记忆 / 随机重排） → turnover ≈ 0.5
   * ρ = -1 （rank 完全反转）   → turnover = 1

   **为什么不用"发生变化的股票占比"**：以 ``method='average'`` 计 rank 时，任何
   一个因子值的微小变动都会在其同值区间两侧引入 ±0.5 的 rank 漂移，导致"变化占比"对几乎
   所有稠密连续因子都恒 ≈1.0，完全丧失区分度（阶段 C 回溯 oracle 实证过）。秩相关则对
   此类微扰鲁棒，只有**整体 rank 顺序**显著改变时才上升。

2. ``rank_autocorr`` —— 每只股票把自己的 daily rank 看作一个时间序列，取 ``lag=1``
   的 Pearson 自相关，再对所有股票取**中位数**（对极端值鲁棒）。值域 ``[-1, 1]``，越大越平滑。

**两指标的分工**：

* ``daily_rank_turnover``  = **截面方向**的 day-to-day rank 稳定度（逐日聚合）
* ``rank_autocorr``        = **时序方向**的 per-stock rank 自相关（逐股票聚合再取中位数）

二者数学上相关但不等价，放在一起能交叉校验。

**阈值字段**（profile 配置）：

* ``max_daily_rank_turnover`` （必填；``[0, 1]``）—— ``daily_rank_turnover`` 上限
* ``min_rank_autocorr``       （必填；``[-1, 1]``）—— ``rank_autocorr`` 下限

**passed**：两个指标都达标。

**score**：
    ``0.5 * (1 - daily_rank_turnover) + 0.5 * (rank_autocorr + 1) / 2``
夹到 ``[0, 1]``。慢因子 / 快因子的评分区分度较好。
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


class TurnoverCheck(CheckBase):
    """因子排名换手 + 时序自相关。"""

    name = "turnover"

    def run(
        self,
        candidate: CandidateFactorPackage,
        context: CheckContext,
        config: dict[str, Any],
    ) -> CheckResult:
        start = time.perf_counter()

        max_turnover = self._float_in(config, "max_daily_rank_turnover", lo=0.0, hi=1.0)
        min_autocorr = self._float_in(config, "min_rank_autocorr", lo=-1.0, hi=1.0)

        df = pd.read_parquet(candidate.values_path)
        if candidate.name not in df.columns:
            raise ValueError(
                f"turnover: parquet 缺列 {candidate.name!r}，"
                f"实际列: {list(df.columns)}"
            )
        series = df[candidate.name]

        # 按 OOS 窗切片（闭区间）
        series = self._slice_oos(series, context)

        if series.empty:
            raise ValueError("turnover: OOS 窗内无样本")

        # unstack 成 (datetime x instrument) 宽表，便于做截面 rank + 时序 autocorr
        wide = series.unstack("instrument")
        # 排除全空行/列，避免 rank 报警
        wide = wide.dropna(axis=0, how="all").dropna(axis=1, how="all")
        if wide.shape[0] < 2 or wide.shape[1] < 2:
            raise ValueError(
                f"turnover: OOS 窗内有效 shape 不足 (n_days={wide.shape[0]}, "
                f"n_instruments={wide.shape[1]}); 无法计算排名变化"
            )

        # ---- 1) daily_rank_turnover （基于截面 Spearman 的 (1-ρ)/2） ----
        # 截面 rank（同日内按因子值升序，NaN 保留 NaN）
        ranks = wide.rank(axis=1, method="average", na_option="keep")
        # 前一交易日 rank
        prev = ranks.shift(1)
        # 对每一对 (today, prev_day) 同时非空的股票子集，计算 Pearson(ranks) == Spearman(raw)
        #   * 只需要两天都有 rank 的股票（inner join 概念）
        #   * 至少 3 个共同股票才能稳定地算 Spearman
        #   * 若共同集上 rank 在某一天是常数（方差 0），Pearson 没有定义 → 记 NaN 跳过
        both_valid = ranks.notna() & prev.notna()
        valid_days_mask = both_valid.sum(axis=1) >= 3
        if not valid_days_mask.any():
            raise ValueError(
                "turnover: 无任何可对比日（全部样本首日或共同股票 < 3）"
            )

        daily_turnover_values: list[float] = []
        for dt in wide.index[valid_days_mask]:
            mask = both_valid.loc[dt]
            r_today = ranks.loc[dt, mask]
            r_prev = prev.loc[dt, mask]
            if r_today.std(ddof=0) == 0 or r_prev.std(ddof=0) == 0:
                # 任一侧 rank 全同 → Spearman 无定义，记为随机水平（0.5）是过度假设；
                # 更保守的做法：剔除该日。
                continue
            rho = float(np.corrcoef(r_today.to_numpy(), r_prev.to_numpy())[0, 1])
            if not np.isfinite(rho):
                continue
            daily_turnover_values.append((1.0 - rho) / 2.0)

        if not daily_turnover_values:
            raise ValueError(
                "turnover: 所有可对比日的截面 Spearman 都无定义（排名退化）"
            )
        daily_rank_turnover = float(np.mean(daily_turnover_values))
        # 数值夹紧到 [0, 1]（浮点误差边界保护）
        daily_rank_turnover = max(0.0, min(1.0, daily_rank_turnover))

        # ---- 2) rank_autocorr（lag=1 per instrument, 取中位数）----
        autocorrs: list[float] = []
        for inst in wide.columns:
            s = ranks[inst].dropna()
            if len(s) < 3:
                continue
            lag1 = s.autocorr(lag=1)
            if pd.notna(lag1):
                autocorrs.append(float(lag1))
        if not autocorrs:
            raise ValueError("turnover: 无任何股票能算出有效的 rank autocorr")
        rank_autocorr = float(np.median(autocorrs))

        # ---- 评分 ----
        part_turnover = max(0.0, 1.0 - daily_rank_turnover)
        part_autocorr = (rank_autocorr + 1.0) / 2.0
        score = 0.5 * part_turnover + 0.5 * part_autocorr
        score = max(0.0, min(1.0, score))

        passed = (
            daily_rank_turnover <= max_turnover
            and rank_autocorr >= min_autocorr
        )

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return CheckResult(
            name=self.name,
            passed=passed,
            score=round(score, 6),
            threshold=None,  # 本 check 两阈值组合，threshold 字段空置，阈值落进 detail
            detail={
                "daily_rank_turnover": round(daily_rank_turnover, 6),
                "rank_autocorr": round(rank_autocorr, 6),
                "max_daily_rank_turnover": max_turnover,
                "min_rank_autocorr": min_autocorr,
                "n_days": int(wide.shape[0]),
                "n_instruments": int(wide.shape[1]),
                "n_autocorr_samples": len(autocorrs),
                "n_spearman_days": len(daily_turnover_values),
                "algorithm": "(1 - cross_sectional_spearman) / 2, mean over days",
            },
            elapsed_ms=elapsed_ms,
        )

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _float_in(
        cfg: dict[str, Any], key: str, *, lo: float, hi: float
    ) -> float:
        if key not in cfg:
            raise ValueError(f"turnover: 配置缺字段 {key!r}")
        v = float(cfg[key])
        if not lo <= v <= hi:
            raise ValueError(
                f"turnover: {key} 必须在 [{lo}, {hi}] 区间: {v}"
            )
        return v

    @staticmethod
    def _slice_oos(series: pd.Series, context: CheckContext) -> pd.Series:
        start_d, end_d = context.oos_window
        start_ts = pd.Timestamp(start_d)
        # 闭区间：end 那天也要包括
        end_ts = pd.Timestamp(end_d) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        idx = series.index.get_level_values("datetime")
        mask = (idx >= start_ts) & (idx <= end_ts)
        return series.loc[mask]
