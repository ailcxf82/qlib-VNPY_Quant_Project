"""``backtest`` check —— 候选因子在 OOS 窗内的"轻量回测"指标。

本 check 是阶段 D 的核心验证环节之一。与阶段 β 的真实 RQAlpha 回测（见
``RqalphaBacktestCheck``）相比，本轻量版：

* **无撮合、无费用、无滑点、无真实持仓约束**
* **纯 NumPy/Pandas 分位 long-short 组合**
* **秒级运行**，适合放进 ``default`` profile 每个候选都跑

**动机**：IC 高的因子不一定能构造出有 Sharpe 的组合（可能集中在极端分位、或信号
强但波动极大）。真实 RQAlpha 回测 5~15 分钟一个，放进 default profile 成本太高。
本 check 用"forward return 直接累加"的方法，能在秒级给出：

* 分位多空组合的年化 Sharpe
* 最大回撤
* 年化收益
* 胜率

这是对 ``ic`` check 的重要补充 —— IC 可能为正但组合无利可图（例如 IC 集中在
尾部极端样本），或 IC 平均但多空 spread 稳定。

**数据源**：

* 候选因子：``candidate.values_path`` 的单列 parquet，MultiIndex ``(datetime, instrument)``
* Forward return（label）：``CheckContext.label_parquet``，与 ``ic_check`` 同源。
  **假定 label = 下一交易日 return 或 5 日 forward return**（由 prepare_oos_labels
  决定；check 自身不 shift，直接用）

**算法**（每日重调仓，等权）：

1. 在 OOS 窗内对齐 candidate + label（内连接，dropna）
2. unstack 成 (date × instrument) 宽表
3. 每个交易日：
   a. 截面按因子值 rank（升序，1..N）
   b. top ``quantile`` 分位作为 **long leg**，bottom ``quantile`` 作为 **short leg**
   c. ``daily_ls_ret = mean(label[long]) - mean(label[short])``
   d. ``daily_long_ret = mean(label[long])`` （仅多头，供 annualized 对照）
4. 聚合：
   * ``annual_return = mean(daily_ls_ret) * trading_days_per_year``
   * ``annual_vol    = std(daily_ls_ret) * sqrt(trading_days_per_year)``
   * ``long_short_sharpe = annual_return / (annual_vol + eps)``
   * ``nav = cumprod(1 + daily_ls_ret)``
   * ``max_drawdown = max(1 - nav / running_max(nav))``
   * ``win_ratio = mean(daily_ls_ret > 0)``

**allow_negative 语义**：若配置 ``allow_negative=True`` 且 ``long_short_sharpe < 0``，
**保留原始指标用于 detail**，但 ``passed`` / ``score`` 按翻转后的 spread 判断（等价
于反向因子 = short the top, long the bottom）。``detail.direction`` 记录 +1/-1。

**阈值字段**（profile 配置）：

* ``quantile``              （必填；``(0, 0.5]``）—— 多/空各自取的分位，如 0.2
* ``min_long_short_sharpe`` （必填；>=0）—— 绝对值下限
* ``max_drawdown``          （必填；``(0, 1]``）—— 最大回撤上限（绝对值）
* ``min_annual_return``     （可选，默认 0.0；>=0）—— 年化 L-S 收益下限（允许负向
                             时对翻转后的年化收益比较）
* ``min_win_ratio``         （可选，默认 0.0；``[0, 1]``）—— 日胜率下限
* ``allow_negative``        （可选，默认 False）
* ``trading_days_per_year`` （可选，默认 252）
* ``min_daily_samples``     （可选，默认 10）—— 截面 top + bottom 合并后的最少股票数；
                             不足该数的交易日剔除（通常 universe 太小或 NaN 过多）
* ``min_valid_days``        （可选，默认 20）—— 有效交易日下限，否则抛错

**passed**：
    ``abs(long_short_sharpe) >= min_long_short_sharpe`` 且
    ``max_drawdown <= max_drawdown`` 且
    ``reference_annual_return >= min_annual_return`` 且
    ``reference_win_ratio >= min_win_ratio``
（reference_* 为方向翻转后的值，``allow_negative=False`` 时等同原值）

**score**：
    ``0.55 * sharpe_ratio + 0.30 * dd_ratio + 0.15 * win_ratio``，其中

    * ``sharpe_ratio = clip(abs_sharpe / min_long_short_sharpe, 0, 1)``
    * ``dd_ratio    = clip(1 - max_drawdown / max_drawdown_limit, 0, 1)``
    * ``win_ratio   = clip((win - 0.5) / 0.15, 0, 1)``  （0.5 = 随机，0.65 及以上满分）
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

_EPS = 1e-12
_DEFAULT_TRADING_DAYS = 252
_DEFAULT_MIN_DAILY_SAMPLES = 10
_DEFAULT_MIN_VALID_DAYS = 20


class BacktestCheck(CheckBase):
    """轻量分位 long-short 回测（无撮合 / 无费用）。"""

    name = "backtest"

    def run(
        self,
        candidate: CandidateFactorPackage,
        context: CheckContext,
        config: dict[str, Any],
    ) -> CheckResult:
        start = time.perf_counter()

        # ---- 读取并校验配置 ----
        quantile = self._float_in(config, "quantile", lo=1e-6, hi=0.5)
        min_sharpe = self._float_ge(config, "min_long_short_sharpe", lo=0.0)
        max_dd_limit = self._float_in(config, "max_drawdown", lo=1e-6, hi=1.0)
        min_ann_ret = float(config.get("min_annual_return", 0.0))
        min_win = float(config.get("min_win_ratio", 0.0))
        if not (0.0 <= min_win <= 1.0):
            raise ValueError(f"backtest: min_win_ratio 必须在 [0, 1]: {min_win}")
        allow_negative = bool(config.get("allow_negative", False))
        trading_days = int(config.get("trading_days_per_year", _DEFAULT_TRADING_DAYS))
        if trading_days <= 0:
            raise ValueError(
                f"backtest: trading_days_per_year 必须 > 0: {trading_days}"
            )
        min_daily_samples = int(
            config.get("min_daily_samples", _DEFAULT_MIN_DAILY_SAMPLES)
        )
        if min_daily_samples < 2:
            raise ValueError(
                f"backtest: min_daily_samples 必须 >= 2: {min_daily_samples}"
            )
        min_valid_days = int(config.get("min_valid_days", _DEFAULT_MIN_VALID_DAYS))
        if min_valid_days < 2:
            raise ValueError(
                f"backtest: min_valid_days 必须 >= 2: {min_valid_days}"
            )

        # ---- 校验上下文 ----
        if context.label_parquet is None:
            raise ValueError(
                "backtest: CheckContext.label_parquet 未设置；"
                "请先跑 scripts/validate/prepare_oos_labels.py 生成 forward return parquet"
            )
        if not context.label_parquet.exists():
            raise ValueError(
                f"backtest: label_parquet 不存在: {context.label_parquet}"
            )

        # ---- 加载候选因子 ----
        cand_df = pd.read_parquet(candidate.values_path)
        if candidate.name not in cand_df.columns:
            raise ValueError(
                f"backtest: parquet 缺列 {candidate.name!r}，"
                f"实际列: {list(cand_df.columns)}"
            )
        cand_series = cand_df[candidate.name]
        cand_series = self._slice_oos(cand_series, context)
        if cand_series.empty:
            raise ValueError("backtest: OOS 窗内候选因子无样本")

        # ---- 加载 label ----
        label_df = pd.read_parquet(context.label_parquet)
        if "label" not in label_df.columns:
            raise ValueError(
                f"backtest: label_parquet 必须包含 'label' 列，"
                f"实际: {list(label_df.columns)}"
            )
        label_series = label_df["label"]
        label_series = self._slice_oos(label_series, context)

        # ---- 对齐 ----
        merged = pd.concat(
            [cand_series.rename("factor"), label_series.rename("label")],
            axis=1,
            join="inner",
        ).dropna()
        if merged.empty:
            raise ValueError(
                "backtest: 候选因子与 label 对齐后无样本；请检查 OOS 窗 + universe"
            )

        factor_wide = merged["factor"].unstack("instrument")
        label_wide = merged["label"].unstack("instrument")
        factor_wide = factor_wide.loc[factor_wide.index.sort_values()]
        label_wide = label_wide.reindex(index=factor_wide.index)

        # ---- 逐日分位 long-short ----
        daily_ls_ret: list[float] = []
        daily_long_ret: list[float] = []
        daily_top_count: list[int] = []
        daily_bottom_count: list[int] = []

        for ts in factor_wide.index:
            f_row = factor_wide.loc[ts]
            l_row = label_wide.loc[ts]
            mask = f_row.notna() & l_row.notna()
            if mask.sum() < min_daily_samples:
                continue
            f = f_row[mask]
            l = l_row[mask]
            ranks = f.rank(method="average", ascending=True)
            n = len(f)
            k = max(1, int(np.floor(n * quantile)))
            # top = 因子值最大的 k 个 (高 rank)；bottom = 最小的 k 个 (低 rank)
            top_idx = ranks.nlargest(k).index
            bottom_idx = ranks.nsmallest(k).index
            # 分位重叠时（极少股票），舍弃该日
            if len(set(top_idx) & set(bottom_idx)) > 0:
                continue
            long_ret = float(l.loc[top_idx].mean())
            short_ret = float(l.loc[bottom_idx].mean())
            if not (np.isfinite(long_ret) and np.isfinite(short_ret)):
                continue
            daily_ls_ret.append(long_ret - short_ret)
            daily_long_ret.append(long_ret)
            daily_top_count.append(k)
            daily_bottom_count.append(k)

        if len(daily_ls_ret) < min_valid_days:
            raise ValueError(
                f"backtest: 有效交易日不足 {min_valid_days} (实际 {len(daily_ls_ret)}); "
                f"OOS 窗过短或 universe 太小"
            )

        arr = np.asarray(daily_ls_ret, dtype=float)
        mean_daily = float(arr.mean())
        std_daily = float(arr.std(ddof=1))
        annual_ret_raw = mean_daily * trading_days
        annual_vol = std_daily * np.sqrt(trading_days)
        sharpe_raw = annual_ret_raw / (annual_vol + _EPS)

        # 最大回撤（基于 nav = cumprod(1 + r)）
        nav = np.cumprod(1.0 + arr)
        running_max = np.maximum.accumulate(nav)
        # 保护：running_max 可能 <= 0 （极端崩盘），用小正数兜底
        with np.errstate(divide="ignore", invalid="ignore"):
            dd = np.where(running_max > 0, 1.0 - nav / running_max, 1.0)
        max_dd_raw = float(np.max(dd))
        max_dd_raw = max(0.0, min(1.0, max_dd_raw))

        win_ratio_raw = float((arr > 0).mean())

        # ---- 负向因子处理 ----
        direction = 1
        if allow_negative and sharpe_raw < 0:
            direction = -1
            # 翻转后：long 与 short 互换，收益符号反转
            arr_ref = -arr
            sharpe_ref = -sharpe_raw
            annual_ret_ref = -annual_ret_raw
            nav_ref = np.cumprod(1.0 + arr_ref)
            running_max_ref = np.maximum.accumulate(nav_ref)
            with np.errstate(divide="ignore", invalid="ignore"):
                dd_ref = np.where(
                    running_max_ref > 0, 1.0 - nav_ref / running_max_ref, 1.0
                )
            max_dd_ref = float(np.max(dd_ref))
            max_dd_ref = max(0.0, min(1.0, max_dd_ref))
            win_ratio_ref = float((arr_ref > 0).mean())
        else:
            sharpe_ref = sharpe_raw
            annual_ret_ref = annual_ret_raw
            max_dd_ref = max_dd_raw
            win_ratio_ref = win_ratio_raw

        # ---- 判决 ----
        passed = (
            abs(sharpe_raw) >= min_sharpe
            and max_dd_ref <= max_dd_limit
            and annual_ret_ref >= min_ann_ret
            and win_ratio_ref >= min_win
        )

        # ---- 评分 ----
        sharpe_ratio = min(1.0, max(0.0, abs(sharpe_raw) / max(min_sharpe, _EPS)))
        dd_ratio = min(1.0, max(0.0, 1.0 - max_dd_ref / max(max_dd_limit, _EPS)))
        # win 分：0.5 = 随机 → 0；0.65+ = 满分
        win_score = min(1.0, max(0.0, (win_ratio_ref - 0.5) / 0.15))
        score = 0.55 * sharpe_ratio + 0.30 * dd_ratio + 0.15 * win_score
        score = float(max(0.0, min(1.0, score)))

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return CheckResult(
            name=self.name,
            passed=passed,
            score=round(score, 6),
            threshold=None,
            detail={
                "long_short_sharpe": round(sharpe_raw, 6),
                "annual_return": round(annual_ret_raw, 6),
                "annual_vol": round(annual_vol, 6),
                "max_drawdown": round(max_dd_raw, 6),
                "win_ratio": round(win_ratio_raw, 6),
                "long_only_annual_return": round(
                    float(np.mean(daily_long_ret)) * trading_days, 6
                ),
                "direction": direction,
                "quantile": quantile,
                "min_long_short_sharpe": min_sharpe,
                "max_drawdown_limit": max_dd_limit,
                "min_annual_return": min_ann_ret,
                "min_win_ratio": min_win,
                "allow_negative": allow_negative,
                "n_valid_days": len(daily_ls_ret),
                "avg_daily_top_bottom_size": round(
                    float(np.mean(daily_top_count + daily_bottom_count)), 2
                ),
                "trading_days_per_year": trading_days,
                "algorithm": (
                    "quantile long-short, equal-weight, no cost, "
                    "forward-return aggregation"
                ),
            },
            elapsed_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------- helpers

    @staticmethod
    def _float_in(cfg: dict[str, Any], key: str, *, lo: float, hi: float) -> float:
        if key not in cfg:
            raise ValueError(f"backtest: 配置缺字段 {key!r}")
        v = float(cfg[key])
        if not lo <= v <= hi:
            raise ValueError(f"backtest: {key} 必须在 [{lo}, {hi}] 区间: {v}")
        return v

    @staticmethod
    def _float_ge(cfg: dict[str, Any], key: str, *, lo: float) -> float:
        if key not in cfg:
            raise ValueError(f"backtest: 配置缺字段 {key!r}")
        v = float(cfg[key])
        if v < lo:
            raise ValueError(f"backtest: {key} 必须 >= {lo}: {v}")
        return v

    @staticmethod
    def _slice_oos(series: pd.Series, context: CheckContext) -> pd.Series:
        start_d, end_d = context.oos_window
        start_ts = pd.Timestamp(start_d)
        end_ts = pd.Timestamp(end_d) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        idx = series.index.get_level_values("datetime")
        return series.loc[(idx >= start_ts) & (idx <= end_ts)]
