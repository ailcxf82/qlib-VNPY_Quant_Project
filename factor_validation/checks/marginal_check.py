"""``marginal`` check —— 候选因子对当前 baseline ensemble 的边际信息量（中性化 IC 法）。

**动机**：``ic`` check 只关心候选因子自身与 forward return 的相关性；但如果当前
production 模型（例如 LGB + GRU 的 dynamic ensemble）已经能解释这份相关，那么把该
因子加进 feature panel **不会带来额外收益**。阶段 D 需要一个"边际贡献"验证来过滤掉
这类"与 baseline 同向冗余"的因子。

**两种实现路线**：

1. **中性化残差法（本 check）**：对候选因子的**截面排名**做"对 baseline 排名的最小
   二乘中性化"，得到残差 rank，然后与 label 排名算 IC：

   * 若 candidate 的排序完全被 baseline 解释（冗余）→ 残差 ≈ 0 → IC ≈ 0 → FAIL
   * 若 candidate 与 baseline 正交并携带新信号 → 残差保留信号 → IC 显著 → PASS

   优点：**秒级**；无需重训模型；对 baseline 预测尺度不敏感（只用 rank）；可放进
   ``default`` profile 每个候选都跑。

   局限：只衡量"与 baseline 线性独立 + 对 label 线性有用"的部分，无法捕捉交互效应。
   这种捕捉需要 D.2B 的微型增量训练。

2. **微型增量训练**（D.2B，见 ``marginal_training_check.py``）。

**数据源**：

* 候选因子：``candidate.values_path``
* Label：``CheckContext.label_parquet``（与 ``ic_check`` 同源；forward return）
* Baseline 预测：``CheckContext.baseline_prediction_parquet``
  （MultiIndex ``(datetime, instrument)``，单列 ``prediction: float64``）

**算法**（每个交易日截面内）：

1. ``cand_rank, label_rank, base_rank = rank(cand), rank(label), rank(baseline)``
2. 对 baseline 做 rank 中性化（最小二乘）::

       x = base_rank - mean(base_rank)
       y = cand_rank - mean(cand_rank)
       beta = (x · y) / (x · x)
       cand_resid = y - beta * x

   ``cand_resid`` 是 candidate 中 **与 baseline 线性无关** 的那一部分。

3. 日 IC（Pearson 与 Spearman 等价，因为已 rank）::

       daily_residual_ic = Pearson(cand_resid, label_rank)

4. 参考指标（诊断用）::

       daily_label_ic   = Pearson(cand_rank, label_rank)   # 候选自身 IC
       daily_overlap_ic = Pearson(cand_rank, base_rank)    # 与 baseline 重叠度

5. 汇总：

   * ``residual_rank_ic  = mean(daily_residual_ic)``
   * ``residual_ic_ir    = mean / (std + eps)``
   * ``label_rank_ic``      （sanity 对照）
   * ``baseline_overlap_ic``（冗余度）

6. 阈值判决同 ``ic_check``（双阈值 + allow_negative）。

**语义速查**：

+---------------------------------+----------------+---------------------+---------------------+----------------------+
| 候选类型                         | label_rank_ic  | baseline_overlap_ic | residual_rank_ic    | 结论                 |
+=================================+================+=====================+=====================+======================+
| 独立新信号                       | 高             | 低                  | **高**              | PASS：有边际贡献     |
+---------------------------------+----------------+---------------------+---------------------+----------------------+
| 与 baseline 同向冗余             | 高             | **高**              | 接近 0              | FAIL：信息已被吸收   |
+---------------------------------+----------------+---------------------+---------------------+----------------------+
| 与 baseline 反向冗余             | 低             | 高（反向）          | 接近 0              | FAIL：no lift        |
+---------------------------------+----------------+---------------------+---------------------+----------------------+
| 无用因子                         | 接近 0         | 接近 0              | 接近 0              | FAIL                 |
+---------------------------------+----------------+---------------------+---------------------+----------------------+

**阈值字段**（profile 配置）：

* ``min_residual_rank_ic`` （必填；典型 0.01）
* ``min_residual_ic_ir``   （必填；典型 0.15）
* ``allow_negative``       （可选，默认 False）
* ``min_cross_samples_per_day`` （可选，默认 5）
* ``min_valid_days``       （可选，默认 20）

**Passed**：同时满足两个阈值。

**Score**：同 ic_check 风格，``0.5 * ratio_ic + 0.5 * ratio_ir``。
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

_DEFAULT_MIN_CROSS = 5
_DEFAULT_MIN_DAYS = 20
_EPS = 1e-12


class MarginalCheck(CheckBase):
    """residual IC 法的边际贡献 check。"""

    name = "marginal"

    def run(
        self,
        candidate: CandidateFactorPackage,
        context: CheckContext,
        config: dict[str, Any],
    ) -> CheckResult:
        start = time.perf_counter()

        if "min_residual_rank_ic" not in config:
            raise ValueError("marginal: 配置缺字段 min_residual_rank_ic")
        if "min_residual_ic_ir" not in config:
            raise ValueError("marginal: 配置缺字段 min_residual_ic_ir")
        min_rank_ic = float(config["min_residual_rank_ic"])
        min_ic_ir = float(config["min_residual_ic_ir"])
        allow_negative = bool(config.get("allow_negative", False))
        min_cross = int(config.get("min_cross_samples_per_day", _DEFAULT_MIN_CROSS))
        min_days = int(config.get("min_valid_days", _DEFAULT_MIN_DAYS))
        if min_cross < 3:
            raise ValueError(
                f"marginal: min_cross_samples_per_day 必须 >= 3: {min_cross}"
            )
        if min_days < 2:
            raise ValueError(f"marginal: min_valid_days 必须 >= 2: {min_days}")

        # ---- 校验 context ----
        if context.label_parquet is None:
            raise ValueError(
                "marginal: CheckContext.label_parquet 未设置；"
                "profile.data_sources.label_parquet 必填"
            )
        if not context.label_parquet.exists():
            raise ValueError(
                f"marginal: label_parquet 不存在: {context.label_parquet}"
            )
        if context.baseline_prediction_parquet is None:
            raise ValueError(
                "marginal: CheckContext.baseline_prediction_parquet 未设置；"
                "profile.data_sources.baseline_prediction_parquet 必填；"
                "它应指向当前 production ensemble 在 OOS 窗的预测（MultiIndex "
                "(datetime, instrument), 列 prediction: float64）"
            )
        if not context.baseline_prediction_parquet.exists():
            raise ValueError(
                f"marginal: baseline_prediction_parquet 不存在: "
                f"{context.baseline_prediction_parquet}"
            )

        # ---- 加载三方数据 ----
        cand_df = pd.read_parquet(candidate.values_path)
        if candidate.name not in cand_df.columns:
            raise ValueError(
                f"marginal: parquet 缺列 {candidate.name!r}，"
                f"实际列: {list(cand_df.columns)}"
            )
        cand_series = self._slice_oos(cand_df[candidate.name], context)
        if cand_series.empty:
            raise ValueError("marginal: OOS 窗内候选因子无样本")

        label_df = pd.read_parquet(context.label_parquet)
        if "label" not in label_df.columns:
            raise ValueError(
                f"marginal: label_parquet 必须包含 'label' 列，"
                f"实际: {list(label_df.columns)}"
            )
        label_series = self._slice_oos(label_df["label"], context)

        base_df = pd.read_parquet(context.baseline_prediction_parquet)
        if "prediction" not in base_df.columns:
            raise ValueError(
                f"marginal: baseline_prediction_parquet 必须包含 'prediction' 列，"
                f"实际: {list(base_df.columns)}"
            )
        base_series = self._slice_oos(base_df["prediction"], context)

        # ---- 三方 inner join ----
        merged = pd.concat(
            [
                cand_series.rename("factor"),
                label_series.rename("label"),
                base_series.rename("baseline"),
            ],
            axis=1,
            join="inner",
        ).dropna()
        if merged.empty:
            raise ValueError(
                "marginal: 候选因子 / label / baseline 三方对齐后无样本；"
                "请检查 OOS 窗 + universe + baseline 预测文件时间范围"
            )

        factor_wide = merged["factor"].unstack("instrument")
        label_wide = merged["label"].unstack("instrument")
        base_wide = merged["baseline"].unstack("instrument")

        daily_residual_ic: list[float] = []
        daily_label_ic: list[float] = []
        daily_overlap_ic: list[float] = []

        def _corr(a: np.ndarray, c: np.ndarray) -> float | None:
            sa, sc = a.std(), c.std()
            if sa == 0 or sc == 0:
                return None
            v = float(np.corrcoef(a, c)[0, 1])
            return v if np.isfinite(v) else None

        for ts in factor_wide.index:
            f_row = factor_wide.loc[ts]
            l_row = label_wide.loc[ts]
            b_row = base_wide.loc[ts]
            mask = f_row.notna() & l_row.notna() & b_row.notna()
            if mask.sum() < min_cross:
                continue
            f = f_row[mask]
            l = l_row[mask]
            b = b_row[mask]

            f_rank = f.rank(method="average").to_numpy()
            l_rank = l.rank(method="average").to_numpy()
            b_rank = b.rank(method="average").to_numpy()

            # ---- 对 baseline rank 做最小二乘中性化 ----
            x = b_rank - b_rank.mean()
            y = f_rank - f_rank.mean()
            xx = float(x @ x)
            if xx <= _EPS:
                # baseline rank 常数 → 退化为 cand_rank 自身（中性化无作用）
                cand_resid = y
            else:
                beta = float(x @ y) / xx
                cand_resid = y - beta * x

            ic_res = _corr(cand_resid, l_rank)
            ic_lbl = _corr(f_rank, l_rank)
            ic_ovr = _corr(f_rank, b_rank)

            if ic_res is None:
                continue
            daily_residual_ic.append(ic_res)
            if ic_lbl is not None:
                daily_label_ic.append(ic_lbl)
            if ic_ovr is not None:
                daily_overlap_ic.append(ic_ovr)

        if len(daily_residual_ic) < min_days:
            raise ValueError(
                f"marginal: 有效日数不足 {min_days} (实际 {len(daily_residual_ic)})；"
                f"OOS 窗过短或截面有效股票 < {min_cross}"
            )

        arr = np.asarray(daily_residual_ic, dtype=float)
        residual_rank_ic = float(arr.mean())
        residual_std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        residual_ic_ir = residual_rank_ic / (residual_std + _EPS)

        label_rank_ic = (
            float(np.mean(daily_label_ic)) if daily_label_ic else float("nan")
        )
        baseline_overlap_ic = (
            float(np.mean(daily_overlap_ic)) if daily_overlap_ic else float("nan")
        )

        # ---- 判决 ----
        cmp_ic = abs(residual_rank_ic) if allow_negative else residual_rank_ic
        cmp_ir = abs(residual_ic_ir) if allow_negative else residual_ic_ir
        passed = cmp_ic >= min_rank_ic and cmp_ir >= min_ic_ir

        # ---- 评分 ----
        def _ratio(v: float, t: float) -> float:
            return max(0.0, v) / max(t, _EPS)

        ratio_ic = min(_ratio(cmp_ic, min_rank_ic), 1.0)
        ratio_ir = min(_ratio(cmp_ir, min_ic_ir), 1.0)
        score = max(0.0, min(1.0, 0.5 * ratio_ic + 0.5 * ratio_ir))

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return CheckResult(
            name=self.name,
            passed=passed,
            score=round(score, 6),
            threshold=None,
            detail={
                "residual_rank_ic": round(residual_rank_ic, 6),
                "residual_ic_ir": round(residual_ic_ir, 6),
                "residual_ic_std": round(residual_std, 6),
                "label_rank_ic": (
                    round(label_rank_ic, 6)
                    if not np.isnan(label_rank_ic)
                    else None
                ),
                "baseline_overlap_ic": (
                    round(baseline_overlap_ic, 6)
                    if not np.isnan(baseline_overlap_ic)
                    else None
                ),
                "n_valid_days": len(daily_residual_ic),
                "min_residual_rank_ic": min_rank_ic,
                "min_residual_ic_ir": min_ic_ir,
                "allow_negative": allow_negative,
                "label_parquet": str(context.label_parquet),
                "baseline_prediction_parquet": str(
                    context.baseline_prediction_parquet
                ),
                "algorithm": "daily cross-section: cand_rank neutralized by "
                "baseline_rank (OLS), Pearson(cand_resid, label_rank)",
            },
            elapsed_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------- helpers
    @staticmethod
    def _slice_oos(series: pd.Series, context: CheckContext) -> pd.Series:
        start_d, end_d = context.oos_window
        start_ts = pd.Timestamp(start_d)
        end_ts = pd.Timestamp(end_d) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        idx = series.index.get_level_values("datetime")
        return series.loc[(idx >= start_ts) & (idx <= end_ts)]
