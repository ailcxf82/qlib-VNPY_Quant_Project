"""``marginal_training`` check —— 基于"微型增量训练"的 A/B 对比。

**动机**：``marginal`` check（D.2A）用线性中性化只能识别"与 baseline 线性独立"的新
信号；但真实 ensemble 是非线性的（LGB / GBDT），一个候选因子可能**只在特定交互下**
才贡献信息。本 check 直接以训练指标说话：

    Baseline 特征集 → 训 LGB → A_IC
    Baseline 特征集 + candidate → 训 LGB → B_IC
    uplift = B_IC - A_IC
    若 uplift >= 阈值 → candidate 带来真正的模型可学到的边际信息

**与 D.2A 的关系**：
* D.2A 是"**每个候选都跑**"的秒级过滤（残差 IC 法），适合 ``default`` profile。
* D.2B（本 check）是"**谨慎放行前的加码检查**"（训 2 个 LGB，秒-分钟级），
  适合 ``strict`` profile；通常放在 D.2A 之后。

**特征来源**：
* Baseline 特征：``CheckContext.reference_factors_parquet`` 的全部 active 列
  （也就是 L3 当前版本 ``factor_registry/parquet/factors_v<N>.parquet``）。
* Candidate：``candidate.values_path``。
* Label：``CheckContext.label_parquet`` 的 ``label`` 列。

**算法**：

1. 按 ``(datetime, instrument)`` inner join 候选 + baseline 矩阵 + label，OOS 窗切片。
2. **时间切分**：按时间升序取前 ``train_ratio`` 作 train，剩下作 test（防止时序泄漏）。
3. **A 模型**：baseline 特征 → LGB regressor → 在 test 上输出 ``y_hat_A``。
4. **B 模型**：baseline + candidate → 同超参 → 在 test 上输出 ``y_hat_B``。
5. **指标**：test 集上的截面 **Rank IC 均值**：

       ic_A = mean( daily Pearson(rank(y_hat_A), rank(label)) )
       ic_B = mean( daily Pearson(rank(y_hat_B), rank(label)) )
       uplift = ic_B - ic_A

6. **辅助指标**：
   * LGB ``feature_importance`` 中 candidate 列占比 ``candidate_importance_ratio``。
     （importance_type='gain'）
   * 可选：训练/测试 RMSE。

7. **判决**：``passed = uplift >= min_ic_uplift``；若 ``allow_negative=True``，
   用 ``abs(uplift) >= min_ic_uplift``（但此时 ``uplift`` 为负意味着候选和 baseline
   信号**相位相反**，需要人工确认而非直接收纳）。

**阈值字段**（profile）：

* ``min_ic_uplift``              必填；典型 0.002 ~ 0.005
* ``train_ratio``                可选；默认 0.6
* ``num_boost_round``            可选；默认 200
* ``learning_rate``              可选；默认 0.05
* ``num_leaves``                 可选；默认 31
* ``min_data_in_leaf``           可选；默认 50
* ``feature_fraction``           可选；默认 0.9
* ``bagging_fraction``           可选；默认 0.9
* ``bagging_freq``               可选；默认 5
* ``random_seed``                可选；默认 42（A/B 同种子保证 reproducibility）
* ``min_cross_samples_per_day``  可选；默认 5
* ``min_test_days``              可选；默认 10
* ``allow_negative``             可选；默认 False

**Score**：``min(1, max(0, uplift / min_ic_uplift))``（线性饱和）。

**性能预期**：baseline 30 列 × 250 天 × 300 股票 ≈ 2.25M 行 × 31 特征，LGB
``num_boost_round=200`` 在 4 核 CPU 大约 5-15 秒；两次训练 10-30 秒。若 ``strict``
profile 一轮跑 50 个候选，总耗时约 8-25 分钟，可接受。
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
_DEFAULT_MIN_CROSS = 5
_DEFAULT_MIN_TEST_DAYS = 10


class MarginalTrainingCheck(CheckBase):
    """微型增量训练 A/B 对比。"""

    name = "marginal_training"

    def run(
        self,
        candidate: CandidateFactorPackage,
        context: CheckContext,
        config: dict[str, Any],
    ) -> CheckResult:
        start = time.perf_counter()

        if "min_ic_uplift" not in config:
            raise ValueError("marginal_training: 配置缺字段 min_ic_uplift")
        min_uplift = float(config["min_ic_uplift"])
        if min_uplift <= 0:
            raise ValueError(
                f"marginal_training: min_ic_uplift 必须 > 0: {min_uplift}"
            )
        train_ratio = float(config.get("train_ratio", 0.6))
        if not 0.2 <= train_ratio <= 0.9:
            raise ValueError(
                f"marginal_training: train_ratio 必须在 [0.2, 0.9]: {train_ratio}"
            )
        min_cross = int(
            config.get("min_cross_samples_per_day", _DEFAULT_MIN_CROSS)
        )
        min_test_days = int(config.get("min_test_days", _DEFAULT_MIN_TEST_DAYS))
        allow_negative = bool(config.get("allow_negative", False))
        seed = int(config.get("random_seed", 42))

        lgb_params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": float(config.get("learning_rate", 0.05)),
            "num_leaves": int(config.get("num_leaves", 31)),
            "min_data_in_leaf": int(config.get("min_data_in_leaf", 50)),
            "feature_fraction": float(config.get("feature_fraction", 0.9)),
            "bagging_fraction": float(config.get("bagging_fraction", 0.9)),
            "bagging_freq": int(config.get("bagging_freq", 5)),
            "verbose": -1,
            "seed": seed,
            "deterministic": True,
        }
        num_boost_round = int(config.get("num_boost_round", 200))

        # ---- context 校验 ----
        if context.label_parquet is None or not context.label_parquet.exists():
            raise ValueError(
                f"marginal_training: label_parquet 未设置或不存在: "
                f"{context.label_parquet}"
            )
        if (
            context.reference_factors_parquet is None
            or not context.reference_factors_parquet.exists()
        ):
            raise ValueError(
                "marginal_training: reference_factors_parquet 未设置或不存在；"
                "profile.data_sources.reference_factors_parquet 必填（指向 L3 当前 parquet）"
            )

        # ---- 加载 + 对齐 ----
        cand_df = pd.read_parquet(candidate.values_path)
        if candidate.name not in cand_df.columns:
            raise ValueError(
                f"marginal_training: parquet 缺列 {candidate.name!r}; "
                f"实际: {list(cand_df.columns)}"
            )
        cand_series = self._slice_oos(cand_df[candidate.name], context)
        if cand_series.empty:
            raise ValueError("marginal_training: OOS 窗内候选因子无样本")

        label_df = pd.read_parquet(context.label_parquet)
        if "label" not in label_df.columns:
            raise ValueError(
                "marginal_training: label_parquet 必须包含 'label' 列，"
                f"实际: {list(label_df.columns)}"
            )
        label_series = self._slice_oos(label_df["label"], context)

        ref_df = pd.read_parquet(context.reference_factors_parquet)
        ref_df = self._slice_oos_frame(ref_df, context)
        if candidate.name in ref_df.columns:
            ref_df = ref_df.drop(columns=[candidate.name])
        if ref_df.shape[1] == 0:
            raise ValueError(
                "marginal_training: 参照因子集为空，无法训练 baseline 模型；"
                "至少要有 1 个已注册因子"
            )
        baseline_cols = list(ref_df.columns)

        # 统一三方 inner join
        merged = ref_df.copy()
        merged["__label__"] = label_series
        merged["__cand__"] = cand_series
        merged = merged.dropna()
        if merged.empty:
            raise ValueError(
                "marginal_training: baseline/label/candidate 对齐后无样本"
            )

        # ---- 时间切分 ----
        dates = merged.index.get_level_values("datetime").unique().sort_values()
        n_total = len(dates)
        n_train = int(n_total * train_ratio)
        if n_train < 2 or (n_total - n_train) < min_test_days:
            raise ValueError(
                f"marginal_training: 有效交易日不足 (total={n_total}, "
                f"train={n_train}, test={n_total - n_train}, "
                f"min_test_days={min_test_days})"
            )
        train_end = dates[n_train - 1]
        test_start = dates[n_train]
        tr_mask = merged.index.get_level_values("datetime") <= train_end
        te_mask = merged.index.get_level_values("datetime") >= test_start
        df_tr = merged.loc[tr_mask]
        df_te = merged.loc[te_mask]

        # ---- 训两个 LGB：A(baseline) vs B(baseline + cand) ----
        ic_A, pred_A = self._train_and_predict_ic(
            df_tr,
            df_te,
            features=baseline_cols,
            params=lgb_params,
            num_boost_round=num_boost_round,
            min_cross=min_cross,
            tag="A",
        )
        ic_B, pred_B, importance_b = self._train_and_predict_ic(
            df_tr,
            df_te,
            features=baseline_cols + ["__cand__"],
            params=lgb_params,
            num_boost_round=num_boost_round,
            min_cross=min_cross,
            tag="B",
            return_importance=True,
        )
        uplift = ic_B - ic_A
        cmp_uplift = abs(uplift) if allow_negative else uplift
        passed = cmp_uplift >= min_uplift

        # candidate importance 占比（importance_type='gain'）
        cand_imp = float(importance_b.get("__cand__", 0.0))
        total_imp = float(sum(importance_b.values()))
        cand_imp_ratio = cand_imp / total_imp if total_imp > 0 else 0.0

        score = max(0.0, min(1.0, cmp_uplift / max(min_uplift, _EPS)))

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return CheckResult(
            name=self.name,
            passed=passed,
            score=round(score, 6),
            threshold=None,
            detail={
                "ic_A_baseline": round(float(ic_A), 6),
                "ic_B_baseline_plus_candidate": round(float(ic_B), 6),
                "ic_uplift": round(float(uplift), 6),
                "min_ic_uplift": min_uplift,
                "allow_negative": allow_negative,
                "candidate_importance_gain": round(cand_imp, 3),
                "candidate_importance_ratio": round(cand_imp_ratio, 6),
                "n_train_days": int(n_train),
                "n_test_days": int(n_total - n_train),
                "n_baseline_features": len(baseline_cols),
                "n_train_rows": int(len(df_tr)),
                "n_test_rows": int(len(df_te)),
                "num_boost_round": num_boost_round,
                "seed": seed,
                "label_parquet": str(context.label_parquet),
                "reference_parquet": str(context.reference_factors_parquet),
            },
            elapsed_ms=elapsed_ms,
        )

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _slice_oos(series: pd.Series, context: CheckContext) -> pd.Series:
        start_d, end_d = context.oos_window
        st = pd.Timestamp(start_d)
        en = pd.Timestamp(end_d) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        idx = series.index.get_level_values("datetime")
        return series.loc[(idx >= st) & (idx <= en)]

    @staticmethod
    def _slice_oos_frame(df: pd.DataFrame, context: CheckContext) -> pd.DataFrame:
        start_d, end_d = context.oos_window
        st = pd.Timestamp(start_d)
        en = pd.Timestamp(end_d) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        idx = df.index.get_level_values("datetime")
        return df.loc[(idx >= st) & (idx <= en)]

    @staticmethod
    def _train_and_predict_ic(
        df_tr: pd.DataFrame,
        df_te: pd.DataFrame,
        features: list[str],
        params: dict[str, Any],
        num_boost_round: int,
        min_cross: int,
        tag: str,
        return_importance: bool = False,
    ) -> Any:
        """训练一个 LGB 回归模型，返回 test 集上的跨日 Rank-IC 均值。

        返回：
            - (ic_mean, pred_series) 或
            - (ic_mean, pred_series, importance_dict) （return_importance=True 时）
        """
        import lightgbm as lgb  # lazy import

        X_tr = df_tr[features].to_numpy(dtype=np.float64)
        y_tr = df_tr["__label__"].to_numpy(dtype=np.float64)
        X_te = df_te[features].to_numpy(dtype=np.float64)
        y_te = df_te["__label__"].to_numpy(dtype=np.float64)

        train_set = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
        booster = lgb.train(
            params=params,
            train_set=train_set,
            num_boost_round=num_boost_round,
            callbacks=[lgb.log_evaluation(period=0)],
        )
        y_hat = booster.predict(X_te)

        pred_series = pd.Series(
            y_hat, index=df_te.index, name=f"y_hat_{tag}", dtype="float64"
        )

        # 跨日 Rank-IC：每日截面 Pearson(rank(y_hat), rank(label))
        wide_pred = pred_series.unstack("instrument")
        wide_lbl = pd.Series(y_te, index=df_te.index, name="label").unstack(
            "instrument"
        )
        daily: list[float] = []
        for ts in wide_pred.index:
            p_row = wide_pred.loc[ts]
            l_row = wide_lbl.loc[ts]
            mask = p_row.notna() & l_row.notna()
            if mask.sum() < min_cross:
                continue
            p_rank = p_row[mask].rank(method="average").to_numpy()
            l_rank = l_row[mask].rank(method="average").to_numpy()
            if p_rank.std() == 0 or l_rank.std() == 0:
                continue
            v = float(np.corrcoef(p_rank, l_rank)[0, 1])
            if np.isfinite(v):
                daily.append(v)

        ic = float(np.mean(daily)) if daily else 0.0

        if return_importance:
            imp_raw = booster.feature_importance(importance_type="gain")
            imp = {f: float(v) for f, v in zip(features, imp_raw)}
            return ic, pred_series, imp
        return ic, pred_series
