"""factor_lab.adapters.diversity_reward
=======================================
Phase-2 扩展：为 RD-Agent 的 composite_score 叠加"多样性奖励"。

动机
----
当因子库已有 N 个认证因子时，新因子与现有因子的相关性越低越好。
但原 composite_score = 1.0*IR + 2.0*IC_IR - 0.5*log(1+turnover) 不包含
多样性项，导致 LLM 倾向于在已有高分家族里继续"微调"。

方案
----
enhanced_score = composite_score + DIVERSITY_WEIGHT * (1 - max_abs_spearman)
  其中 max_abs_spearman = max over existing certified factors of
                          |Spearman(new_factor, certified_factor)|
  DIVERSITY_WEIGHT = 0.4  (约为一个普通 IC_IR 项的 1/5，避免压制真实信号质量)

调用方式
--------
在 rdagent_overrides/factor_template/read_exp_res.py 的末尾注入：

    from factor_lab.adapters.diversity_reward import compute_diversity_bonus
    diversity_bonus = compute_diversity_bonus(result_h5_path, certified_parquet_path)
    composite += diversity_bonus

公开 API
--------
* compute_diversity_bonus(result_h5: str | Path, certified_parquet: str | Path,
                          weight: float = 0.4) -> float
* max_spearman_with_pool(new_series: pd.Series, pool_df: pd.DataFrame) -> float
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

DIVERSITY_WEIGHT = 0.4  # 多样性奖励权重，可通过环境变量 FACTOR_LAB_DIVERSITY_WEIGHT 覆盖


def _load_env_weight() -> float:
    import os
    raw = os.environ.get("FACTOR_LAB_DIVERSITY_WEIGHT", "").strip()
    if not raw:
        return DIVERSITY_WEIGHT
    try:
        return float(raw)
    except ValueError:
        return DIVERSITY_WEIGHT


def max_spearman_with_pool(new_series: "pd.Series", pool_df: "pd.DataFrame") -> float:  # type: ignore[name-defined]
    """计算 new_series 与 pool_df 中每列的截面 Spearman 相关均值的最大绝对值。

    Parameters
    ----------
    new_series : pd.Series
        新因子的面板序列（MultiIndex datetime × instrument）。
    pool_df : pd.DataFrame
        现有因子池面板（同 MultiIndex），每列是一个认证因子。

    Returns
    -------
    float
        max |Spearman(new, existing)| across all existing factors.
        如果 pool_df 为空或计算失败，返回 0.0（不惩罚多样性）。
    """
    try:
        import pandas as pd
        import numpy as np

        if pool_df.empty or new_series.empty:
            return 0.0

        # 对齐 index
        common_idx = new_series.index.intersection(pool_df.index)
        if len(common_idx) < 100:
            return 0.0

        new_aligned = new_series.reindex(common_idx)
        pool_aligned = pool_df.reindex(common_idx)

        # 按日期计算截面 Spearman，再取时序均值
        max_corr = 0.0
        for col in pool_aligned.columns:
            try:
                paired = pd.concat(
                    [new_aligned.rename("new"), pool_aligned[col].rename("old")], axis=1
                ).dropna()
                if len(paired) < 50:
                    continue
                # 截面 rank correlation: groupby date then corr
                daily_corr = (
                    paired.groupby(level="datetime")
                    .apply(lambda g: g["new"].rank().corr(g["old"].rank(), method="spearman"))
                )
                abs_mean = float(daily_corr.abs().mean())
                if abs_mean > max_corr:
                    max_corr = abs_mean
            except Exception:
                continue
        return min(max_corr, 1.0)
    except Exception as exc:
        logger.warning("max_spearman_with_pool failed: %s", exc)
        return 0.0


def compute_diversity_bonus(
    result_h5: Union[str, "Path"],
    certified_parquet: Union[str, "Path"],
    weight: float | None = None,
) -> float:
    """计算新因子的多样性奖励分数。

    Parameters
    ----------
    result_h5 : str | Path
        新因子的 result.h5 路径（workspace 内）。
    certified_parquet : str | Path
        现有认证因子的 combined parquet 路径。
        通常是 ``git_ignore_folder/combined_factors_df.parquet``。
    weight : float, optional
        奖励权重。默认读 FACTOR_LAB_DIVERSITY_WEIGHT 环境变量，回退到 0.4。

    Returns
    -------
    float
        diversity_bonus = weight * (1 - max_abs_spearman).
        范围 [0, weight]；max_corr=0 时奖励最大，max_corr=1 时奖励为 0。
    """
    if weight is None:
        weight = _load_env_weight()
    if weight <= 0:
        return 0.0

    try:
        import pandas as pd

        result_path = Path(result_h5)
        pool_path = Path(certified_parquet)

        if not result_path.exists():
            logger.debug("result_h5 not found, diversity_bonus=0: %s", result_path)
            return 0.0
        if not pool_path.exists():
            logger.debug("certified_parquet not found, diversity_bonus=0: %s", pool_path)
            return weight  # 池子为空 → 新因子完全独特 → 满分奖励

        new_df = pd.read_hdf(result_path, key="data")
        if new_df.empty:
            return 0.0

        # result.h5 只有一列 — 取该列作为新因子序列
        if isinstance(new_df, pd.DataFrame):
            new_series = new_df.iloc[:, 0]
        else:
            new_series = new_df  # type: ignore[assignment]

        pool_df = pd.read_parquet(pool_path)
        if pool_df.empty:
            return weight

        max_corr = max_spearman_with_pool(new_series, pool_df)
        bonus = weight * (1.0 - max_corr)
        logger.info(
            "diversity_bonus=%.4f (max_spearman=%.3f, weight=%.2f)",
            bonus, max_corr, weight,
        )
        return float(bonus)

    except Exception as exc:
        logger.warning("compute_diversity_bonus failed: %s", exc)
        return 0.0


def _normalize_parquet_columns(df: "pd.DataFrame") -> "pd.DataFrame":  # type: ignore[name-defined]
    """Flatten MultiIndex columns to plain names if needed."""
    import pandas as pd

    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            str(c[-1]) if isinstance(c, tuple) else str(c) for c in df.columns
        ]
    return df


def compute_diversity_bonus_from_workspace(
    workspace_dir: Union[str, "Path"],
    certified_parquet: Union[str, "Path"],
    weight: float | None = None,
) -> float:
    """Score diversity using workspace ``combined_factors_df.parquet`` vs certified pool.

    New factor columns = workspace columns not present in the certified parquet.
    Uses max bonus across new columns (most conservative vs pool orthogonality).
    """
    if weight is None:
        weight = _load_env_weight()
    if weight <= 0:
        return 0.0

    try:
        import pandas as pd

        ws_path = Path(workspace_dir) / "combined_factors_df.parquet"
        pool_path = Path(certified_parquet)

        if not ws_path.exists():
            logger.debug("combined_factors_df.parquet not found: %s", ws_path)
            return 0.0
        if not pool_path.exists():
            logger.debug("certified_parquet not found, diversity_bonus=max: %s", pool_path)
            return weight

        ws_df = _normalize_parquet_columns(pd.read_parquet(ws_path))
        pool_df = _normalize_parquet_columns(pd.read_parquet(pool_path))

        if ws_df.empty:
            return 0.0

        pool_cols = {str(c) for c in pool_df.columns}
        new_cols = [str(c) for c in ws_df.columns if str(c) not in pool_cols]
        if not new_cols:
            new_cols = [str(c) for c in ws_df.columns]

        if pool_df.empty:
            return weight

        best_bonus = 0.0
        worst_corr = 0.0
        for col in new_cols:
            try:
                series = ws_df[col]
                max_corr = max_spearman_with_pool(series, pool_df)
                bonus = weight * (1.0 - max_corr)
                if bonus > best_bonus:
                    best_bonus = bonus
                    worst_corr = max_corr
            except Exception:
                continue

        logger.info(
            "diversity_bonus=%.4f (max_spearman=%.3f, n_new_cols=%d, weight=%.2f)",
            best_bonus,
            worst_corr,
            len(new_cols),
            weight,
        )
        return float(best_bonus)

    except Exception as exc:
        logger.warning("compute_diversity_bonus_from_workspace failed: %s", exc)
        return 0.0
