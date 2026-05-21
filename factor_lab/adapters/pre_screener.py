"""factor_lab.adapters.pre_screener
====================================
Phase-4：在 LLM 生成因子代码后、qrun 执行前，进行快速 IC 预筛选。

动机
----
qrun 完整执行一次约 3-10 分钟（含 LightGBM 训练 + PortAnaRecord）。
如果 LLM 提出的因子在训练集上 Rank IC < 0.005，几乎必然失败，
提前过滤可节省 60-80% 无效计算。

工作流
------
1. 从 workspace 中读取 result.h5（因子值面板）。
2. 从 daily_pv.h5 读取未来 1 日收益（或直接用 combined_factors_df.parquet 中已有的 label 列）。
3. 计算最近 252 个交易日的 Rank IC（快速，通常 < 2 秒）。
4. 若 Rank IC < threshold：返回 REJECT，RDAgent 跳过该因子的 qrun。
5. 若 Rank IC >= threshold：返回 PASS，继续 qrun。

集成方式
--------
在 factor_lab/adapters/experiments.py 中的 ``ProjectQlibFactorExperiment``
的 ``execute()`` 方法里，在 qrun 子进程启动前调用本模块：

    from factor_lab.adapters.pre_screener import pre_screen_result_h5
    if not pre_screen_result_h5(workspace_path / "result.h5"):
        logger.info("Pre-screener REJECTED factor, skipping qrun")
        return None  # 或返回失败的 result

注意：本模块是可选优化，任何异常均应 fail-open（不阻塞 qrun）。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

# 最低 Rank IC 阈值（可通过环境变量调整）
_DEFAULT_MIN_RANK_IC = 0.005
_DEFAULT_LOOKBACK_DAYS = 252


def _load_env_threshold() -> float:
    raw = os.environ.get("FACTOR_LAB_PRESCREENER_MIN_IC", "").strip()
    if not raw:
        return _DEFAULT_MIN_RANK_IC
    try:
        return float(raw)
    except ValueError:
        return _DEFAULT_MIN_RANK_IC


def _load_label_series(daily_pv_path: Path, lookback: int = 252) -> "pd.Series | None":  # type: ignore[name-defined]
    """从 daily_pv.h5 读取未来 1 日对数收益作为标签。
    返回 MultiIndex (datetime, instrument) Series，或 None（失败时 fail-open）。
    """
    try:
        import pandas as pd
        df = pd.read_hdf(daily_pv_path, key="data")
        if not isinstance(df.index, pd.MultiIndex):
            return None
        df.index = df.index.set_names(["datetime", "instrument"])
        # 取最近 lookback 天
        dates = df.index.get_level_values("datetime").unique().sort_values()
        if len(dates) > lookback:
            df = df[df.index.get_level_values("datetime") >= dates[-lookback]]
        # 未来1日收益
        close_col = "$close_qfq" if "$close_qfq" in df.columns else "$close"
        close = pd.to_numeric(df[close_col], errors="coerce")
        label = close.groupby(level="instrument").transform(
            lambda s: s.pct_change().shift(-1)  # forward 1-day return
        )
        return label.dropna()
    except Exception as exc:
        logger.debug("_load_label_series failed: %s", exc)
        return None


def _compute_rank_ic(
    factor_series: "pd.Series",
    label_series: "pd.Series",
    min_stocks_per_day: int = 20,
) -> float:
    """计算截面 Rank IC 均值。"""
    try:
        import pandas as pd
        import numpy as np

        common = factor_series.index.intersection(label_series.index)
        if len(common) < 200:
            return float("nan")
        f = factor_series.reindex(common)
        l = label_series.reindex(common)
        # 按日期计算截面 Spearman
        daily_ic = []
        for dt, group in pd.DataFrame({"f": f, "l": l}).groupby(level="datetime"):
            g = group.dropna()
            if len(g) < min_stocks_per_day:
                continue
            try:
                ic = g["f"].rank().corr(g["l"].rank(), method="spearman")
                if not np.isnan(ic):
                    daily_ic.append(ic)
            except Exception:
                continue
        if not daily_ic:
            return float("nan")
        return float(np.mean(daily_ic))
    except Exception as exc:
        logger.debug("_compute_rank_ic failed: %s", exc)
        return float("nan")


def pre_screen_result_h5(
    result_h5: Union[str, Path],
    daily_pv_h5: Union[str, Path, None] = None,
    threshold: float | None = None,
    lookback: int = _DEFAULT_LOOKBACK_DAYS,
) -> bool:
    """对 result.h5 中的因子进行快速 IC 预筛选。

    Parameters
    ----------
    result_h5 : str | Path
        因子值面板（workspace 内 result.h5）。
    daily_pv_h5 : str | Path | None
        daily_pv.h5 路径（用于读取标签）。None 时自动推断。
    threshold : float | None
        最低 Rank IC。None 时读环境变量 FACTOR_LAB_PRESCREENER_MIN_IC。
    lookback : int
        计算 IC 时使用最近多少个交易日。

    Returns
    -------
    bool
        True = PASS（继续 qrun），False = REJECT（跳过 qrun）。
        任何异常均返回 True（fail-open，不阻塞正常流程）。
    """
    if not os.environ.get("FACTOR_LAB_PRESCREENER_ENABLED", "").strip():
        return True  # 默认关闭，需显式启用

    result_path = Path(result_h5)
    if not result_path.exists():
        return True

    min_ic = threshold if threshold is not None else _load_env_threshold()

    try:
        import pandas as pd

        factor_df = pd.read_hdf(result_path, key="data")
        if isinstance(factor_df, pd.DataFrame):
            factor_series = factor_df.iloc[:, 0]
        else:
            factor_series = factor_df

        # 推断 daily_pv.h5 路径
        if daily_pv_h5 is None:
            workspace_dir = result_path.parent
            candidates = [
                workspace_dir / "daily_pv.h5",  # workspace 内（qrun 注入）
                result_path.parents[3] / "git_ignore_folder" / "daily_pv.h5",
            ]
            daily_pv_path = next((p for p in candidates if p.exists()), None)
            if daily_pv_path is None:
                logger.debug("daily_pv.h5 not found; pre-screener passes by default")
                return True
        else:
            daily_pv_path = Path(daily_pv_h5)

        label_series = _load_label_series(daily_pv_path, lookback=lookback)
        if label_series is None:
            return True

        rank_ic = _compute_rank_ic(factor_series, label_series)
        if rank_ic != rank_ic:  # NaN
            logger.debug("pre_screen_result_h5: IC=NaN → PASS (fail-open)")
            return True

        passed = abs(rank_ic) >= min_ic
        logger.info(
            "pre_screen_result_h5: factor=%s IC=%.4f threshold=%.4f → %s",
            result_path.parent.name,
            rank_ic,
            min_ic,
            "PASS" if passed else "REJECT",
        )
        return passed

    except Exception as exc:
        logger.debug("pre_screener error (fail-open): %s", exc)
        return True
