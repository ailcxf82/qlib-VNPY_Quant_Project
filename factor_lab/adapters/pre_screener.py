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
1. CoSTEER 在子 workspace 生成各因子面板（经 ``process_factor_data`` 合并为 DataFrame）。
2. **主路径**：``filter_new_factors_panel()`` 对合并后每个因子列做 Rank IC 检验（merge → qrun 之前）。
3. 从 ``daily_pv.h5`` 读取未来 1 日收益作为标签。
4. 若 |Rank IC| < threshold：剔除该列；全部剔除时跳过 qrun 并写 stub 指标。
5. 若 |Rank IC| >= threshold：保留该列进入 ``combined_factors_df.parquet`` → qrun。

集成方式
--------
- ``patch_qlib_conda._patch_process_factor_data_prescreen()`` 包装 ``process_factor_data``。
- ``QlibFBWorkspace.execute()`` 仍保留 result.h5 单文件检查（子 workspace 有 h5 时生效）。

注意：本模块是可选优化；未启用或计算失败时 fail-open（不阻塞 qrun）。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DAILY_PV = _PROJECT_ROOT / "git_ignore_folder" / "factor_implementation_source_data" / "daily_pv.h5"
_FALLBACK_DAILY_PV = _PROJECT_ROOT / "git_ignore_folder" / "daily_pv.h5"

# 最低 Rank IC 阈值（可通过环境变量调整）
_DEFAULT_MIN_RANK_IC = 0.015
_DEFAULT_LOOKBACK_DAYS = 252


def _resolve_daily_pv_path(workspace_dir: Path | None = None) -> Path | None:
    """Resolve daily_pv.h5 for label series (workspace symlink or project main file)."""
    if workspace_dir is not None:
        ws_h5 = workspace_dir / "daily_pv.h5"
        if ws_h5.exists():
            return ws_h5
    for candidate in (_DEFAULT_DAILY_PV, _FALLBACK_DAILY_PV):
        if candidate.exists():
            return candidate
    return None


def _log_prescreen_decision(factor_name: str, rank_ic: float, min_ic: float, passed: bool) -> None:
    msg = (
        f"pre_screen_factor: name={factor_name} IC={rank_ic:.4f} "
        f"threshold={min_ic:.4f} → {'PASS' if passed else 'REJECT'}"
    )
    logger.info(msg)
    print(msg, flush=True)


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
            lambda s: s.ffill().pct_change(fill_method=None).shift(-1)
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


def evaluate_prescreen(
    result_h5: Union[str, Path],
    daily_pv_h5: Union[str, Path, None] = None,
    threshold: float | None = None,
    lookback: int = _DEFAULT_LOOKBACK_DAYS,
) -> tuple[bool, float]:
    """Run Rank IC screen; return (passed, rank_ic). Disabled → (True, nan)."""
    if not os.environ.get("FACTOR_LAB_PRESCREENER_ENABLED", "").strip():
        return True, float("nan")

    result_path = Path(result_h5)
    if not result_path.exists():
        return True, float("nan")

    min_ic = threshold if threshold is not None else _load_env_threshold()

    try:
        import pandas as pd

        factor_df = pd.read_hdf(result_path, key="data")
        factor_series = factor_df.iloc[:, 0] if isinstance(factor_df, pd.DataFrame) else factor_df

        if daily_pv_h5 is None:
            daily_pv_path = _resolve_daily_pv_path(result_path.parent)
            if daily_pv_path is None:
                logger.debug("daily_pv.h5 not found; pre-screener passes by default")
                return True, float("nan")
        else:
            daily_pv_path = Path(daily_pv_h5)

        label_series = _load_label_series(daily_pv_path, lookback=lookback)
        if label_series is None:
            return True, float("nan")

        rank_ic = _compute_rank_ic(factor_series, label_series)
        if rank_ic != rank_ic:
            logger.debug("pre_screen_result_h5: IC=NaN → PASS (fail-open)")
            return True, float("nan")

        passed = abs(rank_ic) >= min_ic
        _log_prescreen_decision(result_path.stem, rank_ic, min_ic, passed)
        return passed, float(rank_ic)

    except Exception as exc:
        logger.debug("pre_screener error (fail-open): %s", exc)
        return True, float("nan")


def write_prescreen_reject_artifacts(
    workspace: Union[str, Path],
    rank_ic: float = float("nan"),
) -> None:
    """Write stub qlib_res.csv + ret.pkl so RD-Agent feedback can continue without qrun."""
    import pandas as pd

    ws = Path(workspace)
    ic = float(rank_ic) if rank_ic == rank_ic else 0.0
    metrics = pd.Series(
        {
            "IC": ic,
            "ICIR": float("nan"),
            "Rank IC": ic,
            "Rank ICIR": float("nan"),
            "1day.composite_score": 0.0,
            "1day.diversity_bonus": 0.0,
            "1day.enhanced_score": 0.0,
            "1day.excess_return_with_cost.information_ratio": float("nan"),
            "1day.excess_return_with_cost.annualized_turnover": float("nan"),
            "1day.excess_return_with_cost.annualized_return": float("nan"),
            "1day.excess_return_with_cost.max_drawdown": float("nan"),
        }
    )
    metrics.to_csv(ws / "qlib_res.csv", header=False)
    ret = pd.DataFrame({"return": [0.0]}, index=pd.DatetimeIndex(["2000-01-01"]))
    ret.to_pickle(ws / "ret.pkl")
    (ws / ".factor_lab_prescreen_rejected").write_text("1", encoding="utf-8")


def evaluate_prescreen_series(
    factor_series: "pd.Series",
    daily_pv_h5: Union[str, Path, None] = None,
    threshold: float | None = None,
    factor_name: str = "factor",
    lookback: int = _DEFAULT_LOOKBACK_DAYS,
) -> tuple[bool, float]:
    """Rank IC screen on a factor panel column (post process_factor_data merge)."""
    if not os.environ.get("FACTOR_LAB_PRESCREENER_ENABLED", "").strip():
        return True, float("nan")

    min_ic = threshold if threshold is not None else _load_env_threshold()
    try:
        if daily_pv_h5 is None:
            daily_pv_path = _resolve_daily_pv_path()
        else:
            daily_pv_path = Path(daily_pv_h5)
        if daily_pv_path is None:
            logger.debug("daily_pv.h5 not found; pre-screener passes by default")
            return True, float("nan")

        label_series = _load_label_series(daily_pv_path, lookback=lookback)
        if label_series is None:
            return True, float("nan")

        rank_ic = _compute_rank_ic(factor_series, label_series)
        if rank_ic != rank_ic:
            logger.debug("pre_screen_factor: IC=NaN → PASS (fail-open) name=%s", factor_name)
            return True, float("nan")

        passed = abs(rank_ic) >= min_ic
        _log_prescreen_decision(factor_name, rank_ic, min_ic, passed)
        return passed, float(rank_ic)
    except Exception as exc:
        logger.debug("evaluate_prescreen_series fail-open: %s", exc)
        return True, float("nan")


def filter_new_factors_panel(
    new_factors: "pd.DataFrame",
    threshold: float | None = None,
) -> "pd.DataFrame":
    """Drop factor columns that fail Rank IC pre-screen before qrun merge."""
    if not os.environ.get("FACTOR_LAB_PRESCREENER_ENABLED", "").strip():
        return new_factors
    if new_factors is None or new_factors.empty:
        return new_factors

    try:
        from rdagent.core.exception import FactorEmptyError

        kept_cols: list = []
        rejected = 0
        for col in new_factors.columns:
            name = str(col)
            passed, _ = evaluate_prescreen_series(
                new_factors[col],
                threshold=threshold,
                factor_name=name,
            )
            if passed:
                kept_cols.append(col)
            else:
                rejected += 1

        summary = (
            f"pre_screen_panel: kept={len(kept_cols)}/{len(new_factors.columns)} "
            f"rejected={rejected}"
        )
        logger.info(summary)
        print(summary, flush=True)

        if not kept_cols:
            raise FactorEmptyError(
                f"pre_screen: all {len(new_factors.columns)} new factor(s) rejected "
                f"(|Rank IC| < {_load_env_threshold() if threshold is None else threshold})"
            )

        return new_factors[kept_cols] if len(kept_cols) < len(new_factors.columns) else new_factors
    except Exception as exc:
        if exc.__class__.__name__ == "FactorEmptyError":
            raise
        logger.warning("filter_new_factors_panel fail-open: %s", exc)
        return new_factors


def finish_experiment_all_prescreen_rejected(exp: Any) -> Any:
    """When every new factor fails pre-screen and there is no SOTA pool, skip qrun."""
    import pandas as pd

    ws = exp.experiment_workspace.workspace_path
    write_prescreen_reject_artifacts(ws)
    stdout = f"[prescreen] all new factors rejected at {ws}; skipped qrun"
    logger.info(stdout)
    print(stdout, flush=True)
    result = pd.read_csv(ws / "qlib_res.csv", index_col=0).iloc[:, 0]
    exp.result = result
    exp.stdout = stdout
    return exp


def prescreen_workspace_before_qrun(workspace: Union[str, Path]) -> tuple["pd.Series", str] | None:
    """If REJECT, write stub artifacts and return metrics + stdout. None → proceed with qrun."""
    ws = Path(workspace)
    result_h5 = ws / "result.h5"
    if not result_h5.exists():
        return None
    passed, rank_ic = evaluate_prescreen(result_h5)
    if passed:
        return None
    write_prescreen_reject_artifacts(ws, rank_ic=rank_ic)
    stdout = f"[prescreen] REJECTED at {ws}; skipped qrun+read_exp_res (IC={rank_ic:.4f})"
    logger.info(stdout)
    print(stdout, flush=True)
    import pandas as pd

    return pd.read_csv(ws / "qlib_res.csv", index_col=0).iloc[:, 0], stdout


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
    passed, _ = evaluate_prescreen(
        result_h5,
        daily_pv_h5=daily_pv_h5,
        threshold=threshold,
        lookback=lookback,
    )
    return passed
