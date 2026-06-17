"""
Custom RD-Agent reward signal (P3a + Phase-2 diversity).

Executed inside the RD-Agent qlib conda env after `qrun` produces a mlflow recorder.
Loads the latest recorder, then writes:

  - qlib_res.csv : metric Series consumed by RD-Agent feedback (`exp.result`).
                   Keys added beyond standard qlib metrics:
                       1day.excess_return_with_cost.information_ratio
                       1day.excess_return_with_cost.annualized_turnover
                       1day.composite_score          (P3a reward)
                       1day.diversity_bonus          (Phase-2: orthogonality to pool)
                       1day.enhanced_score           (composite + diversity_bonus)
  - ret.pkl      : DataFrame for RD-Agent workspace backtesting chart.

enhanced_score = composite_score + DIVERSITY_WEIGHT * (1 - max_spearman_with_pool)
  where composite_score = 1.0*IR + 2.0*IC_IR - 0.5*log(1+annualized_turnover)
  DIVERSITY_WEIGHT defaults to 0.4; override via FACTOR_LAB_DIVERSITY_WEIGHT env var.
  If certified_parquet is unavailable, diversity_bonus = DIVERSITY_WEIGHT (max reward).
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pandas as pd
import qlib

qlib.init()

from qlib.workflow import R

OUT_DIR = Path(__file__).resolve().parent
RET_PKL = Path("ret.pkl")  # cwd, kept for workspace.log_object compatibility

# Phase-2: certified pool parquet for diversity scoring.
# Path is resolved at runtime; can be overridden via FACTOR_LAB_CERTIFIED_PARQUET env.
_DEFAULT_CERTIFIED_PARQUET = (
    Path(__file__).resolve().parents[3]
    / "git_ignore_folder"
    / "combined_factors_df.parquet"
)
CERTIFIED_PARQUET = Path(
    os.environ.get("FACTOR_LAB_CERTIFIED_PARQUET", str(_DEFAULT_CERTIFIED_PARQUET))
)


def _compute_diversity_bonus(workspace_dir: Path) -> float:
    """Diversity bonus from workspace parquet (primary) or result.h5 (fallback)."""
    try:
        project_root = str(Path(__file__).resolve().parents[3])
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from factor_lab.adapters.diversity_reward import (
            compute_diversity_bonus,
            compute_diversity_bonus_from_workspace,
        )

        parquet_path = workspace_dir / "combined_factors_df.parquet"
        if parquet_path.exists():
            return compute_diversity_bonus_from_workspace(workspace_dir, CERTIFIED_PARQUET)

        result_h5 = workspace_dir / "result.h5"
        if result_h5.exists():
            return compute_diversity_bonus(result_h5, CERTIFIED_PARQUET)
    except ImportError:
        pass
    except Exception as exc:
        print(f"[warn] diversity_bonus unavailable: {exc}")
    return 0.0


def _latest_recorder():
    """Return the most-recently-completed recorder across all experiments."""
    latest = None
    for exp_name in R.list_experiments():
        for rec_id in R.list_recorders(experiment_name=exp_name):
            if rec_id is None:
                continue
            try:
                rec = R.get_recorder(recorder_id=rec_id, experiment_name=exp_name)
                end_time = rec.info.get("end_time")
                if end_time is None:
                    continue
                if latest is None or end_time > latest.info["end_time"]:
                    latest = rec
            except Exception as exc:
                print(f"[warn] skipped recorder {rec_id}: {exc}")
    return latest


def _ic_ir(metrics: pd.Series) -> float:
    """Best-effort extraction of IC information-ratio (IC mean / IC std)."""
    for key in ("ICIR", "Rank ICIR", "ic_ir", "RankICIR"):
        if key in metrics.index:
            try:
                v = float(metrics[key])
                if not math.isnan(v):
                    return v
            except Exception:
                pass
    ic_mean = ic_std = None
    for k in ("IC", "ic"):
        if k in metrics.index:
            try:
                ic_mean = float(metrics[k])
                break
            except Exception:
                pass
    for k in ("ICStd", "IC.std", "ic_std"):
        if k in metrics.index:
            try:
                ic_std = float(metrics[k])
                break
            except Exception:
                pass
    if ic_mean is not None and ic_std and ic_std > 0:
        return ic_mean / ic_std
    return float("nan")


def _annualized_turnover(rec) -> float:
    """Annualised average daily turnover from PortAnaRecord per-day report."""
    try:
        df = rec.load_object("portfolio_analysis/report_normal_1day.pkl")
        if "turnover" in df.columns:
            return float(df["turnover"].mean()) * 252.0
    except Exception as exc:
        print(f"[warn] could not load turnover from report_normal_1day.pkl: {exc}")
    return float("nan")


def _composite_score(ir: float, ic_ir: float, ann_turnover: float) -> float:
    """Larger is better; rewards signal IR/IC_IR, penalises turnover."""
    parts = []
    if ir == ir:  # not NaN
        parts.append(1.0 * ir)
    if ic_ir == ic_ir:
        parts.append(2.0 * ic_ir)
    if ann_turnover == ann_turnover and ann_turnover > 0:
        parts.append(-0.5 * math.log(1.0 + ann_turnover))
    if not parts:
        return float("nan")
    return float(sum(parts))


def _safe_get(metrics: pd.Series, key: str) -> float:
    if key not in metrics.index:
        return float("nan")
    try:
        return float(metrics[key])
    except Exception:
        return float("nan")


def main() -> None:
    rec = _latest_recorder()
    if rec is None:
        print("[error] no recorder found; writing minimal ret.pkl + qlib_res.csv")
        empty = pd.Series(dtype="float64", name="value")
        empty.to_csv(OUT_DIR / "qlib_res.csv")
        pd.DataFrame({0: empty}).to_pickle(RET_PKL)
        return

    print(
        f"[ok] latest recorder: id={rec.info.get('id')} "
        f"end={rec.info.get('end_time')}"
    )

    metrics = pd.Series(rec.list_metrics())
    print(f"[ok] {len(metrics)} mlflow metrics found")

    ann_ret = _safe_get(metrics, "1day.excess_return_with_cost.annualized_return")
    max_dd = _safe_get(metrics, "1day.excess_return_with_cost.max_drawdown")
    info_ratio = _safe_get(metrics, "1day.excess_return_with_cost.information_ratio")
    ic_ir = _ic_ir(metrics)
    ann_turnover = _annualized_turnover(rec)
    composite = _composite_score(info_ratio, ic_ir, ann_turnover)

    # Phase-2: diversity bonus — rewards orthogonality to existing certified pool
    diversity_bonus = _compute_diversity_bonus(OUT_DIR)
    enhanced_score = (composite + diversity_bonus) if composite == composite else float("nan")

    metrics["1day.excess_return_with_cost.information_ratio"] = info_ratio
    metrics["1day.excess_return_with_cost.annualized_turnover"] = ann_turnover
    metrics["1day.composite_score"] = composite
    metrics["1day.diversity_bonus"] = diversity_bonus
    metrics["1day.enhanced_score"] = enhanced_score
    metrics["ICIR_proxy"] = ic_ir

    output_path = OUT_DIR / "qlib_res.csv"
    metrics.to_csv(output_path)
    print(f"[ok] qlib_res.csv -> {output_path} ({len(metrics)} rows)")

    print(
        f"[reward] enhanced={enhanced_score:.4f} | composite={composite:.4f} "
        f"diversity_bonus={diversity_bonus:.4f} | IR={info_ratio:.4f} "
        f"IC_IR={ic_ir:.4f} ann_ret={ann_ret:.4f} "
        f"max_dd={max_dd:.4f} ann_turnover={ann_turnover:.4f}"
    )

    try:
        ret_df = rec.load_object("portfolio_analysis/report_normal_1day.pkl")
        ret_df.to_pickle(RET_PKL)
        print(f"[ok] ret.pkl written from PortAnaRecord ({len(ret_df)} rows)")
    except Exception as exc:
        fallback = pd.DataFrame(
            {
                "composite_score": [composite],
                "information_ratio": [info_ratio],
                "annualized_turnover": [ann_turnover],
                "annualized_return": [ann_ret],
                "max_drawdown": [max_dd],
                "ICIR": [ic_ir],
            }
        )
        fallback.to_pickle(RET_PKL)
        print(f"[warn] PortAnaRecord chart unavailable ({exc}); fallback ret.pkl written")


if __name__ == "__main__":
    main()
