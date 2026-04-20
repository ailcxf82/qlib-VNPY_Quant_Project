"""
Custom RD-Agent reward signal for model_template (P3a).

Same logic as factor_template/read_exp_res.py: the LLM optimisation target now
includes turnover penalty + composite score in addition to the default IC and
annualised return + max drawdown that RD-Agent already tracks.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import qlib

qlib.init()

from qlib.workflow import R

OUT_DIR = Path(__file__).resolve().parent
RET_PKL = Path("ret.pkl")


def _latest_recorder():
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
    try:
        df = rec.load_object("portfolio_analysis/report_normal_1day.pkl")
        if "turnover" in df.columns:
            return float(df["turnover"].mean()) * 252.0
    except Exception as exc:
        print(f"[warn] turnover unavailable: {exc}")
    return float("nan")


def _composite_score(ir: float, ic_ir: float, ann_turnover: float) -> float:
    parts = []
    if ir == ir:
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
        print("[error] no recorder found")
        empty = pd.Series(dtype="float64", name="value")
        empty.to_csv(OUT_DIR / "qlib_res.csv")
        pd.DataFrame({0: empty}).to_pickle(RET_PKL)
        return

    metrics = pd.Series(rec.list_metrics())
    ann_ret = _safe_get(metrics, "1day.excess_return_with_cost.annualized_return")
    max_dd = _safe_get(metrics, "1day.excess_return_with_cost.max_drawdown")
    info_ratio = _safe_get(metrics, "1day.excess_return_with_cost.information_ratio")
    ic_ir = _ic_ir(metrics)
    ann_turnover = _annualized_turnover(rec)
    composite = _composite_score(info_ratio, ic_ir, ann_turnover)

    metrics["1day.excess_return_with_cost.information_ratio"] = info_ratio
    metrics["1day.excess_return_with_cost.annualized_turnover"] = ann_turnover
    metrics["1day.composite_score"] = composite
    metrics["ICIR_proxy"] = ic_ir
    metrics.to_csv(OUT_DIR / "qlib_res.csv")

    print(
        f"[reward] composite={composite:.6f} | IR={info_ratio:.4f} "
        f"IC_IR={ic_ir:.4f} ann_ret={ann_ret:.4f} "
        f"max_dd={max_dd:.4f} ann_turnover={ann_turnover:.4f}"
    )

    try:
        ret_df = rec.load_object("portfolio_analysis/report_normal_1day.pkl")
        ret_df.to_pickle(RET_PKL)
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
        print(f"[warn] PortAnaRecord unavailable ({exc}); fallback ret.pkl written")


if __name__ == "__main__":
    main()
