"""
汇总微调实验结果，生成实验级与窗口级对比表。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IC_COLUMNS = ["ic_lgb", "ic_gru", "ic_stack", "ic_qlib_ensemble"]


def _feature_scope_fields(meta: Dict[str, Any]) -> Dict[str, Any]:
    """从 result.json 的 data_scope 提取便于筛选因子组的列。"""
    ds = meta.get("data_scope") or {}
    afs = ds.get("active_feature_sets")
    out: Dict[str, Any] = {
        "scope_label": ds.get("label", ""),
        "scope_instruments": ds.get("instruments", ""),
        "scope_start_time": ds.get("start_time", ""),
        "scope_end_time": ds.get("end_time", ""),
    }
    if isinstance(afs, list):
        parts = sorted(str(x) for x in afs)
        out["active_feature_sets"] = "|".join(parts)
        out["n_active_feature_sets"] = len(afs)
        out["feature_sets_key"] = "|".join(parts)
    else:
        out["active_feature_sets"] = ""
        out["n_active_feature_sets"] = 0
        out["feature_sets_key"] = ""
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总微调实验结果")
    parser.add_argument("--runs-dir", type=str, default="data/tuning/runs", help="实验运行目录")
    parser.add_argument("--out-dir", type=str, default="data/tuning/summary", help="汇总输出目录")
    parser.add_argument(
        "--include-stale",
        action="store_true",
        help="包含非本次新鲜训练结果（默认跳过 dry_run/失败/历史残留）",
    )
    parser.add_argument("--selection-metric", type=str, default="ic_qlib_ensemble_icir", help="筛选候选因子组使用的指标列")
    parser.add_argument("--min-selection-score", type=float, default=0.0, help="候选最小得分阈值")
    parser.add_argument("--min-valid-windows", type=int, default=3, help="候选至少需要的有效窗口数")
    return parser.parse_args()


def icir(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2:
        return float("nan")
    return float(s.mean() / (s.std(ddof=0) + 1e-12))


def _diag_fields(meta: Dict[str, Any]) -> Dict[str, Any]:
    """提取 preflight 与 error_tail 汇总字段，便于失败汇总。"""
    pf = meta.get("preflight") or {}
    pf_status = str(pf.get("status", "")) if isinstance(pf, dict) else ""
    pf_fails = []
    if isinstance(pf, dict):
        for c in pf.get("checks", []) or []:
            if not bool(c.get("ok")):
                pf_fails.append(f"{c.get('name','?')}:{c.get('detail','')}")
    pf_fail_text = "; ".join(pf_fails)[:500]
    error_tail = str(meta.get("error_tail", "") or "")[:500]
    return {
        "preflight": pf_status,
        "preflight_failures": pf_fail_text,
        "error_tail": error_tail,
    }


def collect_one(run_dir: Path, include_stale: bool = False) -> tuple[Dict[str, Any], pd.DataFrame]:
    result_path = run_dir / "result.json"
    if not result_path.exists():
        return {}, pd.DataFrame()
    meta: Dict[str, Any] = json.loads(result_path.read_text(encoding="utf-8"))
    scope_cols = _feature_scope_fields(meta)
    diag_cols = _diag_fields(meta)
    metrics_path = run_dir / "logs" / "training_metrics.csv"
    trained = bool(meta.get("trained", False))
    status = str(meta.get("status", "unknown"))
    has_fresh_metrics = bool(meta.get("has_fresh_metrics", False))
    if (not include_stale) and (status != "ok" or not trained or not has_fresh_metrics):
        row = {
            "experiment_id": meta.get("experiment_id", run_dir.name),
            "model_type": meta.get("model_type"),
            "status": status,
            "returncode": meta.get("returncode"),
            "elapsed_seconds": meta.get("elapsed_seconds"),
            "valid_windows": 0,
            "all_windows": 0,
            "param_overrides": json.dumps(meta.get("param_overrides", {}), ensure_ascii=False),
            "data_scope": json.dumps(meta.get("data_scope", {}), ensure_ascii=False),
            "notes": meta.get("notes", ""),
            "included_in_ranking": False,
            "skip_reason": meta.get("skip_reason", "not_fresh_or_not_trained"),
        }
        row.update(scope_cols)
        row.update(diag_cols)
        return row, pd.DataFrame()

    if not metrics_path.exists():
        row = {
            "experiment_id": meta.get("experiment_id", run_dir.name),
            "model_type": meta.get("model_type"),
            "status": meta.get("status", "unknown"),
            "returncode": meta.get("returncode"),
            "elapsed_seconds": meta.get("elapsed_seconds"),
            "valid_windows": 0,
            "all_windows": 0,
            "param_overrides": json.dumps(meta.get("param_overrides", {}), ensure_ascii=False),
            "data_scope": json.dumps(meta.get("data_scope", {}), ensure_ascii=False),
            "notes": meta.get("notes", ""),
            "included_in_ranking": False,
            "skip_reason": "metrics_csv_missing",
        }
        row.update(scope_cols)
        row.update(diag_cols)
        return row, pd.DataFrame()

    df = pd.read_csv(metrics_path)
    exp_id = meta.get("experiment_id", run_dir.name)
    valid_df = df[df.get("segment", "valid") == "valid"] if "segment" in df.columns else df

    row: Dict[str, Any] = {
        "experiment_id": exp_id,
        "model_type": meta.get("model_type"),
        "status": meta.get("status", "unknown"),
        "returncode": meta.get("returncode"),
        "elapsed_seconds": meta.get("elapsed_seconds"),
        "valid_windows": int(len(valid_df)),
        "all_windows": int(len(df)),
        "param_overrides": json.dumps(meta.get("param_overrides", {}), ensure_ascii=False),
        "data_scope": json.dumps(meta.get("data_scope", {}), ensure_ascii=False),
        "notes": meta.get("notes", ""),
        "included_in_ranking": True,
        "skip_reason": "",
    }
    row.update(scope_cols)
    row.update(diag_cols)

    for col in IC_COLUMNS:
        if col in valid_df.columns:
            row[f"{col}_mean"] = float(pd.to_numeric(valid_df[col], errors="coerce").mean())
            row[f"{col}_std"] = float(pd.to_numeric(valid_df[col], errors="coerce").std(ddof=0))
            row[f"{col}_icir"] = icir(valid_df[col])
        else:
            row[f"{col}_mean"] = float("nan")
            row[f"{col}_std"] = float("nan")
            row[f"{col}_icir"] = float("nan")

    win = df.copy()
    win.insert(0, "experiment_id", exp_id)
    win.insert(1, "model_type", meta.get("model_type"))
    return row, win


def main() -> None:
    args = parse_args()
    runs_dir = (PROJECT_ROOT / args.runs_dir).resolve()
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    windows: List[pd.DataFrame] = []
    for run_dir in sorted([p for p in runs_dir.iterdir() if p.is_dir()]):
        row, win = collect_one(run_dir, include_stale=args.include_stale)
        if row:
            summary_rows.append(row)
        if not win.empty:
            windows.append(win)

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty and "ic_qlib_ensemble_icir" in summary_df.columns:
        sort_cols = ["included_in_ranking", "ic_qlib_ensemble_icir"]
        sort_asc = [False, False]
        if "feature_sets_key" in summary_df.columns:
            sort_cols.append("feature_sets_key")
            sort_asc.append(True)
        summary_df = summary_df.sort_values(by=sort_cols, ascending=sort_asc, na_position="last")
    if not summary_df.empty:
        score_col = args.selection_metric
        score_series = pd.to_numeric(summary_df.get(score_col, np.nan), errors="coerce")
        valid_windows = pd.to_numeric(summary_df.get("valid_windows", 0), errors="coerce").fillna(0)
        summary_df["selection_metric"] = score_col
        summary_df["selection_score"] = score_series
        summary_df["candidate_keep"] = (
            summary_df.get("included_in_ranking", False).astype(bool)
            & (valid_windows >= int(args.min_valid_windows))
            & (score_series >= float(args.min_selection_score))
        )
        summary_df["selection_reason"] = np.where(
            summary_df["candidate_keep"],
            "pass_threshold",
            "below_threshold_or_unstable",
        )
    summary_df.to_csv(out_dir / "experiment_summary.csv", index=False, encoding="utf-8-sig")

    if windows:
        window_df = pd.concat(windows, ignore_index=True)
    else:
        window_df = pd.DataFrame()
    window_df.to_csv(out_dir / "window_level_metrics.csv", index=False, encoding="utf-8-sig")

    print(f"summary: {out_dir / 'experiment_summary.csv'}")
    print(f"windows: {out_dir / 'window_level_metrics.csv'}")


if __name__ == "__main__":
    main()
