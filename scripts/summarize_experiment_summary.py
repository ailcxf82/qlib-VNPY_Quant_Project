"""
将 experiment_summary.csv 转为易读的终端摘要（成功/失败分组、核心指标、因子组对比）。

用法:
  python scripts/summarize_experiment_summary.py
  python scripts/summarize_experiment_summary.py --csv data/tuning/summary/experiment_summary.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 控制台尽量用 UTF-8（Windows）
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="易读打印 experiment_summary.csv")
    p.add_argument("--csv", type=str, default="data/tuning/summary/experiment_summary.csv", help="汇总 CSV 路径")
    p.add_argument("--metric", type=str, default="ic_qlib_ensemble_icir", help="排序主指标列名")
    p.add_argument("--top", type=int, default=15, help="成功实验中打印前 N 条")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    path = (PROJECT_ROOT / args.csv).resolve()
    if not path.exists():
        print(f"文件不存在: {path}")
        sys.exit(1)

    df = pd.read_csv(path, encoding="utf-8-sig")
    if df.empty:
        print("CSV 为空。")
        return

    print("=" * 72)
    print(f"来源: {path}")
    print(f"实验数: {len(df)}")
    print("=" * 72)

    # 状态分布
    if "status" in df.columns:
        print("\n【状态分布】")
        print(df["status"].value_counts(dropna=False).to_string())

    ok = df["status"].astype(str).str.lower().eq("ok") if "status" in df.columns else pd.Series([True] * len(df))
    failed = ~ok

    if failed.any():
        print("\n【失败 / 未纳入排序】（请先查各实验目录下 run.log）")
        cols = [c for c in ("experiment_id", "status", "returncode", "skip_reason", "notes") if c in df.columns]
        cols += [c for c in ("active_feature_sets",) if c in df.columns]
        sub = df.loc[failed, cols].head(50)
        print(sub.to_string(index=False))
        print(f"\n提示: 打开 data/tuning/runs_<名称>/<experiment_id>/run.log 查看 Traceback。")

    sub_ok = df.loc[ok].copy()
    if sub_ok.empty:
        print("\n【成功实验】无。请先修复失败原因后再做因子对比。")
        return

    metric = args.metric
    if metric not in sub_ok.columns:
        metric = "ic_qlib_ensemble_icir" if "ic_qlib_ensemble_icir" in sub_ok.columns else sub_ok.columns[-1]

    sub_ok[metric] = pd.to_numeric(sub_ok[metric], errors="coerce")
    sub_ok = sub_ok.sort_values(by=metric, ascending=False, na_position="last")

    show_cols = [
        c
        for c in (
            "experiment_id",
            metric,
            "valid_windows",
            "ic_qlib_ensemble_mean",
            "active_feature_sets",
            "n_active_feature_sets",
            "candidate_keep",
            "selection_reason",
        )
        if c in sub_ok.columns
    ]
    print(f"\n【成功实验】按 {metric} 降序（前 {args.top} 条）")
    print(sub_ok[show_cols].head(args.top).to_string(index=False))

    if "included_in_ranking" in sub_ok.columns:
        ranked = sub_ok[sub_ok["included_in_ranking"] == True]  # noqa: E712
        if not ranked.empty and metric in ranked.columns:
            best = ranked.iloc[0]
            print("\n【当前最优（含于 ranking 且指标最高）】")
            print(f"  experiment_id: {best.get('experiment_id', '')}")
            print(f"  {metric}: {best.get(metric, '')}")
            print(f"  active_feature_sets: {best.get('active_feature_sets', '')}")


if __name__ == "__main__":
    main()
