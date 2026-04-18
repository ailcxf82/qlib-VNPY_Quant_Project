"""
GRU 注意力 A/B 分析：比较两个实验的 ic_gru 稳定性。

配合 data/tuning/specs/gru_attention_ab.yaml 使用。
读取各 run 下 logs/training_metrics.csv 的 ic_gru 列，输出：
  - data/gru_ab/ab_summary.md：均值、std、负 IC 窗口占比、胜率
  - data/gru_ab/ic_gru_curve.png：窗口级 IC 对比曲线（若 matplotlib 可用）
  - data/gru_ab/ab_summary.json：机器可读摘要

Usage:
  python scripts/gru_ab_analyze.py \
      --runs-dir data/tuning/runs_gru_ab \
      --out-dir data/gru_ab
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("gru_ab_analyze")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRU 注意力 A/B 分析")
    p.add_argument("--runs-dir", type=str, default="data/tuning/runs_gru_ab")
    p.add_argument("--out-dir", type=str, default="data/gru_ab")
    p.add_argument("--metric", type=str, default="ic_gru",
                   help="要对比的列名（默认 ic_gru）")
    p.add_argument("--exp-ids", type=str, default="gru_attention_none,gru_attention_self",
                   help="逗号分隔的实验 id，顺序即为对比顺序（基线在前）")
    return p.parse_args()


def load_run_metrics(run_dir: Path, metric: str) -> Optional[pd.DataFrame]:
    metrics_path = run_dir / "logs" / "training_metrics.csv"
    if not metrics_path.exists():
        return None
    try:
        df = pd.read_csv(metrics_path)
    except Exception as e:
        logger.warning("读取 %s 失败: %s", metrics_path, e)
        return None
    if metric not in df.columns:
        logger.warning("run=%s 缺少指标列 %s，可用列: %s",
                       run_dir.name, metric, list(df.columns))
        return None
    if "segment" in df.columns:
        df = df[df["segment"].astype(str) == "valid"].copy()
    if "window" not in df.columns:
        df["window"] = np.arange(len(df))
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    return df[["window", metric]].dropna()


def summarize_series(s: pd.Series) -> Dict[str, float]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    n = int(len(s))
    if n == 0:
        return {"n_windows": 0, "ic_mean": float("nan"), "ic_std": float("nan"),
                "ic_icir": float("nan"), "neg_ic_ratio": float("nan"),
                "ic_min": float("nan"), "ic_max": float("nan")}
    std = float(s.std(ddof=0))
    icir = float(s.mean() / (std + 1e-12)) if std > 1e-12 else float("nan")
    return {
        "n_windows": n,
        "ic_mean": float(s.mean()),
        "ic_std": std,
        "ic_icir": icir,
        "neg_ic_ratio": float((s < 0).mean()),
        "ic_min": float(s.min()),
        "ic_max": float(s.max()),
    }


def plot_curve(
    series_map: Dict[str, pd.DataFrame],
    metric: str,
    out_path: Path,
) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning("matplotlib 不可用，跳过曲线: %s", e)
        return False
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name, df in series_map.items():
        ax.plot(df["window"].values, df[metric].values,
                marker="o", linewidth=1.4, label=name, alpha=0.85)
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("window")
    ax.set_ylabel(metric)
    ax.set_title(f"GRU A/B: {metric} per window")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def compute_win_rate(baseline: pd.Series, treat: pd.Series) -> Dict[str, float]:
    """对齐 window，计算逐窗胜率（treat > baseline 的比例）与差值均值。"""
    if baseline is None or treat is None:
        return {"win_rate": float("nan"), "mean_diff": float("nan"), "n_aligned": 0}
    df = pd.concat([baseline.rename("b"), treat.rename("t")], axis=1).dropna()
    n = int(len(df))
    if n == 0:
        return {"win_rate": float("nan"), "mean_diff": float("nan"), "n_aligned": 0}
    return {
        "win_rate": float((df["t"] > df["b"]).mean()),
        "mean_diff": float((df["t"] - df["b"]).mean()),
        "n_aligned": n,
    }


def render_md(
    exp_ids: List[str],
    per_exp: Dict[str, Dict[str, Any]],
    win_stats: Dict[str, float],
    metric: str,
    out_md: Path,
    curve_path: Optional[Path],
) -> None:
    lines: List[str] = []
    lines.append("# GRU 注意力 A/B 分析报告\n")
    lines.append(f"- 指标列：`{metric}`")
    lines.append(f"- 对比实验：{exp_ids}\n")

    lines.append("## 每实验摘要\n")
    lines.append("| experiment_id | n_windows | ic_mean | ic_std | ic_icir | neg_ic_ratio | ic_min | ic_max |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for eid in exp_ids:
        info = per_exp.get(eid) or {}
        s = info.get("summary", {})
        lines.append(
            f"| {eid} | {s.get('n_windows',0)} | "
            f"{s.get('ic_mean', float('nan')):.4f} | "
            f"{s.get('ic_std', float('nan')):.4f} | "
            f"{s.get('ic_icir', float('nan')):.4f} | "
            f"{s.get('neg_ic_ratio', float('nan')):.2%} | "
            f"{s.get('ic_min', float('nan')):.4f} | "
            f"{s.get('ic_max', float('nan')):.4f} |"
        )
    lines.append("")

    lines.append("## 胜率（treatment vs baseline）\n")
    lines.append(f"- baseline = `{exp_ids[0] if exp_ids else ''}`")
    lines.append(f"- treatment = `{exp_ids[1] if len(exp_ids) > 1 else ''}`")
    lines.append(f"- 对齐窗口数：{win_stats.get('n_aligned', 0)}")
    lines.append(f"- win_rate（treatment > baseline）：{win_stats.get('win_rate', float('nan')):.2%}")
    lines.append(f"- mean_diff（treatment - baseline）：{win_stats.get('mean_diff', float('nan')):.4f}\n")

    lines.append("## 结论草稿\n")
    lines.append("> 填写时请结合：1) treatment 的 ic_icir 是否显著高于 baseline；")
    lines.append(">            2) neg_ic_ratio 是否下降；3) win_rate 是否 > 0.5 且 mean_diff 为正。")
    lines.append("> 如 treatment 未展现优势，建议在 `config/model_gru.yaml` 将 `attention_type` 设为 `none`。\n")

    lines.append("## 缺失说明\n")
    for eid in exp_ids:
        info = per_exp.get(eid) or {}
        if not info.get("found"):
            lines.append(f"- `{eid}`：未找到 run（或 `logs/training_metrics.csv` 缺失）")
    if curve_path and curve_path.exists():
        rel = curve_path.name
        lines.append("")
        lines.append(f"![ic_curve]({rel})")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    runs_dir = (PROJECT_ROOT / args.runs_dir).resolve()
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_ids = [x.strip() for x in args.exp_ids.split(",") if x.strip()]
    if len(exp_ids) < 2:
        raise ValueError("--exp-ids 至少需要两个实验 id，例如 baseline,treatment")

    series_map: Dict[str, pd.DataFrame] = {}
    per_exp: Dict[str, Dict[str, Any]] = {}
    for eid in exp_ids:
        run_dir = runs_dir / eid
        df = load_run_metrics(run_dir, metric=args.metric) if run_dir.exists() else None
        if df is None or df.empty:
            per_exp[eid] = {"found": False, "summary": summarize_series(pd.Series(dtype=float))}
            logger.warning("实验 %s 未找到可用 metrics（runs_dir=%s）", eid, run_dir)
            continue
        series_map[eid] = df
        per_exp[eid] = {"found": True, "summary": summarize_series(df[args.metric])}

    baseline_id = exp_ids[0]
    treat_id = exp_ids[1]
    b_df = series_map.get(baseline_id)
    t_df = series_map.get(treat_id)
    if b_df is not None and t_df is not None:
        merged = b_df.merge(t_df, on="window", suffixes=("_b", "_t"))
        win_stats = compute_win_rate(
            merged[f"{args.metric}_b"], merged[f"{args.metric}_t"]
        )
    else:
        win_stats = {"win_rate": float("nan"), "mean_diff": float("nan"), "n_aligned": 0}

    curve_path = out_dir / "ic_gru_curve.png"
    curve_ok = False
    if series_map:
        curve_ok = plot_curve(series_map, metric=args.metric, out_path=curve_path)

    md_path = out_dir / "ab_summary.md"
    render_md(
        exp_ids=exp_ids,
        per_exp=per_exp,
        win_stats=win_stats,
        metric=args.metric,
        out_md=md_path,
        curve_path=curve_path if curve_ok else None,
    )

    json_path = out_dir / "ab_summary.json"
    json_path.write_text(json.dumps({
        "exp_ids": exp_ids,
        "per_exp": per_exp,
        "win_stats": win_stats,
        "metric": args.metric,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("完成。MD: %s", md_path)
    logger.info("JSON: %s", json_path)
    if curve_ok:
        logger.info("曲线: %s", curve_path)


if __name__ == "__main__":
    main()
