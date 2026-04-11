"""
生成离线 HTML 微调对比报告。
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from pandas.errors import EmptyDataError

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="渲染微调对比 HTML 报告")
    parser.add_argument("--summary-csv", type=str, default="data/tuning/summary/experiment_summary.csv")
    parser.add_argument("--window-csv", type=str, default="data/tuning/summary/window_level_metrics.csv")
    parser.add_argument("--out", type=str, default="data/tuning/summary/tuning_report.html")
    parser.add_argument("--topn", type=int, default=12)
    return parser.parse_args()


def _bar_svg(items: List[Tuple[str, float]], title: str, width: int = 1100, height: int = 320) -> str:
    if not items:
        return "<p>暂无可视化数据</p>"
    vals = [v for _, v in items]
    vmin, vmax = min(vals), max(vals)
    if vmax == vmin:
        vmax = vmin + 1e-9
    margin = 40
    chart_w = width - margin * 2
    chart_h = height - margin * 2
    n = len(items)
    bw = max(8, chart_w / (n * 1.6))
    step = chart_w / max(n, 1)
    bars = []
    labels = []
    for i, (name, val) in enumerate(items):
        x = margin + i * step + (step - bw) / 2
        norm = (val - vmin) / (vmax - vmin)
        h = norm * (chart_h - 20)
        y = margin + chart_h - h
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="#4F81BD" />')
        labels.append(f'<text x="{x + bw/2:.1f}" y="{margin + chart_h + 14:.1f}" font-size="10" text-anchor="middle">{html.escape(name[:12])}</text>')
    return (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
        f'<text x="{margin}" y="20" font-size="16">{html.escape(title)}</text>'
        + "".join(bars)
        + "".join(labels)
        + f'<line x1="{margin}" y1="{margin+chart_h}" x2="{margin+chart_w}" y2="{margin+chart_h}" stroke="#444" />'
        + "</svg>"
    )


def _line_svg(df: pd.DataFrame, metric: str, top_ids: List[str], width: int = 1100, height: int = 360) -> str:
    if df.empty or metric not in df.columns:
        return "<p>暂无窗口级趋势数据</p>"
    picked = df[df["experiment_id"].isin(top_ids)].copy()
    if picked.empty:
        return "<p>暂无窗口级趋势数据</p>"
    picked["window"] = pd.to_numeric(picked["window"], errors="coerce")
    picked[metric] = pd.to_numeric(picked[metric], errors="coerce")
    picked = picked.dropna(subset=["window", metric])
    if picked.empty:
        return "<p>暂无窗口级趋势数据</p>"
    margin = 45
    chart_w = width - margin * 2
    chart_h = height - margin * 2
    wmin, wmax = float(picked["window"].min()), float(picked["window"].max())
    vmin, vmax = float(picked[metric].min()), float(picked[metric].max())
    if wmax == wmin:
        wmax = wmin + 1
    if vmax == vmin:
        vmax = vmin + 1e-9
    colors = ["#4F81BD", "#C0504D", "#9BBB59", "#8064A2", "#4BACC6", "#F79646"]
    lines = []
    legends = []
    for i, exp_id in enumerate(top_ids):
        sub = picked[picked["experiment_id"] == exp_id].sort_values("window")
        if sub.empty:
            continue
        points = []
        for _, r in sub.iterrows():
            x = margin + (float(r["window"]) - wmin) / (wmax - wmin) * chart_w
            y = margin + chart_h - (float(r[metric]) - vmin) / (vmax - vmin) * chart_h
            points.append(f"{x:.1f},{y:.1f}")
        color = colors[i % len(colors)]
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(points)}" />')
        legends.append(
            f'<text x="{margin + 10}" y="{margin + 16 + i * 16}" font-size="11" fill="{color}">{html.escape(exp_id[:36])}</text>'
        )
    return (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
        f'<text x="{margin}" y="20" font-size="16">窗口趋势: {html.escape(metric)}</text>'
        + "".join(lines)
        + "".join(legends)
        + f'<line x1="{margin}" y1="{margin+chart_h}" x2="{margin+chart_w}" y2="{margin+chart_h}" stroke="#444" />'
        + f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{margin+chart_h}" stroke="#444" />'
        + "</svg>"
    )


def main() -> None:
    args = parse_args()
    summary_csv = (PROJECT_ROOT / args.summary_csv).resolve()
    window_csv = (PROJECT_ROOT / args.window_csv).resolve()
    out_path = (PROJECT_ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        summary_df = pd.read_csv(summary_csv) if summary_csv.exists() else pd.DataFrame()
    except EmptyDataError:
        summary_df = pd.DataFrame()
    try:
        window_df = pd.read_csv(window_csv) if window_csv.exists() else pd.DataFrame()
    except EmptyDataError:
        window_df = pd.DataFrame()
    score_col = "ic_qlib_ensemble_icir" if "ic_qlib_ensemble_icir" in summary_df.columns else "ic_gru_icir"

    if not summary_df.empty and score_col in summary_df.columns:
        summary_df = summary_df.sort_values(by=score_col, ascending=False, na_position="last")
    top_df = summary_df.head(args.topn).copy() if not summary_df.empty else pd.DataFrame()

    bar_items = []
    if not top_df.empty and score_col in top_df.columns:
        for _, r in top_df.iterrows():
            v = pd.to_numeric(pd.Series([r[score_col]]), errors="coerce").iloc[0]
            if pd.notna(v):
                bar_items.append((str(r["experiment_id"]), float(v)))
    top_ids = [str(x) for x in top_df.get("experiment_id", pd.Series([], dtype=str)).tolist()[:6]]

    bar_svg = _bar_svg(bar_items, f"实验排行榜（{score_col}）")
    line_svg = _line_svg(window_df, "ic_qlib_ensemble", top_ids)

    table_html = (
        top_df.to_html(index=False, classes="summary-table", border=0, float_format=lambda x: f"{x:.6f}")
        if not top_df.empty
        else "<p>暂无实验结果</p>"
    )

    filters = ""
    if not summary_df.empty and "model_type" in summary_df.columns:
        options = sorted(set(str(x) for x in summary_df["model_type"].dropna().tolist()))
        filters = '<label>模型筛选: <select id="modelFilter"><option value="all">all</option>' + "".join(
            [f'<option value="{html.escape(o)}">{html.escape(o)}</option>' for o in options]
        ) + "</select></label>"

    html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<title>Tuning Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; color: #222; }}
h1, h2 {{ margin: 8px 0; }}
.summary-table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
.summary-table th, .summary-table td {{ border: 1px solid #ddd; padding: 6px 8px; font-size: 12px; }}
.summary-table th {{ background: #f3f3f3; }}
.block {{ margin: 16px 0 28px 0; }}
</style>
</head>
<body>
<h1>微调训练结果对比报告</h1>
<p>报告文件: {html.escape(str(out_path))}</p>
<div class="block">{filters}</div>
<div class="block">{bar_svg}</div>
<div class="block">{line_svg}</div>
<h2>实验汇总 TopN</h2>
<div class="block" id="summaryTableWrap">{table_html}</div>
<script>
const filter = document.getElementById('modelFilter');
if (filter) {{
  filter.addEventListener('change', () => {{
    const v = filter.value;
    document.querySelectorAll('.summary-table tbody tr').forEach(tr => {{
      const tds = tr.querySelectorAll('td');
      if (tds.length < 2) return;
      const modelType = tds[1].innerText.trim();
      tr.style.display = (v === 'all' || modelType === v) ? '' : 'none';
    }});
  }});
}}
</script>
</body>
</html>
"""
    out_path.write_text(html_doc, encoding="utf-8")
    print(f"report: {out_path}")


if __name__ == "__main__":
    main()
