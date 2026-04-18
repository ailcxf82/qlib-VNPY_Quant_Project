"""
生成静态 Pipeline Dashboard：接入工程内真实产物摘要（P0-4-b）。

数据来源汇总：docs/DASHBOARD_DATA_SOURCES.md

用法（项目根目录；推荐 conda 环境 qlib_zhengshi，与 run_fin_quant / 训练一致）：

  conda activate qlib_zhengshi
  python scripts/render_pipeline_dashboard.py
  python scripts/render_pipeline_dashboard.py --metrics-pool csi500 --write-regime
  python scripts/render_pipeline_dashboard.py --out data/dashboard --focus train --keep-legacy 30

  # 或：
  conda run -n qlib_zhengshi python scripts/render_pipeline_dashboard.py --write-regime
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from dashboard_collect import CollectResult, collect  # noqa: E402

# 与设计文档第十一节一致：6 盏灯 + 6 张卡（市场单独大卡）
STAGE_LIGHTS = [
    ("regime", "① 市场", "stage_regime.html"),
    ("rdagent", "② RDAgent", "stage_rdagent.html"),
    ("factor_pool", "③ 因子池", "stage_factor_pool.html"),
    ("train", "④ 训练", "stage_train.html"),
    ("ensemble", "⑤ 集成", "stage_ensemble.html"),
    ("backtest", "⑥ 回测", "stage_backtest.html"),
]

CARDS: list[tuple[str, str, str, str, str]] = [
    (
        "rdagent",
        "RDAgent",
        "stage_rdagent.html",
        "git_ignore_folder/RD-Agent_workspace/",
        "RD-Agent 循环与 result.h5",
    ),
    (
        "factor_pool",
        "因子池",
        "stage_factor_pool.html",
        "git_ignore_folder/combined_factors_df.*",
        "export_rdagent_factors 导出",
    ),
    (
        "train",
        "训练 & OOF",
        "stage_train.html",
        "data/logs/<pool>_logs/training_metrics.csv",
        "滚动窗口与各基模型 IC",
    ),
    (
        "ensemble",
        "集成 & Meta",
        "stage_ensemble.html",
        "data/oof/ · data/meta/",
        "OOF 与 meta 权重",
    ),
    (
        "backtest",
        "回测",
        "stage_backtest.html",
        "data/backtest/rqalpha/*/detailed_results.json",
        "RQAlpha 汇总",
    ),
    (
        "audit",
        "审计 / 路线图",
        "stage_audit.html",
        "docs/OPTIMIZATION_ROADMAP_2026.md",
        "路线图与 Phase",
    ),
]

STAGE_PAGES = [
    ("stage_regime.html", "① 市场状态"),
    ("stage_rdagent.html", "② RD-Agent"),
    ("stage_factor_pool.html", "③ 因子池"),
    ("stage_train.html", "④ 训练 & OOF"),
    ("stage_ensemble.html", "⑤ 集成 & Meta"),
    ("stage_backtest.html", "⑥ 回测 & 组合"),
    ("stage_audit.html", "审计 / 路线图"),
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fmt_local() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _default_regime_placeholder() -> dict:
    return {
        "_note": "占位。运行本脚本且加 --write-regime 会合并 config/data.yaml。",
        "universe": [],
        "window": None,
        "regime_label": "pending",
        "regime_label_cn": "暂无数据",
        "signals": {},
        "objective": {"primary": "rank_ic_ensemble", "secondary": "turnover_adj_sharpe"},
        "updated_at": None,
    }


def _ensure_regime_json(out: Path) -> None:
    path = out / "regime_snapshot.json"
    if path.exists():
        return
    path.write_text(
        json.dumps(_default_regime_placeholder(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _ensure_events_log(out: Path) -> None:
    path = out / "events.log"
    if not path.exists():
        line = f"{_utc_now_iso()} [dashboard] INFO dashboard events log initialized\n"
        path.write_text(line, encoding="utf-8")


def _tail_events(out: Path, max_lines: int = 18) -> str:
    path = out / "events.log"
    if not path.exists():
        return "（无 events.log）"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as e:
        return f"（无法读取 events.log: {e}）"
    if not lines:
        return "（events.log 为空）"
    return "\n".join(lines[-max_lines:])


def _dot_class_for_stage(stage_key: str, focus: str, status: str) -> str:
    if stage_key == focus:
        return "status-target"
    if status == "ready":
        return "status-ready"
    if status == "running":
        return "status-running"
    if status == "stale":
        return "status-stale"
    if status == "failed":
        return "status-failed"
    return "status-missing"


def _dot_symbol(status: str, is_target: bool) -> str:
    if is_target:
        return "●"
    if status == "ready":
        return "✓"
    if status == "running":
        return "◐"
    if status == "stale":
        return "⏳"
    if status == "failed":
        return "✗"
    return "○"


def _fmt_ic_means(means: dict | None) -> str:
    if not means:
        return "—"
    parts = []
    for k in sorted(means.keys()):
        parts.append(f"{k.replace('ic_', '')}={means[k]:.4f}")
    return ", ".join(parts[:6])


def _card_body(key: str, c: CollectResult) -> tuple[str, str, str]:
    """返回 (meta_line, inner_html, data_source_short)。"""
    if key == "rdagent":
        r = c.rdagent
        if not r:
            return (
                "数据来源: git_ignore_folder/RD-Agent_workspace · mtime: —",
                "<strong>暂无</strong>：未找到工作区目录或尚无 result.h5。",
                "RD-Agent_workspace",
            )
        meta = f"数据来源: RD-Agent_workspace · 最新: {r.get('latest_result_mtime') or '—'}"
        inner = (
            f"含 result.h5 的工作区: <strong>{r.get('workspaces_with_result_h5', 0)}</strong><br/>"
            f"<span class='hint'>export_rdagent_factors.py 扫描此目录合并因子。</span>"
        )
        return meta, inner, "RD-Agent_workspace/*/result.h5"

    if key == "factor_pool":
        f = c.factor_pool
        if not f:
            return (
                "数据来源: git_ignore_folder/combined_factors_df.* · mtime: —",
                "<strong>暂无</strong>：未运行 export 或文件不存在。",
                "combined_factors_df",
            )
        meta = f"数据来源: combined_factors_df.json · mtime: {f.get('json_mtime') or '—'}"
        mic = f.get("mean_abs_ic")
        mic_s = f"{float(mic):.4f}" if mic is not None else "—"
        inner = (
            f"因子数: <strong>{f.get('n_factors', '—')}</strong> · "
            f"平均|IC|: <strong>{mic_s}</strong>（若可算）<br/>"
            f"导出时间: {html.escape(str(f.get('created_at', '—')))}"
        )
        return meta, inner, "scripts/export_rdagent_factors.py 产出"

    if key == "train":
        t = c.training
        if not t:
            return (
                "数据来源: data/logs/*_logs/training_metrics.csv · mtime: —",
                "<strong>暂无</strong>：未找到 training_metrics.csv。",
                "training_metrics.csv",
            )
        meta = f"数据来源: {html.escape(t.get('metrics_csv_resolved', t.get('path', '')))} · mtime: {t.get('mtime') or '—'}"
        inner = (
            f"滚动窗数: <strong>{t.get('windows', 0)}</strong> · 最近 valid_end: {html.escape(str(t.get('valid_end', '—')))}<br/>"
            f"近20窗 IC 均值: {_fmt_ic_means(t.get('ic_mean_last20'))}<br/>"
            f"最新窗 IC: {_fmt_ic_means(t.get('ic_last_window'))}"
        )
        return meta, inner, "trainer 写入 training_metrics.csv"

    if key == "ensemble":
        e = c.ensemble
        if not e:
            return (
                "数据来源: data/oof/ · data/meta/ · mtime: —",
                "<strong>暂无</strong>：未扫描到 OOF 目录或 meta 文件。",
                "data/oof",
            )
        meta = f"数据来源: data/oof · 最新 OOF 活动: {e.get('oof_latest_mtime') or '—'}"
        tags = e.get("oof_tags_sample") or []
        inner = (
            f"OOF 标签数: <strong>{e.get('oof_tag_count', 0)}</strong> · 样例: {html.escape(', '.join(tags[:5]))}<br/>"
            f"meta 最新文件: {html.escape(str(e.get('meta_latest', '—')))}"
        )
        return meta, inner, "trainer / meta_stacker 产物"

    if key == "backtest":
        b = c.backtest
        if not b:
            return (
                "数据来源: data/backtest/rqalpha/*/detailed_results.json · mtime: —",
                "<strong>暂无</strong>：未找到回测 JSON。",
                "detailed_results.json",
            )
        meta = f"数据来源: {html.escape(b.get('path', ''))} · pool={b.get('pool')} · mtime: {b.get('mtime')}"
        ar = b.get("annualized_returns")
        mdd = b.get("max_drawdown")
        att = b.get("avg_daily_turnover")
        ar_s = f"{float(ar):.2%}" if ar is not None else "—"
        mdd_s = f"{float(mdd):.2%}" if mdd is not None else "—"
        att_s = f"{float(att):.4f}" if att is not None else "—"
        inner = (
            f"区间: {html.escape(str(b.get('start_date')))} ~ {html.escape(str(b.get('end_date')))}<br/>"
            f"年化: <strong>{ar_s}</strong> · 最大回撤: <strong>{mdd_s}</strong> · "
            f"日均换手: <strong>{att_s}</strong>（RQAlpha 字段）<br/>"
            f"Sharpe: {b.get('sharpe')}"
        )
        return meta, inner, "run_backtest → RQAlpha 写 JSON"

    if key == "audit":
        a = c.audit
        if not a:
            return (
                "数据来源: docs/OPTIMIZATION_ROADMAP_2026.md · mtime: —",
                "<strong>暂无</strong>：未找到路线图文件。",
                "OPTIMIZATION_ROADMAP_2026.md",
            )
        meta = f"数据来源: {html.escape(a.get('path', ''))} · mtime: {a.get('mtime')}"
        inner = "打开仓库内路线图查看 Phase 与 P0~P2 任务；本卡仅展示文件更新时间。"
        return meta, inner, "人工维护文档"

    return "—", "—", "—"


def render_index(out: Path, focus: str, generated_at: str, c: CollectResult) -> None:
    regime = c.regime_for_display
    cn = html.escape(str(regime.get("regime_label_cn", "暂无数据")))
    uni = regime.get("universe") or []
    win = regime.get("window")
    note = html.escape(str(regime.get("_display_note", "")))
    uni_s = html.escape(", ".join(uni) if uni else "（未配置）")
    if isinstance(win, (list, tuple)) and len(win) >= 2:
        win_s = html.escape(f"{win[0]} ~ {win[1]}")
    else:
        win_s = "—"
    updated = regime.get("updated_at")
    updated_s = html.escape(str(updated) if updated else "—")
    cfg = regime.get("config_alignment") or {}
    cfg_src = html.escape(str(cfg.get("source_file", "config/data.yaml")))

    lights_html = []
    for key, name, href in STAGE_LIGHTS:
        st_info = c.stages.get(key, {})
        status = st_info.get("status", "missing")
        mtime = st_info.get("mtime") or "—"
        is_focus = key == focus
        cls = _dot_class_for_stage(key, focus, status)
        sym = _dot_symbol(status, is_focus)
        lights_html.append(
            f'<div class="stage-dot {cls}">'
            f'<a href="{html.escape(href)}" title="{html.escape(name)}">'
            f'<div class="dot">{sym}</div></a>'
            f'<div class="name">{html.escape(name)}</div>'
            f'<div class="mtime">{html.escape(str(mtime))}</div></div>'
        )

    cards_html = []
    for key, title, href, _src_hint, _desc in CARDS:
        meta, inner, _ds = _card_body(key, c)
        cards_html.append(
            f'<a class="card" href="{html.escape(href)}">'
            f"<h3>{html.escape(title)}</h3>"
            f'<div class="card-meta">{html.escape(meta)}</div>'
            f'<div class="placeholder data-fill">{inner}</div></a>'
        )

    lineage_rows = []
    for row in c.lineage:
        lineage_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('环节', '')))}</td>"
            f"<td><code>{html.escape(str(row.get('路径', '')))}</code></td>"
            f"<td>{html.escape(str(row.get('说明', '')))}</td>"
            "</tr>"
        )
    lineage_table = (
        "<table class='lineage-table'><thead><tr><th>环节</th><th>路径</th><th>说明</th></tr></thead><tbody>"
        + "".join(lineage_rows)
        + "</tbody></table>"
    )

    events_body = html.escape(_tail_events(out))

    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Qlib + RD-Agent 流水线 · Dashboard</title>
  <link rel="stylesheet" href="assets/style.css"/>
</head>
<body>
  <header class="topbar">
    <h1>Qlib + RD-Agent 量化流水线 · Dashboard</h1>
    <div class="topbar-meta">
      <span>Last update: {html.escape(generated_at)}</span>
      <button type="button" class="btn-refresh" onclick="location.reload()">刷新</button>
    </div>
  </header>
  <div class="wrap">
    <section class="panel">
      <div class="panel-title">当前市场状态（regime_snapshot.json + config/data.yaml 对齐）</div>
      <div class="regime-main">{cn}</div>
      <p class="regime-note">{note}</p>
      <div class="regime-grid">
        <div class="regime-item"><span class="label">股票池</span>{uni_s}</div>
        <div class="regime-item"><span class="label">数据区间</span>{win_s}</div>
        <div class="regime-item"><span class="label">快照时间</span>{updated_s}</div>
        <div class="regime-item"><span class="label">配置来源</span>{cfg_src}</div>
      </div>
    </section>

    <section class="panel">
      <div class="panel-title">Pipeline 阶段（点击圆点进入子页）</div>
      <div class="pipeline-row">
        {"".join(lights_html)}
      </div>
    </section>

    <section class="panel">
      <div class="panel-title">状态卡（点击跳转详情）</div>
      <div class="cards-grid">
        {"".join(cards_html)}
      </div>
    </section>

    <section class="panel">
      <div class="panel-title">数据血缘（本页数字从哪来）</div>
      {lineage_table}
    </section>

    <section class="panel">
      <div class="panel-title">最近事件（events.log 末尾）</div>
      <div class="events-box">{events_body}</div>
    </section>
  </div>
</body>
</html>
"""
    (out / "index.html").write_text(html_doc, encoding="utf-8")


def _stage_detail_html(filename: str, title: str, generated_at: str, c: CollectResult) -> str:
    blocks: list[str] = []
    if filename == "stage_regime.html":
        blocks.append(f"<pre>{html.escape(json.dumps(c.regime_for_display, ensure_ascii=False, indent=2))}</pre>")
    elif filename == "stage_rdagent.html":
        blocks.append(
            f"<pre>{html.escape(json.dumps(c.rdagent or {{}}, ensure_ascii=False, indent=2))}</pre>"
        )
    elif filename == "stage_factor_pool.html":
        blocks.append(f"<pre>{html.escape(json.dumps(c.factor_pool or {{}}, ensure_ascii=False, indent=2))}</pre>")
    elif filename == "stage_train.html":
        blocks.append(f"<pre>{html.escape(json.dumps(c.training or {{}}, ensure_ascii=False, indent=2))}</pre>")
    elif filename == "stage_ensemble.html":
        blocks.append(f"<pre>{html.escape(json.dumps(c.ensemble or {{}}, ensure_ascii=False, indent=2))}</pre>")
    elif filename == "stage_backtest.html":
        blocks.append(f"<pre>{html.escape(json.dumps(c.backtest or {{}}, ensure_ascii=False, indent=2))}</pre>")
    elif filename == "stage_audit.html":
        blocks.append(f"<pre>{html.escape(json.dumps(c.audit or {{}}, ensure_ascii=False, indent=2))}</pre>")
    inner = blocks[0] if blocks else "<p>无数据</p>"
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{html.escape(title)} · Dashboard</title>
  <link rel="stylesheet" href="assets/style.css"/>
</head>
<body>
  <div class="wrap subpage">
    <p class="page-back"><a href="index.html">← 返回首页</a></p>
    <h2>{html.escape(title)}</h2>
    <p class="card-meta">生成时间: {html.escape(generated_at)} · 原始 JSON 摘要如下</p>
    <div class="placeholder-block json-block">
{inner}
    </div>
  </div>
</body>
</html>
"""


def render_stage_pages(out: Path, generated_at: str, c: CollectResult) -> None:
    for filename, title in STAGE_PAGES:
        body = _stage_detail_html(filename, title, generated_at, c)
        (out / filename).write_text(body, encoding="utf-8")


def write_status_snapshot(out: Path, focus: str, c: CollectResult) -> None:
    snap = {
        "version": 1,
        "generated_at": _utc_now_iso(),
        "generated_by": "scripts/render_pipeline_dashboard.py",
        "current_focus": focus,
        "regime": {"see_regime_snapshot_json": True},
        "stages": c.stages,
        "lineage": c.lineage,
        "events_tail_path": "events.log",
    }
    (out / "status_snapshot.json").write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")


def append_event(out: Path, line: str) -> None:
    path = out / "events.log"
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(line.rstrip() + "\n")
    except OSError:
        pass


def prune_legacy(out: Path, keep: int) -> None:
    leg = out / "legacy"
    if keep <= 0 or not leg.is_dir():
        return
    files = sorted(leg.glob("snapshot_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in files[keep:]:
        try:
            p.unlink()
        except OSError:
            pass


def copy_legacy(out: Path, keep: int) -> None:
    if keep <= 0:
        return
    leg = out / "legacy"
    leg.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = leg / f"snapshot_{ts}.json"
    src = out / "status_snapshot.json"
    if src.exists():
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    prune_legacy(out, keep)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render static pipeline dashboard with data summaries")
    ap.add_argument("--out", default="data/dashboard", help="Output directory under project root")
    ap.add_argument(
        "--focus",
        default="regime",
        choices=[s[0] for s in STAGE_LIGHTS],
        help="Highlight pipeline stage (target dot)",
    )
    ap.add_argument("--keep-legacy", type=int, default=0, help="Keep last N legacy snapshots under out/legacy/")
    ap.add_argument(
        "--metrics-pool",
        default="csi500",
        help="Prefer data/logs/<pool>_logs/training_metrics.csv (default: csi500)",
    )
    ap.add_argument(
        "--write-regime",
        action="store_true",
        help="Write merged regime_snapshot.json (config/data.yaml + existing file)",
    )
    args = ap.parse_args()

    out = (ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "assets").mkdir(parents=True, exist_ok=True)

    _ensure_regime_json(out)
    _ensure_events_log(out)

    c = collect(ROOT, out, metrics_pool=args.metrics_pool, write_regime_merged=bool(args.write_regime))
    if c.regime_to_write is not None:
        (out / "regime_snapshot.json").write_text(
            json.dumps(c.regime_to_write, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    generated_local = _fmt_local()
    write_status_snapshot(out, args.focus, c)
    copy_legacy(out, args.keep_legacy)
    render_index(out, args.focus, generated_local, c)
    render_stage_pages(out, generated_local, c)

    summ = []
    if c.training:
        summ.append(f"windows={c.training.get('windows')}")
    if c.factor_pool:
        summ.append(f"factors={c.factor_pool.get('n_factors')}")
    if c.backtest:
        summ.append(f"bt_pool={c.backtest.get('pool')}")
    append_event(
        out,
        f"{_utc_now_iso()} [dashboard] INFO render ok " + (" ".join(summ) if summ else "minimal"),
    )

    render_log = out / "render.log"
    try:
        render_log.write_text(
            f"{_utc_now_iso()} OK out={out} focus={args.focus} metrics_pool={args.metrics_pool}\n",
            encoding="utf-8",
        )
    except OSError:
        pass

    print(f"[dashboard] wrote: {out / 'index.html'}")
    print(f"[dashboard] data: training={bool(c.training)} factors={bool(c.factor_pool)} backtest={bool(c.backtest)}")
    print(f"[dashboard] open file:///{str(out / 'index.html').replace(chr(92), '/')}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
