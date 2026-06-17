"""
为 Pipeline Dashboard 从工程内已有产物收集摘要（无网络、不启动训练）。

数据来源见 docs/DASHBOARD_DATA_SOURCES.md。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class CollectResult:
    """渲染器消费的聚合结果。"""

    regime_for_display: dict[str, Any]
    regime_to_write: dict[str, Any] | None  # None 表示不写盘
    training: dict[str, Any] | None
    factor_pool: dict[str, Any] | None
    backtest: dict[str, Any] | None
    rdagent: dict[str, Any] | None
    ensemble: dict[str, Any] | None
    audit: dict[str, Any] | None
    lineage: list[dict[str, Any]] = field(default_factory=list)
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)


def _mtime_iso(p: Path) -> str | None:
    try:
        return datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    except OSError:
        return None


def _fallback_parse_data_yaml(root: Path) -> dict[str, Any] | None:
    """不依赖 PyYAML：仅解析 data 段常用字段（instruments / start_time / end_time）。"""
    p = root / "config" / "data.yaml"
    if not p.exists():
        return None
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return None
    # 只扫文件前部，避免读完巨大 feature_sets
    head = "\n".join(text.splitlines()[:80])
    out: dict[str, Any] = {}
    m = re.search(r"(?m)^\s*instruments:\s*(.+?)\s*$", head)
    if m:
        out["instruments"] = m.group(1).strip().strip("'\"")
    m = re.search(r"(?m)^\s*start_time:\s*(.+?)\s*$", head)
    if m:
        out["start_time"] = m.group(1).strip().strip("'\"")
    m = re.search(r"(?m)^\s*end_time:\s*(.+?)\s*$", head)
    if m:
        out["end_time"] = m.group(1).strip().strip("'\"")
    return out if out else None


def _safe_read_yaml_data_block(root: Path) -> dict[str, Any] | None:
    p = root / "config" / "data.yaml"
    if not p.exists():
        return None
    try:
        import yaml  # type: ignore
    except ImportError:
        return _fallback_parse_data_yaml(root)
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except Exception:
        return _fallback_parse_data_yaml(root)
    data = raw.get("data") if isinstance(raw, dict) else None
    if isinstance(data, dict) and data:
        return data
    return _fallback_parse_data_yaml(root)


def _merge_regime_snapshot(
    root: Path,
    dashboard_dir: Path,
    write_merged: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """合并 config/data.yaml 口径到 regime 展示；可选写回 regime_snapshot.json。"""
    path = dashboard_dir / "regime_snapshot.json"
    base: dict[str, Any] = {}
    if path.exists():
        try:
            base = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            base = {}

    data_block = _safe_read_yaml_data_block(root)
    cfg_inst = None
    cfg_start = None
    cfg_end = None
    if isinstance(data_block, dict):
        cfg_inst = data_block.get("instruments")
        cfg_start = data_block.get("start_time")
        cfg_end = data_block.get("end_time")

    display = {**base}
    display["config_alignment"] = {
        "instruments": cfg_inst,
        "start_time": str(cfg_start) if cfg_start is not None else None,
        "end_time": str(cfg_end) if cfg_end is not None else None,
        "source_file": "config/data.yaml",
    }
    if cfg_inst:
        if isinstance(cfg_inst, str):
            display["universe"] = [x.strip() for x in cfg_inst.split(",") if x.strip()]
        elif isinstance(cfg_inst, list):
            display["universe"] = [str(x) for x in cfg_inst]
    if cfg_start and cfg_end:
        display["window"] = [str(cfg_start).strip("'\""), str(cfg_end).strip("'\"")]

    if not display.get("regime_label_cn") or display.get("regime_label_cn") == "暂无数据":
        display["regime_label_cn"] = "配置口径（data.yaml，非行情 regime）"
        display["regime_label"] = "from_config"

    display["updated_at"] = display.get("updated_at") or datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    display["_display_note"] = "窗口/股票池来自 config/data.yaml；真实市场状态需 P0-2 sync_regime_snapshot + qlib。"

    to_write = json.loads(json.dumps(display, ensure_ascii=False)) if write_merged else None
    return display, to_write


def _pick_training_metrics_csv(root: Path, pool: str) -> Path | None:
    candidates = [
        root / "data" / "logs" / f"{pool}_logs" / "training_metrics.csv",
        root / "data" / "logs" / "training_metrics.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    # 任意 *_logs
    logs = root / "data" / "logs"
    if not logs.is_dir():
        return None
    found: list[Path] = []
    for d in logs.iterdir():
        if d.is_dir() and d.name.endswith("_logs"):
            t = d / "training_metrics.csv"
            if t.exists():
                found.append(t)
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def _read_training_summary(path: Path) -> dict[str, Any] | None:
    try:
        import pandas as pd
    except ImportError:
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return {"path": str(path), "windows": 0, "mtime": _mtime_iso(path)}
    n = len(df)
    ic_cols = [c for c in df.columns if c.startswith("ic_")]
    tail = df.tail(min(20, n))
    means: dict[str, float] = {}
    for c in ic_cols:
        ser = pd.to_numeric(tail[c], errors="coerce")
        if ser.notna().any():
            means[c] = float(ser.mean())
    last = df.iloc[-1]
    last_ic = {c: float(last[c]) for c in ic_cols if c in last and pd.notna(last.get(c))}
    return {
        "path": str(path),
        "mtime": _mtime_iso(path),
        "windows": n,
        "last_window": int(last.get("window", -1)) if "window" in last else None,
        "valid_end": str(last.get("valid_end", "")) if "valid_end" in last else None,
        "ic_mean_last20": means,
        "ic_last_window": last_ic,
    }


def _read_factor_pool(root: Path) -> dict[str, Any] | None:
    jpath = root / "git_ignore_folder" / "combined_factors_df.json"
    ppath = root / "git_ignore_folder" / "combined_factors_df.parquet"
    if not jpath.exists() and not ppath.exists():
        return None
    out: dict[str, Any] = {"json_path": str(jpath) if jpath.exists() else None, "parquet_path": str(ppath) if ppath.exists() else None}
    if jpath.exists():
        out["json_mtime"] = _mtime_iso(jpath)
        try:
            meta = json.loads(jpath.read_text(encoding="utf-8"))
            out["n_factors"] = meta.get("n_factors")
            out["created_at"] = meta.get("created_at")
            facs = meta.get("factors") or []
            if facs:
                ics = [abs(float(x.get("ic", 0))) for x in facs if x.get("ic") is not None]
                out["mean_abs_ic"] = sum(ics) / len(ics) if ics else None
        except Exception:
            out["parse_error"] = True
    if ppath.exists():
        out["parquet_mtime"] = _mtime_iso(ppath)
    out["source_script"] = "scripts/export_rdagent_factors.py"
    return out


def _read_backtest_latest(root: Path) -> dict[str, Any] | None:
    base = root / "data" / "backtest" / "rqalpha"
    if not base.is_dir():
        return None
    best: tuple[float, Path, str] | None = None
    for pool_dir in base.iterdir():
        if not pool_dir.is_dir():
            continue
        js = pool_dir / "detailed_results.json"
        if not js.exists():
            continue
        try:
            m = js.stat().st_mtime
        except OSError:
            continue
        if best is None or m > best[0]:
            best = (m, js, pool_dir.name)
    if best is None:
        return None
    _, js, pool = best
    try:
        data = json.loads(js.read_text(encoding="utf-8"))
    except Exception:
        return {"path": str(js), "pool": pool, "error": True}
    eff = data.get("效率指标") or {}
    pl = data.get("盈亏状态") or {}
    return {
        "path": str(js),
        "pool": pool,
        "mtime": _mtime_iso(js),
        "annualized_returns": eff.get("annualized_returns"),
        "max_drawdown": eff.get("max_drawdown"),
        "sharpe": eff.get("sharpe"),
        "avg_daily_turnover": eff.get("avg_daily_turnover"),
        "total_returns": eff.get("total_returns") or pl.get("总收益率"),
        "start_date": eff.get("start_date"),
        "end_date": eff.get("end_date"),
        "benchmark": eff.get("benchmark"),
    }


def _scan_rdagent_workspace(root: Path) -> dict[str, Any] | None:
    ws = root / "git_ignore_folder" / "RD-Agent_workspace"
    if not ws.is_dir():
        return None
    n_ok = 0
    latest: float = 0.0
    for d in ws.iterdir():
        if not d.is_dir():
            continue
        h5 = d / "result.h5"
        if h5.exists() and h5.stat().st_size > 1000:
            n_ok += 1
            latest = max(latest, h5.stat().st_mtime)
    return {
        "workspace_root": str(ws),
        "workspaces_with_result_h5": n_ok,
        "latest_result_mtime": datetime.fromtimestamp(latest).strftime("%Y-%m-%d %H:%M:%S") if latest else None,
        "source": "RD-Agent 写入的 result.h5（export 脚本扫此目录）",
    }


def _scan_ensemble(root: Path) -> dict[str, Any] | None:
    oof_root = root / "data" / "oof"
    meta_root = root / "data" / "meta"
    models_root = root / "data" / "models"
    parts: dict[str, Any] = {}

    if oof_root.is_dir():
        tags = [p.name for p in oof_root.iterdir() if p.is_dir()]
        tags.sort(reverse=True)
        parts["oof_tags_sample"] = tags[:8]
        parts["oof_tag_count"] = len(tags)
        if tags:
            newest = max((oof_root / t).stat().st_mtime for t in tags[:20])
            parts["oof_latest_mtime"] = datetime.fromtimestamp(newest).strftime("%Y-%m-%d %H:%M:%S")

    meta_files: list[Path] = []
    for folder, globs in ((meta_root, "*meta*.json"), (models_root, "*meta*.json")):
        if folder.is_dir():
            meta_files.extend(folder.glob(globs))
    if meta_files:
        latest_m = max(meta_files, key=lambda p: p.stat().st_mtime)
        parts["meta_latest"] = str(latest_m)
        parts["meta_latest_mtime"] = _mtime_iso(latest_m)
    parts["source_oof"] = "trainer 滚动训练写入 data/oof/{tag}/"
    parts["source_meta"] = "paths.meta_dir / data/models 下 *meta*.json"
    return parts if parts else None


def _audit_roadmap(root: Path) -> dict[str, Any] | None:
    p = root / "docs" / "OPTIMIZATION_ROADMAP_2026.md"
    if not p.exists():
        return None
    return {"path": str(p), "mtime": _mtime_iso(p)}


def _stage_status(
    regime: dict,
    training: dict | None,
    factor: dict | None,
    backtest: dict | None,
    rdagent: dict | None,
    ensemble: dict | None,
    audit: dict | None,
) -> dict[str, dict[str, Any]]:
    """粗粒度 ready / missing / stale。"""

    def st(has: bool, mtime: str | None) -> str:
        if not has:
            return "missing"
        return "ready"

    out: dict[str, dict[str, Any]] = {}
    out["regime"] = {
        "status": st(bool(regime.get("window")), regime.get("updated_at")),
        "mtime": regime.get("updated_at"),
        "summary": {"label": regime.get("regime_label_cn")},
    }
    out["rdagent"] = {
        "status": st(bool(rdagent and rdagent.get("workspaces_with_result_h5")), rdagent.get("latest_result_mtime") if rdagent else None),
        "mtime": rdagent.get("latest_result_mtime") if rdagent else None,
        "summary": {"result_h5_count": rdagent.get("workspaces_with_result_h5") if rdagent else 0},
    }
    out["factor_pool"] = {
        "status": st(bool(factor and factor.get("n_factors")), factor.get("json_mtime") if factor else None),
        "mtime": factor.get("json_mtime") if factor else None,
        "summary": {"n_factors": factor.get("n_factors") if factor else None},
    }
    out["train"] = {
        "status": st(bool(training and training.get("windows")), training.get("mtime") if training else None),
        "mtime": training.get("mtime") if training else None,
        "summary": {"windows": training.get("windows") if training else 0},
    }
    out["ensemble"] = {
        "status": st(bool(ensemble and ensemble.get("oof_tag_count")), ensemble.get("oof_latest_mtime") if ensemble else None),
        "mtime": ensemble.get("meta_latest_mtime") if ensemble else None,
        "summary": {"oof_tags": ensemble.get("oof_tag_count") if ensemble else 0},
    }
    out["backtest"] = {
        "status": st(bool(backtest and backtest.get("annualized_returns") is not None), backtest.get("mtime") if backtest else None),
        "mtime": backtest.get("mtime") if backtest else None,
        "summary": {"pool": backtest.get("pool") if backtest else None},
    }
    out["audit"] = {
        "status": st(bool(audit), audit.get("mtime") if audit else None),
        "mtime": audit.get("mtime") if audit else None,
        "summary": {"roadmap": "OPTIMIZATION_ROADMAP_2026.md"},
    }
    return out


def collect(root: Path, dashboard_dir: Path, metrics_pool: str, write_regime_merged: bool) -> CollectResult:
    lineage: list[dict[str, Any]] = [
        {
            "环节": "市场大卡·窗口与股票池",
            "路径": "config/data.yaml → data.*",
            "说明": "训练/特征使用的 start_time、end_time、instruments；写入 dashboard 的 config_alignment。",
        },
        {
            "环节": "训练 IC",
            "路径": f"data/logs/{metrics_pool}_logs/training_metrics.csv（或自动选最新 *_logs）",
            "说明": "RollingTrainer 各窗验证集 Rank IC；末窗与近 20 窗均值。",
        },
        {
            "环节": "因子池",
            "路径": "git_ignore_folder/combined_factors_df.json / .parquet",
            "说明": "scripts/export_rdagent_factors.py 导出后的汇总与合并表。",
        },
        {
            "环节": "RD-Agent 工作区",
            "路径": "git_ignore_folder/RD-Agent_workspace/*/result.h5",
            "说明": "因子任务运行成功时落盘，供导出脚本扫描。",
        },
        {
            "环节": "集成 / OOF",
            "路径": "data/oof/{tag}/、data/meta/ 或 data/models/*meta*.json",
            "说明": "OOF 折预测与二层 meta 产物路径（以 pipeline paths 为准）。",
        },
        {
            "环节": "回测",
            "路径": "data/backtest/rqalpha/<pool>/detailed_results.json",
            "说明": "run_backtest / RQAlpha 汇总 JSON，取目录下最新修改时间的池子。",
        },
        {
            "环节": "路线图审计",
            "路径": "docs/OPTIMIZATION_ROADMAP_2026.md",
            "说明": "文档 mtime 供「审计」卡展示。",
        },
    ]

    disp, to_write = _merge_regime_snapshot(root, dashboard_dir, write_regime_merged)

    tm_path = _pick_training_metrics_csv(root, metrics_pool)
    training = _read_training_summary(tm_path) if tm_path else None
    if training:
        training["metrics_csv_resolved"] = str(tm_path)

    factor = _read_factor_pool(root)
    backtest = _read_backtest_latest(root)
    rdagent = _scan_rdagent_workspace(root)
    ensemble = _scan_ensemble(root)
    audit = _audit_roadmap(root)

    stages = _stage_status(disp, training, factor, backtest, rdagent, ensemble, audit)

    return CollectResult(
        regime_for_display=disp,
        regime_to_write=to_write,
        training=training,
        factor_pool=factor,
        backtest=backtest,
        rdagent=rdagent,
        ensemble=ensemble,
        audit=audit,
        lineage=lineage,
        stages=stages,
    )
