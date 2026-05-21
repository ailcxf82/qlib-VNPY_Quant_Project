"""
summarize_rdagent.py
====================
自动汇总 rdagent 运行结果，生成类似 combined_factors_df_expressions_vXX.md 的文档。

用法：
    python scripts/summarize_rdagent.py               # 输出到 git_ignore_folder/
    python scripts/summarize_rdagent.py --out report.md
    python scripts/summarize_rdagent.py --no-code      # 不输出因子代码
    python scripts/summarize_rdagent.py --prev-ver 18  # 指定对比的上一版本号

输出内容：
    1. 概况：当前版本、数据窗口、因子数量
    2. 版本变更摘要：新增 / 移除 / 重加入
    3. 因子性能表（按 IC 降序）
    4. 各因子详情 + 源代码
    5. 组合模型历史性能（从 debug-*.log 解析）
    6. 本次会话候选因子列表（workspace 扫描）
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── 路径常量 ─────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
_GIT_IGNORE = _ROOT / "git_ignore_folder"
_WS_ROOT = _GIT_IGNORE / "RD-Agent_workspace"
_MANIFEST = _ROOT / "factor_registry" / "data" / "manifest.json"
_JSON_PATH = _GIT_IGNORE / "combined_factors_df.json"
_LOG_DIR = _ROOT / "log"


# ═══════════════════════════════════════════════════════════════════════════
# 1. 解析 manifest.json ── 版本历史
# ═══════════════════════════════════════════════════════════════════════════

def _load_manifest() -> list[dict]:
    """返回 manifest 中 factors 列表，按 parquet_version 排序。"""
    if not _MANIFEST.exists():
        return []
    data = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    return sorted(data.get("factors", []), key=lambda x: x.get("parquet_version", 0))


def _group_by_version(factors: list[dict]) -> dict[int, list[dict]]:
    """将 factors 按 parquet_version 分组。"""
    groups: dict[int, list[dict]] = defaultdict(list)
    for f in factors:
        v = f.get("parquet_version", 0)
        groups[v].append(f)
    return dict(groups)


def _get_active_version(factors: list[dict]) -> int:
    """返回当前 active 因子的版本号（最大 parquet_version）。"""
    active = [f for f in factors if f.get("status") == "active"]
    if not active:
        return 0
    return max(f.get("parquet_version", 0) for f in active)


def _factor_names_in_version(groups: dict[int, list[dict]], ver: int) -> set[str]:
    """返回指定版本中所有因子名。"""
    facs = groups.get(ver, [])
    return {f.get("factor_name") or f.get("name", "") for f in facs}


def _compute_version_diff(
    groups: dict[int, list[dict]], cur_ver: int, prev_ver: Optional[int]
) -> tuple[set[str], set[str], set[str]]:
    """
    对比 cur_ver 与 prev_ver，返回 (added, removed, readded)。
    readded = 在 prev_ver 之前某版本出现过、prev_ver 没有、cur_ver 又出现了。
    """
    if prev_ver is None:
        return set(), set(), set()

    cur_names = _factor_names_in_version(groups, cur_ver)
    prev_names = _factor_names_in_version(groups, prev_ver)

    added_raw = cur_names - prev_names
    removed = prev_names - cur_names

    # 历史曾出现过
    all_historical: set[str] = set()
    for v, facs in groups.items():
        if v < cur_ver:
            for f in facs:
                all_historical.add(f.get("factor_name") or f.get("name", ""))

    readded = added_raw & (all_historical - prev_names)
    new_adds = added_raw - readded

    return new_adds, removed, readded


# ═══════════════════════════════════════════════════════════════════════════
# 2. 读取 combined_factors_df.json ── 当前因子性能
# ═══════════════════════════════════════════════════════════════════════════

def _load_current_factors() -> tuple[dict, list[dict]]:
    """返回 (metadata_dict, factors_list)。"""
    if not _JSON_PATH.exists():
        return {}, []
    data = json.loads(_JSON_PATH.read_text(encoding="utf-8"))
    factors = data.get("factors", [])
    meta = {k: v for k, v in data.items() if k != "factors"}
    return meta, factors


# ═══════════════════════════════════════════════════════════════════════════
# 3. 扫描 workspace，建立因子名 → factor.py 路径映射
# ═══════════════════════════════════════════════════════════════════════════

def _build_factor_source_map() -> dict[str, list[Path]]:
    """
    遍历所有 workspace 的 factor.py，
    建立 factor_name → [Path, ...] 的映射（可能有多个 ws 实现同一因子）。
    """
    mapping: dict[str, list[Path]] = defaultdict(list)
    if not _WS_ROOT.exists():
        return mapping
    for ws_dir in _WS_ROOT.iterdir():
        fp = ws_dir / "factor.py"
        if not fp.exists():
            continue
        try:
            lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        for line in lines:
            m = re.match(r"\s*def calculate_([A-Za-z0-9_]+)\s*\(", line)
            if m:
                factor_name = m.group(1)
                mapping[factor_name].append(fp)
    return mapping


def _pick_best_source(paths: list[Path]) -> Optional[Path]:
    """
    多个 ws 实现同一因子时，优先选修改时间最新的。
    """
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def _read_factor_code(factor_name: str, src_map: dict[str, list[Path]]) -> tuple[Optional[str], Optional[Path]]:
    """返回 (code_str, path)，读不到返回 (None, None)。"""
    paths = src_map.get(factor_name, [])
    best = _pick_best_source(paths)
    if best is None:
        return None, None
    try:
        return best.read_text(encoding="utf-8", errors="replace"), best
    except Exception:
        return None, None


# ═══════════════════════════════════════════════════════════════════════════
# 4. 解析 debug-*.log ── 组合模型历史性能
# ═══════════════════════════════════════════════════════════════════════════

_TS_RE = re.compile(r'"timestamp"\s*:\s*(\d+)')
# qrun 产出的行：含 'IC': np.float64(...) 'ICIR': np.float64(...)
_QRUN_IC_RE = re.compile(
    r"'IC':\s*np\.float64\(([\d.]+)\).*?'ICIR':\s*np\.float64\(([\d.]+)\)"
)
# read_exp_res.py 产出的行（P3a 旧格式）: [reward] composite=... IC_IR=...
_REWARD_RE_LEGACY = re.compile(
    r"\[reward\]\s+composite=([\d.nan]+)\s*\|.*?IC_IR=([\d.nan]+)"
)
# read_exp_res.py 产出的行（Phase-2 新格式）:
# [reward] enhanced=... | composite=... diversity_bonus=... | ... IC_IR=...
_REWARD_RE = re.compile(
    r"\[reward\]\s+enhanced=([\d.nan]+)\s*\|\s*composite=([\d.nan]+)\s+"
    r"diversity_bonus=([\d.nan]+).*?IC_IR=([\d.nan]+)"
)
# 从 qlib_res.csv 路径提取 workspace id
_WS_ID_RE = re.compile(r"RD-Agent_workspace[\\/]([a-f0-9]{32})[\\/]")


def _parse_debug_logs(max_logs: int = 10) -> list[dict]:
    """
    扫描所有 debug-*.log，提取组合模型性能条目。

    解析策略：
    - `wsl_direct_exec` 行含 bash_cmd，从中提取当前 workspace ID，存为 last_ws_id
    - `wsl_direct_returned` 行的 stdout_tail 有两种：
      a) qrun 输出：含 'IC': np.float64(...)  → 记录 ic/icir，按 workspace 暂存
      b) read_exp_res 输出：含 [reward] composite=... → 合并同 workspace 的 qrun 结果
    返回按时间戳排序的列表，每条含 {ts, dt_str, workspace, ic, icir, composite,
    enhanced, diversity_bonus}。
    """
    results: list[dict] = []
    pending: dict[str, dict] = {}  # ws_id → 暂存的 qrun 结果
    last_ws_id: str = ""            # 最近一次 wsl_direct_exec 中提取的 ws

    log_files = sorted(
        _ROOT.glob("debug-*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:max_logs]

    for log_path in log_files:
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        lines = text.splitlines()
        for line in lines:
            # ── 1. wsl_direct_exec → 记录当前 workspace ──────────────────
            if "wsl_direct_exec" in line:
                ws_m = _WS_ID_RE.search(line)
                if ws_m:
                    last_ws_id = ws_m.group(1)
                continue

            # ── 2. wsl_direct_returned → 解析结果 ────────────────────────
            if "wsl_direct_returned" not in line:
                continue

            ts_m = _TS_RE.search(line)
            ts = int(ts_m.group(1)) if ts_m else 0
            dt_str = (
                datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
                .astimezone()
                .strftime("%Y-%m-%d %H:%M")
                if ts
                else "unknown"
            )

            # 优先从 stdout_tail 中提取 ws_id（read_exp_res 行含 qlib_res.csv 路径）
            ws_m = _WS_ID_RE.search(line)
            ws_id = ws_m.group(1) if ws_m else last_ws_id

            # ── qrun 行（含 IC/ICIR 字典格式）─────────────────────────────
            ic_m = _QRUN_IC_RE.search(line)
            if ic_m:
                pending[ws_id] = {
                    "ts": ts,
                    "dt_str": dt_str,
                    "workspace": ws_id,
                    "ic": float(ic_m.group(1)),
                    "icir": float(ic_m.group(2)),
                    "composite": float("nan"),
                    "enhanced": float("nan"),
                    "diversity_bonus": float("nan"),
                    "source_log": log_path.name,
                }
                continue

            # ── read_exp_res 行（Phase-2 enhanced 或 P3a legacy composite）──
            reward_m = _REWARD_RE.search(line)
            reward_fmt = "enhanced"
            if not reward_m:
                reward_m = _REWARD_RE_LEGACY.search(line)
                reward_fmt = "legacy"
            if reward_m:
                if reward_fmt == "enhanced":
                    enh_str = reward_m.group(1)
                    comp_str = reward_m.group(2)
                    div_str = reward_m.group(3)
                    icir_str = reward_m.group(4)
                    enhanced = float(enh_str) if enh_str != "nan" else float("nan")
                    diversity_bonus = float(div_str) if div_str != "nan" else float("nan")
                else:
                    comp_str = reward_m.group(1)
                    icir_str = reward_m.group(2)
                    enhanced = float("nan")
                    diversity_bonus = float("nan")
                comp = float(comp_str) if comp_str != "nan" else float("nan")
                icir_reward = float(icir_str) if icir_str != "nan" else float("nan")

                # 尝试合并配对的 qrun 结果（先按 ws_id，再尝试 last_ws_id）
                entry = pending.pop(ws_id, None) or pending.pop(last_ws_id, None)
                if entry:
                    entry["composite"] = comp
                    entry["enhanced"] = enhanced
                    entry["diversity_bonus"] = diversity_bonus
                    if icir_reward == icir_reward:
                        entry["icir"] = icir_reward
                    results.append(entry)
                else:
                    results.append(
                        {
                            "ts": ts,
                            "dt_str": dt_str,
                            "workspace": ws_id,
                            "ic": float("nan"),
                            "icir": icir_reward,
                            "composite": comp,
                            "enhanced": enhanced,
                            "diversity_bonus": diversity_bonus,
                            "source_log": log_path.name,
                        }
                    )

    # 未配对的 pending 条目也收录（qrun 成功但没有 reward 行）
    for entry in pending.values():
        results.append(entry)

    results.sort(key=lambda x: x["ts"])
    return results


# ═══════════════════════════════════════════════════════════════════════════
# 5. 扫描当前 session 的候选因子（RD-Agent_workspace 中只有 factor.py 的 ws）
# ═══════════════════════════════════════════════════════════════════════════

def _scan_candidate_factors(since_ts: float = 0.0) -> list[dict]:
    """
    扫描 since_ts 之后创建的、含 factor.py 但不含 qlib_res.csv 的 workspace，
    视为候选因子。返回 [{name, ws_id, mtime_str}, ...]。
    """
    candidates: list[dict] = []
    if not _WS_ROOT.exists():
        return candidates

    for ws_dir in _WS_ROOT.iterdir():
        fp = ws_dir / "factor.py"
        qr = ws_dir / "qlib_res.csv"
        if not fp.exists():
            continue
        if fp.stat().st_mtime < since_ts:
            continue
        try:
            lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        names = [
            re.match(r"\s*def calculate_([A-Za-z0-9_]+)\s*\(", ln)
            for ln in lines
        ]
        factor_names = [m.group(1) for m in names if m]
        mtime = fp.stat().st_mtime
        mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        for fn in factor_names:
            candidates.append(
                {
                    "name": fn,
                    "ws_id": ws_dir.name,
                    "mtime_str": mtime_str,
                    "has_result": (ws_dir / "result.h5").exists(),
                }
            )

    candidates.sort(key=lambda x: x["mtime_str"])
    return candidates


# ═══════════════════════════════════════════════════════════════════════════
# 6. 扫描 log/ 目录，提取 session 信息
# ═══════════════════════════════════════════════════════════════════════════

def _get_latest_sessions(n: int = 3) -> list[dict]:
    """返回最近 n 个 session 目录的信息。"""
    if not _LOG_DIR.exists():
        return []
    sessions = sorted(
        [d for d in _LOG_DIR.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )[:n]
    result = []
    for s in sessions:
        loops = sorted([d for d in s.iterdir() if d.is_dir() and d.name.startswith("Loop_")])
        result.append(
            {
                "name": s.name,
                "mtime": s.stat().st_mtime,
                "n_loops": len(loops),
            }
        )
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 7. 渲染 Markdown
# ═══════════════════════════════════════════════════════════════════════════

def _fmt_float(v, precision: int = 6) -> str:
    if v is None:
        return "N/A"
    try:
        f = float(v)
        if f != f:
            return "N/A"
        sign = "+" if f >= 0 else ""
        return f"{sign}{f:.{precision}f}"
    except Exception:
        return str(v)


def _render_markdown(
    meta: dict,
    factors: list[dict],
    src_map: dict[str, list[Path]],
    manifest_factors: list[dict],
    groups: dict[int, list[dict]],
    cur_ver: int,
    prev_ver: Optional[int],
    perf_history: list[dict],
    candidate_factors: list[dict],
    include_code: bool = True,
) -> str:
    lines: list[str] = []

    # ── 标题 ──────────────────────────────────────────────────────────────
    n_factors = len(factors)
    created_at = meta.get("created_at", "unknown")
    start = meta.get("start", "?")
    end = meta.get("end", "?")
    instruments = meta.get("instruments", "?")
    mode = meta.get("mode", "?")
    shape = meta.get("shape", [])
    shape_str = f"[{shape[0]}, {shape[1]}]" if len(shape) == 2 else str(shape)

    lines.append(f"# combined_factors_df 完整因子清单（{n_factors} 个）—— v{cur_ver}")
    lines.append("")
    lines.append(f"> 生成时间: {created_at}")
    lines.append(
        f"> 时间窗: {start} ~ {end}; 池: {instruments}; mode: {mode}; shape: {shape_str}"
    )
    lines.append(f"> manifest 当前版本: **v{cur_ver}**; 对比版本: v{prev_ver if prev_ver else '—'}")
    lines.append("")

    # ── 组合模型最新性能摘要 ─────────────────────────────────────────────
    if perf_history:
        latest = perf_history[-1]
        best = max(perf_history, key=lambda x: x["icir"])
        lines.append("## 组合模型性能概况（最新会话）")
        lines.append("")
        lines.append(f"| 指标 | 值 |")
        lines.append(f"|---|---|")
        lines.append(f"| 最新运行时间 | {latest['dt_str']} |")
        lines.append(f"| 最新 IC | {latest['ic']:.4f} |")
        lines.append(f"| 最新 IC_IR | **{latest['icir']:.4f}** |")
        lines.append(f"| 最新 Composite | {latest['composite']:.4f} |")
        if latest.get("enhanced") == latest.get("enhanced"):
            lines.append(f"| 最新 Enhanced | **{latest['enhanced']:.4f}** |")
        if latest.get("diversity_bonus") == latest.get("diversity_bonus"):
            lines.append(f"| 最新 Diversity Bonus | {latest['diversity_bonus']:.4f} |")
        lines.append(f"| 本轮最佳 IC_IR | **{best['icir']:.4f}** ({best['dt_str']}) |")
        lines.append(f"| 本轮最佳 Composite | {best['composite']:.4f} |")
        if best.get("enhanced") == best.get("enhanced"):
            lines.append(f"| 本轮最佳 Enhanced | **{best['enhanced']:.4f}** |")
        lines.append("")

    # ── 版本变更摘要 ─────────────────────────────────────────────────────
    if prev_ver is not None:
        new_adds, removed, readded = _compute_version_diff(groups, cur_ver, prev_ver)
        lines.append(f"## 版本变更：v{prev_ver} → v{cur_ver}")
        lines.append("")
        lines.append("| 变更类型 | 因子 |")
        lines.append("|---|---|")
        if new_adds:
            lines.append(f"| ✅ 新增 | {', '.join(f'`{n}`' for n in sorted(new_adds))} |")
        if readded:
            lines.append(f"| 🔄 重新加入 | {', '.join(f'`{n}`' for n in sorted(readded))} |")
        if removed:
            lines.append(f"| ❌ 移除 | {', '.join(f'`{n}`' for n in sorted(removed))} |")
        if not new_adds and not removed and not readded:
            lines.append("| ℹ️ 无变更 | 因子列表与上一版本完全相同（仅数据窗口刷新）|")
        lines.append("")
    
    # ── 因子性能汇总表 ────────────────────────────────────────────────────
    lines.append("## 因子性能汇总（按 IC 降序）")
    lines.append("")
    lines.append("| # | 因子名 | IC | IC_IR(全期) | IC(样本内) | IC(样本外) | nan比例 | 最大相关特征 |")
    lines.append("|---|---|---|---|---|---|---|---|")

    sorted_factors = sorted(factors, key=lambda x: x.get("ic", 0), reverse=True)
    for i, f in enumerate(sorted_factors, 1):
        name = f.get("name", "?")
        ic = _fmt_float(f.get("ic"), 4)
        ic_ir = _fmt_float(f.get("ic_in_ir"), 4)
        ic_in = _fmt_float(f.get("ic_in_mean"), 4)
        ic_oos = _fmt_float(f.get("ic_oos"), 4)
        nan_r = f.get("nan_ratio", "?")
        nan_str = f"{nan_r:.4f}" if isinstance(nan_r, float) else str(nan_r)
        max_corr = f.get("max_abs_corr_vs_lgb")
        argmax = f.get("argmax_lgb_col", "")
        corr_str = f"{max_corr:.3f} ({argmax})" if max_corr is not None else "N/A"

        # 标记新增/重加入
        tag = ""
        if prev_ver is not None:
            new_adds, _, readded = _compute_version_diff(groups, cur_ver, prev_ver)
            if name in new_adds:
                tag = " ⭐"
            elif name in readded:
                tag = " 🔄"

        lines.append(
            f"| {i} | `{name}`{tag} | {ic} | {ic_ir} | {ic_in} | {ic_oos} | {nan_str} | {corr_str} |"
        )
    lines.append("")

    # ── 各因子详情 ────────────────────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## 各因子详情")
    lines.append("")

    for i, f in enumerate(sorted_factors, 1):
        name = f.get("name", "?")
        ic = f.get("ic", 0)
        nan_r = f.get("nan_ratio", 0)
        ic_in = f.get("ic_in_mean")
        ic_oos = f.get("ic_oos")
        ic_ir = f.get("ic_in_ir")
        max_corr = f.get("max_abs_corr_vs_lgb")
        argmax = f.get("argmax_lgb_col", "")

        # 标记
        tag_str = ""
        if prev_ver is not None:
            new_adds, _, readded = _compute_version_diff(groups, cur_ver, prev_ver)
            if name in new_adds:
                tag_str = " ⭐ 新增"
            elif name in readded:
                tag_str = " 🔄 重新加入"

        lines.append(f"### {i}. `{name}`{tag_str}")
        sign = "+" if ic >= 0 else ""
        lines.append(f"- IC = **{sign}{ic:.6f}**, nan_ratio = {nan_r}")
        if ic_in is not None and ic_oos is not None and ic_ir is not None:
            lines.append(f"- IC(in) = {ic_in:.6f}, IC(oos) = {ic_oos:.6f}, IC_IR = {ic_ir:.4f}")
        if max_corr is not None:
            lines.append(f"- Max abs corr vs LGB = {max_corr:.4f} (`{argmax}`)")
        lines.append("")

        # 源文件
        paths = src_map.get(name, [])
        if paths:
            best_path = _pick_best_source(paths)
            rel = best_path.relative_to(_ROOT)
            rel_str = str(rel).replace("\\", "/")
            lines.append(f"- 源文件: `{rel_str}`")
            if len(paths) > 1:
                lines.append(
                    f"  _(同名实现 {len(paths)} 个，已选最新修改版本)_"
                )
        else:
            lines.append("- 源文件: _未在 RD-Agent_workspace 中找到_")
        lines.append("")

        # 源代码
        if include_code:
            code, _ = _read_factor_code(name, src_map)
            if code:
                lines.append("```python")
                lines.append(code.rstrip())
                lines.append("```")
            else:
                lines.append("_（源代码未找到）_")
            lines.append("")

        lines.append("")
        lines.append("---")
        lines.append("")

    # ── 组合模型历史性能表 ────────────────────────────────────────────────
    if perf_history:
        lines.append("## 组合模型历史性能（debug 日志解析）")
        lines.append("")

        # 只保留最近一段时间 / 最后 N 条
        recent = perf_history[-50:]
        lines.append("| 时间 | IC | IC_IR | Composite | Enhanced | Div.Bonus | 来源日志 |")
        lines.append("|---|---|---|---|---|---|---|")
        for p in recent:
            comp_str = f"{p['composite']:.4f}" if p["composite"] == p["composite"] else "N/A"
            enh = p.get("enhanced", float("nan"))
            div = p.get("diversity_bonus", float("nan"))
            enh_str = f"{enh:.4f}" if enh == enh else "N/A"
            div_str = f"{div:.4f}" if div == div else "N/A"
            lines.append(
                f"| {p['dt_str']} | {p['ic']:.4f} | **{p['icir']:.4f}** | {comp_str} | "
                f"{enh_str} | {div_str} | {p['source_log']} |"
            )
        lines.append("")

    # ── 候选因子列表 ─────────────────────────────────────────────────────
    if candidate_factors:
        lines.append("## 本次会话候选因子（未采纳）")
        lines.append("")
        lines.append("| 因子名 | 工作空间 ID | 修改时间 | result.h5 |")
        lines.append("|---|---|---|---|")
        for c in candidate_factors:
            has_res = "✅" if c["has_result"] else "❌"
            lines.append(
                f"| `{c['name']}` | `{c['ws_id']}` | {c['mtime_str']} | {has_res} |"
            )
        lines.append("")

    # ── 版本历史完整摘要 ──────────────────────────────────────────────────
    if groups:
        lines.append("## manifest 版本历史摘要")
        lines.append("")
        lines.append("| 版本 | 因子数 | 状态 | 注册时间 | 退役时间 |")
        lines.append("|---|---|---|---|---|")

        all_vers = sorted(groups.keys())
        for v in all_vers:
            facs = groups[v]
            statuses = {f.get("status", "?") for f in facs}
            status_str = "/".join(sorted(statuses))
            reg_times = [f.get("registered_at", "") for f in facs if f.get("registered_at")]
            ret_times = [f.get("retired_at", "") for f in facs if f.get("retired_at")]
            reg_str = min(reg_times)[:10] if reg_times else "?"
            ret_str = max(ret_times)[:10] if ret_times else "—"
            factor_names_v = ", ".join(
                f"`{f.get('factor_name') or f.get('name', '?')}`" for f in facs
            )
            lines.append(
                f"| v{v} | {len(facs)} | {status_str} | {reg_str} | {ret_str} |"
            )
        lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="自动汇总 rdagent 运行结果")
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="输出文件路径（默认自动命名到 git_ignore_folder/）",
    )
    parser.add_argument(
        "--no-code",
        action="store_true",
        help="不在文档中输出因子源代码",
    )
    parser.add_argument(
        "--prev-ver",
        type=int,
        default=None,
        help="指定对比的上一版本号（默认自动推断为 cur_ver - 1）",
    )
    parser.add_argument(
        "--since-hours",
        type=float,
        default=48.0,
        help="只统计最近 N 小时内修改的候选 workspace（默认 48h）",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        dest="print_only",
        help="仅打印到终端，不写文件",
    )
    args = parser.parse_args()

    print("[summarize_rdagent] 开始汇总…", file=sys.stderr)

    # 1. manifest
    manifest_factors = _load_manifest()
    groups = _group_by_version(manifest_factors)
    cur_ver = _get_active_version(manifest_factors)

    if cur_ver == 0:
        print("[warn] manifest 中没有 active 因子，将使用最大版本号", file=sys.stderr)
        cur_ver = max(groups.keys(), default=0)

    prev_ver: Optional[int] = args.prev_ver
    if prev_ver is None and cur_ver > 0:
        # 找 cur_ver 之前最近的已有版本
        prev_candidates = sorted([v for v in groups if v < cur_ver], reverse=True)
        prev_ver = prev_candidates[0] if prev_candidates else None

    print(f"[summarize_rdagent] 当前版本 v{cur_ver}，对比版本 v{prev_ver}", file=sys.stderr)

    # 2. 当前因子性能
    meta, factors = _load_current_factors()
    if not factors:
        print("[warn] combined_factors_df.json 为空或不存在", file=sys.stderr)

    # 3. 源码映射
    print("[summarize_rdagent] 扫描 workspace 源码…", file=sys.stderr)
    src_map = _build_factor_source_map()
    print(f"[summarize_rdagent] 找到 {len(src_map)} 个不同因子名的实现", file=sys.stderr)

    # 4. debug 日志
    print("[summarize_rdagent] 解析 debug 日志…", file=sys.stderr)
    perf_history = _parse_debug_logs(max_logs=10)
    print(f"[summarize_rdagent] 找到 {len(perf_history)} 条组合模型性能记录", file=sys.stderr)

    # 5. 候选因子
    import time
    since_ts = time.time() - args.since_hours * 3600
    candidates = _scan_candidate_factors(since_ts=since_ts)
    # 从候选中排除已在 combined 中的因子
    combined_names = {f.get("name", "") for f in factors}
    candidates_new = [c for c in candidates if c["name"] not in combined_names]
    print(
        f"[summarize_rdagent] 找到 {len(candidates_new)} 个候选因子（未采纳，最近 {args.since_hours:.0f}h 内）",
        file=sys.stderr,
    )

    # 6. 渲染
    print("[summarize_rdagent] 渲染 Markdown…", file=sys.stderr)
    md = _render_markdown(
        meta=meta,
        factors=factors,
        src_map=src_map,
        manifest_factors=manifest_factors,
        groups=groups,
        cur_ver=cur_ver,
        prev_ver=prev_ver,
        perf_history=perf_history,
        candidate_factors=candidates_new,
        include_code=not args.no_code,
    )

    # 7. 输出
    if args.print_only:
        import io, sys as _sys
        # Windows 终端可能不支持 emoji/中文，统一用 utf-8 输出
        wrapper = io.TextIOWrapper(_sys.stdout.buffer, encoding="utf-8", errors="replace")
        wrapper.write(md + "\n")
        wrapper.flush()
        return

    out_path: Path
    if args.out:
        out_path = Path(args.out)
    else:
        ts_str = datetime.now().strftime("%Y%m%d_%H%M")
        out_path = _GIT_IGNORE / f"combined_factors_df_expressions_v{cur_ver}_{ts_str}.md"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"[summarize_rdagent] ✅ 文档已写入：{out_path}", file=sys.stderr)

    # 终端摘要
    _print_terminal_summary(factors, cur_ver, prev_ver, groups, perf_history)


def _safe_print(msg: str) -> None:
    """Windows-safe print，将无法编码的字符替换为 '?'。"""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8", errors="replace"))


def _print_terminal_summary(
    factors: list[dict],
    cur_ver: int,
    prev_ver: Optional[int],
    groups: dict[int, list[dict]],
    perf_history: list[dict],
) -> None:
    """在终端输出简洁摘要。"""
    sep = "-" * 70

    _safe_print("")
    _safe_print(sep)
    _safe_print(f"  rdagent 汇总摘要  (combined_factors_df v{cur_ver})")
    _safe_print(sep)

    # 版本变更
    if prev_ver is not None:
        new_adds, removed, readded = _compute_version_diff(groups, cur_ver, prev_ver)
        _safe_print(f"\n  版本变更 v{prev_ver} -> v{cur_ver}:")
        if new_adds:
            _safe_print(f"  [+] 新增:   {', '.join(sorted(new_adds))}")
        if readded:
            _safe_print(f"  [~] 重加入: {', '.join(sorted(readded))}")
        if removed:
            _safe_print(f"  [-] 移除:   {', '.join(sorted(removed))}")
        if not new_adds and not removed and not readded:
            _safe_print("  [i] 仅数据窗口刷新，无因子增减")

    # 因子性能表
    if factors:
        sorted_f = sorted(factors, key=lambda x: x.get("ic", 0), reverse=True)
        _safe_print(f"\n  因子性能表（共 {len(sorted_f)} 个，按 IC 降序）:")
        _safe_print(f"  {'#':>3}  {'因子名':<45} {'IC':>8} {'IC_IR':>8} {'IC_OOS':>8} {'nan%':>6}")
        _safe_print(f"  {'-'*3}  {'-'*45} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
        for i, f in enumerate(sorted_f, 1):
            name = f.get("name", "?")
            ic = f.get("ic", 0)
            ic_ir = f.get("ic_in_ir", float("nan"))
            ic_oos = f.get("ic_oos", float("nan"))
            nan_r = f.get("nan_ratio", float("nan"))
            ic_ir_s = f"{ic_ir:.4f}" if ic_ir == ic_ir else " N/A "
            ic_oos_s = f"{ic_oos:.4f}" if ic_oos == ic_oos else " N/A "
            nan_s = f"{nan_r*100:.2f}" if nan_r == nan_r else " N/A"
            tag = ""
            if prev_ver is not None:
                new_adds, _, readded = _compute_version_diff(groups, cur_ver, prev_ver)
                if name in new_adds:
                    tag = "[+]"
                elif name in readded:
                    tag = "[~]"
            _safe_print(
                f"  {i:>3}  {name:<43} {tag:<4} {ic:>+8.4f} {ic_ir_s:>8} {ic_oos_s:>8} {nan_s:>5}%"
            )

    # 组合模型最新性能
    if perf_history:
        latest = perf_history[-1]
        best = max(perf_history, key=lambda x: x["icir"])
        _safe_print(f"\n  组合模型（最新 {latest['dt_str']}）:")
        _safe_print(
            f"  IC={latest['ic']:.4f}  IC_IR={latest['icir']:.4f}  "
            f"Composite={latest['composite']:.4f}"
        )
        enh = latest.get("enhanced", float("nan"))
        div = latest.get("diversity_bonus", float("nan"))
        if enh == enh:
            _safe_print(f"  Enhanced={enh:.4f}", end="")
        if div == div:
            _safe_print(f"  DiversityBonus={div:.4f}", end="")
        if enh == enh or div == div:
            _safe_print("")
        _safe_print(
            f"  本轮最佳: IC_IR={best['icir']:.4f}  Composite={best['composite']:.4f}  ({best['dt_str']})"
        )

    _safe_print("")
    _safe_print(sep)


if __name__ == "__main__":
    main()
