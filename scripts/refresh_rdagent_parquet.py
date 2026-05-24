"""
A+B：刷新 combined_factors_df.parquet 的时间/股票覆盖到主工程训练窗口。

做法：
1. 从项目的 qlib_data（D:/qlib_data/qlib_data）按目标股票池读 OHLCV/基本面，
   按 RD-Agent schema（$close/$open/.../$rsi12/$macd）构造一份 in-memory daily_pv。
2. monkey-patch pandas，把 `pd.read_hdf('daily_pv.h5')` 拦截为返回上述 in-memory df；
   把 `DataFrame.to_hdf(..., 'result.h5', ...)` 拦截为内存捕获，不落盘。
3. 遍历 RD-Agent_workspace 下所有带 factor.py 的 ws，exec 代码自动触发
   `if __name__=='__main__': calculate_XXX()`，把返回 result 装进 candidates。
4. 计算 |IC|（vs 主工程 label），按阈值+ top-N 挑选，拼接成新 parquet。
5. 旧 parquet 自动备份为 combined_factors_df.parquet.bak.<ts>。

环境：conda run -n qlib_zhengshi --no-capture-output python scripts/refresh_rdagent_parquet.py
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger("refresh_rdagent_parquet")

FACTOR_LAB_YAML = _ROOT / "config" / "factor_lab.yaml"


def _default_refresh_window() -> tuple[str, str]:
    """统一入口：默认时间窗来自 config/factor_lab.yaml → registry.refresh_time_window。"""
    try:
        import yaml

        if not FACTOR_LAB_YAML.exists():
            return "2020-01-01", "2026-04-27"
        data = yaml.safe_load(FACTOR_LAB_YAML.read_text(encoding="utf-8")) or {}
        tw = (data.get("registry") or {}).get("refresh_time_window") or {}
        start = tw.get("start") or "2020-01-01"
        end = tw.get("end") or "2026-04-27"
        return str(start), str(end)
    except Exception as e:
        logger.warning("读取 factor_lab.yaml refresh_time_window 失败（%s），使用内置默认", e)
        return "2020-01-01", "2026-04-27"

PROVIDER_URI = "D:/qlib_data/qlib_data"
GIT_IGNORE = _ROOT / "git_ignore_folder"


def _resolve_provider_uri() -> str:
    """WSL 下使用 /mnt/d/...；Windows 下可用 D:/...；优先 QLIB_PROVIDER_URI。"""
    import os

    candidates = [
        os.environ.get("QLIB_PROVIDER_URI", "").strip(),
        PROVIDER_URI,
        "/mnt/d/qlib_data/qlib_data",
    ]
    for uri in candidates:
        if not uri:
            continue
        p = Path(uri)
        if p.is_dir():
            return str(p.resolve())
    return PROVIDER_URI
WS_ROOT = GIT_IGNORE / "RD-Agent_workspace"


def _refresh_output_paths() -> tuple[Path, Path]:
    """输出路径与 factor_lab.yaml registry.refresh_* 对齐（默认 git_ignore_folder/…）。"""
    try:
        import yaml

        if FACTOR_LAB_YAML.exists():
            reg = (yaml.safe_load(FACTOR_LAB_YAML.read_text(encoding="utf-8")) or {}).get("registry") or {}
            pq_rel = reg.get("refresh_combined_parquet", "git_ignore_folder/combined_factors_df.parquet")
            js_rel = reg.get("refresh_summary_json", "git_ignore_folder/combined_factors_df.json")
            return (_ROOT / pq_rel).resolve(), (_ROOT / js_rel).resolve()
    except Exception:
        pass
    return (GIT_IGNORE / "combined_factors_df.parquet").resolve(), (GIT_IGNORE / "combined_factors_df.json").resolve()


OUT_PATH, OUT_JSON = _refresh_output_paths()

# RD-Agent factor.py 期望的字段 ← qlib_data 实际字段
FIELD_MAP = {
    "$close": "$close_qfq",
    "$open": "$open_qfq",
    "$high": "$high_qfq",
    "$low": "$low_qfq",
    "$volume": "$vol",
    "$amount": "$amount",
    "$turnover_rate": "$turnover_rate",
    "$turnover_rate_f": "$turnover_rate_f",
    "$volume_ratio": "$volume_ratio",
    "$pe": "$pe",
    "$pe_ttm": "$pe_ttm",
    "$pb": "$pb",
    "$ps": "$ps",
    "$ps_ttm": "$ps_ttm",
    "$total_mv": "$total_mv",
    "$dv_ratio": "$dv_ratio",
    "$dv_ttm": "$dv_ttm",
    "$roe": "$roe",
    "$roa": "$roa",
    "$q_profit_yoy": "$q_profit_yoy",
    "$q_eps": "$q_eps",
    "$rsi12": "$rsi_qfq_12",
    "$macd": "$macd_qfq",
    "$atr": "$atr_qfq",
    "$net_amount": "$net_amount",
}


def build_in_memory_daily_pv(start: str, end: str, instruments: str) -> pd.DataFrame:
    """按 RD-Agent schema 从 qlib_data 构造 daily_pv-like DataFrame。"""
    import qlib
    from qlib.config import REG_CN
    from qlib.data import D

    provider = _resolve_provider_uri()
    logger.info("qlib provider_uri=%s", provider)
    qlib.init(provider_uri=provider, region=REG_CN)

    rd_fields = list(FIELD_MAP.keys())
    q_fields = list(FIELD_MAP.values())
    pools = [p.strip() for p in str(instruments).split(",") if p.strip()]
    insts: list[str] = []
    seen: set[str] = set()
    for pool in pools:
        try:
            part = D.list_instruments(D.instruments(pool), start_time=start, end_time=end, as_list=True)
        except Exception as e:
            logger.warning("pool=%s 取列表失败：%s", pool, e)
            continue
        added = 0
        for code in part:
            k = str(code)
            if k not in seen:
                seen.add(k)
                insts.append(k)
                added += 1
        logger.info("pool=%s 贡献 %d 只（累计 %d）", pool, added, len(insts))
    if not insts:
        raise RuntimeError(f"instruments 解析结果为空：{instruments}")
    pv = D.features(insts, q_fields, start_time=start, end_time=end, freq="day")
    if pv.empty:
        raise RuntimeError(f"qlib 返回空数据：instruments={instruments}, {start}~{end}")
    pv.columns = rd_fields
    pv = pv.sort_index()
    logger.info(
        "daily_pv shape=%s, dt=%s~%s, n_inst=%d",
        pv.shape,
        pv.index.get_level_values("datetime").min(),
        pv.index.get_level_values("datetime").max(),
        pv.index.get_level_values("instrument").nunique(),
    )
    return pv


def build_label(pv: pd.DataFrame) -> pd.Series:
    """主工程 label：close[t+3]/close[t+1] - 1（与 config/data.yaml 一致）。"""
    close = pv["$close"].copy()
    fwd = (
        close.groupby(level="instrument", group_keys=False)
        .transform(lambda s: s.shift(-3) / s.shift(1) - 1)
    )
    return fwd.rename("LABEL0")


@contextmanager
def _patch_io(pv: pd.DataFrame, captured: list):
    orig_read_hdf = pd.read_hdf
    orig_to_hdf = pd.DataFrame.to_hdf

    def fake_read_hdf(path_or_buf, *args, **kwargs):
        p = str(path_or_buf)
        if p.endswith("daily_pv.h5") or p == "daily_pv.h5":
            return pv.copy()
        return orig_read_hdf(path_or_buf, *args, **kwargs)

    def fake_to_hdf(self, path_or_buf, *args, **kwargs):
        p = str(path_or_buf)
        if p.endswith("result.h5") or p == "result.h5":
            captured.append(self.copy())
            return None
        return orig_to_hdf(self, path_or_buf, *args, **kwargs)

    pd.read_hdf = fake_read_hdf
    pd.DataFrame.to_hdf = fake_to_hdf
    try:
        yield
    finally:
        pd.read_hdf = orig_read_hdf
        pd.DataFrame.to_hdf = orig_to_hdf


def _factor_name(ws_dir: Path) -> Optional[str]:
    fp = ws_dir / "factor.py"
    if not fp.exists():
        return None
    try:
        for line in fp.read_text(encoding="utf-8", errors="replace").splitlines():
            s = line.strip()
            if s.startswith("def calculate_"):
                return s[len("def calculate_"):].split("(", 1)[0]
    except Exception:
        return None
    return None


def _exec_factor(ws_dir: Path, pv: pd.DataFrame, timeout_warn_sec: float = 20.0) -> Optional[pd.DataFrame]:
    code = (ws_dir / "factor.py").read_text(encoding="utf-8", errors="replace")
    captured: list = []
    ns = {"__name__": "__main__"}
    t0 = time.time()
    with _patch_io(pv, captured):
        exec(compile(code, str(ws_dir / "factor.py"), "exec"), ns)
    if time.time() - t0 > timeout_warn_sec:
        logger.warning("  slow factor (%.1fs): %s", time.time() - t0, ws_dir.name)
    if not captured:
        return None
    return captured[-1]


def _normalize_dt_inst(s: pd.Series) -> pd.Series:
    """把 MultiIndex 调整为 (datetime, instrument) 并按字典序排序。"""
    if s.index.nlevels < 2:
        return s
    names = list(s.index.names)
    if names == ["datetime", "instrument"]:
        return s.sort_index() if not s.index.is_monotonic_increasing else s
    if set(names) == {"datetime", "instrument"} and names[0] != "datetime":
        s = s.swaplevel()
    elif names[0] is None or names[1] is None:
        l0 = s.index.get_level_values(0)
        if l0.dtype == object:
            s = s.swaplevel()
        s.index.set_names(["datetime", "instrument"], inplace=True)
    return s.sort_index()


def _ic(factor: pd.Series, label: pd.Series) -> float:
    try:
        aligned = pd.DataFrame({"f": factor, "l": label}).dropna()
        if len(aligned) < 500:
            return float("nan")
        return float(aligned["f"].rank().corr(aligned["l"].rank(), method="pearson"))
    except Exception:
        return float("nan")


# ════════════════════════════════════════════════════════════════════════
#                     B 模式：严格 OOS 筛选 + 去冗余
# ════════════════════════════════════════════════════════════════════════

def _ic_in_segments(factor: pd.Series, label: pd.Series,
                    segments: list[tuple[str, str]]) -> list[float]:
    out = []
    for s, e in segments:
        ms = pd.Timestamp(s); me = pd.Timestamp(e)
        df = pd.DataFrame({
            "f": factor.loc[(slice(ms, me), slice(None))],
            "l": label.loc[(slice(ms, me), slice(None))],
        }).dropna()
        if len(df) < 200:
            out.append(np.nan)
        else:
            out.append(float(df["f"].rank().corr(df["l"].rank())))
    return out


def _make_in_segments(start: str, cutoff: str, n: int) -> list[tuple[str, str]]:
    sd = pd.Timestamp(start); ed = pd.Timestamp(cutoff)
    bounds = pd.date_range(sd, ed, periods=n + 1)
    return [(bounds[i].strftime("%Y-%m-%d"),
             (bounds[i + 1] - pd.Timedelta(days=1)).strftime("%Y-%m-%d")) for i in range(n)]


def _load_lgb_short_cycle_panel(start: str, end: str, instruments: list[str]) -> pd.DataFrame:
    import yaml
    from qlib.data import D
    data_cfg = yaml.safe_load((_ROOT / "config" / "data.yaml").read_text(encoding="utf-8"))
    expr = list(data_cfg["data"]["feature_sets"]["lgb_short_cycle"])
    df = D.features(instruments, expr, start_time=start, end_time=end, freq="day")
    df.columns = expr
    if list(df.index.names) != ["datetime", "instrument"]:
        df = df.swaplevel()
    df.index.set_names(["datetime", "instrument"], inplace=True)
    return df.sort_index()


def _max_abs_corr_to_panel(series: pd.Series, panel: pd.DataFrame,
                           sample_n: int = 100_000) -> tuple[float, str]:
    """以 spearman 计算 series 与 panel 各列的最大 |corr|；为加速，对长 series 抽样。"""
    s = series.dropna()
    if len(s) > sample_n:
        s = s.sample(sample_n, random_state=42).sort_index()
    best = (0.0, "")
    for col in panel.columns:
        q = panel[col].reindex(s.index)
        df = pd.DataFrame({"a": s, "b": q}).dropna()
        if len(df) < 500:
            continue
        c = float(df["a"].rank().corr(df["b"].rank()))
        if abs(c) > abs(best[0]):
            best = (c, col)
    return abs(best[0]), best[1]


def _max_abs_corr_to_selected(series: pd.Series, picked: list[dict],
                              sample_n: int = 100_000) -> float:
    if not picked:
        return 0.0
    s = series.dropna()
    if len(s) > sample_n:
        s = s.sample(sample_n, random_state=42).sort_index()
    best = 0.0
    for c in picked:
        q = c["series"].reindex(s.index)
        df = pd.DataFrame({"a": s, "b": q}).dropna()
        if len(df) < 500:
            continue
        v = abs(float(df["a"].rank().corr(df["b"].rank())))
        if v > best:
            best = v
    return best


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    d0_start, d0_end = _default_refresh_window()
    ap = argparse.ArgumentParser(
        description="Refresh combined_factors_df.parquet coverage（默认窗口见 factor_lab.yaml registry.refresh_time_window）",
    )
    ap.add_argument(
        "--start",
        default=None,
        help=f"开始日期（默认 {d0_start}，来自 config/factor_lab.yaml）",
    )
    ap.add_argument(
        "--end",
        default=None,
        help=f"结束日期（默认 {d0_end}，来自 config/factor_lab.yaml）",
    )
    ap.add_argument("--instruments", default="csi500")
    ap.add_argument("--ic-threshold", type=float, default=0.02,
                    help="松模式：仅按 |IC| 过滤")
    ap.add_argument("--max-nan-ratio", type=float, default=0.50)
    ap.add_argument("--max-factors", type=int, default=50)
    ap.add_argument("--dry-run", action="store_true")
    # ── 严格 OOS 模式（B 模式）：传 --oos-cutoff 即启用 ──
    ap.add_argument("--oos-cutoff", default=None,
                    help="启用严格模式：仅用 ≤ 该日期的段算 IC_IR；> 该日期的段仅作 OOS 报告")
    ap.add_argument("--ic-ir-threshold", type=float, default=0.30,
                    help="严格模式：in-sample 段 IC_IR 阈值")
    ap.add_argument("--max-corr-vs-lgb", type=float, default=0.70,
                    help="严格模式：与 lgb_short_cycle 任一列 |spearman| 上限")
    ap.add_argument("--max-corr-among-selected", type=float, default=0.70,
                    help="严格模式：保留集合内部两两 |spearman| 上限（greedy 去冗余）")
    ap.add_argument("--n-segments", type=int, default=12,
                    help="严格模式：in-sample 区间被等分的段数")
    args = ap.parse_args()
    start = args.start if args.start is not None else d0_start
    end = args.end if args.end is not None else d0_end
    strict = bool(args.oos_cutoff)

    if not WS_ROOT.exists():
        logger.error("MISSING: %s", WS_ROOT)
        sys.exit(1)

    logger.info("=== Step 1: build in-memory daily_pv from qlib_data ===")
    logger.info("时间窗: %s ~ %s（factor_lab 默认可覆盖）", start, end)
    pv = build_in_memory_daily_pv(start, end, args.instruments)
    logger.info("=== Step 2: compute label ===")
    label = _normalize_dt_inst(build_label(pv))
    logger.info("label non-null=%d", label.notna().sum())

    logger.info("=== Step 3: scan RD-Agent_workspace ===")
    ws_dirs = [d for d in sorted(WS_ROOT.iterdir()) if d.is_dir()]
    factor_ws = [d for d in ws_dirs if (d / "factor.py").exists()]
    logger.info("total ws=%d, with factor.py=%d", len(ws_dirs), len(factor_ws))

    logger.info("=== Step 4: re-exec each factor.py on new daily_pv ===")
    candidates: list[dict] = []
    seen_names: set[str] = set()
    n_ok = n_fail = n_dup = n_noname = n_nanfail = n_icfail = 0

    for idx, ws_dir in enumerate(factor_ws, 1):
        name = _factor_name(ws_dir)
        if name is None:
            n_noname += 1
            continue
        if name in seen_names:
            n_dup += 1
            continue

        try:
            result = _exec_factor(ws_dir, pv)
        except Exception as e:
            n_fail += 1
            logger.debug("exec failed %s (%s): %s", ws_dir.name, name, e)
            continue
        if result is None or result.empty:
            n_fail += 1
            continue

        # 取第一列作为因子值
        col0 = result.columns[0] if result.shape[1] else None
        if col0 is None:
            n_fail += 1
            continue
        series = pd.to_numeric(result[col0], errors="coerce").rename(name)
        series = _normalize_dt_inst(series)

        nan_ratio = float(series.isna().mean())
        if nan_ratio > args.max_nan_ratio:
            n_nanfail += 1
            continue

        ic = _ic(series, label)
        if np.isnan(ic) or abs(ic) < args.ic_threshold:
            n_icfail += 1
            continue

        seen_names.add(name)
        candidates.append({"name": name, "ic": ic, "nan_ratio": nan_ratio, "series": series})
        n_ok += 1

        if idx % 25 == 0 or idx == len(factor_ws):
            logger.info(
                "  [%d/%d] name=%s ic=%.4f nan=%.3f | ok=%d fail=%d nan_skip=%d ic_skip=%d",
                idx, len(factor_ws), name, ic, nan_ratio, n_ok, n_fail, n_nanfail, n_icfail,
            )

    logger.info(
        "scan done: ok=%d fail=%d dup=%d noname=%d nan_skip=%d ic_skip=%d | candidates=%d",
        n_ok, n_fail, n_dup, n_noname, n_nanfail, n_icfail, len(candidates),
    )
    if not candidates:
        logger.error("no candidate factor qualified — abort.")
        sys.exit(2)

    # ── 选因子：严格模式（B） vs 松模式（旧行为，按 |IC| 排序） ──
    extras_per_factor: Dict[str, dict] = {}
    if strict:
        logger.info("=== Step 4.5: STRICT mode (OOS cutoff=%s) ===", args.oos_cutoff)
        in_segs = _make_in_segments(start, args.oos_cutoff, args.n_segments)
        oos_seg = (
            (pd.Timestamp(args.oos_cutoff) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            end,
        )
        logger.info("in-sample 段 = %d 段，OOS 段 = %s ~ %s",
                    args.n_segments, oos_seg[0], oos_seg[1])

        logger.info("加载 lgb_short_cycle 表达式面板（用于共线性筛除）...")
        instruments = sorted(set(label.index.get_level_values("instrument")))
        lgb_panel = _load_lgb_short_cycle_panel(start, end, instruments)
        logger.info("lgb_short_cycle 面板 shape=%s", lgb_panel.shape)

        # (a) 算每个候选 IC_IR + OOS IC + 与 lgb 最大 corr
        n_dropped_ir = n_dropped_corr = 0
        scored: list[dict] = []
        for c in candidates:
            ic_in_vals = _ic_in_segments(c["series"], label, in_segs)
            valid = [v for v in ic_in_vals if not np.isnan(v)]
            if not valid:
                ic_in_mean = np.nan; ic_in_std = np.nan; ir = np.nan
            else:
                ic_in_mean = float(np.mean(valid))
                ic_in_std = float(np.std(valid))
                ir = ic_in_mean / ic_in_std if ic_in_std > 1e-9 else np.nan
            ic_oos = _ic_in_segments(c["series"], label, [oos_seg])[0]
            corr_max, corr_col = _max_abs_corr_to_panel(c["series"], lgb_panel)
            extras_per_factor[c["name"]] = {
                "ic_in_mean": None if np.isnan(ic_in_mean) else round(ic_in_mean, 6),
                "ic_in_std": None if np.isnan(ic_in_std) else round(ic_in_std, 6),
                "ic_in_ir": None if np.isnan(ir) else round(ir, 4),
                "ic_oos": None if np.isnan(ic_oos) else round(ic_oos, 6),
                "max_abs_corr_vs_lgb": round(corr_max, 4),
                "argmax_lgb_col": corr_col,
            }
            if np.isnan(ir) or ir < args.ic_ir_threshold:
                n_dropped_ir += 1
                continue
            if corr_max > args.max_corr_vs_lgb:
                n_dropped_corr += 1
                continue
            scored.append({**c, "ic_in_ir": ir, "ic_oos": ic_oos,
                           "corr_lgb": corr_max, "corr_lgb_col": corr_col})
        scored.sort(key=lambda x: x["ic_in_ir"], reverse=True)
        logger.info("严格筛除：IC_IR<%s 丢 %d；|corr_vs_lgb|>%s 丢 %d；剩余 %d",
                    args.ic_ir_threshold, n_dropped_ir,
                    args.max_corr_vs_lgb, n_dropped_corr, len(scored))

        # (b) greedy 去冗余 + 截 max-factors
        selected: list[dict] = []
        n_dropped_internal = 0
        for c in scored:
            if len(selected) >= args.max_factors:
                break
            cor_in = _max_abs_corr_to_selected(c["series"], selected)
            if cor_in > args.max_corr_among_selected:
                n_dropped_internal += 1
                continue
            selected.append(c)
        logger.info("内部去冗余（|corr_among_selected|>%s）丢 %d；最终入选 %d",
                    args.max_corr_among_selected, n_dropped_internal, len(selected))
    else:
        candidates.sort(key=lambda x: abs(x["ic"]), reverse=True)
        selected = candidates[: args.max_factors]

    logger.info("selected %d factor(s):", len(selected))
    for c in selected:
        if strict:
            logger.info("  %-28s ic=%+.4f IR=%+.2f oos=%+.4f corr_lgb=%.2f(vs %s)",
                        c["name"], c["ic"], c["ic_in_ir"], c["ic_oos"],
                        c["corr_lgb"], c["corr_lgb_col"][:34])
        else:
            logger.info("  %-28s ic=%+.4f nan=%.3f", c["name"], c["ic"], c["nan_ratio"])

    if args.dry_run:
        logger.info("dry-run: not writing parquet")
        return

    merged: Optional[pd.DataFrame] = None
    for c in selected:
        s = c["series"].to_frame(c["name"])
        merged = s if merged is None else merged.join(s, how="outer")
    assert merged is not None
    merged = merged.sort_index()

    logger.info("merged shape=%s, dt=%s~%s",
                merged.shape,
                merged.index.get_level_values("datetime").min(),
                merged.index.get_level_values("datetime").max())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists():
        bak = OUT_PATH.with_suffix(f".parquet.bak.{int(time.time())}")
        shutil.copy2(OUT_PATH, bak)
        logger.info("旧 parquet 已备份: %s", bak.name)
    merged.to_parquet(str(OUT_PATH))
    logger.info("写入: %s", OUT_PATH)

    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": "refresh_rdagent_parquet.py",
        "mode": "strict" if strict else "loose",
        "start": start,
        "end": end,
        "instruments": args.instruments,
        "ic_threshold": args.ic_threshold,
        "max_nan_ratio": args.max_nan_ratio,
        "max_factors": args.max_factors,
        "oos_cutoff": args.oos_cutoff,
        "ic_ir_threshold": args.ic_ir_threshold if strict else None,
        "max_corr_vs_lgb": args.max_corr_vs_lgb if strict else None,
        "max_corr_among_selected": args.max_corr_among_selected if strict else None,
        "n_segments": args.n_segments if strict else None,
        "n_factors": len(selected),
        "shape": list(merged.shape),
        "factors": [
            {
                "name": c["name"],
                "ic": round(c["ic"], 6),
                "nan_ratio": round(c["nan_ratio"], 4),
                **(extras_per_factor.get(c["name"], {}) if strict else {}),
            }
            for c in selected
        ],
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("summary: %s", OUT_JSON)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
