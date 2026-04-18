"""
A+B：刷新 combined_factors_df.parquet 的时间/股票覆盖到主工程训练窗口。

做法：
1. 从项目的 qlib_data（D:/qlib_data/qlib_data）按 csi500+csi300 读 OHLCV/基本面，
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

PROVIDER_URI = "D:/qlib_data/qlib_data"
GIT_IGNORE = _ROOT / "git_ignore_folder"
WS_ROOT = GIT_IGNORE / "RD-Agent_workspace"
OUT_PATH = GIT_IGNORE / "combined_factors_df.parquet"
OUT_JSON = GIT_IGNORE / "combined_factors_df.json"

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

    qlib.init(provider_uri=PROVIDER_URI, region=REG_CN)

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


def _ic(factor: pd.Series, label: pd.Series) -> float:
    try:
        aligned = pd.DataFrame({"f": factor, "l": label}).dropna()
        if len(aligned) < 500:
            return float("nan")
        return float(aligned["f"].rank().corr(aligned["l"].rank(), method="pearson"))
    except Exception:
        return float("nan")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    ap = argparse.ArgumentParser(description="Refresh combined_factors_df.parquet coverage")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-04-07")
    ap.add_argument("--instruments", default="csi500,csi300")
    ap.add_argument("--ic-threshold", type=float, default=0.02)
    ap.add_argument("--max-nan-ratio", type=float, default=0.50)
    ap.add_argument("--max-factors", type=int, default=50)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not WS_ROOT.exists():
        logger.error("MISSING: %s", WS_ROOT)
        sys.exit(1)

    logger.info("=== Step 1: build in-memory daily_pv from qlib_data ===")
    pv = build_in_memory_daily_pv(args.start, args.end, args.instruments)
    logger.info("=== Step 2: compute label ===")
    label = build_label(pv)
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

    candidates.sort(key=lambda x: abs(x["ic"]), reverse=True)
    selected = candidates[: args.max_factors]
    logger.info("selected top %d by |IC|:", len(selected))
    for c in selected:
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
        "start": args.start,
        "end": args.end,
        "instruments": args.instruments,
        "ic_threshold": args.ic_threshold,
        "max_nan_ratio": args.max_nan_ratio,
        "max_factors": args.max_factors,
        "n_factors": len(selected),
        "shape": list(merged.shape),
        "factors": [
            {"name": c["name"], "ic": round(c["ic"], 6), "nan_ratio": round(c["nan_ratio"], 4)}
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
