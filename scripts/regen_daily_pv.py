"""
Regenerate daily_pv.h5 for RD-Agent factor workspaces.

Works on both Windows (qlib_zhengshi env) and WSL/Linux (rdagent env).
Paths are resolved automatically based on the running OS.

Run inside WSL (recommended):
  /home/administrator/.local/share/mamba/envs/rdagent/bin/python scripts/regen_daily_pv.py

Run on Windows:
  python scripts/regen_daily_pv.py

To also patch all existing workspace directories:
  python scripts/regen_daily_pv.py --patch-workspaces

v2 changes (2026-05-21):
  - FIELDS expanded from 27 → 42 columns (full qlib field coverage)
  - Column names kept as qlib native names (no renaming)
  - $roa / $roa2_yearly excluded (all-NaN in this data source)
  - $factor excluded (all-NaN)
  - START_DATE aligned to factor_lab.yaml: 2020-01-01
  - END_DATE updated to 2026-05-12
  - MAX_INSTRUMENTS: None = full market (was 200)
"""
from __future__ import annotations

import os
import platform
import shutil
import sys
from pathlib import Path

# ─── OS-aware path resolution ─────────────────────────────────────────────────
def _is_wsl() -> bool:
    """Return True when running inside WSL (Linux kernel, /proc/version contains Microsoft)."""
    if platform.system() != "Linux":
        return False
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except Exception:
        return False

_ON_WINDOWS = platform.system() == "Windows"
_ON_WSL = _is_wsl()

if _ON_WINDOWS:
    PROVIDER_URI = "D:/qlib_data/qlib_data"
    _GIT_IGNORE = Path(__file__).resolve().parent.parent / "git_ignore_folder"
elif _ON_WSL:
    PROVIDER_URI = "/mnt/d/qlib_data/qlib_data"
    _GIT_IGNORE = Path("/mnt/d/quant_project/Qlib_Quant/qlib-VNPY_Quant_Project/git_ignore_folder")
else:
    PROVIDER_URI = os.environ.get("QLIB_PROVIDER_URI", "")
    if not PROVIDER_URI:
        print("ERROR: Set QLIB_PROVIDER_URI env var when running outside Windows/WSL.", file=sys.stderr)
        sys.exit(1)
    _GIT_IGNORE = Path(os.environ.get(
        "RDAGENT_GIT_IGNORE",
        str(Path(__file__).resolve().parent.parent / "git_ignore_folder"),
    ))

_PROJECT_ROOT = _GIT_IGNORE
OUT_DIR = _PROJECT_ROOT / "factor_implementation_source_data"
OUT_DIR_DEBUG = _PROJECT_ROOT / "factor_implementation_source_data_debug"
WORKSPACE_ROOT = _PROJECT_ROOT / "RD-Agent_workspace"

# ─── Constants (module-level: safe for Windows multiprocessing workers to import)
START_DATE = "2020-01-01"
END_DATE   = "2026-05-12"
MAX_INSTRUMENTS = None   # None = full market; set int to limit (e.g. 500)
TARGET_FILENAME = "daily_pv.h5"

# v2: 42 columns — full qlib field coverage, native names (no renaming)
# Excluded: $factor (100% NaN), $roa/$roa2_yearly (all-NaN in this source)
FIELDS = [
    # ── Price: raw ────────────────────────────────────────────────────────
    "$close",         "$open",          "$high",          "$low",
    # ── Price: forward-adjusted ───────────────────────────────────────────
    "$close_qfq",     "$open_qfq",      "$high_qfq",      "$low_qfq",
    # ── Volume / Amount ───────────────────────────────────────────────────
    "$vol",           "$volume",        "$amount",
    # ── Liquidity ─────────────────────────────────────────────────────────
    "$turnover_rate", "$turnover_rate_f", "$volume_ratio",
    # ── Technical indicators (pre-computed) ───────────────────────────────
    "$rsi_qfq_12",    "$macd_qfq",      "$macd_dif_qfq",  "$macd_dea_qfq",
    "$kdj_k_qfq",     "$kdj_d_qfq",     "$kdj_qfq",       "$atr_qfq",
    "$mtmma_qfq",
    # ── Valuation ─────────────────────────────────────────────────────────
    "$pe",            "$pe_ttm",        "$pb",
    "$ps",            "$ps_ttm",        "$total_mv",       "$dv_ratio",
    "$dv_ttm",
    # ── Profitability / Quality ────────────────────────────────────────────
    "$roe",           "$q_profit_yoy",  "$q_eps",
    "$assets_turn",   "$profit_to_gr",
    # ── Money flow (institutional order imbalance) ─────────────────────────
    "$net_amount",
    "$buy_elg_amount", "$buy_lg_amount", "$buy_md_amount",  "$buy_sm_amount",
    # ── Margin financing ──────────────────────────────────────────────────
    "$rzye",          "$rqye",
]

# Column names = field names (kept identical to qlib native names)
COLUMN_NAMES = FIELDS[:]

# ─── Main logic ───────────────────────────────────────────────────────────────
# MUST be guarded with __name__ == "__main__" on Windows.
# Windows multiprocessing uses "spawn": each worker re-imports this module,
# executing all top-level code. Without this guard, D.features() → spawns
# workers → re-runs D.features() → infinite recursion.
if __name__ == "__main__":
    import qlib
    import pandas as pd
    from qlib.config import REG_CN
    from qlib.data import D

    print(f"[regen] OS: {platform.system()} (WSL={_ON_WSL})")
    print(f"[regen] PROVIDER_URI: {PROVIDER_URI}")
    print(f"[regen] OUT_DIR: {OUT_DIR}")

    print(f"[regen] Initializing qlib with provider_uri={PROVIDER_URI}")
    qlib.init(provider_uri=PROVIDER_URI, region=REG_CN)

    print("[regen] Loading instrument universe...")
    instruments = D.instruments(market="all")

    n_limit = MAX_INSTRUMENTS if MAX_INSTRUMENTS else "all"
    print(f"[regen] Fetching features from {START_DATE} to {END_DATE} ({n_limit} instruments) ...")
    from qlib.data import D as _D
    inst_list = _D.list_instruments(
        instruments=instruments,
        start_time=START_DATE,
        end_time=END_DATE,
        as_list=True,
    )
    if MAX_INSTRUMENTS:
        inst_list = inst_list[:MAX_INSTRUMENTS]
    print(f"[regen] Using {len(inst_list)} instruments. Sample: {inst_list[:5]}")

    data = (
        D.features(inst_list, FIELDS, start_time=START_DATE, end_time=END_DATE, freq="day")
        .swaplevel()
        .sort_index()
    )
    data.columns = COLUMN_NAMES

    print(f"[regen] Fetched shape: {data.shape}")
    if data.empty:
        raise RuntimeError("Fetched data is empty! Check provider_uri and instrument universe.")

    print("[regen] NaN ratios per column:")
    for col in data.columns:
        ratio = data[col].isna().mean()
        tag = " ★ HIGH" if ratio > 0.3 else (" * " if ratio > 0.05 else "")
        print(f"  {col:<30} {ratio:.1%}{tag}")

    for d in (OUT_DIR, OUT_DIR_DEBUG):
        d.mkdir(parents=True, exist_ok=True)
        dst = d / TARGET_FILENAME
        print(f"[regen] Writing to {dst} ...")
        data.to_hdf(str(dst), key="data", mode="w")

    out_path = OUT_DIR / TARGET_FILENAME
    print(f"[regen] Written. Verifying...")
    verify = pd.read_hdf(str(out_path))
    dt = verify.index.get_level_values("datetime")
    inst = verify.index.get_level_values("instrument")
    print(f"[regen] shape          : {verify.shape}")
    print(f"[regen] columns ({len(verify.columns):2d})  : {list(verify.columns)}")
    print(f"[regen] date range     : {dt.min().date()} ~ {dt.max().date()}")
    print(f"[regen] n_instruments  : {inst.nunique()}")
    print(f"[regen] sample insts   : {sorted(inst.unique())[:3]}")
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[regen] file size      : {size_mb:.1f} MB")

    # Workspace patching: symlink canonical daily_pv.h5 (no per-dir 1.2GB copies).
    # Use --patch-workspaces to enable on Windows, or run from WSL.
    _do_patch = "--patch-workspaces" in sys.argv or not _ON_WINDOWS
    patched = 0
    skipped = 0
    canon_abs = out_path.resolve()

    def _link_ok(dest: Path) -> bool:
        if not dest.is_symlink():
            return False
        try:
            return dest.resolve() == canon_abs
        except OSError:
            return False

    if not _do_patch:
        print(f"\n[regen] Skipping workspace patching on Windows (use --patch-workspaces to enable).")
    else:
        print(f"\n[regen] Symlinking workspace dirs under {WORKSPACE_ROOT} -> {canon_abs}")
        for ws_dir in sorted(WORKSPACE_ROOT.iterdir()):
            if not ws_dir.is_dir():
                continue
            dest = ws_dir / TARGET_FILENAME
            if _link_ok(dest):
                skipped += 1
                continue
            try:
                if dest.exists() or dest.is_symlink():
                    dest.unlink()
                os.symlink(str(canon_abs), str(dest))
                patched += 1
            except OSError as _e:
                print(f"[regen]   WARN: skip {ws_dir.name}: {_e}")
                skipped += 1
                continue
            if patched % 100 == 0:
                print(f"[regen]   linked {patched} (skipped {skipped}) ...")

        print(f"[regen] Linked {patched} workspace dirs, skipped {skipped} (already correct symlink).")

    print("[regen] Done.")
