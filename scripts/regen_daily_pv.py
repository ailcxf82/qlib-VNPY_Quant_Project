"""
Regenerate daily_pv.h5 for RD-Agent factor workspaces.

Works on both Windows (qlib_zhengshi env) and WSL/Linux (rdagent env).
Paths are resolved automatically based on the running OS.

Run on Windows:
  python scripts/regen_daily_pv.py

Run inside WSL:
  conda run -n rdagent python scripts/regen_daily_pv.py

To also patch all existing workspace directories (slow on Windows, ~29 GB IO):
  python scripts/regen_daily_pv.py --patch-workspaces
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
END_DATE   = "2022-12-31"
MAX_INSTRUMENTS = 200
TARGET_FILENAME = "daily_pv.h5"

FIELDS = [
    # ── Price / Volume (core) ──────────────────────────────────────────────
    "$close_qfq",       # → $close
    "$open_qfq",        # → $open
    "$high_qfq",        # → $high
    "$low_qfq",         # → $low
    "$vol",             # → $volume
    "$amount",          # → $amount
    # ── Liquidity ─────────────────────────────────────────────────────────
    "$turnover_rate",   # → $turnover_rate
    "$turnover_rate_f", # → $turnover_rate_f
    "$volume_ratio",    # → $volume_ratio
    # ── Fundamental: valuation ────────────────────────────────────────────
    "$pe_ttm",          # → $pe_ttm
    "$pb",              # → $pb
    "$ps_ttm",          # → $ps_ttm
    "$total_mv",        # → $total_mv
    "$dv_ratio",        # → $dv_ratio
    # ── Fundamental: profitability ────────────────────────────────────────
    "$roe",             # → $roe
    "$roa",             # → $roa
    "$q_profit_yoy",    # → $q_profit_yoy
    "$q_eps",           # → $q_eps
    # ── Technical (pre-computed) ──────────────────────────────────────────
    "$rsi_qfq_12",      # → $rsi12
    "$macd_qfq",        # → $macd
    "$macd_dif_qfq",    # → $macd_dif
    "$kdj_k_qfq",       # → $kdj_k
    "$kdj_d_qfq",       # → $kdj_d
    "$atr_qfq",         # → $atr
    # ── Margin financing ──────────────────────────────────────────────────
    "$rzye",            # → $rzye
    "$rqye",            # → $rqye
]

COLUMN_NAMES = [
    "$close", "$open", "$high", "$low", "$volume", "$amount",
    "$turnover_rate", "$turnover_rate_f", "$volume_ratio",
    "$pe_ttm", "$pb", "$ps_ttm", "$total_mv", "$dv_ratio",
    "$roe", "$roa", "$q_profit_yoy", "$q_eps",
    "$rsi12", "$macd", "$macd_dif", "$kdj_k", "$kdj_d", "$atr",
    "$rzye", "$rqye",
]

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

    print(f"[regen] Fetching features from {START_DATE} to {END_DATE} for first {MAX_INSTRUMENTS} instruments ...")
    from qlib.data import D as _D
    inst_list = _D.list_instruments(
        instruments=instruments,
        start_time=START_DATE,
        end_time=END_DATE,
        as_list=True,
    )
    inst_list = inst_list[:MAX_INSTRUMENTS]
    print(f"[regen] Using {len(inst_list)} instruments. Sample: {inst_list[:5]}")

    data = (
        D.features(inst_list, FIELDS, start_time=START_DATE, end_time=END_DATE, freq="day")
        .swaplevel()
        .sort_index()
    )
    data.columns = COLUMN_NAMES

    print("[regen] NaN ratios per column:")
    for col in data.columns:
        ratio = data[col].isna().mean()
        if ratio > 0:
            print(f"  {col}: {ratio:.2%}")

    print(f"[regen] Fetched shape: {data.shape}")
    if data.empty:
        raise RuntimeError("Fetched data is empty! Check provider_uri and instrument universe.")

    all_instruments = data.reset_index()["instrument"].unique()
    selected = all_instruments[:MAX_INSTRUMENTS]
    data = data.swaplevel().loc[selected].swaplevel().sort_index()
    print(f"[regen] After sampling {MAX_INSTRUMENTS} instruments: {data.shape}")

    for d in (OUT_DIR, OUT_DIR_DEBUG):
        d.mkdir(parents=True, exist_ok=True)
        dst = d / TARGET_FILENAME
        print(f"[regen] Writing to {dst} ...")
        data.to_hdf(str(dst), key="data", mode="w")

    out_path = OUT_DIR / TARGET_FILENAME
    print(f"[regen] Written to both data_folder and data_folder_debug. Verifying...")
    verify = pd.read_hdf(str(out_path))
    print(f"[regen] Verified shape: {verify.shape}, dtypes: {verify.dtypes.to_dict()}")
    print(f"[regen] Sample:\n{verify.head(3)}")

    # Workspace patching: copy new daily_pv.h5 into existing workspaces.
    # On Windows, 7000+ dirs × 4MB copy = ~29 GB IO — skipped by default.
    # Use --patch-workspaces to enable on Windows, or run from WSL.
    _do_patch = "--patch-workspaces" in sys.argv or not _ON_WINDOWS
    patched = 0
    skipped = 0

    if not _do_patch:
        print(f"\n[regen] Skipping workspace patching on Windows (use --patch-workspaces to enable).")
    else:
        new_file_size = out_path.stat().st_size
        print(f"\n[regen] Patching workspace dirs under {WORKSPACE_ROOT}...")
        for i, ws_dir in enumerate(sorted(WORKSPACE_ROOT.iterdir())):
            if not ws_dir.is_dir():
                continue
            dest = ws_dir / TARGET_FILENAME
            # Skip if file already matches the new size (already up-to-date)
            if dest.exists() and dest.stat().st_size == new_file_size:
                skipped += 1
                continue
            shutil.copy2(str(out_path), str(dest))
            patched += 1
            if patched % 100 == 0:
                print(f"[regen]   patched {patched} (skipped {skipped}) ...")

        print(f"[regen] Patched {patched} workspace dirs, skipped {skipped} (already up-to-date).")

    print("[regen] Done.")
