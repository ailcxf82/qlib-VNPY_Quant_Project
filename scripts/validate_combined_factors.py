"""
Validate exported RDAgent factors by running a standalone Qlib LGB training.
This is a quick sanity check: compares baseline (no extra factors) vs combined.

Run (Windows, qlib_zhengshi env):
  python scripts/validate_combined_factors.py

Run (WSL, rdagent env):
  python scripts/validate_combined_factors.py
"""
from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

# ── OS-aware paths ─────────────────────────────────────────────────────────
_ON_WIN = platform.system() == "Windows"
PROVIDER_URI = "D:/qlib_data/qlib_data" if _ON_WIN else "/mnt/d/qlib_data/qlib_data"
_GIT_IGNORE = (
    Path(__file__).resolve().parent.parent / "git_ignore_folder"
    if _ON_WIN
    else Path("/mnt/d/quant_project/Qlib_Quant/qlib-VNPY_Quant_Project/git_ignore_folder")
)
PARQUET_PATH = _GIT_IGNORE / "combined_factors_df.parquet"

if not PARQUET_PATH.exists():
    print(f"ERROR: {PARQUET_PATH} not found. Run export_rdagent_factors.py first.")
    sys.exit(1)

print(f"[validate] OS={platform.system()}, provider_uri={PROVIDER_URI}")
print(f"[validate] Parquet: {PARQUET_PATH}")

import pandas as pd
pf = pd.read_parquet(str(PARQUET_PATH))
print(f"[validate] Parquet shape: {pf.shape}")
print(f"[validate] Factors: {list(pf.columns)}")
print(f"[validate] Date range: {pf.index.get_level_values('datetime').min()} ~ "
      f"{pf.index.get_level_values('datetime').max()}")
nan_pct = pf.isna().mean() * 100
print(f"[validate] NaN% per factor:")
for col in pf.columns:
    print(f"  {col:32s} {nan_pct[col]:.1f}%")

# ── Optionally compute IC vs forward return ─────────────────────────────────
if __name__ == "__main__" and "--ic" in sys.argv:
    import qlib
    from qlib.data import D
    from qlib.config import REG_CN

    qlib.init(provider_uri=PROVIDER_URI, region=REG_CN)
    insts = D.list_instruments(D.instruments("all"), start_time="2020-01-01",
                               end_time="2022-12-31", as_list=True)[:200]
    # Forward return label (3-day: close[t+2]/close[t+1]-1)
    label_df = D.features(insts, ["Ref($close_qfq,-2)/Ref($close_qfq,-1)-1"],
                          start_time="2020-01-01", end_time="2022-12-31")
    label_df.columns = ["label"]
    label = label_df["label"].swaplevel().sort_index()

    print(f"\n[validate] IC vs forward return (2020-2022, {len(insts)} instruments):")
    print(f"  {'Factor':<32} {'IC':>8} {'|IC|':>8}")
    print(f"  {'-'*52}")
    results = []
    for col in pf.columns:
        aligned = pd.DataFrame({"f": pf[col], "l": label}).dropna()
        if len(aligned) < 100:
            print(f"  {col:<32} {'N/A':>8}")
            continue
        ic = float(aligned["f"].rank().corr(aligned["l"].rank()))
        results.append((col, ic))
        print(f"  {col:<32} {ic:>8.4f} {abs(ic):>8.4f}")

    if results:
        mean_ic = sum(abs(r[1]) for r in results) / len(results)
        print(f"\n  Mean |IC|: {mean_ic:.4f}")
        print(f"  Positive IC: {sum(1 for r in results if r[1] > 0)}/{len(results)}")

print("\n[validate] Done.")
