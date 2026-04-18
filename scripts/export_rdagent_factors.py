"""
Export high-quality RDAgent-discovered factors into combined_factors_df.parquet
for use in Qlib's NestedDataLoader (conf_combined_factors.yaml).

Computes IC for each factor directly from result.h5 vs the Qlib label,
so no qlib_res.csv is required.

Usage (from project root):
  python scripts/export_rdagent_factors.py                    # default settings
  python scripts/export_rdagent_factors.py --dry-run          # preview only
  python scripts/export_rdagent_factors.py --ic-threshold 0.02 --max-factors 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_WS_ROOT   = Path("git_ignore_folder/RD-Agent_workspace")
DEFAULT_PV_PATH   = Path("git_ignore_folder/factor_implementation_source_data/daily_pv.h5")
DEFAULT_OUT       = Path("git_ignore_folder/combined_factors_df.parquet")
DEFAULT_IC_THRESH = 0.02   # minimum |IC| to include a factor
DEFAULT_MAX_NAN   = 0.50   # skip factor if > 50% NaN
DEFAULT_MAX_FACTS = 50     # top N by |IC| after dedup
# ─────────────────────────────────────────────────────────────────────────────


def _compute_ic(factor_series: pd.Series, label_series: pd.Series) -> float:
    """Compute rank IC between factor and forward label over shared index."""
    try:
        aligned = pd.DataFrame({"f": factor_series, "l": label_series}).dropna()
        if len(aligned) < 200:
            return float("nan")
        return float(aligned["f"].rank().corr(aligned["l"].rank(), method="pearson"))
    except Exception:
        return float("nan")


def _read_factor_name(ws_dir: Path) -> str | None:
    fp = ws_dir / "factor.py"
    if not fp.exists():
        return None
    try:
        for line in fp.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip().startswith("def calculate_"):
                return line.strip()[len("def calculate_"):].split("(")[0]
    except Exception:
        pass
    return None


def _load_result(ws_dir: Path) -> pd.DataFrame | None:
    rp = ws_dir / "result.h5"
    if not rp.exists() or rp.stat().st_size < 1000:
        return None
    try:
        df = pd.read_hdf(str(rp), key="data")
        return df
    except Exception:
        return None


def _make_label(pv: pd.DataFrame) -> pd.Series:
    """Main-project label: close[t+3]/close[t-1] - 1.

    与 config/data.yaml 中 `label: Ref($close_qfq, -3)/Ref($close_qfq, 1) - 1`
    在语义上一致（daily_pv.h5 存储的 `$close` 通常即为后复权价；若实为原始价，
    仍保持与主工程 label 的相对形态一致）。
    """
    close = pv["$close"].copy()
    fwd = (
        close.groupby(level="instrument", group_keys=False)
        .transform(lambda s: s.shift(-3) / s.shift(1) - 1)
    )
    return fwd.rename("LABEL0")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export top RDAgent factors to combined parquet")
    ap.add_argument("--ws-root",       default=str(DEFAULT_WS_ROOT))
    ap.add_argument("--pv-path",       default=str(DEFAULT_PV_PATH),
                    help="Path to daily_pv.h5 used to compute label")
    ap.add_argument("--out",           default=str(DEFAULT_OUT))
    ap.add_argument("--ic-threshold",  type=float, default=DEFAULT_IC_THRESH)
    ap.add_argument("--max-nan-ratio", type=float, default=DEFAULT_MAX_NAN)
    ap.add_argument("--max-factors",   type=int,   default=DEFAULT_MAX_FACTS)
    ap.add_argument("--dry-run",       action="store_true")
    args = ap.parse_args()

    ws_root  = Path(args.ws_root)
    pv_path  = Path(args.pv_path)
    out_path = Path(args.out)

    if not ws_root.exists():
        print(f"[export] ERROR: workspace root not found: {ws_root}", file=sys.stderr)
        sys.exit(1)
    if not pv_path.exists():
        print(f"[export] ERROR: daily_pv.h5 not found: {pv_path}", file=sys.stderr)
        sys.exit(1)

    # ── Load daily_pv and compute label ─────────────────────────────────────
    print(f"[export] Loading daily_pv from {pv_path} ...")
    pv = pd.read_hdf(str(pv_path))
    label = _make_label(pv)
    print(f"[export] daily_pv shape: {pv.shape}, label non-null: {label.notna().sum():,}")

    # ── Scan workspaces ──────────────────────────────────────────────────────
    ws_dirs = [d for d in sorted(ws_root.iterdir()) if d.is_dir()]
    print(f"[export] Scanning {len(ws_dirs):,} workspace dirs ...")

    candidates: list[dict] = []
    seen_names: set[str] = set()
    skipped_nan = skipped_ic = skipped_noname = skipped_noh5 = 0

    for ws_dir in ws_dirs:
        # Must have result.h5
        result_df = _load_result(ws_dir)
        if result_df is None:
            skipped_noh5 += 1
            continue

        # Must have factor name
        name = _read_factor_name(ws_dir)
        if name is None:
            skipped_noname += 1
            continue

        # Deduplicate by name (keep first seen, with valid data)
        if name in seen_names:
            continue

        # Check NaN ratio
        nan_ratio = float(result_df.isna().mean().iloc[0])
        if nan_ratio > args.max_nan_ratio:
            skipped_nan += 1
            continue

        # Compute IC against label
        factor_col = result_df.iloc[:, 0]
        ic = _compute_ic(factor_col, label)
        if np.isnan(ic) or abs(ic) < args.ic_threshold:
            skipped_ic += 1
            continue

        seen_names.add(name)
        candidates.append({
            "name": name,
            "ic": ic,
            "nan_ratio": nan_ratio,
            "ws_dir": str(ws_dir),
            "result_df": result_df,
        })

    print(f"\n[export] Scan complete:")
    print(f"  No result.h5:          {skipped_noh5:,}")
    print(f"  No factor name:        {skipped_noname:,}")
    print(f"  Duplicate name:        {len(ws_dirs) - skipped_noh5 - skipped_noname - skipped_nan - skipped_ic - len(candidates):,}")
    print(f"  Skipped (too many NaN): {skipped_nan:,}")
    print(f"  Skipped (IC too low):  {skipped_ic:,}")
    print(f"  Qualifying candidates: {len(candidates)}")

    # ── Rank by |IC| and select top N ────────────────────────────────────────
    candidates.sort(key=lambda x: abs(x["ic"]), reverse=True)
    selected = candidates[: args.max_factors]

    print(f"\n[export] Top {len(selected)} factors (sorted by |IC|):")
    print(f"  {'Factor Name':<32} {'IC':>8} {'NaN%':>7}")
    print(f"  {'-'*50}")
    for c in selected:
        print(f"  {c['name']:<32} {c['ic']:>8.4f} {c['nan_ratio']*100:>6.1f}%")

    if args.dry_run or not selected:
        print("\n[export] Dry-run or no factors found. Nothing written.")
        return

    # ── Merge and write parquet ───────────────────────────────────────────────
    print(f"\n[export] Merging {len(selected)} factors ...")
    merged: pd.DataFrame | None = None
    for c in selected:
        col_name = c["name"]
        df = c["result_df"].rename(columns={c["result_df"].columns[0]: col_name})
        if merged is None:
            merged = df
        else:
            merged = merged.join(df, how="outer")

    if merged is None:
        print("[export] Nothing to write.", file=sys.stderr)
        return

    merged = merged.sort_index()
    print(f"[export] Merged shape: {merged.shape}")
    print(f"[export] Date range: {merged.index.get_level_values('datetime').min()} ~ "
          f"{merged.index.get_level_values('datetime').max()}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(str(out_path))
    print(f"\n[export] Written to {out_path}")

    # ── Write summary JSON ────────────────────────────────────────────────────
    summary_path = out_path.with_suffix(".json")
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_factors": len(selected),
        "shape": list(merged.shape),
        "factors": [
            {"name": c["name"], "ic": round(c["ic"], 6),
             "nan_ratio": round(c["nan_ratio"], 4)}
            for c in selected
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[export] Summary written to {summary_path}")
    print(f"\n[export] Done. {len(selected)} factors exported.")


if __name__ == "__main__":
    main()
