"""
按模型列输出 ICIR 汇总（支持指定 ic_lgb / ic_gru 等）。
"""
from __future__ import annotations

import argparse
import os
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="输出模型 IC/ICIR 汇总")
    ap.add_argument("--log", type=str, default="data/logs/training_metrics.csv")
    ap.add_argument(
        "--cols",
        type=str,
        default="",
        help="逗号分隔列名，如 ic_lgb,ic_gru；为空则自动检测 ic_*",
    )
    args = ap.parse_args()

    if not os.path.exists(args.log):
        raise FileNotFoundError(f"训练日志不存在: {args.log}")

    df = pd.read_csv(args.log)
    if args.cols.strip():
        cols = [c.strip() for c in args.cols.split(",") if c.strip()]
    else:
        cols = [c for c in df.columns if c.startswith("ic_")]

    print("=" * 80)
    print(f"IC/ICIR 汇总: {args.log}")
    print("=" * 80)
    rows = []
    for c in cols:
        if c not in df.columns:
            print(f"❌ 列不存在: {c}")
            continue
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        n = len(s)
        if n == 0:
            print(f"{c}: 无有效 IC")
            continue
        mean_ic = float(s.mean())
        std_ic = float(s.std(ddof=0)) if n >= 2 else 0.0
        icir = mean_ic / (std_ic + 1e-12) if n >= 2 else float("nan")
        rows.append(
            {
                "col": c,
                "n": n,
                "mean_ic": mean_ic,
                "std_ic": std_ic,
                "icir": icir,
            }
        )
        if n >= 2:
            print(f"{c}: n={n} mean={mean_ic:.6f} std={std_ic:.6f} ICIR={icir:.6f}")
        else:
            print(f"{c}: n={n} ic={mean_ic:.6f}（窗口不足，无法计算 ICIR）")

    if rows:
        out = "data/logs/icir_report.csv"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"已保存: {out}")


if __name__ == "__main__":
    main()


