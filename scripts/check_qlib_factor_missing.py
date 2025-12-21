"""
检查 qlib 数据源中指定字段在某个时间段内是否为空（全 NaN）/缺失率。

用法示例（Windows/conda 环境）：
  c:\\ProgramData\\miniconda3\\envs\\qlib_zhengshi\\python.exe scripts/check_qlib_factor_missing.py ^
    --config config/data.yaml ^
    --market csi101 ^
    --start 2021-01-01 ^
    --end 2025-11-30 ^
    --sample-size 200

说明：
- 为了避免全市场全量拉取过慢，默认对 market 里抽样 sample_size 只股票做统计。
- 如果你确信机器性能足够，可以把 sample-size 设大（如 2000），甚至用 --all-instruments。
"""

from __future__ import annotations

import argparse
import random
from typing import List, Dict

import pandas as pd

from utils import load_yaml_config


FIELDS_DEFAULT = [
    "$buy_elg_amount",
    "$buy_lg_amount",
    "$buy_md_amount",
    "$buy_sm_amount",
    "$short_balance",
    "$margin_balance",
    "Ref($margin_balance, 1)/$margin_balance - 1",
    "$net_mf_amount",
    "$eps_growth",
    "$roe/Mean($roe, 4) - 1",
]


def parse_args():
    p = argparse.ArgumentParser(description="检查 qlib 字段缺失情况（是否全为空）")
    p.add_argument("--config", type=str, default="config/data.yaml", help="data.yaml 路径（含 qlib.provider_uri）")
    p.add_argument("--market", type=str, default="csi101", help="股票池/市场别名，如 csi101/csi300/ETF")
    p.add_argument("--start", type=str, default="2021-01-01")
    p.add_argument("--end", type=str, default="2025-11-30")
    p.add_argument("--sample-size", type=int, default=200, help="抽样股票数量（默认200）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--all-instruments", action="store_true", help="对 market 全量股票统计（可能较慢）")
    p.add_argument("--fields", type=str, default=",".join(FIELDS_DEFAULT), help="逗号分隔字段/表达式列表")
    return p.parse_args()


def _year_stats(s: pd.Series) -> pd.DataFrame:
    if s.empty:
        return pd.DataFrame(columns=["year", "rows", "non_null", "non_null_pct"])
    dt = s.index.get_level_values("datetime")
    df = pd.DataFrame({"v": s.values}, index=pd.Index(dt, name="datetime"))
    df["year"] = pd.to_datetime(df.index).year
    g = df.groupby("year")["v"]
    out = pd.DataFrame(
        {
            "rows": g.size(),
            "non_null": g.apply(lambda x: x.notna().sum()),
        }
    )
    out["non_null_pct"] = (out["non_null"] / out["rows"] * 100).round(2)
    out = out.reset_index()
    return out


def main():
    args = parse_args()
    cfg = load_yaml_config(args.config)
    qcfg = cfg.get("qlib", {})
    provider_uri = qcfg.get("provider_uri")
    region = qcfg.get("region", "cn")
    if not provider_uri:
        raise ValueError("config 中缺少 qlib.provider_uri")

    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    print("provider_uri:", provider_uri)
    print("region:", region)
    print("market:", args.market)
    print("start/end:", args.start, args.end)
    print("fields:", fields)

    import qlib
    from qlib.data import D

    qlib.init(provider_uri=provider_uri, region=region, expression_cache=None)

    # 获取股票列表
    market_cfg = D.instruments(args.market)
    inst_list = D.list_instruments(instruments=market_cfg, as_list=True)
    inst_list = [str(x) for x in inst_list]
    # 清理后缀
    inst_list = [x.split(".", 1)[0] for x in inst_list]
    inst_list = sorted(set(inst_list))
    print("instruments total:", len(inst_list))

    # 抽样
    if args.all_instruments:
        sample = inst_list
    else:
        random.seed(args.seed)
        if len(inst_list) <= args.sample_size:
            sample = inst_list
        else:
            sample = random.sample(inst_list, args.sample_size)
    print("instruments used:", len(sample))

    # 逐字段拉取（避免一次性拉过大）
    summary_rows: List[Dict] = []
    for field in fields:
        try:
            panel = D.features(
                instruments=sample,
                fields=[field],
                start_time=args.start,
                end_time=args.end,
                freq="day",
            )
            s = panel.iloc[:, 0]
        except Exception as e:
            print("=" * 80)
            print("[ERROR] field:", field)
            print("exception:", repr(e))
            summary_rows.append(
                {
                    "field": field,
                    "status": "error",
                    "rows": 0,
                    "non_null": 0,
                    "non_null_pct": 0.0,
                }
            )
            continue

        rows = len(s)
        non_null = int(s.notna().sum())
        pct = round(non_null / rows * 100, 4) if rows else 0.0
        all_nan = non_null == 0
        print("=" * 80)
        print("field:", field)
        print("rows:", rows, "non_null:", non_null, "non_null_pct:", pct, "ALL_NAN:", all_nan)
        # 按年输出（只显示 2021+）
        ys = _year_stats(s)
        if not ys.empty:
            ys = ys[ys["year"] >= 2021]
            print(ys.to_string(index=False))

        summary_rows.append(
            {
                "field": field,
                "status": "ok",
                "rows": rows,
                "non_null": non_null,
                "non_null_pct": pct,
                "all_nan": all_nan,
            }
        )

    print("=" * 80)
    print("SUMMARY (sorted by non_null_pct asc)")
    sdf = pd.DataFrame(summary_rows)
    if not sdf.empty and "non_null_pct" in sdf.columns:
        sdf = sdf.sort_values(["status", "non_null_pct"], ascending=[True, True])
    print(sdf.to_string(index=False))


if __name__ == "__main__":
    main()




