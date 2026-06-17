"""
验证 Tushare Pro 的 income / fina_indicator / fina_audit 接口：
- 参数是否能拿到数据（行数>0）
- 关键字段是否存在/是否大量为空

运行前准备：
1) pip install tushare
2) 设置环境变量 TUSHARE_TOKEN

参考文档：
https://tushare.pro/document
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Dict, Any

import pandas as pd

from .tushare_client import TushareClient


def _print_df_summary(name: str, df: Optional[pd.DataFrame], prefer_cols: List[str]):
    print("=" * 80)
    print(f"[{name}]")
    if df is None:
        print("df=None")
        return
    print(f"rows={len(df)}, cols={len(df.columns)}")
    if df.empty:
        print("EMPTY")
        return
    cols = list(df.columns)
    print("columns:", cols[:40], "..." if len(cols) > 40 else "")

    # 重点字段非空情况
    hit = [c for c in prefer_cols if c in df.columns]
    if hit:
        print("key non-null ratio:")
        for c in hit:
            nn = df[c].notna().mean()
            print(f"  - {c}: {nn:.2%}")
    else:
        print("key fields not found in response (this may be OK; check columns list above).")

    # 打印一行样例
    print("sample row:")
    print(df.head(1).to_string(index=False))


def _call_with_fallbacks(call, candidates: List[Dict[str, Any]]) -> pd.DataFrame:
    last_err = None
    for kwargs in candidates:
        try:
            return call(**kwargs)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"all parameter patterns failed, last_err={last_err}")


def parse_args():
    p = argparse.ArgumentParser(description="验证 tushare income/fina_indicator/fina_audit 是否有值")
    p.add_argument("--ts-code", type=str, default="000001.SZ", help="示例证券代码（Tushare 格式）")
    p.add_argument("--start-date", type=str, default=None, help="YYYYMMDD，可选")
    p.add_argument("--end-date", type=str, default=None, help="YYYYMMDD，可选")
    p.add_argument("--period", type=str, default=None, help="YYYYMMDD（季末/年末），可选")
    p.add_argument("--cache-dir", type=str, default="data/tushare_cache", help="缓存目录")
    return p.parse_args()


def main():
    args = parse_args()
    ts = TushareClient.try_create(cache_dir=args.cache_dir)
    if ts is None:
        print("ERROR: TushareClient 初始化失败。可能原因：")
        print("  1) 未安装 tushare（请先 pip install tushare）")
        print("  2) 未配置 token：环境变量 TUSHARE_TOKEN 或 config/secrets.yaml")
        print("  3) token 无效/接口异常")
        return 2

    ts_code = args.ts_code.strip()

    # 多种参数组合尝试（因为你提到可能参数为空）
    common1 = {"ts_code": ts_code, "period": args.period, "fields": None}
    common2 = {"ts_code": ts_code, "start_date": args.start_date, "end_date": args.end_date, "fields": None}
    common3 = {"ts_code": ts_code, "fields": None}
    patterns = [common1, common2, common3]

    income_df = _call_with_fallbacks(ts.income, patterns)
    indi_df = _call_with_fallbacks(ts.fina_indicator, patterns)
    audit_df = _call_with_fallbacks(ts.fina_audit, patterns)

    _print_df_summary(
        "income",
        income_df,
        prefer_cols=["ts_code", "end_date", "ann_date", "total_revenue", "n_income", "n_income_attr_p", "net_profit"],
    )
    _print_df_summary(
        "fina_indicator",
        indi_df,
        prefer_cols=["ts_code", "end_date", "roa", "roe", "netprofit_yoy", "profit_to_gr", "eps"],
    )
    _print_df_summary(
        "fina_audit",
        audit_df,
        prefer_cols=["ts_code", "end_date", "ann_date", "audit_result", "audit_agency", "audit_sign"],
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


