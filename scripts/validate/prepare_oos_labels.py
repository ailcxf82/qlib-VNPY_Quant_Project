"""离线生成 L2 验证用的 OOS 标签 parquet。

本脚本是 L2 validator 的**离线前置步骤**：跑一次生成 label，后续所有 ic_check 直接读取，
避免在 check 内部触发 qlib.init（保持 L2 check 纯函数化 / 易单测）。

Label 表达式与项目训练端保持一致：
    ``Ref($close_qfq, -5) / Ref($close_qfq, 1) - 1``   （5 日前向收益率）

**典型用法**：

    python -m scripts.validate.prepare_oos_labels \
        --universe csi300 \
        --start 2025-01-01 \
        --end 2026-04-07

默认输出：``factor_validation/data/oos_labels_<universe>.parquet``。

Profile 对齐：``factor_validation/profiles/*.yaml.data_sources.label_parquet`` 必须指向
同一路径，orchestrator 会 resolve 并注入 CheckContext。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


logger = logging.getLogger("prepare_oos_labels")

DEFAULT_PROVIDER_URI = r"D:/qlib_data/qlib_data"
DEFAULT_REGION = "cn"
DEFAULT_LABEL_EXPR = "Ref($close_qfq, -5) / Ref($close_qfq, 1) - 1"
DEFAULT_OUT_DIR = _PROJECT_ROOT / "factor_validation" / "data"


# ------------------------------------------------------------------ core

def prepare_labels(
    *,
    universe: str,
    start: str,
    end: str,
    provider_uri: str = DEFAULT_PROVIDER_URI,
    region: str = DEFAULT_REGION,
    label_expr: str = DEFAULT_LABEL_EXPR,
    out_path: Path | None = None,
) -> Path:
    """调 qlib 拉 label，写 parquet 到 ``out_path``（默认 ``factor_validation/data/<...>``）。"""
    import qlib
    from qlib.data import D

    if not qlib.get_module_logger("qlib").handlers:
        # 抑制 qlib 默认 log 噪音，但仍允许 warning/error
        logging.getLogger("qlib").setLevel(logging.WARNING)

    qlib.init(provider_uri=provider_uri, region=region)
    logger.info(
        "qlib.init OK; universe=%s start=%s end=%s provider_uri=%s",
        universe,
        start,
        end,
        provider_uri,
    )

    instruments = D.instruments(market=universe)
    df = D.features(
        instruments,
        fields=[label_expr],
        start_time=start,
        end_time=end,
        freq="day",
        disk_cache=0,
    )
    if df is None or df.empty:
        raise RuntimeError(
            f"qlib 返回空 label；universe={universe} window=[{start}, {end}]"
        )
    # 规范列名 → "label"；index 规范为 (datetime, instrument)
    df = df.rename(columns={label_expr: "label"})[["label"]]
    if df.index.names != ["datetime", "instrument"]:
        # qlib 默认 (instrument, datetime)；我们统一 (datetime, instrument)
        df = df.reorder_levels(["datetime", "instrument"])
    df = df.sort_index()
    df["label"] = df["label"].astype("float64")

    if out_path is None:
        out_path = DEFAULT_OUT_DIR / f"oos_labels_{universe}.parquet"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 原子写
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_parquet(tmp, engine="pyarrow")
    tmp.replace(out_path)

    n_total = len(df)
    n_non_null = int(df["label"].notna().sum())
    n_instr = df.index.get_level_values("instrument").nunique()
    dt = df.index.get_level_values("datetime")
    logger.info(
        "label parquet 写完：%s  shape=%s non_null=%d (%.4f)  n_instr=%d  window=[%s, %s]",
        out_path,
        df.shape,
        n_non_null,
        n_non_null / max(1, n_total),
        n_instr,
        pd.Timestamp(dt.min()).date(),
        pd.Timestamp(dt.max()).date(),
    )
    return out_path


# ------------------------------------------------------------------ CLI

def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate L2 OOS label parquet via qlib (one-time preparation)"
    )
    p.add_argument("--universe", default="csi300")
    p.add_argument("--start", default="2025-01-01", help="OOS 起始日期，YYYY-MM-DD")
    p.add_argument("--end", default="2026-04-07", help="OOS 结束日期，YYYY-MM-DD（含）")
    p.add_argument(
        "--provider-uri",
        default=DEFAULT_PROVIDER_URI,
        help="qlib provider_uri；默认 D:/qlib_data/qlib_data",
    )
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument(
        "--label-expr",
        default=DEFAULT_LABEL_EXPR,
        help="qlib 表达式，默认 5 日前向收益率",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出 parquet 路径；默认 factor_validation/data/oos_labels_<universe>.parquet",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    out = prepare_labels(
        universe=args.universe,
        start=args.start,
        end=args.end,
        provider_uri=args.provider_uri,
        region=args.region,
        label_expr=args.label_expr,
        out_path=args.out,
    )
    print(str(out))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
