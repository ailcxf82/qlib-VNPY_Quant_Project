from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from .code_utils import qlib_to_rqalpha

logger = logging.getLogger(__name__)


@dataclass
class PredictionBook:
    """按交易日存储信号：{date: {rq_code: score}}"""

    by_date: Dict[pd.Timestamp, Dict[str, float]]

    def get(self, dt: pd.Timestamp) -> Dict[str, float]:
        return self.by_date.get(pd.Timestamp(dt).normalize(), {})


def _calendar(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    try:
        from qlib.data import D  # type: ignore

        cal = D.calendar(start_time=start, end_time=end, freq="day")
        cal = pd.to_datetime(list(cal)).normalize()
        return pd.DatetimeIndex(cal).sort_values().unique()
    except Exception:
        return pd.bdate_range(start=start, end=end).normalize()


def _reverse_shift_to_signal_date_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """
    兼容预测文件的 datetime 已经被 shift 成 trade_date 的情况（_meta_shifted_next_day=1）。
    对于 RQAlpha 的 T+1（next_bar）回测，策略应在 signal_date=t 下单，t+1 成交；
    若预测文件已经是 trade_date，再配合 next_bar 会变成隐性 T+2。
    因此：检测到 shift 标记时，将 trade_date 反向还原为前一交易日(signal_date)。
    """
    if "_meta_shifted_next_day" not in df.columns:
        return df
    try:
        shifted = int(pd.to_numeric(df["_meta_shifted_next_day"], errors="coerce").fillna(0).max()) == 1
    except Exception:
        shifted = False
    if not shifted:
        return df
    if "datetime" not in df.columns:
        return df

    dts = pd.to_datetime(df["datetime"]).dt.normalize()
    if dts.empty:
        return df

    unique_dts = pd.Index(dts.unique()).sort_values()
    min_dt = pd.Timestamp(unique_dts.min()).normalize()
    max_dt = pd.Timestamp(unique_dts.max()).normalize()
    cal = _calendar(min_dt - pd.Timedelta(days=60), max_dt + pd.Timedelta(days=5))
    cal_list = list(cal)
    import bisect

    mapping_prev = {}
    for d in unique_dts:
        i = bisect.bisect_left(cal_list, pd.Timestamp(d))
        if i <= 0:
            mapping_prev[pd.Timestamp(d)] = None
        else:
            mapping_prev[pd.Timestamp(d)] = pd.Timestamp(cal_list[i - 1])

    prev_dt = dts.map(lambda x: mapping_prev.get(pd.Timestamp(x), None))
    before = len(df)
    df2 = df.copy()
    df2["datetime"] = pd.to_datetime(prev_dt)
    df2 = df2.dropna(subset=["datetime"])
    after = len(df2)
    logger.info(
        "MSA 预测文件检测到 _meta_shifted_next_day=1，已将 trade_date 反向还原为 signal_date（丢弃=%d）",
        before - after,
    )
    return df2


def load_prediction_csv(path: str, *, score_col: str = "final") -> PredictionBook:
    """
    支持两种格式：
    1) datetime, instrument, final
    2) datetime, rq_code, final
    """
    df = pd.read_csv(path, dtype={"instrument": str, "rq_code": str})
    if "datetime" not in df.columns:
        raise ValueError(f"预测文件缺少 datetime 列: {path}")
    if score_col not in df.columns:
        raise ValueError(f"预测文件缺少 {score_col} 列: {path}")

    df["datetime"] = pd.to_datetime(df["datetime"]).dt.normalize()
    # 若预测文件 datetime 为 trade_date（已 shift），这里反向还原为 signal_date，避免回测侧变成隐性 T+2
    df = _reverse_shift_to_signal_date_if_needed(df)

    if "rq_code" in df.columns and df["rq_code"].notna().any():
        df["rq_code"] = df["rq_code"].astype(str).str.strip()
    elif "instrument" in df.columns:
        df["rq_code"] = df["instrument"].astype(str).str.strip().apply(qlib_to_rqalpha)
    else:
        raise ValueError(f"预测文件缺少 instrument 或 rq_code 列: {path}")

    # 丢弃空 code 或空分数
    df = df.dropna(subset=["rq_code", score_col])

    by_date: Dict[pd.Timestamp, Dict[str, float]] = {}
    for dt, g in df.groupby("datetime"):
        # 同一日同一证券若重复，取最后一个
        s = pd.Series(g[score_col].values, index=g["rq_code"].values)
        by_date[pd.Timestamp(dt).normalize()] = s.groupby(level=0).last().to_dict()

    logger.info("加载预测文件: %s，交易日=%d", path, len(by_date))
    return PredictionBook(by_date=by_date)


def topk(signals: Dict[str, float], k: int) -> Dict[str, float]:
    if not signals or k <= 0:
        return {}
    items = sorted(signals.items(), key=lambda x: x[1], reverse=True)[:k]
    return dict(items)


