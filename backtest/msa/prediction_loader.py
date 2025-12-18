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


