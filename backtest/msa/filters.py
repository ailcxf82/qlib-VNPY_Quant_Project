from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set

import pandas as pd

from .code_utils import is_kcb_or_bj, rqalpha_to_tushare
from .tushare_client import TushareClient

logger = logging.getLogger(__name__)


@dataclass
class FilterConfig:
    exclude_kcb_bj: bool = True
    exclude_st: bool = True
    min_list_days: int = 360
    pb_min: Optional[float] = None
    pb_max: Optional[float] = None
    exclude_recent_limitup_days: int = 0  # 近N日出现涨停则剔除（需要 Tushare）


def _safe_set(x: Iterable[str]) -> Set[str]:
    return set([str(i).strip() for i in x if str(i).strip()])


def apply_basic_filters(
    rq_codes: List[str],
    dt: pd.Timestamp,
    cfg: FilterConfig,
    ts: Optional[TushareClient],
) -> List[str]:
    """
    对候选 rqalpha 代码做过滤。
    - 不依赖 Tushare 的规则：科创/北交（按代码前缀）
    - 依赖 Tushare 的规则：ST、上市天数、PB、近N日涨停
    """
    dt = pd.Timestamp(dt).normalize()
    ts_codes = [rqalpha_to_tushare(c) for c in rq_codes]

    keep: List[str] = []
    for rq, ts_code in zip(rq_codes, ts_codes):
        if cfg.exclude_kcb_bj and is_kcb_or_bj(ts_code):
            continue
        keep.append(rq)

    if ts is None:
        # 没有 tushare 时只能做最基本过滤
        return keep

    try:
        sb = ts.stock_basic()
    except Exception as e:
        logger.warning("读取 stock_basic 失败，跳过 Tushare 基础过滤: %s", e)
        return keep

    sb = sb.copy()
    sb["ts_code"] = sb["ts_code"].astype(str)
    sb_map = sb.set_index("ts_code").to_dict(orient="index")

    # 上市天数、ST（通过 name 粗略判断）
    out: List[str] = []
    for rq in keep:
        ts_code = rqalpha_to_tushare(rq)
        info = sb_map.get(ts_code)
        if not info:
            out.append(rq)
            continue

        name = str(info.get("name", ""))
        if cfg.exclude_st and ("ST" in name or "*ST" in name or "退" in name):
            continue

        list_date = str(info.get("list_date", "")).strip()
        if list_date and cfg.min_list_days and cfg.min_list_days > 0:
            try:
                ld = pd.to_datetime(list_date)
                if (dt - ld).days < cfg.min_list_days:
                    continue
            except Exception:
                pass

        out.append(rq)

    # PB 过滤
    if cfg.pb_min is not None or cfg.pb_max is not None:
        try:
            db = ts.daily_basic(dt)
            db["ts_code"] = db["ts_code"].astype(str)
            pb_map = db.set_index("ts_code")["pb"].to_dict()
        except Exception as e:
            logger.warning("读取 daily_basic 失败，跳过 PB 过滤: %s", e)
            pb_map = {}

        out2: List[str] = []
        for rq in out:
            ts_code = rqalpha_to_tushare(rq)
            pb = pb_map.get(ts_code)
            if pb is None:
                out2.append(rq)
                continue
            try:
                pb_f = float(pb)
            except Exception:
                out2.append(rq)
                continue
            if cfg.pb_min is not None and pb_f <= cfg.pb_min:
                continue
            if cfg.pb_max is not None and pb_f >= cfg.pb_max:
                continue
            out2.append(rq)
        out = out2

    # 近 N 日涨停过滤（策略2用）
    if cfg.exclude_recent_limitup_days and cfg.exclude_recent_limitup_days > 0:
        limitup: Set[str] = set()
        for i in range(cfg.exclude_recent_limitup_days):
            d = dt - pd.Timedelta(days=i)
            try:
                ll = ts.limit_list(d, limit_type="U")
                if not ll.empty and "ts_code" in ll.columns:
                    limitup |= _safe_set(ll["ts_code"].tolist())
            except Exception:
                # 某些日期非交易日/接口失败：忽略
                continue

        out = [rq for rq in out if rqalpha_to_tushare(rq) not in limitup]

    return out


