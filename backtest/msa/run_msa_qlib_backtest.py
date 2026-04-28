"""
Qlib-driven MSA multi-strategy backtest.

This script reuses the stock-selection logic from ``run_msa_signal.py`` and
uses qlib daily prices to evaluate close-to-close portfolio performance.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backtest.msa.filters import FilterConfig
from backtest.msa.prediction_loader import PredictionBook, load_prediction_csv
from backtest.msa.run_msa_signal import (
    SubStrategyConfig,
    _find_latest_prediction,
    _infer_pred_dates_are,
    _merge_allocations,
    _prev_trading_day,
    _select_for_substrategy,
    _turnover,
)
from backtest.msa.tushare_client import TushareClient

logger = logging.getLogger(__name__)


@dataclass
class RebalanceEvent:
    signal_date: pd.Timestamp
    trade_date: pd.Timestamp
    pred_dt: pd.Timestamp
    target_weights: Dict[str, float]
    picks1: List[str]
    picks2: List[str]


def _resolve_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    p = str(path).strip()
    if p.startswith("@"):
        p = p[1:]
    p = p.replace("/", os.sep)
    if os.path.isabs(p):
        return p
    return os.path.join(_PROJECT_ROOT, p)


def _rqalpha_to_qlib(code: str) -> str:
    s = str(code).strip()
    if s.endswith(".XSHG"):
        return s.replace(".XSHG", ".SH")
    if s.endswith(".XSHE"):
        return s.replace(".XSHE", ".SZ")
    return s


def _qlib_to_rqalpha(code: str) -> str:
    s = str(code).strip()
    if s.startswith("SH"):
        return f"{s[2:].zfill(6)}.XSHG"
    if s.startswith("SZ"):
        return f"{s[2:].zfill(6)}.XSHE"
    if s.endswith(".SH"):
        return s.replace(".SH", ".XSHG")
    if s.endswith(".SZ"):
        return s.replace(".SZ", ".XSHE")
    return s


def _normalize_full(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, float(v)) for v in weights.values())
    if total <= 0:
        return {}
    return {k: max(0.0, float(v)) / total for k, v in weights.items()}


def _load_yaml_config(path: str) -> Dict:
    import yaml

    with open(path, "r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp)
    return data if isinstance(data, dict) else {}


def _init_qlib(data_config: str) -> None:
    cfg = _load_yaml_config(data_config)
    qlib_cfg = cfg.get("qlib", {}) if isinstance(cfg, dict) else {}
    provider_uri = qlib_cfg.get("provider_uri")
    region = qlib_cfg.get("region", "cn")
    if not provider_uri:
        raise ValueError(f"{data_config} 缺少 qlib.provider_uri")

    import qlib

    logger.info("初始化 qlib: provider_uri=%s, region=%s", provider_uri, region)
    qlib.init(provider_uri=provider_uri, region=region, expression_cache=None)
    try:
        import qlib.data  # type: ignore  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "当前 Python 环境中的 qlib 不完整，无法导入 qlib.data；"
            "请确认已安装 pyqlib 并使用正确的 conda/python 环境。"
        ) from exc


def _load_backtest_window(config_path: Optional[str]) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    if not config_path:
        return None, None
    path = _resolve_path(config_path) or config_path
    if not os.path.exists(path):
        return None, None
    cfg = _load_yaml_config(path)
    base = cfg.get("base", {}) if isinstance(cfg, dict) else {}
    if not isinstance(base, dict):
        return None, None
    start = base.get("start_date")
    end = base.get("end_date")
    start_ts = pd.Timestamp(start).normalize() if start else None
    end_ts = pd.Timestamp(end).normalize() if end else None
    return start_ts, end_ts


def _resolve_backtest_window(
    args: argparse.Namespace,
    pred_dates: Iterable[pd.Timestamp],
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    dates = [pd.Timestamp(d).normalize() for d in pred_dates]
    cfg_start, cfg_end = _load_backtest_window(getattr(args, "rqalpha_config", None))
    pred_start = min(dates) if dates else None
    pred_end = max(dates) if dates else None

    start = (
        pd.Timestamp(args.start_date).normalize()
        if getattr(args, "start_date", None)
        else (pred_start if pred_start is not None else cfg_start)
    )
    end = (
        pd.Timestamp(args.end_date).normalize()
        if getattr(args, "end_date", None)
        else (pred_end if pred_end is not None else cfg_end)
    )
    if start is None or end is None:
        raise ValueError("预测文件没有有效日期，且配置中缺少 base.start_date/base.end_date")
    return start, end


def _qlib_calendar(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    from qlib.data import D  # type: ignore

    cal = D.calendar(start_time=pd.Timestamp(start), end_time=pd.Timestamp(end), freq="day")
    return pd.DatetimeIndex(pd.to_datetime(list(cal)).normalize()).sort_values().unique()


def _next_from_calendar(cal: pd.DatetimeIndex, dt: pd.Timestamp) -> Optional[pd.Timestamp]:
    pos = cal.searchsorted(pd.Timestamp(dt).normalize(), side="right")
    if pos >= len(cal):
        return None
    return pd.Timestamp(cal[pos]).normalize()


def _prev_from_calendar(cal: pd.DatetimeIndex, dt: pd.Timestamp) -> Optional[pd.Timestamp]:
    pos = cal.searchsorted(pd.Timestamp(dt).normalize(), side="left") - 1
    if pos < 0:
        return None
    return pd.Timestamp(cal[pos]).normalize()


class QlibPriceFetcher:
    """Compatibility adapter for ``run_msa_signal._select_for_substrategy``."""

    def __init__(self, calendar: pd.DatetimeIndex, price_field: str = "$close_qfq"):
        self.calendar = calendar
        self.price_field = price_field
        self._history_cache: Dict[Tuple[str, pd.Timestamp, int], Optional[pd.Series]] = {}

    def get_close_history(self, rq_code: str, end_dt: pd.Timestamp, n: int) -> Optional[pd.Series]:
        key = (str(rq_code), pd.Timestamp(end_dt).normalize(), int(n))
        if key in self._history_cache:
            return self._history_cache[key]

        end_dt = pd.Timestamp(end_dt).normalize()
        pos = self.calendar.searchsorted(end_dt, side="right") - 1
        if pos < 0:
            self._history_cache[key] = None
            return None
        start_pos = max(0, pos - max(int(n) * 3, int(n) + 5))
        start_dt = pd.Timestamp(self.calendar[start_pos]).normalize()
        qlib_code = _rqalpha_to_qlib(rq_code)
        try:
            prices = _load_qlib_prices([rq_code], start_dt, end_dt, self.price_field)
            if rq_code not in prices.columns:
                self._history_cache[key] = None
                return None
            s = prices[rq_code].dropna().tail(int(n))
            self._history_cache[key] = s if len(s) >= int(n) else None
            return self._history_cache[key]
        except Exception as exc:
            logger.debug("读取 qlib 历史收盘价失败: %s %s", qlib_code, exc)
            self._history_cache[key] = None
            return None

    def get_close(self, rq_code: str, dt: pd.Timestamp) -> Optional[float]:
        s = self.get_close_history(rq_code, pd.Timestamp(dt), 1)
        if s is None or s.empty:
            return None
        v = float(s.iloc[-1])
        return v if math.isfinite(v) and v > 0 else None


def _load_qlib_prices(
    rq_codes: Iterable[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    price_field: str = "$close_qfq",
) -> pd.DataFrame:
    from qlib.data import D  # type: ignore

    codes = sorted({str(c).strip() for c in rq_codes if str(c).strip()})
    if not codes:
        return pd.DataFrame()
    qlib_codes = [_rqalpha_to_qlib(c) for c in codes]
    fields = [price_field]
    fallback_field = "$close" if price_field != "$close" else "$close_qfq"

    def _fetch(field_list: List[str]) -> pd.DataFrame:
        return D.features(
            instruments=qlib_codes,
            fields=field_list,
            start_time=pd.Timestamp(start),
            end_time=pd.Timestamp(end),
            freq="day",
        )

    try:
        raw = _fetch(fields)
        field = price_field
    except Exception:
        raw = _fetch([fallback_field])
        field = fallback_field

    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("qlib D.features 返回结果不是 MultiIndex，无法解析价格矩阵")
    df = df.reset_index()
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.normalize()
    df["rq_code"] = df["instrument"].astype(str).map(_qlib_to_rqalpha)
    if field not in df.columns:
        numeric_cols = [c for c in df.columns if c not in {"datetime", "instrument", "rq_code"}]
        if not numeric_cols:
            return pd.DataFrame()
        field = numeric_cols[0]
    out = df.pivot_table(index="datetime", columns="rq_code", values=field, aggfunc="last")
    out = out.sort_index()
    return out.apply(pd.to_numeric, errors="coerce")


def _build_substrategies(args: argparse.Namespace, pred_csi101: Optional[str], pred_csi300: Optional[str]) -> Tuple[Optional[SubStrategyConfig], Optional[SubStrategyConfig]]:
    use101 = pred_csi101 is not None
    use300 = pred_csi300 is not None
    if use101 and use300:
        alloc1, alloc2 = _merge_allocations(args.alloc1, args.alloc2)
    elif use101:
        alloc1, alloc2 = _merge_allocations(args.alloc1, 0.0)
    else:
        alloc1, alloc2 = _merge_allocations(0.0, args.alloc2)

    s1 = None
    if use101:
        s1 = SubStrategyConfig(
            name="small_cap_csi101",
            allocation=alloc1,
            pred_path=str(pred_csi101),
            topk_pred=int(args.s1_topk),
            target_holdings=int(args.s1_hold),
            filter_cfg=FilterConfig(
                exclude_kcb_bj=True,
                exclude_st=True,
                min_list_days=int(args.s1_min_list_days),
                pb_min=None,
                pb_max=None,
                exclude_recent_limitup_days=0,
            ),
            selection_mode=str(args.msa_selection_mode),
            preselect_topm=int(args.msa_preselect_topm),
            vol_window=int(args.msa_vol_window),
            vol_max=args.msa_vol_max,
            vol_eps=float(args.msa_vol_eps),
            max_vol20=args.s1_max_vol20,
            max_vol60=args.s1_max_vol60,
            max_vol120=args.s1_max_vol120,
        )

    s2 = None
    if use300:
        s2 = SubStrategyConfig(
            name="value_csi300",
            allocation=alloc2,
            pred_path=str(pred_csi300),
            topk_pred=int(args.s2_topk),
            target_holdings=int(args.s2_hold),
            filter_cfg=FilterConfig(
                exclude_kcb_bj=True,
                exclude_st=True,
                min_list_days=int(args.s2_min_list_days),
                pb_min=float(args.s2_pb_min),
                pb_max=float(args.s2_pb_max),
                exclude_recent_limitup_days=int(args.s2_limitup_days),
            ),
            selection_mode=str(args.msa_selection_mode),
            preselect_topm=int(args.msa_preselect_topm),
            vol_window=int(args.msa_vol_window),
            vol_max=args.msa_vol_max,
            vol_eps=float(args.msa_vol_eps),
            max_vol20=args.s2_max_vol20,
            max_vol60=args.s2_max_vol60,
            max_vol120=args.s2_max_vol120,
        )
    return s1, s2


def _select_target_weights(
    *,
    s1: Optional[SubStrategyConfig],
    s2: Optional[SubStrategyConfig],
    book101: Optional[PredictionBook],
    book300: Optional[PredictionBook],
    pred_dt: pd.Timestamp,
    signal_date: pd.Timestamp,
    ts_client: Optional[TushareClient],
    price_fetcher: QlibPriceFetcher,
    args: argparse.Namespace,
) -> Tuple[Dict[str, float], List[str], List[str]]:
    picks1: List[str] = []
    picks2: List[str] = []
    w1: Dict[str, float] = {}
    w2: Dict[str, float] = {}

    if s1 is not None and book101 is not None:
        picks1, _sig1, _vols1, w1 = _select_for_substrategy(
            s1,
            pred_dt,
            signal_date,
            book101,
            ts_client,
            bundle_path="",
            price_fetcher=price_fetcher,
            vol_drop_if_missing=bool(args.vol_drop_if_missing),
            vol_source=str(args.vol_source),
            industry_cap=int(args.industry_cap),
            industry_level=str(args.industry_level),
        )
    if s2 is not None and book300 is not None:
        picks2, _sig2, _vols2, w2 = _select_for_substrategy(
            s2,
            pred_dt,
            signal_date,
            book300,
            ts_client,
            bundle_path="",
            price_fetcher=price_fetcher,
            vol_drop_if_missing=bool(args.vol_drop_if_missing),
            vol_source=str(args.vol_source),
            industry_cap=int(args.industry_cap),
            industry_level=str(args.industry_level),
        )

    merged: Dict[str, float] = {}
    for code, weight in {**w1, **w2}.items():
        merged[code] = merged.get(code, 0.0) + float(weight)
    return _normalize_full(merged), picks1, picks2


def build_rebalance_events(
    *,
    calendar: pd.DatetimeIndex,
    s1: Optional[SubStrategyConfig],
    s2: Optional[SubStrategyConfig],
    book101: Optional[PredictionBook],
    book300: Optional[PredictionBook],
    pred_dates_are: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    rebalance_interval_days: int,
    ts_client: Optional[TushareClient],
    price_fetcher: QlibPriceFetcher,
    args: argparse.Namespace,
) -> List[RebalanceEvent]:
    active_dates: Optional[set[pd.Timestamp]] = None
    for book in (book101, book300):
        if book is None:
            continue
        dates = {pd.Timestamp(d).normalize() for d in book.by_date.keys()}
        active_dates = dates if active_dates is None else active_dates & dates
    pred_dates = sorted(active_dates or [])
    events: List[RebalanceEvent] = []
    last_trade_date: Optional[pd.Timestamp] = None
    interval = max(1, int(rebalance_interval_days))

    for pred_dt in pred_dates:
        pred_dt = pd.Timestamp(pred_dt).normalize()
        if pred_dates_are == "trade_date":
            trade_date = pred_dt
            signal_date = _prev_from_calendar(calendar, trade_date) or _prev_trading_day(trade_date)
        else:
            signal_date = pred_dt
            trade_date = _next_from_calendar(calendar, signal_date)
            if trade_date is None:
                continue
        if trade_date < start or trade_date > end:
            continue
        if last_trade_date is not None and (trade_date - last_trade_date).days < interval:
            continue

        target, picks1, picks2 = _select_target_weights(
            s1=s1,
            s2=s2,
            book101=book101,
            book300=book300,
            pred_dt=pred_dt,
            signal_date=signal_date,
            ts_client=ts_client,
            price_fetcher=price_fetcher,
            args=args,
        )
        if not target:
            logger.warning("调仓日无有效目标权重，跳过: signal=%s trade=%s", signal_date.date(), trade_date.date())
            continue
        events.append(
            RebalanceEvent(
                signal_date=signal_date,
                trade_date=trade_date,
                pred_dt=pred_dt,
                target_weights=target,
                picks1=picks1,
                picks2=picks2,
            )
        )
        last_trade_date = trade_date
    return events


def run_weight_backtest(
    *,
    calendar: pd.DatetimeIndex,
    events: List[RebalanceEvent],
    prices: pd.DataFrame,
    initial_cash: float,
    cost_rate: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    event_by_date = {e.trade_date: e for e in events}
    nav = 1.0
    weights: Dict[str, float] = {}
    daily_rows: List[Dict[str, object]] = []
    rebalance_rows: List[Dict[str, object]] = []
    position_rows: List[Dict[str, object]] = []

    def _position_action(prev_weight: float, target_weight: float) -> str:
        eps = 1e-12
        if prev_weight <= eps and target_weight > eps:
            return "buy"
        if prev_weight > eps and target_weight <= eps:
            return "sell"
        if prev_weight > eps and target_weight > eps and abs(target_weight - prev_weight) <= eps:
            return "hold"
        return "rebalance"

    for i, dt in enumerate(calendar):
        dt = pd.Timestamp(dt).normalize()
        daily_ret = 0.0
        if i > 0 and weights:
            prev_dt = pd.Timestamp(calendar[i - 1]).normalize()
            if prev_dt in prices.index and dt in prices.index:
                p0 = prices.loc[prev_dt]
                p1 = prices.loc[dt]
                for code, weight in weights.items():
                    if code not in prices.columns:
                        continue
                    v0 = p0.get(code)
                    v1 = p1.get(code)
                    if pd.notna(v0) and pd.notna(v1) and float(v0) > 0:
                        daily_ret += float(weight) * (float(v1) / float(v0) - 1.0)
        nav *= 1.0 + daily_ret

        cost = 0.0
        turnover = 0.0
        event = event_by_date.get(dt)
        if event is not None:
            prev_weights = dict(weights)
            target_weights = dict(event.target_weights)
            turnover = float(_turnover(prev_weights, target_weights))
            cost = max(0.0, float(cost_rate)) * turnover
            nav *= max(0.0, 1.0 - cost)
            weights = target_weights
            rebalance_rows.append(
                {
                    "signal_date": event.signal_date.date().isoformat(),
                    "trade_date": event.trade_date.date().isoformat(),
                    "pred_dt": event.pred_dt.date().isoformat(),
                    "turnover": turnover,
                    "cost": cost,
                    "nav_after_cost": nav,
                    "holding_count": len(weights),
                    "picks1": ",".join(event.picks1),
                    "picks2": ",".join(event.picks2),
                }
            )
            for code in sorted(set(prev_weights) | set(target_weights)):
                prev_weight = float(prev_weights.get(code, 0.0))
                target_weight = float(target_weights.get(code, 0.0))
                weight_delta = target_weight - prev_weight
                position_rows.append(
                    {
                        "trade_date": event.trade_date.date().isoformat(),
                        "signal_date": event.signal_date.date().isoformat(),
                        "rq_code": code,
                        "qlib_code": _rqalpha_to_qlib(code),
                        "action": _position_action(prev_weight, target_weight),
                        "prev_weight": prev_weight,
                        "target_weight": target_weight,
                        "weight_delta": weight_delta,
                        "weight": target_weight,
                    }
                )

        daily_rows.append(
            {
                "date": dt.date().isoformat(),
                "nav": nav,
                "portfolio_value": nav * float(initial_cash),
                "daily_return": daily_ret,
                "turnover": turnover,
                "cost": cost,
                "holding_count": len(weights),
            }
        )

    return pd.DataFrame(daily_rows), pd.DataFrame(rebalance_rows), pd.DataFrame(position_rows)


def calculate_metrics(daily: pd.DataFrame) -> Dict[str, float]:
    if daily.empty:
        return {}
    nav = pd.to_numeric(daily["nav"], errors="coerce").dropna()
    if nav.empty:
        return {}
    net_nav = pd.concat([pd.Series([1.0]), nav.reset_index(drop=True)], ignore_index=True)
    ret = net_nav.pct_change().dropna().fillna(0.0)
    total_return = float(nav.iloc[-1] - 1.0)
    ann = 0.0
    if len(nav) > 1 and nav.iloc[-1] > 0:
        ann = float(nav.iloc[-1] ** (252.0 / max(1, len(nav) - 1)) - 1.0)
    vol = float(ret.std(ddof=1) * math.sqrt(252)) if len(ret) > 1 else 0.0
    sharpe = float(ann / vol) if vol > 0 else 0.0
    drawdown = net_nav / net_nav.cummax() - 1.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
    win_rate = float((ret > 0).mean()) if len(ret) > 0 else 0.0
    return {
        "total_return": total_return,
        "annualized_return": ann,
        "annualized_volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "days": float(len(daily)),
        "final_nav": float(nav.iloc[-1]),
    }


def save_plots(daily: pd.DataFrame, output_dir: str) -> Dict[str, str]:
    if daily.empty:
        return {}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning("matplotlib 不可用，跳过图形输出: %s", exc)
        return {}

    out = Path(output_dir)
    dates = pd.to_datetime(daily["date"])
    nav = pd.to_numeric(daily["nav"], errors="coerce")
    drawdown = nav / nav.cummax() - 1.0
    paths: Dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, nav, label="MSA NAV")
    ax.set_title("Qlib MSA Net Asset Value")
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    p = out / "nav_curve.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    paths["nav_curve"] = str(p)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(dates, drawdown, 0, alpha=0.35)
    ax.set_title("Qlib MSA Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    p = out / "drawdown.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    paths["drawdown"] = str(p)

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(dates, pd.to_numeric(daily["turnover"], errors="coerce"), label="Turnover", alpha=0.45)
    ax1.set_ylabel("Turnover")
    ax2 = ax1.twinx()
    ax2.plot(dates, pd.to_numeric(daily["holding_count"], errors="coerce"), color="tab:orange", label="Holdings")
    ax2.set_ylabel("Holdings")
    ax1.set_title("Qlib MSA Turnover and Holdings")
    ax1.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    p = out / "turnover_holdings.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    paths["turnover_holdings"] = str(p)
    return paths


def save_html_report(metrics: Dict[str, float], plot_paths: Dict[str, str], output_dir: str) -> str:
    rows = "\n".join(
        f"<tr><td>{k}</td><td>{v:.6f}</td></tr>" for k, v in sorted(metrics.items()) if isinstance(v, (int, float))
    )
    imgs = "\n".join(
        f'<h2>{name}</h2><img src="{Path(path).name}" style="max-width: 100%;">'
        for name, path in plot_paths.items()
    )
    html = f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>Qlib MSA Backtest Report</title></head>
<body>
<h1>Qlib MSA Backtest Report</h1>
<h2>Metrics</h2>
<table border="1" cellspacing="0" cellpadding="4">{rows}</table>
{imgs}
</body>
</html>
"""
    path = Path(output_dir) / "report.html"
    path.write_text(html, encoding="utf-8")
    return str(path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qlib MSA 多策略自定义回测")
    p.add_argument("--data-config", type=str, default="config/data.yaml")
    p.add_argument("--rqalpha-config", type=str, default="config/rqalpha_config.yaml")
    p.add_argument("--pred-csi101", type=str, default=None)
    p.add_argument("--pred-csi300", type=str, default=None)
    p.add_argument("--allow-missing-csi101", action="store_true")
    p.add_argument("--allow-missing-csi300", action="store_true")
    p.add_argument("--start-date", type=str, default=None)
    p.add_argument("--end-date", type=str, default=None)
    p.add_argument("--pred-dates-are", type=str, default="auto", choices=["auto", "signal_date", "trade_date"])
    p.add_argument("--output-dir", type=str, default="data/backtest/qlib_msa")
    p.add_argument("--price-field", type=str, default="$close_qfq")
    p.add_argument("--rebalance-interval", type=int, default=5)
    p.add_argument("--initial-cash", type=float, default=150000.0)
    p.add_argument("--cost-rate", type=float, default=0.0004)
    p.add_argument("--alloc1", type=float, default=0.5)
    p.add_argument("--alloc2", type=float, default=0.5)
    p.add_argument("--msa-selection-mode", type=str, default="scheme_c", choices=["equal_weight", "scheme_c"])
    p.add_argument("--msa-preselect-topm", type=int, default=50)
    p.add_argument("--msa-vol-window", type=int, default=20)
    p.add_argument("--msa-vol-max", type=float, default=None)
    p.add_argument("--msa-vol-eps", type=float, default=0.05)
    p.add_argument("--s1-topk", type=int, default=100)
    p.add_argument("--s1-hold", type=int, default=5)
    p.add_argument("--s1-min-list-days", type=int, default=360)
    p.add_argument("--s2-topk", type=int, default=100)
    p.add_argument("--s2-hold", type=int, default=5)
    p.add_argument("--s2-min-list-days", type=int, default=360)
    p.add_argument("--s2-pb-min", type=float, default=0.0)
    p.add_argument("--s2-pb-max", type=float, default=1.0)
    p.add_argument("--s2-limitup-days", type=int, default=5)
    p.add_argument("--s1-max-vol20", type=float, default=None)
    p.add_argument("--s1-max-vol60", type=float, default=None)
    p.add_argument("--s1-max-vol120", type=float, default=None)
    p.add_argument("--s2-max-vol20", type=float, default=None)
    p.add_argument("--s2-max-vol60", type=float, default=None)
    p.add_argument("--s2-max-vol120", type=float, default=None)
    p.add_argument("--vol-drop-if-missing", action="store_true")
    p.add_argument(
        "--vol-source",
        type=str,
        default="tushare",
        choices=["qlib", "tushare", "bundle"],
        help="波动率数据源；默认与 run_msa_signal.py 保持一致为 tushare，也可指定 qlib 使用 qlib 行情",
    )
    p.add_argument("--industry-cap", type=int, default=2)
    p.add_argument("--industry-level", type=str, default="l1", choices=["l1", "l2", "l3"])
    p.add_argument("--no-html", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    os.chdir(_PROJECT_ROOT)

    pred_csi101 = _resolve_path(args.pred_csi101)
    pred_csi300 = _resolve_path(args.pred_csi300)
    auto_csi300_only = pred_csi300 is not None and pred_csi101 is None and not args.allow_missing_csi101
    if pred_csi101 is None and not auto_csi300_only:
        try:
            pred_csi101 = _find_latest_prediction("csi101")
        except Exception:
            pred_csi101 = None
    if pred_csi300 is None:
        try:
            pred_csi300 = _find_latest_prediction("csi300")
        except Exception:
            pred_csi300 = None
    if pred_csi101 is None and pred_csi300 is None:
        raise FileNotFoundError("未找到 csi101/csi300 预测文件")
    if pred_csi101 is None and not args.allow_missing_csi101 and not auto_csi300_only:
        hint = "如需单跑 csi300，请传 --allow-missing-csi101"
        if args.allow_missing_csi300 and pred_csi300 is not None:
            hint = (
                "当前只找到了 csi300 预测文件；你传入的 --allow-missing-csi300 表示允许 csi300 缺失、单跑 csi101。"
                "如需单跑 csi300，请改用 --allow-missing-csi101"
            )
        raise FileNotFoundError(f"未找到 csi101 预测文件；{hint}")
    if pred_csi300 is None and not args.allow_missing_csi300:
        raise FileNotFoundError("未找到 csi300 预测文件；如需单跑 csi101，请传 --allow-missing-csi300")

    pred_dates_are = str(args.pred_dates_are).lower()
    if pred_dates_are == "auto":
        flags = set()
        if pred_csi101:
            flags.add(_infer_pred_dates_are(pred_csi101, default="signal_date"))
        if pred_csi300:
            flags.add(_infer_pred_dates_are(pred_csi300, default="signal_date"))
        pred_dates_are = "trade_date" if "trade_date" in flags else "signal_date"

    _init_qlib(str(_resolve_path(args.data_config) or args.data_config))
    book101 = load_prediction_csv(pred_csi101, dates_are=pred_dates_are) if pred_csi101 else None
    book300 = load_prediction_csv(pred_csi300, dates_are=pred_dates_are) if pred_csi300 else None
    s1, s2 = _build_substrategies(args, pred_csi101, pred_csi300)

    all_pred_dates: List[pd.Timestamp] = []
    for book in (book101, book300):
        if book is not None:
            all_pred_dates.extend([pd.Timestamp(d).normalize() for d in book.by_date.keys()])
    start, end = _resolve_backtest_window(args, all_pred_dates)
    logger.info(
        "回测日期窗口: %s~%s（未显式传 --start-date/--end-date 时跟随预测文件实际日期）",
        start.date(),
        end.date(),
    )
    calendar = _qlib_calendar(start - pd.Timedelta(days=30), end + pd.Timedelta(days=5))
    run_calendar = pd.DatetimeIndex([d for d in calendar if start <= pd.Timestamp(d).normalize() <= end])
    if len(run_calendar) < 2:
        raise ValueError(f"qlib 日历在回测区间内不足两个交易日: {start.date()}~{end.date()}")

    ts_client = TushareClient.try_create(cache_dir=os.path.join(_PROJECT_ROOT, "data", "tushare_cache"))
    if ts_client is None:
        logger.warning("未启用 Tushare，将跳过 ST/PB/涨停等依赖 Tushare 的过滤")
    price_fetcher = QlibPriceFetcher(calendar=calendar, price_field=str(args.price_field))
    events = build_rebalance_events(
        calendar=calendar,
        s1=s1,
        s2=s2,
        book101=book101,
        book300=book300,
        pred_dates_are=pred_dates_are,
        start=pd.Timestamp(run_calendar[0]).normalize(),
        end=pd.Timestamp(run_calendar[-1]).normalize(),
        rebalance_interval_days=int(args.rebalance_interval),
        ts_client=ts_client,
        price_fetcher=price_fetcher,
        args=args,
    )
    if not events:
        raise RuntimeError("回测区间内没有生成任何有效调仓事件")

    all_codes = sorted({code for event in events for code in event.target_weights})
    prices = _load_qlib_prices(all_codes, run_calendar[0], run_calendar[-1], str(args.price_field))
    if prices.empty:
        raise RuntimeError("qlib 未返回可用价格数据，无法回测")
    daily, rebalances, positions = run_weight_backtest(
        calendar=run_calendar,
        events=events,
        prices=prices,
        initial_cash=float(args.initial_cash),
        cost_rate=float(args.cost_rate),
    )
    metrics = calculate_metrics(daily)

    output_dir = _resolve_path(args.output_dir) or args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    daily.to_csv(os.path.join(output_dir, "daily_nav.csv"), index=False, encoding="utf-8-sig")
    rebalances.to_csv(os.path.join(output_dir, "rebalance_records.csv"), index=False, encoding="utf-8-sig")
    positions.to_csv(os.path.join(output_dir, "positions.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, ensure_ascii=False, indent=2)
    plot_paths = save_plots(daily, output_dir)
    if not args.no_html:
        save_html_report(metrics, plot_paths, output_dir)

    logger.info("Qlib MSA 回测完成，输出目录: %s", output_dir)
    logger.info("核心指标: %s", metrics)


if __name__ == "__main__":
    main()
