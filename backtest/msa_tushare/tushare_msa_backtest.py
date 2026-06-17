from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backtest.msa.code_utils import rqalpha_to_tushare
from backtest.msa.filters import FilterConfig, apply_basic_filters
from backtest.msa.prediction_loader import PredictionBook, load_prediction_csv, topk
from backtest.msa.tushare_client import _try_load_token_from_secrets_file

logger = logging.getLogger(__name__)


@dataclass
class SubStrategyConfig:
    name: str
    pool: str
    allocation: float
    pred_path: str
    topk_pred: int
    target_holdings: int
    filter_cfg: FilterConfig = field(default_factory=FilterConfig)
    selection_mode: str = "equal_weight"
    preselect_topm: int = 50
    vol_window: int = 20
    vol_max: Optional[float] = None
    vol_eps: float = 0.05


class TushareMarketData:
    """Small cached Tushare data access layer for daily-bar backtests."""

    def __init__(self, *, token: str, cache_dir: str):
        import tushare as ts  # type: ignore

        self.cache_root = Path(cache_dir)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self._pro = ts.pro_api(token)

    @staticmethod
    def create(cache_dir: str) -> "TushareMarketData":
        env_token = os.environ.get("TUSHARE_TOKEN", "").strip()
        api_key_token = os.environ.get("TUSHARE_API_KEY", "").strip()
        secrets_token = _try_load_token_from_secrets_file()
        token = api_key_token or env_token or secrets_token
        if not token:
            raise RuntimeError("Tushare token not found. Set TUSHARE_API_KEY, TUSHARE_TOKEN, or create config/secrets.yaml.")
        return TushareMarketData(token=token, cache_dir=cache_dir)

    def _cache_path(self, name: str) -> Path:
        return self.cache_root / name

    def _load_df(self, name: str) -> Optional[pd.DataFrame]:
        path = self._cache_path(name)
        if not path.exists():
            csv_path = path.with_suffix(".csv")
            if not csv_path.exists():
                return None
            path = csv_path
        try:
            if path.suffix.lower() == ".parquet":
                return pd.read_parquet(path)
            return pd.read_csv(path)
        except Exception:
            return None

    def _save_df(self, name: str, df: pd.DataFrame) -> None:
        path = self._cache_path(name)
        try:
            df.to_parquet(path, index=False)
        except Exception:
            df.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8")

    @staticmethod
    def _dt(dt: pd.Timestamp | str) -> str:
        return pd.Timestamp(dt).strftime("%Y%m%d")

    def trade_calendar(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DatetimeIndex:
        start_s = self._dt(start_date)
        end_s = self._dt(end_date)
        name = f"trade_cal_{start_s}_{end_s}.parquet"
        cached = self._load_df(name)
        if cached is None or cached.empty:
            cached = self._pro.trade_cal(exchange="SSE", start_date=start_s, end_date=end_s, is_open="1")
            self._save_df(name, cached)
        if "cal_date" not in cached.columns:
            raise ValueError("Tushare trade_cal response missing cal_date")
        return pd.DatetimeIndex(pd.to_datetime(cached["cal_date"]).dt.normalize()).sort_values().unique()

    def next_trade_day(self, dt: pd.Timestamp) -> Optional[pd.Timestamp]:
        dt = pd.Timestamp(dt).normalize()
        cal = self.trade_calendar(dt, dt + pd.Timedelta(days=15))
        future = [pd.Timestamp(x).normalize() for x in cal if pd.Timestamp(x).normalize() > dt]
        return future[0] if future else None

    def stock_basic(self) -> pd.DataFrame:
        name = "stock_basic.parquet"
        cached = self._load_df(name)
        if cached is not None and not cached.empty:
            return cached
        df = self._pro.stock_basic(exchange="", list_status="L", fields="ts_code,name,market,list_date,delist_date")
        self._save_df(name, df)
        return df

    def daily_basic(self, trade_date: pd.Timestamp) -> pd.DataFrame:
        d = self._dt(trade_date)
        name = f"daily_basic_{d}.parquet"
        cached = self._load_df(name)
        if cached is not None and not cached.empty:
            return cached
        df = self._pro.daily_basic(trade_date=d, fields="ts_code,pb,total_mv,circ_mv")
        self._save_df(name, df)
        return df

    def limit_list(self, trade_date: pd.Timestamp, *, limit_type: str = "U") -> pd.DataFrame:
        d = self._dt(trade_date)
        name = f"limit_list_{limit_type}_{d}.parquet"
        cached = self._load_df(name)
        if cached is not None and not cached.empty:
            return cached
        df = self._pro.limit_list(trade_date=d, limit_type=limit_type, fields="ts_code")
        self._save_df(name, df)
        return df

    def daily(self, trade_date: pd.Timestamp) -> pd.DataFrame:
        d = self._dt(trade_date)
        name = f"daily_{d}.parquet"
        cached = self._load_df(name)
        if cached is not None and not cached.empty:
            return cached
        df = self._pro.daily(
            trade_date=d,
            fields="ts_code,trade_date,open,high,low,close,pre_close,pct_chg,vol,amount",
        )
        self._save_df(name, df)
        return df

    def close_map(self, trade_date: pd.Timestamp, codes: Iterable[str]) -> Dict[str, float]:
        ts_codes = {rqalpha_to_tushare(c) for c in codes}
        df = self.daily(trade_date)
        if df.empty:
            return {}
        df = df[df["ts_code"].astype(str).isin(ts_codes)].copy()
        return {
            str(row.ts_code): float(row.close)
            for row in df.itertuples(index=False)
            if pd.notna(getattr(row, "close", None))
        }

    def close_history_last_n(self, ts_codes: List[str], signal_date: pd.Timestamp, n: int) -> pd.DataFrame:
        if not ts_codes or n <= 0:
            return pd.DataFrame()
        signal_date = pd.Timestamp(signal_date).normalize()
        cal = self.trade_calendar(signal_date - pd.Timedelta(days=max(40, int(n) * 3)), signal_date)
        cal = pd.DatetimeIndex([d for d in cal if pd.Timestamp(d).normalize() <= signal_date])
        cal = cal[-int(n):]
        rows = []
        wanted = set(ts_codes)
        for d in cal:
            daily = self.daily(pd.Timestamp(d))
            if daily.empty:
                continue
            part = daily[daily["ts_code"].astype(str).isin(wanted)][["trade_date", "ts_code", "close"]]
            rows.append(part)
        if not rows:
            return pd.DataFrame()
        df = pd.concat(rows, ignore_index=True)
        mat = df.pivot_table(index="trade_date", columns="ts_code", values="close", aggfunc="last")
        mat.index = pd.to_datetime(mat.index)
        return mat.sort_index()


def _resolve_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    path = str(path).strip()
    if path.startswith("@"):
        path = path[1:]
    path = path.replace("/", os.sep)
    if os.path.isabs(path):
        return path
    return os.path.join(_PROJECT_ROOT, path)


def _find_prediction(pool: str) -> Optional[str]:
    path = os.path.join(_PROJECT_ROOT, "data", "predictions", f"pred_{pool}.csv")
    return path if os.path.exists(path) else None


def _normalize(weights: Dict[str, float], total: float = 1.0) -> Dict[str, float]:
    s = sum(max(0.0, float(v)) for v in weights.values())
    if s <= 0:
        return {}
    return {k: max(0.0, float(v)) / s * total for k, v in weights.items()}


def _turnover(prev_w: Dict[str, float], new_w: Dict[str, float]) -> float:
    keys = set(prev_w) | set(new_w)
    return 0.5 * sum(abs(float(prev_w.get(k, 0.0)) - float(new_w.get(k, 0.0))) for k in keys)


def _annualized_vol(closes: pd.Series) -> Optional[float]:
    s = pd.to_numeric(closes, errors="coerce").dropna()
    if len(s) < 2:
        return None
    ret = s.pct_change().dropna()
    if ret.empty:
        return None
    return float(ret.std(ddof=1) * (252 ** 0.5))


def _build_sub_weights(
    sub: SubStrategyConfig,
    signals: Dict[str, float],
    signal_date: pd.Timestamp,
    market: TushareMarketData,
) -> Tuple[List[str], Dict[str, float], Dict[str, Optional[float]]]:
    raw_n = max(int(sub.topk_pred), int(sub.preselect_topm), int(sub.target_holdings))
    raw = list(topk(signals, raw_n).keys())
    cand = apply_basic_filters(raw, signal_date, sub.filter_cfg, market)
    scored = [(c, float(signals.get(c, float("-inf")))) for c in cand]
    scored.sort(key=lambda x: x[1], reverse=True)
    if not scored:
        return [], {}, {}

    mode = str(sub.selection_mode).strip().lower()
    vols: Dict[str, Optional[float]] = {}
    if mode in {"scheme_c", "c", "vol"}:
        topm = scored[: min(len(scored), max(int(sub.preselect_topm), int(sub.target_holdings)))]
        ts_codes = [rqalpha_to_tushare(c) for c, _ in topm]
        close_mat = market.close_history_last_n(ts_codes, signal_date, int(sub.vol_window))
        for rq, ts_code in zip([c for c, _ in topm], ts_codes):
            vols[rq] = _annualized_vol(close_mat[ts_code]) if ts_code in close_mat.columns else None
        ranked = []
        for code, score in topm:
            vol = vols.get(code)
            if vol is None:
                continue
            if sub.vol_max is not None and float(vol) > float(sub.vol_max):
                continue
            ranked.append((code, score, float(vol)))
        ranked.sort(key=lambda x: (x[2], -x[1]))
        picks = [code for code, _, _ in ranked[: int(sub.target_holdings)]]
        if not picks:
            picks = [code for code, _ in scored[: int(sub.target_holdings)]]
        raw_weights = {}
        for code in picks:
            vol = vols.get(code)
            raw_weights[code] = 1.0 / (float(sub.vol_eps) + float(vol)) if vol is not None else 1.0
        return picks, _normalize(raw_weights, total=float(sub.allocation)), vols

    picks = [code for code, _ in scored[: int(sub.target_holdings)]]
    if not picks:
        return [], {}, vols
    return picks, {code: float(sub.allocation) / len(picks) for code in picks}, vols


def _portfolio_return(
    weights: Dict[str, float],
    prev_trade_date: pd.Timestamp,
    trade_date: pd.Timestamp,
    market: TushareMarketData,
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    if not weights:
        return 0.0, {}
    prev_close = market.close_map(prev_trade_date, weights.keys())
    cur_close = market.close_map(trade_date, weights.keys())
    detail: Dict[str, Dict[str, float]] = {}
    total = 0.0
    for rq_code, w in weights.items():
        ts_code = rqalpha_to_tushare(rq_code)
        p0 = prev_close.get(ts_code)
        p1 = cur_close.get(ts_code)
        if p0 is None or p1 is None or p0 <= 0:
            stock_ret = 0.0
            missing = 1.0
        else:
            stock_ret = float(p1 / p0 - 1.0)
            missing = 0.0
        contrib = float(w) * stock_ret
        total += contrib
        detail[rq_code] = {
            "close_prev": float(p0) if p0 is not None else float("nan"),
            "close_cur": float(p1) if p1 is not None else float("nan"),
            "stock_return": stock_ret,
            "stock_contrib": contrib,
            "missing_price": missing,
        }
    return total, detail


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tushare-backed MSA multi-strategy close-to-close backtest")
    parser.add_argument("--pred-csi101", type=str, default=None)
    parser.add_argument("--pred-csi300", type=str, default=None)
    parser.add_argument("--strategy-mode", choices=["dual", "csi300_only"], default="csi300_only")
    parser.add_argument("--start-date", type=str, default=None, help="Signal start date, YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="Signal end date, YYYY-MM-DD")
    parser.add_argument("--last-n-signals", type=int, default=2, help="Default runs the latest two signal dates")
    parser.add_argument("--initial-cash", type=float, default=10_000_000.0)
    parser.add_argument("--cost-rate", type=float, default=0.0004)
    parser.add_argument("--rebalance-interval", type=int, default=1)
    parser.add_argument("--alloc1", type=float, default=0.5)
    parser.add_argument("--alloc2", type=float, default=0.5)
    parser.add_argument("--s1-topk", type=int, default=20)
    parser.add_argument("--s1-hold", type=int, default=6)
    parser.add_argument("--s2-topk", type=int, default=20)
    parser.add_argument("--s2-hold", type=int, default=4)
    parser.add_argument("--msa-selection-mode", choices=["equal_weight", "scheme_c"], default="equal_weight")
    parser.add_argument("--msa-preselect-topm", type=int, default=50)
    parser.add_argument("--msa-vol-window", type=int, default=20)
    parser.add_argument("--msa-vol-max", type=float, default=None)
    parser.add_argument("--msa-vol-eps", type=float, default=0.05)
    parser.add_argument("--cache-dir", type=str, default="data/tushare_cache")
    parser.add_argument("--output-dir", type=str, default="data/backtest/msa_tushare")
    return parser.parse_args()


def _active_signal_dates(books: Dict[str, PredictionBook], args: argparse.Namespace) -> List[pd.Timestamp]:
    dates = sorted(set().union(*[set(book.by_date.keys()) for book in books.values()]))
    if args.start_date:
        dates = [d for d in dates if d >= pd.Timestamp(args.start_date).normalize()]
    if args.end_date:
        dates = [d for d in dates if d <= pd.Timestamp(args.end_date).normalize()]
    if not args.start_date and not args.end_date and args.last_n_signals > 0:
        dates = dates[-int(args.last_n_signals):]
    return [pd.Timestamp(d).normalize() for d in dates]


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    os.chdir(_PROJECT_ROOT)

    pred_csi101 = _resolve_path(args.pred_csi101) or (_find_prediction("csi101") if args.strategy_mode == "dual" else None)
    pred_csi300 = _resolve_path(args.pred_csi300) or _find_prediction("csi300")
    if args.strategy_mode == "dual" and not pred_csi101:
        raise FileNotFoundError("dual mode requires --pred-csi101 or data/predictions/pred_csi101.csv")
    if not pred_csi300:
        raise FileNotFoundError("Missing --pred-csi300 or data/predictions/pred_csi300.csv")

    books: Dict[str, PredictionBook] = {}
    if pred_csi101:
        books["csi101"] = load_prediction_csv(pred_csi101)
    if pred_csi300:
        books["csi300"] = load_prediction_csv(pred_csi300)

    signal_dates = _active_signal_dates(books, args)
    if not signal_dates:
        raise ValueError("No signal dates selected")

    alloc1 = float(args.alloc1) if "csi101" in books else 0.0
    alloc2 = float(args.alloc2) if "csi300" in books else 0.0
    total_alloc = alloc1 + alloc2
    if total_alloc <= 0:
        raise ValueError("No active strategy allocation")
    alloc1, alloc2 = alloc1 / total_alloc, alloc2 / total_alloc

    subs: List[SubStrategyConfig] = []
    if pred_csi101 and "csi101" in books:
        subs.append(
            SubStrategyConfig(
                name="small_cap_csi101",
                pool="csi101",
                allocation=alloc1,
                pred_path=pred_csi101,
                topk_pred=int(args.s1_topk),
                target_holdings=int(args.s1_hold),
                filter_cfg=FilterConfig(exclude_kcb_bj=True, exclude_st=True, min_list_days=360),
                selection_mode=str(args.msa_selection_mode),
                preselect_topm=int(args.msa_preselect_topm),
                vol_window=int(args.msa_vol_window),
                vol_max=args.msa_vol_max,
                vol_eps=float(args.msa_vol_eps),
            )
        )
    if pred_csi300 and "csi300" in books:
        subs.append(
            SubStrategyConfig(
                name="value_csi300",
                pool="csi300",
                allocation=alloc2,
                pred_path=pred_csi300,
                topk_pred=int(args.s2_topk),
                target_holdings=int(args.s2_hold),
                filter_cfg=FilterConfig(
                    exclude_kcb_bj=True,
                    exclude_st=True,
                    min_list_days=360,
                    pb_min=0.0,
                    pb_max=1.0,
                    exclude_recent_limitup_days=5,
                ),
                selection_mode=str(args.msa_selection_mode),
                preselect_topm=int(args.msa_preselect_topm),
                vol_window=int(args.msa_vol_window),
                vol_max=args.msa_vol_max,
                vol_eps=float(args.msa_vol_eps),
            )
        )

    market = TushareMarketData.create(cache_dir=_resolve_path(args.cache_dir) or args.cache_dir)
    output_dir = _resolve_path(args.output_dir) or args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    nav_rows: List[Dict[str, Any]] = []
    prev_trade_date: Optional[pd.Timestamp] = None
    prev_weights: Dict[str, float] = {}
    nav = 1.0

    for i, signal_date in enumerate(signal_dates):
        if i % max(1, int(args.rebalance_interval)) != 0:
            continue
        trade_date = market.next_trade_day(signal_date)
        if trade_date is None:
            logger.warning("No next Tushare trade day after signal_date=%s, skipped", signal_date.date())
            continue

        period_return = 0.0
        stock_detail: Dict[str, Dict[str, float]] = {}
        nav_before = nav
        if prev_trade_date is not None:
            period_return, stock_detail = _portfolio_return(prev_weights, prev_trade_date, trade_date, market)
            nav *= 1.0 + float(period_return)

        merged: Dict[str, float] = {}
        sub_details: Dict[str, Tuple[SubStrategyConfig, List[str], Dict[str, float], Dict[str, Optional[float]]]] = {}
        for sub in subs:
            signals = books[sub.pool].get(signal_date)
            picks, weights, vols = _build_sub_weights(sub, signals, signal_date, market)
            sub_details[sub.name] = (sub, picks, signals, vols)
            for code, weight in weights.items():
                merged[code] = merged.get(code, 0.0) + float(weight)
        merged = _normalize(merged, total=1.0)
        turnover = _turnover(prev_weights, merged) if prev_trade_date is not None else sum(merged.values())
        cost = float(turnover) * float(args.cost_rate)
        nav *= 1.0 - cost
        nav_after = nav

        nav_rows.append(
            {
                "signal_date": signal_date.strftime("%Y-%m-%d"),
                "trade_date": trade_date.strftime("%Y-%m-%d"),
                "prev_trade_date": prev_trade_date.strftime("%Y-%m-%d") if prev_trade_date is not None else "",
                "nav_before": nav_before,
                "period_return": period_return,
                "turnover": turnover,
                "cost": cost,
                "nav_after": nav_after,
                "pnl_amount": (nav_after - nav_before) * float(args.initial_cash),
                "holdings": len(merged),
            }
        )

        for sub_name, (sub, picks, signals, vols) in sub_details.items():
            rank_map = {code: rank + 1 for rank, (code, _) in enumerate(sorted(topk(signals, sub.topk_pred).items(), key=lambda x: x[1], reverse=True))}
            for code in picks:
                detail = stock_detail.get(code, {})
                rows.append(
                    {
                        "signal_date": signal_date.strftime("%Y-%m-%d"),
                        "trade_date": trade_date.strftime("%Y-%m-%d"),
                        "prev_trade_date": prev_trade_date.strftime("%Y-%m-%d") if prev_trade_date is not None else "",
                        "sub_strategy": sub_name,
                        "rq_code": code,
                        "ts_code": rqalpha_to_tushare(code),
                        "score": signals.get(code),
                        "rank_in_topk": rank_map.get(code),
                        "target_weight": merged.get(code, 0.0),
                        "prev_weight": prev_weights.get(code, 0.0),
                        "weight_change": merged.get(code, 0.0) - prev_weights.get(code, 0.0),
                        "close_prev": detail.get("close_prev"),
                        "close_cur": detail.get("close_cur"),
                        "stock_return": detail.get("stock_return"),
                        "stock_contrib": detail.get("stock_contrib"),
                        "vol": vols.get(code),
                    }
                )

        logger.info(
            "Tushare MSA rebalance signal=%s trade=%s holdings=%d period_return=%.4f nav=%.4f",
            signal_date.date(),
            trade_date.date(),
            len(merged),
            period_return,
            nav_after,
        )
        prev_trade_date = trade_date
        prev_weights = merged

    nav_df = pd.DataFrame(nav_rows)
    pos_df = pd.DataFrame(rows)
    nav_path = os.path.join(output_dir, "msa_tushare_nav.csv")
    pos_path = os.path.join(output_dir, "msa_tushare_positions.csv")
    summary_path = os.path.join(output_dir, "msa_tushare_summary.json")
    nav_df.to_csv(nav_path, index=False, encoding="utf-8-sig")
    pos_df.to_csv(pos_path, index=False, encoding="utf-8-sig")
    summary = {
        "strategy_mode": args.strategy_mode,
        "signal_start": signal_dates[0].strftime("%Y-%m-%d"),
        "signal_end": signal_dates[-1].strftime("%Y-%m-%d"),
        "rebalance_count": int(len(nav_df)),
        "final_nav": float(nav_df["nav_after"].iloc[-1]) if not nav_df.empty else 1.0,
        "total_return": float(nav_df["nav_after"].iloc[-1] - 1.0) if not nav_df.empty else 0.0,
        "nav_path": nav_path,
        "positions_path": pos_path,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("Saved Tushare MSA outputs: %s, %s, %s", nav_path, pos_path, summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
