"""
MSA 多策略“次日交易建议”脚本（不跑回测，只生成交易清单）。

目标：
- 读取两份预测文件（CSI101 / CSI300）
- 复用 `backtest/msa/rqalpha_msa_strategy.py` 的选股逻辑：TopK 预测 -> filters -> 目标持仓数等权
- 将两子策略按 allocation 合并为一个目标权重
- 输出“年度持仓台账”到：data/trade_plans/MSA_YYYY.csv（每次调仓都会更新台账，并记录调仓盈亏）

说明（非常关键）：
- 预测文件的 datetime 可以有两种语义：
  1) signal_date：用当日可观测数据生成的信号（建议用于回测 next_bar=T+1）
  2) trade_date：已经“对齐到下一交易日”的交易建议日期（更适合人工查看）
  本脚本支持 auto/signal_date/trade_date 三种模式。
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd

# 允许直接运行本文件时也能正确导入项目包
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backtest.msa.filters import FilterConfig, apply_basic_filters
from backtest.msa.prediction_loader import PredictionBook, load_prediction_csv, topk
from backtest.msa.tushare_client import TushareClient
from backtest.msa.code_utils import rqalpha_to_tushare

logger = logging.getLogger(__name__)


@dataclass
class SubStrategyConfig:
    name: str
    allocation: float
    pred_path: str
    topk_pred: int
    target_holdings: int
    rebalance_interval_days: int = 5
    filter_cfg: FilterConfig = field(default_factory=FilterConfig)


def _resolve_path(p: Optional[str]) -> Optional[str]:
    if p is None:
        return None
    p = str(p).strip()
    if p.startswith("@"):
        p = p[1:]
    p = p.replace("/", os.sep)
    if os.path.isabs(p):
        return p
    return os.path.join(_PROJECT_ROOT, p)


def _find_latest_prediction(pool: str) -> str:
    # 兼容两种常见位置：
    # 1) data/predictions 下的 pred_{pool}_*.csv（预测脚本输出）
    # 2) data/backtest/rqalpha 下的 rqalpha_pred_{pool}_*.csv（回测脚本复制到输出目录的文件）
    pred_dir = os.path.join(_PROJECT_ROOT, "data", "predictions")
    rq_dir = os.path.join(_PROJECT_ROOT, "data", "backtest", "rqalpha")
    patterns = [
        # 预测输出
        os.path.join(pred_dir, f"pred_{pool}_*.csv"),
        os.path.join(pred_dir, f"*{pool}*pred*.csv"),
        # 回测输出/复制
        os.path.join(rq_dir, f"rqalpha_pred_{pool}_*.csv"),
        os.path.join(rq_dir, f"*{pool}*rqalpha_pred*.csv"),
    ]
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        raise FileNotFoundError(
            f"未找到 {pool} 的预测文件。请先生成预测文件（建议放在 data/predictions/pred_{pool}_*.csv），"
            f"或传入 --pred-{pool} 指定路径。"
        )
    return max(files, key=os.path.getmtime)


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


def _next_trading_day(d: pd.Timestamp) -> pd.Timestamp:
    d = pd.Timestamp(d).normalize()
    cal = _calendar(d - pd.Timedelta(days=5), d + pd.Timedelta(days=30))
    # 取严格右侧的下一个
    pos = cal.searchsorted(d, side="right")
    if pos >= len(cal):
        # 兜底
        return d + pd.Timedelta(days=1)
    return pd.Timestamp(cal[pos]).normalize()


def _prev_trading_day(d: pd.Timestamp) -> pd.Timestamp:
    d = pd.Timestamp(d).normalize()
    cal = _calendar(d - pd.Timedelta(days=30), d + pd.Timedelta(days=5))
    pos = cal.searchsorted(d, side="left") - 1
    if pos < 0:
        return d - pd.Timedelta(days=1)
    return pd.Timestamp(cal[pos]).normalize()


def _infer_pred_dates_are(path: str, default: str = "signal_date") -> str:
    """
    尝试从预测文件中推断 datetime 的语义。
    - 若存在 _meta_shifted_next_day 且最大值为1：trade_date
    - 否则：default（通常是 signal_date）
    """
    try:
        df = pd.read_csv(path, nrows=200)
        if "_meta_shifted_next_day" in df.columns:
            v = pd.to_numeric(df["_meta_shifted_next_day"], errors="coerce").fillna(0).max()
            if int(v) == 1:
                return "trade_date"
    except Exception:
        pass
    return default


def _load_annual_ledger(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, dtype={"rq_code": str, "row_type": str})
        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.normalize()
        return df
    except Exception as e:
        logger.warning("读取年度台账失败，将创建新文件: %s, err=%s", path, e)
        return pd.DataFrame()


def _latest_rebalance_snapshot(ledger: pd.DataFrame) -> Tuple[Optional[pd.Timestamp], Dict[str, float], float]:
    """
    从年度台账中恢复“上一次调仓”的持仓权重与净值。
    返回：
      - last_trade_date
      - last_holdings: {rq_code: weight}
      - last_nav: float（默认为 1.0）
    """
    if ledger is None or ledger.empty:
        return None, {}, 1.0
    if "row_type" not in ledger.columns or "trade_date" not in ledger.columns:
        return None, {}, 1.0
    # 取最后一个 SUMMARY 行作为调仓点
    summ = ledger[ledger["row_type"].astype(str) == "SUMMARY"].copy()
    if summ.empty:
        return None, {}, 1.0
    summ = summ.dropna(subset=["trade_date"]).sort_values("trade_date")
    if summ.empty:
        return None, {}, 1.0
    last_row = summ.iloc[-1]
    last_dt = pd.Timestamp(last_row["trade_date"]).normalize()
    last_nav = 1.0
    if "nav_after" in last_row and pd.notna(last_row.get("nav_after")):
        try:
            last_nav = float(last_row["nav_after"])
        except Exception:
            last_nav = 1.0

    pos = ledger[(ledger["row_type"].astype(str) == "POSITION") & (ledger["trade_date"] == last_dt)].copy()
    holdings: Dict[str, float] = {}
    if not pos.empty and "rq_code" in pos.columns and "target_weight_total" in pos.columns:
        for _, r in pos.iterrows():
            code = str(r.get("rq_code", "")).strip()
            if not code:
                continue
            try:
                w = float(r.get("target_weight_total", 0.0))
            except Exception:
                w = 0.0
            if w > 0:
                holdings[code] = w
    return last_dt, holdings, last_nav


class _BundlePriceFetcher:
    """
    从 RQAlpha bundle 的 stocks.h5 读取收盘价（close）。
    默认 bundle 路径：~/.rqalpha/bundle
    """

    def __init__(self, bundle_path: str):
        self.bundle_path = bundle_path
        self._h5 = None
        self._cache: Dict[str, Any] = {}

    def _open(self):
        if self._h5 is not None:
            return
        import h5py

        stocks_h5 = os.path.join(self.bundle_path, "stocks.h5")
        self._h5 = h5py.File(stocks_h5, "r")

    def close(self):
        try:
            if self._h5 is not None:
                self._h5.close()
        finally:
            self._h5 = None
            self._cache = {}

    @staticmethod
    def _dt_to_int(dt: pd.Timestamp) -> int:
        # bundle datetime: yyyymmddHHMMSS
        return int(pd.Timestamp(dt).strftime("%Y%m%d") + "000000")

    def get_close(self, rq_code: str, dt: pd.Timestamp) -> Optional[float]:
        rq_code = str(rq_code).strip()
        if not rq_code:
            return None
        dt = pd.Timestamp(dt).normalize()
        try:
            self._open()
        except Exception:
            return None
        try:
            ds = self._h5[rq_code]  # type: ignore[index]
        except Exception:
            return None

        key = rq_code
        cached = self._cache.get(key)
        if cached is None:
            try:
                dts = ds["datetime"][:]
                closes = ds["close"][:]
                self._cache[key] = (dts, closes)
                cached = self._cache[key]
            except Exception:
                return None

        dts, closes = cached
        target = self._dt_to_int(dt)
        import numpy as np

        idx = int(np.searchsorted(dts, target))
        if idx < len(dts) and int(dts[idx]) == target:
            try:
                v = float(closes[idx])
                return v if v > 0 else None
            except Exception:
                return None
        return None


def _compute_portfolio_return(
    holdings: Dict[str, float],
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    price: _BundlePriceFetcher,
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    """
    用 close-to-close 计算持仓组合从 start_dt 到 end_dt 的收益。
    返回：
      - period_return
      - per_stock: {code: {"w":..., "close_start":..., "close_end":..., "ret":..., "contrib":...}}
    """
    start_dt = pd.Timestamp(start_dt).normalize()
    end_dt = pd.Timestamp(end_dt).normalize()
    per: Dict[str, Dict[str, float]] = {}
    total = 0.0
    for code, w in holdings.items():
        try:
            w_f = float(w)
        except Exception:
            continue
        if w_f <= 0:
            continue
        c0 = price.get_close(code, start_dt)
        c1 = price.get_close(code, end_dt)
        if c0 is None or c1 is None or c0 <= 0:
            continue
        r = c1 / c0 - 1.0
        contrib = w_f * r
        total += contrib
        per[code] = {"w": w_f, "close_start": float(c0), "close_end": float(c1), "ret": float(r), "contrib": float(contrib)}
    return float(total), per


def _turnover(prev_w: Dict[str, float], new_w: Dict[str, float]) -> float:
    keys = set(prev_w.keys()) | set(new_w.keys())
    s = 0.0
    for k in keys:
        s += abs(float(prev_w.get(k, 0.0)) - float(new_w.get(k, 0.0)))
    # 通常定义为 0.5 * sum(|Δw|)
    return 0.5 * s


def _select_for_substrategy(
    sub: SubStrategyConfig,
    signal_date: pd.Timestamp,
    book: PredictionBook,
    ts_client: Optional[TushareClient],
) -> Tuple[List[str], Dict[str, float]]:
    """
    返回：
    - picks：最终持仓列表（rq_code）
    - signals：当日信号字典（rq_code->score），用于输出明细
    """
    signals = book.get(signal_date)
    cand = list(topk(signals, sub.topk_pred).keys())
    cand = apply_basic_filters(cand, signal_date, sub.filter_cfg, ts_client)
    scored = [(c, signals.get(c, float("-inf"))) for c in cand]
    scored.sort(key=lambda x: x[1], reverse=True)
    picks = [c for c, _ in scored[: sub.target_holdings]]
    return picks, signals


def _merge_allocations(s1: float, s2: float) -> Tuple[float, float]:
    s1 = float(s1)
    s2 = float(s2)
    s = s1 + s2
    if s <= 0:
        return 0.5, 0.5
    return s1 / s, s2 / s


def parse_args():
    p = argparse.ArgumentParser(description="MSA 多策略次日交易清单生成")
    p.add_argument("--pred-csi101", type=str, default=None, help="CSI101 预测文件（默认自动找 data/predictions 最新）")
    p.add_argument("--pred-csi300", type=str, default=None, help="CSI300 预测文件（默认自动找 data/predictions 最新）")
    p.add_argument(
        "--allow-missing-csi300",
        action="store_true",
        help="允许缺失 CSI300 预测文件：仅运行 CSI101 子策略（alloc2 将被置为 0 并自动归一化）",
    )
    p.add_argument("--output-dir", type=str, default="data/trade_plans", help="输出目录（年度台账会写到这里）")
    p.add_argument("--asof", type=str, default=None, help="指定生成哪个日期的建议（格式YYYY-MM-DD）。含义取决于 --pred-dates-are")
    p.add_argument(
        "--pred-dates-are",
        type=str,
        default="auto",
        choices=["auto", "signal_date", "trade_date"],
        help="预测文件 datetime 的语义：auto/signal_date/trade_date",
    )
    # allocations
    p.add_argument("--alloc1", type=float, default=0.5, help="策略1（csi101）资金占比")
    p.add_argument("--alloc2", type=float, default=0.5, help="策略2（csi300）资金占比")
    # s1
    p.add_argument("--s1-topk", type=int, default=20)
    p.add_argument("--s1-hold", type=int, default=6)
    p.add_argument("--s1-min-list-days", type=int, default=360)
    # s2
    p.add_argument("--s2-topk", type=int, default=20)
    p.add_argument("--s2-hold", type=int, default=4)
    p.add_argument("--s2-min-list-days", type=int, default=360)
    p.add_argument("--s2-pb-min", type=float, default=0.0)
    p.add_argument("--s2-pb-max", type=float, default=1.0)
    p.add_argument("--s2-limitup-days", type=int, default=5)
    # ledger / pnl
    p.add_argument("--initial-cash", type=float, default=150000.0, help="用于把收益率换算为金额的初始资金（默认15万）")
    p.add_argument("--cost-rate", type=float, default=0.0004, help="调仓成本率（用于估算换手成本），如0.0004=万分之4")
    p.add_argument(
        "--bundle-path",
        type=str,
        default=None,
        help="RQAlpha bundle 路径（包含 stocks.h5）。默认使用 ~/.rqalpha/bundle",
    )
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    os.chdir(_PROJECT_ROOT)

    pred_csi101 = _resolve_path(args.pred_csi101) or _find_latest_prediction("csi101")
    pred_csi300 = _resolve_path(args.pred_csi300)
    if not pred_csi300:
        try:
            pred_csi300 = _find_latest_prediction("csi300")
        except Exception:
            pred_csi300 = None

    if not os.path.exists(pred_csi101):
        raise FileNotFoundError(f"csi101 预测文件不存在: {pred_csi101}")
    if pred_csi300 and not os.path.exists(pred_csi300):
        pred_csi300 = None

    if pred_csi300 is None and not args.allow_missing_csi300:
        raise FileNotFoundError(
            "csi300 预测文件不存在/未找到。请先生成 csi300 预测文件并传入 --pred-csi300，"
            "或添加 --allow-missing-csi300 仅运行 csi101 子策略。"
        )

    # 推断/指定预测日期语义
    pred_dates_are = args.pred_dates_are
    if pred_dates_are == "auto":
        # 两份文件只要有一个明确标注为 trade_date，我们就按 trade_date 处理（更符合“次日建议”语义）
        p1 = _infer_pred_dates_are(pred_csi101, default="signal_date")
        p2 = _infer_pred_dates_are(pred_csi300, default="signal_date")
        pred_dates_are = "trade_date" if ("trade_date" in {p1, p2}) else "signal_date"

    book101 = load_prediction_csv(pred_csi101)
    book300 = load_prediction_csv(pred_csi300) if pred_csi300 else None

    # 选用共同可用的最大日期（避免某一份缺日期）
    dates101 = sorted(book101.by_date.keys())
    if not dates101:
        raise ValueError("csi101 预测文件中没有任何日期数据，无法生成交易清单")
    if book300 is not None:
        dates300 = sorted(book300.by_date.keys())
        if not dates300:
            raise ValueError("csi300 预测文件中没有任何日期数据，无法生成交易清单")
        max_common = min(dates101[-1], dates300[-1])
    else:
        max_common = dates101[-1]

    if args.asof:
        asof = pd.Timestamp(args.asof).normalize()
    else:
        asof = pd.Timestamp(max_common).normalize()

    if pred_dates_are == "trade_date":
        trade_date = asof
        signal_date = _prev_trading_day(trade_date)
    else:
        # signal_date：用 asof 当天数据生成下一交易日建议
        signal_date = asof
        trade_date = _next_trading_day(signal_date)

    logger.info("MSA 信号生成日(signal_date)=%s，交易建议日(trade_date)=%s，pred_dates_are=%s", signal_date.date(), trade_date.date(), pred_dates_are)
    logger.info("预测文件：csi101=%s；csi300=%s", pred_csi101, pred_csi300 or "(missing)")

    # 子策略配置（与 rqalpha_msa_strategy 默认一致）
    if book300 is None:
        alloc1, alloc2 = _merge_allocations(args.alloc1, 0.0)
    else:
        alloc1, alloc2 = _merge_allocations(args.alloc1, args.alloc2)
    s1 = SubStrategyConfig(
        name="small_cap_csi101",
        allocation=alloc1,
        pred_path=pred_csi101,
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
    )
    s2 = None
    if book300 is not None:
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
        )

    # Tushare（可选）
    ts_client = TushareClient.try_create(cache_dir=os.path.join(_PROJECT_ROOT, "data", "tushare_cache"))
    if ts_client is None:
        logger.warning("未启用 Tushare（未设置 token 或初始化失败），将跳过 ST/PB/涨停等相关过滤")

    # 子策略选股
    picks1, sig1 = _select_for_substrategy(s1, signal_date, book101, ts_client)
    picks2, sig2 = [], {}
    if s2 is not None and book300 is not None:
        picks2, sig2 = _select_for_substrategy(s2, signal_date, book300, ts_client)

    if not picks1 and not picks2:
        raise RuntimeError("两子策略均未选出股票（可能全部被过滤/当天无信号）")

    # 构建权重：子策略内部等权，组合层按 allocation 合并
    def _eq(picks: List[str], total: float) -> Dict[str, float]:
        if not picks:
            return {}
        w = total / len(picks)
        return {c: w for c in picks}

    w1 = _eq(picks1, s1.allocation)
    w2 = _eq(picks2, s2.allocation) if s2 is not None else {}

    merged: Dict[str, float] = {}
    for code, w in {**w1, **w2}.items():
        merged[code] = merged.get(code, 0.0) + float(w)

    # 归一化到 1.0（满仓）
    total_w = sum(max(0.0, v) for v in merged.values())
    if total_w > 0:
        merged = {k: v / total_w for k, v in merged.items()}

    # ===== 年度持仓台账：每次调仓追加/更新，并记录调仓盈亏 =====
    out_dir = _resolve_path(args.output_dir) or os.path.join(_PROJECT_ROOT, "data", "trade_plans")
    os.makedirs(out_dir, exist_ok=True)
    ledger_year = int(trade_date.year)
    ledger_path = os.path.join(out_dir, f"MSA_{ledger_year}.csv")
    ledger = _load_annual_ledger(ledger_path)

    last_dt, last_holdings, last_nav = _latest_rebalance_snapshot(ledger)

    # 判断是否“发生调仓”：与上次目标持仓权重相比有变化
    eps = 1e-9
    changed = True
    if last_dt is not None:
        keys = set(last_holdings.keys()) | set(merged.keys())
        changed = any(abs(float(last_holdings.get(k, 0.0)) - float(merged.get(k, 0.0))) > eps for k in keys)

    if not changed:
        logger.info("本次目标持仓与上次调仓一致（未发生调仓），不更新年度台账：%s", ledger_path)
        return

    # 计算上一期到本次调仓日的收益（close-to-close），并估算换手成本
    bundle_path = args.bundle_path
    if not bundle_path:
        import pathlib
        bundle_path = str(pathlib.Path.home() / ".rqalpha" / "bundle")
    bundle_path = str(bundle_path).replace("/", os.sep)

    period_return = 0.0
    per_stock_return: Dict[str, Dict[str, float]] = {}
    nav_before = float(last_nav)
    nav_after = float(last_nav)
    turnover = 0.0
    cost = 0.0

    price_fetcher = None
    try:
        if last_dt is not None and last_holdings and os.path.exists(os.path.join(bundle_path, "stocks.h5")):
            price_fetcher = _BundlePriceFetcher(bundle_path)
            period_return, per_stock_return = _compute_portfolio_return(last_holdings, last_dt, trade_date, price_fetcher)
            nav_after = nav_before * (1.0 + float(period_return))
        else:
            # 第一笔调仓没有“上一期”，收益记为0
            period_return = 0.0
            nav_after = nav_before
    finally:
        if price_fetcher is not None:
            price_fetcher.close()

    turnover = _turnover(last_holdings, merged) if last_dt is not None else 0.0
    cost = float(turnover) * float(args.cost_rate)
    nav_after_cost = nav_after * (1.0 - cost)

    pnl_amount = (nav_after_cost - nav_before) * float(args.initial_cash)

    # 输出行：SUMMARY + POSITION（每只股票一行）
    rows: List[Dict[str, Any]] = []

    rows.append(
        {
            "row_type": "SUMMARY",
            "trade_date": trade_date.strftime("%Y-%m-%d"),
            "signal_date": signal_date.strftime("%Y-%m-%d"),
            "prev_trade_date": last_dt.strftime("%Y-%m-%d") if last_dt is not None else "",
            "period_return": float(period_return),
            "turnover": float(turnover),
            "cost_rate": float(args.cost_rate),
            "cost": float(cost),
            "nav_before": float(nav_before),
            "nav_after": float(nav_after_cost),
            "pnl_amount": float(pnl_amount),
            "initial_cash": float(args.initial_cash),
            "note": "close_to_close on bundle; cost=turnover*cost_rate",
        }
    )

    def _add_rows(sub: SubStrategyConfig, picks: List[str], signals: Dict[str, float], sub_weights: Dict[str, float]):
        # rank：在 topk_pred 内的排名（仅用于展示）
        top = list(topk(signals, sub.topk_pred).items())
        top.sort(key=lambda x: x[1], reverse=True)
        rank_map = {code: i + 1 for i, (code, _) in enumerate(top)}
        for code in picks:
            prev_w = float(last_holdings.get(code, 0.0)) if last_dt is not None else 0.0
            new_w = float(merged.get(code, 0.0))
            per_ret = per_stock_return.get(code, {})
            rows.append(
                {
                    "row_type": "POSITION",
                    "trade_date": trade_date.strftime("%Y-%m-%d"),
                    "signal_date": signal_date.strftime("%Y-%m-%d"),
                    "prev_trade_date": last_dt.strftime("%Y-%m-%d") if last_dt is not None else "",
                    "sub_strategy": sub.name,
                    "rq_code": code,
                    "ts_code": rqalpha_to_tushare(code),
                    "score": float(signals.get(code, float("nan"))),
                    "rank_in_topk": int(rank_map.get(code, 0)),
                    "sub_alloc": float(sub.allocation),
                    "sub_target_weight": float(sub_weights.get(code, 0.0)),
                    "target_weight_total": float(merged.get(code, 0.0)),
                    "action": "BUY",
                    "prev_weight": prev_w,
                    "weight_change": new_w - prev_w,
                    "close_prev": float(per_ret.get("close_start", float("nan"))) if per_ret else float("nan"),
                    "close_cur": float(per_ret.get("close_end", float("nan"))) if per_ret else float("nan"),
                    "stock_return": float(per_ret.get("ret", float("nan"))) if per_ret else float("nan"),
                    "stock_contrib": float(per_ret.get("contrib", float("nan"))) if per_ret else float("nan"),
                }
            )

    _add_rows(s1, picks1, sig1, w1)
    if s2 is not None:
        _add_rows(s2, picks2, sig2, w2)

    df_new = pd.DataFrame(rows)
    # POSITION 排序：总权重 -> 子策略 -> 分数；SUMMARY 保持第一行
    if not df_new.empty and "row_type" in df_new.columns:
        pos = df_new[df_new["row_type"] == "POSITION"].copy()
        summ = df_new[df_new["row_type"] == "SUMMARY"].copy()
        if not pos.empty:
            pos = pos.sort_values(["target_weight_total", "sub_strategy", "score"], ascending=[False, True, False])
        df_new = pd.concat([summ, pos], ignore_index=True)

    # 更新/覆盖同一 trade_date 的记录（保证可重复运行）
    if ledger is not None and not ledger.empty and "trade_date" in ledger.columns:
        try:
            ledger["trade_date"] = pd.to_datetime(ledger["trade_date"], errors="coerce").dt.normalize()
            td = pd.Timestamp(trade_date).normalize()
            ledger = ledger[ledger["trade_date"] != td]
        except Exception:
            pass
    ledger_updated = pd.concat([ledger, df_new], ignore_index=True) if ledger is not None and not ledger.empty else df_new
    # 按 trade_date 排序，保证台账可读
    if "trade_date" in ledger_updated.columns:
        ledger_updated["trade_date"] = pd.to_datetime(ledger_updated["trade_date"], errors="coerce").dt.normalize()
        # 自定义排序：SUMMARY 在前，POSITION 在后
        def _rt_order(x: Any) -> int:
            s = str(x).strip().upper()
            if s == "SUMMARY":
                return 0
            if s == "POSITION":
                return 1
            return 9

        if "row_type" in ledger_updated.columns:
            ledger_updated["_row_type_order"] = ledger_updated["row_type"].map(_rt_order)
            ledger_updated = ledger_updated.sort_values(
                ["trade_date", "_row_type_order", "target_weight_total"],
                ascending=[True, True, False],
                na_position="last",
            )
            ledger_updated = ledger_updated.drop(columns=["_row_type_order"], errors="ignore")
        else:
            ledger_updated = ledger_updated.sort_values(["trade_date", "target_weight_total"], ascending=[True, False], na_position="last")

    ledger_updated.to_csv(ledger_path, index=False, encoding="utf-8-sig")
    logger.info("MSA 年度持仓台账已更新: %s（新增 trade_date=%s，股票数=%d，SUMMARY: return=%.4f, pnl=%.2f）",
                ledger_path, trade_date.strftime("%Y-%m-%d"), len(merged), float(period_return), float(pnl_amount))


if __name__ == "__main__":
    main()


