"""
MSA 多策略组合（RQAlpha 策略脚本）

实现自 `backtest/投资策略` 的核心要点（可配置）：
- 总资金分成两份（策略1/策略2）
- 策略1：CSI101 预测 Top20 -> 过滤（科创/北交、ST、上市天数等）-> 选前N只等权，周频调仓；每日若卖出则补齐
- 策略2：CSI300 预测 Top20 -> 过滤（近5日涨停、0<PB<1 等）-> 选前N只等权，周频调仓；每日若卖出则补齐
- 通用风控：每日临近收盘检查个股回撤>阈值则卖出；卖出后立即按各自策略补仓

注意：
- Tushare 过滤为可选：未设置 TUSHARE_TOKEN 也可跑通，但会跳过部分过滤条件
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import pandas as pd

# 允许被 RQAlpha 以“脚本文件”方式加载时也能正确导入 backtest.msa 下的模块
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from rqalpha.api import *  # type: ignore
    RQALPHA_IMPORTED = True
except Exception:
    RQALPHA_IMPORTED = False

# 仅用于 IDE/静态检查，避免 “subscribe/get_positions/order_target_percent/plot 未定义” 的告警
if TYPE_CHECKING:  # pragma: no cover
    from rqalpha.api import subscribe, get_positions, order_target_percent, plot  # type: ignore

from backtest.msa.filters import FilterConfig, apply_basic_filters
from backtest.msa.prediction_loader import PredictionBook, load_prediction_csv, topk
from backtest.msa.tushare_client import TushareClient

logger = logging.getLogger(__name__)


@dataclass
class SubStrategyConfig:
    name: str
    allocation: float
    pred_path: str
    topk_pred: int
    target_holdings: int
    rebalance_interval_days: int = 5
    max_per_industry: int = 2  # 先占位（行业需要额外数据源，这里先不强制）
    filter_cfg: FilterConfig = field(default_factory=FilterConfig)


def _load_context_vars(context) -> Dict:
    extra_cfg = getattr(context.config, "extra", None)
    if extra_cfg and hasattr(extra_cfg, "context_vars"):
        cv = extra_cfg.context_vars
        return cv if isinstance(cv, dict) else cv.__dict__
    if isinstance(extra_cfg, dict) and "context_vars" in extra_cfg:
        return extra_cfg["context_vars"] or {}
    return {}


def _safe_float(x, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _normalize_weights(weights: Dict[str, float], total: float) -> Dict[str, float]:
    if not weights:
        return {}
    s = sum(max(0.0, float(v)) for v in weights.values())
    if s <= 0:
        return {}
    return {k: max(0.0, float(v)) / s * total for k, v in weights.items()}

def _turnover(prev_w: Dict[str, float], new_w: Dict[str, float]) -> float:
    keys = set(prev_w.keys()) | set(new_w.keys())
    s = 0.0
    for k in keys:
        s += abs(float(prev_w.get(k, 0.0)) - float(new_w.get(k, 0.0)))
    # 通常定义为 0.5 * sum(|Δw|)
    return 0.5 * s


def _safe_plot(name: str, value):
    # 兼容：在非 RQAlpha 环境下 import 本文件不报错
    if not RQALPHA_IMPORTED:
        return
    try:
        plot(name, value)  # type: ignore[name-defined]
    except Exception:
        pass


def _plot_daily_metrics(context, *, turnover: float = 0.0, risk_sells: int = 0):
    """
    用 RQAlpha 的 plot() 输出关键指标，供 sys_analyser 收集并绘图。
    风格参考 RQAlpha 教程里 plot("short avg", ...) 的用法：
    https://rqalpha.readthedocs.io/zh-cn/latest/intro/tutorial.html
    """
    try:
        p = getattr(context, "portfolio", None)
        if p is None:
            return
        total_value = float(getattr(p, "total_value", 0.0) or 0.0)
        cash = float(getattr(p, "cash", 0.0) or 0.0)
        unit_nav = float(getattr(p, "unit_net_value", 0.0) or 0.0)
        market_value = float(getattr(p, "market_value", 0.0) or 0.0)
        exposure = (market_value / total_value) if total_value > 0 else 0.0
        cash_ratio = (cash / total_value) if total_value > 0 else 0.0
        _safe_plot("msa/nav", unit_nav)
        _safe_plot("msa/exposure", exposure)
        _safe_plot("msa/cash_ratio", cash_ratio)
        _safe_plot("msa/turnover", float(turnover))
        _safe_plot("msa/risk_sells", int(risk_sells))
        # 持仓数
        try:
            positions = get_positions()  # type: ignore[name-defined]
        except Exception:
            positions = []
        holding_n = 0
        for pos in positions:
            qty = float(getattr(pos, "quantity", 0) or 0)
            if qty > 0:
                holding_n += 1
        _safe_plot("msa/holdings", int(holding_n))
    except Exception:
        pass

def _build_equal_weight_portfolio(codes: List[str], total_weight: float) -> Dict[str, float]:
    if not codes:
        return {}
    w = total_weight / len(codes)
    return {c: w for c in codes}


def init(context):
    cv = _load_context_vars(context)

    # 必填：两份预测文件
    pred_csi101 = cv.get("pred_csi101")
    pred_csi300 = cv.get("pred_csi300")
    if not pred_csi101 or not pred_csi300:
        raise ValueError("MSA 策略需要在 extra.context_vars 中提供 pred_csi101 与 pred_csi300 两个预测文件路径")

    def _resolve(p: str) -> str:
        p = str(p).strip()
        if p.startswith("@"):
            p = p[1:]
        p = p.replace("/", os.sep)
        if os.path.isabs(p):
            return p
        return os.path.join(_PROJECT_ROOT, p)

    pred_csi101 = _resolve(pred_csi101)
    pred_csi300 = _resolve(pred_csi300)
    if not os.path.exists(pred_csi101):
        raise FileNotFoundError(f"MSA: csi101 预测文件不存在: {pred_csi101}")
    if not os.path.exists(pred_csi300):
        raise FileNotFoundError(f"MSA: csi300 预测文件不存在: {pred_csi300}")

    # 资金分配（默认 50/50）
    alloc1 = _safe_float(cv.get("alloc_strategy1", 0.5), 0.5)
    alloc2 = _safe_float(cv.get("alloc_strategy2", 0.5), 0.5)
    s = alloc1 + alloc2
    alloc1, alloc2 = (alloc1 / s, alloc2 / s) if s > 0 else (0.5, 0.5)

    # 子策略参数（默认值按策略文件描述）
    s1 = SubStrategyConfig(
        name="small_cap_csi101",
        allocation=alloc1,
        pred_path=str(pred_csi101),
        topk_pred=int(cv.get("s1_topk_pred", 20)),
        target_holdings=int(cv.get("s1_target_holdings", 6)),
        rebalance_interval_days=int(cv.get("s1_rebalance_interval_days", 5)),
        max_per_industry=int(cv.get("s1_max_per_industry", 2)),
        filter_cfg=FilterConfig(
            exclude_kcb_bj=True,
            exclude_st=True,
            min_list_days=int(cv.get("s1_min_list_days", 360)),
            pb_min=None,
            pb_max=None,
            exclude_recent_limitup_days=0,
        ),
    )

    # 策略2：默认加 PB (0,1) 与近5日涨停过滤
    s2 = SubStrategyConfig(
        name="value_csi300",
        allocation=alloc2,
        pred_path=str(pred_csi300),
        topk_pred=int(cv.get("s2_topk_pred", 20)),
        target_holdings=int(cv.get("s2_target_holdings", 4)),  # 你文档里写 Top2，但又写“保持4只”，这里默认4，可配置为2
        rebalance_interval_days=int(cv.get("s2_rebalance_interval_days", 5)),
        max_per_industry=int(cv.get("s2_max_per_industry", 2)),
        filter_cfg=FilterConfig(
            exclude_kcb_bj=True,
            exclude_st=True,
            min_list_days=int(cv.get("s2_min_list_days", 360)),
            pb_min=float(cv.get("s2_pb_min", 0.0)),
            pb_max=float(cv.get("s2_pb_max", 1.0)),
            exclude_recent_limitup_days=int(cv.get("s2_exclude_recent_limitup_days", 5)),
        ),
    )

    context.sub_strategies = [s1, s2]
    context.pred_books: Dict[str, PredictionBook] = {
        "csi101": load_prediction_csv(s1.pred_path),
        "csi300": load_prediction_csv(s2.pred_path),
    }

    # 诊断：预测信号日期范围 vs 回测配置日期范围（帮助排查“全程无交易导致净值水平线”）
    try:
        pb_dates = []
        for pb in context.pred_books.values():
            pb_dates.extend(list(pb.by_date.keys()))
        pb_dates = sorted(set([pd.Timestamp(d).normalize() for d in pb_dates]))
        pred_min = pb_dates[0] if pb_dates else None
        pred_max = pb_dates[-1] if pb_dates else None
        base_cfg = getattr(context.config, "base", None)
        cfg_start = getattr(base_cfg, "start_date", None) if base_cfg is not None else None
        cfg_end = getattr(base_cfg, "end_date", None) if base_cfg is not None else None
        logger.info("MSA 预测信号日期范围: %s ~ %s", pred_min, pred_max)
        logger.info("MSA 回测配置日期范围: %s ~ %s", cfg_start, cfg_end)
        if pred_min is not None and pred_max is not None and cfg_start and cfg_end:
            try:
                cs = pd.Timestamp(cfg_start).normalize()
                ce = pd.Timestamp(cfg_end).normalize()
                if pred_max < cs or pred_min > ce:
                    logger.warning(
                        "⚠ 预测信号日期与回测区间完全不重叠，策略将全程无交易（净值可能是一条水平线）。pred=%s~%s, cfg=%s~%s",
                        pred_min,
                        pred_max,
                        cs,
                        ce,
                    )
            except Exception:
                pass
    except Exception:
        pass

    # Tushare 可选
    context.ts_client = TushareClient.try_create(cache_dir=str(cv.get("tushare_cache_dir", "data/tushare_cache")))

    # 通用风控参数
    context.drawdown_stop = _safe_float(cv.get("drawdown_stop", 0.08), 0.08)  # 8%
    context.close_check_minute = int(cv.get("close_check_minute", 30))  # 收盘前N分钟

    # 状态
    context.last_rebalance_date = None
    context.high_watermark: Dict[str, float] = {}
    context._last_target_weights: Dict[str, float] = {}

    # 订阅：先尽可能从预测文件把 universe 订阅上（RQAlpha 需要订阅才有 bar）
    all_codes = set()
    for pb in context.pred_books.values():
        for daily in pb.by_date.values():
            all_codes |= set(daily.keys())
    if all_codes:
        try:
            subscribe(list(all_codes))
        except Exception:
            pass

    logger.info("MSA 初始化完成：alloc策略1=%.2f, alloc策略2=%.2f, drawdown_stop=%.2f%%",
                s1.allocation, s2.allocation, context.drawdown_stop * 100)


def before_trading(context):
    # 每日重置标记
    context.need_close_check = True


def _today_ts(context) -> pd.Timestamp:
    # RQAlpha 在策略里常用 context.now 或 pd.Timestamp.now()，这里尽量兼容
    now = getattr(context, "now", None)
    if now is None:
        return pd.Timestamp.today().normalize()
    return pd.Timestamp(now).normalize()


def _should_rebalance(context, today: pd.Timestamp) -> bool:
    last = getattr(context, "last_rebalance_date", None)
    if last is None:
        return True
    delta = (today - pd.Timestamp(last).normalize()).days
    # 任一子策略要求的调仓间隔到了就调
    min_interval = min([max(1, s.rebalance_interval_days) for s in context.sub_strategies])
    return delta >= min_interval


def _select_for_substrategy(
    context,
    sub: SubStrategyConfig,
    today: pd.Timestamp,
    book: PredictionBook,
) -> List[str]:
    signals = book.get(today)
    cand = list(topk(signals, sub.topk_pred).keys())

    # tushare 过滤
    ts_client: Optional[TushareClient] = getattr(context, "ts_client", None)
    cand = apply_basic_filters(cand, today, sub.filter_cfg, ts_client)

    # 先按信号强度排序，取 target_holdings
    scored = [(c, signals.get(c, float("-inf"))) for c in cand]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in scored[: sub.target_holdings]]


def _build_target_weights(context, today: pd.Timestamp) -> Dict[str, float]:
    # 组合层：按 allocation 分配权重，每个子策略内部等权
    target: Dict[str, float] = {}
    for sub in context.sub_strategies:
        book = context.pred_books["csi101"] if "csi101" in sub.name else context.pred_books["csi300"]
        picks = _select_for_substrategy(context, sub, today, book)
        sub_w = _build_equal_weight_portfolio(picks, sub.allocation)
        for code, w in sub_w.items():
            target[code] = target.get(code, 0.0) + w

    # 归一化到 1（满仓）
    return _normalize_weights(target, total=1.0)


def _update_high_watermark(context, bar_dict):
    # 用收盘价更新持仓高水位（回撤止损）
    try:
        positions = get_positions()
    except Exception:
        positions = []
    def _get_bar(code: str):
        # RQAlpha 传入的是 BarMap，不一定有 .get()
        try:
            return bar_dict[code]
        except Exception:
            return None
    for pos in positions:
        code = getattr(pos, "order_book_id", None)
        if not code:
            continue
        bar = _get_bar(code)
        if bar is None:
            continue
        px = float(getattr(bar, "close", 0.0) or 0.0)
        if px <= 0:
            continue
        hw = context.high_watermark.get(code, px)
        context.high_watermark[code] = max(hw, px)


def _risk_sell_list(context, bar_dict) -> List[str]:
    # 回撤 > 阈值：卖出
    stop = float(getattr(context, "drawdown_stop", 0.08))
    to_sell: List[str] = []
    try:
        positions = get_positions()
    except Exception:
        positions = []
    def _get_bar(code: str):
        try:
            return bar_dict[code]
        except Exception:
            return None
    for pos in positions:
        code = getattr(pos, "order_book_id", None)
        qty = float(getattr(pos, "quantity", 0) or 0)
        if not code or qty <= 0:
            continue
        bar = _get_bar(code)
        if bar is None:
            continue
        px = float(getattr(bar, "close", 0.0) or 0.0)
        hw = float(context.high_watermark.get(code, px) or px)
        if hw > 0 and px / hw - 1.0 <= -stop:
            to_sell.append(code)
    return to_sell


def handle_bar(context, bar_dict):
    today = _today_ts(context)
    turnover = 0.0

    # 1) 更新高水位
    _update_high_watermark(context, bar_dict)

    # 2) 调仓：周频/间隔到达
    if _should_rebalance(context, today):
        target = _build_target_weights(context, today)
        try:
            prev = getattr(context, "_last_target_weights", {}) or {}
            turnover = float(_turnover(prev, target))
            context._last_target_weights = dict(target)
        except Exception:
            turnover = 0.0
        _rebalance_to_target(target, bar_dict)
        context.last_rebalance_date = today

    # 3) 临近收盘做风控：回撤止损 + 立刻补仓
    risk_sells = 0
    if getattr(context, "need_close_check", False):
        # 尝试用分钟判断（如果拿不到分钟数据就退化为每根bar都检查一次，但只执行一次）
        now = getattr(context, "now", None)
        if now is not None:
            try:
                # A股 15:00 收盘，收盘前 N 分钟触发
                minute = pd.Timestamp(now).hour * 60 + pd.Timestamp(now).minute
                close_minute = 15 * 60
                if minute < close_minute - int(context.close_check_minute):
                    return
            except Exception:
                pass

        to_sell = _risk_sell_list(context, bar_dict)
        risk_sells = len(to_sell)
        for code in to_sell:
            try:
                order_target_percent(code, 0)
            except Exception:
                pass

        # 卖出后补仓：按最新目标权重再平衡一次
        if to_sell:
            target = _build_target_weights(context, today)
            _rebalance_to_target(target, bar_dict)

        context.need_close_check = False

    # 4) 绘图指标（每日输出）
    _plot_daily_metrics(context, turnover=turnover, risk_sells=risk_sells)


def _rebalance_to_target(target_weights: Dict[str, float], bar_dict=None):
    # 统一用 order_target_percent 做到目标权重
    # 先卖出不在目标里的持仓
    try:
        positions = get_positions()
    except Exception:
        positions = []
    held = set()
    pos_qty: Dict[str, float] = {}
    for pos in positions:
        code = getattr(pos, "order_book_id", None)
        qty = float(getattr(pos, "quantity", 0) or 0)
        if code and qty > 0:
            held.add(code)
            pos_qty[code] = qty

    def _get_bar(code: str):
        if bar_dict is None:
            return None
        try:
            return bar_dict[code]
        except Exception:
            return None

    def _is_limit_up(bar) -> bool:
        try:
            px = float(getattr(bar, "close", 0.0) or 0.0)
            lu = float(getattr(bar, "limit_up", 0.0) or 0.0)
            return lu > 0 and px >= lu - 1e-8
        except Exception:
            return False

    def _is_limit_down(bar) -> bool:
        try:
            px = float(getattr(bar, "close", 0.0) or 0.0)
            ld = float(getattr(bar, "limit_down", 0.0) or 0.0)
            return ld > 0 and px <= ld + 1e-8
        except Exception:
            return False

    for code in held:
        if code not in target_weights:
            try:
                order_target_percent(code, 0)
            except Exception:
                pass

    # 再设置目标权重
    for code, w in target_weights.items():
        # 1) 过滤极小权重，避免触发“下单数量为0”的系统告警
        try:
            w_f = float(w)
        except Exception:
            continue
        if w_f <= 0:
            continue

        # 2) 若无法获取bar，直接交给RQAlpha处理（保持兼容）
        bar = _get_bar(code)
        if bar is not None:
            # 买入方向遇到涨停通常无法成交；卖出方向遇到跌停通常无法成交
            cur_qty = float(pos_qty.get(code, 0.0))
            # 由于我们用 order_target_percent，无法直接知道买/卖方向，但可以用“当前是否持仓”为近似：
            # - 无持仓且目标>0：大概率是买入
            # - 有持仓且目标>0：可能买也可能卖；这里仅对“明显不可成交”的情况做跳过
            if cur_qty <= 0 and _is_limit_up(bar):
                continue
            if cur_qty > 0 and _is_limit_down(bar):
                # 跌停时减仓/清仓往往会被取消，跳过可减少告警（风险控制卖出仍会尝试）
                continue

            # 3) 估算最小可交易单位（A股默认100股）——若目标资金不足1手且当前无持仓，则跳过
            try:
                p = getattr(getattr(bar, "_dict", None), "close", None)  # noqa: F841
            except Exception:
                pass
            try:
                price = float(getattr(bar, "close", 0.0) or 0.0)
            except Exception:
                price = 0.0
            try:
                total_value = float(getattr(getattr(getattr(bar, "_env", None), "portfolio", None), "total_value", 0.0))  # noqa: F841
            except Exception:
                total_value = 0.0
            # 用 context.portfolio 更可靠，但这里没有 context；退化用 None 则不做该判断
            # 因此只在 bar_dict 可用且 bar 上有 close 时做保守判断：低价股票也可能误判，阈值用得更宽松
            if cur_qty <= 0 and price > 0:
                # 经验阈值：目标权重小于 0.0005（0.05%）几乎一定买不到一手，直接跳过
                if w_f < 0.0005:
                    continue
        try:
            order_target_percent(code, w_f)
        except Exception:
            continue


