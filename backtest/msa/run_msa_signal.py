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

如果要更严格（每行业最多1只）
  python backtest/msa/run_msa_signal.py --industry-cap 1
保持默认就行（每行业最多2只）：
  python backtest/msa/run_msa_signal.py
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import math
from collections import defaultdict

# 允许直接运行本文件时也能正确导入项目包
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backtest.msa.filters import FilterConfig, apply_basic_filters
from backtest.msa.prediction_loader import PredictionBook, load_prediction_csv, topk
from backtest.msa.tushare_client import TushareClient
from backtest.msa.code_utils import rqalpha_to_tushare

logger = logging.getLogger(__name__)

def _rqalpha_to_qlib(rq_code: str) -> str:
    """
    000001.XSHE -> SZ000001
    600000.XSHG -> SH600000
    """
    s = str(rq_code).strip()
    if s.endswith(".XSHE"):
        return "SZ" + s.split(".", 1)[0]
    if s.endswith(".XSHG"):
        return "SH" + s.split(".", 1)[0]
    return s


@dataclass
class SubStrategyConfig:
    name: str
    allocation: float
    pred_path: str
    topk_pred: int
    target_holdings: int
    rebalance_interval_days: int = 5
    filter_cfg: FilterConfig = field(default_factory=FilterConfig)
    # 选股/权重模式：equal_weight（旧） / scheme_c（TopM->低波动->风险预算）
    selection_mode: str = "equal_weight"
    # 方案C参数
    preselect_topm: int = 50
    vol_window: int = 20
    vol_max: Optional[float] = None  # 年化波动率上限（可选，超过剔除）
    vol_eps: float = 0.05            # 风险预算 eps：w∝1/(eps+vol)
    # 历史波动率过滤（年化）。None 表示不启用该窗口过滤。
    max_vol20: Optional[float] = None
    max_vol60: Optional[float] = None
    max_vol120: Optional[float] = None


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
    # 新规则：优先使用固定文件名 pred_{pool}.csv
    fixed_pred = os.path.join(pred_dir, f"pred_{pool}.csv")
    if os.path.isfile(fixed_pred):
        return fixed_pred
    patterns = [
        # 预测输出
        os.path.join(pred_dir, f"pred_{pool}_*.csv"),
        os.path.join(pred_dir, f"*{pool}*pred*.csv"),
        # 回测输出/复制
        os.path.join(rq_dir, f"rqalpha_pred_{pool}.csv"),
        os.path.join(rq_dir, f"rqalpha_pred_{pool}_*.csv"),
        os.path.join(rq_dir, f"*{pool}*rqalpha_pred*.csv"),
    ]
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        raise FileNotFoundError(
            f"未找到 {pool} 的预测文件。请先生成预测文件（建议放在 data/predictions/pred_{pool}.csv），"
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


def _build_ts_code_to_industry_map(
    ts_client: Optional[TushareClient],
    *,
    prefer_level: str = "l1",
) -> Dict[str, str]:
    """
    通过 tushare index_member_all 构建 ts_code -> 行业标签 映射。
    若无法获取则返回空 dict（上层会降级为不做行业限制）。
    """
    if ts_client is None:
        return {}
    prefer_level = str(prefer_level).strip().lower()
    try:
        # 尽量只取关键字段；若接口不支持 fields，将回退为全量返回（由客户端缓存）
        df = ts_client.index_member_all(is_new="Y", fields="ts_code,l1_name,l2_name,l3_name,industry,industry_name")
        if df is None or df.empty:
            df = ts_client.index_member_all(is_new="Y")
        if df is None or df.empty:
            return {}
        df.columns = [str(c).strip() for c in df.columns]

        # 兼容股票代码列名
        ts_col = "ts_code" if "ts_code" in df.columns else ("con_code" if "con_code" in df.columns else None)
        if ts_col is None:
            logger.warning("行业限制：index_member_all 返回缺少 ts_code/con_code 列，将跳过同行业限制。cols=%s", list(df.columns))
            return {}

        # 行业列：优先 l1/l2/l3_name，其次 industry/industry_name
        if prefer_level in {"l2", "2"}:
            candidates = ["l2_name", "l1_name", "l3_name", "industry", "industry_name"]
        elif prefer_level in {"l3", "3"}:
            candidates = ["l3_name", "l2_name", "l1_name", "industry", "industry_name"]
        else:
            candidates = ["l1_name", "industry", "industry_name", "l2_name", "l3_name"]
        ind_col = None
        for c in candidates:
            if c in df.columns:
                ind_col = c
                break
        if ind_col is None:
            logger.warning("行业限制：index_member_all 返回缺少行业列，将跳过同行业限制。cols=%s", list(df.columns))
            return {}

        out: Dict[str, str] = {}
        for _, r in df[[ts_col, ind_col]].dropna().iterrows():
            k = str(r[ts_col]).strip()
            v = str(r[ind_col]).strip()
            if k and v:
                out[k] = v
        return out
    except Exception as e:
        logger.warning("行业限制：无法通过 index_member_all 获取行业映射，将跳过同行业限制。err=%s", e)
        return {}


def _pick_with_industry_cap(
    scored: List[Tuple[str, float]],
    *,
    target_holdings: int,
    max_per_industry: int,
    ts_code_to_industry: Dict[str, str],
) -> List[str]:
    """
    按 score 降序遍历候选，应用“每行业最多 N 只”的限制，直到凑够 target_holdings。
    若某只股票行业未知，则不做行业限制（放行）。
    """
    target_holdings = int(target_holdings)
    max_per_industry = int(max_per_industry)
    if target_holdings <= 0:
        return []
    if max_per_industry <= 0 or not ts_code_to_industry:
        return [c for c, _ in scored[:target_holdings]]

    counts: Dict[str, int] = defaultdict(int)
    picks: List[str] = []
    for rq_code, _s in scored:
        if len(picks) >= target_holdings:
            break
        ts_code = rqalpha_to_tushare(rq_code)
        ind = (ts_code_to_industry.get(ts_code) or "").strip()
        if not ind:
            picks.append(rq_code)
            continue
        if counts[ind] >= max_per_industry:
            continue
        picks.append(rq_code)
        counts[ind] += 1
    return picks


def _load_annual_ledger(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, dtype={"rq_code": str, "row_type": str})
        # 兼容历史文件可能出现“列名/值带空格”的情况（例如 'row_type '）
        df.columns = [str(c).strip() for c in df.columns]
        if "row_type" in df.columns:
            df["row_type"] = df["row_type"].astype(str).str.strip()
        if "rq_code" in df.columns:
            df["rq_code"] = df["rq_code"].astype(str).str.strip()
        if "trade_date" in df.columns:
            df["trade_date"] = df["trade_date"].astype(str).str.strip()
        if "signal_date" in df.columns:
            df["signal_date"] = df["signal_date"].astype(str).str.strip()
        if "prev_trade_date" in df.columns:
            df["prev_trade_date"] = df["prev_trade_date"].astype(str).str.strip()
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

    def get_close_history(self, rq_code: str, end_dt: pd.Timestamp, n: int) -> Optional[pd.Series]:
        """
        获取截至 end_dt（含）的最近 n 条收盘价序列（按时间升序）。
        若数据不足返回 None。
        注意：这里按 bundle 内该股票的实际数据条数取样，不依赖交易日历。
        """
        rq_code = str(rq_code).strip()
        if not rq_code or n <= 0:
            return None
        end_dt = pd.Timestamp(end_dt).normalize()
        try:
            self._open()
        except Exception:
            return None
        try:
            ds = self._h5[rq_code]  # type: ignore[index]
        except Exception:
            return None

        cached = self._cache.get(rq_code)
        if cached is None:
            try:
                dts = ds["datetime"][:]
                closes = ds["close"][:]
                self._cache[rq_code] = (dts, closes)
                cached = self._cache[rq_code]
            except Exception:
                return None

        dts, closes = cached
        if len(dts) <= 0:
            return None
        import numpy as np

        target = self._dt_to_int(end_dt)
        # 取 <= end_dt 的最后一个位置
        idx = int(np.searchsorted(dts, target, side="right") - 1)
        if idx < 0:
            return None
        start_idx = max(0, idx - (n - 1))
        sel_dts = dts[start_idx : idx + 1]
        sel_close = closes[start_idx : idx + 1]
        if len(sel_close) < n:
            return None
        # 转为日期索引（yyyymmddHHMMSS -> yyyymmdd）
        dt_index = pd.to_datetime([str(int(x))[:8] for x in sel_dts], format="%Y%m%d", errors="coerce")
        s = pd.Series([float(x) if x is not None else float("nan") for x in sel_close], index=dt_index).sort_index()
        return s


def _bundle_last_trade_day(bundle_path: str) -> Optional[pd.Timestamp]:
    """
    从 RQAlpha bundle 的 trading_dates.npy 推断最后交易日。
    返回 None 表示读取失败。
    """
    try:
        import numpy as np
        import pathlib

        p = pathlib.Path(bundle_path) / "trading_dates.npy"
        if not p.exists():
            return None
        arr = np.load(str(p)).astype(int)
        if arr.size <= 0:
            return None
        last = int(arr.max())
        return pd.to_datetime(str(last), format="%Y%m%d", errors="coerce").normalize()
    except Exception:
        return None


def _trading_calendar_from_bundle(bundle_path: str, *, start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.DatetimeIndex]:
    """优先使用 RQAlpha bundle 的 trading_dates.npy 获取交易日序列。"""
    try:
        import numpy as np
        import pathlib

        p = pathlib.Path(bundle_path) / "trading_dates.npy"
        if not p.exists():
            return None
        arr = np.load(str(p)).astype(int)
        if arr.size <= 0:
            return None
        days = pd.to_datetime(pd.Series(arr.astype(int).astype(str)), format="%Y%m%d", errors="coerce").dropna()
        s = pd.Timestamp(start).normalize()
        e = pd.Timestamp(end).normalize()
        days = days[(days >= s) & (days <= e)].dt.normalize()
        out = pd.DatetimeIndex(days.sort_values().unique())
        return out if len(out) > 0 else None
    except Exception:
        return None


def _annualized_vol_from_closes(closes: pd.Series) -> Optional[float]:
    """
    年化历史波动率 = std(每日收益率) * sqrt(252)
    closes: 按日期升序的收盘价序列
    """
    closes = closes.dropna()
    if len(closes) < 3:
        return None
    rets = closes.pct_change().dropna()
    if len(rets) < 2:
        return None
    vol = float(rets.std(ddof=1) * math.sqrt(252))
    if math.isnan(vol) or math.isinf(vol):
        return None
    return vol


def _annualized_vol_from_close_mat(
    close_mat: Optional[pd.DataFrame],
    *,
    ts_code: str,
    window: int,
) -> Optional[float]:
    """从 close 矩阵中取列并计算末尾 window 的年化波动率。"""
    if close_mat is None or close_mat.empty:
        return None
    if ts_code not in close_mat.columns:
        return None
    s = close_mat[ts_code].dropna()
    window = int(window)
    if window <= 1 or len(s) < window:
        return None
    return _annualized_vol_from_closes(s.tail(window))


def _compute_vols_for_codes(
    codes: List[str],
    signal_date: pd.Timestamp,
    *,
    bundle_path: str,
    price: _BundlePriceFetcher,
    windows: Tuple[int, int, int] = (20, 60, 120),
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    计算每只股票的历史年化波动率（N=20/60/120）。
    返回：{code: {"vol20":..., "vol60":..., "vol120":...}}
    """
    signal_date = pd.Timestamp(signal_date).normalize()
    max_n = max([int(x) for x in windows if x and int(x) > 0] or [0])

    # 只依赖 RQAlpha bundle 行情数据计算波动率。
    # 做法：对每只股票，直接取 bundle 中“截至 signal_date 的最近 N 条收盘价”来计算波动率。
    vols: Dict[str, Dict[str, Optional[float]]] = {}
    for code in codes:
        out: Dict[str, Optional[float]] = {"vol20": None, "vol60": None, "vol120": None}
        for n, key in zip(windows, ["vol20", "vol60", "vol120"]):
            if n <= 0:
                continue
            s = price.get_close_history(code, signal_date, int(n))
            out[key] = _annualized_vol_from_closes(s) if s is not None else None
        vols[code] = out
    return vols


def _compute_vols_for_codes_tushare(
    codes: List[str],
    signal_date: pd.Timestamp,
    *,
    ts_client: TushareClient,
    windows: Tuple[int, int, int] = (20, 60, 120),
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    用 Tushare 的 close（daily_basic/daily）计算历史年化波动率。
    逻辑：先取最大窗口 max_n 的 close 矩阵，再对每只股票截取末尾 N 计算 vol。
    """
    signal_date = pd.Timestamp(signal_date).normalize()
    max_n = max([int(x) for x in windows if x and int(x) > 0] or [0])
    if max_n <= 0 or not codes:
        return {}
    ts_codes = [rqalpha_to_tushare(c) for c in codes]
    close_mat = ts_client.close_history_last_n(ts_codes, signal_date, max_n)
    vols: Dict[str, Dict[str, Optional[float]]] = {}
    for rq, ts_code in zip(codes, ts_codes):
        out: Dict[str, Optional[float]] = {"vol20": None, "vol60": None, "vol120": None}
        if close_mat is not None and not close_mat.empty and ts_code in close_mat.columns:
            s_all = close_mat[ts_code].dropna()
        else:
            s_all = pd.Series(dtype=float)
        for n, key in zip(windows, ["vol20", "vol60", "vol120"]):
            n = int(n)
            if n <= 0:
                continue
            if len(s_all) < n:
                out[key] = None
            else:
                out[key] = _annualized_vol_from_closes(s_all.tail(n))
        vols[rq] = out
    return vols


def _select_scheme_c_weights(
    *,
    scored: List[Tuple[str, float]],
    vols_map: Dict[str, Dict[str, Optional[float]]],
    total_weight: float,
    target_holdings: int,
    preselect_topm: int,
    vol_window: int,
    vol_max: Optional[float],
    vol_eps: float,
) -> Tuple[List[str], Dict[str, float]]:
    """
    方案C：
      - TopM（按 score）候选
      - 在候选中按 vol_window 的年化波动率升序取 K
      - 权重按风险预算：w_i ∝ 1/(eps + vol_i)，归一化到 total_weight
    """
    if total_weight <= 0 or target_holdings <= 0 or not scored:
        return [], {}
    m = int(preselect_topm) if preselect_topm and int(preselect_topm) > 0 else len(scored)
    topm = scored[: min(len(scored), m)]

    # 使用 vols_map 里对应窗口（20/60/120）作为 vol_window；若 vol_window 非这三者，外部应已补齐
    key = f"vol{int(vol_window)}"
    cands: List[Tuple[str, float]] = []
    for code, _s in topm:
        v = (vols_map.get(code, {}) or {}).get(key)
        if v is None:
            continue
        try:
            vf = float(v)
        except Exception:
            continue
        if math.isnan(vf) or math.isinf(vf):
            continue
        if vol_max is not None and vf > float(vol_max):
            continue
        cands.append((code, vf))

    if len(cands) < min(target_holdings, 2):
        return [], {}

    cands.sort(key=lambda x: x[1])  # 低波动优先
    chosen = cands[: min(len(cands), target_holdings)]

    raw: Dict[str, float] = {}
    for code, v in chosen:
        raw[code] = 1.0 / (float(vol_eps) + float(v))
    s_raw = sum(raw.values())
    if s_raw <= 0:
        return [], {}
    weights = {c: (w / s_raw) * float(total_weight) for c, w in raw.items()}
    picks = [c for c, _ in chosen]
    return picks, weights


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
    pred_dt: pd.Timestamp,
    signal_date: pd.Timestamp,
    book: PredictionBook,
    ts_client: Optional[TushareClient],
    *,
    bundle_path: str,
    price_fetcher: Optional[_BundlePriceFetcher],
    vol_drop_if_missing: bool = False,
    vol_source: str = "tushare",
    industry_cap: int = 2,
    industry_level: str = "l1",
) -> Tuple[List[str], Dict[str, float], Dict[str, Dict[str, Optional[float]]], Dict[str, float]]:
    """
    返回：
    - picks：最终持仓列表（rq_code）
    - signals：当日信号字典（rq_code->score），用于输出明细
    - vols_out：最终持仓的 vol20/60/120（若可得）
    - sub_weights：子策略内目标权重（合计为 sub.allocation；若为空则表示“用等权回退”）
    """
    # pred_dt 表示“预测文件 datetime 的语义对应的日期键”：
    # - 若预测文件按 signal_date 存储：pred_dt == signal_date
    # - 若预测文件按 trade_date 存储（已 shift）：pred_dt == trade_date
    # signal_date 仍用于过滤/波动率计算，避免隐性未来信息。
    signals = book.get(pred_dt)
    # 候选池至少取 Top100，保证后续“顺延回填”有足够候选（硬编码在逻辑里）
    topn = max(int(sub.topk_pred) if sub.topk_pred is not None else 0, 100)
    cand_raw = list(topk(signals, topn).keys())
    cand = apply_basic_filters(cand_raw, signal_date, sub.filter_cfg, ts_client)
    vols: Dict[str, Dict[str, Optional[float]]] = {c: {"vol20": None, "vol60": None, "vol120": None} for c in cand}

    # 历史波动率过滤（年化）：过去 N 日收益率标准差 * sqrt(252)
    need_vol = sub.max_vol20 is not None or sub.max_vol60 is not None or sub.max_vol120 is not None
    want_scheme_c = (str(sub.selection_mode).strip().lower() in {"scheme_c", "c", "vol"})
    if (need_vol or want_scheme_c) and cand:
        # 为了支持 scheme_c，我们默认把 20/60/120 都尽量算出来（后续也会写入台账）。
        # 数据源优先级：按 --vol-source 指定，其次 fallback。
        use_ts = (str(vol_source).strip().lower() == "tushare")
        use_bundle = (str(vol_source).strip().lower() == "bundle")
        if use_ts and ts_client is not None and hasattr(ts_client, "close_history_last_n"):
            vols = _compute_vols_for_codes_tushare(cand, signal_date, ts_client=ts_client)
        elif use_bundle and price_fetcher is not None:
            vols = _compute_vols_for_codes(cand, signal_date, bundle_path=bundle_path, price=price_fetcher)
        else:
            # fallback：能用 tushare 就用 tushare，否则 bundle
            if ts_client is not None and hasattr(ts_client, "close_history_last_n"):
                vols = _compute_vols_for_codes_tushare(cand, signal_date, ts_client=ts_client)
            elif price_fetcher is not None:
                vols = _compute_vols_for_codes(cand, signal_date, bundle_path=bundle_path, price=price_fetcher)
        # scheme_c 强依赖波动率：若此时仍然拿不到任何波动率，明确提示用户环境/数据源问题
        if want_scheme_c:
            try:
                any_vol = False
                for _c in cand:
                    vv = vols.get(_c, {}) if isinstance(vols, dict) else {}
                    if (vv.get("vol20") is not None) or (vv.get("vol60") is not None) or (vv.get("vol120") is not None):
                        any_vol = True
                        break
                if not any_vol:
                    logger.warning(
                        "方案C需要历史波动率数据，但当前无法获取（vol_source=%s，tushare=%s，bundle_price_fetcher=%s）。"
                        "将回退为等权TopN。建议：1) pip install tushare 并配置 token；或 2) pip install h5py 并确保 --bundle-path 指向含 stocks.h5 的RQAlpha bundle。",
                        vol_source,
                        "ok" if (ts_client is not None and hasattr(ts_client, "close_history_last_n")) else "unavailable",
                        "ok" if price_fetcher is not None else "unavailable",
                    )
            except Exception:
                pass
        kept: List[str] = []
        for c in cand:
            v = vols.get(c, {})
            v20 = v.get("vol20")
            v60 = v.get("vol60")
            v120 = v.get("vol120")
            # 缺失处理：strict 则剔除；否则仅在有值时判断阈值
            if sub.max_vol20 is not None:
                if v20 is None:
                    if vol_drop_if_missing:
                        continue
                elif float(v20) > float(sub.max_vol20):
                    continue
            if sub.max_vol60 is not None:
                if v60 is None:
                    if vol_drop_if_missing:
                        continue
                elif float(v60) > float(sub.max_vol60):
                    continue
            if sub.max_vol120 is not None:
                if v120 is None:
                    if vol_drop_if_missing:
                        continue
                elif float(v120) > float(sub.max_vol120):
                    continue
            kept.append(c)
        cand = kept
    scored = [(c, signals.get(c, float("-inf"))) for c in cand]
    scored.sort(key=lambda x: x[1], reverse=True)

    # === 方案C：TopM -> 低波动 -> 风险预算权重 ===
    sub_weights: Dict[str, float] = {}
    if want_scheme_c and scored:
        # vol_window 若不在 {20,60,120}，尝试额外计算该窗口并塞进 vols_map（键名 vol{N}）
        vw = int(sub.vol_window) if int(sub.vol_window) > 0 else 20
        if vw not in (20, 60, 120) and cand:
            try:
                # 优先 tushare：一次性取 max_n 的 close，再按 vw 计算
                use_ts = (str(vol_source).strip().lower() == "tushare")
                use_bundle = (str(vol_source).strip().lower() == "bundle")
                if use_ts and ts_client is not None and hasattr(ts_client, "close_history_last_n"):
                    ts_codes = [rqalpha_to_tushare(c) for c in cand]
                    close_mat = ts_client.close_history_last_n(ts_codes, signal_date, vw)
                    for rq, ts_code in zip(cand, ts_codes):
                        vv = _annualized_vol_from_close_mat(close_mat, ts_code=ts_code, window=vw)
                        vols.setdefault(rq, {})[f"vol{vw}"] = vv
                elif use_bundle and price_fetcher is not None:
                    for rq in cand:
                        s = price_fetcher.get_close_history(rq, signal_date, vw)
                        vols.setdefault(rq, {})[f"vol{vw}"] = _annualized_vol_from_closes(s) if s is not None else None
            except Exception:
                pass

        picks_c, w_c = _select_scheme_c_weights(
            scored=scored,
            vols_map=vols,
            total_weight=float(sub.allocation),
            target_holdings=int(sub.target_holdings),
            preselect_topm=int(sub.preselect_topm),
            vol_window=vw,
            vol_max=sub.vol_max,
            vol_eps=float(sub.vol_eps),
        )
        if picks_c and w_c:
            picks = picks_c
            sub_weights = w_c
        else:
            picks = [c for c, _ in scored[: sub.target_holdings]]
    else:
        picks = [c for c, _ in scored[: sub.target_holdings]]

    # === 同行业限制：每行业最多 N 只（超限则顺延选择下一只） ===
    # 规则：只要 tushare 可用且 industry_cap>0，就用 index_member_all 映射行业并执行限制。
    # 若接口不可用/映射缺失，则自动跳过该限制（保证脚本可运行）。
    industry_cap = int(industry_cap) if industry_cap is not None else 0
    if industry_cap > 0 and ts_client is not None and scored:
        ts_code_to_industry = _build_ts_code_to_industry_map(ts_client, prefer_level=industry_level)
        if ts_code_to_industry:
            new_picks = _pick_with_industry_cap(
                scored,
                target_holdings=int(sub.target_holdings),
                max_per_industry=industry_cap,
                ts_code_to_industry=ts_code_to_industry,
            )
            if new_picks and new_picks != picks:
                logger.info(
                    "行业限制生效：子策略 %s 持仓从 %d 调整为 %d（cap=%d，level=%s）",
                    sub.name,
                    len(picks),
                    len(new_picks),
                    industry_cap,
                    industry_level,
                )
                picks = new_picks
                # 若之前是方案C且已有子权重，尝试按波动率重算风险预算权重；否则回退等权
                if want_scheme_c:
                    vw = int(sub.vol_window) if int(sub.vol_window) > 0 else 20
                    key = f"vol{vw}"
                    raw: Dict[str, float] = {}
                    for code in picks:
                        v = (vols.get(code, {}) or {}).get(key)
                        if v is None:
                            continue
                        try:
                            vf = float(v)
                        except Exception:
                            continue
                        if math.isnan(vf) or math.isinf(vf):
                            continue
                        raw[code] = 1.0 / (float(sub.vol_eps) + float(vf))
                    s_raw = sum(raw.values())
                    if s_raw > 0 and len(raw) == len(picks):
                        sub_weights = {c: (w / s_raw) * float(sub.allocation) for c, w in raw.items()}
                    else:
                        sub_weights = {}

    # === 补足持仓数：若过滤/行业cap后不足 target_holdings，则从 TopK 排名继续补 ===
    # 这里“回填”会放松 PB/近N日涨停限制（仍保留基础过滤与行业cap），以提高输出数量的稳定性。
    target_n = int(sub.target_holdings)
    if target_n > 0 and len(picks) < target_n and cand_raw:
        before_n = len(picks)
        relaxed_cfg = copy.deepcopy(sub.filter_cfg)
        try:
            relaxed_cfg.pb_min = None
            relaxed_cfg.pb_max = None
            relaxed_cfg.exclude_recent_limitup_days = 0
        except Exception:
            pass
        cand_relaxed = apply_basic_filters(cand_raw, signal_date, relaxed_cfg, ts_client)
        scored_relaxed = [(c, signals.get(c, float("-inf"))) for c in cand_relaxed]
        scored_relaxed.sort(key=lambda x: x[1], reverse=True)

        # 行业映射/计数
        ts_code_to_industry2: Dict[str, str] = {}
        if industry_cap > 0 and ts_client is not None:
            ts_code_to_industry2 = _build_ts_code_to_industry_map(ts_client, prefer_level=industry_level)
        counts: Dict[str, int] = defaultdict(int)
        if industry_cap > 0 and ts_code_to_industry2:
            for rq in picks:
                ts_code = rqalpha_to_tushare(rq)
                ind = (ts_code_to_industry2.get(ts_code) or "").strip()
                if ind:
                    counts[ind] += 1

        existing = set(picks)
        for rq_code, _s in scored_relaxed:
            if len(picks) >= target_n:
                break
            if rq_code in existing:
                continue
            if industry_cap > 0 and ts_code_to_industry2:
                ts_code = rqalpha_to_tushare(rq_code)
                ind = (ts_code_to_industry2.get(ts_code) or "").strip()
                if ind and counts[ind] >= industry_cap:
                    continue
                if ind:
                    counts[ind] += 1
            picks.append(rq_code)
            existing.add(rq_code)

        # 回填会改变 picks，为避免“权重与 picks 不一致”，这里让权重回退为等权
        if len(picks) != before_n:
            sub_weights = {}
            logger.info(
                "子策略 %s 持仓数不足，已从 TopK 回填：%d -> %d（target=%d）",
                sub.name,
                before_n,
                len(picks),
                target_n,
            )
        if len(picks) < target_n:
            logger.warning(
                "子策略 %s 回填后仍不足 target_holdings=%d，实际=%d（可能是过滤过严/行业cap过小/TopK过小）",
                sub.name,
                target_n,
                len(picks),
            )

    # 回退等权（保证合计为 allocation）
    if not sub_weights and picks:
        w = float(sub.allocation) / float(len(picks))
        sub_weights = {c: w for c in picks}

    # 只返回“最终候选相关”的波动率，便于写入 POSITION（仍保留 vol20/60/120）
    vols_out = {c: vols.get(c, {"vol20": None, "vol60": None, "vol120": None}) for c in picks}
    return picks, signals, vols_out, sub_weights


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
    # selection mode (shared)
    p.add_argument(
        "--msa-selection-mode",
        type=str,
        default="scheme_c",
        choices=["equal_weight", "scheme_c"],
        help="子策略内部选股/权重：equal_weight（旧：等权）/ scheme_c（TopM->低波动->风险预算）",
    )
    p.add_argument("--msa-preselect-topm", type=int, default=50, help="方案C：先按信号取 TopM 候选")
    p.add_argument("--msa-vol-window", type=int, default=20, help="方案C：波动率窗口（天），默认20")
    p.add_argument("--msa-vol-max", type=float, default=None, help="方案C：年化波动率上限（可选，超过剔除）")
    p.add_argument("--msa-vol-eps", type=float, default=0.05, help="方案C：风险预算 eps，w∝1/(eps+vol)")
    # s1
    # 为了确保“同行业cap + 各类过滤”后仍能凑够持仓数，默认把 topk 放大、目标持仓提升到10
    p.add_argument("--s1-topk", type=int, default=100)
    p.add_argument("--s1-hold", type=int, default=10)
    p.add_argument("--s1-min-list-days", type=int, default=360)
    # s2
    p.add_argument("--s2-topk", type=int, default=100)
    p.add_argument("--s2-hold", type=int, default=10)
    p.add_argument("--s2-min-list-days", type=int, default=360)
    p.add_argument("--s2-pb-min", type=float, default=0.0)
    p.add_argument("--s2-pb-max", type=float, default=1.0)
    p.add_argument("--s2-limitup-days", type=int, default=5)
    # 历史波动率过滤（年化）
    p.add_argument("--s1-max-vol20", type=float, default=None, help="策略1：20日年化波动率上限（如0.8表示80%%），不填则不启用")
    p.add_argument("--s1-max-vol60", type=float, default=None, help="策略1：60日年化波动率上限")
    p.add_argument("--s1-max-vol120", type=float, default=None, help="策略1：120日年化波动率上限")
    p.add_argument("--s2-max-vol20", type=float, default=None, help="策略2：20日年化波动率上限")
    p.add_argument("--s2-max-vol60", type=float, default=None, help="策略2：60日年化波动率上限")
    p.add_argument("--s2-max-vol120", type=float, default=None, help="策略2：120日年化波动率上限")
    p.add_argument("--vol-drop-if-missing", action="store_true", help="若历史收盘价不足导致波动率无法计算，则剔除该股票（更严格）")
    p.add_argument(
        "--vol-source",
        type=str,
        default="tushare",
        choices=["tushare", "bundle"],
        help="波动率数据源：tushare（使用daily_basic/daily close，推荐）或 bundle（使用RQAlpha stocks.h5）",
    )
    # 行业限制：每行业最多选 N 只（依赖 tushare index_member_all；不可用时自动跳过）
    p.add_argument("--industry-cap", type=int, default=2, help="同行业最多选择 N 只股票（默认2；0表示关闭）")
    p.add_argument("--industry-level", type=str, default="l1", choices=["l1", "l2", "l3"], help="行业层级：l1/l2/l3（默认l1）")
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

    # 推断/指定预测日期语义（注意：prediction_loader 支持按该语义决定是否“反向还原”为 signal_date）
    pred_dates_are = str(args.pred_dates_are).strip().lower()
    if pred_dates_are == "auto":
        # 两份文件只要有一个明确标注为 trade_date，我们就按 trade_date 处理
        p1 = _infer_pred_dates_are(pred_csi101, default="signal_date")
        p2 = _infer_pred_dates_are(pred_csi300, default="signal_date")
        pred_dates_are = "trade_date" if ("trade_date" in {p1, p2}) else "signal_date"

    book101 = load_prediction_csv(pred_csi101, dates_are=pred_dates_are)
    book300 = load_prediction_csv(pred_csi300, dates_are=pred_dates_are) if pred_csi300 else None

    # 选用共同可用的最大日期（“最后一条预测”），避免某一份缺日期
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
        # 默认使用最后一条预测日期
        asof = pd.Timestamp(max_common).normalize()

    if pred_dates_are == "trade_date":
        # 预测文件的 datetime 代表 trade_date：最后一条预测就是“下一个交易日的交易建议日”
        trade_date = asof
        signal_date = _prev_trading_day(trade_date)
        pred_dt = trade_date
    else:
        # 预测文件的 datetime 代表 signal_date：用最后一条预测生成“下一交易日”建议
        signal_date = asof
        trade_date = _next_trading_day(signal_date)
        pred_dt = signal_date

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
            selection_mode=str(args.msa_selection_mode),
            preselect_topm=int(args.msa_preselect_topm),
            vol_window=int(args.msa_vol_window),
            vol_max=args.msa_vol_max,
            vol_eps=float(args.msa_vol_eps),
            max_vol20=args.s2_max_vol20,
            max_vol60=args.s2_max_vol60,
            max_vol120=args.s2_max_vol120,
        )

    # Tushare（可选）
    ts_client = TushareClient.try_create(cache_dir=os.path.join(_PROJECT_ROOT, "data", "tushare_cache"))
    if ts_client is None:
        logger.warning("未启用 Tushare（未设置 token 或初始化失败），将跳过 ST/PB/涨停等相关过滤")

    # 子策略选股
    # 为波动率过滤准备 bundle price fetcher（作为 tushare 不可用时的回退）
    bundle_path_for_vol = args.bundle_path
    if not bundle_path_for_vol:
        import pathlib
        bundle_path_for_vol = str(pathlib.Path.home() / ".rqalpha" / "bundle")
    bundle_path_for_vol = str(bundle_path_for_vol).replace("/", os.sep)
    vol_price_fetcher: Optional[_BundlePriceFetcher] = None
    if os.path.exists(os.path.join(bundle_path_for_vol, "stocks.h5")):
        try:
            # 显式检查依赖：读取 bundle 需要 h5py
            import h5py  # type: ignore  # noqa: F401
            vol_price_fetcher = _BundlePriceFetcher(bundle_path_for_vol)
        except Exception:
            vol_price_fetcher = None
    else:
        need_any_vol = (
            str(args.msa_selection_mode).strip().lower() in {"scheme_c", "c", "vol"}
            or s1.max_vol20 is not None
            or s1.max_vol60 is not None
            or s1.max_vol120 is not None
            or (s2 and (s2.max_vol20 is not None or s2.max_vol60 is not None or s2.max_vol120 is not None))
        )
        if need_any_vol:
            logger.warning("未找到 RQAlpha bundle 的 stocks.h5（%s），将跳过历史波动率过滤", bundle_path_for_vol)
    need_any_vol = (
        str(args.msa_selection_mode).strip().lower() in {"scheme_c", "c", "vol"}
        or s1.max_vol20 is not None
        or s1.max_vol60 is not None
        or s1.max_vol120 is not None
        or (s2 and (s2.max_vol20 is not None or s2.max_vol60 is not None or s2.max_vol120 is not None))
    )
    if vol_price_fetcher is None and need_any_vol and str(args.vol_source).strip().lower() == "bundle":
        logger.warning(
            "已指定 --vol-source=bundle，但无法初始化 bundle 读取器（请确认安装 h5py 且 bundle 路径正确：%s）。",
            bundle_path_for_vol,
        )

    picks1, sig1, vols1, w1 = _select_for_substrategy(
        s1,
        pred_dt,
        signal_date,
        book101,
        ts_client,
        bundle_path=bundle_path_for_vol,
        price_fetcher=vol_price_fetcher,
        vol_drop_if_missing=bool(args.vol_drop_if_missing),
        vol_source=str(args.vol_source),
        industry_cap=int(args.industry_cap),
        industry_level=str(args.industry_level),
    )
    picks2, sig2, vols2, w2 = [], {}, {}, {}
    if s2 is not None and book300 is not None:
        picks2, sig2, vols2, w2 = _select_for_substrategy(
            s2,
            pred_dt,
            signal_date,
            book300,
            ts_client,
            bundle_path=bundle_path_for_vol,
            price_fetcher=vol_price_fetcher,
            vol_drop_if_missing=bool(args.vol_drop_if_missing),
            vol_source=str(args.vol_source),
            industry_cap=int(args.industry_cap),
            industry_level=str(args.industry_level),
        )

    # 诊断：若因过滤/行业cap导致不足目标持仓数，给出明确提示
    if s1 is not None and int(s1.target_holdings) > 0 and len(picks1) < int(s1.target_holdings):
        logger.warning(
            "策略1(%s) 最终持仓数不足：目标=%d，实际=%d。可能原因：过滤条件过严/行业cap过严/topk过小/当日信号覆盖不足。",
            s1.name,
            int(s1.target_holdings),
            len(picks1),
        )
    if s2 is not None and int(s2.target_holdings) > 0 and len(picks2) < int(s2.target_holdings):
        logger.warning(
            "策略2(%s) 最终持仓数不足：目标=%d，实际=%d。可能原因：过滤条件过严/行业cap过严/topk过小/当日信号覆盖不足。",
            s2.name,
            int(s2.target_holdings),
            len(picks2),
        )

    if vol_price_fetcher is not None:
        vol_price_fetcher.close()

    if not picks1 and not picks2:
        raise RuntimeError("两子策略均未选出股票（可能全部被过滤/当天无信号）")

    # 构建权重：子策略内部按 selection_mode（可能是风险预算权重），组合层按 allocation 合并
    w1 = w1 or {}
    w2 = w2 or {}

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
    bundle_last = _bundle_last_trade_day(bundle_path)
    if bundle_last is not None and pd.Timestamp(trade_date).normalize() > bundle_last:
        logger.warning(
            "RQAlpha bundle 交易日历最后一天=%s 早于本次 trade_date=%s：无法计算 close-to-close 盈亏/归因（将只记录调仓快照）。"
            "请更新 bundle 或设置 --bundle-path 指向新 bundle。",
            bundle_last.date(),
            pd.Timestamp(trade_date).date(),
        )

    period_return = 0.0
    per_stock_return: Dict[str, Dict[str, float]] = {}
    nav_before = float(last_nav)
    nav_after = float(last_nav)
    turnover = 0.0
    cost = 0.0

    price_fetcher = None
    try:
        if (
            last_dt is not None
            and last_holdings
            and os.path.exists(os.path.join(bundle_path, "stocks.h5"))
            and (bundle_last is None or (pd.Timestamp(trade_date).normalize() <= bundle_last and pd.Timestamp(last_dt).normalize() <= bundle_last))
        ):
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
            "note": "close_to_close on bundle; cost=turnover*cost_rate; bundle_last=%s"
            % (bundle_last.date() if bundle_last is not None else "unknown"),
        }
    )

    # 合并两子策略的波动率结果，写入 POSITION 行
    vol_map: Dict[str, Dict[str, Optional[float]]] = {}
    try:
        vol_map.update(vols1 or {})
        vol_map.update(vols2 or {})
    except Exception:
        pass

    def _add_rows(sub: SubStrategyConfig, picks: List[str], signals: Dict[str, float], sub_weights: Dict[str, float]):
        # rank：在 topk_pred 内的排名（仅用于展示）
        top = list(topk(signals, sub.topk_pred).items())
        top.sort(key=lambda x: x[1], reverse=True)
        rank_map = {code: i + 1 for i, (code, _) in enumerate(top)}
        for code in picks:
            prev_w = float(last_holdings.get(code, 0.0)) if last_dt is not None else 0.0
            new_w = float(merged.get(code, 0.0))
            per_ret = per_stock_return.get(code, {})
            vv = vol_map.get(code, {}) if isinstance(vol_map, dict) else {}
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
                    "vol20": vv.get("vol20") if vv else None,
                    "vol60": vv.get("vol60") if vv else None,
                    "vol120": vv.get("vol120") if vv else None,
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


