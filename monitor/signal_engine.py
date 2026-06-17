from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from monitor.ts_client import get_ts_client

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    code: str
    name: str
    signal_type: str
    score: float
    rank: int
    strategy: str
    reason: str
    date: str
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "signal_type": self.signal_type,
            "score": self.score,
            "rank": self.rank,
            "strategy": self.strategy,
            "reason": self.reason,
            "date": self.date,
            "extra": self.extra,
        }


class SignalEngine:
    def __init__(self, config_path: str = "config/monitor.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self._stock_name_map: Dict[str, str] = {}

    def _load_config(self) -> Dict[str, Any]:
        p = Path(self.config_path)
        if not p.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _load_stock_names(self) -> Dict[str, str]:
        if self._stock_name_map:
            return self._stock_name_map
        
        ts = get_ts_client()
        if ts is None:
            return {}
        
        try:
            sb = ts.stock_basic()
            name_map = {}
            for _, row in sb.iterrows():
                ts_code = str(row.get("ts_code", ""))
                name = str(row.get("name", ""))
                if ts_code and name:
                    code = ts_code.split(".")[0]
                    name_map[code] = name
            self._stock_name_map = name_map
            return name_map
        except Exception as e:
            logger.warning(f"加载股票名称失败: {e}")
            return {}

    def _get_stock_name(self, code: str) -> str:
        code_str = str(code).replace(".XSHE", "").replace(".XSHG", "")
        code_clean = code_str.zfill(6)
        return self._stock_name_map.get(code_clean, code_str)

    def load_predictions(self, pool: str, date: Optional[str] = None) -> pd.DataFrame:
        pred_dir = Path(self.config.get("paths", {}).get("prediction_dir", "data/predictions"))
        pred_file = pred_dir / f"pred_{pool}.csv"
        
        if not pred_file.exists():
            raise FileNotFoundError(f"预测文件不存在: {pred_file}")
        
        df = pd.read_csv(pred_file)
        df["datetime"] = pd.to_datetime(df["datetime"])
        
        if "instrument" in df.columns:
            df["instrument"] = df["instrument"].astype(str).str.zfill(6)
        
        if date:
            target_date = pd.to_datetime(date)
            df = df[df["datetime"] == target_date]
        
        if df.empty:
            raise ValueError(f"没有找到预测数据: pool={pool}, date={date}")
        
        return df

    def get_latest_date(self, pool: str) -> str:
        df = self.load_predictions(pool)
        latest = df["datetime"].max()
        return latest.strftime("%Y-%m-%d")

    def apply_buy_rules(self, df: pd.DataFrame, rules: List[Dict], dt: pd.Timestamp) -> List[str]:
        ts = get_ts_client()
        codes = [str(c) for c in df["instrument"].tolist()]
        
        for rule in rules:
            rule_type = rule.get("type")
            
            if rule_type == "top_k":
                k = rule.get("k", 20)
                df = df.nlargest(k, "final")
                codes = [str(c) for c in df["instrument"].tolist()]
                
            elif rule_type == "final_top_k":
                k = rule.get("k", 6)
                df = df.nlargest(k, "final")
                codes = [str(c) for c in df["instrument"].tolist()]
                break
                
            elif rule_type == "exclude_board":
                boards = rule.get("boards", ["科创板", "北交所"])
                filtered = []
                for code in codes:
                    code_clean = str(code).replace(".XSHE", "").replace(".XSHG", "").zfill(6)
                    if code_clean.startswith("688") and "科创板" in boards:
                        continue
                    if code_clean.startswith("8") and len(code_clean) == 6 and "北交所" in boards:
                        continue
                    if code_clean.startswith("4") and len(code_clean) == 6 and "北交所" in boards:
                        continue
                    filtered.append(code)
                codes = filtered
                df = df[df["instrument"].astype(str).isin(codes)]
                
            elif rule_type == "exclude_st":
                if ts is None:
                    continue
                try:
                    sb = ts.stock_basic()
                    st_codes = set()
                    for _, row in sb.iterrows():
                        name = str(row.get("name", ""))
                        ts_code = str(row.get("ts_code", ""))
                        if "ST" in name or "*ST" in name or "退" in name:
                            st_codes.add(ts_code.split(".")[0])
                    codes = [c for c in codes if str(c).replace(".XSHE", "").replace(".XSHG", "").zfill(6) not in st_codes]
                    df = df[df["instrument"].astype(str).isin(codes)]
                except Exception as e:
                    logger.warning(f"排除ST股票失败: {e}")
                    
            elif rule_type == "min_list_days":
                min_days = rule.get("days", 60)
                if ts is None:
                    continue
                try:
                    sb = ts.stock_basic()
                    valid_codes = set()
                    for _, row in sb.iterrows():
                        ts_code = str(row.get("ts_code", ""))
                        list_date = str(row.get("list_date", ""))
                        code = ts_code.split(".")[0]
                        if list_date:
                            try:
                                ld = pd.to_datetime(list_date)
                                if (dt - ld).days >= min_days:
                                    valid_codes.add(code)
                            except:
                                pass
                    codes = [c for c in codes if str(c).replace(".XSHE", "").replace(".XSHG", "").zfill(6) in valid_codes]
                    df = df[df["instrument"].astype(str).isin(codes)]
                except Exception as e:
                    logger.warning(f"最小上市天数过滤失败: {e}")
                    
            elif rule_type == "pb_range":
                pb_min = rule.get("min")
                pb_max = rule.get("max")
                if ts is None:
                    continue
                try:
                    db = ts.daily_basic(dt)
                    pb_map = db.set_index("ts_code")["pb"].to_dict()
                    valid_codes = []
                    for code in codes:
                        code_clean = str(code).replace(".XSHE", "").replace(".XSHG", "").zfill(6)
                        suffix = ".SZ" if code_clean.startswith(("0", "3")) else ".SH"
                        ts_code = code_clean + suffix
                        pb = pb_map.get(ts_code)
                        if pb is None:
                            continue
                        try:
                            pb_f = float(pb)
                            if pb_min is not None and pb_f <= pb_min:
                                continue
                            if pb_max is not None and pb_f >= pb_max:
                                continue
                            valid_codes.append(code)
                        except:
                            continue
                    codes = valid_codes
                    df = df[df["instrument"].astype(str).isin(codes)]
                except Exception as e:
                    logger.warning(f"PB过滤失败: {e}")
                    
            elif rule_type == "exclude_recent_limit_up":
                days = rule.get("days", 5)
                if ts is None:
                    continue
                try:
                    limitup_codes = set()
                    for i in range(days):
                        d = dt - pd.Timedelta(days=i)
                        try:
                            ll = ts.limit_list(d, limit_type="U")
                            if not ll.empty and "ts_code" in ll.columns:
                                for ts_code in ll["ts_code"].tolist():
                                    limitup_codes.add(str(ts_code).split(".")[0])
                        except:
                            continue
                    codes = [c for c in codes if str(c).replace(".XSHE", "").replace(".XSHG", "").zfill(6) not in limitup_codes]
                    df = df[df["instrument"].astype(str).isin(codes)]
                except Exception as e:
                    logger.warning(f"排除近期涨停股失败: {e}")
        
        return codes

    def generate_signals(self, pool: str, date: Optional[str] = None) -> List[Signal]:
        strategy_config = self.config.get("strategies", {}).get(pool, {})
        if not strategy_config.get("enabled", True):
            logger.info(f"策略 {pool} 未启用")
            return []
        
        self._load_stock_names()
        
        df = self.load_predictions(pool, date)
        dt = df["datetime"].iloc[0]
        date_str = dt.strftime("%Y-%m-%d")
        
        df = df.sort_values("final", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
        
        buy_rules = strategy_config.get("rules", {}).get("buy", [])
        buy_codes = self.apply_buy_rules(df.copy(), buy_rules, dt)
        
        signals = []
        for code in buy_codes:
            row = df[df["instrument"].astype(str) == str(code)].iloc[0]
            signal = Signal(
                code=str(code),
                name=self._get_stock_name(code),
                signal_type="buy",
                score=float(row["final"]),
                rank=int(row["rank"]),
                strategy=pool,
                reason=f"预测分数排名 {row['rank']}, 分数 {row['final']:.4f}",
                date=date_str,
            )
            signals.append(signal)
        
        logger.info(f"生成买入信号: pool={pool}, date={date_str}, count={len(signals)}")
        return signals

    def generate_all_signals(self, date: Optional[str] = None) -> Dict[str, List[Signal]]:
        pools = self.config.get("prediction", {}).get("pools", [])
        all_signals = {}
        
        for pool_config in pools:
            pool = pool_config.get("name")
            if not pool_config.get("enabled", True):
                continue
            try:
                signals = self.generate_signals(pool, date)
                all_signals[pool] = signals
            except Exception as e:
                logger.error(f"生成信号失败: pool={pool}, error={e}")
                all_signals[pool] = []
        
        return all_signals
