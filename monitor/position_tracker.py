from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from monitor.ts_client import get_ts_client

logger = logging.getLogger(__name__)


@dataclass
class Position:
    code: str
    name: str
    shares: int
    cost_price: float
    current_price: float
    buy_date: str
    strategy: str
    buy_reason: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def market_value(self) -> float:
        return self.shares * self.current_price
    
    @property
    def cost_value(self) -> float:
        return self.shares * self.cost_price
    
    @property
    def profit(self) -> float:
        return self.market_value - self.cost_value
    
    @property
    def profit_pct(self) -> float:
        if self.cost_value == 0:
            return 0.0
        return (self.current_price - self.cost_price) / self.cost_price
    
    @property
    def holding_days(self) -> int:
        try:
            buy_dt = datetime.strptime(self.buy_date, "%Y-%m-%d")
            today = datetime.now()
            return (today - buy_dt).days
        except:
            return 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "shares": self.shares,
            "cost_price": self.cost_price,
            "current_price": self.current_price,
            "buy_date": self.buy_date,
            "strategy": self.strategy,
            "buy_reason": self.buy_reason,
            "market_value": self.market_value,
            "profit": self.profit,
            "profit_pct": self.profit_pct,
            "holding_days": self.holding_days,
            "extra": self.extra,
        }


@dataclass
class Trade:
    code: str
    name: str
    trade_type: str
    shares: int
    price: float
    amount: float
    commission: float
    stamp_duty: float
    date: str
    strategy: str
    reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PositionTracker:
    def __init__(self, config_path: str = "config/monitor.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.position_config = self.config.get("position", {})
        
        self.initial_capital = self.position_config.get("initial_capital", 1000000)
        self.commission_rate = self.position_config.get("commission_rate", 0.0003)
        self.stamp_duty = self.position_config.get("stamp_duty", 0.001)
        self.slippage = self.position_config.get("slippage", 0.001)
        self.min_trade_amount = self.position_config.get("min_trade_amount", 1000)
        
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.cash = self.initial_capital
        
        self._position_file = Path(self.config.get("paths", {}).get("position_file", "data/monitor/position.json"))
        self._trade_log_file = Path(self.config.get("paths", {}).get("trade_log_file", "data/monitor/trades.csv"))
        
        self._lock = threading.Lock()
        self._load_state()
    
    def _load_config(self) -> Dict[str, Any]:
        p = Path(self.config_path)
        if not p.exists():
            return {}
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    
    def _load_state(self):
        with self._lock:
            if self._position_file.exists():
                try:
                    with open(self._position_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self.cash = data.get("cash", self.initial_capital)
                    positions_data = data.get("positions", {})
                    self.positions = {}
                    for code, pos_data in positions_data.items():
                        self.positions[code] = Position(**pos_data)
                    logger.info(f"加载持仓状态: {len(self.positions)} 只股票, 现金 {self.cash:.2f}")
                except Exception as e:
                    logger.warning(f"加载持仓状态失败: {e}")
            
            if self._trade_log_file.exists():
                try:
                    trade_df = pd.read_csv(self._trade_log_file)
                    self.trades = []
                    for _, row in trade_df.iterrows():
                        trade = Trade(
                            code=row["code"],
                            name=row["name"],
                            trade_type=row["trade_type"],
                            shares=int(row["shares"]),
                            price=float(row["price"]),
                            amount=float(row["amount"]),
                            commission=float(row["commission"]),
                            stamp_duty=float(row["stamp_duty"]),
                            date=row["date"],
                            strategy=row["strategy"],
                            reason=row.get("reason", ""),
                        )
                        self.trades.append(trade)
                except Exception as e:
                    logger.warning(f"加载交易记录失败: {e}")
    
    def _save_state(self):
        with self._lock:
            self._position_file.parent.mkdir(parents=True, exist_ok=True)
            
            positions_data = {}
            for code, pos in self.positions.items():
                positions_data[code] = {
                    "code": pos.code,
                    "name": pos.name,
                    "shares": pos.shares,
                    "cost_price": pos.cost_price,
                    "current_price": pos.current_price,
                    "buy_date": pos.buy_date,
                    "strategy": pos.strategy,
                    "buy_reason": pos.buy_reason,
                    "extra": pos.extra,
                }
            
            data = {
                "cash": self.cash,
                "positions": positions_data,
                "updated_at": datetime.now().isoformat(),
            }
            
            with open(self._position_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            if self.trades:
                trade_df = pd.DataFrame([t.to_dict() for t in self.trades])
                trade_df.to_csv(self._trade_log_file, index=False, encoding="utf-8")
            
            logger.info(f"保存持仓状态: {len(self.positions)} 只股票, 现金 {self.cash:.2f}")
    
    def get_current_price(self, code: str) -> float:
        ts = get_ts_client()
        if ts is None:
            logger.warning("无法获取当前价格: TushareClient 未初始化")
            with self._lock:
                if code in self.positions:
                    return self.positions[code].current_price
            return 0.0
        
        try:
            code_clean = str(code).replace(".XSHE", "").replace(".XSHG", "").zfill(6)
            suffix = ".SZ" if code_clean.startswith(("0", "3")) else ".SH"
            ts_code = code_clean + suffix
            today = datetime.now().strftime("%Y%m%d")
            df = ts._pro.daily(ts_code=ts_code, trade_date=today)
            if not df.empty:
                return float(df.iloc[0]["close"])
            
            df = ts._pro.daily(ts_code=ts_code, end_date=today, limit=1)
            if not df.empty:
                return float(df.iloc[0]["close"])
        except Exception as e:
            logger.warning(f"获取价格失败: code={code}, error={e}")
        
        with self._lock:
            if code in self.positions:
                return self.positions[code].current_price
        return 0.0
    
    def update_prices(self):
        with self._lock:
            for code, pos in list(self.positions.items()):
                price = self.get_current_price(code)
                if price > 0:
                    pos.current_price = price
        self._save_state()
    
    def buy(self, code: str, name: str, price: float, strategy: str, reason: str = "", 
            position_size: Optional[float] = None) -> Optional[Trade]:
        with self._lock:
            if code in self.positions:
                logger.info(f"已持有 {code}，跳过买入")
                return None
            
            if position_size is None:
                strategy_config = self.config.get("strategies", {}).get(strategy, {})
                position_size = strategy_config.get("position_size", 0.1)
            
            target_amount = self.initial_capital * position_size
            actual_price = price * (1 + self.slippage)
            shares = int(target_amount / actual_price / 100) * 100
            
            if shares * actual_price < self.min_trade_amount:
                logger.warning(f"买入金额不足最小交易额: {code}")
                return None
            
            if shares * actual_price > self.cash:
                shares = int(self.cash / actual_price / 100) * 100
                if shares < 100:
                    logger.warning(f"现金不足，无法买入: {code}")
                    return None
            
            amount = shares * actual_price
            commission = max(amount * self.commission_rate, 5)
            total_cost = amount + commission
            
            self.cash -= total_cost
            
            position = Position(
                code=code,
                name=name,
                shares=shares,
                cost_price=actual_price,
                current_price=price,
                buy_date=datetime.now().strftime("%Y-%m-%d"),
                strategy=strategy,
                buy_reason=reason,
            )
            self.positions[code] = position
            
            trade = Trade(
                code=code,
                name=name,
                trade_type="buy",
                shares=shares,
                price=actual_price,
                amount=amount,
                commission=commission,
                stamp_duty=0,
                date=datetime.now().strftime("%Y-%m-%d"),
                strategy=strategy,
                reason=reason,
            )
            self.trades.append(trade)
            
            logger.info(f"买入: {code} {name} {shares}股 @ {actual_price:.2f}")
        
        self._save_state()
        return trade
    
    def sell(self, code: str, reason: str = "") -> Optional[Trade]:
        with self._lock:
            if code not in self.positions:
                logger.warning(f"未持有 {code}，无法卖出")
                return None
            
            pos = self.positions[code]
            price = self.get_current_price(code)
            if price <= 0:
                price = pos.current_price
            
            actual_price = price * (1 - self.slippage)
            amount = pos.shares * actual_price
            commission = max(amount * self.commission_rate, 5)
            stamp = amount * self.stamp_duty
            total_receive = amount - commission - stamp
            
            self.cash += total_receive
            
            trade = Trade(
                code=code,
                name=pos.name,
                trade_type="sell",
                shares=pos.shares,
                price=actual_price,
                amount=amount,
                commission=commission,
                stamp_duty=stamp,
                date=datetime.now().strftime("%Y-%m-%d"),
                strategy=pos.strategy,
                reason=reason,
            )
            self.trades.append(trade)
            
            del self.positions[code]
            
            logger.info(f"卖出: {code} {pos.name} {pos.shares}股 @ {actual_price:.2f}, 原因: {reason}")
        
        self._save_state()
        return trade
    
    def check_sell_signals(self) -> List[Dict[str, Any]]:
        sell_signals = []
        
        with self._lock:
            for code, pos in list(self.positions.items()):
                strategy_config = self.config.get("strategies", {}).get(pos.strategy, {})
                sell_rules = strategy_config.get("rules", {}).get("sell", [])
                
                for rule in sell_rules:
                    rule_type = rule.get("type")
                    
                    if rule_type == "holding_days":
                        max_days = rule.get("max_days", 5)
                        if pos.holding_days >= max_days:
                            sell_signals.append({
                                "code": code,
                                "name": pos.name,
                                "reason": f"持仓超过 {max_days} 天",
                                "profit_pct": pos.profit_pct,
                            })
                            break
                    
                    elif rule_type == "stop_loss":
                        threshold = rule.get("threshold", -0.08)
                        if pos.profit_pct <= threshold:
                            sell_signals.append({
                                "code": code,
                                "name": pos.name,
                                "reason": f"止损: 亏损 {pos.profit_pct:.2%}",
                                "profit_pct": pos.profit_pct,
                            })
                            break
                    
                    elif rule_type == "take_profit":
                        threshold = rule.get("threshold", 0.15)
                        if pos.profit_pct >= threshold:
                            sell_signals.append({
                                "code": code,
                                "name": pos.name,
                                "reason": f"止盈: 盈利 {pos.profit_pct:.2%}",
                                "profit_pct": pos.profit_pct,
                            })
                            break
        
        return sell_signals
    
    def execute_sell_signals(self, sell_signals: List[Dict[str, Any]]) -> List[Trade]:
        trades = []
        for sig in sell_signals:
            trade = self.sell(sig["code"], sig["reason"])
            if trade:
                trades.append(trade)
        return trades
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        self.update_prices()
        
        with self._lock:
            total_market_value = sum(pos.market_value for pos in self.positions.values())
            total_cost = sum(pos.cost_value for pos in self.positions.values())
            total_profit = total_market_value - total_cost
            total_profit_pct = total_profit / total_cost if total_cost > 0 else 0
            
            total_assets = self.cash + total_market_value
            total_return = (total_assets - self.initial_capital) / self.initial_capital
            
            return {
                "cash": self.cash,
                "market_value": total_market_value,
                "total_assets": total_assets,
                "total_profit": total_profit,
                "total_profit_pct": total_profit_pct,
                "total_return": total_return,
                "position_count": len(self.positions),
                "positions": [pos.to_dict() for pos in self.positions.values()],
            }
    
    def reset(self):
        with self._lock:
            self.positions = {}
            self.trades = []
            self.cash = self.initial_capital
        self._save_state()
        logger.info("持仓已重置")
