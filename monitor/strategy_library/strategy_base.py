"""
Strategy Base - 策略基类模块

定义策略的核心接口和数据结构。
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


@dataclass
class ParameterSpace:
    name: str
    param_type: str
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    choices: Optional[List[Any]] = None
    default: Any = None
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.param_type,
            "min": self.min_val,
            "max": self.max_val,
            "choices": self.choices,
            "default": self.default,
            "description": self.description,
        }
    
    def validate(self, value: Any) -> bool:
        if self.param_type == "float":
            if not isinstance(value, (int, float)):
                return False
            if self.min_val is not None and value < self.min_val:
                return False
            if self.max_val is not None and value > self.max_val:
                return False
            return True
        elif self.param_type == "int":
            if not isinstance(value, int):
                return False
            if self.min_val is not None and value < self.min_val:
                return False
            if self.max_val is not None and value > self.max_val:
                return False
            return True
        elif self.param_type == "choice":
            return value in (self.choices or [])
        elif self.param_type == "bool":
            return isinstance(value, bool)
        return True
    
    def sample(self) -> Any:
        if self.param_type == "float":
            return np.random.uniform(self.min_val or 0, self.max_val or 1)
        elif self.param_type == "int":
            return np.random.randint(self.min_val or 0, self.max_val or 10)
        elif self.param_type == "choice":
            return np.random.choice(self.choices or [])
        elif self.param_type == "bool":
            return np.random.choice([True, False])
        return self.default


@dataclass
class StrategyConfig:
    id: str
    name: str
    category: str
    version: str
    enabled: bool
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    parameter_space: Dict[str, ParameterSpace] = field(default_factory=dict)
    optimization: Dict[str, Any] = field(default_factory=dict)
    backtest: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "version": self.version,
            "enabled": self.enabled,
            "description": self.description,
            "parameters": self.parameters,
            "parameter_space": {k: v.to_dict() for k, v in self.parameter_space.items()},
            "optimization": self.optimization,
            "backtest": self.backtest,
            "constraints": self.constraints,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyConfig":
        param_space = {}
        for key, value in data.get("parameter_space", {}).items():
            if isinstance(value, dict):
                param_space[key] = ParameterSpace(
                    name=value.get("name", key),
                    param_type=value.get("type", "float"),
                    min_val=value.get("min"),
                    max_val=value.get("max"),
                    choices=value.get("choices"),
                    default=value.get("default"),
                    description=value.get("description", ""),
                )
        
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            category=data.get("category", ""),
            version=data.get("version", "1.0.0"),
            enabled=data.get("enabled", True),
            description=data.get("description", ""),
            parameters=data.get("parameters", {}),
            parameter_space=param_space,
            optimization=data.get("optimization", {}),
            backtest=data.get("backtest", {}),
            constraints=data.get("constraints", {}),
            metadata=data.get("metadata", {}),
        )
    
    @classmethod
    def from_yaml(cls, path: str) -> "StrategyConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data.get("strategy", data))


@dataclass
class Signal:
    code: str
    name: str
    action: str
    price: float
    shares: int
    confidence: float
    reason: str
    timestamp: str = ""
    strategy_id: str = ""
    strategy_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "action": self.action,
            "price": self.price,
            "shares": self.shares,
            "confidence": self.confidence,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Signal":
        return cls(
            code=data.get("code", ""),
            name=data.get("name", ""),
            action=data.get("action", ""),
            price=data.get("price", 0.0),
            shares=data.get("shares", 0),
            confidence=data.get("confidence", 0.0),
            reason=data.get("reason", ""),
            timestamp=data.get("timestamp", ""),
            strategy_id=data.get("strategy_id", ""),
            strategy_name=data.get("strategy_name", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class BacktestResult:
    strategy_id: str
    strategy_name: str
    parameters: Dict[str, Any]
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_holding_days: float
    trades: List[Dict[str, Any]] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    daily_returns: pd.Series = field(default_factory=pd.Series)
    metrics: Dict[str, float] = field(default_factory=dict)
    in_sample: bool = True
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "parameters": self.parameters,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_capital": self.initial_capital,
            "final_capital": self.final_capital,
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "avg_holding_days": self.avg_holding_days,
            "in_sample": self.in_sample,
            "execution_time": self.execution_time,
            "metrics": self.metrics,
        }
    
    def get_params_hash(self) -> str:
        params_str = json.dumps(self.parameters, sort_keys=True)
        return hashlib.md5(params_str.encode()).hexdigest()[:8]
    
    def check_constraints(self, constraints: Dict[str, Any]) -> Tuple[bool, List[str]]:
        violations = []
        
        if "max_drawdown" in constraints:
            if self.max_drawdown > constraints["max_drawdown"]:
                violations.append(f"最大回撤 {self.max_drawdown:.2%} 超过限制 {constraints['max_drawdown']:.2%}")
        
        if "min_trades" in constraints:
            if self.total_trades < constraints["min_trades"]:
                violations.append(f"交易次数 {self.total_trades} 少于最小要求 {constraints['min_trades']}")
        
        if "min_win_rate" in constraints:
            if self.win_rate < constraints["min_win_rate"]:
                violations.append(f"胜率 {self.win_rate:.2%} 低于最小要求 {constraints['min_win_rate']:.2%}")
        
        if "min_sharpe" in constraints:
            if self.sharpe_ratio < constraints["min_sharpe"]:
                violations.append(f"夏普比率 {self.sharpe_ratio:.2f} 低于最小要求 {constraints['min_sharpe']:.2f}")
        
        return len(violations) == 0, violations


class StrategyBase(ABC):
    STRATEGY_ID = "base"
    STRATEGY_NAME = "基础策略"
    STRATEGY_CATEGORY = "general"
    STRATEGY_VERSION = "1.0.0"
    
    def __init__(
        self, 
        config: Optional[StrategyConfig] = None,
        parameters: Optional[Dict[str, Any]] = None
    ):
        self._config = config or self._get_default_config()
        self._parameters = parameters or dict(self._config.parameters)
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._factor_cache: Dict[str, Dict[str, float]] = {}
        self._initialized = False
    
    @property
    def config(self) -> StrategyConfig:
        return self._config
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return self._parameters
    
    @property
    def strategy_id(self) -> str:
        return self._config.id or self.STRATEGY_ID
    
    @property
    def strategy_name(self) -> str:
        return self._config.name or self.STRATEGY_NAME
    
    @classmethod
    def _get_default_config(cls) -> StrategyConfig:
        return StrategyConfig(
            id=cls.STRATEGY_ID,
            name=cls.STRATEGY_NAME,
            category=cls.STRATEGY_CATEGORY,
            version=cls.STRATEGY_VERSION,
            enabled=True,
        )
    
    @classmethod
    def from_config(cls, config_path: str) -> "StrategyBase":
        config = StrategyConfig.from_yaml(config_path)
        return cls(config=config)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "StrategyBase":
        config = StrategyConfig.from_dict(config_dict)
        return cls(config=config)
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        for key, value in parameters.items():
            if key in self._config.parameter_space:
                space = self._config.parameter_space[key]
                if not space.validate(value):
                    raise ValueError(f"参数 {key} 的值 {value} 不在有效范围内")
        self._parameters.update(parameters)
        logger.debug(f"策略 {self.strategy_id} 参数已更新: {parameters}")
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        return self._parameters.get(key, default)
    
    def get_parameter_space(self) -> Dict[str, ParameterSpace]:
        return self._config.parameter_space
    
    def sample_parameters(self) -> Dict[str, Any]:
        params = {}
        for key, space in self._config.parameter_space.items():
            params[key] = space.sample()
        return params
    
    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []
        for key, value in params.items():
            if key in self._config.parameter_space:
                space = self._config.parameter_space[key]
                if not space.validate(value):
                    errors.append(f"参数 {key} 的值 {value} 无效")
        return len(errors) == 0, errors
    
    @abstractmethod
    def generate_signals(
        self,
        date: str,
        data: Dict[str, pd.DataFrame],
        positions: Dict[str, Any],
        **kwargs
    ) -> List[Signal]:
        pass
    
    @abstractmethod
    def calculate_score(
        self,
        code: str,
        factors: Dict[str, float],
        **kwargs
    ) -> float:
        pass
    
    def initialize(self, data: Dict[str, pd.DataFrame]) -> None:
        self._data_cache = data
        self._initialized = True
        logger.info(f"策略 {self.strategy_id} 初始化完成，加载 {len(data)} 只股票数据")
    
    def clear_cache(self) -> None:
        self._data_cache.clear()
        self._factor_cache.clear()
        self._initialized = False
    
    def get_required_data(self) -> List[Dict[str, Any]]:
        return []
    
    def get_optimization_config(self) -> Dict[str, Any]:
        return self._config.optimization
    
    def get_backtest_config(self) -> Dict[str, Any]:
        return self._config.backtest
    
    def get_constraints(self) -> Dict[str, Any]:
        return self._config.constraints
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.strategy_id} name={self.strategy_name}>"
