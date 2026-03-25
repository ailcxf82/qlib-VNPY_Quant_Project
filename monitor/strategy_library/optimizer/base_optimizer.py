"""
Base Optimizer - 优化器基类

定义优化器的核心接口和通用功能。
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from monitor.strategy_library.strategy_base import StrategyBase, BacktestResult

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    strategy_id: str
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Dict[str, Any]] = field(default_factory=list)
    optimization_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    execution_time: float = 0.0
    n_trials: int = 0
    best_backtest_result: Optional["BacktestResult"] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "best_params": self.best_params,
            "best_score": self.best_score,
            "n_trials": self.n_trials,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
        }
    
    def get_top_n(self, n: int = 10) -> List[Dict[str, Any]]:
        sorted_results = sorted(self.all_results, key=lambda x: x.get("score", 0), reverse=True)
        return sorted_results[:n]


@dataclass
class OptimizationConfig:
    objective: str = "sharpe_ratio"
    direction: str = "maximize"
    n_trials: int = 100
    timeout: Optional[int] = None
    n_jobs: int = 1
    seed: int = 42
    verbose: bool = True
    early_stopping_rounds: Optional[int] = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "objective": self.objective,
            "direction": self.direction,
            "n_trials": self.n_trials,
            "timeout": self.timeout,
            "n_jobs": self.n_jobs,
            "seed": self.seed,
            "verbose": self.verbose,
            "early_stopping_rounds": self.early_stopping_rounds,
            "constraints": self.constraints,
        }


class BaseOptimizer(ABC):
    def __init__(
        self,
        config: Optional[OptimizationConfig] = None,
        backtest_func: Optional[Callable] = None,
    ):
        self.config = config or OptimizationConfig()
        self._backtest_func = backtest_func
        self._results: List[Dict[str, Any]] = []
        self._best_score = -np.inf if self.config.direction == "maximize" else np.inf
        self._best_params: Dict[str, Any] = {}
        self._best_result: Optional["BacktestResult"] = None
        self._start_time: Optional[float] = None
        self._trial_count = 0
        self._no_improvement_count = 0
    
    def set_backtest_func(self, func: Callable) -> None:
        self._backtest_func = func
    
    def optimize(
        self,
        strategy: "StrategyBase",
        data: Dict[str, pd.DataFrame],
        backtest_config: Optional[Dict[str, Any]] = None,
    ) -> OptimizationResult:
        self._reset()
        
        self._start_time = time.time()
        
        logger.info(f"开始优化策略: {strategy.strategy_id}")
        logger.info(f"优化目标: {self.config.objective}")
        logger.info(f"优化方向: {self.config.direction}")
        logger.info(f"试验次数: {self.config.n_trials}")
        
        try:
            self._run_optimization(strategy, data, backtest_config)
        except Exception as e:
            logger.error(f"优化过程出错: {e}")
        
        execution_time = time.time() - self._start_time
        
        result = OptimizationResult(
            strategy_id=strategy.strategy_id,
            best_params=self._best_params,
            best_score=self._best_score,
            all_results=self._results,
            execution_time=execution_time,
            n_trials=self._trial_count,
            best_backtest_result=self._best_result,
            metadata={
                "objective": self.config.objective,
                "direction": self.config.direction,
                "constraints": self.config.constraints,
            }
        )
        
        logger.info(f"优化完成: 最佳得分 {self._best_score:.4f}")
        logger.info(f"最佳参数: {self._best_params}")
        logger.info(f"执行时间: {execution_time:.2f}秒")
        
        return result
    
    @abstractmethod
    def _run_optimization(
        self,
        strategy: "StrategyBase",
        data: Dict[str, pd.DataFrame],
        backtest_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        pass
    
    def _evaluate_params(
        self,
        strategy: "StrategyBase",
        params: Dict[str, Any],
        data: Dict[str, pd.DataFrame],
        backtest_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Optional["BacktestResult"]]:
        if self._backtest_func is None:
            return self._default_backtest(strategy, params, data, backtest_config)
        
        try:
            result = self._backtest_func(strategy, params, data, backtest_config)
            
            if result is None:
                return -np.inf, None
            
            score = self._get_objective_value(result)
            
            return score, result
        except Exception as e:
            logger.debug(f"回测失败: {e}")
            return -np.inf, None
    
    def _default_backtest(
        self,
        strategy: "StrategyBase",
        params: Dict[str, Any],
        data: Dict[str, pd.DataFrame],
        backtest_config: Optional[Dict[str, Any]] = None,
    ) -> Optional["BacktestResult"]:
        from monitor.integrated_backtest import IntegratedBacktestEngine, IntegratedBacktestConfig
        
        try:
            strategy.set_parameters(params)
            
            bt_config = backtest_config or {}
            
            config = IntegratedBacktestConfig(
                start_date=bt_config.get("start_date", "2021-01-01"),
                end_date=bt_config.get("end_date", "2023-12-31"),
                initial_capital=bt_config.get("initial_capital", 1000000),
                commission_rate=bt_config.get("commission_rate", 0.0003),
                stamp_duty=bt_config.get("stamp_duty", 0.001),
                slippage=bt_config.get("slippage", 0.001),
            )
            
            engine = IntegratedBacktestEngine(config=config, strategy=strategy)
            
            codes = list(data.keys())[:50]
            results = engine.run(codes)
            
            return results.get("performance", {})
        except Exception as e:
            logger.debug(f"默认回测失败: {e}")
            return None
    
    def _get_objective_value(self, result: Any) -> float:
        if isinstance(result, dict):
            return result.get(self.config.objective, -np.inf)
        elif hasattr(result, self.config.objective):
            return getattr(result, self.config.objective, -np.inf)
        elif hasattr(result, "metrics"):
            return result.metrics.get(self.config.objective, -np.inf)
        return -np.inf
    
    def _check_constraints(self, result: Any) -> bool:
        if not self.config.constraints:
            return True
        
        metrics = {}
        if isinstance(result, dict):
            metrics = result
        elif hasattr(result, "to_dict"):
            metrics = result.to_dict()
        elif hasattr(result, "metrics"):
            metrics = result.metrics
        
        for key, threshold in self.config.constraints.items():
            value = metrics.get(key, 0)
            
            if key.startswith("min_"):
                if value < threshold:
                    return False
            elif key.startswith("max_"):
                if value > threshold:
                    return False
        
        return True
    
    def _update_best(
        self,
        params: Dict[str, Any],
        score: float,
        result: Optional["BacktestResult"] = None
    ) -> bool:
        improved = False
        
        if self.config.direction == "maximize":
            if score > self._best_score:
                self._best_score = score
                self._best_params = params.copy()
                self._best_result = result
                improved = True
        else:
            if score < self._best_score:
                self._best_score = score
                self._best_params = params.copy()
                self._best_result = result
                improved = True
        
        if improved:
            self._no_improvement_count = 0
            if self.config.verbose:
                logger.info(f"新的最佳得分: {score:.4f}, 参数: {params}")
        else:
            self._no_improvement_count += 1
        
        return improved
    
    def _record_trial(
        self,
        params: Dict[str, Any],
        score: float,
        result: Optional["BacktestResult"] = None
    ) -> None:
        self._trial_count += 1
        
        trial_record = {
            "trial": self._trial_count,
            "params": params.copy(),
            "score": score,
            "timestamp": datetime.now().isoformat(),
        }
        
        if result is not None:
            if isinstance(result, dict):
                trial_record["metrics"] = result
            elif hasattr(result, "to_dict"):
                trial_record["metrics"] = result.to_dict()
        
        self._results.append(trial_record)
    
    def _should_stop(self) -> bool:
        if self.config.timeout:
            elapsed = time.time() - self._start_time
            if elapsed >= self.config.timeout:
                logger.info(f"达到超时限制: {self.config.timeout}秒")
                return True
        
        if self.config.early_stopping_rounds:
            if self._no_improvement_count >= self.config.early_stopping_rounds:
                logger.info(f"早停: {self._no_improvement_count}次无改进")
                return True
        
        return False
    
    def _reset(self) -> None:
        self._results = []
        self._best_score = -np.inf if self.config.direction == "maximize" else np.inf
        self._best_params = {}
        self._best_result = None
        self._start_time = None
        self._trial_count = 0
        self._no_improvement_count = 0
    
    def get_optimization_history(self) -> pd.DataFrame:
        if not self._results:
            return pd.DataFrame()
        
        df = pd.DataFrame(self._results)
        return df
