"""
Grid Search Optimizer - 网格搜索优化器

通过穷举搜索参数空间中的所有组合来寻找最优参数。
适用于参数空间较小的情况。
"""

from __future__ import annotations

import itertools
import logging
import time
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from monitor.strategy_library.optimizer.base_optimizer import (
    BaseOptimizer,
    OptimizationConfig,
    OptimizationResult,
)

if TYPE_CHECKING:
    from monitor.strategy_library.strategy_base import StrategyBase

logger = logging.getLogger(__name__)


class GridSearchOptimizer(BaseOptimizer):
    def __init__(
        self,
        config: Optional[OptimizationConfig] = None,
        backtest_func: Optional[Callable] = None,
        grid_size: int = 5,
    ):
        super().__init__(config=config, backtest_func=backtest_func)
        self.grid_size = grid_size
    
    def _run_optimization(
        self,
        strategy: "StrategyBase",
        data: Dict[str, pd.DataFrame],
        backtest_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        param_space = strategy.get_parameter_space()
        
        if not param_space:
            logger.warning("策略没有定义参数空间")
            return
        
        grid_points = self._generate_grid(param_space)
        
        total_combinations = len(grid_points)
        logger.info(f"网格搜索: 共 {total_combinations} 个参数组合")
        
        for i, params in enumerate(grid_points):
            if self._should_stop():
                break
            
            if self.config.verbose and (i + 1) % max(1, total_combinations // 10) == 0:
                logger.info(f"进度: {i + 1}/{total_combinations}")
            
            score, result = self._evaluate_params(strategy, params, data, backtest_config)
            
            if not self._check_constraints(result):
                logger.debug(f"参数组合 {params} 不满足约束条件")
                score = -np.inf
            
            self._record_trial(params, score, result)
            self._update_best(params, score, result)
    
    def _generate_grid(
        self,
        param_space: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        grid_values = {}
        
        for name, space in param_space.items():
            values = self._generate_param_values(space)
            grid_values[name] = values
        
        keys = list(grid_values.keys())
        value_lists = [grid_values[k] for k in keys]
        
        combinations = []
        for combo in itertools.product(*value_lists):
            params = dict(zip(keys, combo))
            combinations.append(params)
        
        return combinations
    
    def _generate_param_values(
        self,
        space: Any
    ) -> List[Any]:
        from monitor.strategy_library.strategy_base import ParameterSpace
        
        if isinstance(space, ParameterSpace):
            param_type = space.param_type
            min_val = space.min_val
            max_val = space.max_val
            default = space.default
            choices = space.choices
        else:
            param_type = space.get("type", "float")
            min_val = space.get("min", 0.0)
            max_val = space.get("max", 1.0)
            default = space.get("default")
            choices = space.get("choices", [])
        
        if param_type == "float":
            min_v = min_val if min_val is not None else 0.0
            max_v = max_val if max_val is not None else 1.0
            def_v = default if default is not None else (min_v + max_v) / 2
            
            values = list(np.linspace(min_v, max_v, self.grid_size))
            if def_v not in values:
                values.append(def_v)
            return values
        
        elif param_type == "int":
            min_v = int(min_val) if min_val is not None else 0
            max_v = int(max_val) if max_val is not None else 10
            def_v = int(default) if default is not None else (min_v + max_v) // 2
            
            step = max(1, (max_v - min_v) // (self.grid_size - 1))
            values = list(range(min_v, max_v + 1, step))
            if def_v not in values:
                values.append(def_v)
            return sorted(set(values))
        
        elif param_type == "choice":
            return choices if choices else []
        
        elif param_type == "bool":
            return [True, False]
        
        return [default] if default is not None else []
    
    def optimize_with_cross_validation(
        self,
        strategy: "StrategyBase",
        data: Dict[str, pd.DataFrame],
        n_splits: int = 5,
        backtest_config: Optional[Dict[str, Any]] = None,
    ) -> OptimizationResult:
        self._reset()
        
        self._start_time = time.time()
        
        param_space = strategy.get_parameter_space()
        grid_points = self._generate_grid(param_space)
        
        logger.info(f"交叉验证网格搜索: {len(grid_points)} 个参数组合, {n_splits} 折")
        
        cv_scores = {i: [] for i in range(len(grid_points))}
        
        dates = sorted(set(
            date for df in data.values() 
            for date in df["date"].dt.strftime("%Y-%m-%d").tolist()
        ))
        
        split_size = len(dates) // n_splits
        
        for fold in range(n_splits):
            test_start = fold * split_size
            test_end = (fold + 1) * split_size
            
            train_dates = dates[:test_start] + dates[test_end:]
            test_dates = dates[test_start:test_end]
            
            train_data = {
                code: df[df["date"].dt.strftime("%Y-%m-%d").isin(train_dates)]
                for code, df in data.items()
            }
            
            test_data = {
                code: df[df["date"].dt.strftime("%Y-%m-%d").isin(test_dates)]
                for code, df in data.items()
            }
            
            for i, params in enumerate(grid_points):
                if self._should_stop():
                    break
                
                score, _ = self._evaluate_params(strategy, params, test_data, backtest_config)
                cv_scores[i].append(score)
        
        best_idx = -1
        best_mean_score = -np.inf
        
        for i, params in enumerate(grid_points):
            scores = cv_scores[i]
            if not scores:
                continue
            
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            adjusted_score = mean_score - 0.1 * std_score
            
            if adjusted_score > best_mean_score:
                best_mean_score = adjusted_score
                best_idx = i
            
            self._record_trial(params, mean_score, {"std": std_score, "adjusted": adjusted_score})
        
        if best_idx >= 0:
            self._best_params = grid_points[best_idx]
            self._best_score = best_mean_score
        
        execution_time = time.time() - self._start_time
        
        return OptimizationResult(
            strategy_id=strategy.strategy_id,
            best_params=self._best_params,
            best_score=self._best_score,
            all_results=self._results,
            execution_time=execution_time,
            n_trials=self._trial_count,
            metadata={"method": "grid_search_cv", "n_splits": n_splits},
        )
