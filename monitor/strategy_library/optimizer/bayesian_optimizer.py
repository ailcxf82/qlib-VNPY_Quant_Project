"""
Bayesian Optimizer - 贝叶斯优化器

使用贝叶斯优化方法高效搜索参数空间。
适用于参数空间较大、评估成本较高的情况。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

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


@dataclass
class BayesianOptimizerState:
    params_history: List[Dict[str, Any]]
    scores_history: List[float]
    surrogate_model: Any = None
    acquisition_func: str = "ei"
    exploration_weight: float = 0.1


class BayesianOptimizer(BaseOptimizer):
    def __init__(
        self,
        config: Optional[OptimizationConfig] = None,
        backtest_func: Optional[Callable] = None,
        acquisition_func: str = "ei",
        exploration_weight: float = 0.1,
        n_initial_points: int = 5,
    ):
        super().__init__(config=config, backtest_func=backtest_func)
        
        self.acquisition_func = acquisition_func
        self.exploration_weight = exploration_weight
        self.n_initial_points = n_initial_points
        
        self._state = BayesianOptimizerState(
            params_history=[],
            scores_history=[],
            acquisition_func=acquisition_func,
            exploration_weight=exploration_weight,
        )
        
        self._param_bounds: Dict[str, Tuple[float, float]] = {}
        self._param_types: Dict[str, str] = {}
        self._param_choices: Dict[str, List[Any]] = {}
    
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
        
        self._setup_param_space(param_space)
        
        n_initial = min(self.n_initial_points, self.config.n_trials // 4)
        
        logger.info(f"贝叶斯优化: {n_initial} 个初始点, {self.config.n_trials - n_initial} 个贝叶斯迭代")
        
        for i in range(n_initial):
            if self._should_stop():
                break
            
            params = self._sample_random_params()
            score, result = self._evaluate_params(strategy, params, data, backtest_config)
            
            if self._check_constraints(result):
                self._record_trial(params, score, result)
                self._update_best(params, score, result)
                self._update_state(params, score)
            else:
                self._record_trial(params, -np.inf, result)
        
        for i in range(self.config.n_trials - n_initial):
            if self._should_stop():
                break
            
            if self.config.verbose and (i + 1) % 10 == 0:
                logger.info(f"贝叶斯迭代: {i + 1}/{self.config.n_trials - n_initial}")
            
            params = self._suggest_next_params()
            score, result = self._evaluate_params(strategy, params, data, backtest_config)
            
            if self._check_constraints(result):
                self._record_trial(params, score, result)
                self._update_best(params, score, result)
                self._update_state(params, score)
            else:
                self._record_trial(params, -np.inf, result)
    
    def _setup_param_space(self, param_space: Dict[str, Any]) -> None:
        self._param_bounds = {}
        self._param_types = {}
        self._param_choices = {}
        
        for name, space in param_space.items():
            param_type = space.get("type", "float")
            self._param_types[name] = param_type
            
            if param_type in ["float", "int"]:
                min_val = space.get("min", 0.0 if param_type == "float" else 0)
                max_val = space.get("max", 1.0 if param_type == "float" else 10)
                self._param_bounds[name] = (min_val, max_val)
            elif param_type == "choice":
                self._param_choices[name] = space.get("choices", [])
    
    def _sample_random_params(self) -> Dict[str, Any]:
        params = {}
        
        for name, param_type in self._param_types.items():
            if param_type == "float":
                min_val, max_val = self._param_bounds[name]
                params[name] = np.random.uniform(min_val, max_val)
            elif param_type == "int":
                min_val, max_val = self._param_bounds[name]
                params[name] = np.random.randint(min_val, max_val + 1)
            elif param_type == "choice":
                params[name] = np.random.choice(self._param_choices[name])
            elif param_type == "bool":
                params[name] = np.random.choice([True, False])
        
        return params
    
    def _suggest_next_params(self) -> Dict[str, Any]:
        if len(self._state.params_history) < 5:
            return self._sample_random_params()
        
        best_params = self._get_best_params_from_history()
        
        params = {}
        for name, param_type in self._param_types.items():
            if param_type in ["float", "int"]:
                min_val, max_val = self._param_bounds[name]
                current_best = best_params.get(name, (min_val + max_val) / 2)
                
                range_val = max_val - min_val
                exploration_range = range_val * self.exploration_weight
                
                new_val = current_best + np.random.uniform(-exploration_range, exploration_range)
                new_val = max(min_val, min(max_val, new_val))
                
                if param_type == "int":
                    new_val = int(round(new_val))
                
                params[name] = new_val
            
            elif param_type == "choice":
                choices = self._param_choices[name]
                
                if np.random.random() < 0.7:
                    best_choice = best_params.get(name)
                    if best_choice in choices:
                        params[name] = best_choice
                    else:
                        params[name] = np.random.choice(choices)
                else:
                    params[name] = np.random.choice(choices)
            
            elif param_type == "bool":
                params[name] = np.random.choice([True, False])
        
        return params
    
    def _get_best_params_from_history(self) -> Dict[str, float]:
        if not self._state.scores_history:
            return {}
        
        best_idx = np.argmax(self._state.scores_history)
        return self._state.params_history[best_idx]
    
    def _update_state(self, params: Dict[str, Any], score: float) -> None:
        self._state.params_history.append(params.copy())
        self._state.scores_history.append(score)
    
    def _calculate_acquisition(
        self,
        params: Dict[str, Any],
        best_score: float
    ) -> float:
        if self.acquisition_func == "ei":
            return self._expected_improvement(params, best_score)
        elif self.acquisition_func == "ucb":
            return self._upper_confidence_bound(params)
        else:
            return self._expected_improvement(params, best_score)
    
    def _expected_improvement(
        self,
        params: Dict[str, Any],
        best_score: float
    ) -> float:
        if len(self._state.scores_history) < 2:
            return 1.0
        
        mean = np.mean(self._state.scores_history)
        std = np.std(self._state.scores_history)
        
        if std == 0:
            return 0.0
        
        z = (mean - best_score) / std
        
        ei = (mean - best_score) * self._normal_cdf(z) + std * self._normal_pdf(z)
        
        return ei
    
    def _upper_confidence_bound(self, params: Dict[str, Any]) -> float:
        if len(self._state.scores_history) < 2:
            return 0.0
        
        mean = np.mean(self._state.scores_history)
        std = np.std(self._state.scores_history)
        
        kappa = 2.0
        
        return mean + kappa * std
    
    @staticmethod
    def _normal_pdf(x: float) -> float:
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    
    @staticmethod
    def _normal_cdf(x: float) -> float:
        return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    def optimize_with_pruning(
        self,
        strategy: "StrategyBase",
        data: Dict[str, pd.DataFrame],
        pruning_threshold: float = 0.1,
        backtest_config: Optional[Dict[str, Any]] = None,
    ) -> OptimizationResult:
        self._reset()
        
        self._start_time = time.time()
        
        param_space = strategy.get_parameter_space()
        self._setup_param_space(param_space)
        
        n_initial = min(self.n_initial_points, self.config.n_trials // 4)
        
        logger.info(f"带剪枝的贝叶斯优化: 剪枝阈值={pruning_threshold}")
        
        for i in range(n_initial):
            if self._should_stop():
                break
            
            params = self._sample_random_params()
            
            partial_score = self._quick_evaluate(strategy, params, data)
            
            if partial_score < pruning_threshold:
                logger.debug(f"剪枝: 参数 {params} 快速评估得分 {partial_score:.3f}")
                continue
            
            score, result = self._evaluate_params(strategy, params, data, backtest_config)
            
            if self._check_constraints(result):
                self._record_trial(params, score, result)
                self._update_best(params, score, result)
                self._update_state(params, score)
        
        for i in range(self.config.n_trials - n_initial):
            if self._should_stop():
                break
            
            params = self._suggest_next_params()
            
            partial_score = self._quick_evaluate(strategy, params, data)
            
            if partial_score < pruning_threshold:
                continue
            
            score, result = self._evaluate_params(strategy, params, data, backtest_config)
            
            if self._check_constraints(result):
                self._record_trial(params, score, result)
                self._update_best(params, score, result)
                self._update_state(params, score)
        
        execution_time = time.time() - self._start_time
        
        return OptimizationResult(
            strategy_id=strategy.strategy_id,
            best_params=self._best_params,
            best_score=self._best_score,
            all_results=self._results,
            execution_time=execution_time,
            n_trials=self._trial_count,
            metadata={
                "method": "bayesian_with_pruning",
                "pruning_threshold": pruning_threshold,
            },
        )
    
    def _quick_evaluate(
        self,
        strategy: "StrategyBase",
        params: Dict[str, Any],
        data: Dict[str, pd.DataFrame]
    ) -> float:
        if len(self._state.scores_history) < 3:
            return 1.0
        
        recent_scores = self._state.scores_history[-10:]
        mean_score = np.mean(recent_scores)
        std_score = np.std(recent_scores)
        
        if std_score == 0:
            return 0.5
        
        normalized = (mean_score - min(recent_scores)) / (max(recent_scores) - min(recent_scores) + 1e-6)
        
        return normalized
