"""
Overfitting Protection - 过拟合防护模块

提供多种防止过拟合的方法：
1. 样本外测试 (Out-of-Sample Testing)
2. 交叉验证 (Cross-Validation)
3. 正则化约束 (Regularization Constraints)
4. 样本权重衰减 (Sample Weight Decay)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from monitor.strategy_library.strategy_base import StrategyBase, BacktestResult

logger = logging.getLogger(__name__)


@dataclass
class OverfittingCheckResult:
    is_overfitted: bool
    in_sample_score: float
    out_of_sample_score: float
    overfitting_ratio: float
    degradation: float
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_overfitted": self.is_overfitted,
            "in_sample_score": self.in_sample_score,
            "out_of_sample_score": self.out_of_sample_score,
            "overfitting_ratio": self.overfitting_ratio,
            "degradation": self.degradation,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
        }


@dataclass
class CrossValidationResult:
    mean_score: float
    std_score: float
    fold_scores: List[float]
    confidence_interval: Tuple[float, float]
    is_stable: bool
    stability_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_score": self.mean_score,
            "std_score": self.std_score,
            "fold_scores": self.fold_scores,
            "confidence_interval": self.confidence_interval,
            "is_stable": self.is_stable,
            "stability_score": self.stability_score,
        }


class OverfittingProtector:
    DEFAULT_OVERFITTING_THRESHOLD = 0.20
    DEFAULT_DEGRADATION_THRESHOLD = 0.30
    DEFAULT_STABILITY_THRESHOLD = 0.25
    
    def __init__(
        self,
        overfitting_threshold: float = DEFAULT_OVERFITTING_THRESHOLD,
        degradation_threshold: float = DEFAULT_DEGRADATION_THRESHOLD,
        stability_threshold: float = DEFAULT_STABILITY_THRESHOLD,
        n_splits: int = 5,
        verbose: bool = True,
    ):
        self.overfitting_threshold = overfitting_threshold
        self.degradation_threshold = degradation_threshold
        self.stability_threshold = stability_threshold
        self.n_splits = n_splits
        self.verbose = verbose
    
    def check_overfitting(
        self,
        in_sample_result: "BacktestResult",
        out_of_sample_result: "BacktestResult",
        metric: str = "sharpe_ratio",
    ) -> OverfittingCheckResult:
        in_sample_score = self._extract_metric(in_sample_result, metric)
        out_of_sample_score = self._extract_metric(out_of_sample_result, metric)
        
        if in_sample_score == 0:
            overfitting_ratio = 0.0
            degradation = 1.0
        else:
            overfitting_ratio = abs(in_sample_score - out_of_sample_score) / abs(in_sample_score)
            degradation = (in_sample_score - out_of_sample_score) / abs(in_sample_score) if in_sample_score > 0 else 0
        
        warnings = []
        recommendations = []
        
        is_overfitted = False
        
        if overfitting_ratio > self.overfitting_threshold:
            is_overfitted = True
            warnings.append(
                f"过拟合警告: 样本内外差异比例 {overfitting_ratio:.1%} 超过阈值 {self.overfitting_threshold:.1%}"
            )
            recommendations.append("考虑减少参数数量或增加正则化约束")
        
        if degradation > self.degradation_threshold:
            is_overfitted = True
            warnings.append(
                f"性能衰减警告: 样本外性能下降 {degradation:.1%} 超过阈值 {self.degradation_threshold:.1%}"
            )
            recommendations.append("考虑使用更保守的参数或增加训练数据")
        
        if in_sample_score > 2.0 and out_of_sample_score < 1.0:
            is_overfitted = True
            warnings.append("样本内夏普比率异常高，可能存在数据泄露或过拟合")
            recommendations.append("检查数据泄露问题，确保特征工程正确")
        
        if not warnings:
            recommendations.append("策略表现稳定，可以考虑实盘部署")
        
        result = OverfittingCheckResult(
            is_overfitted=is_overfitted,
            in_sample_score=in_sample_score,
            out_of_sample_score=out_of_sample_score,
            overfitting_ratio=overfitting_ratio,
            degradation=degradation,
            warnings=warnings,
            recommendations=recommendations,
        )
        
        if self.verbose:
            self._log_result(result)
        
        return result
    
    def cross_validate(
        self,
        strategy: "StrategyBase",
        data: Dict[str, pd.DataFrame],
        backtest_func: Callable,
        n_splits: Optional[int] = None,
        metric: str = "sharpe_ratio",
    ) -> CrossValidationResult:
        n_splits = n_splits or self.n_splits
        
        dates = self._extract_dates(data)
        
        if len(dates) < n_splits * 20:
            logger.warning("数据量不足以进行交叉验证")
            return CrossValidationResult(
                mean_score=0.0,
                std_score=1.0,
                fold_scores=[],
                confidence_interval=(0.0, 0.0),
                is_stable=False,
                stability_score=0.0,
            )
        
        fold_scores = []
        split_size = len(dates) // n_splits
        
        for fold in range(n_splits):
            test_start_idx = fold * split_size
            test_end_idx = (fold + 1) * split_size
            
            test_dates = dates[test_start_idx:test_end_idx]
            train_dates = dates[:test_start_idx] + dates[test_end_idx:]
            
            test_data = self._filter_data_by_dates(data, test_dates)
            train_data = self._filter_data_by_dates(data, train_dates)
            
            try:
                result = backtest_func(strategy, train_data, test_data)
                score = self._extract_metric(result, metric)
                fold_scores.append(score)
            except Exception as e:
                logger.warning(f"Fold {fold + 1} 回测失败: {e}")
                fold_scores.append(0.0)
        
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        confidence_interval = (
            mean_score - 1.96 * std_score / np.sqrt(n_splits),
            mean_score + 1.96 * std_score / np.sqrt(n_splits),
        )
        
        cv = std_score / mean_score if mean_score != 0 else 1.0
        is_stable = cv < self.stability_threshold
        stability_score = 1.0 - min(cv, 1.0)
        
        result = CrossValidationResult(
            mean_score=mean_score,
            std_score=std_score,
            fold_scores=fold_scores,
            confidence_interval=confidence_interval,
            is_stable=is_stable,
            stability_score=stability_score,
        )
        
        if self.verbose:
            self._log_cv_result(result)
        
        return result
    
    def walk_forward_validation(
        self,
        strategy: "StrategyBase",
        data: Dict[str, pd.DataFrame],
        backtest_func: Callable,
        train_window: int = 252,
        test_window: int = 63,
        step: int = 21,
        metric: str = "sharpe_ratio",
    ) -> Dict[str, Any]:
        dates = self._extract_dates(data)
        
        if len(dates) < train_window + test_window:
            logger.warning("数据量不足以进行滚动验证")
            return {"error": "insufficient_data"}
        
        results = []
        test_scores = []
        
        start_idx = 0
        
        while start_idx + train_window + test_window <= len(dates):
            train_end_idx = start_idx + train_window
            test_end_idx = train_end_idx + test_window
            
            train_dates = dates[start_idx:train_end_idx]
            test_dates = dates[train_end_idx:test_end_idx]
            
            train_data = self._filter_data_by_dates(data, train_dates)
            test_data = self._filter_data_by_dates(data, test_dates)
            
            try:
                result = backtest_func(strategy, train_data, test_data)
                score = self._extract_metric(result, metric)
                
                results.append({
                    "train_period": (train_dates[0], train_dates[-1]),
                    "test_period": (test_dates[0], test_dates[-1]),
                    "score": score,
                })
                test_scores.append(score)
            except Exception as e:
                logger.warning(f"滚动验证失败: {e}")
            
            start_idx += step
        
        if not test_scores:
            return {"error": "no_valid_results"}
        
        return {
            "mean_score": np.mean(test_scores),
            "std_score": np.std(test_scores),
            "min_score": np.min(test_scores),
            "max_score": np.max(test_scores),
            "n_periods": len(test_scores),
            "details": results,
        }
    
    def apply_regularization(
        self,
        params: Dict[str, Any],
        param_space: Dict[str, Any],
        regularization_strength: float = 0.1,
    ) -> Dict[str, Any]:
        from monitor.strategy_library.strategy_base import ParameterSpace
        
        regularized_params = {}
        
        for name, value in params.items():
            if name not in param_space:
                regularized_params[name] = value
                continue
            
            space = param_space[name]
            
            if isinstance(space, ParameterSpace):
                param_type = space.param_type
                min_val = space.min_val
                max_val = space.max_val
                default = space.default
            else:
                param_type = space.get("type", "float")
                min_val = space.get("min", 0)
                max_val = space.get("max", 1)
                default = space.get("default")
            
            if param_type in ["float", "int"]:
                min_v = min_val if min_val is not None else 0
                max_v = max_val if max_val is not None else 1
                def_v = default if default is not None else (min_v + max_v) / 2
                
                shrinkage = regularization_strength * (value - def_v)
                new_value = value - shrinkage
                
                if param_type == "int":
                    new_value = int(round(new_value))
                
                regularized_params[name] = new_value
            else:
                regularized_params[name] = value
        
        return regularized_params
    
    def calculate_complexity_penalty(
        self,
        params: Dict[str, Any],
        n_params: int,
        penalty_factor: float = 0.01,
    ) -> float:
        active_params = sum(1 for v in params.values() if v is not None and v != 0)
        
        complexity = active_params / max(n_params, 1)
        
        penalty = penalty_factor * complexity
        
        return penalty
    
    def _extract_metric(self, result: Any, metric: str) -> float:
        if result is None:
            return 0.0
        
        if isinstance(result, dict):
            return result.get(metric, 0.0)
        elif hasattr(result, metric):
            return getattr(result, metric, 0.0)
        elif hasattr(result, "metrics"):
            return result.metrics.get(metric, 0.0)
        
        return 0.0
    
    def _extract_dates(self, data: Dict[str, pd.DataFrame]) -> List[str]:
        all_dates = set()
        
        for df in data.values():
            if "date" in df.columns:
                dates = df["date"].dt.strftime("%Y-%m-%d").tolist()
                all_dates.update(dates)
        
        return sorted(list(all_dates))
    
    def _filter_data_by_dates(
        self,
        data: Dict[str, pd.DataFrame],
        dates: List[str],
    ) -> Dict[str, pd.DataFrame]:
        filtered = {}
        date_set = set(dates)
        
        for code, df in data.items():
            if "date" in df.columns:
                mask = df["date"].dt.strftime("%Y-%m-%d").isin(date_set)
                filtered[code] = df[mask].copy()
            else:
                filtered[code] = df.copy()
        
        return filtered
    
    def _log_result(self, result: OverfittingCheckResult) -> None:
        logger.info("=" * 50)
        logger.info("过拟合检查结果")
        logger.info("=" * 50)
        logger.info(f"样本内得分: {result.in_sample_score:.4f}")
        logger.info(f"样本外得分: {result.out_of_sample_score:.4f}")
        logger.info(f"过拟合比例: {result.overfitting_ratio:.1%}")
        logger.info(f"性能衰减: {result.degradation:.1%}")
        
        if result.warnings:
            logger.warning("警告:")
            for w in result.warnings:
                logger.warning(f"  - {w}")
        
        if result.recommendations:
            logger.info("建议:")
            for r in result.recommendations:
                logger.info(f"  - {r}")
        
        logger.info("=" * 50)
    
    def _log_cv_result(self, result: CrossValidationResult) -> None:
        logger.info("=" * 50)
        logger.info("交叉验证结果")
        logger.info("=" * 50)
        logger.info(f"平均得分: {result.mean_score:.4f}")
        logger.info(f"标准差: {result.std_score:.4f}")
        logger.info(f"置信区间: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")
        logger.info(f"稳定性得分: {result.stability_score:.2f}")
        logger.info(f"是否稳定: {'是' if result.is_stable else '否'}")
        logger.info("=" * 50)
