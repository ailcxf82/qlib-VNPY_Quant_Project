"""
Config Loader - 策略配置加载器

支持从 YAML 文件加载策略配置，支持参数验证和默认值填充。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from monitor.strategy_library.strategy_base import (
    StrategyConfig,
    ParameterSpace,
)

logger = logging.getLogger(__name__)


@dataclass
class ConfigValidationResult:
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class ConfigLoader:
    DEFAULT_CONFIG_DIR = Path("config/strategies")
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        self.config_dir = Path(config_dir) if config_dir else self.DEFAULT_CONFIG_DIR
        self._cache: Dict[str, StrategyConfig] = {}
    
    def load(self, strategy_id: str) -> Optional[StrategyConfig]:
        if strategy_id in self._cache:
            return self._cache[strategy_id]
        
        config_path = self.config_dir / f"{strategy_id}.yaml"
        
        if not config_path.exists():
            logger.warning(f"策略配置文件不存在: {config_path}")
            return None
        
        try:
            config = self.load_from_file(str(config_path))
            self._cache[strategy_id] = config
            return config
        except Exception as e:
            logger.error(f"加载策略配置失败 {strategy_id}: {e}")
            return None
    
    def load_from_file(self, path: str) -> StrategyConfig:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        if "strategy" in data:
            data = data["strategy"]
        
        config = self._parse_config(data)
        
        validation = self.validate_config(config)
        if not validation.valid:
            raise ValueError(f"配置验证失败: {validation.errors}")
        
        if validation.warnings:
            for warning in validation.warnings:
                logger.warning(f"配置警告: {warning}")
        
        logger.info(f"成功加载策略配置: {config.id} ({config.name})")
        return config
    
    def _parse_config(self, data: Dict[str, Any]) -> StrategyConfig:
        param_space = {}
        
        parameters_data = data.get("parameters", {})
        for key, value in parameters_data.items():
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
        
        defaults = {}
        for key, space in param_space.items():
            if space.default is not None:
                defaults[key] = space.default
        
        parameters = data.get("parameters", {})
        for key, value in parameters_data.items():
            if not isinstance(value, dict) and key not in defaults:
                defaults[key] = value
        
        return StrategyConfig(
            id=data.get("id", ""),
            name=data.get("name", ""),
            category=data.get("category", "general"),
            version=data.get("version", "1.0.0"),
            enabled=data.get("enabled", True),
            description=data.get("description", ""),
            parameters=defaults,
            parameter_space=param_space,
            optimization=data.get("optimization", {}),
            backtest=data.get("backtest", {}),
            constraints=data.get("constraints", {}),
            metadata=data.get("metadata", {}),
        )
    
    def validate_config(self, config: StrategyConfig) -> ConfigValidationResult:
        errors = []
        warnings = []
        
        if not config.id:
            errors.append("策略ID不能为空")
        
        if not config.name:
            warnings.append("策略名称为空，将使用ID作为名称")
        
        if config.optimization:
            opt = config.optimization
            if "objective" not in opt:
                warnings.append("未指定优化目标，将使用默认的夏普比率")
            if "method" not in opt:
                warnings.append("未指定优化方法，将使用网格搜索")
        
        if config.backtest:
            bt = config.backtest
            if "initial_capital" in bt and bt["initial_capital"] <= 0:
                errors.append("初始资金必须大于0")
        
        for key, space in config.parameter_space.items():
            if space.param_type == "float":
                if space.min_val is not None and space.max_val is not None:
                    if space.min_val >= space.max_val:
                        errors.append(f"参数 {key} 的最小值必须小于最大值")
            elif space.param_type == "int":
                if space.min_val is not None and space.max_val is not None:
                    if space.min_val >= space.max_val:
                        errors.append(f"参数 {key} 的最小值必须小于最大值")
            elif space.param_type == "choice":
                if not space.choices:
                    errors.append(f"参数 {key} 的选择类型必须指定 choices")
        
        return ConfigValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )
    
    def load_all(self) -> Dict[str, StrategyConfig]:
        configs = {}
        
        if not self.config_dir.exists():
            logger.warning(f"配置目录不存在: {self.config_dir}")
            return configs
        
        for config_file in self.config_dir.glob("*.yaml"):
            try:
                config = self.load_from_file(str(config_file))
                configs[config.id] = config
            except Exception as e:
                logger.error(f"加载配置文件失败 {config_file}: {e}")
        
        return configs
    
    def get_enabled_strategies(self) -> Dict[str, StrategyConfig]:
        all_configs = self.load_all()
        return {k: v for k, v in all_configs.items() if v.enabled}
    
    def save(self, config: StrategyConfig, path: Optional[str] = None) -> None:
        if path is None:
            path = str(self.config_dir / f"{config.id}.yaml")
        
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        data = {
            "strategy": config.to_dict()
        }
        
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        
        self._cache[config.id] = config
        logger.info(f"策略配置已保存: {path}")
    
    def create_template(self, strategy_id: str, category: str = "general") -> StrategyConfig:
        template = StrategyConfig(
            id=strategy_id,
            name=f"{strategy_id} 策略",
            category=category,
            version="1.0.0",
            enabled=True,
            description="请填写策略描述",
            parameters={
                "lookback_period": 20,
                "threshold": 0.05,
            },
            parameter_space={
                "lookback_period": ParameterSpace(
                    name="lookback_period",
                    param_type="int",
                    min_val=5,
                    max_val=60,
                    default=20,
                    description="回看周期",
                ),
                "threshold": ParameterSpace(
                    name="threshold",
                    param_type="float",
                    min_val=0.01,
                    max_val=0.20,
                    default=0.05,
                    description="信号阈值",
                ),
            },
            optimization={
                "objective": "sharpe_ratio",
                "method": "bayesian",
                "n_trials": 100,
                "timeout": 3600,
            },
            backtest={
                "lookback_years": 3,
                "initial_capital": 1000000,
                "commission_rate": 0.0003,
                "stamp_duty": 0.001,
                "slippage": 0.001,
            },
            constraints={
                "max_drawdown": 0.20,
                "min_trades": 50,
                "min_win_rate": 0.45,
            },
        )
        
        return template
    
    def clear_cache(self) -> None:
        self._cache.clear()
        logger.debug("配置缓存已清除")
