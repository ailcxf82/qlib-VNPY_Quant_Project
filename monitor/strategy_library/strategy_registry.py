"""
Strategy Registry - 策略注册表

管理所有可用策略的注册和发现。
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

from monitor.strategy_library.strategy_base import StrategyBase, StrategyConfig
from monitor.strategy_library.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


@dataclass
class StrategyInfo:
    strategy_id: str
    strategy_name: str
    strategy_class: Type[StrategyBase]
    category: str
    description: str = ""
    config_path: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "category": self.category,
            "description": self.description,
            "config_path": self.config_path,
            "tags": self.tags,
        }


class StrategyRegistry:
    _instance: Optional["StrategyRegistry"] = None
    _strategies: Dict[str, StrategyInfo] = {}
    _categories: Dict[str, List[str]] = {}
    
    def __new__(cls) -> "StrategyRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._strategies = {}
        self._categories = {}
        self._config_loader = ConfigLoader()
    
    @classmethod
    def register(
        cls,
        strategy_id: Optional[str] = None,
        category: str = "general",
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> Callable:
        def decorator(strategy_class: Type[StrategyBase]) -> Type[StrategyBase]:
            registry = cls()
            
            sid = strategy_id or strategy_class.STRATEGY_ID
            sname = strategy_class.STRATEGY_NAME
            scategory = category or strategy_class.STRATEGY_CATEGORY
            
            info = StrategyInfo(
                strategy_id=sid,
                strategy_name=sname,
                strategy_class=strategy_class,
                category=scategory,
                description=description or strategy_class.__doc__ or "",
                tags=tags or [],
            )
            
            registry._strategies[sid] = info
            
            if scategory not in registry._categories:
                registry._categories[scategory] = []
            registry._categories[scategory].append(sid)
            
            logger.debug(f"注册策略: {sid} ({sname}) -> {scategory}")
            
            return strategy_class
        
        return decorator
    
    @classmethod
    def get(cls, strategy_id: str) -> Optional[Type[StrategyBase]]:
        registry = cls()
        info = registry._strategies.get(strategy_id)
        return info.strategy_class if info else None
    
    @classmethod
    def get_info(cls, strategy_id: str) -> Optional[StrategyInfo]:
        registry = cls()
        return registry._strategies.get(strategy_id)
    
    @classmethod
    def list_all(cls) -> Dict[str, StrategyInfo]:
        return cls()._strategies
    
    @classmethod
    def list_by_category(cls, category: str) -> List[StrategyInfo]:
        registry = cls()
        strategy_ids = registry._categories.get(category, [])
        return [registry._strategies[sid] for sid in strategy_ids if sid in registry._strategies]
    
    @classmethod
    def list_categories(cls) -> List[str]:
        return list(cls()._categories.keys())
    
    @classmethod
    def create(
        cls,
        strategy_id: str,
        config: Optional[StrategyConfig] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Optional[StrategyBase]:
        strategy_class = cls.get(strategy_id)
        
        if strategy_class is None:
            logger.error(f"策略未注册: {strategy_id}")
            return None
        
        if config is None:
            config = cls()._config_loader.load(strategy_id)
        
        try:
            strategy = strategy_class(config=config, parameters=parameters)
            return strategy
        except Exception as e:
            logger.error(f"创建策略实例失败 {strategy_id}: {e}")
            return None
    
    @classmethod
    def create_from_config(cls, config_path: str) -> Optional[StrategyBase]:
        config = StrategyConfig.from_yaml(config_path)
        return cls.create(config.id, config=config)
    
    @classmethod
    def discover(cls, package_path: str = "monitor.strategy_library.strategies") -> int:
        count = 0
        
        try:
            package = importlib.import_module(package_path)
            package_dir = Path(package.__file__).parent
            
            for module_file in package_dir.glob("*.py"):
                if module_file.name.startswith("_"):
                    continue
                
                module_name = f"{package_path}.{module_file.stem}"
                
                try:
                    importlib.import_module(module_name)
                    count += 1
                except Exception as e:
                    logger.warning(f"加载策略模块失败 {module_name}: {e}")
        
        except Exception as e:
            logger.error(f"发现策略失败: {e}")
        
        logger.info(f"发现 {count} 个策略模块")
        return count
    
    @classmethod
    def get_strategy_count(cls) -> int:
        return len(cls()._strategies)
    
    @classmethod
    def is_registered(cls, strategy_id: str) -> bool:
        return strategy_id in cls()._strategies
    
    @classmethod
    def unregister(cls, strategy_id: str) -> bool:
        registry = cls()
        
        if strategy_id not in registry._strategies:
            return False
        
        info = registry._strategies.pop(strategy_id)
        
        if info.category in registry._categories:
            registry._categories[info.category] = [
                sid for sid in registry._categories[info.category] 
                if sid != strategy_id
            ]
        
        logger.debug(f"注销策略: {strategy_id}")
        return True
    
    @classmethod
    def clear(cls) -> None:
        registry = cls()
        registry._strategies.clear()
        registry._categories.clear()
        logger.debug("策略注册表已清空")


def register_strategy(
    strategy_id: Optional[str] = None,
    category: str = "general",
    description: str = "",
    tags: Optional[List[str]] = None,
) -> Callable:
    return StrategyRegistry.register(
        strategy_id=strategy_id,
        category=category,
        description=description,
        tags=tags,
    )
