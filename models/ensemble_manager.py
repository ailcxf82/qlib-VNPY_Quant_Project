"""
Qlib 多模型协同封装：负责统一训练/预测接口，并通过 qlib 的 Ensemble 算法完成加权融合。
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import pandas as pd
from qlib.model.ens.ensemble import AverageEnsemble

from models.model_registry import create_model
from models.weighted_ensemble import ICIRWeightedAverageAdapter, MetaLearnerAdapter


class _QlibAverageAdapter:
    """封装 qlib AverageEnsemble，使其可以直接处理 pd.Series 预测结果。"""

    def __init__(self):
        self._ensemble = AverageEnsemble()

    def __call__(self, preds: Dict[str, pd.Series]) -> pd.Series:
        if not preds:
            raise ValueError("无可融合的预测结果")
        if len(preds) == 1:
            return next(iter(preds.values())).rename("qlib_ensemble")
        formatted = {name: series.to_frame(name) for name, series in preds.items()}
        result = self._ensemble(formatted)
        if isinstance(result, pd.DataFrame):
            # 默认只关心单列融合结果；若存在多列，则取平均。
            return result.mean(axis=1).rename("qlib_ensemble")
        if isinstance(result, pd.Series):
            return result.rename("qlib_ensemble")
        raise TypeError(f"无法识别的 qlib Ensemble 输出类型: {type(result)}")


class EnsembleAggregator:
    """将 Ensemble 策略适配为 Series -> Series 的接口。"""

    STRATEGY_MAP = {
        "average": _QlibAverageAdapter,
        "weighted_average": ICIRWeightedAverageAdapter,
        "meta_learner": MetaLearnerAdapter,
        "meta_learner_ridge": lambda: MetaLearnerAdapter(model_type="ridge", alpha=1.0),
        "meta_learner_linear": lambda: MetaLearnerAdapter(model_type="linear"),
    }

    def __init__(self, strategy: str = "average", strategy_params: Optional[Dict] = None):
        """
        参数:
            strategy: 聚合策略名称
            strategy_params: 策略参数（如 meta_learner 的 alpha）
        """
        strategy = (strategy or "average").lower()
        strategy_params = strategy_params or {}
        
        if strategy == "meta_learner":
            # 支持通过参数指定模型类型
            model_type = strategy_params.get("model_type", "ridge")
            alpha = strategy_params.get("alpha", 1.0)
            self._adapter = MetaLearnerAdapter(model_type=model_type, alpha=alpha)
        elif strategy in self.STRATEGY_MAP:
            adapter_cls_or_factory = self.STRATEGY_MAP[strategy]
            if callable(adapter_cls_or_factory) and not isinstance(adapter_cls_or_factory, type):
                # 是工厂函数
                self._adapter = adapter_cls_or_factory()
            else:
                # 是类
                self._adapter = adapter_cls_or_factory()
        else:
            raise ValueError(f"暂不支持的 Ensemble 策略: {strategy}")

    def aggregate(self, preds: Dict[str, pd.Series]) -> pd.Series:
        return self._adapter(preds)
    
    def fit(self, valid_preds: Dict[str, pd.Series], valid_label: pd.Series):
        """
        在验证集上训练聚合器（仅对 weighted_average 和 meta_learner 有效）。
        
        参数:
            valid_preds: {模型名: 验证集预测值}
            valid_label: 验证集标签
        """
        if hasattr(self._adapter, "fit"):
            self._adapter.fit(valid_preds, valid_label)


class EnsembleModelManager:
    """
    统一管理多模型训练与预测。

    config 示例:
    ensemble:
      aggregator: average
      models:
        - name: lgb
          type: lightgbm
          config_key: lightgbm_config
        - name: mlp
          type: mlp
          config_key: mlp_config
    """

    def __init__(self, pipeline_cfg: Dict, ensemble_cfg: Optional[Dict] = None):
        self.pipeline_cfg = pipeline_cfg
        self.ensemble_cfg = ensemble_cfg or {}
        self.models = OrderedDict()
        self._build_models()
        aggregator_strategy = (self.ensemble_cfg or {}).get("aggregator", "average")
        aggregator_params = (self.ensemble_cfg or {}).get("aggregator_params", {})
        self.aggregator: Optional[EnsembleAggregator] = None
        if aggregator_strategy and aggregator_strategy != "disabled":
            self.aggregator = EnsembleAggregator(aggregator_strategy, aggregator_params)

    def _resolve_config_path(self, spec: Dict) -> str:
        if "config" in spec:
            return spec["config"]
        config_key = spec.get("config_key")
        if config_key and config_key in self.pipeline_cfg:
            return self.pipeline_cfg[config_key]
        raise ValueError(f"模型 {spec.get('name')} 未提供 config 或 config_key")

    def _default_specs(self) -> List[Dict]:
        return [
            {"name": "lgb", "type": "lightgbm", "config_key": "lightgbm_config"},
            {"name": "mlp", "type": "mlp", "config_key": "mlp_config"},
        ]

    def _build_models(self):
        specs = (self.ensemble_cfg or {}).get("models")
        if not specs:
            specs = self._default_specs()
        for spec in specs:
            name = spec["name"]
            model_type = spec["type"]
            cfg_path = self._resolve_config_path(spec)
            self.models[name] = create_model(model_type, cfg_path)

    def fit(
        self,
        train_feat: pd.DataFrame,
        train_label: pd.Series,
        valid_feat: Optional[pd.DataFrame] = None,
        valid_label: Optional[pd.Series] = None,
    ):
        # 先训练所有基础模型
        for model in self.models.values():
            model.fit(train_feat, train_label, valid_feat, valid_label)
        
        # 如果聚合器需要训练（如 weighted_average 或 meta_learner），在验证集上训练
        if self.aggregator is not None and hasattr(self.aggregator, "fit"):
            if valid_feat is not None and valid_label is not None and len(valid_feat) > 0 and len(valid_label) > 0:
                # 获取验证集预测（模型已训练完成）
                valid_blend, valid_preds, _ = self.predict(valid_feat)
                if valid_preds:
                    self.aggregator.fit(valid_preds, valid_label)

    def predict(self, feat: pd.DataFrame) -> Tuple[Optional[pd.Series], Dict[str, pd.Series], Dict[str, object]]:
        preds: Dict[str, pd.Series] = {}
        aux: Dict[str, object] = {}
        for name, model in self.models.items():
            output = model.predict(feat)
            if isinstance(output, tuple):
                preds[name], aux[name] = output
            else:
                preds[name] = output
        blended = None
        if self.aggregator is not None:
            blended = self.aggregator.aggregate(preds)
        return blended, preds, aux

    def save(self, output_dir: str, tag: str):
        for model in self.models.values():
            model.save(output_dir, tag)

    def load(self, output_dir: str, tag: str):
        for model in self.models.values():
            model.load(output_dir, tag)

    def get_model(self, name: str):
        return self.models.get(name)

    def list_model_names(self) -> List[str]:
        return list(self.models.keys())

