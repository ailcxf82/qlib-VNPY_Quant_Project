"""
Qlib 多模型协同封装：负责统一训练/预测接口，并通过 qlib 的 Ensemble 算法完成加权融合。
"""

from __future__ import annotations

from collections import OrderedDict
import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from qlib.model.ens.ensemble import AverageEnsemble

from models.model_registry import create_model
from utils import load_yaml_config
from models.weighted_ensemble import ICIRWeightedAverageAdapter, MetaLearnerAdapter

logger = logging.getLogger(__name__)
_DEBUG_LOG_PATH = "debug-78b9cb.log"


def _agent_debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: Dict[str, Any]) -> None:
    payload = {
        "sessionId": "78b9cb",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


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
        "dynamic_weighted": None,  # 将在__init__中动态创建
        "adaptive": None,  # 将在__init__中动态创建
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
        
        # 支持动态权重集成策略
        if strategy == "dynamic_weighted":
            from models.dynamic_ensemble import DynamicWeightedEnsemble
            self._adapter = DynamicWeightedEnsemble(
                window=int(strategy_params.get("window", 60)),
                half_life=int(strategy_params.get("half_life", 20)),
                min_weight=float(strategy_params.get("min_weight", 0.05)),
                max_weight=float(strategy_params.get("max_weight", 0.5)),
                clip_negative=bool(strategy_params.get("clip_negative", True)),
                use_softmax=bool(strategy_params.get("use_softmax", False)),
                temperature=float(strategy_params.get("temperature", 1.0)),
            )
        elif strategy == "adaptive":
            from models.dynamic_ensemble import AdaptiveEnsemble
            self._adapter = AdaptiveEnsemble(
                volatility_window=int(strategy_params.get("volatility_window", 20)),
            )
        elif strategy == "meta_learner":
            # 支持通过参数指定模型类型
            model_type = strategy_params.get("model_type", "ridge")
            alpha = strategy_params.get("alpha", 1.0)
            self._adapter = MetaLearnerAdapter(model_type=model_type, alpha=alpha)
        elif strategy in self.STRATEGY_MAP:
            adapter_cls_or_factory = self.STRATEGY_MAP[strategy]
            if adapter_cls_or_factory is None:
                raise ValueError(f"策略 {strategy} 需要特殊处理，但未正确初始化")
            if callable(adapter_cls_or_factory) and not isinstance(adapter_cls_or_factory, type):
                # 是工厂函数
                self._adapter = adapter_cls_or_factory()
            else:
                # 是类
                self._adapter = adapter_cls_or_factory()
        else:
            raise ValueError(f"暂不支持的 Ensemble 策略: {strategy}")

    def aggregate(self, preds: Dict[str, pd.Series], labels: Optional[pd.Series] = None) -> pd.Series:
        # 支持DynamicWeightedEnsemble和AdaptiveEnsemble
        if hasattr(self._adapter, 'aggregate'):
            return self._adapter.aggregate(preds, labels)
        else:
            # 兼容旧的适配器（可调用对象）
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
        self.specs: List[Dict] = []
        self._feature_sets: Dict[str, List[str]] = {}
        raw_model_features = self.pipeline_cfg.get("model_features", {}) or {}
        # 统一小写 key，避免大小写不一致导致未命中
        self._model_features: Dict[str, object] = {
            str(k).strip().lower(): v for k, v in raw_model_features.items()
        }
        self._feature_log_done: set[str] = set()
        # 归一化参数（预测阶段需要与训练一致）
        # - per_model: {model_name: {"mean": Series, "std": Series}}
        # - global: 兼容旧逻辑（单套 mean/std，按模型列子集使用）
        self._norm_per_model: Dict[str, Dict[str, pd.Series]] = {}
        self._norm_global: Optional[Dict[str, pd.Series]] = None
        # 加载 data.yaml 中的 feature_sets（若存在）
        data_cfg_path = self.pipeline_cfg.get("data_config")
        if isinstance(data_cfg_path, str):
            try:
                data_cfg = load_yaml_config(data_cfg_path)
                self._feature_sets = data_cfg.get("data", {}).get("feature_sets", {}) or {}
            except Exception as e:
                logger.warning(
                    "读取 data_config 失败，将忽略 data.feature_sets（data_config=%s, cwd=%s, err=%s）",
                    data_cfg_path,
                    os.getcwd(),
                    e,
                )
                self._feature_sets = {}
        self._build_models()
        aggregator_strategy = (self.ensemble_cfg or {}).get("aggregator", "average")
        aggregator_params = (self.ensemble_cfg or {}).get("aggregator_params", {})
        self.aggregator: Optional[EnsembleAggregator] = None
        if aggregator_strategy and aggregator_strategy != "disabled":
            self.aggregator = EnsembleAggregator(aggregator_strategy, aggregator_params)

    # =========================
    # Normalization helpers
    # =========================
    @staticmethod
    def _fit_norm(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        mean = df.mean()
        std = df.std().replace(0, 1)
        return mean, std

    @staticmethod
    def _align_to_mean(df: pd.DataFrame, mean: pd.Series) -> pd.DataFrame:
        """
        将 df 对齐到 mean 的列顺序：
        - 缺失列用 mean 填充（归一化后为 0）
        - 多余列忽略
        """
        expected_cols = list(mean.index)
        aligned = pd.DataFrame(index=df.index, columns=expected_cols, dtype=float)
        for c in expected_cols:
            if c in df.columns:
                aligned[c] = df[c]
            else:
                aligned[c] = float(mean[c]) if c in mean.index else 0.0
        return aligned

    @staticmethod
    def _apply_norm(df: pd.DataFrame, mean: pd.Series, std: pd.Series) -> pd.DataFrame:
        aligned = EnsembleModelManager._align_to_mean(df, mean)
        std2 = std.reindex(mean.index).replace(0, 1)
        out = (aligned - mean) / std2
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out.clip(-5, 5)

    def set_norm_meta(self, norm_meta: Dict[str, Any]):
        """
        供预测阶段注入训练时保存的归一化参数（兼容旧格式与 per-model 格式）。

        支持两种 schema：
        1) 旧格式：{"feature_mean": {...}, "feature_std": {...}}
        2) 新格式：{"models": {"lgb": {"mean": {...}, "std": {...}}, ...}}
        """
        self._norm_per_model = {}
        self._norm_global = None
        if not isinstance(norm_meta, dict):
            return
        if "models" in norm_meta and isinstance(norm_meta["models"], dict):
            for name, ms in norm_meta["models"].items():
                if not isinstance(ms, dict):
                    continue
                mean = ms.get("mean")
                std = ms.get("std")
                if isinstance(mean, dict) and isinstance(std, dict):
                    self._norm_per_model[str(name).strip().lower()] = {
                        "mean": pd.Series(mean),
                        "std": pd.Series(std),
                    }
            return
        if "feature_mean" in norm_meta and "feature_std" in norm_meta:
            try:
                self._norm_global = {
                    "mean": pd.Series(norm_meta["feature_mean"]),
                    "std": pd.Series(norm_meta["feature_std"]),
                }
            except Exception:
                self._norm_global = None

    def get_norm_meta(self) -> Dict[str, Any]:
        """训练阶段导出归一化参数，供 RollingTrainer 落盘。"""
        if self._norm_per_model:
            return {
                "mode": "per_model",
                "models": {
                    k: {"mean": v["mean"].to_dict(), "std": v["std"].to_dict()} for k, v in self._norm_per_model.items()
                },
            }
        if self._norm_global is not None:
            return {
                "mode": "global",
                "feature_mean": self._norm_global["mean"].to_dict(),
                "feature_std": self._norm_global["std"].to_dict(),
            }
        return {"mode": "none"}

    def _resolve_config(self, spec: Dict):
        # 1) spec 内直接给 config（允许 str path 或 dict）
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

    def _specs_from_base_models(self) -> List[Dict]:
        """
        支持“只改配置即可启用模型”的声明方式：
          base_models: ["lgb", "mlp", "gru"]
        其中：
        - lgb/mlp 默认沿用现有 config_key（保持原逻辑）
        - gru 优先读取 pipeline_cfg.model_gru（dict），否则回退到 gru_config 文件路径
        """
        base = self.pipeline_cfg.get("base_models")
        if not base:
            return []
        specs: List[Dict] = []
        for name in base:
            n = str(name).strip().lower()
            if n in {"lgb", "lightgbm"}:
                specs.append({"name": "lgb", "type": "lightgbm", "config_key": "lightgbm_config"})
            elif n == "mlp":
                specs.append({"name": "mlp", "type": "mlp", "config_key": "mlp_config"})
            elif n == "gru":
                if "model_gru" in self.pipeline_cfg and isinstance(self.pipeline_cfg["model_gru"], dict):
                    specs.append({"name": "gru", "type": "gru", "config": {"model": self.pipeline_cfg["model_gru"]}})
                elif "gru_config" in self.pipeline_cfg:
                    specs.append({"name": "gru", "type": "gru", "config_key": "gru_config"})
                else:
                    raise ValueError("base_models 包含 gru，但未提供 model_gru(dict) 或 gru_config(path)")
            else:
                raise ValueError(f"未知 base_models 项: {name}")
        return specs

    def _build_models(self):
        specs = (self.ensemble_cfg or {}).get("models")
        if not specs:
            specs = self._specs_from_base_models()
        if not specs:
            specs = self._default_specs()
        self.specs = specs
        for spec in specs:
            name = spec["name"]
            model_type = spec["type"]
            cfg = self._resolve_config(spec)
            self.models[name] = create_model(model_type, cfg)

    def update_feature_set(self, name: str, columns: Optional[List[str]]) -> None:
        """运行时覆盖/补充 data.feature_sets 中的某一集合（如 RD-Agent parquet 列名）。"""
        key = str(name).strip()
        merged = dict(self._feature_sets)
        merged[key] = list(columns or [])
        # #region agent log
        _agent_debug_log(
            "pre-fix",
            "H1",
            "models/ensemble_manager.py:update_feature_set",
            "runtime feature set updated",
            {
                "feature_set_name": key,
                "columns_count": len(merged[key]),
                "columns_sample": [repr(c) for c in merged[key][:5]],
                "column_type_sample": [type(c).__name__ for c in merged[key][:5]],
            },
        )
        # #endregion
        self._feature_sets = merged
        logger.info("已更新 feature_sets[%s]，列数=%d", key, len(merged[key]))

    def _resolve_feature_cols(self, model_name: str, all_cols: List[str]) -> Optional[List[str]]:
        """解析每个模型需要的特征列。"""
        if not self._model_features:
            return None
        missing_mode = str(self.pipeline_cfg.get("feature_missing_mode", "compat") or "compat").strip().lower()
        if missing_mode not in {"compat", "strict"}:
            missing_mode = "compat"
        model_key = str(model_name).strip().lower()
        spec = self._model_features.get(model_key)
        # #region agent log
        _agent_debug_log(
            "pre-fix",
            "H2",
            "models/ensemble_manager.py:_resolve_feature_cols:entry",
            "resolve feature cols entry",
            {
                "model_name": model_name,
                "spec_type": type(spec).__name__ if spec is not None else "None",
                "spec_preview": repr(spec)[:300],
                "all_cols_count": len(all_cols),
                "all_cols_sample": [repr(c) for c in all_cols[:5]],
                "all_col_type_sample": [type(c).__name__ for c in all_cols[:5]],
            },
        )
        # #endregion
        if spec is None:
            raise ValueError(
                f"model_features 已配置，但未包含模型 {model_name}。"
                "请在 pipeline.yaml 的 model_features 中为该模型指定特征集合。"
            )
        # str：引用 data.yaml 的 feature_sets
        if isinstance(spec, str):
            key = spec.strip()
            if key in {"*", "all", "ALL"}:
                return None
            if key in self._feature_sets:
                cols = list(self._feature_sets[key] or [])
            else:
                avail = []
                try:
                    if isinstance(self._feature_sets, dict):
                        avail = sorted([str(k) for k in self._feature_sets.keys()])[:30]
                except Exception:
                    avail = []
                raise ValueError(
                    f"model_features[{model_name}]={key} 未在 data.feature_sets 中定义"
                    + (f"（可用 keys 示例: {avail}）" if avail else "")
                )
        # list：要么全部为 feature_sets 的 key（按顺序拼接列名、去重），要么直接给列名/表达式
        elif isinstance(spec, list):
            raw = list(spec)
            if raw and all(isinstance(x, str) for x in raw):
                keys = [str(x).strip() for x in raw]
                if keys and all(k in self._feature_sets for k in keys):
                    cols = []
                    seen: set[object] = set()
                    for k in keys:
                        for c in self._feature_sets.get(k) or []:
                            if c not in seen:
                                seen.add(c)
                                cols.append(c)
                else:
                    cols = raw
            else:
                cols = raw
        else:
            raise ValueError(f"model_features[{model_name}] 仅支持 str 或 list")
        if not cols:
            return None
        missing = [c for c in cols if c not in all_cols]
        if missing:
            available = [c for c in cols if c in all_cols]
            # #region agent log
            _agent_debug_log(
                "pre-fix",
                "H3",
                "models/ensemble_manager.py:_resolve_feature_cols:missing_check",
                "feature missing check",
                {
                    "model_name": model_name,
                    "missing_count": len(missing),
                    "available_count": len(available),
                    "missing_sample": [repr(c) for c in missing[:5]],
                    "available_sample": [repr(c) for c in available[:5]],
                    "resolved_cols_sample": [repr(c) for c in cols[:5]],
                    "resolved_col_type_sample": [type(c).__name__ for c in cols[:5]],
                },
            )
            # #endregion
            # strict：任何缺失直接中断（用于发现数据/表达式问题）
            if missing_mode == "strict":
                raise ValueError(f"模型 {model_name} 特征缺失（strict）: {missing[:10]}")
            # compat：某些特征可能在特征清理阶段（如全 NaN 列）被移除，不应直接中断训练。
            if available:
                logger.warning(
                    "模型 %s 有 %d 个配置特征在当前窗口缺失，已自动跳过（示例: %s）",
                    model_name,
                    len(missing),
                    missing[:10],
                )
                cols = available
            else:
                raise ValueError(f"模型 {model_name} 特征全部缺失: {missing[:10]}")
        if model_name not in self._feature_log_done:
            logger.info(
                "模型 %s 使用特征集合=%s，列数=%d（示例: %s）",
                model_name,
                spec,
                len(cols),
                cols[:8],
            )
            self._feature_log_done.add(model_name)
        return cols

    def _select_features(self, model_name: str, df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df is None or df.empty:
            return df
        cols = self._resolve_feature_cols(model_name, list(df.columns))
        if not cols:
            return df
        # #region agent log
        _agent_debug_log(
            "pre-fix",
            "H4",
            "models/ensemble_manager.py:_select_features",
            "select feature columns",
            {
                "model_name": model_name,
                "df_cols_count": len(df.columns),
                "selected_cols_count": len(cols),
                "df_cols_sample": [repr(c) for c in list(df.columns)[:5]],
                "selected_cols_sample": [repr(c) for c in cols[:5]],
            },
        )
        # #endregion
        selected = df.loc[:, cols]
        # 仅首次打印，避免日志过多
        log_key = f"{model_name}__select"
        if log_key not in self._feature_log_done:
            logger.info(
                "模型 %s 特征选择: 总列数=%d -> 选择列数=%d",
                model_name,
                len(df.columns),
                len(selected.columns),
            )
            self._feature_log_done.add(log_key)
        return selected

    def fit(
        self,
        train_feat: pd.DataFrame,
        train_label: pd.Series,
        valid_feat: Optional[pd.DataFrame] = None,
        valid_label: Optional[pd.Series] = None,
        *,
        history_feat: Optional[pd.DataFrame] = None,
    ):
        def _tail_by_trading_days(df: pd.DataFrame, n_days: int) -> pd.DataFrame:
            """
            取最近 n 个“交易日”的数据（按 MultiIndex 的 datetime 去重计数，而非按日历天）。
            df 为空则原样返回。
            """
            if df is None or df.empty or n_days <= 0:
                return df
            idx = df.index
            if not isinstance(idx, pd.MultiIndex) or "datetime" not in idx.names:
                return df
            dts = pd.to_datetime(idx.get_level_values("datetime")).normalize()
            unique_days = pd.Index(sorted(pd.unique(dts)))
            if len(unique_days) <= n_days:
                return df
            cutoff = unique_days[-n_days]
            return df.loc[dts >= cutoff]

        # 读取“按模型训练窗口天数”配置（可选）
        rolling = self.pipeline_cfg.get("rolling", {}) or {}
        raw_days_map = rolling.get("model_train_days") or rolling.get("train_days_by_model") or {}
        days_map: Dict[str, int] = {}
        if isinstance(raw_days_map, dict):
            for k, v in raw_days_map.items():
                try:
                    days_map[str(k).strip().lower()] = int(v)
                except Exception:
                    continue

        # 归一化模式：
        # - global: 使用统一 mean/std（来自 RollingTrainer 或 PredictorEngine 注入的 norm_meta）
        # - per_model: 每个模型单独拟合 mean/std（训练阶段默认）
        norm_mode = str(rolling.get("feature_normalization", rolling.get("normalization_mode", "per_model"))).strip().lower()
        if norm_mode not in {"per_model", "global", "none"}:
            norm_mode = "per_model"
        # 训练阶段：若显式要求 global 且尚未注入 global norm，则退回 per_model
        if norm_mode == "global" and self._norm_global is None and not self._norm_per_model:
            norm_mode = "per_model"
        # 训练阶段：per_model 每次 fit 都重算（按窗口），避免跨窗口复用
        if norm_mode == "per_model":
            self._norm_per_model = {}

        # 先训练所有基础模型
        for name, model in self.models.items():
            # 1) 按模型裁剪训练集：保持验证窗口对齐，但允许不同模型使用不同长度的训练历史
            n_days = days_map.get(str(name).strip().lower())
            tr_raw = _tail_by_trading_days(train_feat, n_days) if n_days else train_feat
            tr_lbl = train_label.reindex(tr_raw.index) if tr_raw is not None else train_label

            # 2) 再做按模型特征集合选择（feature_sets / model_features）
            tr = self._select_features(name, tr_raw)
            va = self._select_features(name, valid_feat) if valid_feat is not None else None

            # 3) 归一化（按模型/按窗口）
            if norm_mode == "per_model":
                mean, std = self._fit_norm(tr)
                self._norm_per_model[str(name).strip().lower()] = {"mean": mean, "std": std}
                tr = self._apply_norm(tr, mean, std)
                if va is not None:
                    va = self._apply_norm(va, mean, std)
            elif norm_mode == "global" and self._norm_global is not None:
                gmean, gstd = self._norm_global["mean"], self._norm_global["std"]
                # 仅对子集列应用 global norm
                mean = gmean.reindex(tr.columns)
                std = gstd.reindex(tr.columns)
                tr = self._apply_norm(tr, mean, std)
                if va is not None:
                    mean2 = gmean.reindex(va.columns)
                    std2 = gstd.reindex(va.columns)
                    va = self._apply_norm(va, mean2, std2)

            # 4) 训练
            model.fit(tr, tr_lbl, va, valid_label)
        
        # 如果聚合器需要训练（如 weighted_average 或 meta_learner），在验证集上训练
        if self.aggregator is not None and hasattr(self.aggregator, "fit"):
            if valid_feat is not None and valid_label is not None and len(valid_feat) > 0 and len(valid_label) > 0:
                # 获取验证集预测（模型已训练完成）
                # 对序列模型（如 GRU）需要提供历史特征，避免验证集前期因为缺历史而预测为空
                history = history_feat if history_feat is not None else train_feat
                valid_blend, valid_preds, _ = self.predict(valid_feat, history_feat=history)
                if valid_preds:
                    self.aggregator.fit(valid_preds, valid_label)

    def predict(
        self,
        feat: pd.DataFrame,
        *,
        history_feat: Optional[pd.DataFrame] = None,
    ) -> Tuple[Optional[pd.Series], Dict[str, pd.Series], Dict[str, object]]:
        preds: Dict[str, pd.Series] = {}
        aux: Dict[str, object] = {}
        rolling = self.pipeline_cfg.get("rolling", {}) or {}
        norm_mode = str(rolling.get("feature_normalization", rolling.get("normalization_mode", "per_model"))).strip().lower()
        if norm_mode not in {"per_model", "global", "none"}:
            norm_mode = "per_model"
        for name, model in self.models.items():
            feat_view = self._select_features(name, feat)
            history_view = self._select_features(name, history_feat) if history_feat is not None else None

            # 预测阶段归一化：优先 per_model（训练时保存），否则 global（旧格式）
            mkey = str(name).strip().lower()
            if norm_mode != "none":
                if mkey in self._norm_per_model:
                    mean = self._norm_per_model[mkey]["mean"]
                    std = self._norm_per_model[mkey]["std"]
                    feat_view = self._apply_norm(feat_view, mean, std)
                    if history_view is not None:
                        history_view = self._apply_norm(history_view, mean, std)
                elif self._norm_global is not None:
                    gmean, gstd = self._norm_global["mean"], self._norm_global["std"]
                    mean = gmean.reindex(feat_view.columns)
                    std = gstd.reindex(feat_view.columns)
                    feat_view = self._apply_norm(feat_view, mean, std)
                    if history_view is not None:
                        mean2 = gmean.reindex(history_view.columns)
                        std2 = gstd.reindex(history_view.columns)
                        history_view = self._apply_norm(history_view, mean2, std2)

            # 尽量向序列模型传递历史特征；对不支持的模型自动回退
            if history_view is not None:
                try:
                    output = model.predict(feat_view, history_feat=history_view)
                except TypeError:
                    output = model.predict(feat_view)
            else:
                output = model.predict(feat_view)

            if isinstance(output, tuple):
                preds[name], aux[name] = output
            else:
                preds[name] = output

        blended = None
        if self.aggregator is not None:
            # 动态权重集成需要labels，但在predict阶段没有labels，所以传None
            blended = self.aggregator.aggregate(preds, labels=None)
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

