"""
模型注册表：集中管理可用模型，便于在 Ensemble 中动态扩展。

注意：部分模型支持直接传入 dict 配置（如 MLP/GRU），部分模型仍要求配置文件路径（如 LightGBM）。
"""

from __future__ import annotations

from typing import Dict, Type, Union, Any

from models.lightgbm_model import LightGBMModelWrapper
# torch 相关模型做可选依赖：未安装 torch 时不阻断整个工程（例如仅训练 LightGBM）
try:
    from models.mlp_model import MLPRegressor  # type: ignore
except ModuleNotFoundError:
    MLPRegressor = None  # type: ignore

try:
    from models.gru_model import GRURegressor  # type: ignore
except ModuleNotFoundError:
    GRURegressor = None  # type: ignore
# NOTE: 若后续需要引入更多模型，在此注册即可。
MODEL_REGISTRY: Dict[str, Type] = {
    "lightgbm": LightGBMModelWrapper,
}

if MLPRegressor is not None:
    MODEL_REGISTRY["mlp"] = MLPRegressor
if GRURegressor is not None:
    MODEL_REGISTRY["gru"] = GRURegressor


def create_model(model_type: str, config: Union[str, Dict[str, Any]]):
    """根据类型创建模型实例。"""
    mt = model_type.lower()
    cls = MODEL_REGISTRY.get(mt)
    if cls is None:
        # 给出更清晰的错误：通常是可选依赖未安装
        if mt in {"mlp", "gru"}:
            raise ValueError(f"模型类型 {model_type} 需要可选依赖 torch，但当前环境未安装或导入失败")
        raise ValueError(f"未注册的模型类型: {model_type}")
    # LightGBM 目前只支持从 YAML 文件路径加载
    if mt == "lightgbm" and not isinstance(config, str):
        raise ValueError("LightGBMModelWrapper 仅支持传入配置文件路径（str），不支持 dict")
    return cls(config)

