"""
factor_registry — L3 正式因子注册表。

L3 只接收 ``factor_validation.CertifiedFactorRecord`` 中 ``decision == PASS`` 的因子，
通过 ``FactorRegistry.register()`` 写入 manifest 并把因子值落到 production parquet。
任何 production 模块（``EnsembleModelManager`` / 训练 / 预测 / 回测）必须通过
``feature.production_factor_loader`` 读取因子，**禁止**直接读 ``factor_registry/parquet/``。

L3 不允许：
- 接受未经 L2 认证的因子
- "实验性"启用因子（实验在 L1 / 认证在 L2）
- 直接读 ``factor_lab/workspace/``

详见 ``docs/ARCHITECTURE_FACTOR_LAB.md``。
"""

from factor_registry.schema import ProductionFactorRecord, ProductionStatus

__all__ = ["ProductionFactorRecord", "ProductionStatus"]
