"""
factor_lab — L1 因子试验场。

本包是因子试验场的容器：使用 RD-Agent + Qlib 大量产候选因子，把每个候选打包成
``CandidateFactorPackage`` (契约 C1) 落到 ``workspace/candidates/`` 下，等待
``factor_validation`` (L2) 取走验证。

L1 不允许：
- 直接读写 ``factor_registry/``（L3）
- 直接调用 production 模型 / 回测
- 在试验场内做"是否准入"的判断（这是 L2 的职责）

详见 ``docs/ARCHITECTURE_FACTOR_LAB.md``。
"""

from factor_lab.exporters.schema import CandidateFactorPackage

__all__ = ["CandidateFactorPackage"]
