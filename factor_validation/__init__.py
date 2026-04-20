"""
factor_validation — L2 因子验证环境。

L2 是唯一的"准入把关者"。它读取 ``factor_lab.CandidateFactorPackage`` (契约 C1) +
``ValidationProfile`` (yaml)，按 profile 启用一组 ``Check`` 跑完，输出
``CertifiedFactorRecord`` (契约 C2)。只有 ``decision == 'PASS'`` 的 record 才能被
``factor_registry`` (L3) 接收。

L2 不允许：
- 修改候选因子代码
- 直接写 ``factor_registry/``（promote 是独立动作）
- 重新启动 RD-Agent

详见 ``docs/ARCHITECTURE_FACTOR_LAB.md`` §4 / §5。
"""

from factor_validation.schema import (
    CertifiedFactorRecord,
    CheckResult,
    Decision,
)

__all__ = ["CertifiedFactorRecord", "CheckResult", "Decision"]
