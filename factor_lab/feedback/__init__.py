"""L2 → L1 反馈回路（阶段 E）。

本子包把 L2 的验证结论（C2 证书、cycle 报告、L3 manifest）聚合成 C3
``FeedbackBundle``，供 RD-Agent 下一轮 hypothesis generation 的 RAG 注入使用。

对外唯一出口：

>>> from factor_lab.feedback import FeedbackBundle, build_feedback_bundle

详见 ``docs/STAGE_E_REPORT.md``。
"""

from factor_lab.feedback.schema import (
    ActiveFactorSummary,
    FailedCandidateSummary,
    FeedbackBundle,
    RetiredFactorSummary,
    UniverseSubBundle,
)

__all__ = [
    "ActiveFactorSummary",
    "FailedCandidateSummary",
    "FeedbackBundle",
    "RetiredFactorSummary",
    "UniverseSubBundle",
]
