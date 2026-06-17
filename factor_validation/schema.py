"""
契约 C2：``CertifiedFactorRecord`` —— L2 → L3 之间的唯一数据交换格式。

包含：
* ``CheckResult``：单项检查结果
* ``Decision``：枚举（PASS / FAIL / HOLD）
* ``CertifiedFactorRecord``：完整认证记录，嵌入原始 ``CandidateFactorPackage``

设计目标：
* **不可变**：``model_config = ConfigDict(frozen=True)``
* **可复现**：``profile_hash`` (sha256) 永久绑定本次验证用的标准
* **可审计**：每个 check 单独成 ``CheckResult``，结论 + 评分 + 阈值 + 详情齐全

详见 ``docs/CONTRACT_C2_CERTIFIED.md``。
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from factor_lab.exporters.schema import CandidateFactorPackage

_PROFILE_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.]+)?$")


class Decision(str, Enum):
    """验证决策。"""

    PASS = "PASS"
    FAIL = "FAIL"
    HOLD = "HOLD"  # 需人工复核


class CheckResult(BaseModel):
    """单项检查的结果（不可变）。"""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    name: str = Field(..., min_length=1, max_length=64, description="check 名，例如 'ic'")
    passed: bool = Field(..., description="本项检查是否通过")
    score: float | None = Field(
        default=None, description="本项得分（0~1 区间，可选）；用于聚合"
    )
    threshold: float | None = Field(
        default=None, description="本项判定阈值；便于报告解释"
    )
    detail: dict[str, Any] = Field(
        default_factory=dict, description="任意附加信息（如 IC 序列摘要、p 值等）"
    )
    elapsed_ms: int = Field(..., ge=0, description="本项检查耗时（毫秒）")

    @field_validator("score")
    @classmethod
    def _check_score(cls, v: float | None) -> float | None:
        if v is None:
            return None
        if math.isnan(v) or math.isinf(v):
            raise ValueError(f"score 必须有限: {v!r}")
        if v < 0 or v > 1:
            raise ValueError(f"score 必须在 [0,1]: {v}")
        return float(v)

    @field_validator("threshold")
    @classmethod
    def _check_threshold(cls, v: float | None) -> float | None:
        if v is None:
            return None
        if math.isnan(v) or math.isinf(v):
            raise ValueError(f"threshold 必须有限: {v!r}")
        return float(v)


class CertifiedFactorRecord(BaseModel):
    """L2 颁发的认证记录（不可变，契约 C2）。"""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    factor_id: str = Field(..., description="必须与 candidate.factor_id 相同")
    candidate: CandidateFactorPackage = Field(..., description="原始候选包 snapshot")
    profile_name: str = Field(
        ..., min_length=1, max_length=64, description="使用的 ValidationProfile 名"
    )
    profile_hash: str = Field(
        ...,
        description="profile YAML 的 sha256 十六进制字符串（64 位小写）",
    )
    decision: Decision = Field(..., description="PASS / FAIL / HOLD")
    overall_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="综合分（profile.aggregation 定义聚合方式）",
    )
    check_results: list[CheckResult] = Field(
        ..., min_length=1, description="至少要有一项 check"
    )
    backtest_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="若启用 backtest_check，写入 RQAlpha 真实回测核心指标",
    )
    validated_at: datetime = Field(..., description="验证完成时间（UTC）")
    validator_version: str = Field(
        ...,
        description="factor_validation 包版本（PEP440 风格 'x.y.z'，可附加 '-rc1' 等）",
    )
    notes: str | None = Field(
        default=None, max_length=4000, description="人工备注，可空"
    )

    @field_validator("profile_hash")
    @classmethod
    def _check_profile_hash(cls, v: str) -> str:
        if not _PROFILE_HASH_PATTERN.match(v):
            raise ValueError(f"profile_hash 必须为 64 位小写十六进制 sha256: {v!r}")
        return v

    @field_validator("validator_version")
    @classmethod
    def _check_version(cls, v: str) -> str:
        if not _VERSION_PATTERN.match(v):
            raise ValueError(
                f"validator_version 必须为 'x.y.z' 或 'x.y.z-rc1' 等: {v!r}"
            )
        return v

    @field_validator("backtest_metrics")
    @classmethod
    def _check_metrics_finite(cls, v: dict[str, float]) -> dict[str, float]:
        bad = {
            k: val
            for k, val in v.items()
            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val)
        }
        if bad:
            raise ValueError(f"backtest_metrics 含非有限值: {bad}")
        return {str(k): float(val) for k, val in v.items()}

    @model_validator(mode="after")
    def _check_consistency(self) -> "CertifiedFactorRecord":
        # 1. factor_id 必须与 candidate.factor_id 一致
        if self.factor_id != self.candidate.factor_id:
            raise ValueError(
                f"factor_id ({self.factor_id!r}) 与 candidate.factor_id "
                f"({self.candidate.factor_id!r}) 不一致"
            )

        # 2. check_results 名字必须唯一
        names = [c.name for c in self.check_results]
        if len(set(names)) != len(names):
            from collections import Counter

            dups = [k for k, v in Counter(names).items() if v > 1]
            raise ValueError(f"check_results 出现重名: {dups}")

        # 3. PASS 必须所有 check.passed == True
        if self.decision == Decision.PASS:
            failed = [c.name for c in self.check_results if not c.passed]
            if failed:
                raise ValueError(
                    f"decision=PASS 但有 check 未通过: {failed}"
                )

        # 4. FAIL 必须至少一个 check.passed == False（否则应是 PASS 或 HOLD）
        if self.decision == Decision.FAIL:
            all_passed = all(c.passed for c in self.check_results)
            if all_passed:
                raise ValueError(
                    "decision=FAIL 但所有 check 都通过，应判 PASS 或 HOLD"
                )

        return self
