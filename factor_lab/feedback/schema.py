"""
契约 C3：``FeedbackBundle`` —— L2 验证历史 → L1 RD-Agent 反馈的唯一数据交换格式。

设计目标：

* **不可变**：``model_config = ConfigDict(frozen=True)``。
* **可序列化**：整份 bundle 可落盘为单个 JSON 文件，同时渲染成给 LLM 看的 markdown。
* **可审计**：``cycles_included`` + ``generated_at`` + ``schema_version`` 三元组定位。
* **自包含**：不引用磁盘路径（code/values 等物料），所有信息都已被聚合器提炼成结构化
  字段，RAG 注入器不需要再读别的文件。

C3 不包含因子的完整 C1/C2 数据；它是"跨 cycle 的判决摘要"。全量审计请读
``factor_validation/reports/lab_cycle_*.json`` 与 ``factor_validation/certificates/``。
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_FACTOR_ID_PATTERN = re.compile(r"^[a-z0-9]+_[A-Za-z0-9_]{1,64}_[0-9a-f]{8,16}$")
# 阶段 F.2：universe 标识，与 CandidateFactorPackage.universe 保持同样的长度/字符约束
_UNIVERSE_PATTERN = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def _validate_universe(v: str | None) -> str | None:
    """universe 允许 None（老 bundle 或未知），非 None 时必须匹配 _UNIVERSE_PATTERN。"""
    if v is None:
        return None
    if not isinstance(v, str) or not _UNIVERSE_PATTERN.match(v):
        raise ValueError(f"universe 非法: {v!r}；允许字母数字/_/-，长度 1~64")
    return v

# 允许的失败模式标签（来自 L2 check 名，与 factor_validation/checks/ 一一对应）
_KNOWN_FAILURE_MODES = frozenset(
    {
        "coverage",
        "ic",
        "orthogonality",
        "turnover",
        "backtest",
        "backtest_rqalpha",
        "marginal",
        "marginal_training",
    }
)


class ActiveFactorSummary(BaseModel):
    """L3 当前 active 因子的简要信息（给 RAG 提示『已有的不要重复』）。"""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    factor_id: str = Field(..., min_length=1, max_length=128)
    name: str = Field(..., description="人类可读名，等同 C1.name")
    family: str = Field(
        default="unknown",
        max_length=64,
        description="家族标签，例如 'volume_price_reversal' / 'quality_persist'；未知则 'unknown'",
    )
    universe: str | None = Field(
        default=None,
        description="F.2：因子所属股票池（csi300 / csi500 / all 等）。老 bundle 可能缺此字段",
    )
    parquet_version: int = Field(..., ge=0)
    tags: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("factor_id")
    @classmethod
    def _check_factor_id(cls, v: str) -> str:
        if not _FACTOR_ID_PATTERN.match(v):
            raise ValueError(f"factor_id 格式非法: {v!r}")
        return v

    @field_validator("name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        if not _NAME_PATTERN.match(v):
            raise ValueError(f"name 必须以字母开头，仅字母数字下划线，1~64 长度: {v!r}")
        return v

    @field_validator("universe")
    @classmethod
    def _check_universe(cls, v: str | None) -> str | None:
        return _validate_universe(v)


class RetiredFactorSummary(BaseModel):
    """L3 已退役因子的简要信息（给 RAG 提示『曾经试过、被证伪的』）。"""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    factor_id: str = Field(..., min_length=1, max_length=128)
    name: str = Field(..., description="人类可读名")
    family: str = Field(default="unknown", max_length=64)
    universe: str | None = Field(
        default=None,
        description="F.2：退役因子所属股票池；老 bundle 可能缺此字段",
    )
    retired_at: datetime | None = Field(default=None)
    reason: str = Field(
        ..., min_length=1, max_length=1000, description="退役原因（manifest 里的 reason 字段）"
    )

    @field_validator("factor_id")
    @classmethod
    def _check_factor_id(cls, v: str) -> str:
        if not _FACTOR_ID_PATTERN.match(v):
            raise ValueError(f"factor_id 格式非法: {v!r}")
        return v

    @field_validator("name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        if not _NAME_PATTERN.match(v):
            raise ValueError(f"name 必须以字母开头，仅字母数字下划线，1~64 长度: {v!r}")
        return v

    @field_validator("universe")
    @classmethod
    def _check_universe(cls, v: str | None) -> str | None:
        return _validate_universe(v)


class FailedCandidateSummary(BaseModel):
    """一次 cycle 里某个候选因子没通过 L2 的简要信息。

    **不**包含完整 C2 证书，只提炼以下 4 项给 RAG：

    * 因子名 / 家族标签；
    * cycle_id + stage（exploratory / default）；
    * 失败的 check 列表（``failure_modes``）；
    * 关键指标摘要（``metrics``，如 ``rank_ic=0.007``、``max_abs_corr=0.72``）。
    """

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    name: str = Field(..., description="因子名（C1.name，人类可读）")
    family: str = Field(default="unknown", max_length=64)
    universe: str | None = Field(
        default=None,
        description="F.2：候选因子所属股票池（csi300 / csi500 / all 等）；来自 C2 的 candidate.universe",
    )
    cycle_id: str = Field(..., min_length=1, max_length=128)
    stage: str = Field(
        ...,
        description="exploratory / default / strict；标记候选在哪一档被卡住",
    )
    decision: str = Field(
        ..., description="决议字符串：FAIL / HOLD（PASS 的候选不会被收进本列表）"
    )
    failure_modes: tuple[str, ...] = Field(
        default_factory=tuple,
        description="失败的 check 名列表；来自 CheckResult.name（只收 passed=False 的）",
    )
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="关键指标摘要，例如 {'rank_ic': 0.007, 'ic_ir': 0.08, 'max_abs_corr': 0.72}",
    )

    @field_validator("name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        if not _NAME_PATTERN.match(v):
            raise ValueError(f"name 必须以字母开头，仅字母数字下划线，1~64 长度: {v!r}")
        return v

    @field_validator("universe")
    @classmethod
    def _check_universe(cls, v: str | None) -> str | None:
        return _validate_universe(v)

    @field_validator("stage")
    @classmethod
    def _check_stage(cls, v: str) -> str:
        allowed = {"exploratory", "default", "strict"}
        if v not in allowed:
            raise ValueError(f"stage 必须属于 {sorted(allowed)}，当前: {v!r}")
        return v

    @field_validator("decision")
    @classmethod
    def _check_decision(cls, v: str) -> str:
        allowed = {"FAIL", "HOLD"}
        if v not in allowed:
            raise ValueError(
                f"FailedCandidateSummary.decision 只收 FAIL/HOLD，不收 PASS: {v!r}"
            )
        return v

    @field_validator("failure_modes")
    @classmethod
    def _check_failure_modes(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        clean: list[str] = []
        for m in v:
            if not isinstance(m, str) or not m:
                raise ValueError(f"failure_mode 必须是非空字符串: {m!r}")
            if m not in _KNOWN_FAILURE_MODES:
                raise ValueError(
                    f"failure_mode 必须属于已知 check 名 {sorted(_KNOWN_FAILURE_MODES)}，当前: {m!r}"
                )
            clean.append(m)
        # 去重但保序
        seen: set[str] = set()
        unique: list[str] = []
        for m in clean:
            if m not in seen:
                seen.add(m)
                unique.append(m)
        return tuple(unique)

    @field_validator("metrics")
    @classmethod
    def _check_metrics(cls, v: dict[str, Any]) -> dict[str, float]:
        bad = {
            k: val
            for k, val in v.items()
            if not isinstance(val, (int, float))
            or isinstance(val, bool)
            or math.isnan(val)
            or math.isinf(val)
        }
        if bad:
            raise ValueError(f"metrics 含非有限数值: {bad}")
        return {str(k): float(val) for k, val in v.items()}


class UniverseSubBundle(BaseModel):
    """阶段 G.1：单个 universe 的 C3 子视图（附加在 ``FeedbackBundle.by_universe`` 里）。

    * ``universe`` —— 本桶对应的股票池标识（csi300 / csi500 / ...）。
    * ``active_factors`` / ``retired_factors`` / ``recent_fails`` —— 仅包含该
      universe 下的条目，引用父 bundle 里同样的 Pydantic 模型（对象不共享，
      深拷贝，保持 frozen 语义）。
    * ``failure_family_counts`` / ``discouraged_families`` —— 只统计本 universe
      范围内的失败分布与黑名单；让 LLM 可以单独观察一个 universe 的局部趋势。

    设计原则（与 aggregator 对齐）：

    * 未填 universe 的条目**不会**进入任何 sub-bundle（保留在父 bundle 的
      平铺字段里，避免 "__unknown__" 这类特殊 key 污染 prompt）。
    * ``by_universe`` 不是新增事实，而是"分桶视图"：所有内容在父 bundle 的
      平铺字段里必然也能找到。
    """

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    universe: str = Field(..., min_length=1, max_length=64)
    active_factors: tuple[ActiveFactorSummary, ...] = Field(default_factory=tuple)
    retired_factors: tuple[RetiredFactorSummary, ...] = Field(default_factory=tuple)
    recent_fails: tuple[FailedCandidateSummary, ...] = Field(default_factory=tuple)
    failure_family_counts: dict[str, int] = Field(default_factory=dict)
    discouraged_families: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("universe")
    @classmethod
    def _check_universe(cls, v: str) -> str:
        if not _UNIVERSE_PATTERN.match(v):
            raise ValueError(f"universe 非法: {v!r}；允许字母数字/_/-，长度 1~64")
        return v

    @field_validator("failure_family_counts")
    @classmethod
    def _check_family_counts(cls, v: dict[str, int]) -> dict[str, int]:
        bad = {k: val for k, val in v.items() if not isinstance(val, int) or val < 0}
        if bad:
            raise ValueError(f"failure_family_counts 必须是非负整数: {bad}")
        return {str(k): int(val) for k, val in v.items()}

    @field_validator("discouraged_families")
    @classmethod
    def _check_discouraged_unique(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(v)) != len(v):
            raise ValueError(f"discouraged_families 含重复: {v}")
        return v

    @model_validator(mode="after")
    def _check_inner_universe_consistency(self) -> "UniverseSubBundle":
        for f in self.active_factors:
            if f.universe is not None and f.universe != self.universe:
                raise ValueError(
                    f"UniverseSubBundle[{self.universe}] 下 active_factors 含不同 universe: "
                    f"{f.name} universe={f.universe!r}"
                )
        for f in self.retired_factors:
            if f.universe is not None and f.universe != self.universe:
                raise ValueError(
                    f"UniverseSubBundle[{self.universe}] 下 retired_factors 含不同 universe: "
                    f"{f.name} universe={f.universe!r}"
                )
        for f in self.recent_fails:
            if f.universe is not None and f.universe != self.universe:
                raise ValueError(
                    f"UniverseSubBundle[{self.universe}] 下 recent_fails 含不同 universe: "
                    f"{f.name} universe={f.universe!r}"
                )
        return self


class FeedbackBundle(BaseModel):
    """
    契约 C3：聚合 L2 最近 N 轮 cycle 的判决信息，给 RD-Agent RAG 注入用。

    由 ``factor_lab.feedback.aggregator.build_feedback_bundle(...)`` 产出；
    被 ``factor_lab.adapters.quant_proposal.ProjectQlibQuantHypothesisGen``
    读取后转成 markdown 注入 RAG。
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    schema_version: str = Field(
        default="1.0",
        description="C3 schema 版本；不兼容升级必须 bump major",
    )
    generated_at: datetime = Field(
        ..., description="本 bundle 生成时间（UTC）；每次聚合器重跑都会刷新"
    )
    cycles_included: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "本次聚合纳入的 cycle_id 列表，时间升序；例如 "
            "['labcycle-20260401-weekly', 'labcycle-20260408-weekly', ...]"
        ),
    )
    window_max_cycles: int = Field(
        ..., ge=1, description="聚合器本次使用的 max_cycles 参数（默认 8）"
    )
    active_factors: tuple[ActiveFactorSummary, ...] = Field(default_factory=tuple)
    retired_factors: tuple[RetiredFactorSummary, ...] = Field(default_factory=tuple)
    recent_fails: tuple[FailedCandidateSummary, ...] = Field(default_factory=tuple)
    failure_family_counts: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "按 family 标签计数的失败分布，例如 "
            "{'volume_price_reversal': 3, 'unknown': 1}。由聚合器根据 recent_fails + "
            "活跃/退役因子的 family 推导"
        ),
    )
    discouraged_families: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "从数据推导的『应避免』家族列表：最近 N cycle 里 fails 计数超阈值 OR "
            "退役因子家族。用于注入 RAG 的 discouraged 段"
        ),
    )
    notes: str | None = Field(
        default=None,
        max_length=2000,
        description="聚合器可填的人工注记；LLM 不一定看得到",
    )
    by_universe: dict[str, UniverseSubBundle] = Field(
        default_factory=dict,
        description=(
            "阶段 G.1：按 universe 分桶的子视图。key 必须等于 value.universe；"
            "老 bundle JSON（没有本字段）仍可通过 model_validate_json 加载（默认空 dict）。"
            "bypass 入口：若不需要 LLM 看到分桶视图，聚合器可传空 dict 强制关闭。"
        ),
    )

    @field_validator("cycles_included")
    @classmethod
    def _check_cycles_unique(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(v)) != len(v):
            raise ValueError(f"cycles_included 含重复: {v}")
        return v

    @field_validator("failure_family_counts")
    @classmethod
    def _check_family_counts(cls, v: dict[str, int]) -> dict[str, int]:
        bad = {k: val for k, val in v.items() if not isinstance(val, int) or val < 0}
        if bad:
            raise ValueError(f"failure_family_counts 必须是非负整数: {bad}")
        return {str(k): int(val) for k, val in v.items()}

    @field_validator("discouraged_families")
    @classmethod
    def _check_discouraged_unique(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(v)) != len(v):
            raise ValueError(f"discouraged_families 含重复: {v}")
        return v

    @model_validator(mode="after")
    def _check_consistency(self) -> "FeedbackBundle":
        if len(self.cycles_included) > self.window_max_cycles:
            raise ValueError(
                f"cycles_included 长度 {len(self.cycles_included)} 超过 window_max_cycles "
                f"{self.window_max_cycles}"
            )
        # recent_fails 的 cycle_id 必须 ⊆ cycles_included
        known = set(self.cycles_included)
        for f in self.recent_fails:
            if f.cycle_id not in known:
                raise ValueError(
                    f"recent_fails 里的 cycle_id {f.cycle_id!r} 不在 cycles_included 里"
                )
        for key, sub in self.by_universe.items():
            if key != sub.universe:
                raise ValueError(
                    f"by_universe[{key!r}].universe={sub.universe!r}：键值必须与子桶 universe 一致"
                )
            for ff in sub.recent_fails:
                if ff.cycle_id not in known:
                    raise ValueError(
                        f"by_universe[{key!r}] 里的 cycle_id {ff.cycle_id!r} 不在 cycles_included 里"
                    )
        return self

    def to_markdown(self) -> str:
        """
        渲染成给 LLM 看的 markdown；注入 ProjectQlibQuantHypothesisGen 的 RAG 时用。

        **输出约定**：固定章节顺序（便于 prompt-level 复现），中文标题 + 英文段落。
        字段为空时对应章节省略而不是空表头，避免给 LLM 制造"空类别 = 没有限制"的误读。
        """
        lines: list[str] = []
        lines.append("------Feedback from recent L2 cycles (dynamic)------")
        lines.append(
            f"generated_at = {self.generated_at.isoformat()}    "
            f"cycles_included = {list(self.cycles_included)}"
        )

        if self.active_factors:
            lines.append("")
            lines.append("[L3 active factors — DO NOT propose duplicates]")
            for af in self.active_factors:
                tags_str = f" tags={list(af.tags)}" if af.tags else ""
                universe_str = f" [universe={af.universe}]" if af.universe else ""
                lines.append(f"  - {af.name}  (family={af.family}){tags_str}{universe_str}")

        if self.retired_factors:
            lines.append("")
            lines.append("[L3 retired factors — these failed in production, avoid similar shape]")
            for rf in self.retired_factors:
                when = rf.retired_at.date().isoformat() if rf.retired_at else "n/a"
                universe_str = f" [universe={rf.universe}]" if rf.universe else ""
                lines.append(
                    f"  - {rf.name}  (family={rf.family})  retired_on={when}{universe_str}"
                )
                lines.append(f"    reason: {rf.reason}")

        if self.recent_fails:
            lines.append("")
            lines.append(
                f"[Recent L2 failures across last {len(self.cycles_included)} cycle(s)]"
            )
            for ff in self.recent_fails:
                modes = ",".join(ff.failure_modes) if ff.failure_modes else "n/a"
                metrics = (
                    ", ".join(f"{k}={v:.4g}" for k, v in ff.metrics.items())
                    if ff.metrics
                    else ""
                )
                tail = f"  {metrics}" if metrics else ""
                universe_str = f" [universe={ff.universe}]" if ff.universe else ""
                lines.append(
                    f"  - {ff.name}  (family={ff.family})  "
                    f"stage={ff.stage}  decision={ff.decision}  modes=[{modes}]{tail}{universe_str}"
                )

        if self.failure_family_counts:
            lines.append("")
            lines.append("[Failure family counts (last N cycles)]")
            for family, cnt in sorted(
                self.failure_family_counts.items(), key=lambda x: (-x[1], x[0])
            ):
                lines.append(f"  - {family}: {cnt}")

        if self.discouraged_families:
            lines.append("")
            lines.append(
                "[Discouraged families — empirically blocked by L2; "
                "DO NOT propose new factors in these families]"
            )
            for fam in self.discouraged_families:
                lines.append(f"  - {fam}")

        if self.by_universe:
            lines.append("")
            lines.append(
                "[Per-universe view (G.1) — same facts split by stock pool; use to spot "
                "pool-specific patterns that the global counts hide]"
            )
            for universe in sorted(self.by_universe.keys()):
                sub = self.by_universe[universe]
                lines.append(f"  * universe={universe}")
                lines.append(
                    f"    active={len(sub.active_factors)}  "
                    f"retired={len(sub.retired_factors)}  "
                    f"fails={len(sub.recent_fails)}"
                )
                if sub.failure_family_counts:
                    fam_summary = ", ".join(
                        f"{fam}:{cnt}"
                        for fam, cnt in sorted(
                            sub.failure_family_counts.items(), key=lambda x: (-x[1], x[0])
                        )
                    )
                    lines.append(f"    failure_families: {fam_summary}")
                if sub.discouraged_families:
                    lines.append(
                        f"    discouraged (local): {list(sub.discouraged_families)}"
                    )

        if self.notes:
            lines.append("")
            lines.append("[Aggregator notes]")
            lines.append(f"  {self.notes}")

        lines.append("")
        lines.append(
            "Interpretation rules for LLM: "
            "prefer factor families NOT listed under 'Discouraged'; "
            "if you must revisit a discouraged family, justify explicitly how your new "
            "formulation differs from the retired/failed ones."
        )
        return "\n".join(lines)
