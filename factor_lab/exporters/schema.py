"""
契约 C1：``CandidateFactorPackage`` —— L1 → L2 之间的唯一数据交换格式。

设计目标：

* **不可变**：``model_config = ConfigDict(frozen=True)``，任何下游模块都不能修改候选包。
* **轻量自描述**：仅引用本地物料路径（``code_path``、``values_path``），不在 schema 内
  反序列化重型数据（parquet 内容）。
* **可独立校验**：物料校验通过 ``CandidateFactorPackage.validate_artifacts()`` 显式调用，
  不在 ``__init__`` 自动执行（避免 schema 实例化触发文件 IO）。
* **可审计**：``factor_id`` + ``lab_run_id`` + ``created_at`` 三元组定位任一候选。

详见 ``docs/CONTRACT_C1_CANDIDATE.md``。
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# factor_id 命名规则：<source>_<name>_<short_hash>
# - source: 仅小写字母数字
# - name:   字母数字 + 下划线，长度 1~64
# - short_hash: 8~16 位十六进制
_FACTOR_ID_PATTERN = re.compile(r"^[a-z0-9]+_[A-Za-z0-9_]{1,64}_[0-9a-f]{8,16}$")

# parquet 内单列因子值列名要求：与 ``name`` 相等
_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")


class CandidateFactorPackage(BaseModel):
    """L1 → L2 候选因子包（契约 C1，不可变）。"""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    factor_id: str = Field(
        ...,
        description="全局唯一 ID，格式 '<source>_<name>_<short_hash>'，例如 'rdagent_QualPersist_60D_a1b2c3d4'",
    )
    name: str = Field(
        ...,
        description="人类可读名，必须与 values.parquet 单列列名一致；例如 'QualPersist_60D'",
    )
    source: Literal["rdagent", "manual", "external"] = Field(
        ..., description="候选来源：rdagent 自动 / manual 手工 / external 外部导入"
    )
    hypothesis: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="LLM 提的假设原文 / 人手填的设计动机；不能为空",
    )
    formulation: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="数学公式 / pseudocode；不能为空",
    )
    code_path: Path = Field(
        ...,
        description="factor.py 路径（绝对路径或可解析为绝对路径的相对路径）",
    )
    values_path: Path = Field(
        ...,
        description="values.parquet 路径，含 (datetime, instrument) MultiIndex + 单列 float64",
    )
    universe: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="因子值适用的股票池，例如 'csi300', 'csi500', 'all'",
    )
    date_range: tuple[date, date] = Field(
        ...,
        description="因子值覆盖的时间窗 (start_date, end_date)，闭区间",
    )
    lab_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="L1 自报指标（IC/IC_IR 等）；仅参考，L2 会重新计算",
    )
    parent_loop: int | None = Field(
        default=None,
        ge=0,
        description="RD-Agent loop index；source='manual' 时应为 None",
    )
    created_at: datetime = Field(..., description="候选包创建时间（UTC）")
    lab_run_id: str = Field(
        ...,
        min_length=8,
        max_length=64,
        description="一次 lab run 的 UUID / 短串，便于审计",
    )

    @field_validator("factor_id")
    @classmethod
    def _check_factor_id(cls, v: str) -> str:
        if not _FACTOR_ID_PATTERN.match(v):
            raise ValueError(
                f"factor_id 不符合 '<source>_<name>_<short_hash>' 格式: {v!r}"
            )
        return v

    @field_validator("name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        if not _NAME_PATTERN.match(v):
            raise ValueError(
                f"name 必须以字母开头、仅含字母数字下划线、长度 1~64: {v!r}"
            )
        return v

    @field_validator("code_path", "values_path")
    @classmethod
    def _coerce_path(cls, v: Any) -> Path:
        return Path(v)

    @field_validator("lab_metrics")
    @classmethod
    def _check_metric_finite(cls, v: dict[str, float]) -> dict[str, float]:
        import math

        bad = {
            k: val
            for k, val in v.items()
            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val)
        }
        if bad:
            raise ValueError(f"lab_metrics 含非有限值: {bad}")
        return {str(k): float(val) for k, val in v.items()}

    @model_validator(mode="after")
    def _check_consistency(self) -> "CandidateFactorPackage":
        # 1. factor_id 中的 name 段必须与 name 字段一致
        try:
            _, id_name, _ = self.factor_id.split("_", 2)
            id_name = id_name.rsplit("_", 1)[0]  # 去掉末尾 short_hash 段
        except ValueError:
            raise ValueError(f"无法从 factor_id 解析 name 段: {self.factor_id!r}")
        # 注意：factor_id 拆分后中间段可能含下划线，规则为：第一个 _ 后到最后一个 _ 前
        parts = self.factor_id.split("_")
        if len(parts) < 3:
            raise ValueError(f"factor_id 至少包含三段: {self.factor_id!r}")
        id_name = "_".join(parts[1:-1])
        if id_name != self.name:
            raise ValueError(
                f"factor_id 中的 name 段 ({id_name!r}) 与 name 字段 ({self.name!r}) 不一致"
            )

        # 2. date_range 必须 start <= end
        start, end = self.date_range
        if start > end:
            raise ValueError(f"date_range start ({start}) 晚于 end ({end})")

        # 3. source='manual' 时 parent_loop 必须为 None
        if self.source == "manual" and self.parent_loop is not None:
            raise ValueError("source='manual' 时 parent_loop 必须为 None")

        # 4. source='rdagent' 时 parent_loop 必须给出
        if self.source == "rdagent" and self.parent_loop is None:
            raise ValueError("source='rdagent' 时 parent_loop 不能为 None")

        return self

    # ---------------------------------------------------------- helpers

    def validate_artifacts(self, *, check_parquet_schema: bool = True) -> None:
        """
        显式物料校验（不在 __init__ 触发，避免 schema 实例化时做文件 IO）。

        校验项：
        - ``code_path`` 存在且是 .py 文件
        - ``values_path`` 存在且是 .parquet 文件
        - 若 ``check_parquet_schema=True``，进一步校验 parquet：
          * 含两级 MultiIndex (datetime, instrument)
          * 仅一列且列名 == self.name
          * dtype 为 float64
          * 时间索引落在 self.date_range 内

        校验失败会抛 ValueError；不返回任何东西。
        """
        if not self.code_path.exists():
            raise ValueError(f"code_path 不存在: {self.code_path}")
        if self.code_path.suffix != ".py":
            raise ValueError(f"code_path 必须是 .py: {self.code_path}")
        if not self.values_path.exists():
            raise ValueError(f"values_path 不存在: {self.values_path}")
        if self.values_path.suffix != ".parquet":
            raise ValueError(f"values_path 必须是 .parquet: {self.values_path}")

        if not check_parquet_schema:
            return

        # 仅在需要时 import pandas，避免 schema 模块加载即拉重依赖
        import pandas as pd  # noqa: WPS433

        try:
            df = pd.read_parquet(self.values_path)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"读取 values.parquet 失败: {exc}") from exc

        if not isinstance(df.index, pd.MultiIndex) or df.index.nlevels != 2:
            raise ValueError(
                f"values.parquet 必须有两级 MultiIndex，当前: {df.index!r}"
            )
        levels = list(df.index.names)
        if levels != ["datetime", "instrument"]:
            raise ValueError(
                f"values.parquet MultiIndex 名必须为 ['datetime','instrument']，当前: {levels}"
            )
        if list(df.columns) != [self.name]:
            raise ValueError(
                f"values.parquet 必须仅含一列且列名为 {self.name!r}，当前: {list(df.columns)}"
            )
        if str(df[self.name].dtype) != "float64":
            raise ValueError(
                f"values.parquet 列 dtype 必须为 float64，当前: {df[self.name].dtype}"
            )

        # 时间窗校验：parquet 内最早 / 最晚日期必须 ⊆ self.date_range
        dt_index = df.index.get_level_values("datetime")
        if len(dt_index) == 0:
            raise ValueError("values.parquet 为空")
        actual_start = pd.Timestamp(dt_index.min()).date()
        actual_end = pd.Timestamp(dt_index.max()).date()
        decl_start, decl_end = self.date_range
        if actual_start < decl_start or actual_end > decl_end:
            raise ValueError(
                f"values.parquet 实际时间窗 [{actual_start}, {actual_end}] "
                f"超出声明 date_range [{decl_start}, {decl_end}]"
            )
