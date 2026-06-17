"""
``ProductionFactorRecord`` —— L3 ``factor_registry/data/manifest.json`` 中每条记录的 schema。

manifest.json 是一个对象 ``{"version": ..., "factors": [ProductionFactorRecord, ...]}``。
Manifest 的读写由 ``factor_registry.registry.FactorRegistry`` 负责，本模块仅定义 schema。

设计目标：
* **不可变**：单条记录不可变；任何变更都视为新记录或退役动作。
* **轻引用**：仅持有证书 / parquet 的"指针"，不内嵌 CertifiedFactorRecord 全文（证书副本
  以独立 JSON 文件保存于 ``data/certified/<factor_id>.json``）。
* **可定位**：``parquet_version`` + ``parquet_column`` 唯一决定 production 数据源中的因子值列。
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_FACTOR_ID_PATTERN = re.compile(r"^[a-z0-9]+_[A-Za-z0-9_]{1,96}$")
_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")


class ProductionStatus(str, Enum):
    ACTIVE = "active"
    RETIRED = "retired"


class ProductionFactorRecord(BaseModel):
    """manifest.json 中的单条因子记录（不可变）。"""

    model_config = ConfigDict(frozen=True, extra="allow", str_strip_whitespace=True)

    factor_id: str = Field(..., description="与 C1 / C2 中的 factor_id 一致")
    name: str = Field(..., description="parquet 列名 == 此 name")
    status: ProductionStatus = Field(..., description="active / retired")
    parquet_version: int = Field(
        ..., ge=1, description="生效的 production parquet 版本号 (factors_v<N>.parquet)"
    )
    parquet_column: str = Field(
        ..., description="该因子在 production parquet 中的列名（== name）"
    )
    certificate_path: Path = Field(
        ..., description="data/certified/<factor_id>.json 相对路径"
    )
    registered_at: datetime = Field(..., description="注册到 L3 的时间（UTC）")
    retired_at: datetime | None = Field(
        default=None, description="退役时间（UTC）；status=active 时必须为 None"
    )
    retire_reason: str | None = Field(
        default=None, max_length=2000, description="退役原因；status=active 时必须为 None"
    )
    tags: list[str] = Field(
        default_factory=list,
        description="标签：例如 ['quality', 'long_cycle']，便于检索与分组",
    )

    @model_validator(mode="before")
    @classmethod
    def _legacy_manifest_compat(cls, data: Any) -> Any:
        """兼容早期 manifest：RD-Agent 记录曾使用 factor_name 和额外统计字段。"""
        if not isinstance(data, dict):
            return data
        out = dict(data)
        if "name" not in out and out.get("factor_name"):
            out["name"] = out["factor_name"]
        if "parquet_column" not in out and out.get("name"):
            out["parquet_column"] = out["name"]
        if "certificate_path" not in out and out.get("factor_id"):
            out["certificate_path"] = f"certified/{out['factor_id']}.json"
        return out

    @field_validator("factor_id")
    @classmethod
    def _check_factor_id(cls, v: str) -> str:
        if not _FACTOR_ID_PATTERN.match(v):
            raise ValueError(f"factor_id 不符合命名规则: {v!r}")
        return v

    @field_validator("name", "parquet_column")
    @classmethod
    def _check_name(cls, v: str) -> str:
        if not _NAME_PATTERN.match(v):
            raise ValueError(f"必须以字母开头、仅含字母数字下划线: {v!r}")
        return v

    @field_validator("certificate_path")
    @classmethod
    def _coerce_path(cls, v: Any) -> Path:
        return Path(v)

    @field_validator("tags")
    @classmethod
    def _check_tags(cls, v: list[str]) -> list[str]:
        for t in v:
            if not isinstance(t, str) or not t.strip():
                raise ValueError(f"tags 元素必须为非空字符串: {v!r}")
            if len(t) > 32:
                raise ValueError(f"tag 长度 <= 32: {t!r}")
        return [t.strip() for t in v]

    def model_post_init(self, __context: Any) -> None:
        # active 因子不能有退役信息；retired 因子必须有
        if self.status == ProductionStatus.ACTIVE:
            if self.retired_at is not None or self.retire_reason is not None:
                raise ValueError(
                    "status=active 时 retired_at 与 retire_reason 必须为 None"
                )
        else:  # RETIRED
            if self.retired_at is None:
                raise ValueError(
                    "status=retired 时 retired_at 必须给出"
                )

        # parquet_column 必须等于 name
        if self.parquet_column != self.name:
            raise ValueError(
                f"parquet_column ({self.parquet_column!r}) 必须等于 name ({self.name!r})"
            )
