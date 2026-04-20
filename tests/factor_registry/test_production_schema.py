"""ProductionFactorRecord schema 单元测试。

覆盖：
* active 因子构造 + frozen
* retired 因子要求 retired_at + retire_reason
* active 因子禁止 retired_at / retire_reason
* parquet_column 必须等于 name
* factor_id / name 命名规则
* tags 校验
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from factor_registry import ProductionFactorRecord, ProductionStatus


def _make_active(**overrides) -> ProductionFactorRecord:
    base = dict(
        factor_id="rdagent_QualPersist_60D_a1b2c3d4",
        name="QualPersist_60D",
        status=ProductionStatus.ACTIVE,
        parquet_version=1,
        parquet_column="QualPersist_60D",
        certificate_path=Path("data/certified/rdagent_QualPersist_60D_a1b2c3d4.json"),
        registered_at=datetime(2026, 4, 19, 14, tzinfo=timezone.utc),
        tags=["quality", "long_cycle"],
    )
    base.update(overrides)
    return ProductionFactorRecord(**base)


class TestActive:
    def test_minimal_ok(self):
        r = _make_active()
        assert r.status == ProductionStatus.ACTIVE
        assert r.retired_at is None and r.retire_reason is None

    def test_serialization_roundtrip(self):
        r = _make_active()
        data = r.model_dump(mode="json")
        restored = ProductionFactorRecord.model_validate(data)
        assert restored == r

    def test_active_must_not_have_retired_at(self):
        with pytest.raises(ValidationError, match="active"):
            _make_active(retired_at=datetime.now(timezone.utc))

    def test_active_must_not_have_retire_reason(self):
        with pytest.raises(ValidationError, match="active"):
            _make_active(retire_reason="oops")

    def test_frozen(self):
        r = _make_active()
        with pytest.raises(ValidationError):
            r.status = ProductionStatus.RETIRED


class TestRetired:
    def test_retired_minimal_ok(self):
        r = _make_active(
            status=ProductionStatus.RETIRED,
            retired_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
            retire_reason="Marginal IC dropped below 0.005 over 90 days.",
        )
        assert r.status == ProductionStatus.RETIRED

    def test_retired_requires_retired_at(self):
        with pytest.raises(ValidationError, match="retired"):
            _make_active(
                status=ProductionStatus.RETIRED,
                retire_reason="something",
            )

    def test_retired_requires_retire_reason(self):
        with pytest.raises(ValidationError, match="retired"):
            _make_active(
                status=ProductionStatus.RETIRED,
                retired_at=datetime.now(timezone.utc),
            )


class TestNameConsistency:
    def test_parquet_column_must_equal_name(self):
        with pytest.raises(ValidationError, match="parquet_column"):
            _make_active(parquet_column="OtherName")

    def test_invalid_factor_id(self):
        with pytest.raises(ValidationError):
            _make_active(factor_id="BadID")

    def test_invalid_name(self):
        # 数字开头不行；name 与 parquet_column 一起改
        with pytest.raises(ValidationError):
            _make_active(name="1Bad", parquet_column="1Bad")


class TestTags:
    def test_empty_tag_rejected(self):
        with pytest.raises(ValidationError):
            _make_active(tags=["ok", "  "])

    def test_too_long_tag_rejected(self):
        with pytest.raises(ValidationError):
            _make_active(tags=["x" * 33])

    def test_non_string_rejected(self):
        with pytest.raises(ValidationError):
            _make_active(tags=[123])  # type: ignore[list-item]

    def test_tag_trimmed(self):
        r = _make_active(tags=["  quality  ", " long_cycle "])
        assert r.tags == ["quality", "long_cycle"]


class TestFrozenAndExtraForbid:
    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            _make_active(unknown_field="x")
