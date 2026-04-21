"""阶段 F.1：家族分类 YAML 加载 + 兜底 + 分类器单测。"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

from factor_lab.config.families import (
    DEFAULT_FAMILY_RULES,
    FamilyRule,
    classify_family,
    load_family_rules,
    reload_family_rules,
)


VALID_YAML = """
version: "1.0"
rules:
  - label: custom_family_a
    pattern: "(CustomA|Alpha)"
  - label: custom_family_b
    pattern: "CustomB"
"""


def _write_yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "families.yaml"
    p.write_text(body.strip() + "\n", encoding="utf-8")
    return p


# --------------------------------------------------------------- loader happy path


def test_load_valid_yaml_returns_parsed_rules(tmp_path: Path) -> None:
    yaml_path = _write_yaml(tmp_path, VALID_YAML)
    rules = load_family_rules(yaml_path)

    assert [r.label for r in rules] == ["custom_family_a", "custom_family_b"]
    assert all(isinstance(r, FamilyRule) for r in rules)
    assert rules[0].matches("CustomA_5d")
    assert rules[0].matches("alpha_signal")  # case-insensitive
    assert rules[1].matches("CustomB")


def test_default_yaml_loads_and_matches_python_default() -> None:
    """工程自带的 factor_lab/config/factor_families.yaml 必须与 DEFAULT_FAMILY_RULES 对齐。"""
    rules = load_family_rules()  # path=None → 默认 YAML

    yaml_labels = [r.label for r in rules]
    py_labels = [r.label for r in DEFAULT_FAMILY_RULES]
    assert yaml_labels == py_labels, f"YAML vs PY 默认不一致: {yaml_labels} vs {py_labels}"


# -------------------------------------------------------------- fallback behaviour


def test_missing_yaml_falls_back_to_defaults(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    missing = tmp_path / "does_not_exist.yaml"
    with caplog.at_level(logging.WARNING):
        rules = load_family_rules(missing)
    assert rules == DEFAULT_FAMILY_RULES
    assert any("回退到内置默认值" in rec.message for rec in caplog.records)


def test_broken_yaml_falls_back(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    bad = _write_yaml(tmp_path, "version: 1.0\nrules: [this is :: not yaml")
    with caplog.at_level(logging.WARNING):
        rules = load_family_rules(bad)
    assert rules == DEFAULT_FAMILY_RULES


def test_wrong_schema_version_falls_back(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    bad = _write_yaml(tmp_path, 'version: "9.9"\nrules:\n  - label: x\n    pattern: "y"')
    with caplog.at_level(logging.WARNING):
        rules = load_family_rules(bad)
    assert rules == DEFAULT_FAMILY_RULES
    assert any("schema_version" in rec.message for rec in caplog.records)


def test_empty_rules_list_falls_back(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    bad = _write_yaml(tmp_path, 'version: "1.0"\nrules: []')
    with caplog.at_level(logging.WARNING):
        rules = load_family_rules(bad)
    assert rules == DEFAULT_FAMILY_RULES


def test_non_mapping_root_falls_back(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    bad = _write_yaml(tmp_path, "- just a list")
    with caplog.at_level(logging.WARNING):
        rules = load_family_rules(bad)
    assert rules == DEFAULT_FAMILY_RULES


def test_invalid_regex_falls_back(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    bad = _write_yaml(
        tmp_path,
        'version: "1.0"\nrules:\n  - label: bad\n    pattern: "(unbalanced"',
    )
    with caplog.at_level(logging.WARNING):
        rules = load_family_rules(bad)
    assert rules == DEFAULT_FAMILY_RULES


def test_duplicate_labels_falls_back(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    bad = _write_yaml(
        tmp_path,
        'version: "1.0"\nrules:\n  - label: dup\n    pattern: "a"\n  - label: dup\n    pattern: "b"',
    )
    with caplog.at_level(logging.WARNING):
        rules = load_family_rules(bad)
    assert rules == DEFAULT_FAMILY_RULES
    assert any("校验失败" in rec.message or "dup" in rec.message.lower() for rec in caplog.records)


def test_label_with_whitespace_falls_back(tmp_path: Path) -> None:
    bad = _write_yaml(
        tmp_path,
        'version: "1.0"\nrules:\n  - label: "has space"\n    pattern: "a"',
    )
    rules = load_family_rules(bad)
    assert rules == DEFAULT_FAMILY_RULES


def test_missing_label_field_falls_back(tmp_path: Path) -> None:
    bad = _write_yaml(tmp_path, 'version: "1.0"\nrules:\n  - pattern: "a"')
    rules = load_family_rules(bad)
    assert rules == DEFAULT_FAMILY_RULES


# ------------------------------------------------------------- classify_family


def test_classify_family_uses_yaml(tmp_path: Path) -> None:
    yaml_path = _write_yaml(tmp_path, VALID_YAML)
    rules = load_family_rules(yaml_path)
    assert classify_family("CustomA_5d", rules=rules) == "custom_family_a"
    assert classify_family("CustomB_10", rules=rules) == "custom_family_b"
    assert classify_family("RandomName_20", rules=rules) == "unknown"


def test_classify_family_order_matters() -> None:
    """前面的规则优先命中。"""
    rules = (
        FamilyRule(label="first", pattern=re.compile(r"Vol", re.IGNORECASE)),
        FamilyRule(label="second", pattern=re.compile(r"VolRev", re.IGNORECASE)),
    )
    assert classify_family("VolRev_5d", rules=rules) == "first"


def test_classify_family_case_insensitive() -> None:
    assert classify_family("volrev_5d") == "volume_price_reversal"
    assert classify_family("VOLREV_5D") == "volume_price_reversal"


def test_classify_family_default_path_matches_defaults() -> None:
    # 全默认（YAML 被缓存） + 经典名字
    assert classify_family("VolRev_5d") == "volume_price_reversal"
    assert classify_family("Quality_ROE_5d") == "quality_persist"
    assert classify_family("NoMatchHere_123") == "unknown"


# -------------------------------------------------------------------- reload


def test_reload_family_rules_overrides_cache(tmp_path: Path) -> None:
    custom = _write_yaml(tmp_path, VALID_YAML)
    try:
        rules = reload_family_rules(custom)
        assert [r.label for r in rules] == ["custom_family_a", "custom_family_b"]
        assert classify_family("CustomA_5d") == "custom_family_a"
        assert classify_family("VolRev_5d") == "unknown"  # 默认规则已替换
    finally:
        reload_family_rules()  # 恢复默认，避免污染后续 test
