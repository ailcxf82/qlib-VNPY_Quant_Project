"""阶段 F.3：RD-Agent RAG 静态宪法 YAML 加载 + 渲染 + 兜底 + 挂钩单测。"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from factor_lab.config.constitution import (
    _FALLBACK_CONSTITUTION_TEXT,
    get_constitution_text,
    load_constitution_text,
    reload_constitution_text,
    render_constitution,
)


# ---------------------------------------------------------- byte-level parity


def test_default_yaml_renders_byte_equal_to_fallback() -> None:
    """核心不变式：工程自带 YAML 渲染出的文本必须与 Python 兜底 byte-for-byte 一致。"""
    text = load_constitution_text()
    assert text == _FALLBACK_CONSTITUTION_TEXT


def test_hardcoded_static_constitution_equal_to_fallback() -> None:
    """
    factor_lab.adapters.quant_proposal._STATIC_CONSTITUTION 在正常环境里（YAML
    可读 + pyyaml 可用）必须等于兜底文本，保证 E.3 snapshot 测试继续绿。
    """
    from factor_lab.adapters.quant_proposal import _STATIC_CONSTITUTION

    assert _STATIC_CONSTITUTION == _FALLBACK_CONSTITUTION_TEXT


# ---------------------------------------------------------- YAML → fallback


def _write_yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "rag_constitution.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_missing_yaml_returns_fallback(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    missing = tmp_path / "nope.yaml"
    with caplog.at_level(logging.WARNING):
        text = load_constitution_text(missing)
    assert text == _FALLBACK_CONSTITUTION_TEXT
    assert any("使用兜底文本" in r.message for r in caplog.records)


def test_broken_yaml_returns_fallback(tmp_path: Path) -> None:
    bad = _write_yaml(tmp_path, "::: not yaml :::")
    assert load_constitution_text(bad) == _FALLBACK_CONSTITUTION_TEXT


def test_wrong_schema_version_returns_fallback(tmp_path: Path) -> None:
    bad = _write_yaml(
        tmp_path,
        'version: "9.9"\nallowed_windows: [5]\n',
    )
    assert load_constitution_text(bad) == _FALLBACK_CONSTITUTION_TEXT


def test_non_mapping_root_returns_fallback(tmp_path: Path) -> None:
    bad = _write_yaml(tmp_path, "- just a list\n")
    assert load_constitution_text(bad) == _FALLBACK_CONSTITUTION_TEXT


def test_missing_required_field_returns_fallback(tmp_path: Path) -> None:
    bad = _write_yaml(tmp_path, 'version: "1.0"\n')  # 缺所有必填项
    assert load_constitution_text(bad) == _FALLBACK_CONSTITUTION_TEXT


# ---------------------------------------------------------- renderer direct


def test_renderer_rejects_bad_type() -> None:
    with pytest.raises(ValueError):
        render_constitution({"version": "1.0"})  # 缺 allowed_windows 等


def test_renderer_rejects_wrong_schema() -> None:
    with pytest.raises(ValueError):
        render_constitution({"version": "2.0"})


def test_renderer_rejects_bad_family_entry() -> None:
    import yaml

    cfg = yaml.safe_load(open("factor_lab/config/rag_constitution.yaml", encoding="utf-8"))
    cfg["encouraged_families"] = ["not-a-mapping"]
    with pytest.raises(ValueError):
        render_constitution(cfg)


# ---------------------------------------------------------- cache / reload


def test_get_then_reload_cycle(tmp_path: Path) -> None:
    first = get_constitution_text()
    custom_yaml = _write_yaml(
        tmp_path,
        'version: "1.0"\n# minimal invalid structure\nallowed_windows: [1]\n',
    )
    reloaded = reload_constitution_text(custom_yaml)
    try:
        # 上面 YAML 是最小集，一定缺字段 → 回退到 fallback
        assert reloaded == _FALLBACK_CONSTITUTION_TEXT
    finally:
        # 恢复默认缓存，避免污染后续 test
        reload_constitution_text()

    assert get_constitution_text() == first


# --------------------------------------------------------- content invariants


@pytest.mark.parametrize(
    "marker",
    [
        "composite_score",
        "QualPersist_60D",
        "short-cycle volume-price-reversal factors",
        "|Spearman(new_factor, X)| <= 0.50",
        "cross-sectional rank auto-correlation",
        # F.3 新增 —— 确认 YAML 里的 encouraged/discouraged 条目渲染到了文本
        "(a) Quality persistence",
        "(x) Short-cycle volume-price reversals",
    ],
)
def test_rendered_text_contains_required_markers(marker: str) -> None:
    text = load_constitution_text()
    assert marker in text, f"静态宪法缺关键词: {marker!r}"


# ---------------------------------------------------------- integration: RAG


def test_compose_project_rag_uses_yaml_constitution(tmp_path: Path) -> None:
    """确保 E.3 的 compose_project_rag 现在消费的是 YAML 渲染结果。"""
    from factor_lab.adapters.quant_proposal import compose_project_rag

    rag = compose_project_rag(base_rag="BASE", feedback_dir=tmp_path / "absent")
    assert "BASE" in rag
    assert "(a) Quality persistence" in rag
    assert "(x) Short-cycle volume-price reversals" in rag
