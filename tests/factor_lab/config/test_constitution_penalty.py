"""阶段 G.4：discouraged_families 可选 ``penalty`` 字段 + 软降权渲染。

契约：

1. 缺省 ``penalty`` / ``penalty=.inf`` → 保持 F 阶段硬黑语义，渲染**不**追加任何文字。
   （这一条通过 test_default_yaml_renders_byte_equal_to_fallback 间接锁死，
   本文件再叠加直接断言。）
2. ``penalty=-0.5`` 等具体数值 → 行末追加 ``[penalty=-0.5; soft — new attempts
   allowed only with explicit justification ...]`` 注解。
3. ``penalty`` 非数值 / NaN → ``render_constitution`` 抛 ValueError，load 层回退到 fallback。
4. encouraged_families 不受影响。
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest
import yaml

from factor_lab.config.constitution import (
    _FALLBACK_CONSTITUTION_TEXT,
    load_constitution_text,
    render_constitution,
)


def _load_default_cfg() -> dict:
    path = Path(__file__).resolve().parents[3] / "factor_lab" / "config" / "rag_constitution.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_default_cfg_has_no_penalty_for_any_discouraged() -> None:
    """默认配置里三条 discouraged 必须都没有 penalty 字段，才能满足 byte-for-byte 契约。"""
    cfg = _load_default_cfg()
    for item in cfg["discouraged_families"]:
        assert "penalty" not in item, f"默认配置意外带了 penalty: {item}"


def test_penalty_inf_equals_no_penalty_in_rendering() -> None:
    """显式写 penalty=.inf 应该和不写 penalty 渲染一致（硬黑名单语义）。"""
    cfg = _load_default_cfg()
    cfg_with_inf = copy.deepcopy(cfg)
    for item in cfg_with_inf["discouraged_families"]:
        item["penalty"] = math.inf
    assert render_constitution(cfg) == render_constitution(cfg_with_inf)


def test_finite_penalty_appends_soft_downweight_annotation() -> None:
    cfg = _load_default_cfg()
    cfg["discouraged_families"][0]["penalty"] = -0.5
    out = render_constitution(cfg)
    # 未改动的条目仍无注解
    assert "(y) Same-day volume spike + price reversal patterns\n" in out
    # 改动的条目出现软降权注解
    assert "penalty=-0.5" in out
    assert "soft" in out
    assert "explicit justification" in out
    # 不破坏 encouraged 段
    assert "Encouraged factor families" in out


@pytest.mark.parametrize("pen", ["not-a-number", None, [], {}])
def test_invalid_penalty_type_raises(pen) -> None:
    if pen is None:
        # None 等同于没设置，应该通过
        cfg = _load_default_cfg()
        cfg["discouraged_families"][0]["penalty"] = pen
        # 不应抛
        render_constitution(cfg)
        return
    cfg = _load_default_cfg()
    cfg["discouraged_families"][0]["penalty"] = pen
    with pytest.raises(ValueError):
        render_constitution(cfg)


def test_nan_penalty_rejected() -> None:
    cfg = _load_default_cfg()
    cfg["discouraged_families"][0]["penalty"] = float("nan")
    with pytest.raises(ValueError):
        render_constitution(cfg)


def test_load_constitution_text_falls_back_when_penalty_bad(tmp_path: Path) -> None:
    """非法 penalty 值应让 load 层整体回退到 fallback，不阻塞上线。"""
    cfg = _load_default_cfg()
    cfg["discouraged_families"][0]["penalty"] = "NaN-string"
    bad_path = tmp_path / "rag_constitution.yaml"
    bad_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    text = load_constitution_text(bad_path)
    assert text == _FALLBACK_CONSTITUTION_TEXT


def test_multiple_penalties_mixed_hard_and_soft() -> None:
    cfg = _load_default_cfg()
    cfg["discouraged_families"][0]["penalty"] = -1.0   # soft
    cfg["discouraged_families"][1]["penalty"] = math.inf  # hard
    # 第三条不带 penalty → 硬黑
    out = render_constitution(cfg)
    # 只有第一条带注解
    assert out.count("penalty=-1") == 1
    assert "penalty=inf" not in out
    # 第二条和第三条原样
    assert "(y) Same-day volume spike + price reversal patterns\n" in out
    assert "(z) Anything that ranks the universe with >50% weekly turnover" in out
