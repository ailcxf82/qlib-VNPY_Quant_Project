"""阶段 H.1：retrieval_query 自动推导 + prepare_context 接线测试。

目标：
1) ``_infer_retrieval_query`` 在不同输入形态下行为稳定（ctx override / dict trace /
   object trace / fallback 到 base RAG / 全空返回 None）。
2) ``ProjectQlibQuantHypothesisGen.prepare_context`` 确实把自动推导的 query 传给
   ``compose_project_rag``，且失败时不抛（继承 F/G 的安全降级风格）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from factor_lab.adapters import quant_proposal as qp


@dataclass
class _TraceObj:
    hypothesis_text: str = ""
    summary: str = ""
    hist: list[Any] | None = None


def test_infer_query_prefers_ctx_override() -> None:
    q = qp._infer_retrieval_query(
        trace={"hypothesis": "volume reversal signal"},
        ctx={"RETRIEVAL_QUERY": "manual override query", "RAG": "BASE"},
    )
    assert q == "manual override query"


def test_infer_query_from_dict_trace() -> None:
    q = qp._infer_retrieval_query(
        trace={
            "hypothesis_text": "explore quality persistence with low turnover",
            "summary": "focus on csi300",
        },
        ctx={},
    )
    assert q is not None
    assert "quality persistence" in q


def test_infer_query_from_object_hist() -> None:
    tr = _TraceObj(
        hypothesis_text="",
        summary="",
        hist=[("old", True), {"summary": "avoid short-cycle volume reversal"}],
    )
    q = qp._infer_retrieval_query(trace=tr, ctx={})
    assert q is not None
    assert "volume reversal" in q


def test_infer_query_fallback_to_base_rag_tail() -> None:
    q = qp._infer_retrieval_query(
        trace=None,
        ctx={"RAG": "line1\nline2 final hypothesis about valuation mean reversion"},
    )
    assert q == "line2 final hypothesis about valuation mean reversion"


def test_infer_query_all_empty_returns_none() -> None:
    q = qp._infer_retrieval_query(trace=None, ctx={})
    assert q is None


def test_truncate_query_has_upper_bound() -> None:
    src = "x" * 500
    out = qp._truncate_query(src, max_chars=40)
    assert len(out) <= 40
    assert out.endswith("...")


def test_prepare_context_passes_auto_query_to_compose(monkeypatch: pytest.MonkeyPatch) -> None:
    from rdagent.scenarios.qlib.proposal.quant_proposal import QlibQuantHypothesisGen

    captured: dict[str, Any] = {}

    def _fake_super_prepare(self, trace):  # noqa: ANN001
        return {"RAG": "BASE_RAG_FROM_SUPER"}, True

    def _fake_compose(base_rag: str, **kwargs: Any) -> str:
        captured["base_rag"] = base_rag
        captured.update(kwargs)
        return "FINAL_RAG_SENTINEL"

    monkeypatch.setattr(QlibQuantHypothesisGen, "prepare_context", _fake_super_prepare)
    monkeypatch.setattr(qp, "compose_project_rag", _fake_compose)

    gen = qp.ProjectQlibQuantHypothesisGen.__new__(qp.ProjectQlibQuantHypothesisGen)
    gen._feedback_dir_override = None

    # dict trace 提供 hypothesis_text，验证会被自动提取
    ctx, ok = gen.prepare_context(trace={"hypothesis_text": "quality persistence factor"})
    assert ok is True
    assert ctx["RAG"] == "FINAL_RAG_SENTINEL"
    assert captured["base_rag"] == "BASE_RAG_FROM_SUPER"
    assert captured.get("retrieval_query") is not None
    assert "quality persistence" in captured["retrieval_query"]


def test_prepare_context_rollback_switch_disables_auto_query(monkeypatch: pytest.MonkeyPatch) -> None:
    """auto_retrieval_enabled=False 时，应该完全不传 retrieval_query（退回 G 行为）。"""
    from rdagent.scenarios.qlib.proposal.quant_proposal import QlibQuantHypothesisGen

    captured: dict[str, Any] = {}

    def _fake_super_prepare(self, trace):  # noqa: ANN001
        return {"RAG": "BASE_X"}, True

    def _fake_compose(base_rag: str, **kwargs: Any) -> str:
        captured.update(kwargs)
        return "FINAL"

    monkeypatch.setattr(QlibQuantHypothesisGen, "prepare_context", _fake_super_prepare)
    monkeypatch.setattr(qp, "compose_project_rag", _fake_compose)

    gen = qp.ProjectQlibQuantHypothesisGen.__new__(qp.ProjectQlibQuantHypothesisGen)
    gen._feedback_dir_override = None
    gen.auto_retrieval_enabled = False

    gen.prepare_context(trace={"hypothesis_text": "ignored"})
    assert captured.get("retrieval_query") is None


def test_prepare_context_survives_auto_query_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """自动推导抛异常 → 退化为 None，`compose_project_rag` 仍被调用一次，主流程不崩。"""
    from rdagent.scenarios.qlib.proposal.quant_proposal import QlibQuantHypothesisGen

    def _fake_super_prepare(self, trace):  # noqa: ANN001
        return {"RAG": "BASE_X"}, True

    def _boom(*_args: Any, **_kwargs: Any) -> tuple[str | None, str]:
        raise RuntimeError("simulated failure")

    captured: dict[str, Any] = {}

    def _fake_compose(base_rag: str, **kwargs: Any) -> str:
        captured.update(kwargs)
        return "FINAL"

    monkeypatch.setattr(QlibQuantHypothesisGen, "prepare_context", _fake_super_prepare)
    monkeypatch.setattr(qp, "_infer_retrieval_query_with_source", _boom)
    monkeypatch.setattr(qp, "compose_project_rag", _fake_compose)

    gen = qp.ProjectQlibQuantHypothesisGen.__new__(qp.ProjectQlibQuantHypothesisGen)
    gen._feedback_dir_override = None
    ctx, ok = gen.prepare_context(trace={"hypothesis_text": "x"})
    assert ok is True
    assert ctx["RAG"] == "FINAL"
    assert captured.get("retrieval_query") is None


def test_infer_query_with_source_override_label() -> None:
    q, src = qp._infer_retrieval_query_with_source(
        trace=None, ctx={"RETRIEVAL_QUERY": "explicit text"}
    )
    assert q == "explicit text"
    assert src == "override"


def test_infer_query_with_source_trace_label() -> None:
    q, src = qp._infer_retrieval_query_with_source(
        trace={"hypothesis_text": "use earnings revision strength"}, ctx={}
    )
    assert src == "trace"
    assert q and "earnings revision" in q


def test_infer_query_with_source_rag_tail_label() -> None:
    q, src = qp._infer_retrieval_query_with_source(
        trace=None, ctx={"RAG": "line1\nfinal rag line about valuation"}
    )
    assert src == "rag_tail"
    assert q and "valuation" in q


def test_infer_query_with_source_none_label() -> None:
    q, src = qp._infer_retrieval_query_with_source(trace=None, ctx={})
    assert q is None and src == "none"

