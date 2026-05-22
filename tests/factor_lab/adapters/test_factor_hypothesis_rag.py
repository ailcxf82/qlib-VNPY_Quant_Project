"""Factor hypothesis gen must inject guidance into ctx['RAG'], not ctx['scenario']."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

rdagent = pytest.importorskip("rdagent")

from factor_lab.adapters.proposal import (  # noqa: E402
    _ALPHA158_AVOIDANCE_HINT,
    _enrich_factor_hypothesis_rag,
)


class _EmptyTrace:
    hist: list = []


def test_enrich_writes_rag_not_scenario() -> None:
    ctx = {
        "RAG": "Try the easiest and fastest factors to experiment with from various perspectives first.",
        "hypothesis_specification": "upstream spec",
    }
    _enrich_factor_hypothesis_rag(ctx, _EmptyTrace())  # type: ignore[arg-type]

    rag = ctx.get("RAG") or ""
    assert "Diversity-Driven" in rag
    assert "HIGH-PRIORITY" in rag
    assert "Project factor hypothesis constraints" in rag
    assert "Encouraged factor families" in rag
    assert _ALPHA158_AVOIDANCE_HINT.strip() in rag
    assert "Try the easiest and fastest factors" in rag
    assert "RAG: Academic Factor Examples" in rag or "RAG Example:" in rag
    assert "scenario" not in ctx or not str(ctx.get("scenario", "")).strip()


def test_enrich_is_idempotent_on_base_upstream_line() -> None:
    ctx = {"RAG": "upstream only"}
    _enrich_factor_hypothesis_rag(ctx, _EmptyTrace())  # type: ignore[arg-type]
    assert "upstream only" in (ctx.get("RAG") or "")
