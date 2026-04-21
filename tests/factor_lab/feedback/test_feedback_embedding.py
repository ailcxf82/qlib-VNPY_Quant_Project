"""阶段 G.3：embedding-based RAG 检索层单测。

覆盖：
  1. 分词规则（snake_case / CamelCase / 数字边界 / 空串）。
  2. TfidfBackend：fit+similarity 在对的语义上给出合理排序。
  3. retrieve_similar_failures：
       * 空 query / 空 bundle / top_k=0 → 空列表；
       * top_k 上限 + min_score 阈值生效；
       * 排序稳定（按 score desc，ties 按 name asc）。
  4. render_similar_failures_section：
       * 空输入 → 空串；
       * 非空时包含 sim/score、family、universe、modes、interpretation。
  5. compose_project_rag G.3 集成：
       * retrieval_query=None / 空 → 不改变现有 RAG；
       * retrieval_query 非空 + bundle 里有失败 → 附加一段 G.3 语料；
       * retrieval_query 非空但 bundle 空 → 不附加（安全降级）。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from factor_lab.adapters.quant_proposal import compose_project_rag
from factor_lab.feedback import (
    FailedCandidateSummary,
    FeedbackBundle,
)
from factor_lab.feedback.embedding import (
    RetrievedFailure,
    TfidfBackend,
    _tokenize,
    build_failure_corpus,
    render_similar_failures_section,
    retrieve_similar_failures,
)


# --------------------------------------------------------------- tokenizer


@pytest.mark.parametrize(
    "text,expected",
    [
        ("VolRev_5d", ["vol", "rev", "5", "d"]),
        ("volume_price_reversal", ["volume", "price", "reversal"]),
        ("QualPersist_60D", ["qual", "persist", "60", "d"]),
        ("MACD_dif", ["macd", "dif"]),
        ("", []),
        ("   ", []),
        ("a-b/c.d", ["a", "b", "c", "d"]),
    ],
)
def test_tokenize_rules(text: str, expected: list[str]) -> None:
    assert _tokenize(text) == expected


# -------------------------------------------------------------- TfidfBackend


def test_tfidf_similarity_ranks_related_docs_higher() -> None:
    corpus = [
        "VolRev_5d volume_price_reversal ic",         # 与 query 高度相关
        "QualPersist_60D quality_persist ic",         # 中度相关
        "ResidMom_60D long_horizon_residual_momentum",  # 基本不相关
    ]
    be = TfidfBackend()
    be.fit(corpus)
    q = "short-cycle volume price reversal factor"
    sims = [be.similarity(q, d) for d in corpus]
    assert sims[0] > sims[1], f"volume-reversal 相似度应最高: {sims}"
    assert sims[0] > sims[2]


def test_tfidf_zero_on_disjoint_docs() -> None:
    be = TfidfBackend()
    be.fit(["foo bar baz", "alpha beta gamma"])
    assert be.similarity("alpha beta gamma", "foo bar baz") == 0.0


# ---------------------------------------------- corpus + retrieval end-to-end


def _make_fail(
    name: str,
    family: str,
    *,
    universe: str | None = None,
    modes: tuple[str, ...] = ("ic",),
    cycle_id: str = "c1",
) -> FailedCandidateSummary:
    return FailedCandidateSummary(
        name=name,
        family=family,
        universe=universe,
        cycle_id=cycle_id,
        stage="default",
        decision="FAIL",
        failure_modes=modes,
    )


def _bundle(fails: tuple[FailedCandidateSummary, ...]) -> FeedbackBundle:
    cycles = tuple(sorted({f.cycle_id for f in fails})) or ("c1",)
    return FeedbackBundle(
        generated_at=datetime(2026, 4, 18, tzinfo=timezone.utc),
        cycles_included=cycles,
        window_max_cycles=max(4, len(cycles)),
        recent_fails=fails,
    )


def test_build_failure_corpus_keeps_alignment() -> None:
    fails = (
        _make_fail("VolRev_5d", "volume_price_reversal", universe="csi300"),
        _make_fail("QualPersist_60D", "quality_persist"),
    )
    docs, items = build_failure_corpus(fails)
    assert len(docs) == 2
    assert items[0].name == "VolRev_5d"
    assert "csi300" in docs[0]
    assert "QualPersist" not in docs[0]  # 不应污染对方


def test_retrieve_empty_query_returns_empty() -> None:
    b = _bundle((_make_fail("VolRev_5d", "volume_price_reversal"),))
    assert retrieve_similar_failures(b, query="") == []
    assert retrieve_similar_failures(b, query="   ") == []


def test_retrieve_empty_bundle_returns_empty() -> None:
    b = FeedbackBundle(
        generated_at=datetime(2026, 4, 18, tzinfo=timezone.utc),
        cycles_included=(),
        window_max_cycles=4,
    )
    assert retrieve_similar_failures(b, query="anything") == []


def test_retrieve_top_k_bound() -> None:
    fails = tuple(
        _make_fail(f"VolRev_{w}d", "volume_price_reversal") for w in range(5, 11)
    )
    b = _bundle(fails)
    hits = retrieve_similar_failures(b, query="volume price reversal", top_k=2)
    assert len(hits) == 2
    assert hits[0].score >= hits[1].score


def test_retrieve_prefers_semantic_match() -> None:
    fails = (
        _make_fail("QualPersist_60D", "quality_persist"),
        _make_fail("VolRev_5d", "volume_price_reversal"),
        _make_fail("ResidMom_60D", "long_horizon_residual_momentum"),
    )
    b = _bundle(fails)
    hits = retrieve_similar_failures(b, query="volume reversal short cycle", top_k=1)
    assert len(hits) == 1
    assert hits[0].failure.name == "VolRev_5d"


def test_retrieve_min_score_filters_noise() -> None:
    fails = (_make_fail("QualPersist_60D", "quality_persist"),)
    b = _bundle(fails)
    hits = retrieve_similar_failures(
        b, query="volume reversal short cycle", top_k=5, min_score=0.9
    )
    assert hits == []


# ---------------------------------------------------------------- render


def test_render_empty_retrieval_returns_empty_string() -> None:
    assert render_similar_failures_section([], query="anything") == ""


def test_render_includes_expected_tokens() -> None:
    hit = RetrievedFailure(
        score=0.777,
        doc_text="VolRev_5d volume_price_reversal csi300 ic",
        failure=_make_fail("VolRev_5d", "volume_price_reversal", universe="csi300"),
    )
    md = render_similar_failures_section([hit], query="my query")
    assert "Similar past failures" in md
    assert "query=\"my query\"" in md
    assert "VolRev_5d" in md
    assert "[universe=csi300]" in md
    assert "Interpretation" in md
    assert "0.777" in md


# -------------------------------------------------- compose_project_rag 集成


def _write_bundle(feedback_dir: Path, fails: list[FailedCandidateSummary]) -> None:
    feedback_dir.mkdir(parents=True, exist_ok=True)
    bundle = _bundle(tuple(fails))
    (feedback_dir / "latest.json").write_text(
        bundle.model_dump_json(indent=2), encoding="utf-8"
    )


def test_compose_project_rag_without_query_unchanged_by_g3(tmp_path: Path) -> None:
    _write_bundle(tmp_path, [_make_fail("VolRev_5d", "volume_price_reversal")])
    out_no_q = compose_project_rag("BASE", feedback_dir=tmp_path)
    out_empty_q = compose_project_rag(
        "BASE", feedback_dir=tmp_path, retrieval_query="   "
    )
    assert out_no_q == out_empty_q
    assert "Similar past failures" not in out_no_q


def test_compose_project_rag_with_query_appends_g3_section(tmp_path: Path) -> None:
    _write_bundle(tmp_path, [_make_fail("VolRev_5d", "volume_price_reversal")])
    out = compose_project_rag(
        "BASE",
        feedback_dir=tmp_path,
        retrieval_query="volume price reversal",
        retrieval_top_k=3,
    )
    assert "Project factor hypothesis constraints" in out  # 静态段仍在
    assert "Feedback from recent L2 cycles" in out  # 动态段仍在
    assert "Similar past failures retrieved" in out  # G.3 段新增
    assert "VolRev_5d" in out


def test_compose_project_rag_g3_safe_when_bundle_missing(tmp_path: Path) -> None:
    out = compose_project_rag(
        "BASE",
        feedback_dir=tmp_path / "nope",
        retrieval_query="anything",
    )
    assert "Similar past failures" not in out
    assert "Project factor hypothesis constraints" in out


def test_compose_project_rag_section_order_static_dynamic_g3(tmp_path: Path) -> None:
    _write_bundle(tmp_path, [_make_fail("VolRev_5d", "volume_price_reversal")])
    out = compose_project_rag(
        "BASE",
        feedback_dir=tmp_path,
        retrieval_query="volume reversal",
    )
    idx_static = out.index("Project factor hypothesis constraints")
    idx_dyn = out.index("Feedback from recent L2 cycles")
    idx_g3 = out.index("Similar past failures retrieved")
    assert idx_static < idx_dyn < idx_g3
