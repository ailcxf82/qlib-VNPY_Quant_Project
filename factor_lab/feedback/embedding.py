"""阶段 G.3：把历史失败候选做成一个可检索语料库，按语义相似度挑 top-k 注入 RAG。

背景：
    阶段 F 把「最近 N cycle 的所有失败」统一罗列到 RAG 里，当失败池变大（>30）
    时 prompt 会变长且 LLM 容易忽略。G.3 加一个检索层：
        研究员 / RD-Agent 给定 query（通常是本轮 hypothesis 的简短描述或待试因子名），
        我们从 bundle.recent_fails 里按 query 相似度挑 top-k 条回注 RAG。
    目的不是替代平铺段，而是**补一个"针对性"段**，让 LLM 在大语料下也能看到
    与当前方向最像的历史教训。

设计原则：

* **零外部依赖**：默认实现 ``TfidfBackend`` 用纯 Python + stdlib，不引 scikit-learn /
  sentence-transformers。大规模数据集未来可替换（plugin protocol 已开放）。
* **纯函数可测**：语料、query、top_k 全部显式传入；无全局状态。
* **安全降级**：空语料 / 空 query / tokenization 失败 → 返回空列表，调用方自行判断
  是否注入 RAG；**绝不**对 compose_project_rag 的主路径产生副作用。
* **小而稳**：token 抽取包含 snake_case + CamelCase + 数字切片；TF-IDF 按经典公式，
  cosine 距离归一。

Public API：

* :class:`RetrievedFailure` —— 打包一条召回结果（原 summary + 相似度分）。
* :class:`EmbeddingBackend` —— Protocol；为未来替换 backend 留口。
* :class:`TfidfBackend` —— 默认实现。
* :func:`build_failure_corpus` —— 从 FeedbackBundle 抽取可检索文本语料。
* :func:`retrieve_similar_failures` —— 一站式：bundle + query → top-k。
* :func:`render_similar_failures_section` —— top-k → markdown 段落，给 compose_project_rag 用。
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

from factor_lab.feedback.schema import FailedCandidateSummary, FeedbackBundle

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------- tokenizer


_TOKEN_SPLIT = re.compile(r"[^A-Za-z0-9]+")
# 注意顺序：先吃"大写缩写 + 下一个词头"（MACDFoo → MACD+Foo），
# 再吃"纯大写缩写到结尾 / 末位"（MACD → MACD），最后才是普通 Camel、小写、数字。
_CAMEL_SPLIT = re.compile(
    r"[A-Z]+(?=[A-Z][a-z])|[A-Z]+(?![a-zA-Z])|[A-Z][a-z0-9]*|[a-z]+|[0-9]+"
)


def _tokenize(text: str) -> list[str]:
    """把因子名 / 家族字符串切成小写 token。

    规则：
      * 非字母数字作为分隔符 (``_ - / \\s``)；
      * 每段再按 CamelCase + 数字边界切；
      * 丢弃空串、统一小写。

    例：
      * ``"VolRev_5d"``            → ``["vol", "rev", "5", "d"]``
      * ``"volume_price_reversal"`` → ``["volume", "price", "reversal"]``
      * ``"QualPersist_60D"``       → ``["qual", "persist", "60", "d"]``
    """
    if not isinstance(text, str) or not text:
        return []
    chunks = _TOKEN_SPLIT.split(text)
    out: list[str] = []
    for chunk in chunks:
        if not chunk:
            continue
        for m in _CAMEL_SPLIT.findall(chunk):
            if m:
                out.append(m.lower())
    return out


# -------------------------------------------------------- protocol & backends


class EmbeddingBackend(Protocol):
    """未来可替换为 sentence-transformers / litellm embeddings 等。"""

    def fit(self, corpus: Sequence[str]) -> None:  # pragma: no cover - protocol
        ...

    def similarity(self, query: str, doc: str) -> float:  # pragma: no cover
        ...


@dataclass
class TfidfBackend:
    """轻量 TF-IDF + cosine 相似度。仅依赖 stdlib。

    * ``idf[t] = log((N + 1) / (df[t] + 1)) + 1`` —— 带 smoothing，保证未见 token 权重 > 0。
    * 文档向量按 TF 归一，cosine 在比较时动态计算。
    """

    _doc_tokens: list[list[str]] = None  # type: ignore[assignment]
    _idf: dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._doc_tokens is None:
            self._doc_tokens = []
        if self._idf is None:
            self._idf = {}

    def fit(self, corpus: Sequence[str]) -> None:
        docs = [_tokenize(s) for s in corpus]
        N = max(1, len(docs))
        df: dict[str, int] = {}
        for doc in docs:
            for t in set(doc):
                df[t] = df.get(t, 0) + 1
        self._doc_tokens = docs
        self._idf = {t: math.log((N + 1) / (df[t] + 1)) + 1 for t in df}

    def _vec(self, doc: list[str]) -> dict[str, float]:
        if not doc:
            return {}
        tf = Counter(doc)
        total = sum(tf.values())
        if total == 0:
            return {}
        return {t: (tf[t] / total) * self._idf.get(t, 1.0) for t in tf}

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        if na == 0.0 or nb == 0.0:
            return 0.0
        keys = set(a) & set(b)
        dot = sum(a[k] * b[k] for k in keys)
        return dot / (na * nb)

    def similarity(self, query: str, doc: str) -> float:
        q_vec = self._vec(_tokenize(query))
        d_vec = self._vec(_tokenize(doc))
        return self._cosine(q_vec, d_vec)


# --------------------------------------------------------------- data types


@dataclass(frozen=True)
class RetrievedFailure:
    """检索命中的一条历史失败（原 summary + 相似度得分）。"""

    score: float
    doc_text: str
    failure: FailedCandidateSummary


# ------------------------------------------------------ corpus + retrieval


def _failure_to_doc(f: FailedCandidateSummary) -> str:
    """把一条 FailedCandidateSummary 拼成可索引的字符串文档。

    包含：``name family failure_modes universe`` —— 这四项对 token 相似度最有帮助，
    metrics 的数值噪声大反而拖累检索精度，故不进文档。
    """
    parts: list[str] = [f.name, f.family]
    if f.universe:
        parts.append(f.universe)
    parts.extend(f.failure_modes)
    return " ".join(parts)


def build_failure_corpus(
    fails: Iterable[FailedCandidateSummary],
) -> tuple[list[str], list[FailedCandidateSummary]]:
    """把失败候选序列转成 (docs, 原 summary 列表)，保持同下标对齐。"""
    docs: list[str] = []
    items: list[FailedCandidateSummary] = []
    for f in fails:
        docs.append(_failure_to_doc(f))
        items.append(f)
    return docs, items


def retrieve_similar_failures(
    bundle: FeedbackBundle,
    query: str,
    *,
    top_k: int = 3,
    min_score: float = 0.05,
    backend: EmbeddingBackend | None = None,
) -> list[RetrievedFailure]:
    """从 ``bundle.recent_fails`` 里按 query 相似度召回 top-k。

    参数：
      * ``query``     自然语言 query；通常是本轮 hypothesis 的简短描述或目标家族名。
      * ``top_k``     返回条数上限（默认 3，长 prompt 友好）。
      * ``min_score`` 低于此阈值的结果被过滤掉（避免注入无关失败）。
      * ``backend``   自定义检索 backend；默认 TfidfBackend。

    安全约束：
      * query 为空字符串 / 纯空白 → 返回空列表；
      * bundle 无 recent_fails → 返回空列表；
      * top_k <= 0 → 返回空列表。
    """
    if not isinstance(query, str) or not query.strip():
        return []
    if top_k <= 0:
        return []
    if not bundle.recent_fails:
        return []

    docs, items = build_failure_corpus(bundle.recent_fails)
    if not docs:
        return []

    be = backend if backend is not None else TfidfBackend()
    try:
        be.fit(docs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("embedding backend fit 失败，跳过召回：%s", exc)
        return []

    scored: list[RetrievedFailure] = []
    for i, doc in enumerate(docs):
        try:
            s = be.similarity(query, doc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("similarity 计算失败 doc=%s: %s", doc, exc)
            continue
        if s >= min_score:
            scored.append(RetrievedFailure(score=s, doc_text=doc, failure=items[i]))
    scored.sort(key=lambda r: (-r.score, r.failure.name))
    return scored[:top_k]


# ----------------------------------------------------------- RAG rendering


def render_similar_failures_section(
    retrieved: Sequence[RetrievedFailure],
    *,
    query: str,
    header_note: str = "",
) -> str:
    """把召回结果拼成一段 markdown；空输入返回空串，上游据此决定是否拼入 RAG。"""
    if not retrieved:
        return ""
    lines: list[str] = []
    lines.append(
        f"------Similar past failures retrieved for query=\"{query}\" (G.3 top-{len(retrieved)})------"
    )
    if header_note:
        lines.append(header_note)
    for r in retrieved:
        f = r.failure
        modes = ",".join(f.failure_modes) if f.failure_modes else "n/a"
        u = f" [universe={f.universe}]" if f.universe else ""
        lines.append(
            f"  - sim={r.score:.3f}  {f.name}  (family={f.family})  "
            f"stage={f.stage}  decision={f.decision}  modes=[{modes}]{u}"
        )
    lines.append(
        "Interpretation: these are the closest past failures to the current direction; "
        "if a new factor looks too similar to any of them, explain explicitly how it differs "
        "(feature, window, transformation) before continuing."
    )
    return "\n".join(lines)
