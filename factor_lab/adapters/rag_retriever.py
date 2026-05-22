"""factor_lab.adapters.rag_retriever
====================================
Phase-3：从学术因子知识库检索相似因子示例，注入 LLM proposal 上下文。

架构
----
1. 加载 factor_lab/rag_db/academic_factors/*.yaml → 展平为 List[FactorEntry]。
2. 用关键词匹配（无需 embedding，不依赖外部 API）在 trace history 的 hypothesis
   文本和现有宪法中查找最相关的 1-3 个因子示例。
3. 将示例格式化为 RAG prompt 块，返回给 ProjectQlibFactorHypothesisGen.prepare_context()
   注入到 `scenario` 中。

关键词匹配优先顺序：
  a) 精确匹配 hypothesis 中的 factor family 标签
  b) 当前 trace 中 SOTA 指标判断 — 若 turnover 高 → 优先检索 slow-moving 因子族
  c) 随机均匀采样（保证多样性，每 loop 有机会看到所有族）

未来升级
--------
- 替换关键词匹配为本地 sentence-transformers embedding（BAAI/bge-small-zh-v1.5）
  只需实现 _embed_text() 并改 _score_entry()，外部调用不变。
"""

from __future__ import annotations

import logging
import os
import random
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_RAG_DB_DIR = Path(__file__).resolve().parents[1] / "rag_db" / "academic_factors"
# Additional directories loaded in priority order (academic first, then certified)
_RAG_DB_EXTRA_DIRS: list[Path] = [
    Path(__file__).resolve().parents[1] / "rag_db" / "implemented_factors",
]

# 每次注入的最大示例数（太多会超出 context window）
_MAX_EXAMPLES = int(os.environ.get("FACTOR_LAB_RAG_MAX_EXAMPLES", "3"))

# 每隔几个 loop 强制随机探索（避免总是检索相同家族）
_EXPLORATION_PERIOD = int(os.environ.get("FACTOR_LAB_RAG_EXPLORATION_PERIOD", "3"))


# ─── 数据结构 ──────────────────────────────────────────────────────────────────

_FORBIDDEN_COLUMNS: frozenset[str] = frozenset({"$roa", "$roa2_yearly", "$factor"})


class FactorEntry:
    __slots__ = ("id", "family", "name", "formula", "columns_used", "expected_ic", "notes", "pitfalls")

    def __init__(self, d: dict[str, Any]) -> None:
        self.id: str = str(d.get("id", ""))
        self.family: str = str(d.get("family", ""))
        self.name: str = str(d.get("name", ""))
        self.formula: str = str(d.get("formula", ""))
        self.columns_used: list[str] = [str(c) for c in (d.get("columns_used") or [])]
        self.expected_ic: float = float(d.get("expected_ic") or 0.0)
        self.notes: str = str(d.get("notes", ""))
        raw_pitfalls = d.get("pitfalls") or []
        self.pitfalls: list[str] = [str(p) for p in raw_pitfalls] if isinstance(raw_pitfalls, list) else []

    def has_forbidden_columns(self) -> bool:
        return any(c in _FORBIDDEN_COLUMNS for c in self.columns_used)

    def to_prompt_block(self) -> str:
        cols = ", ".join(self.columns_used)
        lines = [
            f"  [RAG Example: {self.name} | family={self.family} | "
            f"expected_IC≈{self.expected_ic:.3f}]",
            f"  columns: {cols}",
            "  formula sketch:",
        ]
        for line in self.formula.strip().splitlines():
            lines.append(f"    {line}")
        if self.pitfalls:
            lines.append("  pitfalls (avoid these):")
            for p in self.pitfalls:
                lines.append(f"    - {p}")
        lines.append(f"  notes: {self.notes.strip()}")
        return "\n".join(lines) + "\n"

    def to_coder_hint(self, max_chars: int = 800) -> str:
        """Compact hint for the Coder stage: pitfalls + skeleton only."""
        lines = [
            f"  [RAG Coder Ref: {self.name} | family={self.family}]",
            f"  columns used: {', '.join(self.columns_used)}",
        ]
        if self.pitfalls:
            lines.append("  PITFALLS (must avoid):")
            for p in self.pitfalls:
                lines.append(f"    ! {p}")
        lines.append("  formula sketch (adapt, do NOT copy):")
        for line in self.formula.strip().splitlines()[:10]:
            lines.append(f"    {line}")
        result = "\n".join(lines) + "\n"
        return result[:max_chars]


# ─── 加载知识库 ────────────────────────────────────────────────────────────────

_GLOBAL_POOL: list[FactorEntry] | None = None


def _load_entries_from_dir(yaml_dir: Path, pool: list[FactorEntry]) -> int:
    """Load FactorEntry objects from all *.yaml files in yaml_dir into pool.

    Returns the number of entries added.
    """
    if not yaml_dir.exists():
        logger.debug("rag_db dir not found: %s", yaml_dir)
        return 0
    try:
        import yaml
    except ImportError:
        return 0
    added = 0
    for yaml_file in sorted(yaml_dir.glob("*.yaml")):
        try:
            with yaml_file.open("r", encoding="utf-8") as f:
                doc = yaml.safe_load(f) or {}
            for entry_dict in doc.get("factors") or []:
                if isinstance(entry_dict, dict):
                    entry = FactorEntry(entry_dict)
                    if entry.has_forbidden_columns():
                        logger.warning(
                            "RAG entry %s uses forbidden columns %s — skipping",
                            entry.id,
                            [c for c in entry.columns_used if c in _FORBIDDEN_COLUMNS],
                        )
                        continue
                    pool.append(entry)
                    added += 1
        except Exception as exc:
            logger.warning("Failed to load RAG yaml %s: %s", yaml_file, exc)
    return added


def _load_pool() -> list[FactorEntry]:
    global _GLOBAL_POOL
    if _GLOBAL_POOL is not None:
        return _GLOBAL_POOL
    pool: list[FactorEntry] = []
    try:
        import yaml  # noqa: F401 — check availability early
    except ImportError:
        logger.warning("pyyaml not available; RAG retriever disabled")
        _GLOBAL_POOL = pool
        return pool

    n_academic = _load_entries_from_dir(_RAG_DB_DIR, pool)
    n_extra = sum(_load_entries_from_dir(d, pool) for d in _RAG_DB_EXTRA_DIRS)
    logger.info(
        "RAG pool loaded: %d entries (academic=%d, extra=%d)",
        len(pool), n_academic, n_extra,
    )
    _GLOBAL_POOL = pool
    return pool


# ─── 评分 / 检索 ───────────────────────────────────────────────────────────────

def _detect_high_turnover(context_text: str) -> bool:
    """Return True if context indicates a previous high-turnover regime."""
    import re
    text_lower = context_text.lower()
    # Explicit turnover mention with high value
    for m in re.finditer(r"turnover[=\s:]*([0-9]+\.?[0-9]*)", text_lower):
        try:
            if float(m.group(1)) >= 6.0:
                return True
        except ValueError:
            pass
    # Key phrases
    high_turnover_phrases = ["high turnover", "turnover>6", "turnover > 6", "turnover doubled"]
    return any(p in text_lower for p in high_turnover_phrases)


def _score_entry(entry: FactorEntry, context_text: str) -> float:
    """返回 entry 与 context_text 的相关性分数（0-1）。

    当前实现：关键词重叠 + expected_IC 偏好 + turnover 条件族权重调整。
    TODO: 替换为 sentence-transformers cosine similarity。
    """
    score = 0.0
    text_lower = context_text.lower()
    family_lower = entry.family.lower()

    # 家族关键词命中
    family_keywords = {
        "value": ["value", "valuation", "pe", "pb", "price-to-earnings", "low-pe", "pb"],
        "quality": ["quality", "roe", "earnings", "profit", "revision", "eps", "profit_to_gr"],
        "momentum": ["momentum", "trend", "return", "residual", "skip", "overnight", "gap"],
        "liquidity": ["liquidity", "turnover", "illiquidity", "amihud", "bid-ask"],
        "smart_money": ["margin", "short", "smart money", "rzye", "rqye", "northbound"],
        "industry_relative": ["industry", "sector", "intra-sector", "sw_l1"],
    }
    kws = family_keywords.get(family_lower, [])
    hits = sum(1 for kw in kws if kw in text_lower)
    score += hits * 0.15

    # 列名命中（因子用到的列在 context 中出现）
    for col in entry.columns_used:
        clean = col.lstrip("$")
        if clean in text_lower:
            score += 0.05

    # 高 IC 轻微加分
    score += entry.expected_ic * 2.0

    # 高 turnover 条件：降权 liquidity/momentum 族，加权 value/quality 族
    if _detect_high_turnover(context_text):
        if family_lower in ("liquidity", "momentum"):
            score -= 0.2
        elif family_lower in ("value", "quality"):
            score += 0.1

    # 随机扰动保证多样性
    score += random.uniform(0, 0.05)

    return score


def _avoid_duplicates_with_existing(
    entries: list[FactorEntry],
    existing_family_counts: dict[str, int],
) -> list[FactorEntry]:
    """对已经有很多因子的家族进行降权，以保证多样性。"""
    def sort_key(e: FactorEntry) -> float:
        cnt = existing_family_counts.get(e.family, 0)
        # 已有 >=3 个同族认证因子则 penalise
        penalty = 0.2 * max(0, cnt - 2)
        return -penalty  # sort descending

    return sorted(entries, key=sort_key, reverse=True)


# ─── 公开 API ─────────────────────────────────────────────────────────────────

def retrieve_examples(
    hypothesis_text: str,
    trace_length: int = 0,
    existing_families: list[str] | None = None,
    n: int | None = None,
) -> str:
    """检索最相关的 n 个因子示例，返回 prompt 注入块。

    Parameters
    ----------
    hypothesis_text : str
        当前 loop 的 hypothesis 描述（用于关键词匹配）。
    trace_length : int
        当前 trace 长度（用于探索/利用切换）。
    existing_families : list[str], optional
        现有认证因子的家族列表（用于去重偏好）。
    n : int, optional
        返回示例数，默认读 FACTOR_LAB_RAG_MAX_EXAMPLES 环境变量。
    """
    pool = _load_pool()
    if not pool:
        return ""

    max_n = n if n is not None else _MAX_EXAMPLES

    # 探索模式：每 _EXPLORATION_PERIOD 个 loop 强制随机采样
    if trace_length > 0 and (trace_length % _EXPLORATION_PERIOD == 0):
        selected = random.sample(pool, min(max_n, len(pool)))
    else:
        # 利用模式：关键词评分
        scored = sorted(pool, key=lambda e: _score_entry(e, hypothesis_text), reverse=True)
        # 去重：避免同一家族占满所有示例
        family_counts: dict[str, int] = {}
        for f in (existing_families or []):
            family_counts[f] = family_counts.get(f, 0) + 1
        selected: list[FactorEntry] = []
        for entry in scored:
            if len(selected) >= max_n:
                break
            # 同一家族最多选2个示例
            fam_in_selected = sum(1 for e in selected if e.family == entry.family)
            if fam_in_selected >= 2:
                continue
            selected.append(entry)

    if not selected:
        return ""

    lines = [
        "",
        "======  RAG: Academic Factor Examples (use as inspiration, NOT copy-paste)  ======",
        "These factors have been validated in prior research. Adapt their SPIRIT to your hypothesis.",
    ]
    for entry in selected:
        lines.append(entry.to_prompt_block())
    lines.append("======  END RAG EXAMPLES  ======")
    return "\n".join(lines)


def retrieve_coder_hint(hypothesis_text: str, n: int = 1, max_chars: int = 800) -> str:
    """检索 1 条最相关实现示例作为 Coder 阶段提示（pitfalls + skeleton）。

    刻意限制在 1 条、截断到 max_chars，避免 Coder prompt 过长。
    """
    pool = _load_pool()
    if not pool:
        return ""
    scored = sorted(pool, key=lambda e: _score_entry(e, hypothesis_text), reverse=True)
    selected = scored[:n]
    if not selected:
        return ""
    lines = [
        "",
        "======  RAG Coder Reference (adapt structure; do NOT copy verbatim)  ======",
    ]
    for entry in selected:
        lines.append(entry.to_coder_hint(max_chars=max_chars))
    lines.append("======  END CODER REFERENCE  ======")
    return "\n".join(lines)


def get_existing_family_counts() -> dict[str, int]:
    """从 factor_registry/data/manifest.json 统计现有认证因子的家族分布。
    返回 {family_name: count}，用于多样性排序。
    """
    try:
        import json
        manifest_path = (
            Path(__file__).resolve().parents[2] / "factor_registry" / "data" / "manifest.json"
        )
        if not manifest_path.exists():
            return {}
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        counts: dict[str, int] = {}
        for entry in manifest.get("factors") or []:
            family = entry.get("family") or _infer_family(entry.get("factor_name", ""))
            counts[family] = counts.get(family, 0) + 1
        return counts
    except Exception as exc:
        logger.debug("get_existing_family_counts failed: %s", exc)
        return {}


def _infer_family(factor_name: str) -> str:
    """启发式从因子名推断家族标签（manifest 中无 family 字段时的 fallback）。"""
    n = factor_name.lower()
    if any(k in n for k in ("volume", "vol", "turn", "illiq", "amihud")):
        return "liquidity"
    if any(k in n for k in ("roe", "roa", "earn", "profit", "eps", "quality")):
        return "quality"
    if any(k in n for k in ("pe_ttm", "pb", "ps_ttm", "dv_ratio", "value")):
        return "value"
    if any(k in n for k in ("mom", "ret", "return", "momentum", "overnight")):
        return "momentum"
    if any(k in n for k in ("rzye", "rqye", "margin")):
        return "smart_money"
    return "other"
