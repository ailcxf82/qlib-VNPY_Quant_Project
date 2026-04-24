"""Quant-level hypothesis generation: static constitution + dynamic L2 feedback RAG injection.

**阶段 E.3 设计要点**

* ``_STATIC_CONSTITUTION`` 保留"硬编码宪法"：因子生成的格式/列名/打分规则 + 历史一次性经验
  （`csi300_RD_v2` 的失败案例），这部分不依赖数据，每次 RD-Agent loop 都一致。
* 动态反馈（最近 N cycle 的 L2 判决、L3 已入库、L3 已退役）由 ``factor_lab/feedback`` 从
  ``factor_lab/workspace/feedback/latest.json`` 读取，按 schema 渲染成 markdown 追加到 RAG。
* **安全降级**：feedback bundle 缺失 / 损坏 / factor_lab 包未装时，注入器仅注入静态宪法，
  绝不能让 RD-Agent 挂掉。

公开入口：

* :class:`ProjectQlibQuantHypothesisGen`
* :func:`compose_project_rag` —— 纯函数，单元可测。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Tuple

from rdagent.core.proposal import Trace
from rdagent.scenarios.qlib.proposal.quant_proposal import QlibQuantHypothesisGen

from factor_lab.config.constitution import get_constitution_text

logger = logging.getLogger(__name__)


# 阶段 F.3：静态宪法事实源迁到 factor_lab/config/rag_constitution.yaml；
# 通过 factor_lab.config.constitution.get_constitution_text() 懒加载。
# 老名字 ``_STATIC_CONSTITUTION`` 保留为模块级属性（仍是字符串），外部 import 不断。
# 仅在模块首次被访问时懒求值，确保 pyyaml 缺失也不阻塞 import。

_HARDCODED_STATIC_CONSTITUTION_FALLBACK = """
------Project factor hypothesis constraints (mandatory)------
1) Propose at most 2 new factors per hypothesis when trace length < 8; at most 3 afterward.
2) Each factor MUST use a single integer window W chosen from {5, 10, 20, 30, 60} (state W explicitly).
3) Allowed primitives: pct_change, shift, rolling(W).mean/std/sum/min/max/corr, rank, clip.
4) FORBIDDEN: nested rolling correlations across many series, loops, "10 pairs" patterns.
5) Available columns in daily_pv.h5 (use EXACTLY these names, no others):
   Price/Volume: $close, $open, $high, $low, $volume, $amount
   Liquidity:    $turnover_rate, $turnover_rate_f, $volume_ratio
   Valuation:    $pe_ttm, $pb, $ps_ttm, $total_mv, $dv_ratio
   Quality:      $roe, $roa, $q_profit_yoy, $q_eps
   Technical:    $rsi12, $macd, $macd_dif, $kdj_k, $kdj_d, $atr
   Margin:       $rzye, $rqye
   NOTE: $pe_ttm/$pb/$roe/$q_profit_yoy may have NaN for some instruments (quarterly data).
         Always use .fillna(method='ffill') or rolling mean as fallback for fundamental columns.

------Reward / objective (P3a, mandatory reading)------
You are scored by `1day.composite_score = 1.0*IR + 2.0*IC_IR - 0.5*log(1+annualized_turnover)`.
This means three things you MUST optimise simultaneously, NOT just IC:
  · IR  (information_ratio of excess_return_with_cost) — real backtest signal-to-noise
  · IC_IR (Rank IC mean / IC std) — signal stability
  · annualized_turnover — penalised; high-frequency switching hurts the score.

Empirical lesson from previous loops (csi300_RD_v2, 2025-11~2026-03):
  Adding 5 short-cycle volume-price-reversal factors (RangeRatio_10D, VolRatio_20D,
  VolumePriceTrend_10D, VolumeTrend_10D, VolRet_5D) raised LGB valid-IC from 0.148
  to 0.226 BUT the realised Sharpe DROPPED from 3.59 to 1.69 because annualized
  turnover almost doubled (4.7 -> 8.75) and the strategy beta collapsed from 1.50
  to 0.90 (signals fought each other instead of stacking). DO NOT propose more
  short-cycle volume-price-reversal factors; they are the failure family.

------Existing feature universe (`lgb_short_cycle`, 32 columns)------
The project already exposes these columns to LGB; new factors must add INCREMENTAL
information, i.e. low correlation with these:
  Returns/Momentum: RET1, MOM5, MOM10, MOM20, MOM1_5
  Volatility:       VOL5, VOL10, VOL20
  Volume ratios:    VRATIO5, VRATIO10, VRATIO20, VOLVOL
  Range:            HLRANGE, HLRANGE5, HLRANGE10
  Liquidity:        TURN, TURN_F, TURN_REL
  Valuation:        PE_TTM, PB, PS_TTM, LOG_MV
  Quality:          ROE, ROA, PROFIT_YOY
  Technical:        RSI12, MACD, KDJ_DIFF, ATR
  Margin:           MARGIN_L, MARGIN_S

Hard requirement (mandatory):
  6.a) New factor MUST satisfy |Spearman(new_factor, X)| <= 0.50 for every X above
       (when X is a strict superset, prove orthogonality through transformation).
  6.b) New factor's day-over-day cross-sectional rank auto-correlation MUST be
       >= 0.60 (slow-moving) — this directly bounds turnover.
  6.c) Avoid signals that primarily fire on the SAME day as a price jump; prefer
       lagged / smoothed transformations.

------Encouraged factor families (HIGH composite-score expectation)------
   (a) Quality persistence: rolling_mean($roe, 4 quarters) ranked vs sector
   (b) Valuation mean-reversion (slow): ($pe_ttm - rolling_median($pe_ttm, 60)) / rolling_std($pe_ttm, 60)
   (c) Margin financing trend: rolling_mean($rzye / $total_mv, 20) - rolling_mean($rzye / $total_mv, 60)
   (d) Earnings revision strength: $q_profit_yoy minus its 4-quarter rolling median (use shift)
   (e) Low-volatility quality: rank($roe) / (rank(VOL20) + 1)
   (f) Long-horizon residual momentum: pct_change($close, 60) - beta * pct_change(benchmark, 60)
   (g) Liquidity stability: 1 / rolling_std($turnover_rate / Mean($turnover_rate, 60), 20)

------Discouraged factor families (LOW composite-score, DO NOT propose)------
   (x) Short-cycle volume-price reversals on W in {5, 10}
   (y) Same-day volume spike + price reversal patterns
   (z) Anything that ranks the universe with >50% weekly turnover

7) Factor names must encode type and window, e.g. QualPersist_60D, ValueMR_60D,
   MarginTrend_20D, EarnRev_4Q, LowVolQual_20D, ResidMom_60D, LiqStab_20D.
"""


# 懒加载 + 兜底：优先读 YAML 渲染结果；YAML 不可用时用上面的硬编码字符串。
try:
    _STATIC_CONSTITUTION: str = get_constitution_text()
except Exception as exc:  # noqa: BLE001
    logger.warning(
        "factor_lab.config.constitution 加载失败，静态宪法回退到模块内硬编码 err=%s", exc
    )
    _STATIC_CONSTITUTION = _HARDCODED_STATIC_CONSTITUTION_FALLBACK


# ---------------------------------------------------------- feedback injection


def _default_feedback_dir() -> Path:
    """默认 feedback bundle 位置（本项目 ``factor_lab/workspace/feedback``）。"""
    # 本文件位于 factor_lab/adapters/quant_proposal.py，项目根 = parents[2]
    root = Path(__file__).resolve().parents[2]
    return root / "factor_lab" / "workspace" / "feedback"


def _render_dynamic_feedback(feedback_dir: Path | None) -> str:
    """
    尝试加载最新 feedback bundle，渲染成 markdown；失败时返回空串，**绝不抛异常**。

    使用反射式导入保证：即便 ``factor_lab.feedback`` 在某些精简部署里缺失，RD-Agent 依然能跑。
    """
    target_dir = feedback_dir or _default_feedback_dir()
    try:
        from factor_lab.feedback.aggregator import load_latest_feedback_bundle
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "factor_lab.feedback 不可用，跳过动态反馈注入：%s", exc
        )
        return ""
    try:
        bundle = load_latest_feedback_bundle(target_dir)
    except Exception as exc:  # noqa: BLE001
        logger.warning("加载 feedback bundle 失败 %s: %s", target_dir, exc)
        return ""
    if bundle is None:
        logger.info(
            "feedback bundle 缺失（%s），仅注入静态宪法", target_dir
        )
        return ""
    try:
        md = bundle.to_markdown()
    except Exception as exc:  # noqa: BLE001
        logger.warning("渲染 feedback bundle 失败: %s", exc)
        return ""
    return md


def _render_similar_failures(
    feedback_dir: Path | None,
    *,
    query: str,
    top_k: int,
) -> str:
    """阶段 G.3：读 latest bundle，按 query 召回 top-k 历史失败并渲染。失败时安全返回空串。"""
    target_dir = feedback_dir or _default_feedback_dir()
    try:
        from factor_lab.feedback.aggregator import load_latest_feedback_bundle
        from factor_lab.feedback.embedding import (
            render_similar_failures_section,
            retrieve_similar_failures,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("factor_lab.feedback.embedding 不可用，跳过 G.3 检索：%s", exc)
        return ""
    try:
        bundle = load_latest_feedback_bundle(target_dir)
    except Exception as exc:  # noqa: BLE001
        logger.warning("G.3 加载 bundle 失败 %s: %s", target_dir, exc)
        return ""
    if bundle is None or not bundle.recent_fails:
        return ""
    try:
        hits = retrieve_similar_failures(bundle, query, top_k=top_k)
        return render_similar_failures_section(hits, query=query)
    except Exception as exc:  # noqa: BLE001
        logger.warning("G.3 检索/渲染失败: %s", exc)
        return ""


def _truncate_query(text: str, *, max_chars: int = 220) -> str:
    """把自动推导出来的 query 限长，防止 trace 巨大文本灌爆 prompt。"""
    t = " ".join(str(text).split())
    if not t:
        return ""
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 3].rstrip() + "..."


def _extract_candidate_query_parts(trace: Any) -> list[str]:
    """从 trace 的常见字段提取候选 query 片段（最强语义优先）。"""
    if trace is None:
        return []

    out: list[str] = []

    # 1) dict-like trace（测试替身/轻量调用常见）
    if isinstance(trace, dict):
        for key in (
            "retrieval_query",
            "query",
            "hypothesis",
            "hypothesis_text",
            "proposal",
            "summary",
            "goal",
            "task",
            "description",
            "latest_hypothesis",
            "current_hypothesis",
            "current_task",
        ):
            v = trace.get(key)
            if isinstance(v, str) and v.strip():
                out.append(v.strip())
        return out

    # 2) object-like trace：先读直观字段
    for attr in (
        "retrieval_query",
        "query",
        "hypothesis",
        "hypothesis_text",
        "proposal",
        "summary",
        "goal",
        "task",
        "description",
        "latest_hypothesis",
        "current_hypothesis",
        "current_task",
    ):
        try:
            v = getattr(trace, attr)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(v, str) and v.strip():
            out.append(v.strip())

    # 3) trace.hist：抓最后几条历史里的字符串内容
    try:
        hist = getattr(trace, "hist")
    except Exception:  # noqa: BLE001
        hist = None
    if isinstance(hist, (list, tuple)) and hist:
        for item in list(hist)[-3:]:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
                continue
            if isinstance(item, (list, tuple)):
                for x in item:
                    if isinstance(x, str) and x.strip():
                        out.append(x.strip())
            elif isinstance(item, dict):
                for k in ("hypothesis", "summary", "proposal", "description"):
                    x = item.get(k)
                    if isinstance(x, str) and x.strip():
                        out.append(x.strip())
    return out


def _infer_retrieval_query_with_source(
    *, trace: Any, ctx: dict[str, Any]
) -> tuple[str | None, str]:
    """阶段 H.1：自动推导 retrieval_query，同时返回来源标签供观测。

    返回 ``(query or None, source)``。``source`` 取值：
      * ``"override"`` —— 命中 ``ctx['RETRIEVAL_QUERY']``；
      * ``"trace"``    —— 命中 trace 语义字段；
      * ``"rag_tail"`` —— 兜底取 ``ctx['RAG']`` 末行；
      * ``"none"``     —— 全部空。
    """
    override = ctx.get("RETRIEVAL_QUERY")
    if isinstance(override, str) and override.strip():
        q = _truncate_query(override)
        return (q or None), "override"

    parts = _extract_candidate_query_parts(trace)
    if parts:
        q = _truncate_query(" | ".join(parts[:3]))
        return (q or None), "trace"

    base = str(ctx.get("RAG") or "").strip()
    if base:
        tail = base.splitlines()[-1]
        q = _truncate_query(tail)
        return (q or None), "rag_tail"

    return None, "none"


def _infer_retrieval_query(*, trace: Any, ctx: dict[str, Any]) -> str | None:
    """便捷 wrapper：仅返回 query，丢弃来源标签（保留以兼容测试）。"""
    q, _src = _infer_retrieval_query_with_source(trace=trace, ctx=ctx)
    return q


def compose_project_rag(
    base_rag: str,
    *,
    feedback_dir: Path | None = None,
    include_dynamic: bool = True,
    retrieval_query: str | None = None,
    retrieval_top_k: int = 3,
) -> str:
    """
    把 static constitution + dynamic feedback + (可选) G.3 相似检索段拼到 ``base_rag`` 末尾。

    分段顺序固定（便于 prompt 复现）::

        <base_rag>

        ------Project factor hypothesis constraints (mandatory)------ ...

        ------Feedback from recent L2 cycles (dynamic)------ ...

        ------Similar past failures retrieved for query=... (G.3 top-K)------ ...

    参数：
        base_rag         上游 prepare_context 产出的 RAG 字符串（可能为空）。
        feedback_dir     测试可注入；默认 ``factor_lab/workspace/feedback``。
        include_dynamic  False 时仅注入静态段（给需要完全关闭动态的场景：debug / A/B 对照）。
        retrieval_query  阶段 G.3：若非空，从 latest bundle 里按 query 检索 top-k 相似失败
                         并追加一段"针对性历史教训"。None/空串 → 不注入这段。
        retrieval_top_k  G.3 检索返回条数上限（默认 3）。
    """
    parts: list[str] = []
    base = (base_rag or "").rstrip()
    if base:
        parts.append(base)

    parts.append(_STATIC_CONSTITUTION.strip())

    if include_dynamic:
        dyn = _render_dynamic_feedback(feedback_dir)
        if dyn:
            parts.append(dyn.strip())

    if retrieval_query and retrieval_query.strip():
        sim = _render_similar_failures(
            feedback_dir, query=retrieval_query, top_k=retrieval_top_k
        )
        if sim:
            parts.append(sim.strip())

    return "\n\n".join(parts)


# ---------------------------------------------------------------- rdagent hook


class ProjectQlibQuantHypothesisGen(QlibQuantHypothesisGen):
    """
    扩展默认 ``QlibQuantHypothesisGen``：

    * 注入 :data:`_STATIC_CONSTITUTION`（历史经验 + 格式约束）；
    * 运行时读取 ``factor_lab/workspace/feedback/latest.json``，把 L2 最近 N cycle 的
      判决追加到 RAG；读取失败时安全降级到仅静态。

    通过 ``prepare_context`` 作为唯一挂钩点，保持对 RD-Agent 升级的兼容。
    """

    #: 允许运行时环境变量覆盖：``PROJECT_FEEDBACK_DIR``（绝对路径）。
    #: 留空则用默认 ``factor_lab/workspace/feedback``。
    _feedback_dir_override: Path | None = None

    #: 阶段 H.1：自动 retrieval_query 推导总开关；置 False 可退回 G 阶段行为
    #: （不自动传 query；compose_project_rag 不会注入 G.3 段）。
    auto_retrieval_enabled: bool = True

    #: RD-Agent ``scenarios/qlib/prompts.yaml`` 的 ``hypothesis_and_feedback``
    #: / ``last_hypothesis_and_feedback`` / ``sota_hypothesis_and_feedback`` 三
    #: 个模板硬编码 ``experiment.result.loc[[...]]``。当我们的
    #: ``conf_combined_factors.yaml`` 只产出 ``with_cost`` 版本指标时，
    #: jinja StrictUndefined 会把 pandas KeyError 重写为 UndefinedError，整
    #: 个 direct_exp_gen 崩溃。此列表用于在 super().prepare_context 前做
    #: reindex，把缺失索引安全补成 NaN。
    _TEMPLATE_REQUIRED_KEYS: tuple[str, ...] = (
        "IC",
        "1day.excess_return_without_cost.annualized_return",
        "1day.excess_return_without_cost.max_drawdown",
    )

    @classmethod
    def _reindex_trace_results(cls, trace: Trace) -> None:
        """Ensure every historical experiment.result contains the 3 keys.

        Side-effect: rewrites ``experiment.result`` in place to a reindexed
        pandas Series. Keys that did not exist are filled with ``NaN`` — the
        Jinja template will then print them as ``NaN`` instead of raising.
        Non-Series / None results are left untouched.
        """
        try:
            import pandas as pd
        except Exception:  # noqa: BLE001
            return

        hist = getattr(trace, "hist", None)
        if not isinstance(hist, list):
            return

        required = list(cls._TEMPLATE_REQUIRED_KEYS)
        for pair in hist:
            try:
                exp = pair[0]
            except Exception:  # noqa: BLE001
                continue
            result = getattr(exp, "result", None)
            if not isinstance(result, pd.Series):
                continue
            missing = [k for k in required if k not in result.index]
            if not missing:
                continue
            combined_index = list(result.index) + missing
            try:
                exp.result = result.reindex(combined_index)
            except Exception:  # noqa: BLE001
                # Defensive: never let the reindex itself break the loop.
                continue

    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]:
        # Must run BEFORE super() — upstream template consumes trace.hist
        # directly via Jinja StrictUndefined and will crash on missing keys.
        self._reindex_trace_results(trace)
        ctx, ok = super().prepare_context(trace)
        if ok and isinstance(ctx, dict):
            base = str(ctx.get("RAG") or "")

            retrieval_query: str | None = None
            source = "disabled"
            if self.auto_retrieval_enabled:
                try:
                    retrieval_query, source = _infer_retrieval_query_with_source(
                        trace=trace, ctx=ctx
                    )
                except Exception as exc:  # noqa: BLE001
                    # H.1 安全降级：推导失败不能拖垮主流程
                    logger.warning("H.1 自动 retrieval_query 推导失败: %s", exc)
                    retrieval_query, source = None, "error"

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "H.1 auto_retrieval source=%s len=%d",
                    source,
                    len(retrieval_query) if retrieval_query else 0,
                )

            try:
                ctx["RAG"] = compose_project_rag(
                    base,
                    feedback_dir=self._feedback_dir_override,
                    retrieval_query=retrieval_query,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("compose_project_rag 失败，保留 super 的 RAG: %s", exc)

        return ctx, ok


# -------------------- backward-compat alias for existing tests / imports ------


# 旧字段名保留，避免外部 import 断裂。内容与新接口等价。
# 仅用于外部调用（例如审计 / snapshot 测试）；class 内部不再引用。
_PROJECT_FACTOR_RAG = _STATIC_CONSTITUTION
