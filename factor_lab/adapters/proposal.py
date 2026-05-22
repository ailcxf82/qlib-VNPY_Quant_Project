"""factor_lab.adapters.proposal

Hypothesis2Experiment 子类：在 CoSTEER 代码生成 prompt 里强行注入我们仓库的
"I/O rules + column list + safety skeleton"，让 LLM 写出符合 result.h5 约束的 factor 代码。

**阶段 E 搬迁自** ``rdagent_integration/project_proposal.py`` —— 语义等价，仅 import
路径由新家 ``factor_lab.adapters.experiments`` 拉取 ``ProjectQlibFactorExperiment``。
"""

from __future__ import annotations

import json
import logging

from typing import Any, Tuple

logger = logging.getLogger(__name__)

from rdagent.components.coder.factor_coder.factor import FactorExperiment, FactorTask
from rdagent.components.coder.model_coder.model import ModelExperiment, ModelTask
from rdagent.core.proposal import Hypothesis, Trace
from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment
from rdagent.scenarios.qlib.proposal.factor_proposal import (
    QlibFactorHypothesis2Experiment,
    QlibFactorHypothesisGen,
)
from rdagent.scenarios.qlib.proposal.model_proposal import QlibModelHypothesis2Experiment

from factor_lab.adapters.experiments import (
    ProjectQlibFactorExperiment,
    ProjectQlibModelExperiment,
)


# The upstream ``scenarios/qlib/prompts.yaml`` ``hypothesis_and_feedback`` /
# ``last_hypothesis_and_feedback`` templates hard-code
# ``experiment.result.loc[[...]]`` against keys that our
# ``conf_combined_factors.yaml`` does NOT always emit (we only produce
# ``with_cost`` variants, and composite_score is the main metric). Jinja's
# ``StrictUndefined`` then rewrites the pandas KeyError into an
# ``UndefinedError`` and the whole ``direct_exp_gen`` step dies.
#
# ``factor_lab.adapters.quant_proposal`` already solves this for the mixed
# quant loop; we duplicate the minimal fix here so ``FactorRDLoop`` can also
# survive past Loop 0.
_TEMPLATE_REQUIRED_KEYS: tuple[str, ...] = (
    "IC",
    "1day.excess_return_without_cost.annualized_return",
    "1day.excess_return_without_cost.max_drawdown",
)

# Marker injected by ProjectQlibFactorHypothesis2Experiment.convert_response()
# into every factor_description. Everything from this marker onward is verbose
# boilerplate (column whitelist, skeleton code, rules) — ~300 token per factor.
# It is needed during *coding* but wastes tokens in *historical* hypothesis context.
_CODE_RULES_MARKER = "[CODE RULES - MUST FOLLOW]"


def _reindex_trace_results(trace: Trace) -> None:
    """Safely pad every historical ``experiment.result`` Series so the
    prompt templates can index the required keys without blowing up
    on Jinja ``StrictUndefined``. Missing keys become ``NaN``.
    """
    try:
        import pandas as pd
    except Exception:  # noqa: BLE001
        return
    hist = getattr(trace, "hist", None)
    if not isinstance(hist, list):
        return
    required = list(_TEMPLATE_REQUIRED_KEYS)
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
        try:
            exp.result = result.reindex(list(result.index) + missing)
        except Exception:  # noqa: BLE001
            continue


def _strip_code_rules_from_history(trace: Trace) -> None:
    """Strip verbose CODE RULES block from *historical* sub_task descriptions.

    Why this exists
    ---------------
    ``ProjectQlibFactorHypothesis2Experiment.convert_response()`` injects
    ``_CODER_RULES`` (~300 token per factor) into every ``factor_description``
    so the *coding* LLM sees the column whitelist and skeleton. That is correct.

    Problem: the Jinja ``hypothesis_and_feedback`` template also renders
    ``sub_task.factor_description`` for every past trial, so each loop adds
    N_factors × 300 extra tokens to the *propose* prompt. By Loop 2–3 the
    prompt exceeds GLM-5.1's context limit, triggering repeated RateLimitError
    and eventually a silent process death.

    Fix: before calling ``super().prepare_context(trace)``, strip everything
    from ``[CODE RULES...]`` onwards in *historical* sub_task descriptions.
    The trimmed descriptions still contain the human-readable factor name,
    purpose, and formula — exactly what the hypothesis-generation LLM needs.

    The mutation is permanent in-memory (the experiment objects are only used
    inside a single loop run, so there is no need to restore).
    """
    hist = getattr(trace, "hist", None)
    if not isinstance(hist, list):
        return
    for pair in hist:
        try:
            exp = pair[0]
        except Exception:  # noqa: BLE001
            continue
        for task in getattr(exp, "sub_tasks", None) or []:
            # factor_description is a read-only property aliasing task.description;
            # set the underlying writable attribute directly.
            desc = getattr(task, "description", None)
            if isinstance(desc, str) and _CODE_RULES_MARKER in desc:
                try:
                    task.description = desc[: desc.index(_CODE_RULES_MARKER)].rstrip()
                except AttributeError:
                    # Pydantic frozen model — mutate __dict__ directly as last resort
                    try:
                        task.__dict__["description"] = desc[: desc.index(_CODE_RULES_MARKER)].rstrip()
                    except Exception:  # noqa: BLE001
                        pass


_ALPHA158_AVOIDANCE_HINT = """
======  IMPORTANT: Diversity-Driven Proposal Policy  ======
The model already has 20 Alpha158 + 15 certified v19 factors. New factors must add ORTHOGONAL
information to be valuable. Three layers of restrictions apply:

[Layer 1] HARD AVOID — Alpha158 overlap (these signals already in model):
  - Simple price momentum (ROC5/10/20/60), RESI5/10, KLEN, KLOW, VSTD5, STD5
  - WVMA5/WVMA60, CORR5/10/20/60, CORD5/10/60, RSQR5/10/20/60

[Layer 2] CURRENTLY OVER-REPRESENTED in v19 pool — add ONLY with explicit orthogonality argument:
  - Volume ratio variants (sma_volume_ratio_10d, volume_ratio_ma_15d, winsorized_zscore_volume_ratio)
  - Turnover variants (turnover_surprise_20, turnover_acceleration_5d, turnover_accel_5d)
  - Liquidity variants (amihud_illiquidity_20d, log_signed_volume)

[Layer 3] HIGH-PRIORITY GAPS in v19 pool — these families have ZERO or ONE certified factor:
  ★★★ earnings_quality / fundamental revision — ONLY 1 factor (earnings_yield_momentum_5d)
      → Target: ROE acceleration, earnings revision rate, ROE/PE combo, PEG-ratio signal
  ★★★ industry-relative valuation — ZERO factors
      → Target: PE zscore within industry (use $sw_l1_code groupby), industry momentum spread
  ★★   long-horizon momentum (W=90/120) — ZERO factors
      → Target: residual momentum (exclude market beta), skip-1-month momentum
  ★★   overnight/after-hours information — ZERO factors (open/prev_close gap, W=20)
  ★    margin financing DIRECTION (vs level) — only close_location related factors
      → Target: margin net ratio trend, short-selling pressure

PREFER proposing from ★★★ families above. Every proposal should state which gap it fills.
======  END IMPORTANT  ======
"""


def _build_rag_query_from_trace(trace: Any, fallback: str = "") -> str:
    """Extract a focused retrieval query from recent trace history.

    Uses last 3 hypothesis descriptions + SOTA metrics (especially turnover)
    so the RAG retriever can select examples relevant to the *current situation*
    rather than the empty static constitution text.
    """
    hist = getattr(trace, "hist", None) or []
    parts: list[str] = []
    for exp, _ in reversed(hist[-3:]):
        hyp = getattr(exp, "hypothesis", None)
        desc = str(getattr(hyp, "hypothesis", "") or "")[:150].strip()
        if desc:
            parts.append(desc)
        res = getattr(exp, "result", None)
        if hasattr(res, "get"):
            turnover = res.get("1day.excess_return_with_cost.annualized_turnover")
            if turnover is not None:
                try:
                    parts.append(f"prev_turnover={float(turnover):.2f}")
                except (TypeError, ValueError):
                    pass
            composite = res.get("1day.composite_score")
            if composite is not None:
                try:
                    parts.append(f"prev_composite={float(composite):.3f}")
                except (TypeError, ValueError):
                    pass
    if parts:
        return " | ".join(parts)
    return fallback[:300]


def _enrich_factor_hypothesis_rag(ctx: dict[str, Any], trace: Trace) -> None:
    """Append project guidance into ``ctx['RAG']`` for hypothesis-generation LLM.

    ``LLMHypothesisGen.gen()`` reads ``context_dict['RAG']`` in the *user* prompt only;
    it builds *system* ``scenario`` from ``scen.get_scenario_all_desc()``. Writing to
    ``ctx['scenario']`` therefore has no effect — this was the original wiring bug.
    """
    from factor_lab.adapters.quant_proposal import (
        _infer_retrieval_query_with_source,
        compose_project_rag,
    )

    base_rag = str(ctx.get("RAG") or "")
    retrieval_query: str | None = None
    try:
        retrieval_query, _ = _infer_retrieval_query_with_source(trace=trace, ctx=ctx)
    except Exception as exc:  # noqa: BLE001
        logger.warning("factor hypothesis retrieval_query inference failed: %s", exc)

    try:
        rag_text = compose_project_rag(base_rag, retrieval_query=retrieval_query)
    except Exception as exc:  # noqa: BLE001
        logger.warning("compose_project_rag failed, keeping upstream RAG: %s", exc)
        rag_text = base_rag

    rag_text = rag_text.rstrip() + "\n\n" + _ALPHA158_AVOIDANCE_HINT.strip()

    try:
        from factor_lab.adapters.rag_retriever import (
            get_existing_family_counts,
            retrieve_examples,
        )

        trace_len = len(getattr(trace, "hist", None) or [])
        family_counts = get_existing_family_counts()
        rag_query = _build_rag_query_from_trace(trace, fallback=rag_text[:300])
        rag_block = retrieve_examples(
            hypothesis_text=rag_query,
            trace_length=trace_len,
            existing_families=list(family_counts.keys()),
        )
        if rag_block:
            rag_text = rag_text.rstrip() + "\n\n" + rag_block.strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("retrieve_examples failed, skipping academic RAG block: %s", exc)

    ctx["RAG"] = rag_text


class ProjectQlibFactorHypothesisGen(QlibFactorHypothesisGen):
    """Patch point for ``FactorRDLoop`` (pure factor loop).

    Wired via ``.env``::

        QLIB_FACTOR_HYPOTHESIS_GEN=factor_lab.adapters.proposal.ProjectQlibFactorHypothesisGen

    Applies pre-processing steps before the upstream template renders:
    1. ``_reindex_trace_results``  — pads missing metric keys to avoid Jinja crash.
    2. ``_strip_code_rules_from_history`` — trims CODE RULES boilerplate from
       historical factor descriptions to prevent prompt size explosion.
    3. Injects static constitution + diversity policy + academic examples into ``RAG``
       (the field ``LLMHypothesisGen.gen()`` passes to the user prompt).
    """

    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]:
        _reindex_trace_results(trace)
        _strip_code_rules_from_history(trace)
        ctx, ok = super().prepare_context(trace)
        if ok and isinstance(ctx, dict):
            _enrich_factor_hypothesis_rag(ctx, trace)
        return ctx, ok


class ProjectQlibFactorHypothesis2Experiment(QlibFactorHypothesis2Experiment):
    _STRICT_FACTOR_IO_RULES = """
------Project Mandatory I/O Rules (must follow)------
You MUST generate code with the following fixed skeleton and behavior.

[Required structure]
1) Implement helper functions:
   - `_load_daily_pv() -> pd.DataFrame`
   - `_build_result(index: pd.MultiIndex, factor_name: str, values: pd.Series) -> pd.DataFrame`
   - `calculate_<factor_name>() -> None`
2) In `__main__`, call ONLY `calculate_<factor_name>()`.

[Load rules]
3) Read source data ONLY from `daily_pv.h5` in current working directory.
4) Prefer `pd.read_hdf("daily_pv.h5")`; if key is needed, use `key="data"`.
5) After loading, enforce index names exactly `['datetime', 'instrument']` and sort index.
5b) Preserve instrument index AS-IS from loaded dataframe.
    Instrument code format is '000001.SZ' / '600000.SH' (6-digit.EXCHANGE).
    DO NOT rewrite to 'SH600000' prefix format. DO NOT split/rebuild instrument strings.
6) All columns may have NaN: coerce every column with `pd.to_numeric(..., errors="coerce")` before math.
6b) Available columns in daily_pv.h5 — v2 (42 columns, EXACT names — copy-paste verbatim):
    Price/raw:           $close, $open, $high, $low
    Price/fwd-adj:       $close_qfq, $open_qfq, $high_qfq, $low_qfq
    Volume/Amount:       $vol, $volume, $amount
    Liquidity:           $turnover_rate, $turnover_rate_f, $volume_ratio
    Technical:           $rsi_qfq_12, $macd_qfq, $macd_dif_qfq, $macd_dea_qfq,
                         $kdj_k_qfq, $kdj_d_qfq, $kdj_qfq, $atr_qfq, $mtmma_qfq
    Valuation:           $pe, $pe_ttm, $pb, $ps, $ps_ttm, $total_mv, $dv_ratio, $dv_ttm
    Quality:             $roe, $q_profit_yoy, $q_eps, $assets_turn, $profit_to_gr
    Money flow:          $net_amount, $buy_elg_amount, $buy_lg_amount, $buy_md_amount, $buy_sm_amount
    Margin financing:    $rzye, $rqye
    FORBIDDEN (do NOT exist): $rsi12, $macd, $kdj_k, $kdj_d, $atr, $factor,
                               $roa, $roa2_yearly (all-NaN in data source)

[Computation rules]
7) Compute factor with vectorized ops or `groupby(level="instrument").transform(...)`.
8) DO NOT use `groupby(...).apply(...)` to build final output column assignment.
9) Factor series must be explicitly converted to float64.
10) Safe division: use `(num / den)` after masking invalid den (e.g. replace 0 with NaN); never divide then drop all rows.
11) Do NOT `dropna()` on the full panel before save; NaN factor values are allowed. Do NOT shrink the MultiIndex to length 0.

[Output rules]
12) Result must be a DataFrame with exactly one column named factor_name.
13) Result index must be the original MultiIndex (`datetime`, `instrument`) with the same length as input (non-empty panel).
14) Save only to `result.h5` with key='data', mode='w', format='table', index=True (pandas-readable table).
15) FORBIDDEN index/string operations on final output:
    `reorder_levels`, `reset_index`, `set_index`, `stack`, `unstack`,
    `.str.split('.')`, `.str.replace(...)`, `.map(lambda ...)` on instrument index.
    Always build final output by `_build_result(df.index, factor_name, series)` using the original `df.index`.

[Reference skeleton to follow closely]
```python
import pandas as pd

def _coerce_num(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _load_daily_pv() -> pd.DataFrame:
    df = pd.read_hdf("daily_pv.h5")
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("daily_pv.h5 must use MultiIndex")
    df.index = df.index.set_names(["datetime", "instrument"])
    df = df.sort_index()
    return _coerce_num(df)

def _build_result(index: pd.MultiIndex, factor_name: str, values: pd.Series) -> pd.DataFrame:
    s = pd.to_numeric(values, errors="coerce").astype("float64")
    s = s.reindex(index)
    out = pd.DataFrame({factor_name: s}, index=index)
    if len(out.index) == 0:
        raise ValueError("result index is empty")
    out.to_hdf("result.h5", key="data", mode="w", format="table", index=True)
    return out
```
"""

    _FORMULA_IMPLEMENTATION_RULES = """
------Project formula implementation rules (fix2)------
1) Implement ONLY the hypothesis formulation; do not invent extra windows beyond one W in {5,10,20}.
2) Data are sorted by MultiIndex; use `groupby(level="instrument", group_keys=False)` + `transform` for path-safe rolling/pct_change.
3) Momentum sketch: `ret = df["$close_qfq"].groupby(level="instrument").transform(lambda s: s.pct_change(W))`.
4) Volatility sketch: `r = df["$close_qfq"].groupby(level="instrument").transform(lambda s: s.pct_change()); v = r.groupby(level="instrument").transform(lambda s: s.rolling(W, min_periods=W).std())`.
5) Final step: `_build_result(df.index, factor_name, factor_series_aligned_to_df_index)`.
6) If you accidentally produced a non-MultiIndex intermediate, DO NOT fix with `reorder_levels`; discard that path and recompute with `transform` so output series index stays aligned to `df.index`.

------Already-covered signals (Alpha158 built-ins) — DO NOT replicate------
The model already has these 20 Alpha158 features as baseline inputs. Proposing factors that merely
replicate these signals adds noise without incremental alpha. Seek ORTHOGONAL signals instead.
  RESI5/RESI10   : price residual from linear fit (5d / 10d)
  WVMA5/WVMA60   : dollar-volume weighted moving average ratio (5d / 60d) — covers volume×momentum
  RSQR5/RSQR10/RSQR20/RSQR60 : R² of price vs time (trend strength)
  KLEN            : (high-low)/open — daily range / volatility
  CORR5/CORR10/CORR20/CORR60 : Pearson(close, log_volume) — price-volume correlation
  CORD5/CORD10/CORD60 : Pearson(ret, log_vol_ret) — return-volume-change correlation
  ROC60           : 60-day return (long-term momentum)
  VSTD5           : volume coefficient of variation (5d)
  STD5            : close price std (5d) — short-term volatility
  RSQR60          : 60d R²
  KLOW            : (min(open,close)-low)/open — lower-shadow ratio

High-value targets NOT yet covered by Alpha158:
  - Cross-sectional rank interactions of FUNDAMENTALS × TECHNICALS (e.g. low-PE stocks with upward ROA trend)
  - Overnight gap signals: open/prev_close - 1 (info asymmetry, absent from Alpha158)
  - Margin-flow momentum: change in $rzye or $rqye relative to market cap
  - Earnings quality: ROE - ROA spread (leverage quality signal)
  - Intraday amplitude trend: (high-low)/close rolling trend direction
"""

    def prepare_context(self, hypothesis: Hypothesis, trace: Trace):
        ctx, ok = super().prepare_context(hypothesis, trace)
        if isinstance(ctx, dict):
            scenario = str(ctx.get("scenario", ""))
            scenario = scenario + "\n" + self._STRICT_FACTOR_IO_RULES + "\n" + self._FORMULA_IMPLEMENTATION_RULES

            # Phase-3 Coder RAG: inject 1 matching example (pitfalls + skeleton)
            # to help Coder avoid common column/index errors.
            try:
                from factor_lab.adapters.rag_retriever import retrieve_coder_hint
                hyp_text = str(getattr(hypothesis, "hypothesis", "") or "")[:300]
                coder_hint = retrieve_coder_hint(hyp_text, n=1, max_chars=800)
                if coder_hint:
                    scenario += coder_hint
            except Exception:  # noqa: BLE001
                pass  # Coder RAG failure must not break the loop

            ctx["scenario"] = scenario
        return ctx, ok

    _CODER_RULES = (
        "\n[CODE RULES - MUST FOLLOW]\n"
        "0. Available columns in daily_pv.h5 — v2 (42 cols, EXACT names — copy verbatim):\n"
        "   Price/raw:           $close, $open, $high, $low\n"
        "   Price/fwd-adj:       $close_qfq, $open_qfq, $high_qfq, $low_qfq\n"
        "   Volume/Amount:       $vol, $volume, $amount\n"
        "   Liquidity:           $turnover_rate, $turnover_rate_f, $volume_ratio\n"
        "   Technical:           $rsi_qfq_12, $macd_qfq, $macd_dif_qfq, $macd_dea_qfq,\n"
        "                        $kdj_k_qfq, $kdj_d_qfq, $kdj_qfq, $atr_qfq, $mtmma_qfq\n"
        "   Valuation:           $pe, $pe_ttm, $pb, $ps, $ps_ttm, $total_mv, $dv_ratio, $dv_ttm\n"
        "   Quality:             $roe, $q_profit_yoy, $q_eps, $assets_turn, $profit_to_gr\n"
        "   Money flow:          $net_amount, $buy_elg_amount, $buy_lg_amount,\n"
        "                        $buy_md_amount, $buy_sm_amount\n"
        "   Margin financing:    $rzye, $rqye\n"
        "   FORBIDDEN (do NOT exist): $rsi12, $macd, $kdj_k, $atr, $factor,\n"
        "                             $roa, $roa2_yearly (all-NaN in source)\n"
        "0b. Instrument code format: '000001.SZ' / '600000.SH' (6-digit.EXCHANGE).\n"
        "    DO NOT rewrite to 'SH600000' or split on '.'. Preserve index as-is.\n"
        "1. Load: df = pd.read_hdf('daily_pv.h5')  — NO HDFStore, NO h5py.\n"
        "2. Coerce all columns: for c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')\n"
        "3. For fundamental/quarterly columns ($pe_ttm, $pb, $roe, $q_profit_yoy, etc.) that have NaN:\n"
        "   Forward-fill within each instrument:\n"
        "   df['$col'] = df['$col'].groupby(level='instrument').transform(lambda s: s.ffill())\n"
        "4. Compute via: series = df['$close_qfq'].groupby(level='instrument').transform(lambda s: ...)\n"
        "5. Save: result = pd.DataFrame({'FACTOR_NAME': series.astype('float64')}, index=df.index)\n"
        "         result.to_hdf('result.h5', key='data', mode='w', format='table')\n"
        "6. NO dropna() on the panel. NaN rows are allowed. Result must have same len as df.\n"
        "7. NO unstack/stack/reorder_levels on final output.\n"
    )

    @staticmethod
    def _select_improving_experiments(trace: Trace) -> list:
        """Only keep experiments that actually improved over the previous best.

        Root-cause fix for "factor library bloat":
        The original code included ALL historical FactorExperiments in
        based_experiments regardless of whether they beat the previous SOTA.
        By Loop 9 this inflated the combined factor set to 60+ features
        (20 Alpha158 + ~40 user factors from 8 failed loops), diluting
        signal and locking IC at ~0.4441 instead of the SOTA 0.4467.

        This method runs a greedy pass: it only includes experiments whose
        composite_score strictly exceeded the running best at the time they
        were evaluated. Failed loops are excluded from based_experiments so
        they never pollute the combined_factors_df.parquet used by qrun.
        """
        accepted: list = []
        best_score: float = float("-inf")
        for t in trace.hist:
            if not (t[1] and isinstance(t[0], FactorExperiment)):
                continue
            result = getattr(t[0], "result", None)
            if result is None:
                continue
            try:
                score = float(result.get("1day.composite_score", float("-inf")))
            except Exception:  # noqa: BLE001
                score = float("-inf")
            if score > best_score:
                best_score = score
                accepted.append(t[0])
        return accepted

    def convert_response(self, response: str, hypothesis: Hypothesis, trace: Trace) -> FactorExperiment:
        response_dict = json.loads(response)
        tasks = []

        for factor_name in response_dict:
            description = response_dict[factor_name]["description"]
            formulation = response_dict[factor_name]["formulation"]
            variables = response_dict[factor_name]["variables"]
            augmented_desc = description + self._CODER_RULES.replace("FACTOR_NAME", factor_name)
            tasks.append(
                FactorTask(
                    factor_name=factor_name,
                    factor_description=augmented_desc,
                    factor_formulation=formulation,
                    variables=variables,
                )
            )

        exp = ProjectQlibFactorExperiment(tasks, hypothesis=hypothesis)
        # Only include experiments that improved SOTA — see _select_improving_experiments docstring
        exp.based_experiments = [ProjectQlibFactorExperiment(sub_tasks=[])] + \
            self._select_improving_experiments(trace)

        unique_tasks = []
        for task in tasks:
            duplicate = False
            for based_exp in exp.based_experiments:
                if isinstance(based_exp, QlibModelExperiment):
                    continue
                for sub_task in based_exp.sub_tasks:
                    if task.factor_name == sub_task.factor_name:
                        duplicate = True
                        break
                if duplicate:
                    break
            if not duplicate:
                unique_tasks.append(task)

        exp.sub_tasks = unique_tasks
        return exp


class ProjectQlibModelHypothesis2Experiment(QlibModelHypothesis2Experiment):
    def convert_response(self, response: str, hypothesis: Hypothesis, trace: Trace) -> ModelExperiment:
        response_dict = json.loads(response)
        tasks = []
        for model_name in response_dict:
            description = response_dict[model_name]["description"]
            formulation = response_dict[model_name]["formulation"]
            architecture = response_dict[model_name]["architecture"]
            variables = response_dict[model_name]["variables"]
            hyperparameters = response_dict[model_name]["hyperparameters"]
            training_hyperparameters = response_dict[model_name]["training_hyperparameters"]
            model_type = response_dict[model_name]["model_type"]
            tasks.append(
                ModelTask(
                    name=model_name,
                    description=description,
                    formulation=formulation,
                    architecture=architecture,
                    variables=variables,
                    hyperparameters=hyperparameters,
                    training_hyperparameters=training_hyperparameters,
                    model_type=model_type,
                )
            )
        exp = ProjectQlibModelExperiment(tasks, hypothesis=hypothesis)
        exp.based_experiments = [t[0] for t in trace.hist if t[1] and isinstance(t[0], ModelExperiment)]
        return exp
