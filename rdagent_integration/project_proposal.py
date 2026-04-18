"""Hypothesis2Experiment classes that build experiments with project-local Qlib YAML templates."""

from __future__ import annotations

import json

from rdagent.components.coder.factor_coder.factor import FactorExperiment, FactorTask
from rdagent.components.coder.model_coder.model import ModelExperiment, ModelTask
from rdagent.core.proposal import Hypothesis, Trace
from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment
from rdagent.scenarios.qlib.proposal.factor_proposal import QlibFactorHypothesis2Experiment
from rdagent.scenarios.qlib.proposal.model_proposal import QlibModelHypothesis2Experiment

from rdagent_integration.project_experiments import ProjectQlibFactorExperiment, ProjectQlibModelExperiment


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
6) All columns may have NaN: coerce every column with `pd.to_numeric(..., errors="coerce")` before math.
6b) Available columns in daily_pv.h5 (EXACT names):
    Price/Volume: $close, $open, $high, $low, $volume, $amount
    Liquidity:    $turnover_rate, $turnover_rate_f, $volume_ratio
    Valuation:    $pe_ttm, $pb, $ps_ttm, $total_mv, $dv_ratio
    Quality:      $roe, $roa, $q_profit_yoy, $q_eps
    Technical:    $rsi12, $macd, $macd_dif, $kdj_k, $kdj_d, $atr
    Margin:       $rzye, $rqye
    FORBIDDEN: $close_qfq, $open_qfq, $vol, $factor, $pe, $net_amount — do NOT exist.

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
15) FORBIDDEN index-fix operations on final output: `reorder_levels`, `reset_index`, `set_index`, `stack`, `unstack`.
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
3) Momentum sketch: `ret = df["$close"].groupby(level="instrument").transform(lambda s: s.pct_change(W))`.
4) Volatility sketch: `r = df["$close"].groupby(level="instrument").transform(lambda s: s.pct_change()); v = r.groupby(level="instrument").transform(lambda s: s.rolling(W, min_periods=W).std())`.
5) Final step: `_build_result(df.index, factor_name, factor_series_aligned_to_df_index)`.
6) If you accidentally produced a non-MultiIndex intermediate, DO NOT fix with `reorder_levels`; discard that path and recompute with `transform` so output series index stays aligned to `df.index`.
"""

    def prepare_context(self, hypothesis: Hypothesis, trace: Trace):
        ctx, ok = super().prepare_context(hypothesis, trace)
        if isinstance(ctx, dict):
            scenario = str(ctx.get("scenario", ""))
            ctx["scenario"] = (
                scenario + "\n" + self._STRICT_FACTOR_IO_RULES + "\n" + self._FORMULA_IMPLEMENTATION_RULES
            )
        return ctx, ok

    # Injected directly into FactorTask.factor_description so the CoSTEER code-gen LLM sees it
    _CODER_RULES = (
        "\n[CODE RULES - MUST FOLLOW]\n"
        "0. Available columns in daily_pv.h5 (EXACT names only):\n"
        "   Price/Volume: $close, $open, $high, $low, $volume, $amount\n"
        "   Liquidity:    $turnover_rate, $turnover_rate_f, $volume_ratio\n"
        "   Valuation:    $pe_ttm, $pb, $ps_ttm, $total_mv, $dv_ratio\n"
        "   Quality:      $roe, $roa, $q_profit_yoy, $q_eps\n"
        "   Technical:    $rsi12, $macd, $macd_dif, $kdj_k, $kdj_d, $atr\n"
        "   Margin:       $rzye, $rqye\n"
        "   DO NOT use $close_qfq, $vol, $factor, $pe, $net_amount — they do NOT exist.\n"
        "1. Load: df = pd.read_hdf('daily_pv.h5')  — NO HDFStore, NO h5py.\n"
        "2. Coerce all columns: for c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')\n"
        "3. For fundamental columns ($pe_ttm, $pb, $roe, etc.) that have NaN:\n"
        "   Use forward-fill within each instrument before computation:\n"
        "   df['$pe_ttm'] = df['$pe_ttm'].groupby(level='instrument').transform(lambda s: s.ffill())\n"
        "4. Compute via: series = df['$close'].groupby(level='instrument').transform(lambda s: ...)\n"
        "5. Save: result = pd.DataFrame({'FACTOR_NAME': series.astype('float64')}, index=df.index)\n"
        "         result.to_hdf('result.h5', key='data', mode='w', format='table')\n"
        "6. NO dropna() on the panel. NaN rows are allowed. Result must have same len as df.\n"
        "7. NO unstack/stack/reorder_levels on final output.\n"
    )

    def convert_response(self, response: str, hypothesis: Hypothesis, trace: Trace) -> FactorExperiment:
        response_dict = json.loads(response)
        tasks = []

        for factor_name in response_dict:
            description = response_dict[factor_name]["description"]
            formulation = response_dict[factor_name]["formulation"]
            variables = response_dict[factor_name]["variables"]
            # Inject code rules into description so CoSTEER code-gen LLM receives them
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
        exp.based_experiments = [ProjectQlibFactorExperiment(sub_tasks=[])] + [
            t[0] for t in trace.hist if t[1] and isinstance(t[0], FactorExperiment)
        ]

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
