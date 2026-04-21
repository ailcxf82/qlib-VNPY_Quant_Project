"""Point RD-Agent's Qlib conda runs at env QLIB_RDAGENT_CONDA_ENV (default: rdagent)."""

from __future__ import annotations

import os
from typing import Any

import rdagent.utils.env as env_mod

# Absolute bin path of the WSL rdagent micromamba environment.
# Contains: rdagent 0.8.0 + pyqlib 0.9.7 + qrun
_RDAGENT_ENV_BIN = "/home/administrator/.local/share/mamba/envs/rdagent/bin"


def _patch_embedding_graceful_fallback() -> None:
    """
    RD-Agent calls APIBackend().create_embedding() in two places:
      1. KnowledgeMetaData.create_embedding  — when a successful task is stored in the graph
      2. calculate_embedding_distance_between_str_list — when the RAG queries similar tasks

    Both paths crash if EMBEDDING_MODEL points to an API that has no valid key.
    This patch wraps the root method (APIBackend.create_embedding) so that any API
    failure is silently replaced by a zero-vector, allowing the run to continue without
    a working embedding endpoint.

    Fast-path optimisation: if ``OPENAI_API_KEY`` is empty AND the configured
    embedding model is an OpenAI model (which is the project default — chat is
    DeepSeek but embedding still points at ``openai/text-embedding-3-small``),
    return the zero-vector **without** calling the backend. Otherwise every single
    embedding call would first burn ~10 seconds on LiteLLM's auth-retry loop
    (``max_retry=10, retry_wait_seconds=1``) before our ``except`` fires, which
    in a CoSTEER loop with ~20 stored successes quickly accumulates into minutes
    of wall time and hundreds of 401 warnings. CoSTEER's knowledge-graph
    similarity search still works — zero-vectors just mean "all embeddings are
    equidistant", so retrieval becomes a no-op, which is the correct graceful
    degradation when embeddings are unavailable.
    """
    try:
        import os as _os
        import numpy as np
        from rdagent.oai.backend.base import APIBackend

        _EMBED_DIM = 1536  # standard OpenAI/DeepSeek-compatible dimension

        def _embedding_api_unavailable() -> bool:
            emb_model = (_os.environ.get("EMBEDDING_MODEL") or "").lower()
            if emb_model.startswith("openai/") or emb_model.startswith("azure/openai/"):
                return not (_os.environ.get("OPENAI_API_KEY") or "").strip()
            return False

        def _zero_vec(input_content):
            if isinstance(input_content, str):
                return np.zeros(_EMBED_DIM, dtype=np.float32)
            return [np.zeros(_EMBED_DIM, dtype=np.float32) for _ in input_content]

        _original_create = APIBackend.create_embedding

        def _safe_create_embedding(self: APIBackend, input_content, **kwargs):  # type: ignore[override]
            if _embedding_api_unavailable():
                return _zero_vec(input_content)
            try:
                return _original_create(self, input_content, **kwargs)
            except Exception:
                return _zero_vec(input_content)

        APIBackend.create_embedding = _safe_create_embedding  # type: ignore[method-assign]
    except Exception:
        pass  # Silently skip if rdagent internals change in a future version.


def _patch_feedback_important_metrics() -> None:
    """
    P3a: extend RD-Agent's hard-coded IMPORTANT_METRICS so the LLM feedback loop
    also sees turnover + composite_score + information_ratio.

    Without this patch the LLM only optimises against {IC, annualized_return,
    max_drawdown}. After this patch the comparison table presented to the LLM
    additionally contains:
      - 1day.excess_return_with_cost.information_ratio
      - 1day.excess_return_with_cost.annualized_turnover  (added in read_exp_res.py)
      - 1day.composite_score                              (added in read_exp_res.py)
    Steering the LLM to propose factors with high signal-to-noise ratio AND low
    turnover, instead of high-IC short-cycle reversal factors that blow up
    turnover (the failure mode observed in csi300_RD_v2).
    """
    try:
        import rdagent.scenarios.qlib.developer.feedback as feedback_mod

        extra = [
            "1day.excess_return_with_cost.information_ratio",
            "1day.excess_return_with_cost.annualized_turnover",
            "1day.composite_score",
        ]
        existing = list(getattr(feedback_mod, "IMPORTANT_METRICS", []))
        for k in extra:
            if k not in existing:
                existing.append(k)
        feedback_mod.IMPORTANT_METRICS = existing
    except Exception:
        pass


def _patch_inject_code_utf8() -> None:
    """
    RD-Agent bug (Windows + GBK locale):
    ``rdagent.core.experiment.FBWorkspace.inject_code_from_folder`` calls
    ``file_path.read_text()`` without ``encoding=``. On Windows boxes whose
    locale default is GBK/CP936 this raises ``UnicodeDecodeError`` the moment
    it hits any UTF-8 template file containing non-ASCII bytes (e.g. a BOM,
    `—`, `…`, Chinese comment, etc.), which happens early in the
    Hypothesis2Experiment step — long before qrun is ever reached.

    We monkey-patch the method to read every file as UTF-8. This matches the
    template files' actual encoding (RD-Agent writes them UTF-8). Applied
    once at startup by :func:`apply_qlib_conda_env_patch`; safe to double-apply.
    """
    try:
        from rdagent.core.experiment import FBWorkspace

        _orig = FBWorkspace.inject_code_from_folder

        # idempotence guard so repeated apply_qlib_conda_env_patch() calls are safe
        if getattr(_orig, "__factor_lab_utf8_patched__", False):
            return

        def _safe_inject_code_from_folder(self, folder_path):  # type: ignore[override]
            for file_path in folder_path.rglob("*"):
                if file_path.suffix in (".py", ".yaml", ".md"):
                    relative_path = file_path.relative_to(folder_path)
                    try:
                        text = file_path.read_text(encoding="utf-8")
                    except UnicodeDecodeError:
                        # defensive fallback: template shouldn't have mojibake,
                        # but if it does, don't kill the whole loop — log-and-skip
                        text = file_path.read_text(encoding="utf-8", errors="replace")
                    self.inject_files(**{str(relative_path): text})

        _safe_inject_code_from_folder.__factor_lab_utf8_patched__ = True  # type: ignore[attr-defined]
        FBWorkspace.inject_code_from_folder = _safe_inject_code_from_folder  # type: ignore[method-assign]
    except Exception:
        pass


def _patch_feedback_process_results() -> None:
    """
    Wrap feedback.process_results to gracefully handle missing benchmark metrics.

    When PortAnaRecord succeeds but the SOTA result only has IC metrics (e.g. from a
    previous run before the benchmark fix), the concat/reindex in process_results raises
    a KeyError for 'annualized_return' and 'max_drawdown'. This patch catches that
    specific KeyError and returns a NaN-filled fallback DataFrame so the workflow
    continues instead of crashing.

    P3a addendum: also reindex with the extended IMPORTANT_METRICS list installed
    by ``_patch_feedback_important_metrics`` so missing turnover/composite keys
    do not crash older runs.
    """
    try:
        import pandas as pd
        import rdagent.scenarios.qlib.developer.feedback as feedback_mod

        _orig_process_results = feedback_mod.process_results

        def _safe_process_results(current_result, sota_result):  # type: ignore[override]
            try:
                return _orig_process_results(current_result, sota_result)
            except KeyError as exc:
                important = [
                    "1day.excess_return_with_cost.annualized_return",
                    "1day.excess_return_with_cost.max_drawdown",
                    "1day.excess_return_with_cost.information_ratio",
                    "1day.excess_return_with_cost.annualized_turnover",
                    "1day.composite_score",
                ]
                try:
                    if isinstance(current_result, pd.Series) and isinstance(sota_result, pd.Series):
                        combined_df = pd.concat(
                            [current_result.rename("Current"), sota_result.rename("SOTA")], axis=1
                        )
                        fallback_result = combined_df.reindex(important)
                        return fallback_result
                except Exception:
                    pass
                raise

        feedback_mod.process_results = _safe_process_results  # type: ignore[method-assign]
    except Exception:
        pass


def apply_qlib_conda_env_patch() -> None:
    """Must run before any import of rdagent.scenarios.qlib.experiment.workspace."""
    # Windows has no select.poll; disable live stream mode in LocalEnv to avoid that code path.
    env_mod.LocalConf.live_output = False
    env_mod.CondaConf.live_output = False

    # ── Core fix: wrap __init__ on the ORIGINAL class objects ────────────────────────────
    # Pydantic v2 compiles validators into __pydantic_core_schema__ at class creation time.
    # Patching validator methods or model_post_init after class creation has NO effect.
    # BUT __init__ is a regular Python function (confirmed via testing) and CAN be wrapped.
    # We wrap __init__ so our code runs AFTER pydantic validation finishes, giving us the
    # last word on bin_path regardless of what `conda run` in change_bin_path produced.
    # This affects ALL instances created from these classes, even stale references in
    # workspace.py and other modules.
    for _cls in (env_mod.CondaConf, env_mod.QlibCondaConf):
        try:
            _orig_init = _cls.__init__

            def _make_wrapped_init(_orig):
                def _wrapped_init(self: Any, **kwargs: Any) -> None:
                    # RD-Agent calls CondaConf(conda_env_name=os.environ.get("CONDA_DEFAULT_ENV"))
                    # (see rdagent.components.coder.factor_coder.config.get_factor_env). When the
                    # worker process is launched via "export PATH=.../mamba/envs/rdagent/bin:$PATH"
                    # instead of "conda activate", CONDA_DEFAULT_ENV is unset and pydantic rejects
                    # conda_env_name=None before we ever get to override bin_path. Fill a harmless
                    # placeholder here — bin_path gets overridden below, so conda_env_name is never
                    # actually used by the patched environment.
                    if kwargs.get("conda_env_name") in (None, ""):
                        kwargs["conda_env_name"] = "rdagent"
                    _orig(self, **kwargs)
                    object.__setattr__(self, "bin_path", _RDAGENT_ENV_BIN)
                return _wrapped_init

            _cls.__init__ = _make_wrapped_init(_orig_init)  # type: ignore[method-assign]
        except Exception:
            pass

    # Prevent embedding-API crash from killing runs when a factor task first succeeds.
    _patch_embedding_graceful_fallback()
    # Level-2 fix: force UTF-8 read in inject_code_from_folder (Windows GBK crash).
    _patch_inject_code_utf8()
    # P3a: extend the LLM's reward signal BEFORE the feedback wrapper is installed.
    _patch_feedback_important_metrics()
    _patch_feedback_process_results()


def _patch_qlib_runner_env() -> None:
    """Also patch workspace.py's QlibCondaConf reference (belt-and-suspenders)."""
    try:
        import rdagent.scenarios.qlib.experiment.workspace as ws_mod
        ws_mod.QlibCondaConf = env_mod.QlibCondaConf  # type: ignore[attr-defined]
    except Exception:
        pass
