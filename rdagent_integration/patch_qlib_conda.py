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
    """
    try:
        import numpy as np
        from rdagent.oai.backend.base import APIBackend

        _EMBED_DIM = 1536  # standard OpenAI/DeepSeek-compatible dimension

        _original_create = APIBackend.create_embedding

        def _safe_create_embedding(self: APIBackend, input_content, **kwargs):  # type: ignore[override]
            try:
                return _original_create(self, input_content, **kwargs)
            except Exception:
                # Return a zero-vector (or list of zero-vectors) so callers don't crash.
                if isinstance(input_content, str):
                    return np.zeros(_EMBED_DIM, dtype=np.float32)
                return [np.zeros(_EMBED_DIM, dtype=np.float32) for _ in input_content]

        APIBackend.create_embedding = _safe_create_embedding  # type: ignore[method-assign]
    except Exception:
        pass  # Silently skip if rdagent internals change in a future version.


def _patch_feedback_process_results() -> None:
    """
    Wrap feedback.process_results to gracefully handle missing benchmark metrics.

    When PortAnaRecord succeeds but the SOTA result only has IC metrics (e.g. from a
    previous run before the benchmark fix), the concat/reindex in process_results raises
    a KeyError for 'annualized_return' and 'max_drawdown'. This patch catches that
    specific KeyError and returns a NaN-filled fallback DataFrame so the workflow
    continues instead of crashing.
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
                    _orig(self, **kwargs)
                    object.__setattr__(self, "bin_path", _RDAGENT_ENV_BIN)
                return _wrapped_init

            _cls.__init__ = _make_wrapped_init(_orig_init)  # type: ignore[method-assign]
        except Exception:
            pass

    # Prevent embedding-API crash from killing runs when a factor task first succeeds.
    _patch_embedding_graceful_fallback()
    _patch_feedback_process_results()


def _patch_qlib_runner_env() -> None:
    """Also patch workspace.py's QlibCondaConf reference (belt-and-suspenders)."""
    try:
        import rdagent.scenarios.qlib.experiment.workspace as ws_mod
        ws_mod.QlibCondaConf = env_mod.QlibCondaConf  # type: ignore[attr-defined]
    except Exception:
        pass
