"""Point RD-Agent's Qlib conda runs at env QLIB_RDAGENT_CONDA_ENV (default: rdagent)."""

from __future__ import annotations

import os
import pickle
import re
import uuid
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
                        # utf-8-sig strips a leading BOM (\ufeff) automatically,
                        # preventing GBK write failures on Windows when inject_files
                        # calls write_text() without an explicit encoding.
                        text = file_path.read_text(encoding="utf-8-sig")
                    except UnicodeDecodeError:
                        text = file_path.read_text(encoding="utf-8-sig", errors="replace")
                    self.inject_files(**{str(relative_path): text})

        _safe_inject_code_from_folder.__factor_lab_utf8_patched__ = True  # type: ignore[attr-defined]
        FBWorkspace.inject_code_from_folder = _safe_inject_code_from_folder  # type: ignore[method-assign]

        # Also patch inject_files so every workspace file is written UTF-8.
        # Without this, write_text(v) uses the system locale (GBK on Windows),
        # and qrun inside WSL fails to read them as UTF-8.
        _orig_inject = FBWorkspace.inject_files
        if not getattr(_orig_inject, "__factor_lab_utf8_write_patched__", False):
            def _utf8_inject_files(self, **files: str) -> None:  # type: ignore[override]
                self.prepare()
                for k, v in files.items():
                    target_file_path = self.workspace_path / k
                    if v == self.DEL_KEY:
                        if target_file_path.exists():
                            target_file_path.unlink()
                        self.file_dict.pop(k, None)
                    else:
                        self.file_dict[k] = v
                        target_file_path.parent.mkdir(parents=True, exist_ok=True)
                        target_file_path.write_text(v, encoding="utf-8")  # explicit UTF-8

            _utf8_inject_files.__factor_lab_utf8_write_patched__ = True  # type: ignore[attr-defined]
            FBWorkspace.inject_files = _utf8_inject_files  # type: ignore[method-assign]
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
                # Keep both legacy keys (IC, without_cost.*) and new project keys
                # so RD-Agent's feedback Jinja templates never crash on missing rows.
                important = [
                    "IC",
                    "1day.excess_return_without_cost.annualized_return",
                    "1day.excess_return_without_cost.max_drawdown",
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


def _patch_feedback_generate_feedback_safe_metrics() -> None:
    """Guard generate_feedback against missing IMPORTANT_METRICS rows.

    Upstream model/factor feedback uses ``exp.result.loc[IMPORTANT_METRICS]``
    and ``SOTA_experiment.result.loc[IMPORTANT_METRICS]`` directly, which crash
    whenever the metric naming schema changes (with_cost vs without_cost,
    composite field availability, IC naming, etc.) or when the experiment
    produces an empty/NaN result.

    We wrap both factor/model generate_feedback with a three-layer defense:

    1. Shrink ``IMPORTANT_METRICS`` to the intersection of keys actually
       present in both ``exp.result`` *and* every SOTA experiment's result.
    2. If no intersection exists, fall back to an empty list — pandas'
       ``.loc[[]]`` returns an empty Series (no KeyError), and the downstream
       Jinja template tolerates an empty rendering.
    3. If upstream still raises ``KeyError`` (e.g., because a template
       hard-indexes something else), temporarily set ``exp.result`` and all
       SOTA ``result`` attributes to ``None`` and retry so that upstream takes
       the ``"execution failed"`` placeholder branch.
    """
    try:
        import pandas as pd
        import rdagent.scenarios.qlib.developer.feedback as feedback_mod

        def _collect_sota_results(exp) -> list:
            sota_exp = getattr(exp, "based_experiments", None)
            if not isinstance(sota_exp, list):
                return []
            out = []
            for e in sota_exp:
                r = getattr(e, "result", None)
                if isinstance(r, pd.Series):
                    out.append((e, r))
            return out

        def _safe_metric_subset(exp) -> list[str]:
            base = list(getattr(feedback_mod, "IMPORTANT_METRICS", []))
            cur = getattr(exp, "result", None)

            if not isinstance(cur, pd.Series) or len(cur.index) == 0:
                # No current result at all — caller will get "execution failed"
                # via the None-result branch after we neutralize it below.
                return []

            cur_idx = set(cur.index.tolist())
            common = set(cur_idx)
            for _, r in _collect_sota_results(exp):
                common &= set(r.index.tolist())

            # Prefer keeping the canonical base order where possible.
            metrics = [m for m in base if m in common]
            if metrics:
                return metrics

            # No canonical metric available in ALL of current+SOTA.
            # Returning [] is intentional: pandas' .loc[[]] is safe and the
            # downstream template simply renders an empty block.
            return []

        def _neutralize_results(exp):
            """Temporarily blank out result fields so upstream hits the
            ``execution failed`` placeholder branch. Returns a callable that
            restores the original values."""
            saved: list[tuple[object, object]] = []
            try:
                saved.append((exp, getattr(exp, "result", None)))
                exp.result = None  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                for e, _ in _collect_sota_results(exp):
                    saved.append((e, getattr(e, "result", None)))
                    try:
                        e.result = None  # type: ignore[attr-defined]
                    except Exception:
                        pass
            except Exception:
                pass

            def _restore():
                for target, val in saved:
                    try:
                        target.result = val  # type: ignore[attr-defined]
                    except Exception:
                        pass

            return _restore

        def _wrap_generate_feedback(orig):
            def _wrapped(self, exp, trace):  # type: ignore[override]
                original_metrics = list(getattr(feedback_mod, "IMPORTANT_METRICS", []))
                try:
                    feedback_mod.IMPORTANT_METRICS = _safe_metric_subset(exp)
                    try:
                        return orig(self, exp, trace)
                    except KeyError:
                        # Template still hit a missing metric. Blank out all
                        # result fields and retry so upstream routes into the
                        # "execution failed" placeholder path. On the retry
                        # pass IMPORTANT_METRICS is restored to its canonical
                        # form so the placeholder renders consistently.
                        feedback_mod.IMPORTANT_METRICS = original_metrics
                        restore = _neutralize_results(exp)
                        try:
                            return orig(self, exp, trace)
                        finally:
                            restore()
                finally:
                    feedback_mod.IMPORTANT_METRICS = original_metrics

            return _wrapped

        factor_cls = getattr(feedback_mod, "QlibFactorExperiment2Feedback", None)
        model_cls = getattr(feedback_mod, "QlibModelExperiment2Feedback", None)
        if factor_cls is not None and hasattr(factor_cls, "generate_feedback"):
            if not getattr(factor_cls.generate_feedback, "__factor_lab_safe_metrics_patched__", False):
                wrapped = _wrap_generate_feedback(factor_cls.generate_feedback)
                setattr(wrapped, "__factor_lab_safe_metrics_patched__", True)
                factor_cls.generate_feedback = wrapped  # type: ignore[method-assign]
        if model_cls is not None and hasattr(model_cls, "generate_feedback"):
            if not getattr(model_cls.generate_feedback, "__factor_lab_safe_metrics_patched__", False):
                wrapped = _wrap_generate_feedback(model_cls.generate_feedback)
                setattr(wrapped, "__factor_lab_safe_metrics_patched__", True)
                model_cls.generate_feedback = wrapped  # type: ignore[method-assign]
    except Exception:
        pass


def _patch_env_python_entry() -> None:
    """Force RD-Agent ad-hoc code execution to use absolute python path.

    RD-Agent's Env.dump_python_code_run_and_get_results hard-codes
    ``python <tmp_file>.py``. On WSL machines where ``python`` is not in PATH
    (only ``python3`` exists), factor/model evaluator loops fail with
    ``/bin/sh: 1: python: not found``. We patch this method to call the
    rdagent env interpreter directly.
    """
    try:
        env_cls = env_mod.Env
        original = env_cls.dump_python_code_run_and_get_results
        if getattr(original, "__factor_lab_python_entry_patched__", False):
            return

        py_exec = f"{_RDAGENT_ENV_BIN}/python"

        def _safe_dump_python_code_run_and_get_results(
            self,
            code: str,
            dump_file_names: list[str],
            local_path: str,
            env: dict | None = None,
            running_extra_volume: Any = None,
            code_dump_file_py_name: str | None = None,
        ) -> tuple[str, list]:
            random_file_name = (
                f"{uuid.uuid4()}.py" if code_dump_file_py_name is None else f"{code_dump_file_py_name}.py"
            )
            with open(os.path.join(local_path, random_file_name), "w", encoding="utf-8") as f:
                f.write(code)

            entry = f"{py_exec} {random_file_name}"
            extra_volume = {} if running_extra_volume is None else dict(running_extra_volume)
            log_output = self.check_output(entry, local_path, env, running_extra_volume=extra_volume)

            results = []
            os.remove(os.path.join(local_path, random_file_name))
            for name in dump_file_names:
                dump_path = os.path.join(local_path, f"{name}")
                if os.path.exists(dump_path):
                    with open(dump_path, "rb") as fh:
                        results.append(pickle.load(fh))
                    os.remove(dump_path)
                else:
                    return log_output, []
            return log_output, results

        _safe_dump_python_code_run_and_get_results.__factor_lab_python_entry_patched__ = True  # type: ignore[attr-defined]
        env_cls.dump_python_code_run_and_get_results = _safe_dump_python_code_run_and_get_results  # type: ignore[method-assign]
    except Exception:
        pass


def _patch_local_env_bin_path() -> None:
    """Ensure LocalEnv execution PATH contains a resolvable `python` binary."""
    try:
        current = getattr(env_mod.LocalConf, "bin_path", "") or ""
        parts = [p for p in current.split(":") if p]
        for must in (_RDAGENT_ENV_BIN, "/usr/bin"):
            if must not in parts:
                parts.insert(0, must)
        env_mod.LocalConf.bin_path = ":".join(parts)
    except Exception:
        pass


def _patch_local_env_entry_python() -> None:
    """Rewrite shell entry `python ...` to absolute interpreter for LocalEnv."""
    try:
        local_run = env_mod.LocalEnv._run
        if getattr(local_run, "__factor_lab_local_entry_patched__", False):
            return

        py_exec = f"{_RDAGENT_ENV_BIN}/python"
        qrun_exec = f"{_RDAGENT_ENV_BIN}/qrun"
        py_pattern = re.compile(r"(^|\s)python(\s+)")
        qrun_pattern = re.compile(r"(^|\s)qrun(\s+)")

        def _rewrite_entry(entry: str) -> str:
            entry = py_pattern.sub(lambda m: f"{m.group(1)}{py_exec}{m.group(2)}", entry, count=1)
            entry = qrun_pattern.sub(lambda m: f"{m.group(1)}{qrun_exec}{m.group(2)}", entry, count=1)
            return entry

        def _win_path_to_wsl(win_path: str) -> str:
            """Convert a Windows path like D:\\foo\\bar to /mnt/d/foo/bar."""
            import re as _re
            p = win_path.replace("\\", "/")
            p = _re.sub(r"^([A-Za-z]):/", lambda m: f"/mnt/{m.group(1).lower()}/", p)
            return p

        def _wrapped_run(self, entry=None, local_path=None, env=None, running_extra_volume=None, **kwargs):  # type: ignore[override]
            import json as _json, platform as _pl, time as _t, subprocess as _sp, os as _os
            from pathlib import Path as _Path
            from rich.console import Console as _Console
            from rich.rule import Rule as _Rule
            from rich.table import Table as _Table

            resolved = entry if entry is not None else getattr(self.conf, "default_entry", None)

            # #region agent log fc2594
            open("debug-fc2594.log", "a", encoding="utf-8").write(_json.dumps({
                "sessionId": "fc2594", "timestamp": int(_t.time()*1000),
                "hypothesisId": "H-D", "location": "patch_qlib_conda.py:_wrapped_run",
                "message": "local_env_run_called",
                "data": {"entry": str(resolved), "cwd": str(local_path), "platform": _pl.system()}
            }) + "\n")
            # #endregion

            if isinstance(resolved, str) and ("python " in resolved or " qrun " in resolved or resolved.lstrip().startswith("qrun ")):
                resolved = _rewrite_entry(resolved)

            # ── WSL execution path ──────────────────────────────────────────────────
            # On Windows, /bin/sh commands must run inside WSL.
            # We bypass local_run entirely because Popen(shell=True) + WSL returns
            # err=None from communicate(), crashing LocalEnv._run.
            if (
                _pl.system() == "Windows"
                and isinstance(resolved, str)
                and resolved.startswith("/bin/sh")
                and local_path is not None
            ):
                wsl_cwd = _win_path_to_wsl(str(local_path))

                # Build volume symlink commands for WSL (/tmp/full → wsl-mapped path)
                vol_cmds: list[str] = []
                for lp, rp in (running_extra_volume or {}).items():
                    wsl_real = _win_path_to_wsl(str(rp))
                    vol_cmds.append(f"rm -f {lp} && mkdir -p $(dirname {lp}) && ln -sfn {wsl_real} {lp}")
                extra_vols = getattr(self.conf, "extra_volumes", None) or {}
                if extra_vols:
                    cache_key = "/tmp/sample" if any("/sample/" in k for k in extra_vols) else "/tmp/full"
                    wsl_cache_target = _win_path_to_wsl(str(local_path)) + "/workspace_cache"
                    vol_cmds.append(f"rm -f {cache_key} && mkdir -p $(dirname {cache_key}) && ln -sfn {wsl_cache_target} {cache_key}")

                vol_setup = " && ".join(vol_cmds) + " && " if vol_cmds else ""
                # Prepend PYTHONPATH so AI-generated model.py in workspace is importable
                bash_cmd = f"{vol_setup}cd {wsl_cwd} && export PYTHONPATH={wsl_cwd}:$PYTHONPATH && {resolved}"

                # #region agent log fc2594
                open("debug-fc2594.log", "a", encoding="utf-8").write(_json.dumps({
                    "sessionId": "fc2594", "timestamp": int(_t.time()*1000),
                    "hypothesisId": "H-WSL2", "location": "patch_qlib_conda.py:wsl_direct",
                    "message": "wsl_direct_exec",
                    "data": {"bash_cmd": bash_cmd[:400], "wsl_cwd": wsl_cwd}
                }) + "\n")
                # #endregion

                _con = _Console()
                _con.print(_Rule("[bold green]LocalEnv Logs Begin[/bold green]", style="dark_orange"))
                _tbl = _Table(title="Run Info", show_header=False)
                _tbl.add_column("Key", style="bold cyan")
                _tbl.add_column("Value", style="bold magenta")
                _tbl.add_row("Entry (WSL)", bash_cmd[:200])
                _tbl.add_row("Local Path", str(local_path))
                _con.print(_tbl)

                proc = _sp.Popen(
                    [r"C:\Windows\System32\wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc", bash_cmd],
                    cwd=str(local_path),
                    stdout=_sp.PIPE,
                    stderr=_sp.STDOUT,   # merge stderr→stdout to avoid err=None
                    encoding="utf-8",    # WSL outputs UTF-8, not GBK
                    errors="replace",    # don't crash on un-decodable bytes
                    env=_os.environ.copy(),
                )
                out, _ = proc.communicate()
                out = out or ""
                _con.print(out, end="", markup=False)
                _con.print(_Rule("[bold green]LocalEnv Logs End[/bold green]", style="dark_orange"))

                # #region agent log fc2594
                open("debug-fc2594.log", "a", encoding="utf-8").write(_json.dumps({
                    "sessionId": "fc2594", "timestamp": int(_t.time()*1000),
                    "hypothesisId": "H-WSL2", "location": "patch_qlib_conda.py:wsl_direct",
                    "message": "wsl_direct_returned",
                    "data": {"return_code": proc.returncode, "stdout_tail": out[-600:]}
                }) + "\n")
                # #endregion

                return out, proc.returncode
            # ── end WSL path ────────────────────────────────────────────────────────

            result = local_run(
                self,
                entry=resolved,
                local_path=local_path,
                env=env,
                running_extra_volume=running_extra_volume if running_extra_volume is not None else {},
                **kwargs,
            )
            # #region agent log fc2594
            _rc  = result[1] if isinstance(result, tuple) else getattr(result, "return_code", "?")
            _out = result[0] if isinstance(result, tuple) else getattr(result, "stdout", "")
            open("debug-fc2594.log", "a", encoding="utf-8").write(_json.dumps({
                "sessionId": "fc2594", "timestamp": int(_t.time()*1000),
                "hypothesisId": "H-I", "location": "patch_qlib_conda.py:_wrapped_run",
                "message": "local_run_returned",
                "data": {
                    "entry_snippet": str(resolved)[:120],
                    "return_code": _rc,
                    "stdout_tail": str(_out)[-400:],
                }
            }) + "\n")
            # #endregion
            return result

        _wrapped_run.__factor_lab_local_entry_patched__ = True  # type: ignore[attr-defined]
        env_mod.LocalEnv._run = _wrapped_run  # type: ignore[method-assign]
    except Exception:
        pass


def _patch_qlib_conda_env_prepare_noop() -> None:
    """Short-circuit ``QlibCondaEnv.prepare`` so it never runs ``conda create``.

    Root cause (observed repeatedly in ``logs/live_loop/wsl_level2_loop5_retry*``):
    ``QlibCondaEnv.prepare()`` executes ``subprocess.run("conda env list",
    shell=True, ...)``. The shell spawned by ``shell=True`` is ``/bin/sh`` with a
    minimal ``PATH=/usr/bin:/bin`` — it does **not** inherit the caller's
    ``~/.local/bin`` or the ``micromamba envs`` bin directory. So ``conda`` is
    ``not found`` (exit 127) and the prepare code wrongly concludes the conda
    env is missing. It then tries ``conda create -y -n rdagent4qlib ...`` which
    fails with exit 127 again (``/usr/bin/env: 'bash': No such file or directory``
    comes from ``conda``'s own shebang resolution in that reduced PATH). The
    outer ``try/except`` swallows the error silently, the backtest never runs,
    and the Loop ends with ``exit_code=0`` but no ``Loop_*/running`` directory.

    We already ship an ``rdagent`` micromamba env with pyqlib/pandas/torch
    preinstalled, so the whole ``prepare`` step is redundant. Replace it with
    a no-op.
    """
    try:
        qlib_conda_env_cls = getattr(env_mod, "QlibCondaEnv", None)
        if qlib_conda_env_cls is None:
            return
        if getattr(qlib_conda_env_cls.prepare, "__factor_lab_noop_patched__", False):
            return

        def _noop_prepare(self) -> None:  # type: ignore[override]
            return None

        _noop_prepare.__factor_lab_noop_patched__ = True  # type: ignore[attr-defined]
        qlib_conda_env_cls.prepare = _noop_prepare  # type: ignore[method-assign]
    except Exception:
        pass


def _patch_factor_costeer_python_bin() -> None:
    """Force ``FACTOR_COSTEER_SETTINGS.python_bin`` to the absolute rdagent python.

    In ``rdagent/components/coder/factor_coder/factor.py`` the evaluator runs

        subprocess.check_output(
            f"{FACTOR_COSTEER_SETTINGS.python_bin} {execution_code_path}",
            shell=True, cwd=self.workspace_path, ...
        )

    with the default ``python_bin = "python"``. As above, ``/bin/sh`` launched
    by ``shell=True`` cannot resolve ``python`` from the user's PATH, so every
    factor evaluation fails with ``/bin/sh: 1: python: not found`` and CoSTEER
    reports ``Final decisions: [False, False]`` for the whole loop.

    Pin to the absolute path of the rdagent env's python (this env has qlib +
    pandas + torch installed, verified in :mod:`logs/live_loop/_env_check.sh`).
    """
    try:
        from rdagent.components.coder.factor_coder.config import (
            FACTOR_COSTEER_SETTINGS,
        )

        abs_python = f"{_RDAGENT_ENV_BIN}/python"
        if os.path.exists(abs_python):
            try:
                object.__setattr__(FACTOR_COSTEER_SETTINGS, "python_bin", abs_python)
            except Exception:
                FACTOR_COSTEER_SETTINGS.python_bin = abs_python  # type: ignore[attr-defined]
    except Exception:
        pass


def _patch_model_costeer_python_bin() -> None:
    """Mirror of :func:`_patch_factor_costeer_python_bin` for the model coder."""
    try:
        from rdagent.components.coder.model_coder.conf import (
            MODEL_COSTEER_SETTINGS,
        )

        abs_python = f"{_RDAGENT_ENV_BIN}/python"
        if os.path.exists(abs_python):
            try:
                object.__setattr__(MODEL_COSTEER_SETTINGS, "python_bin", abs_python)
            except Exception:
                MODEL_COSTEER_SETTINGS.python_bin = abs_python  # type: ignore[attr-defined]
    except Exception:
        pass


def cap_n_epochs_value(raw_n_epochs: Any, cap: int) -> Any:
    """Public helper (used by unit tests).

    Parameters
    ----------
    raw_n_epochs:
        Whatever the LLM put in ``training_hyperparameters["n_epochs"]``
        (typically ``"100"`` — a string, since RD-Agent serialises training
        hyperparameters through env vars to Jinja2).
    cap:
        Upper bound. Values ``<= 0`` disable clamping (returns input).

    Returns
    -------
    Clamped value (always coerced to ``str`` when clamped, to match
    the storage format RD-Agent's ``model_runner.develop`` expects).
    Inputs that can't be parsed as ``int`` or are already below the cap
    are returned untouched.
    """
    if cap <= 0:
        return raw_n_epochs
    if raw_n_epochs is None:
        return raw_n_epochs
    try:
        n = int(str(raw_n_epochs))
    except (TypeError, ValueError):
        return raw_n_epochs
    if n > cap:
        return str(cap)
    return raw_n_epochs


def _patch_cap_n_epochs() -> None:
    """Clamp ``training_hyperparameters['n_epochs']`` on every model experiment
    so LLM-proposed GRU/MLP/Transformer tasks can actually finish within
    RD-Agent's 3600s qrun hard-timeout.

    Background
    ----------
    LLM (DeepSeek / GLM-5.1 / OpenAI) tends to propose ``n_epochs=100`` by
    default (it literally copies the example shown in RD-Agent's
    ``prompts.yaml``). On this project's CSI300 2022–2025 slice, 100 epochs
    of a 2-layer GRU runs **~60 minutes or more**, and ~100% of the time
    hits ``timeout --kill-after=10 3600``, producing a neutralised
    (``nan``) feedback that the LLM then (a) can't learn from and (b)
    wastes the next round re-proposing similar hyperparameters.

    The fix
    -------
    Monkey-patch ``QlibModelRunner.develop`` and ``QlibFactorRunner.develop``
    to clamp ``training_hyperparameters['n_epochs']`` to
    ``FACTOR_LAB_MAX_N_EPOCHS`` right before the qrun subprocess is
    spawned. Both runner classes resolve ``n_epochs`` identically —
    see ``rdagent/scenarios/qlib/developer/model_runner.py:L67`` and
    ``rdagent/scenarios/qlib/developer/factor_runner.py:L144``.

    Activation
    ----------
    Set ``FACTOR_LAB_MAX_N_EPOCHS=<int>`` in the environment. Unset or
    ``<=0`` → no-op (full LLM autonomy preserved). Typical value for this
    project: ``20`` (finishes in ~15 min, well under 3600s timeout, while
    still giving the neural net enough iterations to converge on the
    2022–2024 train/valid split).
    """
    try:
        raw_cap = os.environ.get("FACTOR_LAB_MAX_N_EPOCHS", "").strip()
        if not raw_cap:
            return
        try:
            cap = int(raw_cap)
        except (TypeError, ValueError):
            return
        if cap <= 0:
            return

        def _clamp_dict(th: dict[str, Any] | None) -> None:
            if not th:
                return
            raw = th.get("n_epochs")
            if raw is None:
                return
            new_val = cap_n_epochs_value(raw, cap)
            if new_val != raw:
                th["n_epochs"] = new_val

        def _clamp_exp(exp: Any) -> None:
            try:
                sub_tasks = getattr(exp, "sub_tasks", None) or []
                for task in sub_tasks:
                    th = getattr(task, "training_hyperparameters", None)
                    if isinstance(th, dict):
                        _clamp_dict(th)
            except Exception:
                pass

        try:
            import rdagent.scenarios.qlib.developer.model_runner as mr_mod

            _orig_model_develop = mr_mod.QlibModelRunner.develop

            def _model_develop_capped(self: Any, exp: Any) -> Any:
                _clamp_exp(exp)
                return _orig_model_develop(self, exp)

            mr_mod.QlibModelRunner.develop = _model_develop_capped  # type: ignore[method-assign]
        except Exception:
            pass

        try:
            import rdagent.scenarios.qlib.developer.factor_runner as fr_mod

            _orig_factor_develop = fr_mod.QlibFactorRunner.develop

            def _factor_develop_capped(self: Any, exp: Any) -> Any:
                try:
                    trace = getattr(self, "trace", None)
                    hist = getattr(trace, "hist", None) if trace is not None else None
                    if hist:
                        for past_exp, _fb in hist:
                            _clamp_exp(past_exp)
                except Exception:
                    pass
                return _orig_factor_develop(self, exp)

            fr_mod.QlibFactorRunner.develop = _factor_develop_capped  # type: ignore[method-assign]
        except Exception:
            pass
    except Exception:
        pass


def apply_qlib_conda_env_patch() -> None:
    """Must run before any import of rdagent.scenarios.qlib.experiment.workspace."""
    # Windows has no select.poll; disable live stream mode in LocalEnv to avoid that code path.
    env_mod.LocalConf.live_output = False
    env_mod.CondaConf.live_output = False
    _patch_local_env_bin_path()
    _patch_local_env_entry_python()

    # ── Core fix: wrap __init__ on the ORIGINAL class objects ────────────────────────────
    # Pydantic v2 compiles validators into __pydantic_core_schema__ at class creation time.
    # Patching validator methods or model_post_init after class creation has NO effect.
    # BUT __init__ is a regular Python function (confirmed via testing) and CAN be wrapped.
    # We wrap __init__ so our code runs AFTER pydantic validation finishes, giving us the
    # last word on bin_path regardless of what `conda run` in change_bin_path produced.
    # This affects ALL instances created from these classes, even stale references in
    # workspace.py and other modules.
    #
    # IMPORTANT: we wrap both CondaConf and QlibCondaConf because they are separate
    # Pydantic classes. But we must NOT force ``conda_env_name="rdagent"`` on the
    # QlibCondaConf path (previous iteration of this patch did — and it silently
    # mapped L3 qrun onto the fully-populated ``rdagent`` env which works, but it
    # also masked the real underlying bug where QlibCondaEnv.prepare tries to run
    # ``conda create`` in a sh-with-minimal-PATH context, exits 127, and then
    # silently skips the backtest. We now short-circuit prepare() entirely and
    # pin bin_path instead, so we can leave QlibCondaConf.conda_env_name alone
    # for cosmetic accuracy in logs.)
    try:
        _orig_cond_init = env_mod.CondaConf.__init__

        def _wrapped_cond_init(self: Any, **kwargs: Any) -> None:
            if kwargs.get("conda_env_name") in (None, ""):
                kwargs["conda_env_name"] = "rdagent"
            _orig_cond_init(self, **kwargs)
            object.__setattr__(self, "bin_path", _RDAGENT_ENV_BIN)

        env_mod.CondaConf.__init__ = _wrapped_cond_init  # type: ignore[method-assign]
    except Exception:
        pass

    try:
        _orig_qcond_init = env_mod.QlibCondaConf.__init__

        def _wrapped_qcond_init(self: Any, **kwargs: Any) -> None:
            _orig_qcond_init(self, **kwargs)
            # Always override bin_path to point at the fully-populated rdagent env.
            # Leave conda_env_name to whatever pydantic resolved (default
            # ``rdagent4qlib``) so logs stay accurate — prepare() is a no-op anyway.
            object.__setattr__(self, "bin_path", _RDAGENT_ENV_BIN)

        env_mod.QlibCondaConf.__init__ = _wrapped_qcond_init  # type: ignore[method-assign]
    except Exception:
        pass

    # ── Core fix #2: neuter QlibCondaEnv.prepare so it can't mis-trigger conda create.
    _patch_qlib_conda_env_prepare_noop()

    # ── Core fix #3: pin CoSTEER's python_bin to the rdagent env absolute path,
    # otherwise every ``subprocess.check_output(f"python <file>.py", shell=True)``
    # in factor/model coders fails with "python: not found".
    _patch_factor_costeer_python_bin()
    _patch_model_costeer_python_bin()

    # Prevent embedding-API crash from killing runs when a factor task first succeeds.
    _patch_embedding_graceful_fallback()
    _patch_env_python_entry()
    # Level-2 fix: force UTF-8 read in inject_code_from_folder (Windows GBK crash).
    _patch_inject_code_utf8()
    # P3a: extend the LLM's reward signal BEFORE the feedback wrapper is installed.
    _patch_feedback_important_metrics()
    _patch_feedback_process_results()
    _patch_feedback_generate_feedback_safe_metrics()
    # Stage I: cap LLM-proposed model training n_epochs so GRU qrun never hits
    # RD-Agent's 3600s hard-timeout (opt-in via FACTOR_LAB_MAX_N_EPOCHS).
    _patch_cap_n_epochs()
    # Cap CoSTEER evo loops so coding step stays under ~20 min.
    _patch_costeer_max_loop()
    # Windows/Linux cross-platform: remap PosixPath in pkl cache → PurePosixPath.
    _patch_pickle_cache_posixpath()
    # numpy 2.x compat: ret.pkl from WSL uses numpy._core; Windows may have numpy 1.x.
    _patch_workspace_numpy_pkl_compat()


def _patch_pickle_cache_posixpath() -> None:
    """Patch rdagent's cache_wrapper so PosixPath pkl files created in WSL/Linux
    can be loaded transparently on Windows by remapping PosixPath → PurePosixPath.

    Root cause: pickle cache files written in WSL embed pathlib.PosixPath objects;
    Windows Python cannot instantiate PosixPath, raising NotImplementedError.
    Fix: replace pickle.load in rdagent.core.utils with a safe loader that falls
    back to a custom Unpickler mapping PosixPath → PurePosixPath on failure.
    """
    try:
        import io
        import pathlib
        import pickle as _pickle
        import rdagent.core.utils as _rdu

        class _PosixSafeUnpickler(_pickle.Unpickler):
            """Replace PosixPath with PurePosixPath so Windows can load Linux pkl files."""
            def find_class(self, module, name):
                if module == "pathlib" and name == "PosixPath":
                    return pathlib.PurePosixPath
                return super().find_class(module, name)

        def _safe_load(f):
            data = f.read()
            try:
                return _pickle.loads(data)
            except NotImplementedError:
                return _PosixSafeUnpickler(io.BytesIO(data)).load()

        _rdu_pickle = _pickle

        class _SafePickleModule:
            @staticmethod
            def load(f):
                return _safe_load(f)

            @staticmethod
            def dump(obj, f, *a, **kw):
                return _rdu_pickle.dump(obj, f, *a, **kw)

            @staticmethod
            def dumps(obj, *a, **kw):
                return _rdu_pickle.dumps(obj, *a, **kw)

            @staticmethod
            def loads(data, *a, **kw):
                return _rdu_pickle.loads(data, *a, **kw)

        _rdu.pickle = _SafePickleModule  # type: ignore[attr-defined]

    except Exception as _e:
        import logging
        logging.getLogger(__name__).warning("_patch_pickle_cache_posixpath failed: %s", _e)


def _patch_costeer_max_loop() -> None:
    """Cap CoSTEER evo-loop count via env FACTOR_LAB_COSTEER_MAX_LOOP (default 4).

    The CoSTEER default is max_loop=10 — 10 evolution rounds × 3 factors ×
    execution time = 90+ minutes per coding step. With our large daily_pv.h5
    (181 MB / 6.8 M rows) each execution takes ~60 s, pushing the coding step
    to 90 min. Lowering max_loop to 4 keeps coding under ~25 minutes.
    """
    try:
        import os
        from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
        limit = int(os.environ.get("FACTOR_LAB_COSTEER_MAX_LOOP", "4"))
        object.__setattr__(FACTOR_COSTEER_SETTINGS, "max_loop", limit)
    except Exception:
        pass


def _patch_qlib_runner_env() -> None:
    """Also patch workspace.py's QlibCondaConf reference (belt-and-suspenders)."""
    try:
        import rdagent.scenarios.qlib.experiment.workspace as ws_mod
        ws_mod.QlibCondaConf = env_mod.QlibCondaConf  # type: ignore[attr-defined]
    except Exception:
        pass


def _patch_workspace_numpy_pkl_compat() -> None:
    """Patch workspace.execute so ret.pkl written by WSL numpy 2.x can be
    read by a Windows environment with numpy 1.x (or different 2.x build).

    Root cause: numpy 2.x moved internals to numpy._core.*; when the pkl is
    loaded on a system where only numpy.core.* exists the unpickler raises
    ModuleNotFoundError: No module named 'numpy._core.numeric'.
    Fix: wrap pd.read_pickle with a custom unpickler that remaps the module
    path before instantiating objects.
    """
    try:
        import io
        import pickle as _pkl
        import rdagent.scenarios.qlib.experiment.workspace as ws_mod

        _NUMPY_REMAP = {
            "numpy._core.numeric": "numpy.core.numeric",
            "numpy._core.multiarray": "numpy.core.multiarray",
            "numpy._core.umath": "numpy.core.umath",
            "numpy._core.fromnumeric": "numpy.core.fromnumeric",
            "numpy._core._methods": "numpy.core._methods",
            "numpy._core.arrayprint": "numpy.core.arrayprint",
        }
        # Reverse map for numpy 1.x → 2.x
        _NUMPY_REMAP.update({v: k for k, v in _NUMPY_REMAP.items()})

        class _NumpyCompatUnpickler(_pkl.Unpickler):
            def find_class(self, module, name):
                module = _NUMPY_REMAP.get(module, module)
                # Also handle pathlib cross-platform
                import pathlib
                if module == "pathlib" and name == "PosixPath":
                    return pathlib.PurePosixPath
                return super().find_class(module, name)

        def _compat_read_pickle(path):
            with open(path, "rb") as f:
                data = f.read()
            try:
                return _pkl.loads(data)
            except Exception:
                return _NumpyCompatUnpickler(io.BytesIO(data)).load()

        orig_execute = ws_mod.QlibFBWorkspace.execute
        if getattr(orig_execute, "__numpy_compat_patched__", False):
            return

        import pandas as _pd

        def _patched_execute(self, *args, **kwargs):
            orig_read_pickle = _pd.read_pickle
            _pd.read_pickle = _compat_read_pickle  # type: ignore[assignment]
            try:
                return orig_execute(self, *args, **kwargs)
            finally:
                _pd.read_pickle = orig_read_pickle  # type: ignore[assignment]

        _patched_execute.__numpy_compat_patched__ = True  # type: ignore[attr-defined]
        ws_mod.QlibFBWorkspace.execute = _patched_execute  # type: ignore[method-assign]
    except Exception as _e:
        import logging
        logging.getLogger(__name__).warning("_patch_workspace_numpy_pkl_compat failed: %s", _e)
