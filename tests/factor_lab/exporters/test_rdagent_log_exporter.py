"""
D.4 C1 exporter 单测：``factor_lab.exporters.rdagent_log_exporter``。

通过在 ``tmp_path`` 下构造 **假的 RD-Agent log 目录树 + 假的 workspace**（全部用
``SimpleNamespace`` + ``pickle.dump`` 写盘）来覆盖：

* 正常路径：单 loop / 单 task / 全 pkl 齐全 → 成功导 1 个 C1 package
* 多 loop：2 个 Loop_N 都能导
* 幂等：第二次跑 ``skipped_idempotent``
* overwrite=True 强制覆盖
* 缺 hypothesis pkl → placeholder 回退
* 缺 workspace_path / factor.py / result.h5 → 各自 skipped
* decision=False → 仍然导出，lab_metrics.rdagent_self_eval=0.0
* factor_name 含非法字符 → sanitize
* 多 evo_loop → 取最大 index
* export_rdagent_log_tree 扫多 run
* CLI 端到端（``python -m scripts.lab.export_rdagent_candidates``）
"""

from __future__ import annotations

import json
import pickle
import runpy
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from factor_lab.exporters.rdagent_log_exporter import (
    ExportRunSummary,
    _read_result_h5,
    _sanitize_name,
    export_rdagent_log_tree,
    export_rdagent_run,
    scan_loops,
)


# --------------------------------------------------------------------------- fixtures / builders


def _make_hdf_result(
    ws_dir: Path,
    factor_name: str,
    *,
    start: str = "2025-01-02",
    end: str = "2025-01-10",
    instruments: tuple[str, ...] = ("SH600000", "SH600001"),
) -> pd.DataFrame:
    """写一个符合 C1 schema 的 result.h5（key='data', MultiIndex, 单列 float64）。"""
    idx = pd.MultiIndex.from_product(
        [pd.date_range(start, end, freq="B"), list(instruments)],
        names=["datetime", "instrument"],
    )
    df = pd.DataFrame(
        {factor_name: [0.1 * (i % 7) for i in range(len(idx))]},
        index=idx,
        dtype="float64",
    )
    df.to_hdf(str(ws_dir / "result.h5"), key="data", mode="w", format="table")
    return df


def _make_factor_py(ws_dir: Path, factor_name: str, *, marker: str = "") -> str:
    src = (
        '"""Fake RD-Agent factor for tests."""\n'
        "import pandas as pd\n"
        f"def calculate_{factor_name}():\n"
        f"    # marker={marker}\n"
        "    df = pd.read_hdf('daily_pv.h5')\n"
        "    return df\n"
    )
    (ws_dir / "factor.py").write_text(src, encoding="utf-8")
    return src


def _pickle_at(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def _write_loop(
    run_dir: Path,
    *,
    loop_index: int,
    hypothesis: dict | None,
    tasks: list[dict],
    workspaces: list[dict],  # 每项必含 ws_dir (Path) + target_task_name
    feedback_decision: bool | None,
    evo_loop_indices: tuple[int, ...] = (0,),
) -> Path:
    """在 ``run_dir/Loop_N`` 下铺完整 pkl 目录树，返回该 Loop 目录。"""
    loop_dir = run_dir / f"Loop_{loop_index}"
    # hypothesis
    if hypothesis is not None:
        hypo_obj = SimpleNamespace(**hypothesis)
        _pickle_at(
            loop_dir / "direct_exp_gen" / "hypothesis generation" / "123" / "1.pkl",
            hypo_obj,
        )
    # experiment generation
    task_objs = [SimpleNamespace(**t) for t in tasks]
    _pickle_at(
        loop_dir / "direct_exp_gen" / "experiment generation" / "123" / "1.pkl",
        task_objs,
    )
    # coding evo_loop_N — 写多个 evo_loop，仅最大 index 的应被读取
    ws_objs = []
    for ws in workspaces:
        target_name = ws["target_task_name"]
        ws_ns = SimpleNamespace(
            target_task=SimpleNamespace(factor_name=target_name),
            workspace_path=ws.get("workspace_path"),
            file_dict=ws.get("file_dict", {}),
        )
        ws_objs.append(ws_ns)
    for evo_idx in evo_loop_indices:
        _pickle_at(
            loop_dir / "coding" / f"evo_loop_{evo_idx}" / "evolving code" / "123" / "1.pkl",
            ws_objs,
        )
    # feedback
    if feedback_decision is not None:
        fb_obj = SimpleNamespace(decision=feedback_decision)
        _pickle_at(loop_dir / "feedback" / "feedback" / "123" / "1.pkl", fb_obj)
    return loop_dir


# --------------------------------------------------------------------------- helpers: 全景场景


def _build_minimal_run(
    tmp_path: Path,
    *,
    run_name: str = "2026-04-20_01-19-42-374811",
    loop_index: int = 0,
    factor_name: str = "QualPersist_60D",
    ws_hash: str = "abcdef1234567890deadbeef",
    decision: bool | None = True,
) -> tuple[Path, Path, Path, Path]:
    """构造一份 "完美" 的 RD-Agent run 目录 + workspace，返回关键路径。"""
    log_root = tmp_path / "log"
    run_dir = log_root / run_name
    rdagent_ws_root = tmp_path / "git_ignore_folder" / "RD-Agent_workspace"
    ws_dir = rdagent_ws_root / ws_hash
    ws_dir.mkdir(parents=True, exist_ok=True)

    _make_factor_py(ws_dir, factor_name, marker="orig")
    _make_hdf_result(ws_dir, factor_name)

    _write_loop(
        run_dir,
        loop_index=loop_index,
        hypothesis={
            "hypothesis": "Quality factor persists over 60D horizon.",
            "reason": "Documented in literature.",
        },
        tasks=[
            {
                "factor_name": factor_name,
                "factor_formulation": "rank(rolling_mean($roe, 60))",
                "factor_description": "Rolling quality",
                "variables": {"$roe": "return on equity"},
            }
        ],
        workspaces=[
            {
                "target_task_name": factor_name,
                "workspace_path": str(ws_dir),
            }
        ],
        feedback_decision=decision,
    )

    cand_dir = tmp_path / "factor_lab" / "workspace" / "candidates"
    return log_root, run_dir, rdagent_ws_root, cand_dir


# =========================================================================== 基础单元


class TestSanitize:
    def test_clean_name_passthrough(self):
        assert _sanitize_name("QualPersist_60D") == "QualPersist_60D"

    def test_replace_illegal_chars(self):
        assert _sanitize_name("Vol-Ret.5D") == "Vol_Ret_5D"

    def test_prepend_F_when_not_alpha(self):
        assert _sanitize_name("60DMomentum").startswith("F_")

    def test_truncate_to_64(self):
        n = _sanitize_name("A" * 200)
        assert len(n) <= 64

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _sanitize_name("")


class TestReadResultH5:
    def test_reads_and_renames(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        _make_hdf_result(ws, "X_10D")
        df = _read_result_h5(ws / "result.h5", expected_name="X_10D")
        assert list(df.columns) == ["X_10D"]
        assert list(df.index.names) == ["datetime", "instrument"]
        assert str(df["X_10D"].dtype) == "float64"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ValueError):
            _read_result_h5(tmp_path / "nope.h5", expected_name="X")


# =========================================================================== 扫描 + 导出


class TestExportHappyPath:
    def test_single_loop_single_factor(self, tmp_path):
        log_root, run_dir, ws_root, cand_dir = _build_minimal_run(tmp_path)
        summary = export_rdagent_run(
            log_run_dir=run_dir,
            workspace_candidates_dir=cand_dir,
            rdagent_workspace_root=ws_root,
            now=datetime(2026, 4, 20, tzinfo=timezone.utc),
        )
        assert summary.loops_scanned == 1
        assert len(summary.exports) == 1
        assert summary.skipped == []

        exp = summary.exports[0]
        assert exp.name == "QualPersist_60D"
        assert exp.factor_id.startswith("rdagent_QualPersist_60D_")
        assert exp.action == "written"
        assert exp.decision is True

        # 物料齐全
        assert (exp.target_dir / "factor.py").exists()
        assert (exp.target_dir / "values.parquet").exists()
        assert exp.package_path.exists()

        # c1.json schema 可往返
        payload = json.loads(exp.package_path.read_text(encoding="utf-8"))
        assert payload["source"] == "rdagent"
        assert payload["parent_loop"] == 0
        assert payload["lab_metrics"]["rdagent_self_eval"] == 1.0
        assert payload["lab_run_id"] == run_dir.name
        assert "_meta" in payload
        assert payload["_meta"]["rdagent_workspace_hash"].startswith("abcdef")

    def test_two_loops_two_factors(self, tmp_path):
        log_root = tmp_path / "log"
        run_dir = log_root / "run-xyz-2026-04-20"
        ws_root = tmp_path / "git_ignore_folder" / "RD-Agent_workspace"

        # Loop 0: FactorA
        ws_a = ws_root / "aaaaaaaaaaaaaaaaaaaaaaaa"
        ws_a.mkdir(parents=True)
        _make_factor_py(ws_a, "FactorA_5D")
        _make_hdf_result(ws_a, "FactorA_5D")
        _write_loop(
            run_dir,
            loop_index=0,
            hypothesis={"hypothesis": "Short horizon", "reason": "reason A"},
            tasks=[{"factor_name": "FactorA_5D", "factor_formulation": "A()"}],
            workspaces=[{"target_task_name": "FactorA_5D", "workspace_path": str(ws_a)}],
            feedback_decision=True,
        )

        # Loop 1: FactorB
        ws_b = ws_root / "bbbbbbbbbbbbbbbbbbbbbbbb"
        ws_b.mkdir(parents=True)
        _make_factor_py(ws_b, "FactorB_20D")
        _make_hdf_result(ws_b, "FactorB_20D")
        _write_loop(
            run_dir,
            loop_index=1,
            hypothesis={"hypothesis": "Long horizon", "reason": "reason B"},
            tasks=[{"factor_name": "FactorB_20D", "factor_formulation": "B()"}],
            workspaces=[{"target_task_name": "FactorB_20D", "workspace_path": str(ws_b)}],
            feedback_decision=False,
        )

        summary = export_rdagent_run(
            log_run_dir=run_dir,
            workspace_candidates_dir=tmp_path / "cand",
            rdagent_workspace_root=ws_root,
        )
        assert summary.loops_scanned == 2
        assert len(summary.exports) == 2
        names = {e.name for e in summary.exports}
        assert names == {"FactorA_5D", "FactorB_20D"}
        loops = {e.loop_index for e in summary.exports}
        assert loops == {0, 1}
        # decision=False 也被导出，self_eval=0
        b_export = next(e for e in summary.exports if e.name == "FactorB_20D")
        payload = json.loads(b_export.package_path.read_text(encoding="utf-8"))
        assert payload["lab_metrics"]["rdagent_self_eval"] == 0.0


class TestIdempotency:
    def test_second_run_skipped(self, tmp_path):
        log_root, run_dir, ws_root, cand_dir = _build_minimal_run(tmp_path)
        s1 = export_rdagent_run(
            log_run_dir=run_dir,
            workspace_candidates_dir=cand_dir,
            rdagent_workspace_root=ws_root,
        )
        assert s1.exports[0].action == "written"
        # 二次
        s2 = export_rdagent_run(
            log_run_dir=run_dir,
            workspace_candidates_dir=cand_dir,
            rdagent_workspace_root=ws_root,
        )
        assert s2.exports[0].action == "skipped_idempotent"

    def test_overwrite_forces_rewrite(self, tmp_path):
        log_root, run_dir, ws_root, cand_dir = _build_minimal_run(tmp_path)
        s1 = export_rdagent_run(
            log_run_dir=run_dir,
            workspace_candidates_dir=cand_dir,
            rdagent_workspace_root=ws_root,
        )
        tgt = s1.exports[0].target_dir
        # 改一下 staging 的 factor.py 内容
        (tgt / "factor.py").write_text("# tampered\n", encoding="utf-8")

        s2 = export_rdagent_run(
            log_run_dir=run_dir,
            workspace_candidates_dir=cand_dir,
            rdagent_workspace_root=ws_root,
            overwrite=True,
        )
        # 不管原有是否幂等，overwrite=True 且 _should_skip_idempotent=False（已 tampered），
        # 会重写
        assert s2.exports[0].action == "written"
        assert (tgt / "factor.py").read_text(encoding="utf-8").startswith('"""Fake RD-Agent')


# =========================================================================== 断点场景


class TestMissingArtifacts:
    def test_missing_hypothesis_falls_back(self, tmp_path):
        log_root, run_dir, ws_root, cand_dir = _build_minimal_run(tmp_path)
        # 删 hypothesis pkl
        for p in (run_dir / "Loop_0" / "direct_exp_gen" / "hypothesis generation").rglob("*.pkl"):
            p.unlink()
        s = export_rdagent_run(
            log_run_dir=run_dir,
            workspace_candidates_dir=cand_dir,
            rdagent_workspace_root=ws_root,
        )
        assert len(s.exports) == 1
        payload = json.loads(s.exports[0].package_path.read_text(encoding="utf-8"))
        # hypothesis 退回 placeholder
        assert "rdagent loop 0" in payload["hypothesis"]

    def test_missing_workspace_dir_skipped(self, tmp_path):
        log_root = tmp_path / "log"
        run_dir = log_root / "run-xyz-00000000"
        _write_loop(
            run_dir,
            loop_index=0,
            hypothesis={"hypothesis": "h", "reason": "r"},
            tasks=[{"factor_name": "Ghost_10D", "factor_formulation": "f()"}],
            workspaces=[
                {
                    "target_task_name": "Ghost_10D",
                    "workspace_path": str(tmp_path / "no_such_dir"),
                }
            ],
            feedback_decision=True,
        )
        s = export_rdagent_run(
            log_run_dir=run_dir,
            workspace_candidates_dir=tmp_path / "cand",
            rdagent_workspace_root=tmp_path / "git_ignore_folder" / "RD-Agent_workspace",
        )
        assert s.exports == []
        assert len(s.skipped) == 1
        assert s.skipped[0]["reason"] == "workspace_missing"

    def test_missing_result_h5_skipped(self, tmp_path):
        log_root, run_dir, ws_root, cand_dir = _build_minimal_run(tmp_path)
        # 删 result.h5
        for p in ws_root.rglob("result.h5"):
            p.unlink()
        s = export_rdagent_run(
            log_run_dir=run_dir,
            workspace_candidates_dir=cand_dir,
            rdagent_workspace_root=ws_root,
        )
        assert s.exports == []
        assert any(x["reason"] == "result_h5_missing" for x in s.skipped)

    def test_missing_factor_py_but_file_dict_ok(self, tmp_path):
        log_root = tmp_path / "log"
        run_dir = log_root / "run-xyz-00000001"
        ws_root = tmp_path / "git_ignore_folder" / "RD-Agent_workspace"
        ws = ws_root / "hash-no-py"
        ws.mkdir(parents=True)
        _make_hdf_result(ws, "DictBacked_5D")
        # 不写 factor.py；改通过 file_dict 注入
        src = "# fallback from file_dict\npass\n"
        _write_loop(
            run_dir,
            loop_index=0,
            hypothesis={"hypothesis": "h", "reason": "r"},
            tasks=[{"factor_name": "DictBacked_5D", "factor_formulation": "x()"}],
            workspaces=[
                {
                    "target_task_name": "DictBacked_5D",
                    "workspace_path": str(ws),
                    "file_dict": {"factor.py": src},
                }
            ],
            feedback_decision=True,
        )
        s = export_rdagent_run(
            log_run_dir=run_dir,
            workspace_candidates_dir=tmp_path / "cand",
            rdagent_workspace_root=ws_root,
        )
        assert len(s.exports) == 1
        code_out = s.exports[0].target_dir / "factor.py"
        assert code_out.read_text(encoding="utf-8") == src

    def test_task_workspace_mismatch_skipped(self, tmp_path):
        log_root = tmp_path / "log"
        run_dir = log_root / "run-xyz-00000002"
        ws_root = tmp_path / "git_ignore_folder" / "RD-Agent_workspace"
        ws = ws_root / "hash-mismatch"
        ws.mkdir(parents=True)
        _make_factor_py(ws, "A_5D")
        _make_hdf_result(ws, "A_5D")
        # task 名和 ws target_task.factor_name 不匹配
        _write_loop(
            run_dir,
            loop_index=0,
            hypothesis={"hypothesis": "h", "reason": "r"},
            tasks=[{"factor_name": "A_5D", "factor_formulation": "a()"}],
            workspaces=[{"target_task_name": "OTHER", "workspace_path": str(ws)}],
            feedback_decision=True,
        )
        s = export_rdagent_run(
            log_run_dir=run_dir,
            workspace_candidates_dir=tmp_path / "cand",
            rdagent_workspace_root=ws_root,
        )
        assert s.exports == []
        assert any(x["reason"] == "no_task_workspace_pairs" for x in s.skipped)


# =========================================================================== 其他分支


class TestEdgeCases:
    def test_weird_factor_name_sanitized(self, tmp_path):
        log_root = tmp_path / "log"
        run_dir = log_root / "run-xyz-00000003"
        ws_root = tmp_path / "git_ignore_folder" / "RD-Agent_workspace"
        ws = ws_root / "hash-weird"
        ws.mkdir(parents=True)
        # result.h5 里列名是 sanitize 之后的
        _make_factor_py(ws, "Vol_Ret_5D")
        _make_hdf_result(ws, "Vol_Ret_5D")
        _write_loop(
            run_dir,
            loop_index=0,
            hypothesis={"hypothesis": "h", "reason": "r"},
            # 注入非法 name（带 .）
            tasks=[{"factor_name": "Vol.Ret-5D", "factor_formulation": "v()"}],
            workspaces=[
                {"target_task_name": "Vol.Ret-5D", "workspace_path": str(ws)}
            ],
            feedback_decision=None,  # 无 feedback
        )
        s = export_rdagent_run(
            log_run_dir=run_dir,
            workspace_candidates_dir=tmp_path / "cand",
            rdagent_workspace_root=ws_root,
        )
        assert len(s.exports) == 1
        e = s.exports[0]
        assert e.name == "Vol_Ret_5D"
        assert e.factor_id.startswith("rdagent_Vol_Ret_5D_")
        # 无 feedback 时没 rdagent_self_eval
        payload = json.loads(e.package_path.read_text(encoding="utf-8"))
        assert "rdagent_self_eval" not in payload["lab_metrics"]

    def test_multiple_evo_loops_picks_highest(self, tmp_path):
        """多个 evo_loop_N 目录，应读取 N 最大的那个（最后一轮演化）。"""
        log_root = tmp_path / "log"
        run_dir = log_root / "run-xyz-00000004"
        ws_root = tmp_path / "git_ignore_folder" / "RD-Agent_workspace"

        # 两个 workspace，evo_loop_0 指向 A（旧），evo_loop_3 指向 B（新）
        ws_old = ws_root / "hash-old-evo0"
        ws_new = ws_root / "hash-new-evo3"
        for ws, name, marker in [
            (ws_old, "EvoTarget_10D", "old"),
            (ws_new, "EvoTarget_10D", "new"),
        ]:
            ws.mkdir(parents=True)
            _make_factor_py(ws, name, marker=marker)
            _make_hdf_result(ws, name)

        loop_dir = run_dir / "Loop_0"
        # hypothesis + experiment generation
        _pickle_at(
            loop_dir / "direct_exp_gen" / "hypothesis generation" / "1" / "1.pkl",
            SimpleNamespace(hypothesis="h", reason="r"),
        )
        _pickle_at(
            loop_dir / "direct_exp_gen" / "experiment generation" / "1" / "1.pkl",
            [SimpleNamespace(factor_name="EvoTarget_10D", factor_formulation="e()")],
        )
        # evo_loop_0 -> old
        _pickle_at(
            loop_dir / "coding" / "evo_loop_0" / "evolving code" / "1" / "1.pkl",
            [
                SimpleNamespace(
                    target_task=SimpleNamespace(factor_name="EvoTarget_10D"),
                    workspace_path=str(ws_old),
                    file_dict={},
                )
            ],
        )
        # evo_loop_3 -> new
        _pickle_at(
            loop_dir / "coding" / "evo_loop_3" / "evolving code" / "1" / "1.pkl",
            [
                SimpleNamespace(
                    target_task=SimpleNamespace(factor_name="EvoTarget_10D"),
                    workspace_path=str(ws_new),
                    file_dict={},
                )
            ],
        )
        _pickle_at(
            loop_dir / "feedback" / "feedback" / "1" / "1.pkl",
            SimpleNamespace(decision=True),
        )

        s = export_rdagent_run(
            log_run_dir=run_dir,
            workspace_candidates_dir=tmp_path / "cand",
            rdagent_workspace_root=ws_root,
        )
        assert len(s.exports) == 1
        # factor_id 应基于 ws_new 的 hash
        assert "hash-new" in json.loads(
            s.exports[0].package_path.read_text(encoding="utf-8")
        )["_meta"]["rdagent_workspace_path"]

    def test_scan_loops_orders_by_index(self, tmp_path):
        run_dir = tmp_path / "run"
        # 故意乱序写
        for idx in (2, 0, 1):
            _write_loop(
                run_dir,
                loop_index=idx,
                hypothesis={"hypothesis": f"h{idx}", "reason": "r"},
                tasks=[{"factor_name": f"F_{idx}"}],
                workspaces=[],
                feedback_decision=None,
            )
        loops = scan_loops(run_dir)
        assert [lp.loop_index for lp in loops] == [0, 1, 2]


# =========================================================================== 全树扫描


class TestExportLogTree:
    def test_scan_multiple_runs(self, tmp_path):
        log_root = tmp_path / "log"
        ws_root = tmp_path / "git_ignore_folder" / "RD-Agent_workspace"
        cand_dir = tmp_path / "cand"

        for run_name, fac_name, ws_hash in [
            ("2026-04-19_aaaaaa", "Alpha_5D", "aaaaaaaaaaaaaaaaaaaaaaaa"),
            ("2026-04-20_bbbbbb", "Beta_10D", "bbbbbbbbbbbbbbbbbbbbbbbb"),
        ]:
            ws = ws_root / ws_hash
            ws.mkdir(parents=True, exist_ok=True)
            _make_factor_py(ws, fac_name)
            _make_hdf_result(ws, fac_name)
            _write_loop(
                log_root / run_name,
                loop_index=0,
                hypothesis={"hypothesis": "h", "reason": "r"},
                tasks=[{"factor_name": fac_name, "factor_formulation": "x()"}],
                workspaces=[{"target_task_name": fac_name, "workspace_path": str(ws)}],
                feedback_decision=True,
            )

        # 加一个无 Loop_N 的脏目录，要被忽略
        (log_root / "zzz_no_loop").mkdir()
        (log_root / "zzz_no_loop" / "some.txt").write_text("ignore me")

        summaries = export_rdagent_log_tree(
            log_root=log_root,
            workspace_candidates_dir=cand_dir,
            rdagent_workspace_root=ws_root,
        )
        assert len(summaries) == 2
        all_names = {e.name for s in summaries for e in s.exports}
        assert all_names == {"Alpha_5D", "Beta_10D"}

    def test_run_filter(self, tmp_path):
        log_root = tmp_path / "log"
        ws_root = tmp_path / "ws"
        cand_dir = tmp_path / "cand"
        for run_name, fac_name, ws_hash in [
            ("2026-04-19_aaaaaa", "Alpha_5D", "aaaaaaaaaaaaaaaaaaaaaaaa"),
            ("2026-04-20_bbbbbb", "Beta_10D", "bbbbbbbbbbbbbbbbbbbbbbbb"),
        ]:
            ws = ws_root / ws_hash
            ws.mkdir(parents=True)
            _make_factor_py(ws, fac_name)
            _make_hdf_result(ws, fac_name)
            _write_loop(
                log_root / run_name,
                loop_index=0,
                hypothesis={"hypothesis": "h", "reason": "r"},
                tasks=[{"factor_name": fac_name, "factor_formulation": "x()"}],
                workspaces=[{"target_task_name": fac_name, "workspace_path": str(ws)}],
                feedback_decision=True,
            )
        summaries = export_rdagent_log_tree(
            log_root=log_root,
            workspace_candidates_dir=cand_dir,
            rdagent_workspace_root=ws_root,
            run_filter="2026-04-20",
        )
        assert len(summaries) == 1
        assert summaries[0].run_id == "2026-04-20_bbbbbb"


# =========================================================================== CLI


class TestCLI:
    def test_cli_end_to_end(self, tmp_path, monkeypatch, capsys):
        log_root, run_dir, ws_root, cand_dir = _build_minimal_run(tmp_path)
        summary_json = tmp_path / "out" / "summary.json"

        argv = [
            "export_rdagent_candidates",
            "--log-run-dir",
            str(run_dir),
            "--workspace-candidates-dir",
            str(cand_dir),
            "--rdagent-workspace-root",
            str(ws_root),
            "--summary-json",
            str(summary_json),
            "--log-level",
            "WARNING",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        from scripts.lab.export_rdagent_candidates import main

        rc = main(argv[1:])
        assert rc == 0
        assert summary_json.exists()
        payload = json.loads(summary_json.read_text(encoding="utf-8"))
        assert payload["n_runs"] == 1
        assert payload["runs"][0]["n_exports"] == 1
