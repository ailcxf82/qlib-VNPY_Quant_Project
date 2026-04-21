"""阶段 E/F/G 累积契约：

* E.4 搬家：``factor_lab.adapters.*`` 是 RD-Agent 适配层的唯一事实源（新家）；
* F：YAML 驱动静态宪法 + 动态 feedback；
* **G.2（2026-04）**：``rdagent_integration.*`` + ``scripts/run_fin_quant.py`` 的旧 shim
  升级为 **import 时立即 raise RuntimeError**，且异常信息里必须包含新 import 路径的迁移指引；
  DeprecationWarning 阶段结束。

本文件把三段契约合并成可执行测试，任何一次无意识改回或遗漏都会红。
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest


# ---------- 新家可用 ----------


def test_new_home_adapters_experiments_imports_ok() -> None:
    from factor_lab.adapters.experiments import (  # noqa: F401
        ProjectQlibFactorExperiment,
        ProjectQlibModelExperiment,
    )


def test_new_home_adapters_proposal_imports_ok() -> None:
    from factor_lab.adapters.proposal import (  # noqa: F401
        ProjectQlibFactorHypothesis2Experiment,
        ProjectQlibModelHypothesis2Experiment,
    )


def test_new_home_adapters_quant_proposal_imports_ok() -> None:
    from factor_lab.adapters.quant_proposal import (  # noqa: F401
        ProjectQlibQuantHypothesisGen,
        _STATIC_CONSTITUTION,
        compose_project_rag,
    )


def test_new_home_adapters_patch_qlib_conda_imports_ok() -> None:
    from factor_lab.adapters.patch_qlib_conda import (  # noqa: F401
        apply_qlib_conda_env_patch,
        _patch_qlib_runner_env,
    )


# ---------- 老 shim：G.2 起 import 必须立即 raise RuntimeError ----------


def _fresh_import(name: str):
    """清 sys.modules 缓存以保证 top-level 代码被重新执行。"""
    to_drop = [k for k in list(sys.modules) if k == name or k.startswith(name + ".")]
    for k in to_drop:
        del sys.modules[k]
    return importlib.import_module(name)


@pytest.mark.parametrize(
    "mod_name",
    [
        "rdagent_integration",
        "rdagent_integration.project_experiments",
        "rdagent_integration.project_proposal",
        "rdagent_integration.project_quant_proposal",
        "rdagent_integration.patch_qlib_conda",
    ],
)
def test_old_shim_modules_raise_runtime_error_on_import(mod_name: str) -> None:
    with pytest.raises(RuntimeError) as excinfo:
        _fresh_import(mod_name)
    msg = str(excinfo.value)
    # 异常信息必须显式引导到 factor_lab.adapters.*
    assert "factor_lab.adapters" in msg, f"迁移提示缺失 factor_lab.adapters: {msg}"
    assert "G.2" in msg or "下线" in msg


def test_scripts_run_fin_quant_raises_runtime_error() -> None:
    """旧入口 scripts/run_fin_quant.py 在 load 时就必须抛 RuntimeError。"""
    path = Path(__file__).resolve().parents[3] / "scripts" / "run_fin_quant.py"
    assert path.exists(), f"scripts/run_fin_quant.py 文件缺失：{path}"
    spec = importlib.util.spec_from_file_location("scripts_run_fin_quant_removed_test", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    with pytest.raises(RuntimeError) as excinfo:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    msg = str(excinfo.value)
    assert "scripts.lab.run_rdagent_loop" in msg
    assert "G.2" in msg or "下线" in msg


# ---------- 核心行为必须仍可从新家取得 ----------


def test_compose_project_rag_in_new_home_is_callable() -> None:
    from factor_lab.adapters.quant_proposal import compose_project_rag

    out = compose_project_rag("BASE_UPSTREAM_RAG", feedback_dir=None)
    assert out.startswith("BASE_UPSTREAM_RAG")
    assert "Project factor hypothesis constraints" in out


def test_legacy_alias_project_factor_rag_preserved_in_new_home() -> None:
    """``_PROJECT_FACTOR_RAG`` 别名仍在新家保留（外部审计脚本依赖这个名字）。"""
    from factor_lab.adapters.quant_proposal import (
        _PROJECT_FACTOR_RAG,
        _STATIC_CONSTITUTION,
    )
    assert _PROJECT_FACTOR_RAG == _STATIC_CONSTITUTION
