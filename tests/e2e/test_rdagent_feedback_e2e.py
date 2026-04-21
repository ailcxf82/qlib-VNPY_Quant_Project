"""阶段 F.4：RD-Agent + Feedback 端到端集成测试。

两层设计：

1. **非 marker**（每次跑都会触发）：整链路 aggregator → bundle → compose_project_rag
   → 最终 RAG 字符串，断言 discouraged 家族真的回流到了 prompt 层。这部分替代
   了"阶段 E 的合成器单测只在纯函数层面跑"的不足。

2. **``@pytest.mark.e2e_rdagent``**（默认 skip，要跑加 `pytest -m e2e_rdagent`）：
   加载真实的 RD-Agent ``QlibQuantHypothesisGen`` 基类，验证我们的子类
   ``ProjectQlibQuantHypothesisGen`` 挂钩位置正确 + 类继承链完整 + ``prepare_context``
   重写确实调用到 ``compose_project_rag``。不调 LLM，用 monkeypatch 替换父类方法。

阶段 F.4 放弃"直接调 RD-Agent 主循环 + 真实 LLM"这个 variant：成本高、依赖 API key
和真实 fin_quant 数据；真实 live 验证请走 ``docs/STAGE_F_TEST_PLAYBOOK.md`` 里的
手工冒烟步骤。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from factor_lab.adapters.quant_proposal import (
    ProjectQlibQuantHypothesisGen,
    compose_project_rag,
)
from factor_lab.feedback.aggregator import build_feedback_bundle, write_feedback_bundle


# ------------------------------------------------------------------ fixtures


def _seed_cycle_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    """
    造一份最小但"真实味道"的 cycle artifacts：
    - 2 个 cycle report
    - 每个 cycle 3 张 cert（2 FAIL 同 family + 1 PASS 其它 family）
    - manifest 有 1 active + 1 retired，全带 universe
    """
    reports_dir = tmp_path / "reports"
    cert_dir = tmp_path / "certificates"
    registry_data = tmp_path / "registry_data"
    reports_dir.mkdir()
    cert_dir.mkdir()
    registry_data.mkdir()

    now_ts = datetime(2026, 4, 20, tzinfo=timezone.utc).isoformat()

    for cycle_id in ("cyc-2026-04-13", "cyc-2026-04-20"):
        (reports_dir / f"lab_cycle_{cycle_id}.json").write_text(
            json.dumps({"cycle_id": cycle_id}), encoding="utf-8"
        )
        cyc_dir = cert_dir / cycle_id
        cyc_dir.mkdir()
        # 两个 volume_price_reversal FAIL（driving discouraged）
        for i in range(2):
            payload = {
                "decision": "FAIL",
                "candidate": {
                    "name": f"VolRev_{5 + i * 5}D",
                    "universe": "csi300",
                },
                "check_results": [
                    {"name": "ic", "passed": False, "detail": {"rank_ic": 0.002 + i * 0.001}},
                    {"name": "orthogonality", "passed": False, "detail": {"max_abs_corr": 0.80}},
                ],
            }
            (cyc_dir / f"fid{i}.default.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
        # 一个 PASS（Quality）：不进 recent_fails
        pass_payload = {
            "decision": "PASS",
            "candidate": {"name": "QualPersist_60D", "universe": "csi300"},
            "check_results": [{"name": "ic", "passed": True, "detail": {"rank_ic": 0.03}}],
        }
        (cyc_dir / "fidp.default.json").write_text(
            json.dumps(pass_payload), encoding="utf-8"
        )

    manifest = {
        "factors": [
            {
                "factor_id": "prod_alpha_abcdef0001",
                "name": "QualPersist_60D",
                "status": "active",
                "parquet_version": 1,
                "universe": "csi300",
            },
            {
                "factor_id": "prod_alpha_abcdef0002",
                "name": "VolTwist_10D",
                "status": "retired",
                "retired_at": now_ts,
                "retire_reason": "Sharpe dropped 3.59 -> 1.69 under turnover regime",
                "universe": "csi300",
            },
        ]
    }
    (registry_data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return reports_dir, cert_dir, registry_data


# ---------------------------------------------- integration: full pipeline


def test_feedback_pipeline_end_to_end_into_final_rag(tmp_path: Path) -> None:
    """
    aggregator → bundle → latest.json → compose_project_rag → final prompt.

    关键断言：
    - 两个 discouraged 家族（volume_price_reversal 来自 fails + retired）真的出现在 prompt；
    - L3 active 名字 + retired 名字 + 最近 fails 名字都能被 LLM 看到；
    - [universe=csi300] 每种 summary 都渲染；
    - 静态宪法 + 动态段顺序稳定。
    """
    r, c, reg = _seed_cycle_artifacts(tmp_path)
    bundle = build_feedback_bundle(
        reports_dir=r, cert_dir=c, registry_data_dir=reg, max_cycles=8,
        min_fail_count_for_discouraged=2,
    )
    assert "volume_price_reversal" in bundle.discouraged_families

    fb_dir = tmp_path / "workspace_feedback"
    write_feedback_bundle(bundle, workspace_feedback_dir=fb_dir)

    final = compose_project_rag(base_rag="SENTINEL_BASE", feedback_dir=fb_dir)

    assert final.startswith("SENTINEL_BASE")
    # 静态宪法必须在
    assert "Project factor hypothesis constraints" in final
    assert "(x) Short-cycle volume-price reversals" in final
    # 动态段必须在
    assert "Feedback from recent L2 cycles (dynamic)" in final
    # discouraged family 从 L2 数据层一路出现到 prompt 层
    assert "volume_price_reversal" in final
    # L3 active / retired / recent fails 名字都在
    assert "QualPersist_60D" in final
    assert "VolTwist_10D" in final
    assert "VolRev_5D" in final
    # universe 标签标准化渲染
    assert "[universe=csi300]" in final


def test_pipeline_discouraged_threshold_respected(tmp_path: Path) -> None:
    """
    只有 1 个 FAIL + 无 retired → discouraged 为空，prompt 里不应该误列。
    """
    reports_dir = tmp_path / "reports"
    cert_dir = tmp_path / "certificates"
    registry_data = tmp_path / "registry_data"
    reports_dir.mkdir()
    cert_dir.mkdir()
    registry_data.mkdir()
    (reports_dir / "lab_cycle_cyc1.json").write_text(
        json.dumps({"cycle_id": "cyc1"}), encoding="utf-8"
    )
    (cert_dir / "cyc1").mkdir()
    (cert_dir / "cyc1" / "x.default.json").write_text(
        json.dumps(
            {
                "decision": "FAIL",
                "candidate": {"name": "VolRev_5D", "universe": "csi300"},
                "check_results": [{"name": "ic", "passed": False, "detail": {}}],
            }
        ),
        encoding="utf-8",
    )
    (registry_data / "manifest.json").write_text(json.dumps({"factors": []}), encoding="utf-8")

    bundle = build_feedback_bundle(
        reports_dir=reports_dir, cert_dir=cert_dir, registry_data_dir=registry_data,
        max_cycles=8, min_fail_count_for_discouraged=2,
    )
    assert bundle.discouraged_families == ()

    fb_dir = tmp_path / "feedback"
    write_feedback_bundle(bundle, workspace_feedback_dir=fb_dir)
    final = compose_project_rag(base_rag="", feedback_dir=fb_dir)

    # markdown 里不应该出现 discouraged 段
    assert "Discouraged families — empirically blocked" not in final
    # 但单条 fail 仍在
    assert "VolRev_5D" in final


# ---------------------------------- marker-gated: real RD-Agent subclass


@pytest.mark.e2e_rdagent
def test_real_rdagent_base_class_in_mro() -> None:
    """
    断言 ProjectQlibQuantHypothesisGen 真的继承自 RD-Agent 的 QlibQuantHypothesisGen，
    并把 prepare_context 挂到正确位置。
    """
    from rdagent.scenarios.qlib.proposal.quant_proposal import QlibQuantHypothesisGen

    assert issubclass(ProjectQlibQuantHypothesisGen, QlibQuantHypothesisGen)
    # prepare_context 必须是子类自己定义的那一份
    assert (
        ProjectQlibQuantHypothesisGen.prepare_context  # type: ignore[comparison-overlap]
        is not QlibQuantHypothesisGen.prepare_context
    )


@pytest.mark.e2e_rdagent
def test_prepare_context_chains_compose_project_rag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    真实 ProjectQlibQuantHypothesisGen 实例 + 真实 feedback 盘存 +
    monkeypatch 掉父类的 prepare_context（避免需要 Scenario/LLM），
    证明 prepare_context 会把 discouraged 家族写进 ctx['RAG']。
    """
    from rdagent.scenarios.qlib.proposal.quant_proposal import QlibQuantHypothesisGen

    r, c, reg = _seed_cycle_artifacts(tmp_path)
    fb_dir = tmp_path / "feedback"
    bundle = build_feedback_bundle(
        reports_dir=r, cert_dir=c, registry_data_dir=reg, max_cycles=8,
        min_fail_count_for_discouraged=2,
    )
    write_feedback_bundle(bundle, workspace_feedback_dir=fb_dir)

    # 替换父类 prepare_context：返回一个 sentinel RAG
    def _fake_super_prepare(self, trace):  # noqa: ANN001
        return {"RAG": "UPSTREAM_BASE_SENTINEL"}, True

    monkeypatch.setattr(QlibQuantHypothesisGen, "prepare_context", _fake_super_prepare)

    gen = ProjectQlibQuantHypothesisGen.__new__(ProjectQlibQuantHypothesisGen)
    gen._feedback_dir_override = fb_dir
    ctx, ok = gen.prepare_context(trace=None)
    assert ok is True
    rag = ctx["RAG"]
    assert rag.startswith("UPSTREAM_BASE_SENTINEL")
    assert "Project factor hypothesis constraints" in rag
    assert "volume_price_reversal" in rag
    assert "QualPersist_60D" in rag
    assert "[universe=csi300]" in rag


@pytest.mark.e2e_rdagent
def test_prepare_context_degrades_when_feedback_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    feedback 盘缺失 → 应该只注入静态宪法，不抛；保证 live loop 不会被回流失败搞崩。
    """
    from rdagent.scenarios.qlib.proposal.quant_proposal import QlibQuantHypothesisGen

    def _fake_super_prepare(self, trace):  # noqa: ANN001
        return {"RAG": "BASE_X"}, True

    monkeypatch.setattr(QlibQuantHypothesisGen, "prepare_context", _fake_super_prepare)

    gen = ProjectQlibQuantHypothesisGen.__new__(ProjectQlibQuantHypothesisGen)
    gen._feedback_dir_override = tmp_path / "ghost"
    ctx, ok = gen.prepare_context(trace=None)
    assert ok is True
    rag = ctx["RAG"]
    assert rag.startswith("BASE_X")
    assert "Project factor hypothesis constraints" in rag
    assert "Feedback from recent L2 cycles" not in rag
