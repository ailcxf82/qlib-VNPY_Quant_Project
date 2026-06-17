"""
E.2：C3 FeedbackBundle 的聚合器。

输入：

* `factor_validation/reports/lab_cycle_*.json` —— D.5 cycle 报告。
* `factor_validation/certificates/<cycle_id>/<factor_id>.(exploratory|default).json` ——
  D.5 每 cycle 逐候选的 C2 证书副本。
* `factor_registry/data/manifest.json` —— L3 active 因子列表。
* `factor_registry/data/retired/<factor_id>.json` —— L3 retired 因子的 C2 副本。

输出：单个 ``FeedbackBundle`` 实例；由调用方选择落盘为 JSON + 渲染 Markdown。

**边界约束**（阶段 E 明确放弃的东西，避免范围蔓延）：

* 聚合器**只读**。不修改 manifest、不删除 cert、不 retire 因子。
* family 标签采用**启发式**：基于因子名里的家族关键词 + 窗口；未命中统一归 ``unknown``。
  真正的 family 分类属于 factor_lab 的研究问题，本模块只做最小可用映射。
* 不跨版本迁移：``schema_version`` 不兼容时聚合器原地失败，而非静默适配。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from factor_lab.config.families import classify_family as _classify_family_yaml
from factor_lab.feedback.schema import (
    ActiveFactorSummary,
    FailedCandidateSummary,
    FeedbackBundle,
    RetiredFactorSummary,
    UniverseSubBundle,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- families
#
# 阶段 F.1 起：家族规则的事实源迁到 ``factor_lab/config/factor_families.yaml``，
# 通过 ``factor_lab.config.classify_family`` 加载。本模块保留 ``classify_family``
# 的 re-export，仅为兼容老 import（``from factor_lab.feedback.aggregator import
# classify_family``）。新代码请直接用 ``factor_lab.config``。


def classify_family(name: str) -> str:
    """从因子名推导 family 标签；无命中返回 ``unknown``（YAML 主源 + Python 兜底）。"""
    return _classify_family_yaml(name)


# -------------------------------------------------------------- data containers


@dataclass(frozen=True)
class AggregatorInputs:
    """聚合器的输入路径集合；便于单测替换。"""

    reports_dir: Path
    cert_dir: Path
    registry_data_dir: Path


# ------------------------------------------------------------------- file utils


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_iso_datetime(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        ts = datetime.fromisoformat(s)
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


# ---------------------------------------------------------- cycle report reader


def _list_cycle_reports(reports_dir: Path) -> list[Path]:
    """返回 reports_dir 下所有 lab_cycle_*.json，按 mtime 升序（老在前）。"""
    if not reports_dir.exists():
        return []
    items = sorted(
        reports_dir.glob("lab_cycle_*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    return list(items)


def _pick_recent_cycles(
    reports_dir: Path, *, max_cycles: int
) -> list[tuple[str, dict[str, Any]]]:
    """按 mtime 选最近 max_cycles 个 cycle 报告，返回 [(cycle_id, payload), ...] 时间升序。"""
    files = _list_cycle_reports(reports_dir)
    if max_cycles <= 0:
        return []
    chosen = files[-max_cycles:]
    out: list[tuple[str, dict[str, Any]]] = []
    for f in chosen:
        try:
            payload = _load_json(f)
        except Exception as exc:  # noqa: BLE001
            logger.warning("跳过无法解析的 cycle 报告 %s: %s", f, exc)
            continue
        cycle_id = payload.get("cycle_id")
        if not cycle_id:
            logger.warning("跳过缺 cycle_id 的报告 %s", f)
            continue
        out.append((str(cycle_id), payload))
    return out


# -------------------------------------------------------------- cert extraction


_BACKTEST_METRIC_KEYS = ("sharpe", "max_drawdown", "annual_return", "long_short_sharpe")


def _extract_cert_failure_modes(cert_payload: dict[str, Any]) -> tuple[list[str], dict[str, float]]:
    """
    从 C2 证书 JSON 提炼 (failure_modes, metrics)。

    * failure_modes = 所有 ``passed=False`` 的 check.name；按 schema 白名单过滤。
    * metrics = 汇总关键指标：
        * ic check → rank_ic / ic_ir
        * orthogonality → max_abs_corr
        * turnover → daily_rank_turnover / rank_autocorr
        * marginal → residual_rank_ic / residual_ic_ir
        * backtest / backtest_rqalpha → sharpe / max_drawdown / annual_return
      只保留数值、有限的条目。
    """
    from factor_lab.feedback.schema import _KNOWN_FAILURE_MODES  # noqa: WPS433 — 内部常量引用

    failure_modes: list[str] = []
    metrics: dict[str, float] = {}
    checks = cert_payload.get("check_results") or []
    for c in checks:
        if not isinstance(c, dict):
            continue
        name = c.get("name")
        if not isinstance(name, str):
            continue
        passed = bool(c.get("passed", False))
        if not passed and name in _KNOWN_FAILURE_MODES:
            failure_modes.append(name)
        detail = c.get("detail") or {}
        if not isinstance(detail, dict):
            continue
        # 常用指标：尽量用 detail 扁平字段；backtest 则进 detail["metrics"] 再扁平
        for k, v in detail.items():
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            val = float(v)
            if val != val or val in (float("inf"), float("-inf")):
                continue
            pretty = f"{name}.{k}" if name not in {"ic", "orthogonality", "turnover", "marginal"} else k
            metrics[pretty] = val
        nested = detail.get("metrics")
        if isinstance(nested, dict):
            for k, v in nested.items():
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    continue
                val = float(v)
                if val != val or val in (float("inf"), float("-inf")):
                    continue
                if k in _BACKTEST_METRIC_KEYS:
                    metrics[f"{name}.{k}"] = val
    return failure_modes, metrics


def _iter_cert_files(cert_cycle_dir: Path) -> Iterable[tuple[str, Path]]:
    """
    枚举一次 cycle 对应的证书文件，产出 (stage, path)。

    stage 来自文件名后缀：
        <fid>.exploratory.json -> "exploratory"
        <fid>.default.json     -> "default"
    其它后缀忽略。
    """
    if not cert_cycle_dir.exists():
        return []
    pairs: list[tuple[str, Path]] = []
    for f in sorted(cert_cycle_dir.iterdir()):
        if not f.is_file() or f.suffix != ".json":
            continue
        stem_parts = f.stem.split(".")
        if len(stem_parts) >= 2 and stem_parts[-1] in {"exploratory", "default", "strict"}:
            pairs.append((stem_parts[-1], f))
    return pairs


def _build_failed_candidates(
    cycle_id: str,
    cert_cycle_dir: Path,
) -> list[FailedCandidateSummary]:
    """
    读一次 cycle 的全部证书，挑出 FAIL / HOLD 的候选，打包成 FailedCandidateSummary 列表。

    同一个因子可能出现在 exploratory + default 两档；我们只保留**最深档**的记录
    （default > exploratory），这样 LLM 看到的是"最后卡住的那一档"。
    """
    records: dict[str, FailedCandidateSummary] = {}
    stage_priority = {"exploratory": 1, "default": 2, "strict": 3}

    for stage, cert_path in _iter_cert_files(cert_cycle_dir):
        try:
            payload = _load_json(cert_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("跳过无法解析的证书 %s: %s", cert_path, exc)
            continue
        decision = str(payload.get("decision") or "").upper()
        if decision not in {"FAIL", "HOLD"}:
            continue
        candidate = payload.get("candidate") or {}
        name = candidate.get("name")
        if not isinstance(name, str) or not name:
            continue
        # F.2：候选 universe 来自 C1 的 candidate.universe（C2 把 C1 内嵌了）
        universe_raw = candidate.get("universe")
        universe = universe_raw if isinstance(universe_raw, str) and universe_raw else None

        modes, metrics = _extract_cert_failure_modes(payload)
        try:
            summary = FailedCandidateSummary(
                name=name,
                family=classify_family(name),
                universe=universe,
                cycle_id=cycle_id,
                stage=stage,
                decision=decision,
                failure_modes=tuple(modes),
                metrics=metrics,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("跳过构造失败的 FailedCandidateSummary name=%s: %s", name, exc)
            continue

        existing = records.get(name)
        if existing is None:
            records[name] = summary
            continue
        if stage_priority.get(stage, 0) > stage_priority.get(existing.stage, 0):
            records[name] = summary

    return sorted(records.values(), key=lambda x: x.name)


# --------------------------------------------------------------- registry reader


def _load_manifest_records(registry_data_dir: Path) -> list[dict[str, Any]]:
    manifest_path = registry_data_dir / "manifest.json"
    if not manifest_path.exists():
        return []
    try:
        payload = _load_json(manifest_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("manifest 无法解析 %s: %s", manifest_path, exc)
        return []
    factors = payload.get("factors") or []
    if not isinstance(factors, list):
        return []
    return [f for f in factors if isinstance(f, dict)]


def _build_active_summaries(
    manifest_records: list[dict[str, Any]],
) -> list[ActiveFactorSummary]:
    out: list[ActiveFactorSummary] = []
    for rec in manifest_records:
        if rec.get("status") != "active":
            continue
        try:
            name = rec["name"]
            # F.2：manifest 里 universe 是可选字段；若不存在保持 None（向后兼容）
            u_raw = rec.get("universe")
            universe = u_raw if isinstance(u_raw, str) and u_raw else None
            summary = ActiveFactorSummary(
                factor_id=rec["factor_id"],
                name=name,
                family=classify_family(name),
                universe=universe,
                parquet_version=int(rec["parquet_version"]),
                tags=tuple(rec.get("tags") or ()),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("跳过非法 active 记录 %s: %s", rec, exc)
            continue
        out.append(summary)
    out.sort(key=lambda x: x.name)
    return out


def _build_retired_summaries(
    manifest_records: list[dict[str, Any]],
) -> list[RetiredFactorSummary]:
    out: list[RetiredFactorSummary] = []
    for rec in manifest_records:
        if rec.get("status") != "retired":
            continue
        try:
            name = rec["name"]
            retired_at = _parse_iso_datetime(rec.get("retired_at"))
            reason = rec.get("retire_reason") or "(no reason recorded)"
            u_raw = rec.get("universe")
            universe = u_raw if isinstance(u_raw, str) and u_raw else None
            summary = RetiredFactorSummary(
                factor_id=rec["factor_id"],
                name=name,
                family=classify_family(name),
                universe=universe,
                retired_at=retired_at,
                reason=reason,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("跳过非法 retired 记录 %s: %s", rec, exc)
            continue
        out.append(summary)
    out.sort(key=lambda x: (x.retired_at or datetime.min.replace(tzinfo=timezone.utc), x.name))
    return out


# --------------------------------------------------------- discouraged families


def _derive_discouraged(
    recent_fails: Iterable[FailedCandidateSummary],
    retired: Iterable[RetiredFactorSummary],
    *,
    min_fail_count: int,
) -> tuple[dict[str, int], list[str]]:
    """
    统计 family → 失败计数；超过阈值 OR 在 retired 集合里出现过的 family 被标记为 discouraged。

    - ``failure_family_counts`` 给 LLM 看绝对数字；
    - ``discouraged_families`` 给 LLM 看硬结论。
    """
    counts: dict[str, int] = {}
    for ff in recent_fails:
        counts[ff.family] = counts.get(ff.family, 0) + 1

    retired_families = {r.family for r in retired if r.family != "unknown"}

    discouraged: list[str] = []
    for fam, cnt in counts.items():
        if fam == "unknown":
            continue
        if cnt >= min_fail_count:
            discouraged.append(fam)

    for fam in retired_families:
        if fam not in discouraged:
            discouraged.append(fam)

    discouraged.sort()
    return counts, discouraged


# ------------------------------------------------------------ by_universe view


def _derive_by_universe(
    active: Iterable[ActiveFactorSummary],
    retired: Iterable[RetiredFactorSummary],
    recent_fails: Iterable[FailedCandidateSummary],
    *,
    min_fail_count: int,
) -> dict[str, UniverseSubBundle]:
    """阶段 G.1：按 universe 分桶，每桶独立推导 failure_family_counts + discouraged。

    规则：
      * 仅带 universe 标签的条目进入对应桶；universe 为 None 的一律留在父 bundle 平铺段，
        **不**进任何 sub-bundle。
      * 每桶 discouraged = (本桶失败数 >= min_fail_count 的 family)
                       ∪ (本桶 retired 出现过的 family - unknown)。
      * 桶按 universe 字典序排序，方便 diff 与 snapshot 复现。
    """
    buckets: dict[str, dict[str, list]] = {}

    def _ensure(u: str) -> dict[str, list]:
        return buckets.setdefault(
            u,
            {"active": [], "retired": [], "fails": []},
        )

    for a in active:
        if a.universe:
            _ensure(a.universe)["active"].append(a)
    for r in retired:
        if r.universe:
            _ensure(r.universe)["retired"].append(r)
    for f in recent_fails:
        if f.universe:
            _ensure(f.universe)["fails"].append(f)

    out: dict[str, UniverseSubBundle] = {}
    for u in sorted(buckets.keys()):
        b = buckets[u]
        fams_ret = {r.family for r in b["retired"] if r.family != "unknown"}
        counts: dict[str, int] = {}
        for ff in b["fails"]:
            counts[ff.family] = counts.get(ff.family, 0) + 1
        discouraged_list: list[str] = []
        for fam, cnt in counts.items():
            if fam == "unknown":
                continue
            if cnt >= min_fail_count:
                discouraged_list.append(fam)
        for fam in fams_ret:
            if fam not in discouraged_list:
                discouraged_list.append(fam)
        discouraged_list.sort()

        out[u] = UniverseSubBundle(
            universe=u,
            active_factors=tuple(sorted(b["active"], key=lambda x: x.name)),
            retired_factors=tuple(
                sorted(
                    b["retired"],
                    key=lambda x: (
                        x.retired_at or datetime.min.replace(tzinfo=timezone.utc),
                        x.name,
                    ),
                )
            ),
            recent_fails=tuple(sorted(b["fails"], key=lambda x: x.name)),
            failure_family_counts=counts,
            discouraged_families=tuple(discouraged_list),
        )
    return out


# --------------------------------------------------------- public: build bundle


def build_feedback_bundle(
    *,
    reports_dir: Path,
    cert_dir: Path,
    registry_data_dir: Path,
    max_cycles: int = 8,
    min_fail_count_for_discouraged: int = 2,
    now: datetime | None = None,
    notes: str | None = None,
) -> FeedbackBundle:
    """
    聚合最近 ``max_cycles`` 次 cycle 的 L2 结论，加上当前 L3 状态，构造 FeedbackBundle。

    Parameters
    ----------
    reports_dir : Path
        ``factor_validation/reports/`` —— 含 ``lab_cycle_<id>.json``。
    cert_dir : Path
        ``factor_validation/certificates/`` —— 含子目录 ``<cycle_id>/<factor_id>.<stage>.json``。
    registry_data_dir : Path
        ``factor_registry/data/`` —— 含 ``manifest.json``、``retired/``。
    max_cycles : int
        参与本次聚合的最近 cycle 数（默认 8）。
    min_fail_count_for_discouraged : int
        家族失败数达到此阈值进 discouraged（默认 2）。
    now : datetime
        测试可注入，默认 UTC 当前。
    notes : str | None
        写进 bundle.notes 的人工注记。
    """
    now = now or datetime.now(timezone.utc)
    if max_cycles < 1:
        raise ValueError(f"max_cycles 必须 >= 1，当前 {max_cycles}")

    cycle_records = _pick_recent_cycles(reports_dir, max_cycles=max_cycles)
    cycle_ids: list[str] = [cid for cid, _ in cycle_records]

    recent_fails: list[FailedCandidateSummary] = []
    for cid, _payload in cycle_records:
        recent_fails.extend(_build_failed_candidates(cid, cert_dir / cid))

    manifest_records = _load_manifest_records(registry_data_dir)
    active = _build_active_summaries(manifest_records)
    retired = _build_retired_summaries(manifest_records)

    counts, discouraged = _derive_discouraged(
        recent_fails, retired, min_fail_count=min_fail_count_for_discouraged
    )

    by_universe = _derive_by_universe(
        active, retired, recent_fails, min_fail_count=min_fail_count_for_discouraged
    )

    bundle = FeedbackBundle(
        generated_at=now,
        cycles_included=tuple(cycle_ids),
        window_max_cycles=max_cycles,
        active_factors=tuple(active),
        retired_factors=tuple(retired),
        recent_fails=tuple(recent_fails),
        failure_family_counts=counts,
        discouraged_families=tuple(discouraged),
        notes=notes,
        by_universe=by_universe,
    )
    return bundle


# ---------------------------------------------------------- public: disk writer


def write_feedback_bundle(
    bundle: FeedbackBundle,
    *,
    workspace_feedback_dir: Path,
    also_update_history: bool = True,
    also_write_per_universe: bool = True,
) -> tuple[Path, Path]:
    """
    把 bundle 落到 ``workspace_feedback_dir``：

    * ``latest.json``                   —— 最近一次聚合结果（供 RAG 注入器读取）。
    * ``latest.md``                     —— markdown 渲染版（人工阅读）。
    * ``history/<cycle>.json``          —— 归档（``also_update_history=True`` 时写入；
      ``<cycle>`` 以 bundle.cycles_included[-1] 为准，若为空用 ``generated_at``）。
    * ``latest_<universe>.json``        —— 阶段 G.1：每个 universe 的子视图 JSON，
      字段结构为 ``UniverseSubBundle.model_dump``；``also_write_per_universe=False``
      时跳过。老消费者可以完全无视这些文件。

    返回 ``(latest_json, latest_md)``（与 F 阶段签名兼容，不把 per-universe 路径
    塞进返回值以免下游 CLI 误解析）。
    """
    workspace_feedback_dir = Path(workspace_feedback_dir)
    workspace_feedback_dir.mkdir(parents=True, exist_ok=True)

    latest_json = workspace_feedback_dir / "latest.json"
    latest_md = workspace_feedback_dir / "latest.md"

    payload_json = bundle.model_dump_json(indent=2)
    latest_json.write_text(payload_json, encoding="utf-8")
    latest_md.write_text(bundle.to_markdown(), encoding="utf-8")

    if also_update_history:
        hist_dir = workspace_feedback_dir / "history"
        hist_dir.mkdir(parents=True, exist_ok=True)
        tail = bundle.cycles_included[-1] if bundle.cycles_included else None
        stamp = tail or bundle.generated_at.strftime("snap-%Y%m%d-%H%M%S")
        hist_path = hist_dir / f"{stamp}.json"
        hist_path.write_text(payload_json, encoding="utf-8")

    if also_write_per_universe and bundle.by_universe:
        for u, sub in bundle.by_universe.items():
            per_path = workspace_feedback_dir / f"latest_{u}.json"
            try:
                per_path.write_text(sub.model_dump_json(indent=2), encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                logger.warning("写 per-universe 子视图失败 %s: %s", per_path, exc)

    return latest_json, latest_md


# -------------------------------------------------------------- public: loader


def load_latest_feedback_bundle(workspace_feedback_dir: Path) -> FeedbackBundle | None:
    """从 ``workspace_feedback_dir/latest.json`` 加载最新 bundle；缺失返回 None。"""
    latest = Path(workspace_feedback_dir) / "latest.json"
    if not latest.exists():
        return None
    try:
        return FeedbackBundle.model_validate_json(latest.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("读取 latest feedback bundle 失败 %s: %s", latest, exc)
        return None
