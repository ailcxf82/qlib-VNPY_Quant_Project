"""
RD-Agent log → CandidateFactorPackage (契约 C1) 导出器。

本模块是 **L1 Factor Lab 唯一的 rdagent 源出口**。它做三件事：

1. 扫 RD-Agent 一次 run 的 log 目录（``log/<run-ts>/Loop_N/...``），
   用 **反射** 读 pickle 里的 ``Hypothesis`` / ``FactorTask`` /
   ``FactorFBWorkspace`` / ``HypothesisFeedback`` 对象——**不** ``import rdagent.xxx``
   类（防止 rdagent 依赖污染 factor_lab 包）。
2. 沿 ``FBWorkspace.workspace_path`` 拿到 ``git_ignore_folder/RD-Agent_workspace/<hash>/``
   里的 ``factor.py`` + ``result.h5``，把 ``result.h5`` 转成 parquet
   （schema 对齐 C1 合约）。
3. 组装 ``CandidateFactorPackage(source='rdagent', parent_loop=N, ...)`` 并落到
   ``factor_lab/workspace/candidates/<factor_id>/{factor.py, values.parquet, c1.json}``。

**决策与边界**：

* ``factor_id = rdagent_<sanitized_name>_<workspace_hash[:8]>``；同一份 code+data
  （= RD-Agent 内容哈希目录）始终得到相同 id → **内容级幂等**。
* ``lab_run_id = log_dir.name``（run 时间戳 basename）。
* 不做任何 IC / backtest / 有效性判断——那是 L2 的职责。``decision=False`` 的
  hypothesis 也会导出，只在 ``lab_metrics['rdagent_self_eval']`` 里标记 0.0，
  供 D.5 ``run_lab_cycle.py`` 按需过滤。
* 幂等：若目标 staging 目录已存在且 ``factor.py`` 内容一致、``values.parquet``
  schema 一致，则跳过重写；``--overwrite`` 可强制覆盖。
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from factor_lab.exporters.schema import CandidateFactorPackage

logger = logging.getLogger("factor_lab.exporters.rdagent_log_exporter")

# --------------------------------------------------------------------------- 常量

# RD-Agent log 树约定路径（相对 Loop_N/ 目录）
_REL_HYPOTHESIS = "direct_exp_gen/hypothesis generation"
_REL_EXP_GEN = "direct_exp_gen/experiment generation"
_REL_CODING = "coding"  # 下一级是 evo_loop_N/ "evolving code"
_REL_FEEDBACK = "feedback/feedback"
_EVO_LOOP_PREFIX = "evo_loop_"
_EVOLVING_CODE_SUBDIR = "evolving code"

# C1 schema 限制
_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_NAME_SANITIZE = re.compile(r"[^A-Za-z0-9_]+")
_HYPO_MAX_LEN = 4000
_FORM_MAX_LEN = 4000

# Loop 目录名形如 "Loop_0", "Loop_1", ...
_LOOP_DIR_PATTERN = re.compile(r"^Loop_(\d+)$")

# 默认路径
_DEFAULT_WORKSPACE_CANDIDATES_DIR = "factor_lab/workspace/candidates"
_DEFAULT_RDAGENT_WORKSPACE_ROOT = "git_ignore_folder/RD-Agent_workspace"
_DEFAULT_LOG_ROOT = "log"


# --------------------------------------------------------------------------- 数据结构


@dataclass
class LoopArtifacts:
    """一次 RD-Agent Loop_N 里抽出来的关键 pkl 对象集合（全部可选）。"""

    loop_index: int
    loop_dir: Path
    hypothesis_obj: Any = None  # 预期含 .hypothesis / .reason
    experiment_tasks: list[Any] = field(default_factory=list)  # list[FactorTask]
    evolving_workspaces: list[Any] = field(default_factory=list)  # list[FactorFBWorkspace]
    feedback_obj: Any = None  # 预期含 .decision: bool


@dataclass
class ExportResult:
    """一次 factor 导出的摘要。"""

    factor_id: str
    name: str
    loop_index: int
    target_dir: Path
    package_path: Path
    decision: bool | None
    action: str  # "written" | "skipped_idempotent" | "overwritten"


@dataclass
class ExportRunSummary:
    """一次 CLI run 的汇总。"""

    log_root: Path
    run_id: str
    loops_scanned: int
    exports: list[ExportResult] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "log_root": str(self.log_root),
            "run_id": self.run_id,
            "loops_scanned": self.loops_scanned,
            "n_exports": len(self.exports),
            "n_skipped": len(self.skipped),
            "exports": [
                {
                    "factor_id": e.factor_id,
                    "name": e.name,
                    "loop_index": e.loop_index,
                    "target_dir": str(e.target_dir),
                    "package_path": str(e.package_path),
                    "decision": e.decision,
                    "action": e.action,
                }
                for e in self.exports
            ],
            "skipped": list(self.skipped),
        }


# --------------------------------------------------------------------------- 工具函数


def _sanitize_name(raw_name: str) -> str:
    """把 rdagent 给出的 factor_name 规整到 C1 ``_NAME_PATTERN``。

    * 非法字符替换为 ``_``。
    * 如果首字符不是字母，强制前缀 ``F_``（rdagent 极少出现这种情况，但要兜底）。
    * 截断至 64。
    """
    if not raw_name:
        raise ValueError("factor_name 为空")
    s = _NAME_SANITIZE.sub("_", raw_name.strip())
    s = s.strip("_") or "factor"
    if not s[0].isalpha():
        s = "F_" + s
    s = s[:64]
    if not _NAME_PATTERN.match(s):
        raise ValueError(
            f"sanitize 后仍不符合 C1 name 规则: {s!r} (原: {raw_name!r})"
        )
    return s


def _truncate(text: str, limit: int) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    if not s:
        return ""
    if len(s) <= limit:
        return s
    # 保留尾部摘要哈希以便追溯
    tail = hashlib.sha1(s.encode("utf-8", errors="replace")).hexdigest()[:8]
    return s[: limit - 20] + f"...[truncated:{tail}]"


def _latest_pkl(dirpath: Path) -> Path | None:
    """在目录下找最新的 ``*.pkl`` 文件（递归一层子目录；RD-Agent log 习惯把 pkl
    嵌在 ``<pid>/<ts>.pkl`` 的结构下）。"""
    if not dirpath.exists() or not dirpath.is_dir():
        return None
    pkls: list[Path] = []
    for p in dirpath.rglob("*.pkl"):
        if p.is_file():
            pkls.append(p)
    if not pkls:
        return None
    pkls.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return pkls[0]


def _safe_pickle_load(p: Path) -> Any | None:
    try:
        with p.open("rb") as f:
            return pickle.load(f)
    except Exception as exc:  # noqa: BLE001
        logger.warning("pickle load failed %s: %s", p, exc)
        return None


def _getattr_or_key(obj: Any, field_name: str, default: Any = None) -> Any:
    """反射取属性/key：先 attr，再 dict[key]，都没有就返回 default。"""
    if obj is None:
        return default
    if hasattr(obj, field_name):
        return getattr(obj, field_name)
    if isinstance(obj, dict) and field_name in obj:
        return obj[field_name]
    return default


def _ensure_list(obj: Any) -> list[Any]:
    if obj is None:
        return []
    if isinstance(obj, (list, tuple)):
        return list(obj)
    return [obj]


# --------------------------------------------------------------------------- log 扫描


def scan_loops(log_run_dir: Path) -> list[LoopArtifacts]:
    """扫描一次 RD-Agent run 目录，返回每个 Loop_N 的关键 pkl 集合（按 loop index 升序）。"""
    if not log_run_dir.exists():
        raise FileNotFoundError(f"log run 目录不存在: {log_run_dir}")

    loops: list[LoopArtifacts] = []
    for child in sorted(log_run_dir.iterdir()):
        if not child.is_dir():
            continue
        m = _LOOP_DIR_PATTERN.match(child.name)
        if not m:
            continue
        loop_idx = int(m.group(1))

        # hypothesis
        hypo_pkl = _latest_pkl(child / _REL_HYPOTHESIS)
        hypo_obj = _safe_pickle_load(hypo_pkl) if hypo_pkl else None

        # experiment tasks
        exp_pkl = _latest_pkl(child / _REL_EXP_GEN)
        exp_tasks_raw = _safe_pickle_load(exp_pkl) if exp_pkl else None
        exp_tasks = _ensure_list(exp_tasks_raw)

        # coding evolving workspaces: 选 evo_loop_N 里 N 最大的那个
        coding_dir = child / _REL_CODING
        evolving_ws_raw: Any = None
        if coding_dir.exists():
            evo_subdirs = [
                d
                for d in coding_dir.iterdir()
                if d.is_dir() and d.name.startswith(_EVO_LOOP_PREFIX)
            ]

            def _evo_idx(p: Path) -> int:
                try:
                    return int(p.name[len(_EVO_LOOP_PREFIX) :])
                except ValueError:
                    return -1

            evo_subdirs.sort(key=_evo_idx, reverse=True)
            for evo in evo_subdirs:
                code_pkl = _latest_pkl(evo / _EVOLVING_CODE_SUBDIR)
                if code_pkl is None:
                    continue
                loaded = _safe_pickle_load(code_pkl)
                if loaded is not None:
                    evolving_ws_raw = loaded
                    break
        evolving_workspaces = _ensure_list(evolving_ws_raw)

        # feedback
        fb_pkl = _latest_pkl(child / _REL_FEEDBACK)
        fb_obj = _safe_pickle_load(fb_pkl) if fb_pkl else None

        loops.append(
            LoopArtifacts(
                loop_index=loop_idx,
                loop_dir=child,
                hypothesis_obj=hypo_obj,
                experiment_tasks=exp_tasks,
                evolving_workspaces=evolving_workspaces,
                feedback_obj=fb_obj,
            )
        )

    loops.sort(key=lambda x: x.loop_index)
    return loops


# --------------------------------------------------------------------------- 单个 factor 导出


def _extract_hypothesis_text(hypo_obj: Any) -> str:
    """从 hypothesis pkl 对象拼出 C1 的 hypothesis 字符串。"""
    if hypo_obj is None:
        return ""
    hyp = _getattr_or_key(hypo_obj, "hypothesis", "")
    reason = _getattr_or_key(hypo_obj, "reason", "")
    concise = _getattr_or_key(hypo_obj, "concise_justification", "")
    parts = [str(x).strip() for x in (hyp, reason, concise) if x]
    return "\n\n".join(p for p in parts if p)


def _extract_formulation_text(task_obj: Any) -> str:
    """从 FactorTask 对象拼出 C1 的 formulation 字符串。"""
    if task_obj is None:
        return ""
    pieces: list[str] = []
    form = _getattr_or_key(task_obj, "factor_formulation", "")
    if form:
        pieces.append(f"[formulation]\n{form}")
    desc = _getattr_or_key(task_obj, "factor_description", None) or _getattr_or_key(
        task_obj, "description", ""
    )
    if desc:
        pieces.append(f"[description]\n{desc}")
    variables = _getattr_or_key(task_obj, "variables", None)
    if variables:
        try:
            pieces.append("[variables]\n" + json.dumps(variables, ensure_ascii=False))
        except (TypeError, ValueError):
            pieces.append("[variables]\n" + repr(variables))
    return "\n\n".join(pieces)


def _resolve_workspace_path(
    fb_ws: Any,
    *,
    rdagent_workspace_root: Path,
) -> Path | None:
    """解析 FBWorkspace 对象的 workspace_path；做一点容错（相对路径、软链接、
    用项目根兜底）。"""
    raw = _getattr_or_key(fb_ws, "workspace_path", None)
    if raw is None:
        return None
    try:
        p = Path(str(raw))
    except Exception:  # noqa: BLE001
        return None
    if p.is_absolute() and p.exists():
        return p
    # 相对路径：相对项目根
    proj = rdagent_workspace_root.resolve().parents[1] if rdagent_workspace_root.exists() else Path.cwd()
    cand = (proj / p).resolve()
    if cand.exists():
        return cand
    # 用 basename 在 rdagent_workspace_root 下找
    by_name = rdagent_workspace_root / p.name
    if by_name.exists():
        return by_name
    return None


def _read_result_h5(result_h5: Path, expected_name: str) -> pd.DataFrame:
    """读 result.h5 → 规范 C1 schema：MultiIndex(datetime, instrument) + 单列 ``expected_name``(float64)。

    ``result.h5`` 约定：``key='data'``，MultiIndex，单列但列名可能与 ``expected_name``
    不一致（例如大小写漂移），这里做重命名。
    """
    if not result_h5.exists() or result_h5.stat().st_size < 64:
        raise ValueError(f"result.h5 不存在或过小: {result_h5}")
    df = pd.read_hdf(str(result_h5), key="data")
    if isinstance(df, pd.Series):
        df = df.to_frame()

    # 规范化 index
    if not isinstance(df.index, pd.MultiIndex) or df.index.nlevels != 2:
        raise ValueError(
            f"result.h5 必须带两级 MultiIndex，当前: {df.index!r}"
        )
    level_names = list(df.index.names)
    if level_names != ["datetime", "instrument"]:
        # 尝试自动识别
        lc = [str(n).lower() if n is not None else "" for n in level_names]
        try:
            dt_pos = next(i for i, n in enumerate(lc) if n in {"datetime", "date", "dt"})
            inst_pos = next(
                i for i, n in enumerate(lc) if n in {"instrument", "code", "symbol"}
            )
        except StopIteration as exc:
            raise ValueError(
                f"无法从 MultiIndex {level_names} 推断 (datetime, instrument)"
            ) from exc
        new_names = list(df.index.names)
        new_names[dt_pos] = "datetime"
        new_names[inst_pos] = "instrument"
        df.index = df.index.set_names(new_names)
        if list(df.index.names) != ["datetime", "instrument"]:
            df = df.reorder_levels(["datetime", "instrument"])

    # 规范化列
    if df.shape[1] == 0:
        raise ValueError(f"result.h5 没有数据列: {result_h5}")
    if df.shape[1] > 1:
        # 优先取 expected_name；否则取第一列
        if expected_name in df.columns:
            df = df[[expected_name]]
        else:
            df = df.iloc[:, :1]

    src_col = df.columns[0]
    if src_col != expected_name:
        df = df.rename(columns={src_col: expected_name})
    df[expected_name] = df[expected_name].astype("float64")
    df = df.sort_index()
    # 去重 (datetime, instrument) 重复
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="last")]
    return df


def _match_task_to_workspace(
    tasks: list[Any], workspaces: list[Any]
) -> list[tuple[Any, Any]]:
    """把 FactorTask 和 FBWorkspace 按 factor_name 配对。配不上的直接丢弃。"""
    pairs: list[tuple[Any, Any]] = []
    # 建 ws 索引：name → ws
    ws_by_name: dict[str, Any] = {}
    for ws in workspaces:
        target = _getattr_or_key(ws, "target_task", None)
        ws_name = _getattr_or_key(target, "factor_name", None) if target else None
        if ws_name:
            ws_by_name.setdefault(str(ws_name), ws)
    for t in tasks:
        t_name = _getattr_or_key(t, "factor_name", None)
        if not t_name:
            continue
        ws = ws_by_name.get(str(t_name))
        if ws is None:
            continue
        pairs.append((t, ws))
    return pairs


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _should_skip_idempotent(
    target_dir: Path,
    factor_py_text: str,
    values_df: pd.DataFrame,
    expected_name: str,
) -> bool:
    """若目标目录已存在、且 factor.py 内容一致、values.parquet 行数/shape 一致，
    判定为幂等已写，无需重写。"""
    if not target_dir.exists():
        return False
    code_p = target_dir / "factor.py"
    vals_p = target_dir / "values.parquet"
    pkg_p = target_dir / "c1.json"
    if not (code_p.exists() and vals_p.exists() and pkg_p.exists()):
        return False
    try:
        existing_code = code_p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    if existing_code != factor_py_text:
        return False
    try:
        existing_df = pd.read_parquet(vals_p)
    except Exception:  # noqa: BLE001
        return False
    if (
        list(existing_df.columns) != [expected_name]
        or existing_df.shape != values_df.shape
    ):
        return False
    return True


# --------------------------------------------------------------------------- 主流程


def export_loop(
    loop: LoopArtifacts,
    *,
    log_run_dir: Path,
    workspace_candidates_dir: Path,
    rdagent_workspace_root: Path,
    universe: str,
    now: datetime,
    overwrite: bool = False,
    lab_run_id: str | None = None,
) -> tuple[list[ExportResult], list[dict]]:
    """把一个 Loop_N 里的所有 (task, workspace) 对导成 C1 package。

    返回 ``(exports, skipped)``，``skipped`` 里记录每条跳过原因的字典，便于审计。
    """
    exports: list[ExportResult] = []
    skipped: list[dict] = []

    run_id = lab_run_id or log_run_dir.name
    hypo_text_full = _extract_hypothesis_text(loop.hypothesis_obj)
    feedback_decision = _getattr_or_key(loop.feedback_obj, "decision", None)
    try:
        feedback_decision_bool: bool | None = (
            bool(feedback_decision) if feedback_decision is not None else None
        )
    except Exception:  # noqa: BLE001
        feedback_decision_bool = None

    pairs = _match_task_to_workspace(loop.experiment_tasks, loop.evolving_workspaces)
    if not pairs:
        skipped.append(
            {
                "loop_index": loop.loop_index,
                "reason": "no_task_workspace_pairs",
                "n_tasks": len(loop.experiment_tasks),
                "n_workspaces": len(loop.evolving_workspaces),
            }
        )
        return exports, skipped

    for task, ws in pairs:
        raw_name = _getattr_or_key(task, "factor_name", None)
        try:
            name = _sanitize_name(str(raw_name))
        except ValueError as exc:
            skipped.append(
                {
                    "loop_index": loop.loop_index,
                    "reason": "bad_factor_name",
                    "raw_name": repr(raw_name),
                    "error": str(exc),
                }
            )
            continue

        # 1) 定位 workspace_path
        ws_path = _resolve_workspace_path(
            ws, rdagent_workspace_root=rdagent_workspace_root
        )
        # 即使 workspace_path 没有，也可能能从 file_dict 取 factor.py，但 result.h5 没
        # 了就完全没法导——必须要 workspace。
        if ws_path is None or not ws_path.exists():
            skipped.append(
                {
                    "loop_index": loop.loop_index,
                    "name": name,
                    "reason": "workspace_missing",
                    "raw_workspace_path": repr(_getattr_or_key(ws, "workspace_path")),
                }
            )
            continue

        factor_py_path = ws_path / "factor.py"
        result_h5_path = ws_path / "result.h5"
        if not factor_py_path.exists():
            # 退而求其次：file_dict['factor.py']
            fd = _getattr_or_key(ws, "file_dict", None)
            text_from_dict: str | None = None
            if isinstance(fd, dict):
                text_from_dict = fd.get("factor.py")
            if not text_from_dict:
                skipped.append(
                    {
                        "loop_index": loop.loop_index,
                        "name": name,
                        "reason": "factor_py_missing",
                        "workspace": str(ws_path),
                    }
                )
                continue
            factor_py_text = str(text_from_dict)
        else:
            factor_py_text = factor_py_path.read_text(
                encoding="utf-8", errors="replace"
            )
        if not result_h5_path.exists():
            skipped.append(
                {
                    "loop_index": loop.loop_index,
                    "name": name,
                    "reason": "result_h5_missing",
                    "workspace": str(ws_path),
                }
            )
            continue

        # 2) 读 result.h5 → 规范 df
        try:
            df = _read_result_h5(result_h5_path, expected_name=name)
        except ValueError as exc:
            skipped.append(
                {
                    "loop_index": loop.loop_index,
                    "name": name,
                    "reason": "result_h5_unreadable",
                    "error": str(exc),
                    "workspace": str(ws_path),
                }
            )
            continue
        if df.empty:
            skipped.append(
                {
                    "loop_index": loop.loop_index,
                    "name": name,
                    "reason": "result_h5_empty",
                    "workspace": str(ws_path),
                }
            )
            continue

        # 3) factor_id = rdagent_<name>_<workspace_hash[:8]>
        ws_hash = ws_path.name  # RD-Agent 内容哈希目录名
        short_hash = re.sub(r"[^0-9a-f]", "", ws_hash.lower())[:8]
        if len(short_hash) < 8:
            short_hash = (short_hash + hashlib.sha256(ws_hash.encode()).hexdigest())[:8]
        factor_id = f"rdagent_{name}_{short_hash}"

        # 4) 目标目录
        target_dir = workspace_candidates_dir / factor_id
        code_out = target_dir / "factor.py"
        values_out = target_dir / "values.parquet"
        c1_out = target_dir / "c1.json"

        idempotent = _should_skip_idempotent(
            target_dir, factor_py_text, df, expected_name=name
        )
        if idempotent and not overwrite:
            exports.append(
                ExportResult(
                    factor_id=factor_id,
                    name=name,
                    loop_index=loop.loop_index,
                    target_dir=target_dir,
                    package_path=c1_out,
                    decision=feedback_decision_bool,
                    action="skipped_idempotent",
                )
            )
            continue

        # 5) 落盘 factor.py + values.parquet（原子性：先写 tmp 再 replace）
        target_dir.mkdir(parents=True, exist_ok=True)
        tmp_code = code_out.with_suffix(code_out.suffix + ".tmp")
        tmp_vals = values_out.with_suffix(values_out.suffix + ".tmp")
        tmp_code.write_text(factor_py_text, encoding="utf-8")
        df.to_parquet(tmp_vals, engine="pyarrow")
        tmp_code.replace(code_out)
        tmp_vals.replace(values_out)

        # 6) 造 CandidateFactorPackage
        dt_index = df.index.get_level_values("datetime")
        start_date = pd.Timestamp(dt_index.min()).date()
        end_date = pd.Timestamp(dt_index.max()).date()

        hypothesis_text = _truncate(
            hypo_text_full or f"(rdagent loop {loop.loop_index}; 无 hypothesis pkl)",
            _HYPO_MAX_LEN,
        )
        formulation_text = _truncate(
            _extract_formulation_text(task) or f"(rdagent task={name}; 无 formulation)",
            _FORM_MAX_LEN,
        )

        lab_metrics: dict[str, float] = {
            "non_null_ratio": float(df[name].notna().mean()),
            "n_rows": float(len(df)),
            "n_instruments": float(df.index.get_level_values("instrument").nunique()),
        }
        if feedback_decision_bool is not None:
            lab_metrics["rdagent_self_eval"] = 1.0 if feedback_decision_bool else 0.0

        pkg = CandidateFactorPackage(
            factor_id=factor_id,
            name=name,
            source="rdagent",
            hypothesis=hypothesis_text,
            formulation=formulation_text,
            code_path=code_out,
            values_path=values_out,
            universe=universe,
            date_range=(start_date, end_date),
            lab_metrics=lab_metrics,
            parent_loop=loop.loop_index,
            created_at=now,
            lab_run_id=run_id,
        )
        # 校验物料对齐 schema
        pkg.validate_artifacts(check_parquet_schema=True)

        # 7) c1.json（持久化，方便 D.5 / 审计离线读）
        c1_dict = json.loads(pkg.model_dump_json())
        # lab_run_dir 侧录，便于反查
        c1_dict["_meta"] = {
            "rdagent_workspace_path": str(ws_path),
            "rdagent_workspace_hash": ws_hash,
            "rdagent_log_run_dir": str(log_run_dir),
            "exported_at": now.isoformat(),
        }
        tmp_c1 = c1_out.with_suffix(c1_out.suffix + ".tmp")
        tmp_c1.write_text(
            json.dumps(c1_dict, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        tmp_c1.replace(c1_out)

        action = "overwritten" if idempotent else "written"
        exports.append(
            ExportResult(
                factor_id=factor_id,
                name=name,
                loop_index=loop.loop_index,
                target_dir=target_dir,
                package_path=c1_out,
                decision=feedback_decision_bool,
                action=action,
            )
        )

    return exports, skipped


def export_rdagent_run(
    *,
    log_run_dir: Path,
    workspace_candidates_dir: Path,
    rdagent_workspace_root: Path,
    universe: str = "csi300",
    overwrite: bool = False,
    now: datetime | None = None,
    lab_run_id: str | None = None,
) -> ExportRunSummary:
    """导出一次 RD-Agent run 目录下所有 loop 的候选因子。"""
    log_run_dir = Path(log_run_dir)
    workspace_candidates_dir = Path(workspace_candidates_dir)
    rdagent_workspace_root = Path(rdagent_workspace_root)
    now = now or datetime.now(timezone.utc)
    lab_run_id = lab_run_id or log_run_dir.name

    workspace_candidates_dir.mkdir(parents=True, exist_ok=True)

    loops = scan_loops(log_run_dir)
    logger.info(
        "扫到 %d 个 Loop_N 目录 @ %s（lab_run_id=%s）",
        len(loops),
        log_run_dir,
        lab_run_id,
    )

    summary = ExportRunSummary(
        log_root=log_run_dir,
        run_id=lab_run_id,
        loops_scanned=len(loops),
    )

    for loop in loops:
        exports, skipped = export_loop(
            loop,
            log_run_dir=log_run_dir,
            workspace_candidates_dir=workspace_candidates_dir,
            rdagent_workspace_root=rdagent_workspace_root,
            universe=universe,
            now=now,
            overwrite=overwrite,
            lab_run_id=lab_run_id,
        )
        summary.exports.extend(exports)
        summary.skipped.extend(skipped)

    logger.info(
        "run %s 完成：exports=%d skipped=%d",
        lab_run_id,
        len(summary.exports),
        len(summary.skipped),
    )
    return summary


def export_rdagent_log_tree(
    *,
    log_root: Path,
    workspace_candidates_dir: Path,
    rdagent_workspace_root: Path,
    run_filter: str | None = None,
    universe: str = "csi300",
    overwrite: bool = False,
    now: datetime | None = None,
) -> list[ExportRunSummary]:
    """扫 ``log/`` 根目录下所有（或匹配的）RD-Agent run 时间戳目录。"""
    log_root = Path(log_root)
    if not log_root.exists():
        raise FileNotFoundError(f"log 根目录不存在: {log_root}")

    run_dirs: list[Path] = []
    for child in sorted(log_root.iterdir()):
        if not child.is_dir():
            continue
        # 只挑带 Loop_N 子目录的目录；避免把不相关目录当作 run
        if not any(_LOOP_DIR_PATTERN.match(p.name) for p in child.iterdir() if p.is_dir()):
            continue
        if run_filter and run_filter not in child.name:
            continue
        run_dirs.append(child)

    summaries: list[ExportRunSummary] = []
    for rd in run_dirs:
        summary = export_rdagent_run(
            log_run_dir=rd,
            workspace_candidates_dir=workspace_candidates_dir,
            rdagent_workspace_root=rdagent_workspace_root,
            universe=universe,
            overwrite=overwrite,
            now=now,
        )
        summaries.append(summary)
    return summaries


# --------------------------------------------------------------------------- 便捷 import 导出


__all__ = [
    "LoopArtifacts",
    "ExportResult",
    "ExportRunSummary",
    "scan_loops",
    "export_loop",
    "export_rdagent_run",
    "export_rdagent_log_tree",
]
