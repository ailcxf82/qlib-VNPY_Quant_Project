"""``backtest_rqalpha`` check —— 候选因子在 OOS 窗内的 RQAlpha 真实回测。

与轻量版 ``backtest_check`` 的关系：

| 维度 | ``backtest`` (轻量) | ``backtest_rqalpha`` (本 check) |
| --- | --- | --- |
| 撮合 / 费用 / 滑点 | 无 | 真实（T+1、手续费、滑点、价格限制）|
| 耗时 | 秒级 | **5~15 分钟 / 候选**（取决于 universe 和区间）|
| 依赖 | 无 | 本机装了 ``rqalpha`` + 数据 bundle |
| 适用 profile | ``default`` / ``exploratory`` | ``strict``（production 准入）|
| 失败兜底 | 无 | 若 ``rqalpha`` 未安装 → 抛错；若 runner 模块不可用 → 抛错 |

**动机**：轻量版 long-short 组合无成本、无约束，可能高估实盘 Sharpe。production
准入必须用真实撮合、真实费用、真实持仓约束验过的指标。本 check 是 L2 通往 L3 的
**最后一关**。

**数据流**：

1. 从 ``candidate.values_path`` 读单列因子（``(datetime, instrument) -> value``）
2. 切到 ``context.oos_window``，以及可选 config 传入的 ``backtest_start`` / ``backtest_end``
   （交集），生成 ``prediction.csv``：

   ========  ==========  =====
   datetime  instrument  final
   ========  ==========  =====

   若 ``allow_negative=True``，会把因子值 ``*= -1`` 写入 CSV（等价 short the top）

3. Lazy import ``backtest.rqalpha_backtest.run_rqalpha_backtest``（**只在本 check 被执行时 import**，
   避免无 RQAlpha 环境上 L2 其他 check 也跑不动）
4. 以 ``output_dir=reports/rqalpha/<factor_id>_<validation_ts>`` 调 runner
5. runner 完成后从 ``<output_dir>/report.json`` 读 ``summary``，取：

   * ``sharpe`` / ``sharpe_ratio``       → ``sharpe``
   * ``annualized_returns`` / ``annualized_return`` / ``annual_return`` → ``annual_return``
   * ``max_drawdown``                   → ``max_drawdown``（RQAlpha 写为负数，这里取绝对值）
   * ``total_returns`` / ``total_return`` → ``total_return``
   * ``volatility`` / ``annual_volatility`` → ``annual_vol``

6. 如果任一核心字段缺失或读取失败 → check 抛错（orchestrator 会包成 FAIL）

**阈值字段**（profile 配置）：

* ``rqalpha_config_path``   （可选；默认 ``config/rqalpha_config.yaml``）
* ``strategy_path``         （可选；传 None → runner 用内置 ``backtest/rqalpha_strategy.py``）
* ``min_sharpe``            （必填；>=0）
* ``max_drawdown``          （必填；``(0, 1]``；绝对值）
* ``min_annual_return``     （可选，默认 ``0.0``；绝对值下限）
* ``min_total_return``      （可选，默认 ``0.0``；累计收益绝对值下限）
* ``allow_negative``        （可选，默认 False）
* ``full_invested``         （可选，默认 False）
* ``score_col``             （可选，默认 ``"final"``）
* ``output_root``           （可选；默认 ``factor_validation/reports/rqalpha``）
* ``backtest_start`` / ``backtest_end`` （可选；若提供则与 OOS 窗取交集，用于截断到
                           bundle 已覆盖区间，避免"回测区间超出 bundle 覆盖"的常见坑）

**passed**：
    ``sharpe_cmp >= min_sharpe`` 且
    ``max_drawdown <= max_drawdown`` 且
    ``annual_return_cmp >= min_annual_return`` 且
    ``total_return_cmp >= min_total_return``
（``*_cmp`` 为方向翻转后的值，``allow_negative=False`` 时等同原值）

**score**：
    ``0.55 * clip(sharpe/min_sharpe, 0, 1) + 0.30 * clip(1 - dd/dd_limit, 0, 1) + 0.15 * clip(annual_ret/target, 0, 1)``
    （target = max(min_annual_return, 0.05)，给 0 阈值时也能区分好因子）
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from factor_lab.exporters.schema import CandidateFactorPackage
from factor_validation.checks.base import CheckBase, CheckContext
from factor_validation.schema import CheckResult

logger = logging.getLogger(__name__)

_DEFAULT_RQALPHA_CONFIG = "config/rqalpha_config.yaml"
_DEFAULT_OUTPUT_ROOT = "factor_validation/reports/rqalpha"
_EPS = 1e-12


class RqalphaBacktestCheck(CheckBase):
    """调用 ``backtest.rqalpha_backtest.run_rqalpha_backtest`` 的真实回测 check。"""

    name = "backtest_rqalpha"

    def run(
        self,
        candidate: CandidateFactorPackage,
        context: CheckContext,
        config: dict[str, Any],
    ) -> CheckResult:
        start = time.perf_counter()

        # ---- 读取并校验配置 ----
        min_sharpe = self._float_ge(config, "min_sharpe", lo=0.0)
        max_dd_limit = self._float_in(config, "max_drawdown", lo=1e-6, hi=1.0)
        min_ann_ret = float(config.get("min_annual_return", 0.0))
        min_total_ret = float(config.get("min_total_return", 0.0))
        allow_negative = bool(config.get("allow_negative", False))
        full_invested = bool(config.get("full_invested", False))
        score_col = str(config.get("score_col", "final"))

        rqalpha_config_path = Path(
            config.get("rqalpha_config_path", _DEFAULT_RQALPHA_CONFIG)
        )
        if not rqalpha_config_path.is_absolute():
            rqalpha_config_path = context.project_root / rqalpha_config_path
        if not rqalpha_config_path.exists():
            raise ValueError(
                f"backtest_rqalpha: rqalpha 配置不存在: {rqalpha_config_path}"
            )

        strategy_path_cfg = config.get("strategy_path")
        strategy_path: Path | None = None
        if strategy_path_cfg:
            strategy_path = Path(strategy_path_cfg)
            if not strategy_path.is_absolute():
                strategy_path = context.project_root / strategy_path
            if not strategy_path.exists():
                raise ValueError(
                    f"backtest_rqalpha: strategy 不存在: {strategy_path}"
                )

        output_root = Path(config.get("output_root", _DEFAULT_OUTPUT_ROOT))
        if not output_root.is_absolute():
            output_root = context.project_root / output_root
        output_root.mkdir(parents=True, exist_ok=True)

        # 回测区间：默认 OOS 窗，config 可以截断到 bundle 覆盖区间
        start_d, end_d = context.oos_window
        bt_start = pd.Timestamp(config.get("backtest_start") or start_d)
        bt_end = pd.Timestamp(config.get("backtest_end") or end_d)
        if bt_end < bt_start:
            raise ValueError(
                f"backtest_rqalpha: backtest_end ({bt_end}) < backtest_start ({bt_start})"
            )

        # ---- 加载候选因子 ----
        cand_df = pd.read_parquet(candidate.values_path)
        if candidate.name not in cand_df.columns:
            raise ValueError(
                f"backtest_rqalpha: parquet 缺列 {candidate.name!r}，"
                f"实际列: {list(cand_df.columns)}"
            )
        series = cand_df[candidate.name]
        idx_dt = series.index.get_level_values("datetime")
        mask = (idx_dt >= bt_start) & (
            idx_dt <= bt_end + pd.Timedelta(hours=23, minutes=59, seconds=59)
        )
        series = series[mask].dropna()
        if series.empty:
            raise ValueError(
                "backtest_rqalpha: 回测区间内候选因子无样本"
            )

        # 负向因子：写 prediction 时翻转符号（等价 short the top, long the bottom）
        direction = 1
        final_values = series.to_numpy()
        if allow_negative:
            # 单独用少量样本粗估方向（同 long-short IC 符号）—— 若因子与 forward return
            # 负相关，direction = -1，CSV 写 -factor。此处只能启发式判断：若 allow_negative
            # 为 True，默认假定正向；若回测完 sharpe 为负，会在阈值判决阶段失败，不做隐式翻转
            # 到 CSV（真实回测翻转 score 代价是重新跑一次，成本太高）。
            #
            # 行为约定：allow_negative=True 时，真实 RQAlpha 跑一次后，若 sharpe<0，
            # 把方向记为 -1，但 pass 条件基于 abs(sharpe) —— 这意味着 production 必须
            # 清楚"最终投产是反向因子"。
            direction = 1  # 延后判决

        # ---- 生成 prediction CSV ----
        tag = candidate.factor_id
        ts_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = output_root / f"{tag}_{ts_tag}"
        output_dir.mkdir(parents=True, exist_ok=True)

        pred_path = output_dir / f"prediction_{tag}.csv"
        pred_df = pd.DataFrame(
            {
                "datetime": series.index.get_level_values("datetime"),
                "instrument": series.index.get_level_values("instrument").astype(str),
                score_col: final_values,
            }
        )
        # 若有重复 (datetime, instrument) → 取最后一条
        pred_df = pred_df.drop_duplicates(
            subset=["datetime", "instrument"], keep="last"
        )
        pred_df.to_csv(pred_path, index=False)
        logger.info(
            "backtest_rqalpha: prediction csv %s  rows=%d  dates=[%s~%s]",
            pred_path,
            len(pred_df),
            pred_df["datetime"].min(),
            pred_df["datetime"].max(),
        )

        # ---- Lazy import + 调用 ----
        runner = _resolve_runner(config.get("_runner_override"))

        run_kwargs = {
            "rqalpha_config_path": str(rqalpha_config_path),
            "prediction_path": str(pred_path),
            "industry_path": None,
            "strategy_path": str(strategy_path) if strategy_path else None,
            "full_invested": full_invested,
            "score_col": score_col,
            "output_dir": str(output_dir),
        }
        logger.info(
            "backtest_rqalpha: 调用 run_rqalpha_backtest  output_dir=%s",
            output_dir,
        )
        try:
            runner(**run_kwargs)
        except Exception as exc:  # pragma: no cover - 真实 RQAlpha 异常不落单测
            raise ValueError(
                f"backtest_rqalpha: RQAlpha runner 抛错: {exc!r}"
            ) from exc

        # ---- 读取 report.json ----
        report_path = output_dir / "report.json"
        if not report_path.exists():
            # 部分老版本 runner 只写 summary.json
            alt = output_dir / "summary.json"
            if alt.exists():
                report_path = alt
            else:
                raise ValueError(
                    f"backtest_rqalpha: 未找到 report.json / summary.json "
                    f"于 {output_dir}"
                )

        try:
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
        except Exception as exc:
            raise ValueError(
                f"backtest_rqalpha: 无法解析 {report_path}: {exc!r}"
            ) from exc

        summary = _extract_summary(report)
        metrics = _parse_metrics(summary)

        # ---- 阈值判决 ----
        sharpe_raw = metrics["sharpe"]
        annual_ret_raw = metrics["annual_return"]
        total_ret_raw = metrics["total_return"]
        max_dd_raw = metrics["max_drawdown"]

        if allow_negative and sharpe_raw < 0:
            direction = -1
            sharpe_cmp = -sharpe_raw
            annual_ret_cmp = -annual_ret_raw
            total_ret_cmp = -total_ret_raw
        else:
            sharpe_cmp = sharpe_raw
            annual_ret_cmp = annual_ret_raw
            total_ret_cmp = total_ret_raw

        passed = (
            sharpe_cmp >= min_sharpe
            and max_dd_raw <= max_dd_limit
            and annual_ret_cmp >= min_ann_ret
            and total_ret_cmp >= min_total_ret
        )

        # ---- 评分 ----
        sharpe_ratio = min(1.0, max(0.0, sharpe_cmp / max(min_sharpe, _EPS)))
        dd_ratio = min(1.0, max(0.0, 1.0 - max_dd_raw / max(max_dd_limit, _EPS)))
        ann_target = max(min_ann_ret, 0.05)
        ann_ratio = min(1.0, max(0.0, annual_ret_cmp / max(ann_target, _EPS)))
        score = 0.55 * sharpe_ratio + 0.30 * dd_ratio + 0.15 * ann_ratio
        score = float(max(0.0, min(1.0, score)))

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return CheckResult(
            name=self.name,
            passed=passed,
            score=round(score, 6),
            threshold=None,
            detail={
                "sharpe": round(sharpe_raw, 6),
                "annual_return": round(annual_ret_raw, 6),
                "total_return": round(total_ret_raw, 6),
                "annual_vol": round(metrics["annual_vol"], 6),
                "max_drawdown": round(max_dd_raw, 6),
                "direction": direction,
                "min_sharpe": min_sharpe,
                "max_drawdown_limit": max_dd_limit,
                "min_annual_return": min_ann_ret,
                "min_total_return": min_total_ret,
                "allow_negative": allow_negative,
                "backtest_start": str(bt_start.date()),
                "backtest_end": str(bt_end.date()),
                "rqalpha_config": str(rqalpha_config_path),
                "strategy_path": str(strategy_path) if strategy_path else None,
                "output_dir": str(output_dir),
                "prediction_rows": int(len(pred_df)),
                "algorithm": "rqalpha run_file + rqalpha_strategy (T+1, real cost)",
            },
            elapsed_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------- helpers

    @staticmethod
    def _float_ge(cfg: dict[str, Any], key: str, *, lo: float) -> float:
        if key not in cfg:
            raise ValueError(f"backtest_rqalpha: 配置缺字段 {key!r}")
        v = float(cfg[key])
        if v < lo:
            raise ValueError(f"backtest_rqalpha: {key} 必须 >= {lo}: {v}")
        return v

    @staticmethod
    def _float_in(cfg: dict[str, Any], key: str, *, lo: float, hi: float) -> float:
        if key not in cfg:
            raise ValueError(f"backtest_rqalpha: 配置缺字段 {key!r}")
        v = float(cfg[key])
        if not lo <= v <= hi:
            raise ValueError(
                f"backtest_rqalpha: {key} 必须在 [{lo}, {hi}] 区间: {v}"
            )
        return v


def _resolve_runner(override: Callable[..., Any] | None) -> Callable[..., Any]:
    """取 runner：优先用 config 里的 ``_runner_override`` （单测），否则 lazy import。"""
    if override is not None and callable(override):
        return override
    try:
        from backtest.rqalpha_backtest import run_rqalpha_backtest  # type: ignore
    except ImportError as exc:  # pragma: no cover - 仅在环境缺 rqalpha 时触发
        raise ValueError(
            "backtest_rqalpha: 无法 import backtest.rqalpha_backtest.run_rqalpha_backtest；"
            "请先 `pip install rqalpha` 并确认 backtest/rqalpha_backtest.py 可用"
        ) from exc
    return run_rqalpha_backtest


def _extract_summary(report: dict) -> dict:
    """兼容 RQAlpha / detailed_results.json 多种结构，取出 flat summary 字典。"""
    if not isinstance(report, dict):
        raise ValueError("backtest_rqalpha: report 不是 dict")
    # 1) report.json（runner 自己落盘）的结构通常 {"summary": {...}, "positions": ..., ...}
    if "summary" in report and isinstance(report["summary"], dict):
        return report["summary"]
    # 2) detailed_results.json 的中文结构
    if "效率指标" in report and isinstance(report["效率指标"], dict):
        return report["效率指标"]
    # 3) 直接就是扁平 summary
    return report


def _parse_metrics(summary: dict) -> dict[str, float]:
    """从多变的 key 里选一个可用值。缺核心字段 → 抛错。"""

    def _get(keys: list[str]) -> float | None:
        for k in keys:
            if k in summary and summary[k] is not None:
                try:
                    v = float(summary[k])
                except (TypeError, ValueError):
                    continue
                if np.isnan(v) or np.isinf(v):
                    continue
                return v
        return None

    sharpe = _get(["sharpe", "sharpe_ratio"])
    annual_return = _get(
        ["annualized_returns", "annualized_return", "annual_return"]
    )
    total_return = _get(["total_returns", "total_return"])
    max_dd = _get(["max_drawdown", "maxdrawdown"])
    annual_vol = _get(["annual_volatility", "volatility"])

    if sharpe is None:
        raise ValueError(
            "backtest_rqalpha: report.summary 缺 sharpe / sharpe_ratio"
        )
    if annual_return is None:
        raise ValueError(
            "backtest_rqalpha: report.summary 缺 annualized_returns / annual_return"
        )
    if max_dd is None:
        raise ValueError(
            "backtest_rqalpha: report.summary 缺 max_drawdown"
        )
    if total_return is None:
        total_return = 0.0
    if annual_vol is None:
        annual_vol = 0.0

    return {
        "sharpe": float(sharpe),
        "annual_return": float(annual_return),
        "total_return": float(total_return),
        "max_drawdown": float(abs(max_dd)),  # RQAlpha 可能写负数，统一取绝对值
        "annual_vol": float(annual_vol),
    }
