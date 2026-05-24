"""update_data_window.py

每次更新 Qlib 数据后运行，自动把所有配置文件里的时间窗推进到最新数据日期。

用法::

    # 指定新数据截止日（推荐）
    python scripts/update_data_window.py --data-end 2026-04-23

    # 从 Qlib 数据目录自动探测最新交易日（需要 qlib 已安装）
    python scripts/update_data_window.py --auto

    # 预览改什么，不真正写文件
    python scripts/update_data_window.py --data-end 2026-04-23 --dry-run

日期划分策略（固定规则，每次自动计算）::

    data_end   = 你的最新数据日（如 2026-04-23）
    test_start = data_end 所在年份的 1 月 1 日（如 2026-01-01）
    valid_end  = test_start - 1 天                （如 2025-12-31）
    valid_start= test_start - 6 个月              （如 2025-07-01）
    train_end  = valid_start - 1 天               （如 2025-06-30）
    fit_end    = train_end（标准化器拟合截止，与训练集对齐）
    train_start= 2022-01-01（固定，历史数据起点）
    test_end    = 不晚于 data_end 的最后一根 Qlib 交易日
    backtest_end= test_end 的前一交易日（PortAna 最后一步需要 calendar[i+1]）
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, NamedTuple

# ── 项目根 ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]

# ── 需要更新的文件（相对路径）──────────────────────────────────────────────
TARGET_FILES = [
    "config/factor_lab.yaml",
    "rdagent_overrides/factor_template/conf_baseline.yaml",
    "rdagent_overrides/factor_template/conf_combined_factors.yaml",
    "rdagent_overrides/factor_template/conf_combined_factors_sota_model.yaml",
    "rdagent_overrides/model_template/conf_baseline_factors_model.yaml",
    "rdagent_overrides/model_template/conf_sota_factors_model.yaml",
]

TRAIN_START = date(2022, 1, 1)
DEFAULT_PROVIDER_URI = "/mnt/d/qlib_data/qlib_data"


# ── 日期窗口计算 ────────────────────────────────────────────────────────────
class DateWindow(NamedTuple):
    data_end: date
    fit_end: date
    train_start: date
    train_end: date
    valid_start: date
    valid_end: date
    test_start: date
    test_end: date
    backtest_end: date

    def show(self) -> str:
        lines = [
            "  数据窗口  : 2022-01-01 ～ " + fmt(self.data_end),
            "  训练集    : " + fmt(self.train_start) + " ～ " + fmt(self.train_end),
            "  验证集    : " + fmt(self.valid_start) + " ～ " + fmt(self.valid_end),
            "  测试集    : " + fmt(self.test_start)  + " ～ " + fmt(self.test_end),
            "  PortAna   : " + fmt(self.test_start)  + " ～ " + fmt(self.backtest_end),
            "  fit_end   : " + fmt(self.fit_end) + "（标准化器拟合截止）",
        ]
        return "\n".join(lines)


class FileUpdateResult(NamedTuple):
    path: Path
    rel_path: str
    diffs: list[str]
    missing: bool = False


class DataWindowUpdateResult(NamedTuple):
    window: DateWindow
    files: list[FileUpdateResult]
    dry_run: bool

    @property
    def changed_count(self) -> int:
        return sum(1 for item in self.files if item.diffs)


def fmt(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def _ts_to_date(ts) -> date:
    if hasattr(ts, "date"):
        return ts.date()
    return date.fromisoformat(str(ts)[:10])


def safe_backtest_end_index(start_idx: int, end_idx: int) -> int:
    """PortAna 最后一步 get_step_time 需要 calendar[end_idx+1]，故回测 inclusive end 至少比 test 少 1 根。"""
    if end_idx <= start_idx:
        raise ValueError(
            f"not enough calendar bars for PortAna: start_idx={start_idx}, end_idx={end_idx}"
        )
    return end_idx - 1


def resolve_trading_endpoints(
    test_start: date,
    data_end: date,
    provider_uri: str | None = None,
) -> tuple[date, date]:
    """Return (test_end, backtest_end) on the Qlib day calendar."""
    uri = provider_uri or os.environ.get("QLIB_PROVIDER_URI", DEFAULT_PROVIDER_URI)
    try:
        import qlib
        from qlib.config import REG_CN
        from qlib.data.data import Cal

        qlib.init(provider_uri=uri, region=REG_CN)
        cal = Cal.calendar(freq="day", future=True)
        _, _, start_idx, end_idx = Cal.locate_index(
            str(test_start), str(data_end), freq="day", future=True
        )
        safe_backtest_end_index(start_idx, end_idx)
        return _ts_to_date(cal[end_idx]), _ts_to_date(cal[safe_backtest_end_index(start_idx, end_idx)])
    except Exception as exc:
        print(f"[warn] Qlib calendar resolve failed ({exc}); using calendar-day fallback")
        test_end = data_end
        backtest_end = data_end - timedelta(days=1)
        return test_end, backtest_end


def calc_window(data_end: date, *, provider_uri: str | None = None) -> DateWindow:
    test_start = date(data_end.year, 1, 1)
    valid_end = test_start - timedelta(days=1)
    # valid_start = 6 个月前
    vs_month = valid_end.month - 5           # 12 - 5 = 7 → 7 月
    vs_year = valid_end.year
    if vs_month <= 0:
        vs_month += 12
        vs_year -= 1
    valid_start = date(vs_year, vs_month, 1)
    train_end = valid_start - timedelta(days=1)
    fit_end = train_end
    test_end, backtest_end = resolve_trading_endpoints(test_start, data_end, provider_uri)
    return DateWindow(
        data_end=data_end,
        fit_end=fit_end,
        train_start=TRAIN_START,
        train_end=train_end,
        valid_start=valid_start,
        valid_end=valid_end,
        test_start=test_start,
        test_end=test_end,
        backtest_end=backtest_end,
    )


# ── 替换规则（每条规则：正则 → 新值） ──────────────────────────────────────
def build_replacements(w: DateWindow) -> list[tuple[re.Pattern, str]]:
    """返回 (pattern, replacement) 列表，按顺序应用到文件内容。"""
    # 匹配任何 YYYY-MM-DD 格式的日期字符串，用具名组方便替换
    _D = r"\d{4}-\d{2}-\d{2}"

    rules = [
        # fit_end_time
        (r"(fit_end_time:\s*)(" + _D + r")",      r"\g<1>" + fmt(w.fit_end)),
        # PortAna backtest end（8 空格缩进，仅此一处；勿用可变行数正则，易被注释行干扰）
        (r"(^        end_time:\s*)(" + _D + r")", r"\g<1>" + fmt(w.backtest_end)),
        # data_handler end_time（4 空格缩进，勿与 backtest 的 end_time 混用同一规则）
        (r"(^    end_time:\s*)(" + _D + r")",      r"\g<1>" + fmt(w.test_end)),
        # factor_lab.yaml time_window end
        (r"(end:\s*\"?)(" + _D + r")(\"?)",       r"\g<1>" + fmt(w.test_end) + r"\g<3>"),
        # train / valid / test segments
        (r"(train:\s*\[)(" + _D + r")(,\s*)(" + _D + r")(\])",
         r"\g<1>" + fmt(w.train_start) + r"\g<3>" + fmt(w.train_end) + r"\g<5>"),
        (r"(valid:\s*\[)(" + _D + r")(,\s*)(" + _D + r")(\])",
         r"\g<1>" + fmt(w.valid_start) + r"\g<3>" + fmt(w.valid_end) + r"\g<5>"),
        (r"(test:\s*\[)(" + _D + r")(,\s*)(" + _D + r")(\])",
         r"\g<1>" + fmt(w.test_start)  + r"\g<3>" + fmt(w.test_end)  + r"\g<5>"),
        # backtest start_time（port_analysis_config 块里）
        (r"(backtest:\n(?:[^\n]*\n)*?\s*start_time:\s*)(" + _D + r")",
         r"\g<1>" + fmt(w.test_start)),
    ]
    return [(re.compile(p, re.MULTILINE), r) for p, r in rules]


def update_file(path: Path, w: DateWindow, dry_run: bool) -> list[str]:
    """更新单个文件，返回实际改动的行描述列表。"""
    original = path.read_text(encoding="utf-8")
    content  = original

    # 逐条规则替换（只替换值变化的部分）
    for pattern, repl in build_replacements(w):
        content = pattern.sub(repl, content)

    if content == original:
        return []

    # 收集改动行（简单对比）
    diffs = []
    for i, (old_line, new_line) in enumerate(
        zip(original.splitlines(), content.splitlines()), start=1
    ):
        if old_line != new_line:
            diffs.append(f"  行{i:4d}: {old_line.strip()} → {new_line.strip()}")

    if not dry_run:
        path.write_text(content, encoding="utf-8")

    return diffs


def apply_data_window(
    data_end: date,
    *,
    dry_run: bool = False,
    root: Path = ROOT,
    target_files: Iterable[str] = TARGET_FILES,
    provider_uri: str | None = None,
) -> DataWindowUpdateResult:
    """Apply the derived data window to all configured target files."""
    w = calc_window(data_end, provider_uri=provider_uri)
    results: list[FileUpdateResult] = []
    for rel in target_files:
        path = root / rel
        if not path.exists():
            results.append(FileUpdateResult(path=path, rel_path=rel, diffs=[], missing=True))
            continue
        diffs = update_file(path, w, dry_run)
        results.append(FileUpdateResult(path=path, rel_path=rel, diffs=diffs, missing=False))
    return DataWindowUpdateResult(window=w, files=results, dry_run=dry_run)


def print_update_result(result: DataWindowUpdateResult) -> None:
    print()
    print("=" * 60)
    print("  数据窗口更新" + ("（DRY-RUN，不写文件）" if result.dry_run else ""))
    print("=" * 60)
    print(result.window.show())
    print()

    for item in result.files:
        if item.missing:
            print(f"  [跳过] {item.rel_path}（文件不存在）")
            continue
        if item.diffs:
            tag = "[预览]" if result.dry_run else "[已更新]"
            print(f"  {tag} {item.rel_path}")
            for d in item.diffs:
                print(d)
        else:
            print(f"  [无变化] {item.rel_path}")

    print()
    if result.dry_run:
        print(f"共 {result.changed_count} 个文件需要更新（dry-run，未写入）。")
    else:
        print(f"共 {result.changed_count} 个文件已更新完毕。")
        if result.changed_count > 0:
            print("现在可以直接启动循环：")
            print("  .\\scripts\\lab\\run_loop.ps1 -Mode factor -LoopN 10")
    print()


# ── 自动探测最新交易日 ──────────────────────────────────────────────────────
def auto_detect_data_end() -> date:
    """从 Qlib 本地数据目录探测最新有数据的交易日。"""
    try:
        import qlib
        from qlib.data import D

        uri = os.environ.get("QLIB_PROVIDER_URI", DEFAULT_PROVIDER_URI)
        qlib.init(provider_uri=uri, region="cn")
        cal = D.calendar(freq="day")
        if len(cal) == 0:
            raise RuntimeError("日历为空")
        last = cal[-1]
        if hasattr(last, "date"):
            return last.date()
        return date.fromisoformat(str(last)[:10])
    except Exception as e:
        print(f"[auto] 探测失败：{e}")
        print("[auto] 请改用 --data-end YYYY-MM-DD 手动指定。")
        sys.exit(1)


# ── 主入口 ──────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="更新所有配置文件的时间窗到最新数据日期。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--data-end", metavar="YYYY-MM-DD",
        help="新数据截止日期（如 2026-04-23）",
    )
    grp.add_argument(
        "--auto", action="store_true",
        help="从 Qlib 数据目录自动探测最新交易日",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只打印将要修改的内容，不写文件",
    )
    args = parser.parse_args()

    if args.auto:
        data_end = auto_detect_data_end()
        print(f"[auto] 探测到最新交易日：{fmt(data_end)}")
    else:
        try:
            data_end = date.fromisoformat(args.data_end)
        except ValueError:
            print(f"[错误] 日期格式不正确：{args.data_end}，请用 YYYY-MM-DD。")
            sys.exit(1)

    print_update_result(apply_data_window(data_end, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
