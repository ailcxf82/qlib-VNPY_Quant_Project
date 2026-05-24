"""PortAna 回测截止日：避免 TradeCalendarManager 访问 calendar[end+1] 越界。"""
from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

_BACKTEST_END_RE = re.compile(r"(^        end_time:\s*)(\d{4}-\d{2}-\d{2})", re.MULTILINE)
_TEST_SEG_END_RE = re.compile(r"test:\s*\[[^\]]+,\s*(\d{4}-\d{2}-\d{2})\]")
_HANDLER_END_RE = re.compile(r"^    end_time:\s*(\d{4}-\d{2}-\d{2})", re.MULTILINE)


def patch_workspace_portana_dates(workspace: Path, *, provider_uri: str | None = None) -> bool:
    """将 workspace 内 conf 的 PortAna backtest end_time 校正为 test_end 的前一交易日。"""
    conf = Path(workspace) / "conf_combined_factors.yaml"
    if not conf.is_file():
        return False

    text = conf.read_text(encoding="utf-8")
    m_be = _BACKTEST_END_RE.search(text)
    if not m_be:
        return False

    data_end = _infer_data_end(text)
    if data_end is None:
        return False

    root = Path(__file__).resolve().parents[2]
    import sys

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts.update_data_window import resolve_trading_endpoints

    test_start = date(data_end.year, 1, 1)
    _test_end, backtest_end = resolve_trading_endpoints(
        test_start, data_end, provider_uri=provider_uri
    )
    safe = backtest_end.isoformat()
    current = m_be.group(2)
    if current == safe:
        return False

    new_text = _BACKTEST_END_RE.sub(rf"\g<1>{safe}", text, count=1)
    conf.write_text(new_text, encoding="utf-8")
    logger.info(
        "PortAna backtest end_time %s -> %s in %s", current, safe, conf
    )
    return True


def _infer_data_end(text: str) -> date | None:
    m = _TEST_SEG_END_RE.search(text)
    if m:
        return date.fromisoformat(m.group(1))
    m = _HANDLER_END_RE.search(text)
    if m:
        return date.fromisoformat(m.group(1))
    return None
