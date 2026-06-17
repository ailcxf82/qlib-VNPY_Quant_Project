"""Resolve qlib benchmark instrument codes for local provider layouts."""
from __future__ import annotations

from typing import Sequence

# Wind/qlib 文档常用 SH000905；本仓库 D:/qlib_data 存为 000905.SZ（见 instruments/all.txt）
BENCHMARK_ALIASES: dict[str, tuple[str, ...]] = {
    "SH000905": ("000905.SZ", "000905.sz", "SH000905", "sh000905"),
    "SH000300": ("000300.SH", "000300.sh", "SH000300", "sh000300"),
}


def benchmark_candidates(code: str) -> tuple[str, ...]:
    """Return instrument ids to try, in order (deduplicated)."""
    key = str(code).strip().upper()
    extras = BENCHMARK_ALIASES.get(key, ())
    out: list[str] = [str(code).strip()]
    for alt in extras:
        if alt not in out:
            out.append(alt)
    return tuple(out)


def resolve_benchmark_in_provider(
    feature_fetcher,
    codes: Sequence[str],
    fields: Sequence[str],
    start_time: str,
    end_time: str,
    freq: str = "day",
):
    """Try each candidate code; return (resolved_code, dataframe) or raise last error."""
    last_exc: Exception | None = None
    for code in codes:
        try:
            df = feature_fetcher(
                [code],
                list(fields),
                start_time=start_time,
                end_time=end_time,
                freq=freq,
            )
            if df is not None and not df.empty:
                return code, df
        except Exception as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise ValueError(f"No benchmark rows for {list(codes)!r} in [{start_time}, {end_time}]")
