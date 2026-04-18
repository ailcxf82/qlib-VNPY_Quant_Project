"""Quant-level hypothesis generation: bias toward simple, implementable factors (higher pass rate)."""

from __future__ import annotations

from typing import Tuple

from rdagent.core.proposal import Trace
from rdagent.scenarios.qlib.proposal.quant_proposal import QlibQuantHypothesisGen


class ProjectQlibQuantHypothesisGen(QlibQuantHypothesisGen):
    """Extend default QlibQuantHypothesisGen RAG so fin_quant proposes factors that match our code template."""

    _PROJECT_FACTOR_RAG = """
------Project factor hypothesis constraints (mandatory)------
1) Propose at most 2 new factors per hypothesis when trace length < 8; at most 3 afterward.
2) Each factor MUST use a single integer window W chosen from {5, 10, 20, 30, 60} (state W explicitly).
3) Allowed primitives: pct_change, shift, rolling(W).mean/std/sum/min/max/corr, rank, clip.
4) FORBIDDEN: nested rolling correlations across many series, loops, "10 pairs" patterns.
5) Available columns in daily_pv.h5 (use EXACTLY these names, no others):
   Price/Volume: $close, $open, $high, $low, $volume, $amount
   Liquidity:    $turnover_rate, $turnover_rate_f, $volume_ratio
   Valuation:    $pe_ttm, $pb, $ps_ttm, $total_mv, $dv_ratio
   Quality:      $roe, $roa, $q_profit_yoy, $q_eps
   Technical:    $rsi12, $macd, $macd_dif, $kdj_k, $kdj_d, $atr
   Margin:       $rzye, $rqye
   NOTE: $pe_ttm/$pb/$roe/$q_profit_yoy may have NaN for some instruments (quarterly data).
         Always use .fillna(method='ffill') or rolling mean as fallback for fundamental columns.
6) High-value factor ideas to explore (go beyond simple momentum):
   (a) Value momentum: $roe change over 4 quarters (use shift) vs $pb
   (b) Earnings surprise: $q_profit_yoy relative to its rolling mean
   (c) Margin pressure reversal: $rqye / $total_mv spike then reversal
   (d) Volume-price divergence: rolling corr($close, $volume, W) sign change
   (e) Valuation mean-reversion: ($pe_ttm - rolling_mean($pe_ttm, W)) / rolling_std
   (f) ATR-normalized momentum: pct_change($close, W) / rolling_mean($atr, W)
7) Factor names must encode type and window, e.g. MomRet_10D, ValueMom_20D, EarnSurp_5D.
"""

    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]:
        ctx, ok = super().prepare_context(trace)
        if ok and isinstance(ctx, dict):
            base = str(ctx.get("RAG") or "")
            ctx["RAG"] = base.rstrip() + "\n\n" + self._PROJECT_FACTOR_RAG
        return ctx, ok
