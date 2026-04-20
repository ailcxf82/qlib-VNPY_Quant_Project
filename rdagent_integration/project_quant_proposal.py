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

------Reward / objective (P3a, mandatory reading)------
You are scored by `1day.composite_score = 1.0*IR + 2.0*IC_IR - 0.5*log(1+annualized_turnover)`.
This means three things you MUST optimise simultaneously, NOT just IC:
  · IR  (information_ratio of excess_return_with_cost) — real backtest signal-to-noise
  · IC_IR (Rank IC mean / IC std) — signal stability
  · annualized_turnover — penalised; high-frequency switching hurts the score.

Empirical lesson from previous loops (csi300_RD_v2, 2025-11~2026-03):
  Adding 5 short-cycle volume-price-reversal factors (RangeRatio_10D, VolRatio_20D,
  VolumePriceTrend_10D, VolumeTrend_10D, VolRet_5D) raised LGB valid-IC from 0.148
  to 0.226 BUT the realised Sharpe DROPPED from 3.59 to 1.69 because annualized
  turnover almost doubled (4.7 -> 8.75) and the strategy beta collapsed from 1.50
  to 0.90 (signals fought each other instead of stacking). DO NOT propose more
  short-cycle volume-price-reversal factors; they are the failure family.

------Existing feature universe (`lgb_short_cycle`, 32 columns)------
The project already exposes these columns to LGB; new factors must add INCREMENTAL
information, i.e. low correlation with these:
  Returns/Momentum: RET1, MOM5, MOM10, MOM20, MOM1_5
  Volatility:       VOL5, VOL10, VOL20
  Volume ratios:    VRATIO5, VRATIO10, VRATIO20, VOLVOL
  Range:            HLRANGE, HLRANGE5, HLRANGE10
  Liquidity:        TURN, TURN_F, TURN_REL
  Valuation:        PE_TTM, PB, PS_TTM, LOG_MV
  Quality:          ROE, ROA, PROFIT_YOY
  Technical:        RSI12, MACD, KDJ_DIFF, ATR
  Margin:           MARGIN_L, MARGIN_S

Hard requirement (mandatory):
  6.a) New factor MUST satisfy |Spearman(new_factor, X)| <= 0.50 for every X above
       (when X is a strict superset, prove orthogonality through transformation).
  6.b) New factor's day-over-day cross-sectional rank auto-correlation MUST be
       >= 0.60 (slow-moving) — this directly bounds turnover.
  6.c) Avoid signals that primarily fire on the SAME day as a price jump; prefer
       lagged / smoothed transformations.

------Encouraged factor families (HIGH composite-score expectation)------
   (a) Quality persistence: rolling_mean($roe, 4 quarters) ranked vs sector
   (b) Valuation mean-reversion (slow): ($pe_ttm - rolling_median($pe_ttm, 60)) / rolling_std($pe_ttm, 60)
   (c) Margin financing trend: rolling_mean($rzye / $total_mv, 20) - rolling_mean($rzye / $total_mv, 60)
   (d) Earnings revision strength: $q_profit_yoy minus its 4-quarter rolling median (use shift)
   (e) Low-volatility quality: rank($roe) / (rank(VOL20) + 1)
   (f) Long-horizon residual momentum: pct_change($close, 60) - beta * pct_change(benchmark, 60)
   (g) Liquidity stability: 1 / rolling_std($turnover_rate / Mean($turnover_rate, 60), 20)
------Discouraged factor families (LOW composite-score, DO NOT propose)------
   (x) Short-cycle volume-price reversals on W in {5, 10}
   (y) Same-day volume spike + price reversal patterns
   (z) Anything that ranks the universe with >50% weekly turnover

7) Factor names must encode type and window, e.g. QualPersist_60D, ValueMR_60D,
   MarginTrend_20D, EarnRev_4Q, LowVolQual_20D, ResidMom_60D, LiqStab_20D.
"""

    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]:
        ctx, ok = super().prepare_context(trace)
        if ok and isinstance(ctx, dict):
            base = str(ctx.get("RAG") or "")
            ctx["RAG"] = base.rstrip() + "\n\n" + self._PROJECT_FACTOR_RAG
        return ctx, ok
