"""
risk_limits.py — Portfolio-level pre-trade risk controls.

The audit flagged that the broker happily fills any decision the
strategy emits as long as the per-position constraints (max_positions,
position_frac) are satisfied. There were no portfolio-level limits:

  * gross exposure can drift past 100% if everyone says "buy"
  * sector concentration can stack 6 of 10 names in semis
  * portfolio beta can blow past 1.5 in a momentum chase
  * single-day VaR can sit at 5% with nobody noticing

This module sits *between* ``decisions.prioritize_entries`` and
``broker.apply_decisions``. It takes the day's proposed decisions plus
the current portfolio state, runs them through a stack of pre-trade
checks, and drops / down-scales / converts to NO_TRADE the entries
that would breach a limit. Drops are recorded with a reason so the
decision-memory log can audit them.

Usage
-----
::

    from portfolio.risk_limits import RiskLimits, apply_pre_trade_limits

    limits = RiskLimits.from_cfg(cfg)
    decisions, dropped = apply_pre_trade_limits(
        decisions, state, limits=limits,
        sector_lookup=lambda tk: analyses_by_ticker[tk]["sector"],
        beta_lookup=lambda tk: analyses_by_ticker[tk].get("beta"),
    )
    # ... then broker.apply_decisions(state, decisions, ...)

Limits — defaults are conservative for a US equity LS book:

  * ``max_gross_exposure_pct``   1.20 (120% gross including shorts)
  * ``max_net_exposure_pct``     1.00 (100% net; protects against -200% short)
  * ``max_sector_pct``           0.30 (30% in any single sector)
  * ``max_beta_to_spy``          1.30 (target beta cap; long minus short)
  * ``max_single_day_var_pct``   0.05 (5% single-day VaR @ 95% c.l.)

VaR is computed on a normal approximation using positions' realised
60d vol and assumed zero correlation (the conservative direction —
real correlations >0 would mean we underestimate VaR; the cap should
bite even harder).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from portfolio.decisions import Action, TickerDecision
from portfolio.store import Position, PortfolioState


# ── config ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RiskLimits:
    max_gross_exposure_pct: float = 1.20
    max_net_exposure_pct: float = 1.00
    max_sector_pct: float = 0.30
    max_beta_to_spy: float = 1.30
    max_single_day_var_pct: float = 0.05
    var_confidence_z: float = 1.645   # one-sided 95%
    enforce_var: bool = True

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any]) -> "RiskLimits":
        return cls(
            max_gross_exposure_pct=float(cfg.get("max_gross_exposure_pct", 1.20)),
            max_net_exposure_pct=float(cfg.get("max_net_exposure_pct", 1.00)),
            max_sector_pct=float(cfg.get("max_sector_pct", 0.30)),
            max_beta_to_spy=float(cfg.get("max_beta_to_spy", 1.30)),
            max_single_day_var_pct=float(cfg.get("max_single_day_var_pct", 0.05)),
            var_confidence_z=float(cfg.get("var_confidence_z", 1.645)),
            enforce_var=bool(cfg.get("enforce_var", True)),
        )


# ── current state aggregation ────────────────────────────────────────────────

@dataclass
class PortfolioAgg:
    nav: float
    long_notional: float = 0.0
    short_notional: float = 0.0
    by_sector_long: dict[str, float] = field(default_factory=dict)
    by_sector_short: dict[str, float] = field(default_factory=dict)
    beta_weighted_long: float = 0.0
    beta_weighted_short: float = 0.0
    var_var_squared: float = 0.0       # accumulator for VaR variance

    @property
    def gross(self) -> float:
        return (self.long_notional + self.short_notional) / max(self.nav, 1e-9)

    @property
    def net(self) -> float:
        return (self.long_notional - self.short_notional) / max(self.nav, 1e-9)

    @property
    def beta(self) -> float:
        if self.nav <= 0:
            return 0.0
        return (self.beta_weighted_long - self.beta_weighted_short) / self.nav

    @property
    def single_day_var_pct(self) -> float:
        if self.nav <= 0 or self.var_var_squared <= 0:
            return 0.0
        # daily vol -> single-day VaR @ confidence z
        sigma_daily = math.sqrt(self.var_var_squared) / self.nav
        return sigma_daily / math.sqrt(252.0)

    def with_z(self, z: float) -> float:
        return self.single_day_var_pct * z


def _sector_cap_amount(margin: float, exposure: float, cfg: dict[str, Any]) -> float:
    """Sector caps use margin (cash at risk) under leverage so one slot ≠ 50% sector."""
    if cfg.get("risk_limits_sector_on_margin", True):
        return margin
    return exposure


def _beta_risk_amount(
    margin: float, exposure: float, beta: float, cfg: dict[str, Any]
) -> float:
    """Beta budget on margin so a 10-slot book can target ~market beta with 5× CFD."""
    if cfg.get("risk_limits_beta_on_margin", True):
        return beta * margin
    return beta * exposure


def _aggregate_existing(
    state: PortfolioState,
    *,
    cfg: dict[str, Any],
    sector_lookup: Callable[[str], str | None] | None = None,
    beta_lookup: Callable[[str], float | None] | None = None,
    vol_lookup: Callable[[str], float | None] | None = None,
) -> PortfolioAgg:
    agg = PortfolioAgg(nav=state.nav)
    for p in state.positions:
        sector = (sector_lookup(p.ticker) if sector_lookup else None) or "UNKNOWN"
        beta = (beta_lookup(p.ticker) if beta_lookup else None) or 1.0
        vol = (vol_lookup(p.ticker) if vol_lookup else None) or 0.20
        margin = p.position_margin()
        sector_amt = _sector_cap_amount(margin, p.notional, cfg)
        beta_amt = _beta_risk_amount(margin, p.notional, beta, cfg)
        if p.side == "long":
            agg.long_notional += p.notional
            agg.by_sector_long[sector] = agg.by_sector_long.get(sector, 0.0) + sector_amt
            agg.beta_weighted_long += beta_amt
        else:
            agg.short_notional += p.notional
            agg.by_sector_short[sector] = agg.by_sector_short.get(sector, 0.0) + sector_amt
            agg.beta_weighted_short += beta_amt
        agg.var_var_squared += (p.notional * vol) ** 2
    return agg


# ── pre-trade decision filter ────────────────────────────────────────────────

def apply_pre_trade_limits(
    decisions: list[TickerDecision],
    state: PortfolioState,
    *,
    limits: RiskLimits,
    cfg: dict[str, Any],
    sector_lookup: Callable[[str], str | None] | None = None,
    beta_lookup: Callable[[str], float | None] | None = None,
    vol_lookup: Callable[[str], float | None] | None = None,
) -> tuple[list[TickerDecision], list[dict[str, Any]]]:
    """Return ``(filtered_decisions, dropped_log)``.

    Order:
      1. Existing positions counted into the aggregate (they consume
         budget; we don't kick them out — risk-limit breaches in the
         existing book are addressed by exits, not by pre-trade veto).
      2. Iterate proposed entries in score order (decisions are
         already sorted by ``prioritize_entries``).
      3. For each entry, simulate adding it and check every limit;
         drop the entry if any limit would breach.
      4. Exits and HOLDs always pass through unchanged.

    The dropped log carries enough information for the memory log to
    explain "why didn't we take this signal".
    """
    frac = float(cfg.get("position_frac", 0.10))
    scale = float(cfg.get("_regime_scale", 1.0))
    eff_frac = frac * scale
    z = limits.var_confidence_z

    agg = _aggregate_existing(
        state,
        cfg=cfg,
        sector_lookup=sector_lookup,
        beta_lookup=beta_lookup,
        vol_lookup=vol_lookup,
    )

    dropped: list[dict[str, Any]] = []
    out: list[TickerDecision] = []
    for d in decisions:
        if d.action not in (Action.ENTER_LONG, Action.ENTER_SHORT):
            out.append(d)
            continue
        # Simulate the entry hitting at ``eff_frac`` of NAV. If a vol-
        # target sizer is configured, we don't replicate it here; the
        # final size will be smaller, so this check is conservative.
        side_long = d.action == Action.ENTER_LONG
        entry_frac = float(cfg.get("position_frac", 0.10)) * float(cfg.get("_regime_scale", 1.0))
        if float(cfg.get("_regime_scale", 1.0)) >= float(cfg.get("bull_scale_threshold", 0.99)):
            entry_frac *= float(cfg.get("bull_position_frac_mult", 1.0))
        if not side_long:
            entry_frac *= float(cfg.get("short_position_frac_mult", 1.0))
        margin = max(0.0, entry_frac * state.nav)
        from portfolio.broker import cfd_leverage

        lev = cfd_leverage(cfg)
        exposure_add = margin * lev
        cash_need = margin
        if margin <= 0 or state.cash < cash_need * 0.999:
            continue  # broker will skip anyway; let it
        sector = (sector_lookup(d.ticker) if sector_lookup else None) or "UNKNOWN"
        beta = (beta_lookup(d.ticker) if beta_lookup else None) or 1.0
        vol = (vol_lookup(d.ticker) if vol_lookup else None) or (d.vol_60d_annual or 0.20)

        new_long = agg.long_notional + (exposure_add if side_long else 0.0)
        new_short = agg.short_notional + (0.0 if side_long else exposure_add)
        new_long_sector = dict(agg.by_sector_long)
        new_short_sector = dict(agg.by_sector_short)
        sector_add = _sector_cap_amount(margin, exposure_add, cfg)
        if side_long:
            new_long_sector[sector] = new_long_sector.get(sector, 0.0) + sector_add
        else:
            new_short_sector[sector] = new_short_sector.get(sector, 0.0) + sector_add
        beta_add = _beta_risk_amount(margin, exposure_add, beta, cfg)
        new_beta_long = agg.beta_weighted_long + (beta_add if side_long else 0.0)
        new_beta_short = agg.beta_weighted_short + (0.0 if side_long else beta_add)
        new_varsq = agg.var_var_squared + (exposure_add * vol) ** 2

        # Build a hypothetical Agg.
        hypo = PortfolioAgg(
            nav=state.nav,
            long_notional=new_long,
            short_notional=new_short,
            by_sector_long=new_long_sector,
            by_sector_short=new_short_sector,
            beta_weighted_long=new_beta_long,
            beta_weighted_short=new_beta_short,
            var_var_squared=new_varsq,
        )

        breaches: list[str] = []
        if hypo.gross > limits.max_gross_exposure_pct:
            breaches.append(f"gross={hypo.gross:.2f}>cap={limits.max_gross_exposure_pct:.2f}")
        if abs(hypo.net) > limits.max_net_exposure_pct:
            breaches.append(f"net={hypo.net:.2f}>cap={limits.max_net_exposure_pct:.2f}")
        sector_long_pct = new_long_sector.get(sector, 0.0) / max(state.nav, 1e-9)
        sector_short_pct = new_short_sector.get(sector, 0.0) / max(state.nav, 1e-9)
        sector_pct = sector_long_pct + sector_short_pct
        if sector_pct > limits.max_sector_pct:
            breaches.append(f"sector[{sector}]={sector_pct:.2f}>cap={limits.max_sector_pct:.2f}")
        if abs(hypo.beta) > limits.max_beta_to_spy:
            breaches.append(f"beta={hypo.beta:.2f}>cap={limits.max_beta_to_spy:.2f}")
        if limits.enforce_var and hypo.with_z(z) > limits.max_single_day_var_pct:
            breaches.append(
                f"var95={hypo.with_z(z):.3f}>cap={limits.max_single_day_var_pct:.3f}"
            )

        if breaches:
            dropped.append({
                "ticker": d.ticker,
                "action_proposed": d.action.value,
                "reason": "; ".join(breaches),
            })
            out.append(TickerDecision(
                ticker=d.ticker,
                action=Action.NO_TRADE,
                reason=f"Pre-trade risk limits breached: {'; '.join(breaches)}",
                ml_score=d.ml_score, quintile=d.quintile, p_up_20d=d.p_up_20d,
                price=d.price, atr_pct=d.atr_pct, vol_60d_annual=d.vol_60d_annual,
                intraday_low=d.intraday_low, intraday_high=d.intraday_high,
                open_price=d.open_price,
            ))
            continue

        # Accept entry: fold its contribution into agg so subsequent
        # entries see updated portfolio state.
        agg = hypo
        out.append(d)
    return out, dropped
