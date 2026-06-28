"""Target-weight / multi-leg portfolio backtesting (opt-in, generic).

This is the third execution paradigm in :mod:`eventbt`, alongside the core
signal -> tick engine (:func:`eventbt.run_backtest`) and the limit-TTL futures
engine (:func:`eventbt.run_futures_backtest`).

Where ``run_backtest`` and ``run_futures_backtest`` model **discrete trades**
(an entry, an exit, a realised PnL), this module models a **continuously held
target position** that is rebalanced every bar.  At each bar the strategy holds
a signed weight; the per-bar return is

    r_t = pos_{t-1} * (C_t / C_{t-1} - 1)        # carried part, close-to-close
        + (pos_t - pos_{t-1}) * (C_t / O_t - 1)  # freshly added part, open-to-close
        - |pos_t - pos_{t-1}| * cost             # turnover cost
        + funding_t                              # optional funding/borrow effect

This matches the standard "vectorised weight x return" accounting used by
trend / carry / vol-targeting books.  Signals are still decided on coarse bars
(e.g. 1h) and executed on fine bars (e.g. 1m): the coarse position is shifted by
one coarse bar (no look-ahead) and forward-filled onto the fine grid, exactly
mirroring an event-driven "decide at close t, act from t+1" rule.

The module is deliberately bot-agnostic: it knows nothing about funding-rate
file formats or strategy gating.  The caller injects those via a ``funding_fn``
callback and by supplying pre-built leverage / funding-daily series.

Public surface
--------------
``PortfolioConfig``        cost / annualisation parameters
``PositionResult``         output of :func:`run_position_backtest`
``run_position_backtest``  one directional leg, fine-bar accounting
``to_daily``               compound a fine net-return series to daily
``carry_leg_returns``      a funding-harvest (carry) leg, daily
``combine_daily``          weight-combine several daily legs
``portfolio_metrics``      headline metrics for a daily return series
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "PortfolioConfig",
    "PositionResult",
    "run_position_backtest",
    "to_daily",
    "carry_leg_returns",
    "combine_daily",
    "portfolio_metrics",
]


@dataclass
class PortfolioConfig:
    """Cost and annualisation parameters for portfolio-style backtests.

    Costs are expressed exactly as the classic vectorised model does:
    a per-side fee plus slippage, both in basis points, charged on the traded
    notional ``|Δpos|``.  ``carry_cost_day`` is the daily hedge-maintenance drag
    of a delta-neutral carry leg (in return units, e.g. ``2e-5`` ~ 0.7%/yr).
    """

    fee_bps_side: float = 2.0
    slip_bps: float = 1.0
    carry_cost_day: float = 2e-5
    months_per_year: int = 12          # Sharpe annualisation: monthly -> *sqrt(12)
    days_per_year: float = 365.25      # CAGR year length
    capital_mode: str = "reinvest"     # "reinvest" (compounding) is the only mode used


@dataclass
class PositionResult:
    """Result of a single directional leg run on fine bars."""

    net_ret: pd.Series                 # per-fine-bar net return
    equity: pd.Series                  # cumprod(1 + net_ret)
    dd: pd.Series                      # drawdown fraction
    daily_returns: pd.Series           # net_ret compounded to calendar days
    metrics: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------
# signal -> fine-bar mapping (decide at close t, act from t + tf)
# ----------------------------------------------------------------------------
def map_signal_to_fine(position: pd.Series, fine_index: pd.DatetimeIndex,
                       signal_tf: pd.Timedelta | None = None) -> pd.Series:
    """Map a coarse-bar signed position onto a fine-bar index, lag-safe.

    The decision taken at the *close* of coarse bar ``t`` only becomes effective
    on the next bar, so the position timestamp is shifted forward by one coarse
    timeframe before being forward-filled onto the fine grid.  ``signal_tf``
    defaults to the spacing of the first two coarse bars (``index[1] - index[0]``).
    """
    if signal_tf is None:
        signal_tf = position.index[1] - position.index[0]
    decision_time = position.index + signal_tf
    return (pd.Series(position.values, index=decision_time)
            .reindex(fine_index, method="ffill")
            .fillna(0.0))


# ----------------------------------------------------------------------------
# directional leg
# ----------------------------------------------------------------------------
def run_position_backtest(fine_bars: pd.DataFrame, position: pd.Series, *,
                          signal_tf: pd.Timedelta | None = None,
                          funding_fn=None,
                          config: PortfolioConfig | None = None,
                          n_trades: int = 0) -> PositionResult:
    """Backtest one continuously-held signed position on fine bars.

    Parameters
    ----------
    fine_bars
        Fine-grained OHLC bars (needs ``open`` and ``close`` columns); its index
        is the execution clock (e.g. 1-minute).
    position
        Signed target weight on the coarse signal grid (e.g. 1h), as decided at
        each coarse bar's close.  It is shifted +1 coarse bar and forward-filled.
    signal_tf
        Coarse timeframe for the lag; inferred from ``position`` if omitted.
    funding_fn
        Optional ``(fine_index, pos_fine) -> Series`` returning the per-bar
        funding PnL effect.  ``pos_fine`` is the mapped fine-grid position, so a
        caller can implement perp funding (``-pos*rate``), a "longs-on-spot"
        hybrid (funding only on the short side: ``pos.clip(upper=0)``), etc.
        ``None`` means no funding.
    config
        Cost / annualisation parameters (defaults to :class:`PortfolioConfig`).
    n_trades
        Round-trip trade count (sign changes) for reporting; PnL-neutral.
    """
    cfg = config or PortfolioConfig()
    pos_fine = map_signal_to_fine(position, fine_bars.index, signal_tf)

    open_, close = fine_bars["open"], fine_bars["close"]
    pos_prev = pos_fine.shift(1).fillna(0.0)
    dpos = pos_fine - pos_prev
    ret_cc = close.pct_change().fillna(0.0)
    ret_oc = (close / open_ - 1.0).fillna(0.0)
    strat_ret = pos_prev * ret_cc + dpos * ret_oc

    cost = dpos.abs() * (cfg.fee_bps_side + cfg.slip_bps) / 1e4
    if funding_fn is not None:
        fund = funding_fn(fine_bars.index, pos_fine)
    else:
        fund = pd.Series(0.0, index=fine_bars.index)
    net_ret = strat_ret - cost + fund

    equity = (1.0 + net_ret).cumprod()
    dd = (equity - equity.cummax()) / equity.cummax()
    daily = to_daily(net_ret)
    metrics = _series_metrics(net_ret, equity, dd, n_trades, cfg)
    return PositionResult(net_ret=net_ret, equity=equity, dd=dd,
                          daily_returns=daily, metrics=metrics)


# ----------------------------------------------------------------------------
# carry / funding-harvest leg
# ----------------------------------------------------------------------------
def carry_leg_returns(funding_daily: pd.Series, leverage: pd.Series,
                      cost_day: float = 2e-5) -> pd.Series:
    """Daily returns of a (levered) delta-neutral funding-carry leg.

    ``funding_daily`` is the per-day funding sum a 1x delta-neutral book earns;
    ``leverage`` is the (possibly time-varying / gated) carry leverage on the
    same index.  Returns ``leverage * (funding_daily - cost_day)``.
    """
    lev = leverage.reindex(funding_daily.index) if not leverage.index.equals(
        funding_daily.index) else leverage
    return (lev * (funding_daily - cost_day)).rename("carry")


# ----------------------------------------------------------------------------
# daily aggregation / combination
# ----------------------------------------------------------------------------
def to_daily(net_ret: pd.Series) -> pd.Series:
    """Compound a fine per-bar net-return series into calendar-day returns."""
    return ((1.0 + net_ret).groupby(net_ret.index.normalize()).prod() - 1.0)


def combine_daily(legs: list[pd.Series], weights: list[float],
                  index: pd.DatetimeIndex | None = None) -> pd.Series:
    """Weight-combine several daily return legs onto a common index.

    Each leg is reindexed to ``index`` (default: the first leg's index) and
    missing days are treated as flat (0.0), then summed with its weight.
    """
    if not legs:
        raise ValueError("combine_daily needs at least one leg")
    if len(legs) != len(weights):
        raise ValueError("legs and weights must be the same length")
    idx = legs[0].index if index is None else index
    out = pd.Series(0.0, index=idx)
    for leg, w in zip(legs, weights):
        out = out + w * leg.reindex(idx).fillna(0.0)
    return out.rename("combo")


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def portfolio_metrics(daily_returns: pd.Series, n_trades: int = 0, *,
                      config: PortfolioConfig | None = None) -> dict:
    """Headline metrics for a daily return series (CAGR, MaxDD, Sharpe, ...).

    Sharpe is computed on calendar-month returns annualised by
    ``sqrt(months_per_year)`` with ``ddof=1``; CAGR uses an actual-days year
    length; MaxDD is the worst close-to-peak on the daily equity curve.  The
    rounding matches the canonical research reporting so dicts compare equal.
    """
    cfg = config or PortfolioConfig()
    daily = daily_returns.dropna()
    eq = (1.0 + daily).cumprod()
    mdd = float((eq / eq.cummax() - 1.0).min() * 100)
    monthly = (1.0 + daily).groupby(daily.index.to_period("M")).prod() - 1.0
    sd = monthly.std(ddof=1)
    sharpe = float(monthly.mean() / sd * np.sqrt(cfg.months_per_year)) if sd > 0 else 0.0
    yrs = (eq.index[-1] - eq.index[0]).days / cfg.days_per_year
    cagr = float(eq.iloc[-1] ** (1.0 / yrs) - 1.0) * 100 if yrs > 0 else 0.0
    yearly = ((1.0 + daily).groupby(daily.index.year).prod() - 1.0) * 100
    out = {
        "CAGR_%": round(cagr, 1),
        "tr/yr": round(n_trades / yrs, 0) if yrs > 0 else 0.0,
        "max_dd_%": round(mdd, 1),
        "sharpe_m": round(sharpe, 2),
        "calmar": round(cagr / abs(mdd), 2) if mdd else np.inf,
        "pos_months": f"{int((monthly > 0).sum())}/{len(monthly)}",
    }
    out.update({f"y{y}_%": round(float(v), 1) for y, v in yearly.items()})
    return out


def _series_metrics(net_ret: pd.Series, equity: pd.Series, dd: pd.Series,
                    n_trades: int, cfg: PortfolioConfig) -> dict:
    """Metrics computed on the *native* (fine) return series of a single leg.

    Mirrors the per-leg reporting block of the research engine: equity-based
    return / CAGR / MaxDD, month-grouped Sharpe, positive-month count, per-year.
    """
    max_dd = float(dd.min() * 100)
    monthly = (1.0 + net_ret).groupby(net_ret.index.to_period("M")).prod() - 1.0
    sd = monthly.std(ddof=1)
    sharpe = float(monthly.mean() / sd * np.sqrt(cfg.months_per_year)) if sd > 0 else 0.0
    yrs = (equity.index[-1] - equity.index[0]).days / cfg.days_per_year
    cagr = float(equity.iloc[-1] ** (1.0 / yrs) - 1.0) if yrs > 0 else 0.0
    calmar = cagr * 100 / abs(max_dd) if max_dd != 0 else np.inf
    yearly = ((1.0 + net_ret).groupby(net_ret.index.year).prod() - 1.0) * 100
    return {
        "return_%": round(float(equity.iloc[-1] - 1) * 100, 1),
        "CAGR_%": round(cagr * 100, 1),
        "tr/yr": round(n_trades / yrs, 0) if yrs > 0 else 0.0,
        "max_dd_%": round(max_dd, 1),
        "sharpe_m": round(sharpe, 2),
        "calmar": round(float(calmar), 2),
        "pos_months": f"{int((monthly > 0).sum())}/{len(monthly)}",
        **{f"y{y}_%": round(float(v), 1) for y, v in yearly.items()},
    }
