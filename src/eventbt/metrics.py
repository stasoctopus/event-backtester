"""Performance metrics computed from an equity curve and a trade log.

Every function is pure and deterministic. The annualization factor
(``periods_per_year``) is always an explicit argument -- never inferred from
wall-clock timestamps -- so results are reproducible regardless of the data's
sampling frequency. A zero risk-free rate is assumed throughout.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import pandas as pd

from .engine import BacktestResult, Trade

__all__ = [
    "total_return",
    "win_rate",
    "profit_factor",
    "max_drawdown",
    "sharpe",
    "calmar",
    "positive_months_pct",
    "summary",
    "summary_table",
]


def _period_returns(series: pd.Series) -> pd.Series:
    """Simple period-over-period returns, computed without ``pct_change`` to avoid
    its ``fill_method`` deprecation across pandas versions."""
    s = series.astype(float)
    return (s / s.shift(1) - 1.0).dropna()


def total_return(equity: pd.Series) -> float:
    """Total return over the curve: ``equity[-1] / equity[0] - 1``."""
    if len(equity) < 1 or equity.iloc[0] == 0:
        return float("nan")
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def win_rate(trades: Sequence[Trade]) -> float:
    """Fraction of trades with positive PnL. ``nan`` if there are no trades."""
    if not trades:
        return float("nan")
    wins = sum(1 for t in trades if t.pnl > 0)
    return wins / len(trades)


def profit_factor(trades: Sequence[Trade]) -> float:
    """Gross profit divided by gross loss.

    Returns ``inf`` when there are winners but no losers, and ``nan`` when there are
    no trades at all.
    """
    if not trades:
        return float("nan")
    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = -sum(t.pnl for t in trades if t.pnl < 0)
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else float("nan")
    return gross_profit / gross_loss


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a non-positive fraction (e.g. -0.20)."""
    if len(equity) < 1:
        return float("nan")
    eq = equity.astype(float)
    running_max = eq.cummax()
    drawdown = eq / running_max - 1.0
    return float(drawdown.min())


def sharpe(equity: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio of the per-period equity returns (risk-free = 0).

    ``sharpe = mean(r) / std(r, ddof=1) * sqrt(periods_per_year)``. Returns ``nan``
    when fewer than two returns exist or the return series has zero variance.
    """
    if len(equity) < 2:
        return float("nan")
    returns = _period_returns(equity)
    if len(returns) < 2:
        return float("nan")
    std = returns.std(ddof=1)
    if std == 0 or math.isnan(std):
        return float("nan")
    return float(returns.mean() / std * math.sqrt(periods_per_year))


def calmar(equity: pd.Series, periods_per_year: int = 252) -> float:
    """Calmar ratio: annualized CAGR divided by the absolute max drawdown.

    Returns ``nan`` when the curve is degenerate or has no drawdown.
    """
    if len(equity) < 2:
        return float("nan")
    returns = _period_returns(equity)
    n = len(returns)
    start, end = float(equity.iloc[0]), float(equity.iloc[-1])
    if n < 1 or start <= 0 or end <= 0:
        return float("nan")
    years = n / periods_per_year
    if years <= 0:
        return float("nan")
    cagr = (end / start) ** (1.0 / years) - 1.0
    mdd = max_drawdown(equity)
    if mdd == 0 or math.isnan(mdd):
        return float("nan")
    return cagr / abs(mdd)


def positive_months_pct(equity: pd.Series) -> float:
    """Percentage of calendar months with a positive return.

    The curve is resampled to month-end (``"ME"``) and consecutive month-end values
    are differenced. The first month-end serves as the baseline and is not itself
    scored (the common convention, as in e.g. quantstats), so an N-month curve yields
    N-1 monthly returns. Requires a ``DatetimeIndex``; returns ``nan`` if fewer than
    two months exist.
    """
    if len(equity) < 2:
        return float("nan")
    if not isinstance(equity.index, pd.DatetimeIndex):
        raise TypeError("positive_months_pct requires a DatetimeIndex")
    monthly = equity.astype(float).resample("ME").last()
    monthly_returns = _period_returns(monthly)
    if len(monthly_returns) == 0:
        return float("nan")
    return float((monthly_returns > 0).mean() * 100.0)


def summary(result: BacktestResult, periods_per_year: int = 252) -> dict[str, float]:
    """Compute all metrics for a :class:`~eventbt.engine.BacktestResult`."""
    equity = result.equity_curve
    trades = result.trades
    return {
        "total_return_pct": total_return(equity) * 100.0,
        "win_rate_pct": win_rate(trades) * 100.0,
        "profit_factor": profit_factor(trades),
        "max_drawdown_pct": max_drawdown(equity) * 100.0,
        "sharpe": sharpe(equity, periods_per_year),
        "calmar": calmar(equity, periods_per_year),
        "positive_months_pct": positive_months_pct(equity),
        "num_trades": float(len(trades)),
        "final_balance": float(result.final_balance),
    }


def summary_table(result: BacktestResult, periods_per_year: int = 252) -> str:
    """Render :func:`summary` as an aligned, human-readable table."""
    labels = {
        "total_return_pct": "Total Return %",
        "win_rate_pct": "Win Rate %",
        "profit_factor": "Profit Factor",
        "max_drawdown_pct": "Max Drawdown %",
        "sharpe": "Sharpe",
        "calmar": "Calmar",
        "positive_months_pct": "Positive Months %",
        "num_trades": "Trades",
        "final_balance": "Final Balance",
    }
    data = summary(result, periods_per_year)
    width = max(len(v) for v in labels.values())
    lines = ["Metric".ljust(width) + "  Value", "-" * (width + 14)]
    for key, label in labels.items():
        value = data[key]
        if key == "num_trades":
            formatted = f"{int(value)}"
        elif math.isinf(value):
            formatted = "inf"
        elif math.isnan(value):
            formatted = "n/a"
        else:
            formatted = f"{value:,.2f}"
        lines.append(label.ljust(width) + "  " + formatted)
    return "\n".join(lines)
