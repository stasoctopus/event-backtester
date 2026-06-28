"""Metric correctness on hand-computed vectors."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from eventbt import (
    Direction,
    ExitReason,
    Trade,
    calmar,
    max_drawdown,
    positive_months_pct,
    profit_factor,
    sharpe,
    total_return,
    win_rate,
)


def mk_trade(pnl: float) -> Trade:
    ts = pd.Timestamp("2020-01-01")
    return Trade(
        entry_time=ts,
        entry_price=100.0,
        exit_time=ts,
        exit_price=100.0,
        direction=Direction.LONG,
        lots=1,
        pnl=pnl,
        exit_reason=ExitReason.TAKE,
    )


# Vector A: a simple 3-point equity curve.
EQ_A = pd.Series([100.0, 110.0, 99.0])
# Vector B: a mixed trade ledger.
TRADES_B = [mk_trade(100.0), mk_trade(-50.0), mk_trade(25.0), mk_trade(-25.0)]


def test_total_return() -> None:
    assert total_return(EQ_A) == pytest.approx(-0.01)  # 99/100 - 1


def test_win_rate() -> None:
    assert win_rate(TRADES_B) == pytest.approx(0.5)  # 2 of 4


def test_profit_factor() -> None:
    assert profit_factor(TRADES_B) == pytest.approx(125.0 / 75.0)  # 1.6667


def test_profit_factor_all_winners_is_inf() -> None:
    assert math.isinf(profit_factor([mk_trade(10.0), mk_trade(5.0)]))


def test_max_drawdown() -> None:
    # Peak 110 -> trough 99 = -10%.
    assert max_drawdown(EQ_A) == pytest.approx(-0.10)


def test_sharpe_zero_mean_returns_zero() -> None:
    # Returns [+0.10, -0.10] have ~zero mean -> Sharpe ~0. This path does NOT exercise
    # the annualization factor; see the non-trivial test below.
    assert sharpe(EQ_A) == pytest.approx(0.0, abs=1e-9)


def test_sharpe_nontrivial_checks_annualization_and_ddof() -> None:
    # Returns [0.01, 0.02, 0.01, 0.02]: mean 0.015, std(ddof=1) ~0.0057735.
    # Sharpe = 0.015 / 0.0057735 * sqrt(252) ~= 41.246. A bug in sqrt(ppy) or ddof
    # would be caught here (the zero-mean vector cannot catch it).
    returns = [0.01, 0.02, 0.01, 0.02]
    equity = [100.0]
    for r in returns:
        equity.append(equity[-1] * (1.0 + r))
    eq = pd.Series(equity)
    assert sharpe(eq, periods_per_year=252) == pytest.approx(41.2459, rel=1e-3)


def test_calmar() -> None:
    # cagr = (99/100)**(252/2) - 1 ~= -0.7181; max_dd ~= -0.10 -> calmar ~= -7.181.
    assert calmar(EQ_A) == pytest.approx(-7.181, abs=0.02)


def test_positive_months_pct() -> None:
    idx = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30"])
    eq = pd.Series([100.0, 105.0, 102.0, 108.0], index=idx)
    # Monthly returns [+5%, -2.86%, +5.88%] -> 2 of 3 positive -> 66.67%.
    assert positive_months_pct(eq) == pytest.approx(200.0 / 3.0)


def test_empty_inputs_do_not_crash() -> None:
    assert math.isnan(win_rate([]))
    assert math.isnan(profit_factor([]))
    assert math.isnan(sharpe(pd.Series([100.0])))
    assert math.isnan(total_return(pd.Series(dtype=float)))
