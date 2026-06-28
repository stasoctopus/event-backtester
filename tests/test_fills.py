"""Fill model and the headline tick-accuracy property.

The core claim of the engine is that intrabar exits are resolved by the *actual tick
order*, not by an optimistic bar-only guess. These tests pin that behaviour down.
"""

from __future__ import annotations

import pytest

from _helpers import OneShotStrategy, make_series, naive_bar_exit_pnl
from eventbt import Direction, EngineConfig, ExitReason, Signal, run_backtest, size_position

# A long signal emitted on bar 0; the fill happens on the first tick of bar 1.
LONG = Signal(Direction.LONG, stop_distance=1.0, take_distance=2.0)
SHORT = Signal(Direction.SHORT, stop_distance=1.0, take_distance=2.0)
NO_COST = EngineConfig(initial_balance=10_000, risk_pct=0.01, point_value=1.0)


def test_entry_fills_on_first_tick_of_next_bar() -> None:
    # Far bracket so nothing triggers; position is force-closed at the end.
    bars, ticks = make_series(
        ohlc=[(100, 100, 100, 100), (100, 101, 100, 101)],
        tick_lists=[[100.0], [100.0, 100.5, 101.0]],
    )
    sig = Signal(Direction.LONG, stop_distance=100.0, take_distance=100.0)
    result = run_backtest(OneShotStrategy(sig, at_bar=0), bars, ticks, NO_COST)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.entry_price == pytest.approx(100.0)  # first tick of bar 1
    assert trade.entry_time == bars[1].time  # bar 1's left edge == its first tick


def test_tick_accuracy_stop_before_take_long() -> None:
    # Both 99 (stop) and 102 (take) are inside the bar's [low, high]. The ticks reach
    # the stop FIRST, so a tick-accurate engine must record a loss -- even though a
    # naive bar-only backtest would optimistically book the take-profit.
    fill_bar = (100.0, 102.0, 99.0, 101.0)
    bars, ticks = make_series(
        ohlc=[(100, 100, 100, 100), fill_bar],
        tick_lists=[[100.0], [100.0, 99.5, 99.0, 100.5, 102.0]],
    )
    result = run_backtest(OneShotStrategy(LONG, at_bar=0), bars, ticks, NO_COST)

    assert len(result.trades) == 1
    trade = result.trades[0]
    lots = size_position(10_000, 0.01, LONG.stop_distance, 1.0)

    assert trade.exit_reason == ExitReason.STOP
    assert trade.pnl < 0
    assert trade.pnl == pytest.approx(-1.0 * lots)  # stop distance 1.0, point_value 1.0

    naive = naive_bar_exit_pnl(fill_bar, Direction.LONG, 100.0, 1.0, 2.0, lots, 1.0)
    assert naive == pytest.approx(2.0 * lots)  # bar-only rule books the +2 take
    assert trade.pnl < naive  # the engine refuses to overstate the result


def test_tick_accuracy_take_before_stop_long() -> None:
    # Same levels, reversed tick order: the take is reached first -> a win.
    bars, ticks = make_series(
        ohlc=[(100, 100, 100, 100), (100.0, 102.0, 99.0, 101.0)],
        tick_lists=[[100.0], [100.0, 100.5, 102.0, 99.5, 99.0]],
    )
    result = run_backtest(OneShotStrategy(LONG, at_bar=0), bars, ticks, NO_COST)

    trade = result.trades[0]
    lots = size_position(10_000, 0.01, LONG.stop_distance, 1.0)
    assert trade.exit_reason == ExitReason.TAKE
    assert trade.pnl == pytest.approx(2.0 * lots)


def test_tick_accuracy_short_symmetry() -> None:
    # Short entry at 100: stop at 101, take at 98. Ticks reach the stop first -> loss.
    bars, ticks = make_series(
        ohlc=[(100, 100, 100, 100), (100.0, 101.0, 98.0, 99.0)],
        tick_lists=[[100.0], [100.0, 100.5, 101.0, 99.0, 98.0]],
    )
    result = run_backtest(OneShotStrategy(SHORT, at_bar=0), bars, ticks, NO_COST)

    trade = result.trades[0]
    lots = size_position(10_000, 0.01, SHORT.stop_distance, 1.0)
    assert trade.exit_reason == ExitReason.STOP
    assert trade.pnl == pytest.approx(-1.0 * lots)


def test_spread_reduces_pnl_by_exactly_spread_times_size() -> None:
    # Far bracket, force-close at the last bar's close (1.0 above the entry), so the
    # only difference between runs is the spread cost on entry+exit.
    bars, ticks = make_series(
        ohlc=[(100, 100, 100, 100), (100, 101, 100, 101)],
        tick_lists=[[100.0], [100.0, 100.5, 101.0]],
    )
    sig = Signal(Direction.LONG, stop_distance=100.0, take_distance=100.0)
    lots = size_position(10_000, 0.01, sig.stop_distance, 1.0)

    base = run_backtest(OneShotStrategy(sig, at_bar=0), bars, ticks, NO_COST)
    spread_cfg = EngineConfig(initial_balance=10_000, risk_pct=0.01, point_value=1.0, spread=0.4)
    with_spread = run_backtest(OneShotStrategy(sig, at_bar=0), bars, ticks, spread_cfg)

    expected_cost = 0.4 * lots * 1.0  # full spread paid across both fills
    assert with_spread.trades[0].pnl == pytest.approx(base.trades[0].pnl - expected_cost)


def test_commission_reduces_pnl_by_round_turn() -> None:
    bars, ticks = make_series(
        ohlc=[(100, 100, 100, 100), (100, 101, 100, 101)],
        tick_lists=[[100.0], [100.0, 100.5, 101.0]],
    )
    sig = Signal(Direction.LONG, stop_distance=100.0, take_distance=100.0)
    lots = size_position(10_000, 0.01, sig.stop_distance, 1.0)

    base = run_backtest(OneShotStrategy(sig, at_bar=0), bars, ticks, NO_COST)
    comm_cfg = EngineConfig(initial_balance=10_000, risk_pct=0.01, point_value=1.0, commission=2.0)
    with_comm = run_backtest(OneShotStrategy(sig, at_bar=0), bars, ticks, comm_cfg)

    expected_cost = 2.0 * 2.0 * lots  # commission per side, round-turn
    assert with_comm.trades[0].pnl == pytest.approx(base.trades[0].pnl - expected_cost)


def test_balance_delta_equals_trade_pnl() -> None:
    # The realized balance change must equal the reported trade PnL exactly, even
    # with both spread and commission applied.
    bars, ticks = make_series(
        ohlc=[(100, 100, 100, 100), (100, 101, 100, 101)],
        tick_lists=[[100.0], [100.0, 100.5, 101.0]],
    )
    sig = Signal(Direction.LONG, stop_distance=100.0, take_distance=100.0)
    cfg = EngineConfig(
        initial_balance=10_000, risk_pct=0.01, point_value=1.0, spread=0.4, commission=1.5
    )
    result = run_backtest(OneShotStrategy(sig, at_bar=0), bars, ticks, cfg)

    delta = result.final_balance - cfg.initial_balance
    assert delta == pytest.approx(result.trades[0].pnl)
