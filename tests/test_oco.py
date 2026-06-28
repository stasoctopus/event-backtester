"""OCO bracket semantics: one exit per position, no double counting, forced close."""

from __future__ import annotations

from _helpers import OneShotStrategy, make_series
from eventbt import Direction, EngineConfig, ExitReason, Signal, run_backtest

NO_COST = EngineConfig(initial_balance=10_000, risk_pct=0.01, point_value=1.0)
LONG = Signal(Direction.LONG, stop_distance=1.0, take_distance=2.0)


def test_only_one_exit_when_both_levels_in_range() -> None:
    # The stop is reached first; the later take-profit tick must NOT open a 2nd trade.
    bars, ticks = make_series(
        ohlc=[(100, 100, 100, 100), (100.0, 102.0, 99.0, 101.0)],
        tick_lists=[[100.0], [100.0, 99.5, 99.0, 100.5, 102.0]],
    )
    result = run_backtest(OneShotStrategy(LONG, at_bar=0), bars, ticks, NO_COST)

    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == ExitReason.STOP


def test_no_reentry_after_oneshot_signal() -> None:
    # The OneShot strategy fires once; after the bracket closes mid-bar, the engine
    # asks for a new signal while flat but gets None -> no further trades.
    bars, ticks = make_series(
        ohlc=[
            (100, 100, 100, 100),
            (100.0, 102.0, 99.0, 101.0),
            (101, 103, 100, 102),
        ],
        tick_lists=[[100.0], [100.0, 99.5, 99.0], [101.0, 102.0, 103.0]],
    )
    result = run_backtest(OneShotStrategy(LONG, at_bar=0), bars, ticks, NO_COST)
    assert len(result.trades) == 1


def test_open_position_is_force_closed_at_end() -> None:
    # Bracket far away -> never hit -> liquidated at the final bar's close.
    bars, ticks = make_series(
        ohlc=[(100, 100, 100, 100), (100, 101, 100, 101)],
        tick_lists=[[100.0], [100.0, 100.5, 101.0]],
    )
    sig = Signal(Direction.LONG, stop_distance=100.0, take_distance=100.0)
    result = run_backtest(OneShotStrategy(sig, at_bar=0), bars, ticks, NO_COST)

    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == ExitReason.END
    # final equity point equals the realized balance after liquidation
    assert float(result.equity_curve.iloc[-1]) == result.final_balance
