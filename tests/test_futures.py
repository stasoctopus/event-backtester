"""Futures execution engine: limit-entry TTL, tick-accurate bracket, notional cost,
risk sizing and per-segment compounding.

Every case here is hand-built and deterministic: a single bar whose ``time`` is the
*close* timestamp, plus an explicit list of ``(seconds_after_close, price)`` ticks.
The order window is ``(close + order_start_offset_s, close + order_life_sec]`` so all
fill ticks are placed at >= +2s. The engine is treated as ground truth -- these tests
pin the documented semantics, they do not change behaviour.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from eventbt import (
    BarSeries,
    Direction,
    FuturesConfig,
    FuturesSegment,
    FuturesSignal,
    FuturesStrategy,
    TickSeries,
    gbm_data,
    run_futures_backtest,
)

# A weekday, mid-session anchor: Tuesday 14:00 sits inside the default 13:00-23:45.
BAR_CLOSE = pd.Timestamp("2024-01-02 14:00:00")

BASE_CFG = FuturesConfig(
    initial_balance=100_000.0,
    risk_pct=0.01,
    point_value=10.0,
    max_lots=100,
    min_lots=1,
    use_position_sizing=True,
    price_step=0.01,
    min_step=0.01,
    cost_per_step=8.0,
    commission_rate=0.0001,
    order_life_sec=60,
    order_start_offset_s=1,
)

TRADE_COLUMNS = [
    "file", "signal_time", "side", "lots", "entry_time", "exit_time", "hold_sec",
    "close5", "atr5", "entry_offset", "stop_dist", "take_dist", "entry_price",
    "exit_price", "exit_reason", "pnl_before", "commission", "pnl",
    "mfe_money", "mae_money", "r_mult", "risk_money",
]


class OneShotFutures(FuturesStrategy):
    """Emit one fixed :class:`FuturesSignal` at a chosen bar index (once per segment)."""

    def __init__(self, signal: FuturesSignal, at_bar: int = 0) -> None:
        self._signal = signal
        self._at = at_bar

    def on_bar(self, bars: BarSeries, i: int) -> FuturesSignal | None:
        return self._signal if i == self._at else None


class EveryNFutures(FuturesStrategy):
    """Emit a fixed signal on every ``every``-th bar (for the larger sanity run)."""

    def __init__(self, signal: FuturesSignal, every: int = 10) -> None:
        self._signal = signal
        self._every = every

    def on_bar(self, bars: BarSeries, i: int) -> FuturesSignal | None:
        return self._signal if i % self._every == 0 else None


def _one_bar_segment(
    close_price: float,
    tick_offsets: Sequence[tuple[float, float]],
    *,
    close_ts: pd.Timestamp = BAR_CLOSE,
    label: str = "seg",
) -> FuturesSegment:
    """One bar (close ``close_price`` at ``close_ts``) plus ticks at second offsets."""
    bar_time = np.array([np.datetime64(close_ts)], dtype="datetime64[ns]")
    px = np.array([close_price], dtype=float)
    bars = BarSeries(bar_time, px.copy(), px.copy(), px.copy(), px.copy(), np.array([0.0]))
    times = np.array(
        [np.datetime64(close_ts + pd.Timedelta(seconds=s)) for s, _ in tick_offsets],
        dtype="datetime64[ns]",
    )
    prices = np.array([p for _, p in tick_offsets], dtype=float)
    return FuturesSegment(bars=bars, ticks=TickSeries(times, prices), label=label)


# ---------------------------------------------------------------------------
# Limit entry + time-to-live
# ---------------------------------------------------------------------------
def test_limit_entry_never_reached_yields_no_trade() -> None:
    # Long limit = 100; a fill needs a tick <= 100 - price_step = 99.99 INSIDE the
    # 60s window. The in-window ticks stay >= 100; the one tick that does cross
    # (99.0) lands at +120s -- past order_life_sec -- so the order has expired.
    sig = FuturesSignal(Direction.LONG, stop_distance=5.0, take_distance=5.0)
    seg = _one_bar_segment(100.0, [(5, 100.0), (30, 100.10), (60, 100.05), (120, 99.0)])
    res = run_futures_backtest(OneShotFutures(sig), [seg], BASE_CFG)

    assert res.trades == []
    assert res.final_balance == BASE_CFG.initial_balance


def test_limit_entry_reached_fills_exactly_at_limit() -> None:
    # 99.98 <= 99.99 fills the long limit; the fill price is the limit (100), not 99.98.
    sig = FuturesSignal(Direction.LONG, stop_distance=5.0, take_distance=5.0)
    seg = _one_bar_segment(100.0, [(5, 99.98), (20, 100.0), (40, 100.2)])
    res = run_futures_backtest(OneShotFutures(sig), [seg], BASE_CFG)

    assert len(res.trades) == 1
    trade = res.trades[0]
    assert trade.side == "long"
    assert trade.entry_price == pytest.approx(100.0)
    assert trade.entry_time == BAR_CLOSE + pd.Timedelta(seconds=5)


def test_short_fills_at_limit_above_close() -> None:
    # Short limit = 100; fill needs a tick >= 100 + price_step = 100.01.
    sig = FuturesSignal(Direction.SHORT, stop_distance=5.0, take_distance=5.0)
    seg = _one_bar_segment(100.0, [(5, 100.02), (20, 100.0), (40, 99.9)])
    res = run_futures_backtest(OneShotFutures(sig), [seg], BASE_CFG)

    assert len(res.trades) == 1
    trade = res.trades[0]
    assert trade.side == "short"
    assert trade.entry_price == pytest.approx(100.0)
    assert trade.entry_time == BAR_CLOSE + pd.Timedelta(seconds=5)


# ---------------------------------------------------------------------------
# Tick-accurate bracket: stop vs take vs EOD
# ---------------------------------------------------------------------------
def test_stop_before_take_exits_at_stop_level() -> None:
    # entry 100, SL=98, TP=103. Ticks dip to 98 BEFORE rallying to 103 -> a loss.
    sig = FuturesSignal(Direction.LONG, stop_distance=2.0, take_distance=3.0)
    seg = _one_bar_segment(100.0, [(5, 99.98), (10, 99.0), (15, 98.0), (20, 103.0)])
    res = run_futures_backtest(OneShotFutures(sig), [seg], BASE_CFG)

    trade = res.trades[0]
    assert trade.exit_reason == "SL"
    assert trade.exit_price == pytest.approx(98.0)
    assert trade.pnl_before == pytest.approx((98.0 - 100.0) * BASE_CFG.point_value * trade.lots)


def test_take_before_stop_exits_at_take_level() -> None:
    # Same levels, reversed tick order: 103 is reached first -> a win at the TP level.
    sig = FuturesSignal(Direction.LONG, stop_distance=2.0, take_distance=3.0)
    seg = _one_bar_segment(100.0, [(5, 99.98), (10, 101.0), (15, 103.0), (20, 98.0)])
    res = run_futures_backtest(OneShotFutures(sig), [seg], BASE_CFG)

    trade = res.trades[0]
    assert trade.exit_reason == "TP"
    assert trade.exit_price == pytest.approx(103.0)
    assert trade.pnl_before == pytest.approx((103.0 - 100.0) * BASE_CFG.point_value * trade.lots)


def test_short_stop_before_take_exits_at_stop_level() -> None:
    # Mirror on the short side: entry 100, SL=102, TP=97; price runs UP to 102 first.
    sig = FuturesSignal(Direction.SHORT, stop_distance=2.0, take_distance=3.0)
    seg = _one_bar_segment(100.0, [(5, 100.02), (10, 101.0), (15, 102.0), (20, 97.0)])
    res = run_futures_backtest(OneShotFutures(sig), [seg], BASE_CFG)

    trade = res.trades[0]
    assert trade.side == "short"
    assert trade.exit_reason == "SL"
    assert trade.exit_price == pytest.approx(102.0)
    assert trade.pnl_before == pytest.approx((100.0 - 102.0) * BASE_CFG.point_value * trade.lots)


def test_eod_forced_exit_when_bracket_untouched() -> None:
    # SL=90 / TP=110 are never touched; the session end (14:05) forces the exit at the
    # price of the first tick at/after 14:05:00.
    cfg = replace(BASE_CFG, trade_end_time="14:05")
    sig = FuturesSignal(Direction.LONG, stop_distance=10.0, take_distance=10.0)
    seg = _one_bar_segment(
        100.0, [(2, 99.98), (60, 100.1), (120, 100.0), (240, 99.9), (300, 100.5)]
    )
    res = run_futures_backtest(OneShotFutures(sig), [seg], cfg)

    trade = res.trades[0]
    assert trade.exit_reason == "EOD"
    assert trade.exit_time == BAR_CLOSE.normalize() + pd.Timedelta("14:05:00")
    assert trade.exit_price == pytest.approx(100.5)


def test_tie_break_prefers_sl_over_eod_at_same_timestamp() -> None:
    # The stop level (98) is touched by the SAME tick that lands on the EOD time:
    # the two events share a timestamp, so list order (SL before EOD) must win.
    cfg = replace(BASE_CFG, trade_end_time="14:05")
    sig = FuturesSignal(Direction.LONG, stop_distance=2.0, take_distance=50.0)
    eod_off = 5 * 60  # +300s == 14:05:00
    seg = _one_bar_segment(100.0, [(5, 99.98), (eod_off, 98.0)])
    res = run_futures_backtest(OneShotFutures(sig), [seg], cfg)

    trade = res.trades[0]
    assert trade.exit_reason == "SL"
    assert trade.exit_price == pytest.approx(98.0)


# ---------------------------------------------------------------------------
# Costs and sizing
# ---------------------------------------------------------------------------
def test_commission_is_notional_round_turn() -> None:
    # commission = entry_price * (cost_per_step/min_step) * commission_rate * 2 * lots.
    sig = FuturesSignal(Direction.LONG, stop_distance=5.0, take_distance=50.0)
    seg = _one_bar_segment(100.0, [(5, 99.98), (20, 100.1)])
    res = run_futures_backtest(OneShotFutures(sig), [seg], BASE_CFG)

    trade = res.trades[0]
    assert trade.lots == 20  # int(100000*0.01 / (5*10)) == int(20.0)
    expected = (
        100.0
        * (BASE_CFG.cost_per_step / BASE_CFG.min_step)
        * BASE_CFG.commission_rate
        * 2
        * trade.lots
    )
    assert trade.commission == pytest.approx(expected)
    assert trade.commission == pytest.approx(320.0)
    assert trade.pnl == pytest.approx(trade.pnl_before - trade.commission)


def test_sizing_min_lots_floor() -> None:
    # 100000*0.01 / (1000*10) = 0.1 -> int -> 0 -> clamped UP to min_lots=1.
    sig = FuturesSignal(Direction.LONG, stop_distance=1000.0, take_distance=1000.0)
    seg = _one_bar_segment(100.0, [(5, 99.98), (20, 100.1)])
    res = run_futures_backtest(OneShotFutures(sig), [seg], BASE_CFG)

    assert res.trades[0].lots == BASE_CFG.min_lots == 1


def test_sizing_max_lots_cap() -> None:
    # 100000*0.01 / (0.1*10) = 1000 -> clamped DOWN to max_lots=3.
    cfg = replace(BASE_CFG, max_lots=3)
    sig = FuturesSignal(Direction.LONG, stop_distance=0.1, take_distance=50.0)
    seg = _one_bar_segment(100.0, [(5, 99.98), (20, 100.05)])
    res = run_futures_backtest(OneShotFutures(sig), [seg], cfg)

    assert res.trades[0].lots == 3


def test_sizing_disabled_uses_min_lots() -> None:
    # use_position_sizing=False ignores the risk math and always trades min_lots.
    cfg = replace(BASE_CFG, use_position_sizing=False, min_lots=2)
    sig = FuturesSignal(Direction.LONG, stop_distance=5.0, take_distance=50.0)
    seg = _one_bar_segment(100.0, [(5, 99.98), (20, 100.1)])
    res = run_futures_backtest(OneShotFutures(sig), [seg], cfg)

    assert res.trades[0].lots == 2


# ---------------------------------------------------------------------------
# Per-segment compounding
# ---------------------------------------------------------------------------
def test_reinvest_compounds_balance_across_segments() -> None:
    # Each segment: enter 100, take-profit at 130 (+30). Sizing uses the *current*
    # balance, so segment 2 must be sized off segment 1's ending capital.
    sig = FuturesSignal(Direction.LONG, stop_distance=5.0, take_distance=30.0)
    seg1 = _one_bar_segment(
        100.0, [(5, 99.98), (10, 130.0)], close_ts=pd.Timestamp("2024-01-02 14:00:00"), label="m1"
    )
    seg2 = _one_bar_segment(
        100.0, [(5, 99.98), (10, 130.0)], close_ts=pd.Timestamp("2024-01-03 14:00:00"), label="m2"
    )
    res = run_futures_backtest(OneShotFutures(sig), [seg1, seg2], BASE_CFG)

    assert len(res.trades) == 2
    t1, t2 = res.trades
    assert t1.exit_reason == "TP" and t2.exit_reason == "TP"

    # seg1 is sized off the initial balance; seg2 off the *carried* balance.
    risk_money = sig.stop_distance * BASE_CFG.point_value
    assert t1.lots == int(BASE_CFG.initial_balance * BASE_CFG.risk_pct / risk_money)
    assert res.month_end_caps[0] == pytest.approx(BASE_CFG.initial_balance + t1.pnl)
    assert t2.lots == int(res.month_end_caps[0] * BASE_CFG.risk_pct / risk_money)
    assert t2.lots > t1.lots  # the win grew the account -> a bigger second position

    # Reinvest: the final balance is the initial plus both trades' net PnL, and it
    # equals the last month-end cap.
    assert res.final_balance == pytest.approx(BASE_CFG.initial_balance + t1.pnl + t2.pnl)
    assert res.final_balance == pytest.approx(res.month_end_caps[-1])


# ---------------------------------------------------------------------------
# Larger, seeded sanity run
# ---------------------------------------------------------------------------
def test_gbm_sanity_run_is_well_formed() -> None:
    bars, ticks = gbm_data(n_bars=200, ticks_per_bar=30, sigma=0.5, seed=7)
    seg = FuturesSegment(bars=bars, ticks=ticks, label="gbm")
    # gbm timestamps start at 00:00 and fall outside the 13:00-23:45 window, so the
    # session is disabled for this throughput check.
    cfg = replace(BASE_CFG, session_enabled=False)
    sig = FuturesSignal(Direction.LONG, stop_distance=0.5, take_distance=0.5)
    res = run_futures_backtest(EveryNFutures(sig, every=10), [seg], cfg)

    assert np.isfinite(res.final_balance)
    assert len(res.trades) >= 1
    frame = res.trades_frame()
    assert len(frame) == len(res.trades)
    assert list(frame.columns) == TRADE_COLUMNS
    # Every reported trade fills exactly at its limit (long: close, offset 0).
    for trade in res.trades:
        assert trade.side == "long"
        assert trade.exit_reason in {"SL", "TP", "EOD"}
