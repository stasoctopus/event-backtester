"""Shared test helpers: hand-built series, a naive bar-only exit, and stub strategies."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from eventbt import Bar, BarSeries, Direction, Signal, Strategy, TickSeries


def make_series(
    ohlc: Sequence[tuple[float, float, float, float]],
    tick_lists: Sequence[Sequence[float]],
    start: str = "2020-01-01",
    freq_min: int = 1,
) -> tuple[BarSeries, TickSeries]:
    """Build aligned bar and tick series from explicit OHLC and per-bar tick prices.

    Bar ``i`` is placed at ``start + i*freq_min`` minutes; its ticks are spread evenly
    inside that minute, with the first tick exactly on the bar's left edge so the
    engine's grouping assigns it to that bar.
    """
    base = pd.Timestamp(start)
    bar_delta = pd.Timedelta(minutes=freq_min)

    bars: list[Bar] = []
    tick_times: list[np.datetime64] = []
    tick_prices: list[float] = []
    for i, (o, h, low, c) in enumerate(ohlc):
        t_i = base + i * bar_delta
        bars.append(Bar(time=t_i, open=o, high=h, low=low, close=c, volume=0.0))
        prices = list(tick_lists[i])
        k = len(prices)
        if k == 0:
            continue
        tdelta = bar_delta / k
        for j, price in enumerate(prices):
            tick_times.append(np.datetime64(t_i + j * tdelta))
            tick_prices.append(float(price))

    bar_series = BarSeries.from_bars(bars)
    tick_series = TickSeries(
        np.array(tick_times, dtype="datetime64[ns]"),
        np.array(tick_prices, dtype=float),
    )
    return bar_series, tick_series


def naive_bar_exit_pnl(
    ohlc: tuple[float, float, float, float],
    direction: Direction,
    entry_price: float,
    stop_distance: float,
    take_distance: float,
    lots: int,
    point_value: float,
) -> float:
    """A bar-only exit rule that, when both levels sit inside the bar, optimistically
    assumes the take-profit was hit first -- the bias this engine avoids.
    """
    _o, high, low, _c = ohlc
    if direction == Direction.LONG:
        stop = entry_price - stop_distance
        take = entry_price + take_distance
        hit_stop = low <= stop
        hit_take = high >= take
        sign = 1
    else:
        stop = entry_price + stop_distance
        take = entry_price - take_distance
        hit_stop = high >= stop
        hit_take = low <= take
        sign = -1

    if hit_take:  # optimistic assumption
        return sign * (take - entry_price) * lots * point_value
    if hit_stop:
        return sign * (stop - entry_price) * lots * point_value
    return 0.0


class OneShotStrategy(Strategy):
    """Emits a single, fixed :class:`Signal` once, at a chosen bar index."""

    def __init__(self, signal: Signal, at_bar: int = 0) -> None:
        self._signal = signal
        self._at = at_bar
        self._fired = False

    def on_bar(self, bars: BarSeries) -> Signal | None:
        idx = len(bars) - 1
        if not self._fired and idx == self._at:
            self._fired = True
            return self._signal
        return None


class FlatStrategy(Strategy):
    """Never trades -- useful for accounting-only assertions."""

    def on_bar(self, bars: BarSeries) -> Signal | None:
        return None
