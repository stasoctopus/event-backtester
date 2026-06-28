"""Synthetic data generator: reproducibility and bar/tick consistency."""

from __future__ import annotations

import numpy as np

from eventbt import gbm_data

TICKS_PER_BAR = 30
N_BARS = 50


def _gen(seed: int) -> tuple:
    return gbm_data(n_bars=N_BARS, ticks_per_bar=TICKS_PER_BAR, sigma=0.4, seed=seed)


def test_reproducible_with_same_seed() -> None:
    b1, t1 = _gen(123)
    b2, t2 = _gen(123)
    assert np.array_equal(t1.price, t2.price)
    assert np.array_equal(b1.close, b2.close)
    assert np.array_equal(t1.time, t2.time)


def test_different_seed_differs() -> None:
    _, t1 = _gen(1)
    _, t2 = _gen(2)
    assert not np.array_equal(t1.price, t2.price)


def test_bar_tick_consistency() -> None:
    bars, ticks = _gen(7)
    grid = ticks.price.reshape(N_BARS, TICKS_PER_BAR)
    assert np.allclose(bars.open, grid[:, 0])
    assert np.allclose(bars.close, grid[:, -1])
    assert np.allclose(bars.high, grid.max(axis=1))
    assert np.allclose(bars.low, grid.min(axis=1))


def test_ohlc_invariants() -> None:
    bars, _ = _gen(7)
    assert np.all(bars.high >= bars.low)
    assert np.all(bars.high >= bars.open)
    assert np.all(bars.high >= bars.close)
    assert np.all(bars.low <= bars.open)
    assert np.all(bars.low <= bars.close)


def test_bar_count_matches_ticks() -> None:
    bars, ticks = _gen(7)
    assert len(bars) == N_BARS
    assert len(ticks) == N_BARS * TICKS_PER_BAR


def test_prices_strictly_positive() -> None:
    _, ticks = _gen(7)
    assert np.all(ticks.price > 0)


def test_time_is_monotonic() -> None:
    bars, ticks = _gen(7)
    assert np.all(np.diff(ticks.time.astype("int64")) > 0)
    assert np.all(np.diff(bars.time.astype("int64")) > 0)
