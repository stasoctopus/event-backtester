"""Walk-forward window generation, splitting, and the optimize/evaluate/stitch loop."""

from __future__ import annotations

import numpy as np
import pytest

from eventbt import (
    Backtester,
    BarSeries,
    EngineConfig,
    SMACrossover,
    TickSeries,
    WFWindow,
    generate_windows,
    total_return,
    train_test_split,
    walk_forward,
)
from eventbt.engine import _group_ticks
from eventbt.walkforward import _param_combos

CFG = EngineConfig(initial_balance=10_000, risk_pct=0.01, point_value=1.0)
GRID = {"fast": [5, 10], "slow": [20, 30]}


def _factory(fast: int, slow: int) -> SMACrossover:
    return SMACrossover(fast=fast, slow=slow, stop_distance=1.0, take_distance=2.0)


def _objective(result: object) -> float:
    return total_return(result.equity_curve)  # type: ignore[attr-defined]


# --- generate_windows ------------------------------------------------------------


def test_window_count_and_bounds() -> None:
    windows = generate_windows(n=100, train_size=50, test_size=10, step=10)
    assert len(windows) == 5
    assert windows[0] == WFWindow(0, 50, 50, 60)
    assert windows[-1] == WFWindow(40, 90, 90, 100)


def test_windows_are_contiguous_when_step_equals_test() -> None:
    windows = generate_windows(n=100, train_size=50, test_size=10, step=10)
    for prev, nxt in zip(windows, windows[1:], strict=False):
        assert nxt.test_start == prev.test_end
        assert nxt.train_start == prev.train_start + 10


def test_overlapping_windows_when_step_smaller_than_test() -> None:
    windows = generate_windows(n=100, train_size=50, test_size=10, step=5)
    assert len(windows) == 9  # starts 0,5,...,40


def test_insufficient_data_yields_no_windows() -> None:
    assert generate_windows(n=40, train_size=50, test_size=10) == []


def test_invalid_sizes_raise() -> None:
    with pytest.raises(ValueError):
        generate_windows(n=100, train_size=0, test_size=10)
    with pytest.raises(ValueError):
        generate_windows(n=100, train_size=50, test_size=10, step=0)


def test_train_test_split() -> None:
    train, test = train_test_split(100, 0.7)
    assert train == slice(0, 70)
    assert test == slice(70, 100)


def test_train_test_split_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        train_test_split(100, 1.5)


# --- full walk_forward -----------------------------------------------------------


def test_walk_forward_selects_best_and_stitches(synth: tuple[BarSeries, TickSeries]) -> None:
    bars, ticks = synth
    wf = walk_forward(
        bars,
        ticks,
        _factory,
        GRID,
        _objective,
        CFG,
        train_size=80,
        test_size=40,
        step=40,
    )

    n = len(wf.windows)
    assert n > 0
    assert len(wf.oos_results) == n
    assert len(wf.best_params) == n
    assert len(wf.is_scores) == n

    valid_combos = _param_combos(GRID)
    for chosen in wf.best_params:
        assert chosen in valid_combos

    # Independently recompute the in-sample argmax for each window and compare. This
    # replicates the engine's tie-break (first combo in product order wins on ties).
    starts, ends = _group_ticks(bars.time, ticks.time)
    backtester = Backtester(CFG)
    for window, chosen in zip(wf.windows, wf.best_params, strict=True):
        is_bars = bars[window.train_start : window.train_end]
        is_ticks = ticks[int(starts[window.train_start]) : int(ends[window.train_end - 1])]
        best_combo = valid_combos[0]
        best_score = float("-inf")
        for combo in valid_combos:
            score = _objective(backtester.run(_factory(**combo), is_bars, is_ticks))
            if score > best_score:
                best_score = score
                best_combo = combo
        assert chosen == best_combo

    # Stitched OOS curve: length equals the sum of segment lengths and the value is
    # continuous across segment boundaries (each segment is rebased to the running end).
    seg_lengths = [len(r.equity_curve) for r in wf.oos_results]
    assert len(wf.stitched_equity) == sum(seg_lengths)
    boundaries = np.cumsum([n for n in seg_lengths if n > 0])[:-1]
    for b in boundaries:
        assert float(wf.stitched_equity.iloc[b]) == pytest.approx(
            float(wf.stitched_equity.iloc[b - 1])
        )
