"""In-sample / out-of-sample splitting and rolling walk-forward validation.

Walk-forward analysis is the antidote to overfitting: parameters are chosen on an
*in-sample* (IS) window and then judged only on the immediately following, never-seen
*out-of-sample* (OOS) window. Rolling the window forward produces a continuous OOS
equity path -- the closest a backtest gets to honest, unbiased performance.

The module is strategy-agnostic: the caller supplies a parameter grid, a factory that
builds a strategy from a parameter combination, and an objective to maximize.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any

import pandas as pd

from .data import Bar, BarSeries, Tick, TickSeries
from .engine import Backtester, BacktestResult, EngineConfig, Trade, _group_ticks
from .strategy import Strategy

__all__ = [
    "WFWindow",
    "WalkForwardResult",
    "generate_windows",
    "train_test_split",
    "walk_forward",
]

ParamGrid = Mapping[str, Sequence[Any]]
Objective = Callable[[BacktestResult], float]
StrategyFactory = Callable[..., Strategy]


@dataclass(frozen=True, slots=True)
class WFWindow:
    """A single walk-forward window. All bounds are ``[start, end)`` (end-exclusive).

    ``train`` and ``test`` are contiguous: ``train_end == test_start``.
    """

    train_start: int
    train_end: int
    test_start: int
    test_end: int


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    """Aggregated output of :func:`walk_forward`."""

    windows: list[WFWindow]
    best_params: list[dict[str, Any]]
    is_scores: list[float]
    oos_results: list[BacktestResult]
    stitched_equity: pd.Series
    stitched_trades: list[Trade]


def generate_windows(
    n: int, train_size: int, test_size: int, step: int | None = None
) -> list[WFWindow]:
    """Generate rolling walk-forward windows over ``n`` bars.

    ``step`` defaults to ``test_size`` (non-overlapping, contiguous OOS segments).
    A window is emitted only if both its train and test halves fit fully within ``n``.
    """
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive")
    step = test_size if step is None else step
    if step <= 0:
        raise ValueError("step must be positive")

    windows: list[WFWindow] = []
    start = 0
    while start + train_size + test_size <= n:
        windows.append(
            WFWindow(
                train_start=start,
                train_end=start + train_size,
                test_start=start + train_size,
                test_end=start + train_size + test_size,
            )
        )
        start += step
    return windows


def train_test_split(n: int, train_frac: float) -> tuple[slice, slice]:
    """Split ``n`` items into a single in-sample / out-of-sample pair of slices.

    Useful for a blind hold-out: train on the first ``train_frac`` of the data, keep
    the rest untouched for a final OOS evaluation.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in the open interval (0, 1)")
    cut = int(n * train_frac)
    return slice(0, cut), slice(cut, n)


def _param_combos(param_grid: ParamGrid) -> list[dict[str, Any]]:
    keys = list(param_grid.keys())
    if not keys:
        return [{}]
    return [
        dict(zip(keys, values, strict=True)) for values in product(*(param_grid[k] for k in keys))
    ]


def _stitch(
    results: Sequence[BacktestResult], initial_balance: float
) -> tuple[pd.Series, list[Trade]]:
    """Chain OOS segments into one compounded equity curve.

    Each segment restarts at ``initial_balance``; we rebase it so that it begins where
    the previous stitched segment ended, preserving continuity by compounded return.
    This assumes the test windows do not overlap (the default ``step == test_size``);
    overlapping windows would produce a non-monotonic, duplicated index.
    """
    pieces: list[pd.Series] = []
    trades: list[Trade] = []
    running_end = float(initial_balance)
    for res in results:
        equity = res.equity_curve
        trades.extend(res.trades)
        if len(equity) == 0:
            continue
        base = float(equity.iloc[0])
        scaled = (equity / base * running_end) if base != 0 else (equity * 0.0 + running_end)
        pieces.append(scaled)
        running_end = float(scaled.iloc[-1])
    if pieces:
        stitched = pd.concat(pieces)
        stitched.name = "equity"
    else:
        stitched = pd.Series(dtype=float, name="equity")
    return stitched, trades


def walk_forward(
    bars: BarSeries | Sequence[Bar],
    ticks: TickSeries | Sequence[Tick],
    strategy_factory: StrategyFactory,
    param_grid: ParamGrid,
    objective: Objective,
    config: EngineConfig | None = None,
    *,
    train_size: int,
    test_size: int,
    step: int | None = None,
) -> WalkForwardResult:
    """Run rolling walk-forward optimization and OOS evaluation.

    For each window: grid-search ``param_grid`` on the in-sample bars, pick the
    combination that maximizes ``objective``, then evaluate that single combination
    on the out-of-sample bars. OOS segments are stitched into one continuous curve.

    Parameters
    ----------
    bars, ticks:
        The full coarse bar series and fine tick series.
    strategy_factory:
        ``(**params) -> Strategy`` builder.
    param_grid:
        Mapping of parameter name to the list of values to search.
    objective:
        ``(BacktestResult) -> float``; higher is better.
    config:
        Engine configuration (defaults to :class:`EngineConfig`).
    train_size, test_size, step:
        Window geometry (see :func:`generate_windows`).
    """
    cfg = config if config is not None else EngineConfig()
    bar_series = BarSeries.coerce(bars)
    tick_series = TickSeries.coerce(ticks)
    starts, ends = _group_ticks(bar_series.time, tick_series.time)

    windows = generate_windows(len(bar_series), train_size, test_size, step)
    combos = _param_combos(param_grid)
    backtester = Backtester(cfg)

    best_params: list[dict[str, Any]] = []
    is_scores: list[float] = []
    oos_results: list[BacktestResult] = []

    for window in windows:
        # --- In-sample: search the grid, keep the best objective ---
        is_bars = bar_series[window.train_start : window.train_end]
        is_ticks = tick_series[int(starts[window.train_start]) : int(ends[window.train_end - 1])]

        best_combo = combos[0]
        best_score = float("-inf")
        for combo in combos:
            result = backtester.run(strategy_factory(**combo), is_bars, is_ticks)
            score = objective(result)
            if score > best_score:
                best_score = score
                best_combo = combo

        # --- Out-of-sample: evaluate the winner once ---
        oos_bars = bar_series[window.test_start : window.test_end]
        oos_ticks = tick_series[int(starts[window.test_start]) : int(ends[window.test_end - 1])]
        oos_result = backtester.run(strategy_factory(**best_combo), oos_bars, oos_ticks)

        best_params.append(dict(best_combo))
        is_scores.append(best_score)
        oos_results.append(oos_result)

    stitched_equity, stitched_trades = _stitch(oos_results, cfg.initial_balance)
    return WalkForwardResult(
        windows=windows,
        best_params=best_params,
        is_scores=is_scores,
        oos_results=oos_results,
        stitched_equity=stitched_equity,
        stitched_trades=stitched_trades,
    )
