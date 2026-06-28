"""The event-driven, tick-accurate backtesting engine.

Why "tick-accurate"? Signals are generated on a *coarse* bar series, but order
fills and the protective bracket are resolved on a *finer* tick series. When both
the stop-loss and the take-profit fall inside a single bar's range, a bar-only
backtester cannot know which was touched first and must guess -- typically in the
strategy's favour, which inflates results. This engine replays the actual tick
order, so the first level the price reaches wins (a genuine one-cancels-other).

Execution model (per bar, one position at a time):

1. **Entry** of a pending signal fills at the *first tick of the next bar* (never at
   the signal bar's close -- that would be look-ahead). The lot count is computed by
   risk-based sizing from the stop distance.
2. **Exit (OCO)**: the bar's ticks are scanned in time order; the first of stop /
   take reached closes the position and cancels the other.
3. **Mark-to-market** at the bar close records the equity curve point.
4. A **new signal** is requested only while flat.

Costs: an optional ``spread`` (half charged on each side) and a per-lot, per-side
``commission`` (round-turn = twice). Stop and take orders fill exactly at their level;
gap-through slippage beyond the spread is intentionally not modeled (a standard,
documented simplification).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pandas as pd

from .data import Bar, BarSeries, Tick, TickSeries
from .strategy import Direction, Signal, Strategy

__all__ = [
    "ExitReason",
    "Trade",
    "EngineConfig",
    "BacktestResult",
    "Backtester",
    "run_backtest",
    "size_position",
]


class ExitReason(IntEnum):
    """Why a position was closed."""

    STOP = 1
    TAKE = 2
    END = 3  # forced liquidation at the end of the data


@dataclass(frozen=True, slots=True)
class Trade:
    """A completed round-turn trade. ``pnl`` is net of spread and commission."""

    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    direction: Direction
    lots: int
    pnl: float
    exit_reason: ExitReason


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Execution and risk configuration.

    Attributes
    ----------
    initial_balance:
        Starting account balance.
    risk_pct:
        Fraction of the current balance risked per trade (used for sizing).
    point_value:
        Money gained/lost per 1.0 of price movement, per lot.
    max_lots:
        Optional hard cap on the lot count per trade.
    spread:
        Total spread in price units; half is charged on each fill (entry and exit).
    commission:
        Money per lot, per side. A round-turn therefore costs ``2 * commission``.
    """

    initial_balance: float = 10_000.0
    risk_pct: float = 0.01
    point_value: float = 1.0
    max_lots: int | None = None
    spread: float = 0.0
    commission: float = 0.0


def size_position(
    balance: float,
    risk_pct: float,
    stop_distance: float,
    point_value: float,
    max_lots: int | None = None,
) -> int:
    """Risk-based position sizing.

    ``lots = floor(balance * risk_pct / (stop_distance * point_value))``, clamped to
    ``[0, max_lots]``. Returns ``0`` (i.e. skip the trade) when any input is
    non-positive or the risk budget cannot afford a single lot.
    """
    if balance <= 0 or risk_pct <= 0 or stop_distance <= 0 or point_value <= 0:
        return 0
    raw = (balance * risk_pct) / (stop_distance * point_value)
    lots = max(0, math.floor(raw))
    if max_lots is not None:
        lots = min(lots, int(max_lots))
    return lots


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """The output of a backtest run."""

    equity_curve: pd.Series
    trades: list[Trade]
    initial_balance: float
    final_balance: float
    config: EngineConfig

    def trades_frame(self) -> pd.DataFrame:
        """Return the trade log as a ``DataFrame`` (empty but typed if no trades)."""
        if not self.trades:
            return pd.DataFrame(
                columns=[
                    "entry_time",
                    "entry_price",
                    "exit_time",
                    "exit_price",
                    "direction",
                    "lots",
                    "pnl",
                    "exit_reason",
                ]
            )
        return pd.DataFrame(
            {
                "entry_time": [t.entry_time for t in self.trades],
                "entry_price": [t.entry_price for t in self.trades],
                "exit_time": [t.exit_time for t in self.trades],
                "exit_price": [t.exit_price for t in self.trades],
                "direction": [t.direction.name for t in self.trades],
                "lots": [t.lots for t in self.trades],
                "pnl": [t.pnl for t in self.trades],
                "exit_reason": [t.exit_reason.name for t in self.trades],
            }
        )


@dataclass(slots=True)
class _Position:
    """Engine-private open position state (mutable)."""

    direction: Direction
    entry_time: pd.Timestamp
    entry_price: float  # effective (spread-inclusive) fill
    lots: int
    stop_price: float
    take_price: float


def _group_ticks(bar_times: np.ndarray, tick_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map each bar to the half-open ``[start, end)`` slice of ticks it contains.

    Bar ``i`` owns the ticks with ``bar_times[i] <= t < bar_times[i + 1]`` (the last
    bar runs to the end of the data). Computed in O(n log n) with ``searchsorted``
    instead of an O(bars * ticks) scan.
    """
    n = len(bar_times)
    starts = np.searchsorted(tick_times, bar_times, side="left").astype(np.intp)
    ends = np.empty(n, dtype=np.intp)
    if n > 0:
        ends[:-1] = starts[1:]
        ends[-1] = len(tick_times)
    return starts, ends


class Backtester:
    """Runs a strategy over aligned bar and tick streams."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config if config is not None else EngineConfig()

    def run(
        self,
        strategy: Strategy,
        bars: BarSeries | Sequence[Bar],
        ticks: TickSeries | Sequence[Tick],
    ) -> BacktestResult:
        """Execute ``strategy`` over ``bars`` (signals) and ``ticks`` (fills)."""
        cfg = self.config
        bar_series = BarSeries.coerce(bars)
        tick_series = TickSeries.coerce(ticks)

        n_bars = len(bar_series)
        half_spread = cfg.spread / 2.0
        pv = cfg.point_value

        strategy.on_start(bar_series)

        if n_bars == 0:
            empty_eq = pd.Series(dtype=float)
            return BacktestResult(empty_eq, [], cfg.initial_balance, cfg.initial_balance, cfg)

        starts, ends = _group_ticks(bar_series.time, tick_series.time)
        tick_price = tick_series.price
        tick_time = tick_series.time
        bar_close = bar_series.close

        balance = cfg.initial_balance
        position: _Position | None = None
        pending: Signal | None = None
        trades: list[Trade] = []
        equity_times: list[np.datetime64] = []
        equity_values: list[float] = []

        for i in range(n_bars):
            lo, hi = int(starts[i]), int(ends[i])
            n_in_bar = hi - lo

            # --- Step 1: fill a pending entry at the first tick of this bar ---
            opened_this_bar = False
            if position is None and pending is not None and n_in_bar > 0:
                sig = pending
                lots = size_position(balance, cfg.risk_pct, sig.stop_distance, pv, cfg.max_lots)
                if lots <= 0:
                    pending = None  # cannot afford a lot -> drop the signal
                else:
                    sign = int(sig.direction)
                    raw_fill = float(tick_price[lo])
                    entry_eff = raw_fill + sign * half_spread  # pay the spread on entry
                    balance -= cfg.commission * lots
                    stop_price = entry_eff - sign * sig.stop_distance
                    take_price = entry_eff + sign * sig.take_distance
                    position = _Position(
                        direction=sig.direction,
                        entry_time=pd.Timestamp(tick_time[lo]),
                        entry_price=entry_eff,
                        lots=lots,
                        stop_price=stop_price,
                        take_price=take_price,
                    )
                    pending = None
                    opened_this_bar = True

            # --- Step 2: resolve the OCO bracket on this bar's ticks, in order ---
            if position is not None and n_in_bar > 0:
                scan_start = lo + 1 if opened_this_bar else lo
                sign = int(position.direction)
                for k in range(scan_start, hi):
                    price = float(tick_price[k])
                    if sign == 1:  # long
                        hit_stop = price <= position.stop_price
                        hit_take = price >= position.take_price
                    else:  # short
                        hit_stop = price >= position.stop_price
                        hit_take = price <= position.take_price
                    if not (hit_stop or hit_take):
                        continue
                    # Take the level this tick reached. Stop and take cannot both be
                    # reached by one scalar tick price, but if they ever were we prefer
                    # the stop (the adverse outcome) to avoid overstating results.
                    if hit_stop:
                        exit_level, reason = position.stop_price, ExitReason.STOP
                    else:
                        exit_level, reason = position.take_price, ExitReason.TAKE
                    balance, trade = _close_position(
                        position,
                        exit_level,
                        pd.Timestamp(tick_time[k]),
                        reason,
                        balance,
                        half_spread,
                        pv,
                        cfg.commission,
                    )
                    trades.append(trade)
                    position = None
                    break

            # --- Step 3: mark-to-market at the bar close ---
            close_price = float(bar_close[i])
            if position is not None:
                sign = int(position.direction)
                unrealized = sign * (close_price - position.entry_price) * position.lots * pv
            else:
                unrealized = 0.0
            equity_times.append(bar_series.time[i])
            equity_values.append(balance + unrealized)

            # --- Step 4: request a new signal, only while flat ---
            if position is None:
                next_sig = strategy.on_bar(bar_series.head(i + 1))
                if next_sig is not None:
                    pending = next_sig

        # Force-close any position still open at the end of the data.
        if position is not None:
            last_close = float(bar_close[n_bars - 1])
            balance, trade = _close_position(
                position,
                last_close,
                pd.Timestamp(bar_series.time[n_bars - 1]),
                ExitReason.END,
                balance,
                half_spread,
                pv,
                cfg.commission,
            )
            trades.append(trade)
            position = None
            # Reflect the realized liquidation in the final equity point.
            equity_values[-1] = balance

        equity_curve = pd.Series(
            equity_values,
            index=pd.DatetimeIndex(np.array(equity_times, dtype="datetime64[ns]"), name="time"),
            name="equity",
            dtype=float,
        )
        return BacktestResult(equity_curve, trades, cfg.initial_balance, balance, cfg)


def _close_position(
    position: _Position,
    exit_level: float,
    exit_time: pd.Timestamp,
    reason: ExitReason,
    balance: float,
    half_spread: float,
    point_value: float,
    commission: float,
) -> tuple[float, Trade]:
    """Close ``position`` at ``exit_level``, returning the new balance and the trade.

    The effective exit price pays the spread on the closing side; PnL is net of both
    spread sides (embedded in the effective prices) and the round-turn commission, so
    that the change in ``balance`` over the trade equals ``Trade.pnl`` exactly.
    """
    sign = int(position.direction)
    exit_eff = exit_level - sign * half_spread  # pay the spread on exit
    gross = sign * (exit_eff - position.entry_price) * position.lots * point_value
    balance += gross
    balance -= commission * position.lots  # exit commission
    pnl = gross - 2.0 * commission * position.lots
    trade = Trade(
        entry_time=position.entry_time,
        entry_price=position.entry_price,
        exit_time=exit_time,
        exit_price=exit_eff,
        direction=position.direction,
        lots=position.lots,
        pnl=pnl,
        exit_reason=reason,
    )
    return balance, trade


def run_backtest(
    strategy: Strategy,
    bars: BarSeries | Sequence[Bar],
    ticks: TickSeries | Sequence[Tick],
    config: EngineConfig | None = None,
) -> BacktestResult:
    """Functional convenience wrapper around :class:`Backtester`."""
    return Backtester(config).run(strategy, bars, ticks)
