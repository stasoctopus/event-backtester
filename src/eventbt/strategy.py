"""Strategy interface and a demo strategy.

The design deliberately separates *alpha* from *execution and risk*:

* A :class:`Strategy` decides **when**, in **which direction**, and **how far** the
  protective stop and take-profit sit -- and returns a :class:`Signal`.
* The engine owns everything else: position **sizing** (lots), fills, the OCO
  bracket, costs, and accounting.

A strategy therefore never sees the account balance or the number of lots, which
keeps it pure and trivially testable.

:class:`SMACrossover` is a textbook simple-moving-average crossover. It is a *demo*
only -- it carries no real edge and exists purely to exercise the engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum

from .data import BarSeries

__all__ = ["Direction", "Signal", "Strategy", "SMACrossover"]


class Direction(IntEnum):
    """Trade direction. The integer value doubles as the PnL sign (+1 / -1)."""

    LONG = 1
    SHORT = -1


@dataclass(frozen=True, slots=True)
class Signal:
    """A request to open a position.

    ``stop_distance`` and ``take_distance`` are absolute price distances from the
    fill price to the protective stop and the take-profit, respectively. The engine
    converts ``stop_distance`` into a lot count via risk-based sizing.
    """

    direction: Direction
    stop_distance: float
    take_distance: float

    def __post_init__(self) -> None:
        if self.stop_distance <= 0:
            raise ValueError("stop_distance must be positive")
        if self.take_distance <= 0:
            raise ValueError("take_distance must be positive")


class Strategy(ABC):
    """Abstract base class for trading strategies."""

    def on_start(self, bars: BarSeries) -> None:
        """Optional warm-up hook, called once before the event loop.

        Override to precompute state. The per-bar decision must still be made in
        :meth:`on_bar`, which only ever receives data up to the current bar.
        """
        return None

    @abstractmethod
    def on_bar(self, bars: BarSeries) -> Signal | None:
        """Decide what to do at the close of the latest bar.

        Parameters
        ----------
        bars:
            A view truncated to the current bar -- ``bars[-1]`` is the bar that just
            closed and there is no access to future bars, so look-ahead bias is
            structurally impossible.

        Returns
        -------
        Signal | None
            A :class:`Signal` to request a new position, or ``None`` to do nothing.
            (The engine only asks for a signal while flat -- one position at a time.)
        """
        raise NotImplementedError


class SMACrossover(Strategy):
    """Demo strategy: go long when the fast SMA crosses above the slow SMA, and
    short on the opposite cross. Fixed stop/take distances keep it deterministic.

    Parameters
    ----------
    fast, slow:
        Fast and slow simple-moving-average windows (``fast < slow``).
    stop_distance, take_distance:
        Absolute price distances for the protective stop and take-profit.
    """

    def __init__(
        self,
        fast: int = 10,
        slow: int = 30,
        stop_distance: float = 1.0,
        take_distance: float = 2.0,
    ) -> None:
        if fast < 1 or slow < 1:
            raise ValueError("fast and slow must be >= 1")
        if fast >= slow:
            raise ValueError("fast window must be strictly smaller than slow window")
        self.fast = fast
        self.slow = slow
        self.stop_distance = stop_distance
        self.take_distance = take_distance

    def on_bar(self, bars: BarSeries) -> Signal | None:
        close = bars.close
        # Need one extra bar so both the current and previous SMA pair are defined.
        if len(close) <= self.slow:
            return None

        fast_now = float(close[-self.fast :].mean())
        slow_now = float(close[-self.slow :].mean())
        fast_prev = float(close[-self.fast - 1 : -1].mean())
        slow_prev = float(close[-self.slow - 1 : -1].mean())

        crossed_up = fast_prev <= slow_prev and fast_now > slow_now
        crossed_down = fast_prev >= slow_prev and fast_now < slow_now

        if crossed_up:
            return Signal(Direction.LONG, self.stop_distance, self.take_distance)
        if crossed_down:
            return Signal(Direction.SHORT, self.stop_distance, self.take_distance)
        return None
