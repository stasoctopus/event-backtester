"""Market data primitives and synthetic data generation.

This module defines the immutable scalar types (:class:`Bar`, :class:`Tick`) and
the columnar containers (:class:`BarSeries`, :class:`TickSeries`) that the engine
consumes. It also provides:

* :func:`gbm_data` -- a seeded Geometric Brownian Motion generator that produces a
  fine *tick* series and aggregates it into a fully consistent OHLC *bar* series.
  Because the bars are derived from the same ticks, the two streams never disagree.
* :func:`load_yfinance` -- an optional loader for public-market OHLC data. It is a
  thin convenience wrapper; ``yfinance`` is an optional dependency.

No proprietary data, instruments, or market specifics live here -- everything is
either synthetic or supplied by the caller.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import overload

import numpy as np
import pandas as pd

__all__ = [
    "Bar",
    "Tick",
    "BarSeries",
    "TickSeries",
    "gbm_data",
    "load_yfinance",
]


@dataclass(frozen=True, slots=True)
class Bar:
    """A single OHLCV bar. ``time`` is the bar's left edge (open) timestamp."""

    time: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass(frozen=True, slots=True)
class Tick:
    """A single price observation at a point in time (mid price)."""

    time: pd.Timestamp
    price: float


class BarSeries:
    """Columnar, immutable-by-convention container of OHLCV bars.

    Stored as parallel NumPy arrays so that strategy math can be vectorized and
    slicing returns cheap O(1) *views* (no copy). The engine hands a strategy a
    truncated view (:meth:`head`) so look-ahead bias is impossible by construction.
    """

    __slots__ = ("time", "open", "high", "low", "close", "volume")

    def __init__(
        self,
        time: np.ndarray,
        open: np.ndarray,  # noqa: A002 - mirrors the OHLC vocabulary
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
    ) -> None:
        self.time = np.asarray(time, dtype="datetime64[ns]")
        self.open = np.asarray(open, dtype=float)
        self.high = np.asarray(high, dtype=float)
        self.low = np.asarray(low, dtype=float)
        self.close = np.asarray(close, dtype=float)
        self.volume = np.asarray(volume, dtype=float)

    def __len__(self) -> int:
        return int(self.time.shape[0])

    @overload
    def __getitem__(self, idx: int) -> Bar: ...
    @overload
    def __getitem__(self, idx: slice) -> BarSeries: ...
    def __getitem__(self, idx: int | slice) -> Bar | BarSeries:
        if isinstance(idx, slice):
            return BarSeries(
                self.time[idx],
                self.open[idx],
                self.high[idx],
                self.low[idx],
                self.close[idx],
                self.volume[idx],
            )
        return Bar(
            time=pd.Timestamp(self.time[idx]),
            open=float(self.open[idx]),
            high=float(self.high[idx]),
            low=float(self.low[idx]),
            close=float(self.close[idx]),
            volume=float(self.volume[idx]),
        )

    def head(self, n: int) -> BarSeries:
        """Return a view of the first ``n`` bars (O(1), no copy)."""
        return self[:n]

    @classmethod
    def from_bars(cls, bars: Sequence[Bar]) -> BarSeries:
        """Build a :class:`BarSeries` from a sequence of :class:`Bar` objects."""
        if not bars:
            empty_t = np.empty(0, dtype="datetime64[ns]")
            empty_f = np.empty(0, dtype=float)
            return cls(empty_t, empty_f, empty_f, empty_f, empty_f, empty_f)
        return cls(
            time=np.array([np.datetime64(b.time) for b in bars], dtype="datetime64[ns]"),
            open=np.array([b.open for b in bars], dtype=float),
            high=np.array([b.high for b in bars], dtype=float),
            low=np.array([b.low for b in bars], dtype=float),
            close=np.array([b.close for b in bars], dtype=float),
            volume=np.array([b.volume for b in bars], dtype=float),
        )

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> BarSeries:
        """Build a :class:`BarSeries` from an OHLCV ``DataFrame``.

        Column names are matched case-insensitively. The timestamp is taken from a
        ``DatetimeIndex`` if present, otherwise from a ``time``/``date`` column.
        ``volume`` is optional and defaults to zero.
        """
        frame = df.copy()
        # Flatten a possible MultiIndex (e.g. yfinance with a single ticker).
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        lower = {str(c).lower(): c for c in frame.columns}

        if isinstance(frame.index, pd.DatetimeIndex):
            time = frame.index.to_numpy(dtype="datetime64[ns]")
        elif "time" in lower or "date" in lower:
            time_col = lower["time"] if "time" in lower else lower["date"]
            time = pd.to_datetime(frame[time_col]).to_numpy(dtype="datetime64[ns]")
        else:
            raise ValueError("from_frame requires a DatetimeIndex or a 'time'/'date' column")

        required = ("open", "high", "low", "close")
        missing = [c for c in required if c not in lower]
        if missing:
            raise ValueError(f"from_frame missing required column(s): {missing}")

        def col(name: str) -> np.ndarray:
            return frame[lower[name]].to_numpy(dtype=float)

        volume = col("volume") if "volume" in lower else np.zeros(len(frame), dtype=float)
        return cls(time, col("open"), col("high"), col("low"), col("close"), volume)

    def to_frame(self) -> pd.DataFrame:
        """Return the series as a ``DataFrame`` indexed by bar time."""
        return pd.DataFrame(
            {
                "open": self.open,
                "high": self.high,
                "low": self.low,
                "close": self.close,
                "volume": self.volume,
            },
            index=pd.DatetimeIndex(self.time, name="time"),
        )

    @classmethod
    def coerce(cls, bars: BarSeries | Sequence[Bar]) -> BarSeries:
        """Accept either a :class:`BarSeries` or a sequence of :class:`Bar`."""
        return bars if isinstance(bars, BarSeries) else cls.from_bars(bars)


class TickSeries:
    """Columnar container of price ticks (parallel ``time``/``price`` arrays)."""

    __slots__ = ("time", "price")

    def __init__(self, time: np.ndarray, price: np.ndarray) -> None:
        self.time = np.asarray(time, dtype="datetime64[ns]")
        self.price = np.asarray(price, dtype=float)

    def __len__(self) -> int:
        return int(self.time.shape[0])

    @overload
    def __getitem__(self, idx: int) -> Tick: ...
    @overload
    def __getitem__(self, idx: slice) -> TickSeries: ...
    def __getitem__(self, idx: int | slice) -> Tick | TickSeries:
        if isinstance(idx, slice):
            return TickSeries(self.time[idx], self.price[idx])
        return Tick(time=pd.Timestamp(self.time[idx]), price=float(self.price[idx]))

    def slice_by_time(self, start: pd.Timestamp, end: pd.Timestamp) -> TickSeries:
        """Return the ticks with ``start <= t < end`` (half-open) via binary search."""
        lo = int(np.searchsorted(self.time, np.datetime64(start), side="left"))
        hi = int(np.searchsorted(self.time, np.datetime64(end), side="left"))
        return self[lo:hi]

    @classmethod
    def from_ticks(cls, ticks: Sequence[Tick]) -> TickSeries:
        """Build a :class:`TickSeries` from a sequence of :class:`Tick` objects."""
        if not ticks:
            return cls(np.empty(0, dtype="datetime64[ns]"), np.empty(0, dtype=float))
        return cls(
            time=np.array([np.datetime64(t.time) for t in ticks], dtype="datetime64[ns]"),
            price=np.array([t.price for t in ticks], dtype=float),
        )

    @classmethod
    def coerce(cls, ticks: TickSeries | Sequence[Tick]) -> TickSeries:
        """Accept either a :class:`TickSeries` or a sequence of :class:`Tick`."""
        return ticks if isinstance(ticks, TickSeries) else cls.from_ticks(ticks)


def gbm_data(
    n_bars: int = 500,
    ticks_per_bar: int = 60,
    *,
    s0: float = 100.0,
    mu: float = 0.0,
    sigma: float = 0.20,
    seed: int = 0,
    start: str | pd.Timestamp = "2020-01-01",
    bar_freq: str = "1min",
    periods_per_year: int = 252,
) -> tuple[BarSeries, TickSeries]:
    """Generate a synthetic tick series via GBM and aggregate it into OHLC bars.

    The price path is a discrete Geometric Brownian Motion evaluated at every tick.
    Bars are then formed from consecutive groups of ``ticks_per_bar`` ticks, so the
    bar OHLC values are *exactly* the open/high/low/close of the underlying ticks --
    the two streams are guaranteed consistent.

    Parameters
    ----------
    n_bars:
        Number of bars to produce.
    ticks_per_bar:
        Number of ticks aggregated into each bar (the "fine" execution resolution).
    s0:
        Initial price.
    mu, sigma:
        Annualized drift and volatility of the GBM.
    seed:
        Seed for ``numpy.random.default_rng`` -- makes the output fully reproducible.
    start:
        Timestamp of the first tick.
    bar_freq:
        Pandas-style bar duration (e.g. ``"1min"``). Ticks are spaced evenly within.
    periods_per_year:
        Number of bars treated as one year for scaling the random walk.

    Returns
    -------
    (BarSeries, TickSeries)
        The aggregated bars and the underlying ticks.
    """
    if n_bars <= 0 or ticks_per_bar <= 0:
        raise ValueError("n_bars and ticks_per_bar must be positive")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")

    rng = np.random.default_rng(seed)
    n_ticks = n_bars * ticks_per_bar

    # Per-tick time step, derived from the (bar-level) annualization.
    dt = 1.0 / (periods_per_year * ticks_per_bar)
    drift = (mu - 0.5 * sigma * sigma) * dt
    diffusion = sigma * np.sqrt(dt)

    shocks = rng.standard_normal(n_ticks)
    log_returns = drift + diffusion * shocks
    log_price = np.log(s0) + np.cumsum(log_returns)
    prices = np.exp(log_price)

    # Even tick time grid: bar_delta / ticks_per_bar between consecutive ticks,
    # computed in integer nanoseconds to stay exact and version-agnostic.
    start_ns = pd.Timestamp(start).value
    bar_ns = pd.Timedelta(bar_freq).value
    tick_ns = bar_ns // ticks_per_bar
    tick_time_ints = start_ns + tick_ns * np.arange(n_ticks, dtype=np.int64)
    tick_times = tick_time_ints.astype("datetime64[ns]")

    ticks = TickSeries(tick_times, prices)

    # Aggregate consecutive ticks_per_bar ticks into one bar.
    grid = prices.reshape(n_bars, ticks_per_bar)
    bar_open = grid[:, 0]
    bar_close = grid[:, -1]
    bar_high = grid.max(axis=1)
    bar_low = grid.min(axis=1)
    bar_time = tick_times[::ticks_per_bar]
    bar_volume = np.full(n_bars, float(ticks_per_bar))

    bars = BarSeries(bar_time, bar_open, bar_high, bar_low, bar_close, bar_volume)
    return bars, ticks


def load_yfinance(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
) -> BarSeries:
    """Load public-market OHLC bars for ``symbol`` via the optional ``yfinance`` dep.

    The caller always supplies the symbol -- nothing is hardcoded. ``yfinance`` is an
    optional dependency; install it with ``pip install eventbt[data]``.

    Note: this returns *bars* only. To run the tick-accurate engine you must also
    provide a finer tick series (e.g. request an intraday ``interval`` and treat it
    as the execution resolution, or supply your own ticks).

    Example
    -------
    >>> bars = load_yfinance("SPY", start="2023-01-01", end="2023-06-30")  # doctest: +SKIP
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "load_yfinance() requires the optional 'yfinance' dependency. "
            "Install it with: pip install eventbt[data]"
        ) from exc

    df = yf.download(
        symbol,
        start=start,
        end=end,
        interval=interval,
        progress=False,
        auto_adjust=True,
    )
    if df is None or len(df) == 0:
        raise ValueError(f"No data returned for symbol {symbol!r}")
    return BarSeries.from_frame(df)
