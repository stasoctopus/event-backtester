"""Limit-order, session-aware futures execution engine (tick-accurate).

This is an **opt-in** execution mode that complements the core market-fill engine
in :mod:`eventbt.engine`. The core engine fills a pending signal as a *market*
order at the first tick of the next bar; that is the right model for many
strategies, but intraday futures desks usually work a more specific order
workflow that materially changes which signals become trades and at what price.

This module models that workflow exactly:

* **Limit entry with a time-to-live.** A signal places a limit order at
  ``close ∓ entry_offset``. It is live only for ``order_life_sec`` seconds after
  the bar closes; if price never reaches the limit, the signal produces *no
  trade* (a market order would always fill -- a real and often large difference).
  A filled order executes at the limit price exactly (no slippage past the
  limit, a standard documented simplification).
* **Tick-accurate protective bracket + EOD.** Stop, take, an optional breakeven
  stop, and a session end-of-day forced exit all compete; the earliest event in
  time wins, resolved by replaying ticks in order.
* **Notional commission.** Commission is charged on the traded notional
  (``entry_price * cost_per_step / min_step``) per side, round-turn = twice --
  the exchange-style futures fee, not a flat per-lot fee.
* **Risk-based sizing with a floor.** ``lots = clamp(floor(balance * risk_pct /
  (stop_distance * point_value)), min_lots, max_lots)`` -- ``min_lots`` defaults
  to 1 (always trade at least one lot when sizing is on).
* **Trading session.** A window ``[trade_start_time, trade_end_time)``, weekend
  skip, an entry cutoff, and the EOD exit are honoured.
* **Per-segment compounding.** Data is supplied as a list of segments (e.g. one
  per month/contract file); balance compounds across them in ``reinvest`` mode.

Nothing here is instrument- or project-specific: every market constant is a
:class:`FuturesConfig` field supplied by the caller, so the module is safe to
ship as part of the public library.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import BarSeries, TickSeries
from .strategy import Direction

__all__ = [
    "FuturesConfig",
    "FuturesSignal",
    "FuturesStrategy",
    "FuturesTrade",
    "FuturesSegment",
    "FuturesResult",
    "run_futures_backtest",
]

# A breakeven trigger at or above this (price units) means "breakeven disabled".
_BE_DISABLED = 900.0

_TRADE_COLUMNS = [
    "file", "signal_time", "side", "lots", "entry_time", "exit_time", "hold_sec",
    "close5", "atr5", "entry_offset", "stop_dist", "take_dist", "entry_price",
    "exit_price", "exit_reason", "pnl_before", "commission", "pnl",
    "mfe_money", "mae_money", "r_mult", "risk_money",
]


@dataclass(frozen=True, slots=True)
class FuturesConfig:
    """Execution, cost, sizing and session configuration for the futures engine.

    All money math mirrors a real intraday futures account. ``point_value`` drives
    PnL (money per 1.0 price move per lot); ``cost_per_step``/``min_step`` drive the
    notional used for commission. They are kept separate (rather than folding into
    one multiplier) so the commission arithmetic is byte-for-byte reproducible.
    """

    # account / risk
    initial_balance: float = 100_000.0
    risk_pct: float = 0.01
    point_value: float = 1.0
    max_lots: int = 10
    min_lots: int = 1
    use_position_sizing: bool = True
    one_position_at_a_time: bool = True
    capital_mode: str = "reinvest"  # "reinvest" | "fixed"

    # contract / costs (generic placeholders; the caller supplies real constants)
    price_step: float = 0.001
    min_step: float = 0.001
    cost_per_step: float = 1.0
    commission_rate: float = 0.0  # per side; round-turn = 2x

    # order workflow
    order_life_sec: int = 60
    order_start_offset_s: int = 1

    # session
    session_enabled: bool = True
    session_entry_filter: bool = True
    session_force_exit: bool = True
    allow_weekend: bool = False
    trade_start_time: str = "09:00"
    trade_end_time: str = "17:00"
    entry_cutoff_hour: int = 16
    entry_cutoff_time: str = "16:30"
    block_weekdays: frozenset[int] = frozenset()
    block_hours: frozenset[int] = frozenset()
    session_exit_reason: str = "EOD"


@dataclass(frozen=True, slots=True)
class FuturesSignal:
    """A request to open a position with a limit entry and a protective bracket.

    ``stop_distance`` / ``take_distance`` / ``entry_offset`` are absolute price
    distances. ``be_trigger`` >= 900 disables the breakeven stop. ``atr`` is carried
    purely for reporting (the ``atr5`` column) and never affects money math.
    """

    direction: Direction
    stop_distance: float
    take_distance: float
    entry_offset: float = 0.0
    be_trigger: float = 999.0
    be_offset: float = 0.0
    atr: float = float("nan")


@dataclass(frozen=True, slots=True)
class FuturesTrade:
    """A completed round-turn trade. ``pnl`` is net of (notional) commission."""

    file: str
    signal_time: pd.Timestamp
    side: str
    lots: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    hold_sec: float
    close5: float
    atr5: float
    entry_offset: float
    stop_dist: float
    take_dist: float
    entry_price: float
    exit_price: float
    exit_reason: str
    pnl_before: float
    commission: float
    pnl: float
    mfe_money: float
    mae_money: float
    r_mult: float
    risk_money: float


@dataclass(frozen=True, slots=True)
class FuturesSegment:
    """One independently-cached data segment (e.g. a month/contract file).

    ``bars.time[i]`` is the bar's **close** timestamp (right-labelled): the limit
    order opens ``order_start_offset_s`` seconds after it. ``ticks`` is the fine
    execution stream used for fills and the bracket scan.
    """

    bars: BarSeries
    ticks: TickSeries
    label: str = ""


@dataclass(frozen=True, slots=True)
class FuturesResult:
    """Output of a futures backtest: trades, monthly PnL series, and balances."""

    trades: list[FuturesTrade]
    month_pnl: list[float]
    month_end_caps: list[float]
    initial_balance: float
    final_balance: float
    config: FuturesConfig

    def trades_frame(self) -> pd.DataFrame:
        """Return the trade log as a DataFrame (typed but empty if no trades)."""
        if not self.trades:
            return pd.DataFrame(columns=_TRADE_COLUMNS)
        return pd.DataFrame([{c: getattr(t, c) for c in _TRADE_COLUMNS} for t in self.trades])


class FuturesStrategy(ABC):
    """Abstract base class for futures strategies.

    The strategy supplies, per bar, a :class:`FuturesSignal` with *absolute* stop /
    take / entry-offset distances (it owns the indicator math, e.g. ATR-derived
    distances); the engine owns fills, the bracket, sizing, costs and accounting.
    """

    def on_segment(self, segment: FuturesSegment) -> None:
        """Optional hook called once at the start of each segment (precompute here)."""
        return None

    @abstractmethod
    def on_bar(self, bars: BarSeries, i: int) -> FuturesSignal | None:
        """Return the signal at the close of bar ``i`` (use only ``bars[:i+1]``)."""
        raise NotImplementedError


# ----------------------------------------------------------------------------
# Session helpers (generic, parameterized by FuturesConfig)
# ----------------------------------------------------------------------------
def _hhmm_td(hhmm: str) -> pd.Timedelta:
    return pd.to_timedelta(hhmm + ":00")


def _is_trading_day(ts: pd.Timestamp, cfg: FuturesConfig) -> bool:
    if cfg.allow_weekend:
        return True
    return pd.Timestamp(ts).weekday() < 5


def _is_in_window(ts: pd.Timestamp, cfg: FuturesConfig) -> bool:
    ts = pd.Timestamp(ts)
    if not _is_trading_day(ts, cfg):
        return False
    if ts.weekday() in cfg.block_weekdays:
        return False
    t = ts.time()
    start = pd.to_datetime(cfg.trade_start_time).time()
    end = pd.to_datetime(cfg.trade_end_time).time()
    if not ((t >= start) and (t < end)):
        return False
    if ts.hour in cfg.block_hours:
        return False
    cutoff_end = pd.to_datetime(cfg.entry_cutoff_time).time()
    return not (ts.hour == cfg.entry_cutoff_hour and t > cutoff_end)


def _session_eod(ts: pd.Timestamp, cfg: FuturesConfig) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.normalize() + _hhmm_td(cfg.trade_end_time)


# ----------------------------------------------------------------------------
# Tick-scan primitives (vectorized, half-open windows)
# ----------------------------------------------------------------------------
def _idx_range(t_ns: np.ndarray, t_start: np.datetime64, t_end: np.datetime64) -> tuple[int, int]:
    i0 = int(np.searchsorted(t_ns, t_start, side="right"))
    i1 = int(np.searchsorted(t_ns, t_end, side="right"))
    return i0, i1


def _first_leq(t_ns, p, i0, i1, level):
    if i1 <= i0:
        return None
    seg = p[i0:i1]
    mask = seg <= level
    if not mask.any():
        return None
    return pd.Timestamp(t_ns[i0 + int(np.argmax(mask))])


def _first_geq(t_ns, p, i0, i1, level):
    if i1 <= i0:
        return None
    seg = p[i0:i1]
    mask = seg >= level
    if not mask.any():
        return None
    return pd.Timestamp(t_ns[i0 + int(np.argmax(mask))])


def _slice_min_max(p, j0, j1):
    if j1 <= j0:
        return np.nan, np.nan
    seg = p[j0:j1]
    return float(np.min(seg)), float(np.max(seg))


def _calc_lots(balance: float, stop_dist: float, cfg: FuturesConfig) -> int:
    if not cfg.use_position_sizing or stop_dist <= 0:
        return cfg.min_lots
    stop_money = stop_dist * cfg.point_value
    lots = int(balance * cfg.risk_pct / stop_money)
    return max(cfg.min_lots, min(lots, cfg.max_lots))


def _run_segment(
    strategy: FuturesStrategy,
    segment: FuturesSegment,
    cfg: FuturesConfig,
    balance: float,
) -> tuple[float, list[FuturesTrade]]:
    """Execute one segment, returning the new balance and the trades produced."""
    bars = segment.bars
    ticks = segment.ticks
    fname = segment.label
    t_ns = ticks.time
    p = ticks.price
    bar_time = bars.time
    bar_close = bars.close
    n_bars = len(bars)

    trades: list[FuturesTrade] = []
    in_pos_until: pd.Timestamp | None = None

    for i in range(n_bars):
        t_close_ts = pd.Timestamp(bar_time[i])

        if cfg.session_enabled and not _is_trading_day(t_close_ts, cfg):
            continue
        if cfg.one_position_at_a_time and in_pos_until is not None and t_close_ts <= in_pos_until:
            continue

        sig = strategy.on_bar(bars, i)
        if sig is None:
            continue

        stop_dist = float(sig.stop_distance)
        take_dist = float(sig.take_distance)
        entry_offset = float(sig.entry_offset)
        be_trigger = float(sig.be_trigger)
        be_offset = float(sig.be_offset)
        close5 = float(bar_close[i])
        signal_time = t_close_ts

        if cfg.session_enabled and cfg.session_entry_filter and not _is_in_window(signal_time, cfg):
            continue

        order_start = signal_time + pd.Timedelta(seconds=cfg.order_start_offset_s)
        order_end = order_start + pd.Timedelta(seconds=cfg.order_life_sec)
        i0, i1 = _idx_range(t_ns, np.datetime64(order_start), np.datetime64(order_end))
        if i1 <= i0:
            continue

        sign = int(sig.direction)
        if sign == 1:  # long
            limit = close5 - entry_offset
            entry_time = _first_leq(t_ns, p, i0, i1, limit - cfg.price_step)
            if entry_time is None:
                continue
            if cfg.session_enabled and cfg.session_entry_filter:
                if not _is_in_window(pd.Timestamp(entry_time), cfg):
                    continue
                eod = _session_eod(signal_time, cfg)
                if pd.Timestamp(entry_time) >= eod:
                    continue
            side = "long"
            entry_price = float(limit)
        else:  # short
            limit = close5 + entry_offset
            entry_time = _first_geq(t_ns, p, i0, i1, limit + cfg.price_step)
            if entry_time is None:
                continue
            if cfg.session_enabled and cfg.session_entry_filter:
                if not _is_in_window(pd.Timestamp(entry_time), cfg):
                    continue
                eod = _session_eod(signal_time, cfg)
                if pd.Timestamp(entry_time) >= eod:
                    continue
            side = "short"
            entry_price = float(limit)

        j0 = int(np.searchsorted(t_ns, np.datetime64(entry_time), side="left"))
        j1 = len(p)

        if side == "long":
            tp = entry_price + take_dist
            sl0 = entry_price - stop_dist
            t_be = (
                _first_geq(t_ns, p, j0, j1, entry_price + be_trigger)
                if be_trigger < _BE_DISABLED else None
            )
            if t_be is None:
                t_sl0 = _first_leq(t_ns, p, j0, j1, sl0)
                t_sl_be = None
            else:
                k_be = int(np.searchsorted(t_ns, np.datetime64(t_be), side="left"))
                t_sl0 = _first_leq(t_ns, p, j0, k_be, sl0)
                t_sl_be = _first_leq(t_ns, p, k_be, j1, entry_price + be_offset)
            t_tp = _first_geq(t_ns, p, j0, j1, tp)
            eod_time = (
                _session_eod(signal_time, cfg)
                if (cfg.session_enabled and cfg.session_force_exit) else None
            )
            if eod_time is not None and eod_time <= entry_time:
                eod_time = None

            events = []
            if t_sl0 is not None:
                events.append((t_sl0, sl0, "SL"))
            if t_tp is not None:
                events.append((t_tp, tp, "TP"))
            if t_sl_be is not None:
                events.append((t_sl_be, entry_price + be_offset, "BE"))
            if eod_time is not None:
                idx_eod = int(np.searchsorted(t_ns, np.datetime64(eod_time), side="left"))
                if idx_eod < len(p):
                    events.append((eod_time, float(p[idx_eod]), cfg.session_exit_reason))

            if not events:
                exit_time = pd.Timestamp(t_ns[-1])
                exit_price = float(p[-1])
                exit_reason = "EOD"
            else:
                exit_time, exit_price, exit_reason = min(events, key=lambda x: x[0])
            pnl_before = (exit_price - entry_price) * cfg.point_value
        else:  # short
            tp = entry_price - take_dist
            sl0 = entry_price + stop_dist
            t_be = (
                _first_leq(t_ns, p, j0, j1, entry_price - be_trigger)
                if be_trigger < _BE_DISABLED else None
            )
            if t_be is None:
                t_sl0 = _first_geq(t_ns, p, j0, j1, sl0)
                t_sl_be = None
            else:
                k_be = int(np.searchsorted(t_ns, np.datetime64(t_be), side="left"))
                t_sl0 = _first_geq(t_ns, p, j0, k_be, sl0)
                t_sl_be = _first_geq(t_ns, p, k_be, j1, entry_price - be_offset)
            t_tp = _first_leq(t_ns, p, j0, j1, tp)
            eod_time = (
                _session_eod(signal_time, cfg)
                if (cfg.session_enabled and cfg.session_force_exit) else None
            )
            if eod_time is not None and eod_time <= entry_time:
                eod_time = None

            events = []
            if t_sl0 is not None:
                events.append((t_sl0, sl0, "SL"))
            if t_tp is not None:
                events.append((t_tp, tp, "TP"))
            if t_sl_be is not None:
                events.append((t_sl_be, entry_price - be_offset, "BE"))
            if eod_time is not None:
                idx_eod = int(np.searchsorted(t_ns, np.datetime64(eod_time), side="left"))
                if idx_eod < len(p):
                    events.append((eod_time, float(p[idx_eod]), cfg.session_exit_reason))

            if not events:
                exit_time = pd.Timestamp(t_ns[-1])
                exit_price = float(p[-1])
                exit_reason = "EOD"
            else:
                exit_time, exit_price, exit_reason = min(events, key=lambda x: x[0])
            pnl_before = (entry_price - exit_price) * cfg.point_value

        lots = _calc_lots(balance, stop_dist, cfg)
        pnl_before = pnl_before * lots
        contract_value = entry_price * (cfg.cost_per_step / cfg.min_step)
        commission = contract_value * cfg.commission_rate * 2 * lots
        pnl = pnl_before - commission

        exit_idx = int(np.searchsorted(t_ns, np.datetime64(exit_time), side="right"))
        px_min, px_max = _slice_min_max(p, j0, exit_idx)
        if side == "long":
            mfe_price = (px_max - entry_price) if np.isfinite(px_max) else np.nan
            mae_price = (px_min - entry_price) if np.isfinite(px_min) else np.nan
        else:
            mfe_price = (entry_price - px_min) if np.isfinite(px_min) else np.nan
            mae_price = (entry_price - px_max) if np.isfinite(px_max) else np.nan
        mfe_money = mfe_price * cfg.point_value * lots if np.isfinite(mfe_price) else np.nan
        mae_money = mae_price * cfg.point_value * lots if np.isfinite(mae_price) else np.nan

        balance += pnl

        hold_sec = (pd.Timestamp(exit_time) - pd.Timestamp(entry_time)).total_seconds()
        risk_money = stop_dist * cfg.point_value
        r_mult = (pnl_before / risk_money) if (risk_money and risk_money > 0) else np.nan

        trades.append(FuturesTrade(
            file=fname,
            signal_time=signal_time,
            side=side,
            lots=int(lots),
            entry_time=pd.Timestamp(entry_time),
            exit_time=pd.Timestamp(exit_time),
            hold_sec=float(hold_sec),
            close5=float(close5),
            atr5=float(sig.atr),
            entry_offset=float(entry_offset),
            stop_dist=float(stop_dist),
            take_dist=float(take_dist),
            entry_price=float(entry_price),
            exit_price=float(exit_price),
            exit_reason=exit_reason,
            pnl_before=float(pnl_before),
            commission=float(commission),
            pnl=float(pnl),
            mfe_money=float(mfe_money) if np.isfinite(mfe_money) else np.nan,
            mae_money=float(mae_money) if np.isfinite(mae_money) else np.nan,
            r_mult=float(r_mult) if np.isfinite(r_mult) else np.nan,
            risk_money=float(risk_money),
        ))

        if cfg.one_position_at_a_time:
            in_pos_until = pd.Timestamp(exit_time)

    return balance, trades


def run_futures_backtest(
    strategy: FuturesStrategy,
    segments: Sequence[FuturesSegment],
    config: FuturesConfig | None = None,
) -> FuturesResult:
    """Run ``strategy`` over labelled ``segments`` with per-segment compounding.

    In ``reinvest`` mode the account balance carries continuously across segments;
    in ``fixed`` mode every segment restarts from ``initial_balance`` and the final
    balance is ``initial_balance + sum(month_pnl)``.
    """
    cfg = config if config is not None else FuturesConfig()
    capital = cfg.initial_balance
    all_trades: list[FuturesTrade] = []
    month_pnl: list[float] = []
    month_end_caps: list[float] = []

    for segment in segments:
        strategy.on_segment(segment)
        start_cap = capital if cfg.capital_mode == "reinvest" else float(cfg.initial_balance)
        balance = float(start_cap)
        balance, seg_trades = _run_segment(strategy, segment, cfg, balance)
        all_trades.extend(seg_trades)
        month_pnl.append(float(balance - start_cap))
        month_end_caps.append(float(balance))
        if cfg.capital_mode == "reinvest":
            capital = float(balance)

    if cfg.capital_mode == "reinvest":
        final_balance = float(capital)
    else:
        final_balance = float(cfg.initial_balance + sum(month_pnl))

    return FuturesResult(
        trades=all_trades,
        month_pnl=month_pnl,
        month_end_caps=month_end_caps,
        initial_balance=float(cfg.initial_balance),
        final_balance=final_balance,
        config=cfg,
    )
