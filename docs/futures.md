# Futures execution mode (`eventbt.futures`)

[← back to README](../README.md)

An **opt-in** execution engine for intraday futures, layered on top of the same
tick-accurate machinery as the core engine. The core
[`run_backtest`](../README.md#architecture) fills a pending signal as a *market*
order at the first tick of the next bar — the right model for many strategies. A
futures desk usually works a more specific order workflow that materially changes
*which* signals become trades and *at what price*. This module models that workflow
exactly, with every market constant supplied by the caller, so nothing here is
instrument- or project-specific.

It lives in [`src/eventbt/futures.py`](../src/eventbt/futures.py) and is exported from
the top-level package:

```python
from eventbt import (
    FuturesConfig, FuturesSignal, FuturesSegment, FuturesStrategy,
    FuturesTrade, FuturesResult, run_futures_backtest,
)
```

The core `run_backtest` and its existing test suite are **untouched** — this is purely
additive.

---

## Why a second execution mode?

Three differences from the core market-fill engine, and why each one matters.

### 1. Limit entry with a time-to-live (limit-TTL)

A signal does **not** become a guaranteed trade. It places a **limit order** at
`close ∓ entry_offset` that is live only for `order_life_sec` seconds after the bar
closes. If price never trades through the limit inside that window, the signal produces
**no trade at all**.

This is the single biggest behavioural gap versus a market-fill model, which *always*
fills. A passive limit both **skips** signals (the market ran away before you were
filled) and **improves** the ones it does take (you are filled at your price, not at the
next print). A backtest that fills everything at market systematically over-counts trades
and misprices entries.

A filled order executes at the **limit price exactly** — no slippage past the limit. That
is a standard, explicitly-documented simplification (a real limit can only fill *at or
better than* its price; modelling it *at* the price is the conservative-on-improvement,
neutral-on-fill choice).

### 2. Trading session + forced end-of-day exit

Real futures sessions are not 24/7. The engine honours a `[trade_start_time,
trade_end_time)` window, an optional weekend skip, an entry cutoff near the close, and a
**forced end-of-day (EOD) exit** that flattens any open position at `trade_end_time`. A
strategy cannot accidentally "hold overnight" a position the live desk would have closed.

### 3. Notional (exchange-style) commission

Commission is charged on the **traded notional**, per side, round-turn:

```
commission = entry_price * (cost_per_step / min_step) * commission_rate * 2 * lots
```

This is the exchange/broker fee structure for many futures contracts — a fraction of
contract value — not a flat per-lot or per-share fee like the core engine's `commission`.
`point_value` (money per 1.0 price move per lot) and `cost_per_step / min_step` (the
notional multiplier) are kept as **separate** config fields rather than folded into one
number, so the commission arithmetic is byte-for-byte reproducible.

### At a glance

| | Core engine (`run_backtest`) | Futures engine (`run_futures_backtest`) |
|---|---|---|
| Entry | market, first tick of next bar (always fills) | **limit** at `close ∓ entry_offset`, TTL `order_life_sec` (**may not fill**) |
| Fill price | next tick (± spread) | the limit price exactly |
| Exit bracket | OCO stop / take (tick order) | stop / take / **breakeven** / **EOD** (tick order) |
| Session | none | window + weekend skip + entry cutoff + forced EOD |
| Commission | flat per trade | **notional**, round-turn |
| Data shape | one `BarSeries` + `TickSeries` | a list of **segments** (per file/contract), compounded |

Everything else — tick-accurate intrabar resolution, risk-based sizing, one position at a
time — is shared with the core engine.

---

## Per-bar execution order

For each bar `i` in a segment (where **`bars.time[i]` is the bar's _close_ timestamp**):

1. **Session / occupancy gate.** Skip the bar if the session is enabled and it is not a
   trading day, or if a position is still open (`one_position_at_a_time`).
2. **Ask the strategy.** `signal = strategy.on_bar(bars, i)`. `None` ⇒ nothing to do.
   The strategy must look only at `bars[:i+1]`.
3. **Entry-time filter.** If session filtering is on and the signal bar's close is outside
   the trading window (or past the entry cutoff), drop the signal.
4. **Place the limit.** Limit `= close − entry_offset` (long) or `close + entry_offset`
   (short). It is live in the window starting `order_start_offset_s` seconds after the bar
   close and lasting `order_life_sec` seconds.
5. **Try to fill.** Replay the window's ticks in order. A **long** fills on the first tick
   `≤ limit − price_step`; a **short** on the first tick `≥ limit + price_step`. No
   qualifying tick ⇒ **no trade**. The fill price is the limit, exactly.
6. **Resolve the exit.** From the fill onward, compute the times of stop, take, the
   optional breakeven stop, and the EOD exit, then take the **earliest** event. On a tie
   the list order wins — **SL → TP → BE → EOD**. EOD is that day's `trade_end_time`.
7. **Size, cost, book.** Compute `lots` from the risk budget, the notional round-turn
   commission, the net PnL, and `MFE`/`MAE`/`R`-multiple over the holding window; append a
   `FuturesTrade`; mark the position open until its exit time.

After all bars in a segment, the per-segment PnL and ending balance are recorded; in
`reinvest` mode the balance carries into the next segment.

---

## API

### `FuturesConfig` (frozen dataclass)

Execution, cost, sizing and session configuration. All fields have defaults.

| Field | Default | Meaning |
|---|---|---|
| `initial_balance` | `100_000.0` | Starting account balance. |
| `risk_pct` | `0.01` | Fraction of balance risked per trade (sizing). |
| `point_value` | `1.0` | Money per 1.0 price move, per lot (drives PnL). |
| `max_lots` | `10` | Upper clamp on position size. |
| `min_lots` | `1` | Lower clamp; also the size when sizing is off. |
| `use_position_sizing` | `True` | If `False`, every trade is `min_lots`. |
| `one_position_at_a_time` | `True` | Block new signals while a position is open. |
| `capital_mode` | `"reinvest"` | `"reinvest"` compounds across segments; `"fixed"` restarts each segment from `initial_balance`. |
| `price_step` | `0.001` | Tick size used for the fill-through test (`limit ∓ price_step`). |
| `min_step` | `0.001` | Price step in the commission notional denominator. |
| `cost_per_step` | `1.0` | Money per `min_step` of price (commission notional numerator). |
| `commission_rate` | `0.0` | Commission rate **per side** (round-turn applies `×2`); `0.0` = no fee. |
| `order_life_sec` | `60` | Limit-order time-to-live, in seconds. |
| `order_start_offset_s` | `1` | Seconds after the bar close before the order goes live. |
| `session_enabled` | `True` | Master switch for the whole session model. |
| `session_entry_filter` | `True` | Reject entries outside the window / past the cutoff. |
| `session_force_exit` | `True` | Force-flatten open positions at EOD. |
| `allow_weekend` | `False` | If `False`, Sat/Sun bars are skipped. |
| `trade_start_time` | `"09:00"` | Session open (`HH:MM`). |
| `trade_end_time` | `"17:00"` | Session close and EOD exit time (`HH:MM`). |
| `entry_cutoff_hour` | `16` | Hour in which the late-entry cutoff applies. |
| `entry_cutoff_time` | `"16:30"` | No new entries after this time within the cutoff hour. |
| `block_weekdays` | `frozenset()` | Weekday indices (`0=Mon`) to skip entirely. |
| `block_hours` | `frozenset()` | Hours of day to skip entirely. |
| `session_exit_reason` | `"EOD"` | Label written to the trade log for forced exits. |

### `FuturesSignal` (frozen dataclass)

A request to open a position with a limit entry and a protective bracket. All distances are
**absolute price distances**.

| Field | Default | Meaning |
|---|---|---|
| `direction` | — | `eventbt.Direction.LONG` (`+1`) or `eventbt.Direction.SHORT` (`-1`). |
| `stop_distance` | — | Stop distance from the fill price (also drives sizing). |
| `take_distance` | — | Take-profit distance from the fill price. |
| `entry_offset` | `0.0` | Limit offset from the bar close (passive entry). |
| `be_trigger` | `999.0` | Favourable move that arms the breakeven stop. **`≥ 900` disables breakeven.** |
| `be_offset` | `0.0` | Where the breakeven stop sits once armed (`entry ± be_offset`). |
| `atr` | `nan` | Reporting only (the `atr5` column); never affects money math. |

The strategy owns the indicator math (e.g. ATR-derived distances); the engine owns fills,
the bracket, sizing, costs and accounting — so a signal never sees the balance or lot count.

### `FuturesSegment` (frozen dataclass)

One independently-cached data segment, e.g. a single month or contract file.

| Field | Default | Meaning |
|---|---|---|
| `bars` | — | `eventbt.BarSeries`. **`bars.time[i]` is the bar's _close_ time.** |
| `ticks` | — | `eventbt.TickSeries` — the fine execution stream for fills and the bracket scan. |
| `label` | `""` | Free-form tag; copied into each trade's `file` column. |

### `FuturesStrategy` (ABC)

```python
class FuturesStrategy(ABC):
    def on_segment(self, segment: FuturesSegment) -> None: ...      # optional precompute
    @abstractmethod
    def on_bar(self, bars: BarSeries, i: int) -> FuturesSignal | None: ...
```

- `on_segment(segment)` — optional hook called once at the start of each segment (cache
  indicators here).
- `on_bar(bars, i)` — **required.** Return a `FuturesSignal` to request a trade at the close
  of bar `i`, or `None`. Use only `bars[:i+1]`; reaching past `i` is look-ahead.

### `run_futures_backtest(strategy, segments, config=None) -> FuturesResult`

Runs `strategy` over the labelled `segments` with per-segment compounding. In `reinvest`
mode the balance carries continuously across segments; in `fixed` mode each segment restarts
from `initial_balance` and `final_balance = initial_balance + sum(month_pnl)`.

`FuturesResult` fields:

| Field | Type | Meaning |
|---|---|---|
| `trades` | `list[FuturesTrade]` | Every completed round-turn trade. |
| `month_pnl` | `list[float]` | PnL per segment (segment-end balance − segment-start balance). |
| `month_end_caps` | `list[float]` | Account balance at the end of each segment. |
| `initial_balance` | `float` | Echo of the configured start balance. |
| `final_balance` | `float` | Final account balance. |
| `config` | `FuturesConfig` | The config that produced this result. |
| `trades_frame()` | `pd.DataFrame` | The trade log (typed but empty when there are no trades). |

Each `FuturesTrade` (and each row of `trades_frame()`) carries:
`file`, `signal_time`, `side`, `lots`, `entry_time`, `exit_time`, `hold_sec`, `close5`,
`atr5`, `entry_offset`, `stop_dist`, `take_dist`, `entry_price`, `exit_price`,
`exit_reason`, `pnl_before`, `commission`, `pnl`, `mfe_money`, `mae_money`, `r_mult`,
`risk_money`. `pnl` is net of commission; `pnl_before` is gross.

---

## Minimal example

Build a tiny deterministic segment by hand and fire one signal. The strategy goes long at
bar 0; the limit fills passively and the position reaches its take-profit.

```python
import numpy as np
import pandas as pd
from eventbt import (
    BarSeries, TickSeries, Direction,
    FuturesConfig, FuturesSignal, FuturesSegment, FuturesStrategy,
    run_futures_backtest,
)

# One trading day, three 5-minute bars. NOTE: bars.time[i] is the bar's CLOSE.
bar_time = np.array(
    ["2023-01-02T13:00:00", "2023-01-02T13:05:00", "2023-01-02T13:10:00"],
    dtype="datetime64[ns]",
)
bars = BarSeries(
    time=bar_time,
    open=np.array([100.0, 100.0, 100.0]),
    high=np.array([100.5, 105.5, 100.5]),
    low=np.array([99.5, 99.5, 99.5]),
    close=np.array([100.0, 100.0, 100.0]),
    volume=np.array([10.0, 10.0, 10.0]),
)

# Ticks after bar 0's close (13:00:00). The limit sits at 99.95; it fills on the
# first tick <= 99.95 - price_step (= 99.94), then price runs up to the take.
t0 = pd.Timestamp("2023-01-02T13:00:00")
seq = [100.00, 99.96, 99.92, 100.50, 102.0, 104.0, 104.95, 105.10]
times = [np.datetime64(t0 + pd.Timedelta(seconds=2 * (k + 1))) for k in range(len(seq))]
ticks = TickSeries(np.array(times, dtype="datetime64[ns]"), np.array(seq, dtype=float))

segment = FuturesSegment(bars=bars, ticks=ticks, label="2023-01")


class OneShotLong(FuturesStrategy):
    """Go long exactly once, at the first bar; otherwise stand aside."""

    def on_bar(self, bars, i):
        if i == 0:
            return FuturesSignal(
                direction=Direction.LONG,
                stop_distance=2.0,    # SL = entry - 2.0
                take_distance=5.0,    # TP = entry + 5.0
                entry_offset=0.05,    # limit = close - 0.05 = 99.95
            )
        return None


cfg = FuturesConfig(
    initial_balance=100_000.0, risk_pct=0.01, point_value=10.0, max_lots=5,
    price_step=0.01, min_step=0.01, cost_per_step=2.0, commission_rate=0.0001,
    trade_start_time="09:00", trade_end_time="17:00",
)

result = run_futures_backtest(OneShotLong(), [segment], cfg)
print(result.trades_frame()[
    ["side", "lots", "entry_price", "exit_price", "exit_reason",
     "pnl_before", "commission", "pnl"]
].to_string(index=False))
print("final_balance:", round(result.final_balance, 2))
```

Output:

```
side  lots  entry_price  exit_price exit_reason  pnl_before  commission    pnl
long     5        99.95      104.95          TP       250.0       19.99 230.01
final_balance: 100230.01
```

Reading the numbers:

- **Limit fill.** `limit = 100.00 − 0.05 = 99.95`; it arms on the first tick
  `≤ 99.95 − price_step = 99.94`, which is the `99.92` print. The fill price is the limit,
  `99.95` (had no tick reached `99.94`, there would have been **no trade**).
- **Sizing.** `lots = clamp(int(100_000 × 0.01 / (2.0 × 10)), 1, 5) = clamp(50, 1, 5) = 5`.
- **Exit.** Take-profit at `99.95 + 5.0 = 104.95` is hit; `pnl_before = 5.0 × 10 × 5 = 250`.
- **Commission.** `99.95 × (2 / 0.01) × 0.0001 × 2 × 5 = 19.99`, so net `pnl = 230.01`.

### Larger sanity run with `gbm_data`

For a quick exercise of the full loop you can reuse the bundled GBM generator. Its
timestamps are not intraday-futures hours, so disable the session for the demo:

```python
from eventbt import (gbm_data, Direction, FuturesConfig, FuturesSignal,
                     FuturesSegment, FuturesStrategy, run_futures_backtest)

bars, ticks = gbm_data(n_bars=2000, ticks_per_bar=60, sigma=0.25, seed=7)

class MomentumDemo(FuturesStrategy):
    """Demo only (no edge): long after an up-bar, short after a down-bar."""
    def on_bar(self, bars, i):
        if i < 1:
            return None
        d = Direction.LONG if bars.close[i] > bars.close[i - 1] else Direction.SHORT
        return FuturesSignal(direction=d, stop_distance=0.5, take_distance=1.0,
                             entry_offset=0.02)

cfg = FuturesConfig(point_value=1.0, price_step=0.01, min_step=0.01,
                    cost_per_step=1.0, commission_rate=0.0001,
                    session_enabled=False)   # GBM data is not intraday hours

res = run_futures_backtest(MomentumDemo(), [FuturesSegment(bars, ticks, "gbm")], cfg)
print(len(res.trades), round(res.final_balance, 2))   # -> 1703 85434.68
```

As with the core engine's demo, this strategy carries **no edge** — it exists only to
exercise the engine end-to-end.

---

## Sizing and commission, precisely

**Sizing** (per trade):

```
lots = clamp(int(balance * risk_pct / (stop_distance * point_value)), min_lots, max_lots)
```

`int(...)` truncates toward zero. If `use_position_sizing` is `False`, or
`stop_distance ≤ 0`, the size is `min_lots`. Because risk is anchored to `stop_distance`,
every trade risks roughly the same fraction of equity regardless of stop width.

**Commission** (round-turn, notional):

```
commission = entry_price * (cost_per_step / min_step) * commission_rate * 2 * lots
```

The `× 2` is the round turn (entry + exit). `point_value` is deliberately *not* part of this
formula — PnL and commission use independent multipliers.

---

## Session model

When `session_enabled` is on:

- **Trading window.** Entries (and EOD) respect `[trade_start_time, trade_end_time)`.
- **Weekends.** Skipped unless `allow_weekend`.
- **Entry cutoff.** No new entries after `entry_cutoff_time` within `entry_cutoff_hour`
  (e.g. nothing new in the final part of the session) — controlled by `session_entry_filter`.
- **Blocks.** `block_weekdays` and `block_hours` remove whole weekdays / hours.
- **Forced EOD exit.** With `session_force_exit`, any position still open at
  `trade_end_time` is flattened and labelled `session_exit_reason` (default `"EOD"`).

Set `session_enabled=False` to run the engine as a pure limit-TTL model with no calendar.

---

## Segments and compounding

Data is supplied as a **list of `FuturesSegment`** — typically one per month or contract
file. Each segment is processed in order, and:

- `capital_mode="reinvest"` (default) carries the balance from one segment into the next, so
  the equity compounds across the whole list.
- `capital_mode="fixed"` restarts every segment from `initial_balance`; the final balance is
  `initial_balance + sum(month_pnl)`.

`month_pnl[k]` and `month_end_caps[k]` report the PnL and ending balance of segment `k`.

---

## Validated bit-for-bit against an external engine

The execution loop is not just internally consistent — it has been validated against an
independent production backtester on real intraday futures data, across two different
contracts and multiple years. Every trade-log column matched with `max|Δ| = 0.0` and all
headline metrics were identical.

The fidelity trick: feed both engines the *identical* signal cache, so only the execution
loop differs. Achieving zero column-wise difference across hundreds of real trades is strong
evidence that the limit-TTL fills, the tick-ordered bracket, the session/EOD logic, and the
notional commission all behave correctly. The comparison harness, the instrument constants,
and the strategies themselves live outside this repository — the engine here stays a clean,
self-contained, strategy-agnostic library.

---

## Limitations

- One position at a time (`one_position_at_a_time`); no portfolio of simultaneous positions.
- A filled limit executes *at* its price — the engine does not model partial fills, queue
  position, or price improvement beyond the limit.
- The session model assumes a single daily `[start, end)` window per calendar day.
- Not financial advice. The bundled demo strategies carry no edge and exist only to exercise
  the engine.
```
