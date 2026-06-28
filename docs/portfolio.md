# Portfolio execution mode (`eventbt.portfolio`)

[← back to README](../README.md)

An **opt-in** execution engine for **target-weight, multi-leg portfolios**, layered on top
of the same package as the core and futures engines. The core
[`run_backtest`](../README.md#architecture) and the
[futures `run_futures_backtest`](futures.md) both model **discrete trades** — an entry, an
exit, a realised PnL booked one position at a time. A trend / carry / vol-targeting book does
not work that way: it holds a **continuously rebalanced signed weight** and accrues a
mark-to-market return every bar. This module models that workflow exactly, with every cost
and annualisation constant supplied by the caller, so nothing here is instrument- or
project-specific.

It lives in [`src/eventbt/portfolio.py`](../src/eventbt/portfolio.py) and is exported from the
top-level package:

```python
from eventbt import (
    PortfolioConfig, PositionResult, run_position_backtest,
    to_daily, carry_leg_returns, combine_daily, portfolio_metrics,
)
```

The core `run_backtest`, the futures engine, and their existing test suites are **untouched** —
this is purely additive.

---

## Why a third execution mode?

The two discrete-trade engines answer the question *"when I enter and exit, what is the
realised PnL of this trade?"* A weight-based book asks a different question: *"given a target
exposure that changes every bar, what is the mark-to-market return of holding and rebalancing
it?"* Three differences follow.

### 1. A continuously held weight, not a trade

The strategy does not emit entries and exits. At every coarse bar it emits a **signed target
weight** `pos_t` (e.g. `+1` fully long, `-0.3` 30% short, `0` flat). The engine carries that
weight close-to-close and books the cost of *changing* it open-to-close. There is no
one-position-at-a-time constraint and no bracket: the position is whatever the weight says,
continuously. This is the standard "vectorised weight × return" accounting used by trend,
carry and vol-targeting books.

### 2. Turnover cost on `|Δpos|`, not a per-trade fee

Cost is charged on the **traded notional**, i.e. the absolute change in weight, every time the
weight moves:

```
cost_t = |pos_t - pos_{t-1}| * (fee_bps_side + slip_bps) / 1e4
```

A weight that drifts from `+0.3` to `-0.3` pays turnover on the full `0.6` it crossed. There is
no notion of "a trade" in the money math — only weight changes — so cost is continuous in
turnover rather than quantised per round-turn. (`n_trades`, the count of sign changes, is
carried purely for the `tr/yr` report and never touches PnL.)

### 3. Optional funding / borrow, and multi-leg combination

Two features serve the carry / delta-neutral world that discrete-trade engines cannot express:

- a caller-injected **`funding_fn`** adds a per-bar funding or borrow effect to any directional
  leg (perp funding `−pos·rate`, a "longs-on-spot" hybrid that funds only the short side, etc.);
- **`carry_leg_returns`** + **`combine_daily`** build a separate delta-neutral funding-harvest
  leg and **weight-combine** several daily legs into one portfolio — so a trend leg and a carry
  leg can be backtested independently and then blended, which a single-position engine has no
  way to represent.

### At a glance

| | Core / futures engines | Portfolio engine (`run_position_backtest`) |
|---|---|---|
| Unit of execution | a **trade** (entry → exit) | a **target weight** held every bar |
| Position | one at a time, bracketed | continuous signed weight, no bracket |
| Return | realised PnL per trade | per-bar `pos·ret` (mark-to-market) |
| Cost | per-trade / notional round-turn | turnover on `|Δpos|`, every bar |
| Funding / borrow | — | optional `funding_fn`, plus a dedicated carry leg |
| Composition | one instrument, compounded | **multi-leg**, weight-combined to a portfolio |
| Data shape | bars + ticks | one **fine** OHLC frame + a **coarse** weight series |

Everything is plain pandas: legs are `pd.Series`, the portfolio is a `pd.Series`, metrics are a
`dict`.

---

## Per-bar return and the signal → fine-bar lag

Signals are decided on **coarse** bars (e.g. daily) and executed on **fine** bars (e.g. hourly
or 1-minute). The coarse weight is mapped onto the fine grid by `map_signal_to_fine` (see the
API below), which encodes a strict "decide at close `t`, act from `t + tf`" rule: the weight decided at the close of coarse
bar `t` is **shifted forward by one coarse timeframe** and then forward-filled onto the fine
index. There is no look-ahead — a weight is never in force during the bar that produced it.

On the fine grid, for each fine bar `t` with open `O_t` and close `C_t`, the **net return** is:

```
ret_cc_t = C_t / C_{t-1} - 1            # close-to-close
ret_oc_t = C_t / O_t     - 1            # open-to-close
dpos_t   = pos_t - pos_{t-1}

net_t = pos_{t-1} * ret_cc_t            # carried part (held since last bar)
      + dpos_t    * ret_oc_t            # freshly added part (executed at this open)
      - |dpos_t|  * (fee_bps_side + slip_bps) / 1e4   # turnover cost
      + funding_t                       # optional, from funding_fn (else 0)
```

The split between `ret_cc` and `ret_oc` is the key detail: the weight you were **already
holding** earns the close-to-close move, while the increment you **add this bar** is assumed
filled at the bar's open and therefore earns only the open-to-close move. Equity is
`cumprod(1 + net_t)`; drawdown is `equity / equity.cummax() − 1`.

---

## API

### `PortfolioConfig` (dataclass)

Cost and annualisation parameters. All fields have defaults.

| Field | Default | Meaning |
|---|---|---|
| `fee_bps_side` | `2.0` | Fee in basis points **per side**, charged on `|Δpos|`. |
| `slip_bps` | `1.0` | Slippage in basis points, added to the fee on `|Δpos|`. |
| `carry_cost_day` | `2e-5` | Daily hedge-maintenance drag of a delta-neutral carry leg, in return units (`2e-5` ≈ 0.7%/yr). |
| `months_per_year` | `12` | Sharpe annualisation factor: monthly Sharpe × `sqrt(months_per_year)`. |
| `days_per_year` | `365.25` | Year length (in days) used for the CAGR exponent. |
| `capital_mode` | `"reinvest"` | Compounding mode; `"reinvest"` is the only supported mode. |

### `run_position_backtest(fine_bars, position, *, signal_tf=None, funding_fn=None, config=None, n_trades=0) -> PositionResult`

Backtest **one** continuously-held signed leg on fine bars.

| Argument | Meaning |
|---|---|
| `fine_bars` | Fine-grained OHLC `pd.DataFrame` with at least `open` and `close` columns; its `DatetimeIndex` is the execution clock (e.g. 1-minute). |
| `position` | Signed target weight on the **coarse** signal grid (e.g. daily), as decided at each coarse bar's close. It is shifted `+1` coarse bar and forward-filled. |
| `signal_tf` | `pd.Timedelta` of the coarse timeframe used for the lag; inferred from `position.index[1] − position.index[0]` when omitted. |
| `funding_fn` | Optional `(fine_index, pos_fine) -> pd.Series` returning the per-bar funding effect. `pos_fine` is the mapped fine-grid weight, so the caller can implement perp funding (`−pos·rate`), a short-only-funded hybrid (`pos.clip(upper=0)`), etc. `None` ⇒ no funding. |
| `config` | A `PortfolioConfig` (defaults to `PortfolioConfig()`). |
| `n_trades` | Round-trip count (sign changes) for the `tr/yr` report only; PnL-neutral. |

Returns a **`PositionResult`** (dataclass):

| Field | Type | Meaning |
|---|---|---|
| `net_ret` | `pd.Series` | Per-fine-bar net return (the formula above). |
| `equity` | `pd.Series` | `cumprod(1 + net_ret)`. |
| `dd` | `pd.Series` | Drawdown fraction, `equity / equity.cummax() − 1`. |
| `daily_returns` | `pd.Series` | `net_ret` compounded to calendar days (via `to_daily`). |
| `metrics` | `dict` | Headline metrics on the fine series: the same keys as `portfolio_metrics` (below) **plus** `return_%` (total equity return). |

### `to_daily(net_ret) -> pd.Series`

Compound a fine per-bar net-return series into **calendar-day** returns:
`(1 + net_ret).groupby(index.normalize()).prod() − 1`. The day key is the bar timestamp
normalised to midnight.

### `carry_leg_returns(funding_daily, leverage, cost_day=2e-5) -> pd.Series`

Daily returns of a **levered, delta-neutral funding-carry** leg, named `"carry"`:

```
carry_daily = leverage * (funding_daily - cost_day)
```

`funding_daily` is the per-day funding a 1× delta-neutral book earns; `leverage` is the
(possibly time-varying or signal-gated) carry leverage. `leverage` is reindexed onto
`funding_daily.index` when their indices differ. `cost_day` is the daily hedge-maintenance drag
(typically `PortfolioConfig.carry_cost_day`).

### `combine_daily(legs, weights, index=None) -> pd.Series`

Weight-combine several daily return legs into one portfolio series named `"combo"`:
`sum_i weights[i] * legs[i].reindex(index).fillna(0.0)`. Each leg is reindexed to `index`
(default: the **first** leg's index) and any missing day is treated as flat. Raises
`ValueError` if `legs` is empty or `len(legs) != len(weights)`.

### `portfolio_metrics(daily_returns, n_trades=0, *, config=None) -> dict`

Headline metrics for a **daily** return series. Definitions:

- **`CAGR_%`** — `equity[-1] ** (1 / years) − 1`, where `years = (last − first).days / days_per_year`.
- **`max_dd_%`** — worst close-to-peak on the daily equity curve, `min(equity / equity.cummax() − 1)`.
- **`sharpe_m`** — calendar-**month** returns: `mean / std(ddof=1) * sqrt(months_per_year)` (`0.0` if `std == 0`).
- **`calmar`** — `CAGR_% / |max_dd_%|`.
- **`pos_months`** — `"<positive months>/<total months>"`.
- **`tr/yr`** — `round(n_trades / years)`.
- **`y<year>_%`** — calendar-year return for each year present.

All values are rounded to match the canonical research reporting (so two dicts compare equal).

### `map_signal_to_fine(position, fine_index, signal_tf=None) -> pd.Series`

The lag primitive used internally by `run_position_backtest`, exposed for direct use. Shifts the
coarse `position` timestamps forward by `signal_tf` (default: the spacing of the first two coarse
bars), forward-fills onto `fine_index`, and `fillna(0.0)` before the first decision. This is the
"decide at close `t`, act from `t + tf`" mapping in one call.

---

## Minimal example

Reuse the bundled GBM generator as the **fine** (hourly) execution frame, derive a **daily**
toy-trend weight, run the directional leg, add a delta-neutral carry leg, and combine the two
into a portfolio. The strategy carries **no edge** — like the other engines' demos, it exists
only to exercise the loop end-to-end. Numbers are fully deterministic with `seed=7`.

```python
import numpy as np
import pandas as pd
from eventbt import (gbm_data, PortfolioConfig, run_position_backtest, to_daily,
                     carry_leg_returns, combine_daily, portfolio_metrics)

# --- Fine execution bars: two years of hourly GBM, as a DataFrame. ---
bars, _ = gbm_data(n_bars=24 * 365 * 2, ticks_per_bar=1, sigma=0.2,
                   seed=7, bar_freq="1h")
fine = pd.DataFrame({"open": bars.open, "close": bars.close},
                    index=pd.DatetimeIndex(bars.time))

# --- Coarse signal grid (daily): toy trend = 0.3 x sign of 5-day momentum. ---
coarse_close = fine["close"].resample("1D").last().dropna()
position = 0.3 * np.sign(coarse_close.pct_change(5)).fillna(0.0)    # +0.3 / 0 / -0.3
n_trades = int((position.diff().fillna(0.0) != 0).sum())

cfg = PortfolioConfig(fee_bps_side=2.0, slip_bps=1.0)

# --- Leg 1: the directional trend leg, on fine bars. ---
trend = run_position_backtest(fine, position, config=cfg, n_trades=n_trades)
trend_daily = trend.daily_returns
print("trend leg metrics:", trend.metrics)

# --- Leg 2: a delta-neutral funding-carry leg, daily. ---
funding_daily = pd.Series(2e-4, index=trend_daily.index)           # ~7.4%/yr at 1x
leverage = pd.Series(3.0, index=trend_daily.index)                 # 3x carry dial
carry_daily = carry_leg_returns(funding_daily, leverage, cost_day=cfg.carry_cost_day)
print("carry leg CAGR_%:", portfolio_metrics(carry_daily)["CAGR_%"])

# --- Combine 50/50 and report the portfolio. ---
combo = combine_daily([trend_daily, carry_daily], [0.5, 0.5])
print("combo metrics:", portfolio_metrics(combo, n_trades=n_trades, config=cfg))
print("to_daily rows:", len(to_daily(trend.net_ret)))
```

Output:

```
trend leg metrics: {'return_%': 9.8, 'CAGR_%': 4.8, 'tr/yr': 71.0, 'max_dd_%': -54.9, 'sharpe_m': 0.31, 'calmar': 0.09, 'pos_months': '12/24', 'y2020_%': -10.9, 'y2021_%': 23.3}
carry leg CAGR_%: 21.8
combo metrics: {'CAGR_%': 14.9, 'tr/yr': 71.0, 'max_dd_%': -26.6, 'sharpe_m': 0.8, 'calmar': 0.56, 'pos_months': '12/24', 'y2020_%': 6.0, 'y2021_%': 24.4}
to_daily rows: 730
```

Reading the numbers:

- **The trend leg** holds a fractional `±0.3` weight, decided daily and executed hourly. On a
  driftless random walk it has no edge (CAGR `4.8%`, a `−54.9%` drawdown) — exactly the
  no-edge behaviour the other engines' demos show.
- **The carry leg** harvests `2e-4`/day of funding at `3×` leverage net of the `2e-5`/day hedge
  cost: `3 × (2e-4 − 2e-5) ≈ 5.4e-4`/day ≈ `21.8%` CAGR, with no exposure to the price path.
- **The 50/50 portfolio** blends them: the steady carry stream lifts CAGR to `14.9%` and, being
  uncorrelated with the trend leg's path, cuts the drawdown to `−26.6%` and roughly doubles the
  Sharpe (`0.31 → 0.8`). This diversification across legs is the whole point of the engine, and
  is something a single-position engine cannot express.

---

## Validated bit-for-bit against an external research stack

The engine is not just internally consistent — it has been validated against an independent
production research backtester on real multi-year market data. Feeding both stacks the
*identical* coarse position and leg inputs, every output matched with `max|Δ| = 0.0`: the fine
per-bar `net_ret` (millions of rows), the daily directional series, the carry leg, the
weight-combined portfolio, its equity curve, and every headline metric. Achieving zero
difference across millions of fine bars is strong evidence that the lag mapping, the `ret_cc`
/ `ret_oc` split, the turnover cost, and the daily / metric aggregation all behave correctly.
The comparison harness, the instrument data, and all strategy-specific code live **outside this
repository** — the engine itself stays a clean, self-contained, strategy-agnostic library.

---

## Limitations

- **Frictionless within a bar.** A weight change is filled at the bar's open with a flat
  basis-point cost; there is no tick-level fill model, partial fills, or queue position (use the
  core or futures engines when intrabar fill order matters).
- **Caller owns funding and leverage.** The engine never reads a funding-rate file or gates a
  signal; `funding_fn`, `funding_daily` and `leverage` are all supplied by the caller.
- **`reinvest` (compounding) only** — there is no fixed-notional mode.
- Not financial advice. The bundled demo strategy carries no edge and exists only to exercise the
  engine.
```