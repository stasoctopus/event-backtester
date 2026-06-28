<!-- Language switcher -->
**English** | [Русский](README.ru.md)

# event-backtester (`eventbt`)

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Lint: ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![Types: mypy](https://img.shields.io/badge/types-mypy-blue.svg)](https://mypy-lang.org/)
[![CI](https://github.com/stasoctopus/event-backtester/actions/workflows/ci.yml/badge.svg)](https://github.com/stasoctopus/event-backtester/actions/workflows/ci.yml)

An **event-driven, tick-accurate backtesting engine** in Python. Strategy signals are
generated on a coarse bar series, while order fills and the protective bracket are
resolved on a finer tick series — so intrabar exits reflect the *actual* order in
which prices were touched, rather than an optimistic bar-only guess.

> Inspired by a production trading system I run on a VPS.

![Equity curve](docs/equity_curve.png)

*Equity curve of the demo SMA-crossover strategy on synthetic data (reproducible with
`seed=7`). The demo strategy carries **no edge** — it exists to exercise the engine.*

---

## Why this design?

A backtest is only as trustworthy as its execution model. Three decisions drive the
design of this engine; each one removes a common way that backtests lie.

### 1. Tick-accurate fills (avoiding intrabar fill bias)

When both a stop-loss and a take-profit fall inside a single bar's high–low range, a
bar-only backtester **cannot know which was hit first**. Most libraries silently
resolve the ambiguity in the strategy's favour, which inflates results.

This engine replays the **actual tick order** within the bar: whichever level the
price reaches first closes the position and cancels the other (a true one-cancels-other
bracket). The unit test [`tests/test_fills.py`](tests/test_fills.py) pins this down with
a controlled scenario:

| | Stop @ 99 | Take @ 102 | Result |
|---|---|---|---|
| Tick order `100 → 99.5 → **99** → 100.5 → 102` | reached **first** | reached later | **−1.00 / lot (loss)** |
| Naive bar-only rule (both in range → assume take) | ignored | assumed first | **+2.00 / lot (win)** |

Same bar, opposite conclusions. The engine refuses to book the phantom win.

The bundled demo shows the same effect on a full run — bar-only execution looks rosier
across the board (see the comparison below).

### 2. Risk-based position sizing

Position size is derived from the account's risk budget, not a fixed lot count:

```
lots = floor(balance * risk_pct / (stop_distance * point_value))
```

So every trade risks roughly the same fraction of equity regardless of how wide its
stop is. The lot count is clamped to `[0, max_lots]`; a trade whose risk budget cannot
afford a single lot is skipped rather than forced.

### 3. Methodology against overfitting

The library ships the tools to validate a strategy honestly, not just to fit one:

- **In-sample / out-of-sample split** — `train_test_split` for a blind hold-out.
- **Rolling walk-forward** — `walk_forward` optimizes parameters on each in-sample
  window and evaluates the winner *only* on the immediately following, never-seen
  out-of-sample window, then stitches the OOS segments into one continuous equity curve.

### What I tried and discarded (an honest negative result)

I evaluated a machine-learning signal filter (gradient boosting on engineered features)
to gate entries. Its **out-of-sample AUC was ≈ 0.53** — statistically indistinguishable
from a coin flip — so it was rejected. Reporting the negative result is the point: a
filter that doesn't generalize has no business in a backtest, however good it looks
in-sample.

---

## Demo results

Running `python examples/run_demo.py` (synthetic GBM data, `seed=7`, ~3 years of daily
bars, demo SMA crossover) prints two metric tables — the tick-accurate run and a
bar-only approximation that can only see one price per bar:

| Metric | Tick-accurate | Bar-only (close-only) |
|---|---:|---:|
| Total Return % | **−8.44** | −7.32 |
| Win Rate % | 32.35 | 33.33 |
| Profit Factor | 0.69 | 0.72 |
| Max Drawdown % | −10.23 | −11.57 |
| Sharpe | **−0.61** | −0.45 |
| Calmar | −0.29 | −0.22 |
| Positive Months % | 29.17 | 29.17 |
| Trades | 34 | 33 |

The bar-only column reports a higher return, a better Sharpe, and a better profit
factor — the systematic optimism this engine is built to avoid. (The strategy loses
money either way; it has no edge, and that is fine — the subject here is the engine.)

---

## Install

```bash
git clone https://github.com/stasoctopus/event-backtester.git
cd event-backtester
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,plot]"      # core + tests/lint/types + plotting
```

`numpy` and `pandas` are the only runtime dependencies. `matplotlib` (plotting) and
`yfinance` (the optional public-data loader) are extras: `pip install -e ".[plot,data]"`.

## Quickstart

```python
from eventbt import gbm_data, SMACrossover, EngineConfig, run_backtest, summary_table

# 1. Synthetic, reproducible market: GBM ticks aggregated into consistent OHLC bars.
bars, ticks = gbm_data(n_bars=750, ticks_per_bar=60, sigma=0.25, seed=7, bar_freq="1D")

# 2. A strategy returns Signals (direction + stop/take distances); the engine sizes them.
strategy = SMACrossover(fast=10, slow=40, stop_distance=0.5, take_distance=1.0)

# 3. Run, with realistic costs.
config = EngineConfig(initial_balance=10_000, risk_pct=0.01, point_value=1.0,
                      spread=0.02, commission=0.05)
result = run_backtest(strategy, bars, ticks, config)

print(summary_table(result))
print(result.trades_frame().head())
```

Run the full demo (prints tables, saves `docs/equity_curve.png`):

```bash
python examples/run_demo.py
```

## Walk-forward

```python
from eventbt import gbm_data, SMACrossover, walk_forward, total_return

bars, ticks = gbm_data(n_bars=2000, ticks_per_bar=30, seed=1)

wf = walk_forward(
    bars, ticks,
    strategy_factory=lambda fast, slow: SMACrossover(fast, slow, 0.5, 1.0),
    param_grid={"fast": [5, 10], "slow": [20, 40]},
    objective=lambda res: total_return(res.equity_curve),  # maximized in-sample
    train_size=400, test_size=100, step=100,
)
print(wf.best_params)                 # parameters chosen per window
print(len(wf.stitched_equity))        # continuous out-of-sample equity curve
```

---

## Architecture

```
                signals (coarse bars)          fills + bracket (fine ticks)
 Strategy  ───────────────────────────►  Backtester  ◄───────────────────────────  ticks
 (on_bar → Signal)                          │
                                            ├─ size_position()  risk-based lots
                                            ├─ OCO bracket      tick-order exits
                                            ├─ costs            spread + commission
                                            └─ mark-to-market   equity curve
                                            │
                                            ▼
                                       BacktestResult ──►  metrics / walk_forward
```

| Module | Responsibility |
|---|---|
| `eventbt.data` | `Bar`/`Tick`, columnar `BarSeries`/`TickSeries`, `gbm_data` generator, `load_yfinance` |
| `eventbt.strategy` | `Strategy` ABC, `Signal`, demo `SMACrossover` |
| `eventbt.engine` | `Backtester`, `EngineConfig`, `size_position`, OCO fills, `Trade`, `BacktestResult` |
| `eventbt.metrics` | return, win rate, profit factor, max drawdown, Sharpe, Calmar, % positive months |
| `eventbt.walkforward` | `generate_windows`, `train_test_split`, `walk_forward` |

**Per-bar order of operations:** ① fill a pending entry at the *first tick of the next
bar* (filling at the signal bar's close would be look-ahead); ② scan the bar's ticks in
order and resolve the OCO bracket; ③ mark-to-market at the bar close; ④ request a new
signal only while flat. The strategy receives a *truncated view* of the bars, so
look-ahead bias is impossible by construction. Ticks are grouped to bars once with
`np.searchsorted` (O(n log n)).

Scope choice: one position at a time. This keeps the accounting auditable and is
documented rather than hidden.

## Metrics

All metrics are pure functions; the annualization factor (`periods_per_year`, default
252) is always explicit, never inferred from timestamps, so results are reproducible.
A zero risk-free rate is assumed.

- **Total Return** `equity[-1] / equity[0] − 1`
- **Win Rate** winning trades / all trades
- **Profit Factor** gross profit / gross loss (`inf` if no losers)
- **Max Drawdown** worst peak-to-trough on the equity curve
- **Sharpe** `mean(r) / std(r, ddof=1) * sqrt(periods_per_year)`
- **Calmar** annualized CAGR / |max drawdown|
- **Positive Months %** share of calendar months with a positive return (`ME` resample)

## Testing

```bash
pytest                       # 41 tests; equity/sizing/fills/OCO/metrics/data/walk-forward
ruff check . && ruff format --check .
mypy src
```

The suite uses hand-computed known-input vectors (e.g. Sharpe of returns
`[0.01, 0.02, 0.01, 0.02]` ≈ 41.25, exercising the annualization and `ddof`) and the
controlled tick-accuracy scenario above. CI runs the same checks on Python 3.10–3.13.

## Limitations

- Synthetic demo data only; the `yfinance` loader returns bars, so a finer tick series
  must be supplied for tick-accurate execution on real data.
- One open position at a time; no portfolio of simultaneous positions.
- Not financial advice. The included strategy is a demo with no edge.

## License

[MIT](LICENSE)
