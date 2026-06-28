"""End-to-end demo for eventbt.

Generates a synthetic market, runs the demo SMA-crossover strategy through the
tick-accurate engine, prints a metrics table, and saves an equity-curve plot to
``docs/equity_curve.png``.

It also runs a *bar-only* approximation (execution resolved on bar closes alone)
to illustrate that execution granularity changes the result -- the engine's headline
point. The rigorous, sign-controlled proof of intrabar fill bias lives in
``tests/test_fills.py``.

Run with::

    python examples/run_demo.py
"""

from __future__ import annotations

from pathlib import Path

from eventbt import (
    EngineConfig,
    SMACrossover,
    TickSeries,
    gbm_data,
    run_backtest,
    summary,
    summary_table,
)

SEED = 7
N_BARS = 750  # ~3 years of daily bars
TICKS_PER_BAR = 60
BAR_FREQ = "1D"
SIGMA = 0.25  # annualized volatility
FAST, SLOW = 10, 40
STOP_DISTANCE, TAKE_DISTANCE = 0.5, 1.0
CONFIG = EngineConfig(
    initial_balance=10_000.0,
    risk_pct=0.01,
    point_value=1.0,
    spread=0.02,
    commission=0.05,
)


def _plot_equity(equity, out_path: Path) -> None:
    """Save the equity curve to ``out_path`` using a headless backend."""
    import matplotlib

    matplotlib.use("Agg")  # headless / reproducible -- must precede pyplot import
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(equity.index, equity.values, color="#1f77b4", linewidth=1.3)
    ax.set_title("eventbt -- SMA crossover on synthetic data (tick-accurate)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    bars, ticks = gbm_data(
        n_bars=N_BARS,
        ticks_per_bar=TICKS_PER_BAR,
        sigma=SIGMA,
        seed=SEED,
        bar_freq=BAR_FREQ,
    )
    strategy = SMACrossover(
        fast=FAST, slow=SLOW, stop_distance=STOP_DISTANCE, take_distance=TAKE_DISTANCE
    )

    # 1) Tick-accurate run (fills resolved on the fine tick series).
    result = run_backtest(strategy, bars, ticks, CONFIG)

    # 2) Bar-only approximation: execution sees only one price per bar (the close),
    #    so it cannot honor intrabar bracket order. This is a deliberate simplification
    #    for contrast -- it stamps each bar's close at the bar's timestamp, so entries
    #    effectively fill at the next bar's close. The rigorous proof of the bias is the
    #    controlled unit test in tests/test_fills.py.
    coarse_ticks = TickSeries(bars.time, bars.close)
    bar_only = run_backtest(
        SMACrossover(
            fast=FAST, slow=SLOW, stop_distance=STOP_DISTANCE, take_distance=TAKE_DISTANCE
        ),
        bars,
        coarse_ticks,
        CONFIG,
    )

    print("=" * 48)
    print("Tick-accurate execution")
    print("=" * 48)
    print(summary_table(result))

    print("\n" + "=" * 48)
    print("Bar-only execution (close-only, for contrast)")
    print("=" * 48)
    print(summary_table(bar_only))

    tick_ret = summary(result)["total_return_pct"]
    bar_ret = summary(bar_only)["total_return_pct"]
    print(
        f"\nReturn: tick-accurate {tick_ret:,.2f}%  vs  bar-only {bar_ret:,.2f}%  "
        f"(delta {bar_ret - tick_ret:+,.2f} pp). Execution granularity matters."
    )

    out_path = Path(__file__).resolve().parent.parent / "docs" / "equity_curve.png"
    _plot_equity(result.equity_curve, out_path)
    print(f"\nSaved equity curve to: {out_path}")


if __name__ == "__main__":
    main()
