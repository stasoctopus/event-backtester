"""Smoke tests for the public surface used in the README and examples."""

from __future__ import annotations

import pandas as pd
import pytest

from eventbt import (
    BarSeries,
    EngineConfig,
    SMACrossover,
    TickSeries,
    run_backtest,
    summary,
    summary_table,
)

SUMMARY_KEYS = {
    "total_return_pct",
    "win_rate_pct",
    "profit_factor",
    "max_drawdown_pct",
    "sharpe",
    "calmar",
    "positive_months_pct",
    "num_trades",
    "final_balance",
}


def test_summary_and_table(synth: tuple[BarSeries, TickSeries]) -> None:
    bars, ticks = synth
    result = run_backtest(SMACrossover(5, 20, 0.5, 1.0), bars, ticks, EngineConfig())
    data = summary(result)
    assert SUMMARY_KEYS.issubset(data.keys())

    table = summary_table(result)
    assert "Total Return %" in table
    assert "Sharpe" in table


def test_equity_curve_has_one_point_per_bar(synth: tuple[BarSeries, TickSeries]) -> None:
    bars, ticks = synth
    result = run_backtest(SMACrossover(5, 20, 0.5, 1.0), bars, ticks)
    assert len(result.equity_curve) == len(bars)


def test_barseries_from_frame_roundtrip() -> None:
    idx = pd.date_range("2021-01-01", periods=4, freq="D")
    df = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0, 4.0],
            "high": [2.0, 3.0, 4.0, 5.0],
            "low": [0.5, 1.5, 2.5, 3.5],
            "close": [1.5, 2.5, 3.5, 4.5],
            "volume": [10.0, 20.0, 30.0, 40.0],
        },
        index=idx,
    )
    bars = BarSeries.from_frame(df)
    assert len(bars) == 4
    out = bars.to_frame()
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert out["close"].tolist() == [1.5, 2.5, 3.5, 4.5]


def test_from_frame_missing_columns_raises() -> None:
    df = pd.DataFrame(
        {"open": [1.0], "high": [2.0]},
        index=pd.date_range("2021-01-01", periods=1, freq="D"),
    )
    with pytest.raises(ValueError, match="missing required column"):
        BarSeries.from_frame(df)


def test_empty_backtest_is_a_no_op() -> None:
    bars = BarSeries.from_bars([])
    ticks = TickSeries.from_ticks([])
    result = run_backtest(SMACrossover(), bars, ticks)
    assert result.trades == []
    assert result.final_balance == result.initial_balance
    assert len(result.equity_curve) == 0
