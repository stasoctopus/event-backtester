"""Target-weight / multi-leg portfolio engine: lag mapping, return decomposition,
costs, funding, daily aggregation, leg combination and headline metrics.

Every fixture here is hand-built and deterministic. Where possible a tiny 3-4 bar
OHLC series with round numbers is used so the documented per-bar accounting

    r_t = pos_{t-1} * (C_t/C_{t-1} - 1) + (pos_t - pos_{t-1}) * (C_t/O_t - 1)
        - |pos_t - pos_{t-1}| * cost + funding_t

can be checked by eye. The engine is treated as ground truth: these tests pin the
documented semantics, they do not change behaviour. Only synthetic numpy/pandas
data and eventbt's own :func:`gbm_data` are used -- nothing from the trading bot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eventbt import (
    PortfolioConfig,
    PositionResult,
    carry_leg_returns,
    combine_daily,
    gbm_data,
    portfolio_metrics,
    run_position_backtest,
    to_daily,
)
from eventbt.portfolio import map_signal_to_fine

ONE_MIN = pd.Timedelta(minutes=1)


def _bars(index: pd.DatetimeIndex, open_, close) -> pd.DataFrame:
    """Minimal OHLC frame on ``index`` (high/low unused by the engine)."""
    open_ = np.asarray(open_, dtype=float)
    close = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close),
            "low": np.minimum(open_, close),
            "close": close,
            "volume": np.zeros(len(index)),
        },
        index=index,
    )


def _flat_bars(index: pd.DatetimeIndex, price: float = 100.0) -> pd.DataFrame:
    """A flat market: open == close == ``price`` on every bar (zero ret_cc/ret_oc)."""
    px = np.full(len(index), float(price))
    return _bars(index, px, px)


def _pos_on_fine(values, fine_index: pd.DatetimeIndex) -> pd.Series:
    """Build a coarse position whose +1-bar lag lands it *exactly* on ``fine_index``.

    Placing the decision timestamps at ``fine_index - signal_tf`` makes
    :func:`map_signal_to_fine` reproduce ``values`` verbatim on the fine grid, so a
    test can dictate the realised ``pos_fine`` and hand-check the return algebra.
    """
    return pd.Series(np.asarray(values, dtype=float), index=fine_index - ONE_MIN)


# ---------------------------------------------------------------------------
# signal -> fine-bar mapping: decide at close t, act from t + tf, forward-fill
# ---------------------------------------------------------------------------
def test_map_signal_to_fine_lags_one_coarse_bar() -> None:
    # A 1h signal decided at each coarse close only becomes effective +1h later and
    # is then held (forward-filled) on the 1m grid until the next decision.
    coarse = pd.date_range("2021-03-01 00:00", periods=4, freq="1h")
    position = pd.Series([1.0, -1.0, 0.5, 2.0], index=coarse)
    fine = pd.date_range("2021-03-01 00:00", "2021-03-01 04:00", freq="1min")

    pos_fine = map_signal_to_fine(position, fine)  # signal_tf inferred as 1h

    assert pos_fine.index.equals(fine)
    # Nothing is effective before the first close + 1h; the lag is exact.
    assert pos_fine.loc["2021-03-01 00:00"] == 0.0
    assert pos_fine.loc["2021-03-01 00:59"] == 0.0
    assert pos_fine.loc["2021-03-01 01:00"] == 1.0  # first decision, +1h exactly
    assert pos_fine.loc["2021-03-01 01:59"] == 1.0  # held until the next bar
    assert pos_fine.loc["2021-03-01 02:00"] == -1.0
    assert pos_fine.loc["2021-03-01 02:59"] == -1.0
    assert pos_fine.loc["2021-03-01 03:00"] == 0.5
    assert pos_fine.loc["2021-03-01 04:00"] == 2.0
    # The first non-zero fine bar is precisely one coarse timeframe after close[0].
    first_live = pos_fine[pos_fine != 0.0].index[0]
    assert first_live == coarse[0] + pd.Timedelta(hours=1)


def test_map_signal_to_fine_respects_explicit_signal_tf() -> None:
    # An explicit 2h timeframe shifts the same decisions two hours forward instead.
    coarse = pd.date_range("2021-03-01 00:00", periods=4, freq="1h")
    position = pd.Series([1.0, -1.0, 0.5, 2.0], index=coarse)
    fine = pd.date_range("2021-03-01 00:00", "2021-03-01 04:00", freq="1min")

    pos_fine = map_signal_to_fine(position, fine, signal_tf=pd.Timedelta(hours=2))

    assert pos_fine.loc["2021-03-01 01:59"] == 0.0
    assert pos_fine.loc["2021-03-01 02:00"] == 1.0  # now effective at +2h
    assert pos_fine.loc["2021-03-01 03:00"] == -1.0
    assert pos_fine.loc["2021-03-01 04:00"] == 0.5


# ---------------------------------------------------------------------------
# per-bar return decomposition: pos_prev*ret_cc + dpos*ret_oc - turnover
# ---------------------------------------------------------------------------
def test_return_decomposition_on_tiny_known_series() -> None:
    # pos_fine = [0, 1, 2, 0]; pos_prev = [0, 0, 1, 2]; dpos = [0, 1, 1, -2].
    # OHLC chosen so every ret_cc == ret_oc == 0.10 on the moving bars.
    fine = pd.date_range("2021-01-01 00:00", periods=4, freq="1min")
    bars = _bars(fine, open_=[100, 100, 110, 121], close=[100, 110, 121, 121])
    position = _pos_on_fine([0.0, 1.0, 2.0, 0.0], fine)

    res = run_position_backtest(bars, position, signal_tf=ONE_MIN)

    assert isinstance(res, PositionResult)
    # bar1: 0*0.10 + 1*0.10 - 1*0.0003 ; bar2: 1*0.10 + 1*0.10 - 1*0.0003 ;
    # bar3: 2*0.0   + (-2)*0.0  - 2*0.0003  (default cost = (2+1)bps = 3e-4).
    expected = np.array([0.0, 0.0997, 0.1997, -0.0006])
    assert res.net_ret.values == pytest.approx(expected)
    # equity is the running compound of (1 + net_ret); dd is non-positive.
    assert res.equity.values == pytest.approx(np.cumprod(1.0 + expected))
    assert (res.dd.values <= 1e-12).all()


def test_carried_and_fresh_legs_use_different_prices() -> None:
    # A position already on the books before this bar earns the close-to-close move,
    # while a position opened this bar only earns open-to-close. Make the two moves
    # differ (gap up: prev close 100, open 105, close 110) and hold pos flat at 1.
    fine = pd.date_range("2021-01-01 00:00", periods=2, freq="1min")
    bars = _bars(fine, open_=[100, 105], close=[100, 110])
    position = _pos_on_fine([1.0, 1.0], fine)  # pos_prev[1] = 1, dpos[1] = 0

    res = run_position_backtest(bars, position, signal_tf=ONE_MIN)

    # bar1 carries the full close-to-close 110/100-1 = 0.10 (no fresh leg, no cost).
    assert res.net_ret.iloc[1] == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# turnover cost: |dpos| * (fee_bps_side + slip_bps) / 1e4
# ---------------------------------------------------------------------------
def test_turnover_cost_is_exact_on_a_known_step() -> None:
    # Flat market => strat_ret == 0, so net_ret is pure cost. A single +0.5 step at
    # bar1 with fee 5bps + slip 3bps = 8bps costs 0.5 * 8 / 1e4 = 4e-4.
    fine = pd.date_range("2021-01-01 00:00", periods=3, freq="1min")
    bars = _flat_bars(fine, 100.0)
    position = _pos_on_fine([0.0, 0.5, 0.5], fine)
    cfg = PortfolioConfig(fee_bps_side=5.0, slip_bps=3.0)

    res = run_position_backtest(bars, position, signal_tf=ONE_MIN, config=cfg)

    assert res.net_ret.iloc[0] == 0.0  # no trade, no move
    assert res.net_ret.iloc[1] == pytest.approx(-0.5 * (5.0 + 3.0) / 1e4)
    assert res.net_ret.iloc[2] == 0.0  # pos unchanged -> no further cost


# ---------------------------------------------------------------------------
# funding callback: applied additively; a short-only clip charges only shorts
# ---------------------------------------------------------------------------
def test_funding_is_additive_and_clip_charges_only_shorts() -> None:
    # Flat market with costs switched off, so net_ret is exactly the funding leg.
    fine = pd.date_range("2021-01-01 00:00", periods=4, freq="1min")
    bars = _flat_bars(fine, 100.0)
    position = _pos_on_fine([1.0, 1.0, -1.0, -1.0], fine)  # long, long, short, short
    cfg = PortfolioConfig(fee_bps_side=0.0, slip_bps=0.0)

    base = run_position_backtest(bars, position, signal_tf=ONE_MIN, config=cfg)
    assert base.net_ret.values == pytest.approx(np.zeros(4))

    rate = 0.001
    # "longs-on-spot" hybrid: funding (here a credit) only touches the short side.
    funding_fn = lambda fine_index, pos_fine: -rate * pos_fine.clip(upper=0)
    res = run_position_backtest(
        bars, position, signal_tf=ONE_MIN, funding_fn=funding_fn, config=cfg
    )

    expected = np.array([0.0, 0.0, rate, rate])  # clip(upper=0) zeroes the longs
    assert res.net_ret.values == pytest.approx(expected)
    # Additivity: the funded run differs from the bare run by exactly the funding leg.
    assert (res.net_ret - base.net_ret).values == pytest.approx(expected)


def test_funding_constant_is_added_to_every_bar() -> None:
    # A flat per-bar funding credit lands on top of the (here zero) strategy return.
    fine = pd.date_range("2021-01-01 00:00", periods=3, freq="1min")
    bars = _flat_bars(fine, 100.0)
    position = _pos_on_fine([0.0, 0.0, 0.0], fine)
    const = 2.5e-4
    funding_fn = lambda fine_index, pos_fine: pd.Series(const, index=fine_index)

    res = run_position_backtest(bars, position, signal_tf=ONE_MIN, funding_fn=funding_fn)

    assert res.net_ret.values == pytest.approx(np.full(3, const))


# ---------------------------------------------------------------------------
# daily aggregation: compound within a calendar day
# ---------------------------------------------------------------------------
def test_to_daily_compounds_within_each_calendar_day() -> None:
    # Two bars per day; the day return is the product of (1 + bar) minus one.
    idx = pd.to_datetime(
        [
            "2021-01-01 09:00",
            "2021-01-01 15:00",
            "2021-01-02 09:00",
            "2021-01-02 15:00",
        ]
    )
    net = pd.Series([0.10, 0.20, -0.10, 0.05], index=idx)

    daily = to_daily(net)

    assert list(daily.index) == [pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-02")]
    assert daily.iloc[0] == pytest.approx(1.10 * 1.20 - 1.0)  # 0.32
    assert daily.iloc[1] == pytest.approx(0.90 * 1.05 - 1.0)  # -0.055


# ---------------------------------------------------------------------------
# carry leg: leverage * (funding_daily - cost_day), elementwise
# ---------------------------------------------------------------------------
def test_carry_leg_returns_is_levered_net_funding() -> None:
    days = pd.date_range("2021-01-01", periods=3, freq="D")
    funding = pd.Series([0.0010, 0.0020, -0.0005], index=days)
    lev = pd.Series([2.0, 3.0, 1.0], index=days)  # same index -> the .equals branch

    leg = carry_leg_returns(funding, lev, cost_day=2e-5)

    assert leg.name == "carry"
    assert leg.values == pytest.approx((lev * (funding - 2e-5)).values)


def test_carry_leg_reindexes_leverage_to_funding_index() -> None:
    # Leverage supplied on a longer/different index is reindexed onto funding's days.
    days = pd.date_range("2021-01-01", periods=3, freq="D")
    funding = pd.Series([0.001, 0.001, 0.001], index=days)
    lev = pd.Series([2.0, 2.0, 2.0, 9.9], index=pd.date_range("2021-01-01", periods=4, freq="D"))

    leg = carry_leg_returns(funding, lev, cost_day=0.0)

    assert leg.index.equals(funding.index)  # the stray 4th leverage day is dropped
    assert leg.values == pytest.approx(np.full(3, 2.0 * 0.001))


# ---------------------------------------------------------------------------
# leg combination: reindex to a common grid, fill gaps with 0, weighted sum
# ---------------------------------------------------------------------------
def test_combine_daily_reindexes_fills_and_weights() -> None:
    days = pd.date_range("2021-01-01", periods=4, freq="D")
    leg1 = pd.Series([0.1, 0.2, 0.3], index=days[:3])  # d1, d2, d3
    leg2 = pd.Series([1.0, 1.0, 1.0], index=days[1:4])  # d2, d3, d4

    # Default index = first leg's index (d1..d3); leg2's missing d1 fills to 0.
    out = combine_daily([leg1, leg2], [1.0, 0.5])
    assert out.name == "combo"
    assert out.index.equals(days[:3])
    assert out.values == pytest.approx([0.1, 0.2 + 0.5, 0.3 + 0.5])  # [0.1, 0.7, 0.8]

    # Explicit union index (d1..d4): each leg's absent days are treated as flat (0).
    out_union = combine_daily([leg1, leg2], [1.0, 0.5], index=days)
    assert out_union.values == pytest.approx([0.1, 0.7, 0.8, 0.0 + 0.5])


def test_combine_daily_validates_inputs() -> None:
    leg = pd.Series([0.1], index=pd.date_range("2021-01-01", periods=1, freq="D"))
    with pytest.raises(ValueError):
        combine_daily([], [])
    with pytest.raises(ValueError):
        combine_daily([leg], [1.0, 0.5])


# ---------------------------------------------------------------------------
# headline metrics: CAGR / MaxDD / Sharpe / Calmar on a constructed series
# ---------------------------------------------------------------------------
def test_portfolio_metrics_known_answer() -> None:
    # One non-zero return on the 1st of three months => monthly returns are exactly
    # [+0.20, -0.10, +0.10] and the rest of the equity path is flat between jumps.
    idx = pd.date_range("2021-01-01", "2021-03-31", freq="D")
    daily = pd.Series(0.0, index=idx)
    daily.loc["2021-01-01"] = 0.20
    daily.loc["2021-02-01"] = -0.10
    daily.loc["2021-03-01"] = 0.10

    m = portfolio_metrics(daily)

    # Equity: 1 -> 1.20 -> 1.08 -> 1.188; peak 1.20 with a -10% trough in February.
    assert m["max_dd_%"] == -10.0
    assert m["pos_months"] == "2/3"  # Jan & Mar up, Feb down
    assert m["y2021_%"] == 18.8  # full-year compound = 1.188 - 1
    assert m["tr/yr"] == 0.0  # n_trades defaulted to 0
    # CAGR annualises 1.188 over (Mar 31 - Jan 1) = 89 days; Sharpe & Calmar follow.
    assert m["CAGR_%"] == pytest.approx(102.8, abs=0.05)
    assert m["sharpe_m"] == pytest.approx(1.51, abs=0.01)
    assert m["calmar"] == pytest.approx(10.28, abs=0.01)


def test_portfolio_metrics_all_zero_series_is_degenerate_but_safe() -> None:
    # A flat (all-zero) book must not raise and reports zeroed headline numbers.
    idx = pd.date_range("2021-01-01", "2021-03-31", freq="D")
    m = portfolio_metrics(pd.Series(0.0, index=idx))

    assert m["CAGR_%"] == 0.0
    assert m["max_dd_%"] == 0.0
    assert m["sharpe_m"] == 0.0  # zero variance -> Sharpe falls back to 0, no divide
    assert m["y2021_%"] == 0.0
    assert np.isinf(m["calmar"])  # zero drawdown -> Calmar is +inf by definition


# ---------------------------------------------------------------------------
# seeded end-to-end sanity on GBM bars
# ---------------------------------------------------------------------------
def test_gbm_position_backtest_is_well_formed() -> None:
    # A simple one-flip position (long first half, short second half) run on seeded
    # GBM fine bars. The GBM "year" is bar-count based while metrics annualise off the
    # 1m calendar span, so magnitudes are not meaningful -- we only pin finiteness
    # and the structural invariants of the result.
    bars, _ticks = gbm_data(n_bars=20_000, ticks_per_bar=3, mu=0.0, sigma=0.05, seed=7)
    fine = bars.to_frame()
    coarse = fine["close"].resample("30min").last()
    half = len(coarse) // 2
    position = pd.Series(
        np.where(np.arange(len(coarse)) < half, 1.0, -1.0), index=coarse.index
    )

    res = run_position_backtest(fine, position, n_trades=1)

    assert isinstance(res, PositionResult)
    assert len(res.net_ret) == len(fine)
    assert res.equity.index.equals(fine.index)
    assert np.isfinite(res.net_ret.values).all()
    assert (np.isfinite(res.equity.values) & (res.equity.values > 0)).all()
    assert (res.dd.values <= 1e-12).all()
    assert len(res.daily_returns) > 0
    for key in ("return_%", "CAGR_%", "max_dd_%", "sharpe_m", "calmar"):
        assert np.isfinite(res.metrics[key])
