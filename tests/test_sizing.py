"""Risk-based position sizing: lots = floor(balance*risk_pct/(stop_distance*point_value))."""

from __future__ import annotations

from eventbt import size_position


def test_size_basic() -> None:
    # 10000 * 0.01 / (2.0 * 1.0) = 50
    assert size_position(10_000, 0.01, 2.0, 1.0) == 50


def test_size_floors_fractional_result() -> None:
    # 10000 * 0.01 / (3.0 * 1.0) = 33.33... -> floor -> 33
    assert size_position(10_000, 0.01, 3.0, 1.0) == 33


def test_size_accounts_for_point_value() -> None:
    # 10000 * 0.02 / (5.0 * 10.0) = 200 / 50 = 4
    assert size_position(10_000, 0.02, 5.0, 10.0) == 4


def test_size_respects_max_lots() -> None:
    assert size_position(10_000, 0.01, 2.0, 1.0, max_lots=10) == 10


def test_size_below_one_lot_is_skipped() -> None:
    # 100 * 0.01 / (2.0 * 1.0) = 0.5 -> floor -> 0 (skip the trade)
    assert size_position(100, 0.01, 2.0, 1.0) == 0


def test_size_guards_against_nonpositive_inputs() -> None:
    assert size_position(10_000, 0.01, 0.0, 1.0) == 0  # zero stop distance
    assert size_position(10_000, 0.01, 2.0, 0.0) == 0  # zero point value
    assert size_position(0, 0.01, 2.0, 1.0) == 0  # zero balance
    assert size_position(10_000, -0.01, 2.0, 1.0) == 0  # negative risk
    assert size_position(10_000, 0.01, -2.0, 1.0) == 0  # negative stop distance
