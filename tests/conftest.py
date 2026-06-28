"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from eventbt import BarSeries, TickSeries, gbm_data


@pytest.fixture
def synth() -> tuple[BarSeries, TickSeries]:
    """A deterministic synthetic dataset with enough volatility to trigger trades."""
    return gbm_data(n_bars=200, ticks_per_bar=30, sigma=0.5, seed=42)
