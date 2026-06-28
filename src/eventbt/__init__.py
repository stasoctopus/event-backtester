"""eventbt -- an event-driven, tick-accurate backtesting engine.

Quick start
-----------
>>> from eventbt import gbm_data, SMACrossover, run_backtest, summary_table
>>> bars, ticks = gbm_data(n_bars=300, seed=7)
>>> result = run_backtest(SMACrossover(fast=10, slow=30), bars, ticks)
>>> print(summary_table(result))  # doctest: +SKIP
"""

from __future__ import annotations

from .data import Bar, BarSeries, Tick, TickSeries, gbm_data, load_yfinance
from .engine import (
    Backtester,
    BacktestResult,
    EngineConfig,
    ExitReason,
    Trade,
    run_backtest,
    size_position,
)
from .futures import (
    FuturesConfig,
    FuturesResult,
    FuturesSegment,
    FuturesSignal,
    FuturesStrategy,
    FuturesTrade,
    run_futures_backtest,
)
from .portfolio import (
    PortfolioConfig,
    PositionResult,
    carry_leg_returns,
    combine_daily,
    portfolio_metrics,
    run_position_backtest,
    to_daily,
)
from .metrics import (
    calmar,
    max_drawdown,
    positive_months_pct,
    profit_factor,
    sharpe,
    summary,
    summary_table,
    total_return,
    win_rate,
)
from .strategy import Direction, Signal, SMACrossover, Strategy
from .walkforward import (
    WalkForwardResult,
    WFWindow,
    generate_windows,
    train_test_split,
    walk_forward,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # data
    "Bar",
    "Tick",
    "BarSeries",
    "TickSeries",
    "gbm_data",
    "load_yfinance",
    # strategy
    "Direction",
    "Signal",
    "Strategy",
    "SMACrossover",
    # engine
    "EngineConfig",
    "ExitReason",
    "Trade",
    "BacktestResult",
    "Backtester",
    "run_backtest",
    "size_position",
    # futures execution (opt-in limit-order / session-aware engine)
    "FuturesConfig",
    "FuturesSignal",
    "FuturesStrategy",
    "FuturesTrade",
    "FuturesSegment",
    "FuturesResult",
    "run_futures_backtest",
    # portfolio / target-weight execution (opt-in multi-leg weight engine)
    "PortfolioConfig",
    "PositionResult",
    "run_position_backtest",
    "to_daily",
    "carry_leg_returns",
    "combine_daily",
    "portfolio_metrics",
    # metrics
    "total_return",
    "win_rate",
    "profit_factor",
    "max_drawdown",
    "sharpe",
    "calmar",
    "positive_months_pct",
    "summary",
    "summary_table",
    # walk-forward
    "WFWindow",
    "WalkForwardResult",
    "generate_windows",
    "train_test_split",
    "walk_forward",
]
