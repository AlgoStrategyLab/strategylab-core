"""strategylab — a small, transparent backtest harness for the Strategy Lab article series.

Design goals
------------
* Daily bars, long-only or long/flat portfolios of ETFs/equities (extensible).
* A Strategy returns *target weights* on each rebalance date; the engine does the rest
  (lagged execution, costs, metrics, robustness, JSON export).
* One JSON schema per run so articles and the platform read the same artifact.
"""
from .data import load_prices
from .engine import Backtest, BacktestResult
from .metrics import compute_metrics, yearly_table, drawdown_series
from .robustness import parameter_sweep, walk_forward, in_vs_out_of_sample
from .strategy import Strategy

__all__ = [
    "load_prices", "Backtest", "BacktestResult", "compute_metrics", "yearly_table",
    "drawdown_series", "parameter_sweep", "walk_forward", "in_vs_out_of_sample", "Strategy",
]
