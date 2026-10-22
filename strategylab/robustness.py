"""Robustness tools: parameter sweeps, in-sample / out-of-sample split, walk-forward."""
from __future__ import annotations

import itertools

import pandas as pd

from .engine import Backtest
from .strategy import Strategy

KEEP = ["cagr", "volatility", "sharpe", "max_drawdown", "calmar", "annual_turnover", "excess_cagr"]


def _run(prices, strategy, **bt_kwargs) -> dict:
    r = Backtest(prices, strategy, **bt_kwargs).run()
    return {k: r.metrics.get(k) for k in KEEP}


def parameter_sweep(prices: pd.DataFrame, strategy: Strategy, grid: dict[str, list], **bt_kwargs) -> pd.DataFrame:
    """Run every combination in ``grid`` ({param: [values]}) and return a tidy DataFrame."""
    keys = list(grid)
    rows = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        s = strategy.with_params(**dict(zip(keys, combo)))
        m = _run(prices, s, **bt_kwargs)
        rows.append({**dict(zip(keys, combo)), **m})
    return pd.DataFrame(rows)


def in_vs_out_of_sample(prices: pd.DataFrame, strategy: Strategy, split: str, **bt_kwargs) -> dict:
    """Metrics before and after a fixed calendar split date (no re-optimisation)."""
    kw = dict(bt_kwargs)
    a = Backtest(prices, strategy, **{**kw, "end": split}).run()
    b = Backtest(prices, strategy, **{**kw, "start": split}).run()
    return {
        "split": split,
        "in_sample": {k: a.metrics.get(k) for k in KEEP + ["start", "end"]},
        "out_of_sample": {k: b.metrics.get(k) for k in KEEP + ["start", "end"]},
        "benchmark_in_sample": {k: a.benchmark_metrics.get(k) for k in ["cagr", "sharpe", "max_drawdown"]},
        "benchmark_out_of_sample": {k: b.benchmark_metrics.get(k) for k in ["cagr", "sharpe", "max_drawdown"]},
    }


def walk_forward(prices: pd.DataFrame, strategy: Strategy, grid: dict[str, list], train_years: int = 5,
                 test_years: int = 1, select_by: str = "sharpe", first_test_start: str | None = None,
                 **bt_kwargs) -> pd.DataFrame:
    """Anchored-window walk-forward: pick the best params on the train window, apply to the next test window.

    Returns one row per test window with the chosen params and the out-of-sample metrics,
    plus the same window's metrics for the strategy's *default* params for comparison.
    """
    idx = prices.index
    start = pd.Timestamp(first_test_start) if first_test_start else idx[0] + pd.DateOffset(years=train_years)
    rows = []
    t = start
    keys = list(grid)
    while t + pd.DateOffset(years=test_years) <= idx[-1] + pd.Timedelta(days=1):
        tr_start = t - pd.DateOffset(years=train_years)
        te_end = t + pd.DateOffset(years=test_years) - pd.Timedelta(days=1)
        best, best_params = None, None
        for combo in itertools.product(*[grid[k] for k in keys]):
            p = dict(zip(keys, combo))
            m = Backtest(prices, strategy.with_params(**p), start=str(tr_start.date()), end=str(t.date()), **bt_kwargs).run().metrics
            if best is None or (m[select_by] or -1e9) > best:
                best, best_params = m[select_by], p
        oos = Backtest(prices, strategy.with_params(**best_params), start=str(t.date()), end=str(te_end.date()), **bt_kwargs).run()
        base = Backtest(prices, strategy, start=str(t.date()), end=str(te_end.date()), **bt_kwargs).run()
        rows.append({
            "test_start": str(t.date()), "test_end": str(te_end.date()), "chosen": best_params,
            "train_" + select_by: best,
            "oos_return": oos.metrics["total_return"], "oos_sharpe": oos.metrics["sharpe"], "oos_max_dd": oos.metrics["max_drawdown"],
            "default_return": base.metrics["total_return"], "default_sharpe": base.metrics["sharpe"],
            "benchmark_return": base.benchmark_metrics["total_return"],
        })
        t = t + pd.DateOffset(years=test_years)
    return pd.DataFrame(rows)
