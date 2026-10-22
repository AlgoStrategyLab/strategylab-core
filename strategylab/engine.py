"""Vectorised daily backtest engine.

Execution model (deliberately conservative):
  * Signals are computed on the rebalance date's close.
  * Trades fill at the NEXT trading day's close (one-day lag -> no look-ahead).
  * Costs: ``cost_bps`` per unit of turnover (sum of |weight changes|), charged on the fill day.
  * Between rebalances weights drift with returns (buy-and-hold inside the month).
  * Unallocated weight sits in cash earning ``cash_rate`` (annualised, constant) — set to 0
    for the harshest reading, or include a T-bill ETF in the universe instead.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .strategy import Strategy
from .metrics import compute_metrics, yearly_table, drawdown_series


def rebalance_calendar(index: pd.DatetimeIndex, freq: str = "M") -> pd.DatetimeIndex:
    """Last trading day of each period in ``index`` (M, W, Q)."""
    s = pd.Series(index, index=index)
    code = {"M": "ME", "W": "W-FRI", "Q": "QE"}[freq]
    return pd.DatetimeIndex(s.groupby(pd.Grouper(freq=code)).last().dropna().values)


@dataclass
class BacktestResult:
    strategy: dict
    config: dict
    metrics: dict
    benchmark_metrics: dict
    yearly: list
    equity: pd.Series
    benchmark_equity: pd.Series
    drawdown: pd.Series
    weights: pd.DataFrame
    daily_returns: pd.Series
    turnover: pd.Series
    trades: int

    def to_json_dict(self, include_series: bool = True) -> dict:
        d = {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "strategy": self.strategy,
            "config": self.config,
            "metrics": self.metrics,
            "benchmark_metrics": self.benchmark_metrics,
            "yearly": self.yearly,
            "trades": self.trades,
        }
        if include_series:
            d["series"] = {
                "dates": [str(x.date()) for x in self.equity.index],
                "equity": self.equity.round(6).tolist(),
                "benchmark_equity": self.benchmark_equity.round(6).tolist(),
                "drawdown": self.drawdown.round(6).tolist(),
            }
            # weights only on days they changed, to keep the file small
            w = self.weights
            changed = w[(w.diff().abs().sum(axis=1) > 1e-9) | (w.index == w.index[0])]
            d["allocations"] = [
                {"date": str(i.date()), **{c: round(float(v), 4) for c, v in row.items() if abs(v) > 1e-9}}
                for i, row in changed.iterrows()
            ]
        return d

    def save(self, path: str, include_series: bool = True) -> None:
        with open(path, "w") as f:
            json.dump(self.to_json_dict(include_series), f, indent=2)


class Backtest:
    def __init__(self, prices: pd.DataFrame, strategy: Strategy, benchmark: str = "SPY",
                 rebalance: str = "M", cost_bps: float = 10.0, cash_rate: float = 0.0,
                 start: str | None = None, end: str | None = None, warmup_days: int = 0):
        self.prices = prices.sort_index()
        self.strategy = strategy
        self.benchmark = benchmark
        self.rebalance = rebalance
        self.cost_bps = cost_bps
        self.cash_rate = cash_rate
        self.start = start
        self.end = end
        self.warmup_days = warmup_days

    def run(self) -> BacktestResult:
        px = self.prices
        rets = px.pct_change().fillna(0.0)
        idx = px.index
        # evaluation window (strategy may still read prices before `start` for warmup)
        eval_start = pd.Timestamp(self.start) if self.start else idx[max(self.warmup_days, 0)]
        eval_end = pd.Timestamp(self.end) if self.end else idx[-1]
        rb_dates = rebalance_calendar(idx, self.rebalance)
        rb_dates = rb_dates[(rb_dates >= eval_start) & (rb_dates <= eval_end)]

        targets = self.strategy.target_weights(px, rb_dates).reindex(columns=px.columns).fillna(0.0)
        if (targets.sum(axis=1) > 1.0 + 1e-9).any():
            raise ValueError("target weights exceed 1.0 on some rebalance date")

        # shift execution to the next trading day
        pos = idx.get_indexer(targets.index)
        fill_pos = np.minimum(pos + 1, len(idx) - 1)
        fill_dates = idx[fill_pos]
        targets.index = fill_dates
        targets = targets[~targets.index.duplicated(keep="last")]

        # daily loop over the evaluation window (fast enough: ~6k days x few assets)
        eval_idx = idx[(idx >= eval_start) & (idx <= eval_end)]
        n = len(eval_idx)
        tickers = list(px.columns)
        w = np.zeros(len(tickers))                      # weights at start of day (before returns)
        daily_cash = (1 + self.cash_rate) ** (1 / 252) - 1
        port_ret = np.zeros(n)
        turnover = np.zeros(n)
        w_hist = np.zeros((n, len(tickers)))
        trades = 0
        tgt_map = {d: targets.loc[d].values for d in targets.index}
        r_eval = rets.loc[eval_idx].values

        for i, d in enumerate(eval_idx):
            # drift through the day's returns
            r = r_eval[i]
            gross = float(np.dot(w, r)) + (1 - w.sum()) * daily_cash
            w = w * (1 + r)
            total = 1 + gross
            w = w / total if total != 0 else w
            cost = 0.0
            if d in tgt_map:                            # rebalance at today's close
                new_w = tgt_map[d]
                delta = np.abs(new_w - w).sum()
                if delta > 1e-9:
                    trades += int((np.abs(new_w - w) > 1e-9).sum())
                cost = delta * self.cost_bps / 1e4
                turnover[i] = delta
                w = new_w.copy()
            port_ret[i] = gross - cost
            w_hist[i] = w

        daily = pd.Series(port_ret, index=eval_idx, name="strategy")
        equity = (1 + daily).cumprod()
        bench_daily = rets[self.benchmark].loc[eval_idx]
        bench_equity = (1 + bench_daily).cumprod()
        weights = pd.DataFrame(w_hist, index=eval_idx, columns=tickers)
        turn = pd.Series(turnover, index=eval_idx)

        metrics = compute_metrics(daily, bench_daily)
        metrics["annual_turnover"] = round(float(turn.sum() / (n / 252)), 3)
        metrics["time_in_market"] = round(float((weights.sum(axis=1) > 0.01).mean()), 4)
        metrics["avg_cost_drag_annual"] = round(float((turn.sum() * self.cost_bps / 1e4) / (n / 252)), 5)
        bmetrics = compute_metrics(bench_daily, bench_daily)

        return BacktestResult(
            strategy=self.strategy.describe(),
            config={
                "benchmark": self.benchmark, "rebalance": self.rebalance, "cost_bps": self.cost_bps,
                "cash_rate": self.cash_rate, "start": str(eval_idx[0].date()), "end": str(eval_idx[-1].date()),
                "execution": "signal at close, fill next close", "data": "yfinance adjusted close",
            },
            metrics=metrics, benchmark_metrics=bmetrics,
            yearly=yearly_table(daily, bench_daily), equity=equity, benchmark_equity=bench_equity,
            drawdown=drawdown_series(equity), weights=weights, daily_returns=daily, turnover=turn, trades=trades,
        )
