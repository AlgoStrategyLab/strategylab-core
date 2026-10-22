# strategylab-core

The shared backtest engine behind every strategy in the [AlgoStrategyLab](https://github.com/AlgoStrategyLab)
organization. Each strategy lives in its own repository and installs this package.

```bash
pip install git+https://github.com/AlgoStrategyLab/strategylab-core.git
```

## What it does

- **Data** — `load_prices(tickers, start)`: Yahoo Finance adjusted closes (total return), cached as parquet in `./data/`.
- **Strategy interface** — subclass `Strategy`, implement `target_weights(prices, rebalance_dates)` returning target
  weights per rebalance date (rows sum to ≤ 1.0; the remainder is cash).
- **Engine** — `Backtest(prices, strategy, benchmark, rebalance="M", cost_bps=10).run()`. Signals on the rebalance
  close, fills at the **next** close (no same-bar look-ahead), weights drift between rebalances, costs in bps per unit
  of turnover.
- **Metrics** — CAGR, volatility, Sharpe, Sortino, Calmar, max drawdown with peak/trough/recovery, longest underwater,
  best/worst month and year, turnover, time in market, beta/alpha/tracking error vs benchmark, yearly table.
- **Robustness** — `parameter_sweep`, `in_vs_out_of_sample`, anchored-window `walk_forward`.
- **Output** — `BacktestResult.save(path)` writes one JSON (`schema_version` 1.0): strategy, config, metrics,
  benchmark metrics, yearly table, equity/drawdown series and allocation changes. Every strategy emits the same shape.

## Minimal example

```python
import pandas as pd
from strategylab import load_prices, Backtest, Strategy

class SmaTrend(Strategy):
    def target_weights(self, prices, rb):
        sma = prices["SPY"].rolling(self.params["n"]).mean()
        on = (prices["SPY"] > sma).reindex(rb)
        return pd.DataFrame({"SPY": on.astype(float)}, index=rb)

px = load_prices(["SPY"], start="2000-01-01")
res = Backtest(px, SmaTrend(name="SMA", params={"n": 200}), benchmark="SPY", cost_bps=10, start="2001-01-01").run()
print(res.metrics)
res.save("sma.json")
```

## Strategies built on it

| Repo | Strategy | Article |
| --- | --- | --- |
| [dual-momentum](https://github.com/AlgoStrategyLab/dual-momentum) | Antonacci's Global Equities Momentum, 1997–2026 | [read](https://algostrategylab.github.io/dual-momentum/) |

MIT licensed. Research and education only, not investment advice.
