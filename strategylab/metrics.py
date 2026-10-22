"""Performance metrics on a daily return series."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def _max_dd_info(equity: pd.Series) -> dict:
    dd = drawdown_series(equity)
    trough = dd.idxmin()
    peak = equity.loc[:trough].idxmax()
    after = equity.loc[trough:]
    rec = after[after >= equity.loc[peak]]
    recovery = rec.index[0] if len(rec) else None
    return {
        "max_drawdown": round(float(dd.min()), 4),
        "max_dd_peak": str(peak.date()), "max_dd_trough": str(trough.date()),
        "max_dd_recovery": str(recovery.date()) if recovery is not None else None,
        "max_dd_days": int((trough - peak).days),
        "longest_underwater_days": int(_longest_underwater(dd)),
    }


def _longest_underwater(dd: pd.Series) -> int:
    longest = cur = 0
    start = None
    for d, v in dd.items():
        if v < 0:
            start = start or d
            cur = (d - start).days
            longest = max(longest, cur)
        else:
            start = None
            cur = 0
    return longest


def compute_metrics(daily: pd.Series, bench_daily: pd.Series | None = None, rf: float = 0.0) -> dict:
    daily = daily.dropna()
    n = len(daily)
    years = n / TRADING_DAYS
    equity = (1 + daily).cumprod()
    total = float(equity.iloc[-1] - 1)
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan
    vol = float(daily.std() * np.sqrt(TRADING_DAYS))
    ex = daily - rf / TRADING_DAYS
    sharpe = float(ex.mean() / daily.std() * np.sqrt(TRADING_DAYS)) if daily.std() > 0 else np.nan
    downside = daily[daily < 0].std() * np.sqrt(TRADING_DAYS)
    sortino = float(ex.mean() * TRADING_DAYS / downside) if downside > 0 else np.nan
    dd = _max_dd_info(equity)
    calmar = float(cagr / abs(dd["max_drawdown"])) if dd["max_drawdown"] < 0 else np.nan
    monthly = (1 + daily).resample("ME").prod() - 1
    out = {
        "start": str(daily.index[0].date()), "end": str(daily.index[-1].date()), "years": round(years, 2),
        "total_return": round(total, 4), "cagr": round(cagr, 4), "volatility": round(vol, 4),
        "sharpe": round(sharpe, 3), "sortino": round(sortino, 3), "calmar": round(calmar, 3),
        **dd,
        "best_month": round(float(monthly.max()), 4), "worst_month": round(float(monthly.min()), 4),
        "pct_positive_months": round(float((monthly > 0).mean()), 4),
        "best_year": None, "worst_year": None,
    }
    yearly = (1 + daily).groupby(daily.index.year).prod() - 1
    out["best_year"] = round(float(yearly.max()), 4)
    out["worst_year"] = round(float(yearly.min()), 4)
    if bench_daily is not None:
        b = bench_daily.reindex(daily.index).fillna(0.0)
        beq = (1 + b).cumprod()
        bcagr = float(beq.iloc[-1] ** (1 / years) - 1)
        cov = np.cov(daily, b)
        beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else np.nan
        alpha = float((daily.mean() - beta * b.mean()) * TRADING_DAYS)
        active = daily - b
        te = float(active.std() * np.sqrt(TRADING_DAYS))
        out.update({
            "beta": round(beta, 3), "alpha_annual": round(alpha, 4),
            "tracking_error": round(te, 4),
            "information_ratio": round(float(active.mean() * TRADING_DAYS / te), 3) if te > 0 else np.nan,
            "excess_cagr": round(cagr - bcagr, 4),
            "correlation": round(float(daily.corr(b)), 3),
        })
    return out


def yearly_table(daily: pd.Series, bench_daily: pd.Series | None = None) -> list[dict]:
    rows = []
    g = (1 + daily).groupby(daily.index.year)
    for yr, grp in g:
        eq = grp.cumprod()
        row = {"year": int(yr), "strategy": round(float(eq.iloc[-1] - 1), 4),
               "max_drawdown": round(float((eq / eq.cummax() - 1).min()), 4)}
        if bench_daily is not None:
            b = bench_daily.reindex(grp.index).fillna(0.0)
            row["benchmark"] = round(float((1 + b).prod() - 1), 4)
            row["excess"] = round(row["strategy"] - row["benchmark"], 4)
        rows.append(row)
    return rows
