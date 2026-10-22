"""Performance metrics on a daily return series.

``compute_metrics`` returns the core set used in every article.
``extended_metrics`` returns the full suite (returns, risk, tail, risk-adjusted, benchmark-relative).
``drawdown_table`` / ``monthly_returns_table`` return the classic exhibits.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- helpers
def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


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
        "max_dd_recovery_days": int((recovery - trough).days) if recovery is not None else None,
        "longest_underwater_days": int(_longest_underwater(dd)),
    }


def _streak(x: pd.Series, positive: bool) -> int:
    best = cur = 0
    for v in x:
        if (v > 0) == positive and v != 0:
            cur += 1; best = max(best, cur)
        else:
            cur = 0
    return best


def _ann_return(daily: pd.Series) -> float:
    n = len(daily)
    return float((1 + daily).prod() ** (TRADING_DAYS / n) - 1) if n > 0 else np.nan


# --------------------------------------------------------------------------- core
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
    downside = float(np.sqrt((np.minimum(ex, 0) ** 2).mean()) * np.sqrt(TRADING_DAYS))
    sortino = float(ex.mean() * TRADING_DAYS / downside) if downside > 0 else np.nan
    dd = _max_dd_info(equity)
    calmar = float(cagr / abs(dd["max_drawdown"])) if dd["max_drawdown"] < 0 else np.nan
    monthly = (1 + daily).resample("ME").prod() - 1
    yearly = (1 + daily).groupby(daily.index.year).prod() - 1
    out = {
        "start": str(daily.index[0].date()), "end": str(daily.index[-1].date()), "years": round(years, 2),
        "total_return": round(total, 4), "cagr": round(cagr, 4), "volatility": round(vol, 4),
        "sharpe": round(sharpe, 3), "sortino": round(sortino, 3), "calmar": round(calmar, 3),
        **dd,
        "best_month": round(float(monthly.max()), 4), "worst_month": round(float(monthly.min()), 4),
        "pct_positive_months": round(float((monthly > 0).mean()), 4),
        "best_year": round(float(yearly.max()), 4), "worst_year": round(float(yearly.min()), 4),
    }
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


# --------------------------------------------------------------------------- extended
def extended_metrics(daily: pd.Series, bench_daily: pd.Series | None = None, rf: float = 0.0) -> dict:
    """Full metric suite, grouped. All ratios use rf (annual) where applicable."""
    daily = daily.dropna()
    n = len(daily); years = n / TRADING_DAYS
    equity = (1 + daily).cumprod()
    dd = drawdown_series(equity)
    monthly = (1 + daily).resample("ME").prod() - 1
    quarterly = (1 + daily).resample("QE").prod() - 1
    yearly = (1 + daily).groupby(daily.index.year).prod() - 1
    rf_d = rf / TRADING_DAYS
    ex = daily - rf_d
    cagr = float(equity.iloc[-1] ** (1 / years) - 1)
    vol = float(daily.std() * np.sqrt(TRADING_DAYS))
    neg = daily[daily < 0]; pos = daily[daily > 0]
    downside_dev = float(np.sqrt((np.minimum(daily - rf_d, 0) ** 2).mean()) * np.sqrt(TRADING_DAYS))
    mdd = float(dd.min())

    # trailing returns (annualised past 1y)
    def trailing(days):
        if n < days: return None
        return round(float((1 + daily.iloc[-days:]).prod() ** (TRADING_DAYS / days if days > TRADING_DAYS else 1) - 1), 4)
    ytd = daily[daily.index.year == daily.index[-1].year]

    # drawdown stats
    ulcer = float(np.sqrt((dd ** 2).mean()))
    pain = float(dd.abs().mean())
    dd_tbl = drawdown_table(equity, top=100)
    dd_over10 = sum(1 for r in dd_tbl if r["depth"] <= -0.10)
    dd_over20 = sum(1 for r in dd_tbl if r["depth"] <= -0.20)
    avg_dd = float(np.mean([r["depth"] for r in dd_tbl])) if dd_tbl else 0.0
    avg_dd_len = float(np.mean([r["days_peak_to_recovery"] for r in dd_tbl if r["days_peak_to_recovery"] is not None])) if dd_tbl else 0.0

    # tail
    var95 = float(np.percentile(daily, 5)); var99 = float(np.percentile(daily, 1))
    cvar95 = float(daily[daily <= var95].mean()); cvar99 = float(daily[daily <= var99].mean())
    roll5 = (1 + daily).rolling(5).apply(np.prod, raw=True) - 1
    roll20 = (1 + daily).rolling(20).apply(np.prod, raw=True) - 1
    roll63 = (1 + daily).rolling(63).apply(np.prod, raw=True) - 1
    roll252 = (1 + daily).rolling(252).apply(np.prod, raw=True) - 1
    tail_ratio = float(abs(np.percentile(daily, 95) / np.percentile(daily, 5))) if np.percentile(daily, 5) != 0 else np.nan

    # ratios
    sharpe = float(ex.mean() / daily.std() * np.sqrt(TRADING_DAYS))
    sortino = float(ex.mean() * TRADING_DAYS / downside_dev) if downside_dev > 0 else np.nan
    calmar = float(cagr / abs(mdd)) if mdd < 0 else np.nan
    omega = float(pos.sum() / abs(neg.sum())) if len(neg) else np.nan            # threshold 0
    gain_to_pain_m = float(monthly.sum() / abs(monthly[monthly < 0].sum())) if (monthly < 0).any() else np.nan
    profit_factor_m = float(monthly[monthly > 0].sum() / abs(monthly[monthly < 0].sum())) if (monthly < 0).any() else np.nan
    martin = float((cagr - rf) / ulcer) if ulcer > 0 else np.nan                 # Ulcer Performance Index
    sterling = float((cagr - rf) / abs(avg_dd - 0.10)) if avg_dd - 0.10 != 0 else np.nan  # Sterling (avg DD - 10%)
    burke = float((cagr - rf) / np.sqrt(sum(r["depth"] ** 2 for r in dd_tbl))) if dd_tbl else np.nan
    kelly = float(daily.mean() / daily.var()) if daily.var() > 0 else np.nan
    skew = float(daily.skew()); kurt = float(daily.kurt())
    sharpe_se = float(np.sqrt((1 + 0.5 * sharpe ** 2) / years))                  # Lo (2002) approx, iid
    win_days = float((daily > 0).mean())
    avg_up_m = float(monthly[monthly > 0].mean()); avg_dn_m = float(monthly[monthly < 0].mean())
    payoff_m = float(avg_up_m / abs(avg_dn_m)) if avg_dn_m < 0 else np.nan
    expectancy_m = float(monthly.mean())

    out = {
        "returns": {
            "total_return": round(float(equity.iloc[-1] - 1), 4), "final_equity_per_dollar": round(float(equity.iloc[-1]), 2),
            "cagr": round(cagr, 4), "arithmetic_mean_annual": round(float(daily.mean() * TRADING_DAYS), 4),
            "geometric_mean_monthly": round(float((1 + monthly).prod() ** (1 / len(monthly)) - 1), 4),
            "avg_monthly_return": round(expectancy_m, 4), "median_monthly_return": round(float(monthly.median()), 4),
            "avg_positive_month": round(avg_up_m, 4), "avg_negative_month": round(avg_dn_m, 4),
            "best_day": round(float(daily.max()), 4), "worst_day": round(float(daily.min()), 4),
            "best_month": round(float(monthly.max()), 4), "worst_month": round(float(monthly.min()), 4),
            "best_quarter": round(float(quarterly.max()), 4), "worst_quarter": round(float(quarterly.min()), 4),
            "best_year": round(float(yearly.max()), 4), "worst_year": round(float(yearly.min()), 4),
            "pct_positive_days": round(win_days, 4), "pct_positive_months": round(float((monthly > 0).mean()), 4),
            "pct_positive_quarters": round(float((quarterly > 0).mean()), 4), "pct_positive_years": round(float((yearly > 0).mean()), 4),
            "max_consecutive_up_days": _streak(daily, True), "max_consecutive_down_days": _streak(daily, False),
            "max_consecutive_up_months": _streak(monthly, True), "max_consecutive_down_months": _streak(monthly, False),
            "ytd": round(float((1 + ytd).prod() - 1), 4), "trailing_1y": trailing(252), "trailing_3y_ann": trailing(756),
            "trailing_5y_ann": trailing(1260), "trailing_10y_ann": trailing(2520),
            "rolling_1y_min": round(float(roll252.min()), 4), "rolling_1y_median": round(float(roll252.median()), 4),
            "rolling_1y_max": round(float(roll252.max()), 4), "pct_rolling_1y_positive": round(float((roll252.dropna() > 0).mean()), 4),
        },
        "risk": {
            "volatility": round(vol, 4), "downside_deviation": round(downside_dev, 4),
            "upside_volatility": round(float(pos.std() * np.sqrt(TRADING_DAYS)), 4),
            "monthly_volatility": round(float(monthly.std()), 4),
            "max_drawdown": round(mdd, 4), "avg_drawdown": round(avg_dd, 4), "avg_drawdown_days": round(avg_dd_len, 0),
            "drawdowns_over_10pct": dd_over10, "drawdowns_over_20pct": dd_over20,
            "ulcer_index": round(ulcer, 4), "pain_index": round(pain, 4),
            "time_underwater_pct": round(float((dd < 0).mean()), 4), "time_under_10pct_dd": round(float((dd < -0.10).mean()), 4),
            **_max_dd_info(equity),
            "var_95_daily": round(var95, 4), "var_99_daily": round(var99, 4),
            "cvar_95_daily": round(cvar95, 4), "cvar_99_daily": round(cvar99, 4),
            "worst_5_day": round(float(roll5.min()), 4), "worst_20_day": round(float(roll20.min()), 4), "worst_63_day": round(float(roll63.min()), 4),
            "skewness": round(skew, 3), "excess_kurtosis": round(kurt, 3), "tail_ratio": round(tail_ratio, 3),
        },
        "risk_adjusted": {
            "sharpe": round(sharpe, 3), "sharpe_std_error": round(sharpe_se, 3), "sortino": round(sortino, 3),
            "calmar": round(calmar, 3), "mar_ratio": round(calmar, 3), "omega": round(omega, 3),
            "gain_to_pain_monthly": round(gain_to_pain_m, 3), "profit_factor_monthly": round(profit_factor_m, 3),
            "payoff_ratio_monthly": round(payoff_m, 3), "martin_ratio_upi": round(martin, 3),
            "sterling_ratio": round(sterling, 3), "burke_ratio": round(burke, 3),
            "return_to_vol": round(cagr / vol, 3), "kelly_fraction": round(kelly, 3),
            "risk_of_ruin_proxy_pct_months_below_minus10": round(float((monthly < -0.10).mean()), 4),
        },
    }
    if bench_daily is not None:
        b = bench_daily.reindex(daily.index).fillna(0.0)
        bm = (1 + b).resample("ME").prod() - 1
        beq = (1 + b).cumprod(); bcagr = float(beq.iloc[-1] ** (1 / years) - 1)
        bvol = float(b.std() * np.sqrt(TRADING_DAYS))
        cov = np.cov(daily, b); beta = float(cov[0, 1] / cov[1, 1])
        alpha = float(((daily - rf_d) - beta * (b - rf_d)).mean() * TRADING_DAYS)
        active = daily - b; te = float(active.std() * np.sqrt(TRADING_DAYS))
        up = b > 0; dn = b < 0
        up_cap = float(((1 + daily[up]).prod() ** (TRADING_DAYS / up.sum()) - 1) / ((1 + b[up]).prod() ** (TRADING_DAYS / up.sum()) - 1))
        dn_cap = float(((1 + daily[dn]).prod() ** (TRADING_DAYS / dn.sum()) - 1) / ((1 + b[dn]).prod() ** (TRADING_DAYS / dn.sum()) - 1))
        upm = bm > 0; dnm = bm < 0
        up_cap_m = float(monthly[upm].mean() / bm[upm].mean()); dn_cap_m = float(monthly[dnm].mean() / bm[dnm].mean())
        beta_up = float(np.cov(daily[up], b[up])[0, 1] / b[up].var()); beta_dn = float(np.cov(daily[dn], b[dn])[0, 1] / b[dn].var())
        rel = equity / beq; rel_dd = float((rel / rel.cummax() - 1).min())
        m2 = float(rf + (sharpe * bvol))                                            # Modigliani M²
        treynor = float((cagr - rf) / beta) if beta != 0 else np.nan
        out["vs_benchmark"] = {
            "benchmark_cagr": round(bcagr, 4), "excess_cagr": round(cagr - bcagr, 4),
            "benchmark_volatility": round(bvol, 4), "benchmark_max_drawdown": round(float(drawdown_series(beq).min()), 4),
            "benchmark_sharpe": round(float((b - rf_d).mean() / b.std() * np.sqrt(TRADING_DAYS)), 3),
            "beta": round(beta, 3), "beta_up_markets": round(beta_up, 3), "beta_down_markets": round(beta_dn, 3),
            "jensen_alpha_annual": round(alpha, 4), "correlation": round(float(daily.corr(b)), 3),
            "r_squared": round(float(daily.corr(b) ** 2), 3), "tracking_error": round(te, 4),
            "information_ratio": round(float(active.mean() * TRADING_DAYS / te), 3) if te > 0 else np.nan,
            "treynor_ratio": round(treynor, 4), "m2_measure": round(m2, 4), "m2_excess": round(m2 - bcagr, 4),
            "up_capture_daily": round(up_cap, 3), "down_capture_daily": round(dn_cap, 3),
            "up_capture_monthly": round(up_cap_m, 3), "down_capture_monthly": round(dn_cap_m, 3),
            "capture_ratio_monthly": round(up_cap_m / dn_cap_m, 3) if dn_cap_m != 0 else np.nan,
            "batting_average_monthly": round(float((monthly > bm).mean()), 4),
            "batting_average_yearly": round(float((yearly > ((1 + b).groupby(b.index.year).prod() - 1)).mean()), 4),
            "max_relative_drawdown": round(rel_dd, 4),
            "months_outperformed": int((monthly > bm).sum()), "months_underperformed": int((monthly < bm).sum()),
            "avg_excess_in_benchmark_down_months": round(float((monthly[dnm] - bm[dnm]).mean()), 4),
            "avg_excess_in_benchmark_up_months": round(float((monthly[upm] - bm[upm]).mean()), 4),
        }
    return out


def drawdown_table(equity: pd.Series, top: int = 5, min_depth: float = 0.0) -> list[dict]:
    """Top-N drawdowns: peak, trough, recovery, depth, durations."""
    dd = drawdown_series(equity)
    rows = []
    in_dd = False
    for d, v in dd.items():
        if v < 0 and not in_dd:
            in_dd = True; start = d; trough = d; depth = v
        elif v < 0 and in_dd:
            if v < depth: depth, trough = v, d
        elif v == 0 and in_dd:
            in_dd = False
            peak = equity.loc[:start].idxmax()
            rows.append({"peak": peak, "trough": trough, "recovery": d, "depth": float(depth)})
    if in_dd:
        peak = equity.loc[:start].idxmax()
        rows.append({"peak": peak, "trough": trough, "recovery": None, "depth": float(depth)})
    rows = [r for r in rows if r["depth"] <= -abs(min_depth)]
    rows.sort(key=lambda r: r["depth"])
    out = []
    for r in rows[:top]:
        out.append({
            "peak": str(r["peak"].date()), "trough": str(r["trough"].date()),
            "recovery": str(r["recovery"].date()) if r["recovery"] is not None else None,
            "depth": round(r["depth"], 4),
            "days_peak_to_trough": int((r["trough"] - r["peak"]).days),
            "days_trough_to_recovery": int((r["recovery"] - r["trough"]).days) if r["recovery"] is not None else None,
            "days_peak_to_recovery": int((r["recovery"] - r["peak"]).days) if r["recovery"] is not None else None,
        })
    return out


def monthly_returns_table(daily: pd.Series) -> pd.DataFrame:
    """Years x months grid of returns plus a Year column."""
    m = (1 + daily).resample("ME").prod() - 1
    tbl = pd.DataFrame({"year": m.index.year, "month": m.index.month, "ret": m.values}).pivot(index="year", columns="month", values="ret")
    tbl.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][: len(tbl.columns)] if len(tbl.columns) == 12 else [pd.Timestamp(2000, c, 1).strftime("%b") for c in tbl.columns]
    tbl["Year"] = (1 + daily).groupby(daily.index.year).prod() - 1
    return tbl


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
