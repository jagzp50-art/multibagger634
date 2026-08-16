"""
Sovereign Lite v7 — one-click backtest (Phase 5).

A clean, lookahead-free momentum strategy over the stored price history:

  - Rebalance monthly: rank universe by 6M momentum + trend template at that date
  - Buy the top-N names, equal weight
  - Exit: -20% trailing stop, or close < 50-DMA
  - Position closed at the next rebalance

Outputs an equity curve, trade log, and summary stats (CAGR, max drawdown,
Sharpe, win rate). Price-only by design — no historical fundamentals needed.
"""
from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd

from . import db, indicators

DEFAULT_PARAMS = {
    "years": 3,
    "top_n": 10,
    "initial_capital": 1_000_000,  # ₹10L
    "trailing_stop_pct": 0.20,
}


def run_backtest(
    prices_by_symbol: dict[str, pd.DataFrame],
    params: dict | None = None,
) -> dict:
    p = {**DEFAULT_PARAMS, **(params or {})}
    years = max(1, min(int(p.get("years", 3)), 10))
    top_n = max(1, int(p.get("top_n", 10)))
    initial = float(p.get("initial_capital", 1_000_000))
    stop = float(p.get("trailing_stop_pct", 0.20))

    # Align all series on a common trading calendar.
    frames = {}
    for sym, df in prices_by_symbol.items():
        if df is None or df.empty or "close" not in df.columns:
            continue
        s = df["close"].dropna()
        if len(s) > 260:  # need history for momentum + SMA200
            frames[sym] = s
    if not frames:
        return _empty_result(p, "Not enough price history stored. Run a scan first.")

    aligned = pd.concat(frames, axis=1).sort_index().ffill()
    end_date = aligned.index[-1]
    start_date = end_date - pd.DateOffset(years=years)
    aligned = aligned[aligned.index >= start_date]
    if len(aligned) < 60:
        return _empty_result(p, "Not enough history in the requested window.")

    closes = aligned
    sma50 = aligned.rolling(50).mean()
    sma200 = aligned.rolling(200).mean()

    # Monthly rebalance dates.
    rebalance_dates = list(
        pd.Series(closes.index).groupby([closes.index.year, closes.index.month]).last()
    )
    rebalance_dates = [d for d in rebalance_dates if d >= start_date]

    equity = initial
    peak = initial
    cash = initial
    holdings: dict[str, dict] = {}
    curve: list[dict] = []
    trades: list[dict] = []
    daily_equity = []
    buy_notional = 0.0
    sell_notional = 0.0

    for i in range(1, len(closes)):
        day = closes.index[i]
        prev = closes.index[i - 1]

        # Mark-to-market (iterate a snapshot — positions may be closed mid-loop)
        day_equity = cash
        for sym, pos in list(holdings.items()):
            px = closes.loc[day, sym]
            if px is None or pd.isna(px):
                continue
            day_equity += pos["shares"] * px
            pos["peak"] = max(pos["peak"], px)
            # Trailing stop
            if pos["peak"] * (1 - stop) > px:
                proceeds = pos["shares"] * px
                cash += proceeds
                sell_notional += proceeds
                trades.append(_trade(pos, sym, day, px, "trailing_stop"))
                del holdings[sym]
            # 50-DMA exit
            elif sma50.loc[day, sym] is not None and not pd.isna(sma50.loc[day, sym]) and px < sma50.loc[day, sym]:
                proceeds = pos["shares"] * px
                cash += proceeds
                sell_notional += proceeds
                trades.append(_trade(pos, sym, day, px, "sma50_break"))
                del holdings[sym]

        daily_equity.append(day_equity)
        curve.append({"date": day.date().isoformat(), "equity": round(day_equity, 2)})

        # Rebalance at month end
        if day in rebalance_dates and day != rebalance_dates[0]:
            # Rank candidates by momentum + trend at this date (no lookahead)
            candidates = []
            for sym in closes.columns:
                px = closes.loc[day, sym]
                if px is None or pd.isna(px) or px <= 0:
                    continue
                window = closes[sym].loc[:day]
                if len(window) < 60:
                    continue
                ret6 = _ret(window, 126)
                above200 = not (sma200.loc[day, sym] is None or pd.isna(sma200.loc[day, sym])) and px > sma200.loc[day, sym]
                above50 = not (sma50.loc[day, sym] is None or pd.isna(sma50.loc[day, sym])) and px > sma50.loc[day, sym]
                if ret6 is None:
                    continue
                score = ret6 * 2.0 + (1.0 if above200 else 0.0) + (0.5 if above50 else 0.0)
                candidates.append((score, sym))
            candidates.sort(reverse=True)
            picks = [s for _, s in candidates[:top_n]]

            # Liquidate positions not in picks
            for sym in list(holdings.keys()):
                if sym not in picks:
                    px = closes.loc[day, sym]
                    if px is not None and not pd.isna(px):
                        proceeds = holdings[sym]["shares"] * px
                        cash += proceeds
                        sell_notional += proceeds
                        trades.append(_trade(holdings[sym], sym, day, px, "rebalance"))
                    del holdings[sym]

            # Deploy cash into picks, equal weight
            if picks:
                budget = cash / len(picks)
                for sym in picks:
                    px = closes.loc[day, sym]
                    if px is None or pd.isna(px) or px <= 0:
                        continue
                    shares = math.floor(budget / px)
                    if shares <= 0:
                        continue
                    cost = shares * px
                    cash -= cost
                    buy_notional += cost
                    holdings[sym] = {"shares": shares, "entry": px, "peak": px, "entry_date": day.date().isoformat()}

    # Close remaining positions at the end
    for sym in list(holdings.keys()):
        px = closes.iloc[-1][sym]
        if px is not None and not pd.isna(px):
            proceeds = holdings[sym]["shares"] * px
            cash += proceeds
            sell_notional += proceeds
            trades.append(_trade(holdings[sym], sym, closes.index[-1], px, "end"))

    final = cash
    equity_series = pd.Series([p["equity"] for p in curve])
    summary = _summary(initial, final, equity_series, trades, len(rebalance_dates) - 1, buy_notional, sell_notional)

    # Universe equal-weight benchmark: the average member of the candidate pool.
    # The model must beat its own universe, not just a stored index.
    universe_curve: list[dict] = []
    bench = aligned.mean(axis=1).dropna()
    if len(bench) >= 60 and float(bench.iloc[0]) > 0:
        norm = bench / float(bench.iloc[0]) * initial
        by_date = {d.date().isoformat(): float(v) for d, v in zip(norm.index, norm)}
        last_v = None
        for pt in curve:
            if pt["date"] in by_date:
                last_v = by_date[pt["date"]]
            if last_v is not None:
                universe_curve.append({"date": pt["date"], "value": round(last_v, 2)})
        b_ret = float(norm.iloc[-1] / norm.iloc[0] - 1)
        b_years = max(len(norm) / 252, 1 / 252)
        b_cagr = ((norm.iloc[-1] / norm.iloc[0]) ** (1 / b_years) - 1) * 100
        b_mdd = float(((norm - norm.cummax()) / norm.cummax()).min() * 100)
        strat_ret = final / initial - 1
        summary.update(
            {
                "universe_benchmark": "Universe EW",
                "universe_return_pct": round(b_ret * 100, 2),
                "universe_cagr_pct": round(b_cagr, 2),
                "universe_max_dd_pct": round(b_mdd, 2),
                "universe_alpha_pct": round((strat_ret - b_ret) * 100, 2),
            }
        )

    return {
        "params": p,
        "equity_curve": curve,
        "universe_curve": universe_curve,
        "trades": trades,
        "summary": summary,
        "warnings": [],
    }


def walk_forward(
    prices_by_symbol: dict[str, pd.DataFrame],
    params: dict | None = None,
) -> dict:
    """Walk-forward validation: run the exact same rule-based strategy on N
    consecutive 12-month test windows (each with a 12-month warmup before it).

    The strategy is parameter-free, so this measures how it performed across
    different regimes instead of one cherry-picked window — hit rate, average
    return and worst drawdown across folds flag overfit/regime dependence.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    fold_months = max(6, min(int(p.get("fold_months", 12)), 24))
    folds = max(2, min(int(p.get("folds", 3)), 6))
    top_n = max(1, int(p.get("top_n", 10)))
    initial = float(p.get("initial_capital", 1_000_000))
    stop = float(p.get("trailing_stop_pct", 0.20))

    frames = {}
    for sym, df in prices_by_symbol.items():
        if df is None or df.empty or "close" not in df.columns:
            continue
        s = df["close"].dropna()
        if len(s) > 260:
            frames[sym] = s
    if not frames:
        return _empty_result(p, "Not enough price history stored. Run a scan first.")

    aligned = pd.concat(frames, axis=1).sort_index().ffill()
    n_bars = len(aligned)
    fold_days = fold_months * 21
    warmup = 260
    need = warmup + folds * fold_days
    if n_bars < need:
        return _empty_result(
            p,
            f"Walk-forward needs {need} bars ({folds} × {fold_months}m folds + warmup) but only {n_bars} are stored. "
            "Run a scan after 5y of prices are fetched.",
        )

    fold_results = []
    for i in range(folds):
        test_end = n_bars - i * fold_days
        test_start = test_end - fold_days
        slice_start = test_start - warmup
        sub = {
            sym: pd.DataFrame({"close": aligned[sym].iloc[slice_start:test_end]})
            for sym in aligned.columns
        }
        res = run_backtest(
            sub,
            {"years": 1, "top_n": top_n, "initial_capital": initial, "trailing_stop_pct": stop},
        )
        sm = res.get("summary") or {}
        if sm.get("error"):
            continue
        start_label = aligned.index[test_start].date().isoformat()
        end_label = aligned.index[test_end - 1].date().isoformat()
        fold_results.append(
            {
                "fold": i + 1,
                "window": f"{start_label} → {end_label}",
                "net_return_pct": sm.get("net_return_pct"),
                "cagr_pct": sm.get("cagr_pct"),
                "max_drawdown_pct": sm.get("max_drawdown_pct"),
                "sharpe": sm.get("sharpe"),
                "win_rate_pct": sm.get("win_rate_pct"),
                "trades": sm.get("trades"),
            }
        )

    if not fold_results:
        return _empty_result(p, "No folds could be evaluated from the stored history.")

    returns = [f["net_return_pct"] for f in fold_results if f["net_return_pct"] is not None]
    cagrs = [f["cagr_pct"] for f in fold_results if f["cagr_pct"] is not None]
    dds = [f["max_drawdown_pct"] for f in fold_results if f["max_drawdown_pct"] is not None]
    sharpes = [f["sharpe"] for f in fold_results if f["sharpe"] is not None]
    win_rates = [f["win_rate_pct"] for f in fold_results if f["win_rate_pct"] is not None]

    summary = {
        "folds_evaluated": len(fold_results),
        "fold_months": fold_months,
        "avg_return_pct": round(sum(returns) / len(returns), 2) if returns else None,
        "hit_rate_pct": round(sum(1 for r in returns if r > 0) / len(returns) * 100, 1) if returns else None,
        "avg_cagr_pct": round(sum(cagrs) / len(cagrs), 2) if cagrs else None,
        "worst_max_drawdown_pct": round(min(dds), 2) if dds else None,
        "avg_sharpe": round(sum(sharpes) / len(sharpes), 2) if sharpes else None,
        "avg_win_rate_pct": round(sum(win_rates) / len(win_rates), 1) if win_rates else None,
        "total_trades": sum(f["trades"] or 0 for f in fold_results),
        "run_date": date.today().isoformat(),
    }
    return {"params": p, "folds": fold_results, "summary": summary, "warnings": []}


def _ret(series: pd.Series, days: int) -> float | None:
    if len(series) < days + 1:
        return None
    start, end = series.iloc[-days - 1], series.iloc[-1]
    if start is None or pd.isna(start) or start <= 0 or pd.isna(end):
        return None
    return float(end / start - 1)


def _trade(pos: dict, symbol: str, day, price: float, reason: str) -> dict:
    return {
        "symbol": symbol,
        "exit_date": day.date().isoformat(),
        "entry_date": pos.get("entry_date"),
        "exit_price": round(price, 2),
        "entry_price": round(pos["entry"], 2),
        "return_pct": round((price / pos["entry"] - 1) * 100, 2),
        "shares": pos.get("shares"),
        "reason": reason,
    }


def _summary(
    initial: float,
    final: float,
    equity_series: pd.Series,
    trades: list,
    n_rebalances: int,
    buy_notional: float = 0.0,
    sell_notional: float = 0.0,
) -> dict:
    n_days = len(equity_series)
    years = max(n_days / 252, 1 / 252)
    cagr = ((final / initial) ** (1 / years) - 1) * 100 if final > 0 and initial > 0 else 0.0
    mdd = float(((equity_series - equity_series.cummax()) / equity_series.cummax()).min() * 100) if n_days else 0.0
    rets = equity_series.pct_change().dropna()
    sharpe = float(rets.mean() / rets.std() * math.sqrt(252)) if len(rets) > 2 and rets.std() > 0 else 0.0
    downside = rets[rets < 0]
    dstd = float(downside.std(ddof=0)) if len(downside) > 2 else 0.0
    sortino = float(rets.mean() / dstd * math.sqrt(252)) if dstd > 0 else 0.0
    avg_equity = float(equity_series.mean()) if n_days else initial
    turnover = ((buy_notional + sell_notional) / 2 / avg_equity / years) if avg_equity > 0 else 0.0
    wins = [t for t in trades if t.get("return_pct", 0) > 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0.0
    avg_win = sum(t["return_pct"] for t in wins) / len(wins) if wins else 0.0
    losses = [t for t in trades if t.get("return_pct", 0) <= 0]
    avg_loss = sum(t["return_pct"] for t in losses) / len(losses) if losses else 0.0
    return {
        "initial_capital": round(initial, 2),
        "final_value": round(final, 2),
        "net_return_pct": round((final / initial - 1) * 100, 2) if initial else 0.0,
        "cagr_pct": round(cagr, 2),
        "max_drawdown_pct": round(mdd, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "turnover_annual": round(turnover, 2),
        "win_rate_pct": round(win_rate, 1),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "trades": len(trades),
        "rebalances": n_rebalances,
        "run_date": date.today().isoformat(),
    }


def _empty_result(params: dict, message: str) -> dict:
    return {
        "params": params,
        "equity_curve": [],
        "trades": [],
        "summary": {"error": message},
        "warnings": [message],
    }
