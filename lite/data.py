"""
Sovereign Lite v7 — yFinance-only data layer.

  - Price history: batched `yf.download` (one request per symbol, threaded by yf)
  - Fundamentals: per-symbol `Ticker.info` + quarterly income statement,
    cached in SQLite for 24 h so repeated scans are instant
  - Benchmarks: NIFTY 50 (^NSEI) and India VIX (^INDIAVIX)

Everything degrades gracefully: a failed symbol is skipped, never fatal.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from . import db

HISTORY_PERIOD = "2y"        # enough bars for 200-SMA + 52w + 12M momentum
FUNDAMENTAL_TTL_HOURS = 24
PRICE_TTL_HOURS = 4
_DOWNLOAD_CHUNK = 25
_MAX_WORKERS = 6


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


# ── Price history ───────────────────────────────────────────────────────────

def fetch_prices(symbols: list[str], period: str = HISTORY_PERIOD) -> dict[str, pd.DataFrame]:
    """Fetch daily OHLCV for all symbols. Returns {symbol: DataFrame}."""
    import yfinance as yf

    result: dict[str, pd.DataFrame] = {}
    for chunk in _chunks(list(dict.fromkeys(symbols)), _DOWNLOAD_CHUNK):
        try:
            df = yf.download(
                tickers=chunk,
                period=period,
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
        except Exception as exc:  # pragma: no cover - network
            print(f"  ⚠️ download chunk failed ({len(chunk)} symbols): {exc}")
            continue
        if df is None or df.empty:
            continue
        if len(chunk) == 1 and not isinstance(df.columns, pd.MultiIndex):
            result[chunk[0]] = df[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")
        else:
            for sym in chunk:
                if sym in df.columns.get_level_values(0):
                    try:
                        sub = df[sym][["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")
                        if not sub.empty:
                            result[sym] = sub
                    except KeyError:
                        continue
    return result


def persist_prices(prices: dict[str, pd.DataFrame]) -> int:
    total = 0
    for symbol, df in prices.items():
        rows = []
        for idx, r in df.iterrows():
            if pd.isna(r["Close"]):
                continue
            rows.append(
                {
                    "date": idx.date().isoformat(),
                    "open": _clean(r["Open"]),
                    "high": _clean(r["High"]),
                    "low": _clean(r["Low"]),
                    "close": _clean(r["Close"]),
                    "volume": int(r["Volume"]) if not pd.isna(r["Volume"]) else 0,
                }
            )
        if rows:
            total += db.upsert_prices(symbol, rows)
    return total


# ── Fundamentals ────────────────────────────────────────────────────────────

def _ticker_fast_info(symbol: str) -> Optional[dict]:
    """Lightweight quote via yfinance fast_info (fallback share counts)."""
    try:
        import yfinance as yf

        fi = yf.Ticker(symbol).fast_info
        return {"shares": fi.get("shares")}
    except Exception:
        return None


def _from_info(info: dict, symbol: str = "") -> dict:
    """Extract the metrics we need from a yfinance `Ticker.info` dict."""
    shares = info.get("sharesOutstanding") or 0
    book = info.get("bookValue") or 0
    total_debt = info.get("totalDebt") or 0
    ebitda = info.get("ebitda") or 0
    equity = shares * book if shares and book else 0
    capital_employed = equity + total_debt

    market_cap_inr = info.get("marketCap") or 0
    roce = None
    if ebitda and capital_employed > 0:
        roce = ebitda / capital_employed  # EBIT(≈EBITDA) / capital employed
        roce = min(roce, 3.0)  # clamp: tiny capital bases produce absurd ROCE
    roe = info.get("returnOnEquity")
    if roe is not None:
        roe = roe * 100
    else:
        # Fallback: Net Income / book equity (yfinance often omits returnOnEquity
        # for NSE names; shares derived from market cap when sharesOutstanding
        # is missing).
        ni = info.get("netIncomeToCommon")
        bv = info.get("bookValue")
        shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
        if not shares:
            px = info.get("currentPrice") or info.get("regularMarketPrice")
            mcap = info.get("marketCap")
            if mcap and px:
                shares = mcap / px
        if not shares:
            try:
                fi = _ticker_fast_info(symbol)
                if fi:
                    shares = fi.get("shares")
            except Exception:
                pass
        if ni is not None and bv and shares:
            equity = bv * shares
            if equity > 0:
                roe = ni / equity * 100
    debt_equity = info.get("debtToEquity")
    if debt_equity is not None:
        debt_equity = debt_equity / 100.0

    revenue = info.get("totalRevenue") or 0
    fcf = info.get("freeCashflow")
    fcf_margin = (fcf / revenue * 100) if (fcf is not None and revenue and revenue > 0) else None

    return {
        "market_cap": market_cap_inr / 1e7 if market_cap_inr else None,  # → ₹ Cr
        "roe": roe,
        "roce": roce * 100 if roce is not None else None,
        "debt_equity": debt_equity,
        "sales_growth": (info.get("revenueGrowth") or 0) * 100 if info.get("revenueGrowth") is not None else None,
        "profit_growth": (info.get("earningsGrowth") or 0) * 100 if info.get("earningsGrowth") is not None else None,
        "pe": info.get("trailingPE"),
        "pb": info.get("priceToBook"),
        "fcf_margin": fcf_margin,
        "eps_growth": (info.get("earningsGrowth") or 0) * 100 if info.get("earningsGrowth") is not None else None,
        "promoter_holding": (info.get("heldPercentInsiders") or 0) * 100 if info.get("heldPercentInsiders") is not None else None,
        "sector": info.get("sector") or "Unknown",
        "name": info.get("longName") or info.get("shortName") or "",
        "beta": info.get("beta"),
    }


def _yoy_growths(series: pd.Series) -> list[Optional[float]]:
    """YoY growth (as fraction) for the last 4 points of a quarterly series."""
    growths = []
    for i in range(len(series) - 4, len(series)):
        base = series.iloc[i - 4]
        cur = series.iloc[i]
        if base is None or cur is None or base == 0:
            growths.append(None)
        else:
            growths.append(max(min(cur / base - 1, 5.0), -5.0))
    return growths


def _trend_score(growths: list[Optional[float]]) -> Optional[float]:
    """0-100 acceleration score: latest YoY growth (60%) + slope (40%).

    Slope is first→last quarter growth delta, so 10→18→25→38% scores far
    higher than flat 20% — that's the earnings-acceleration signal.
    """
    valid = [g for g in growths if g is not None]
    if len(valid) < 2:
        return None
    latest = valid[-1]
    slope = valid[-1] - valid[0]
    base_score = 100 / (1 + math.exp(-(latest * 100 - 20) / 15))
    slope_score = 100 / (1 + math.exp(-(slope * 100 - 5) / 10))
    return max(0.0, min(100.0, base_score * 0.6 + slope_score * 0.4))


def _quarterly_metrics(symbol: str) -> dict:
    """Quarterly income-statement metrics for a symbol.

    Returns {"eps_accel": blended 0-100 Revenue/EPS/PAT acceleration,
    "rev_accel"/"pat_accel": individual 0-100 trend scores,
    "eps_quarters": YoY PAT-growth list, "rev_quarters": YoY revenue-growth
    list, "margin_expansion": pp change in net margin (last 4Q avg vs prior
    4Q avg), "quarters": [{period_end, quarter, revenue, net_income,
    net_margin}]}. Every field degrades to None/[] when unavailable.
    """
    import yfinance as yf

    out = {
        "eps_accel": None,
        "rev_accel": None,
        "pat_accel": None,
        "eps_quarters": [],
        "rev_quarters": [],
        "margin_expansion": None,
        "quarters": [],
    }
    try:
        ticker = yf.Ticker(symbol)
        stmt = ticker.quarterly_income_stmt
        if stmt is None or stmt.empty:
            return out
        ni = (
            stmt.loc["Net Income"].dropna().astype(float).sort_index()
            if "Net Income" in stmt.index
            else pd.Series(dtype=float)
        )
        rev = (
            stmt.loc["Total Revenue"].dropna().astype(float).sort_index()
            if "Total Revenue" in stmt.index
            else pd.Series(dtype=float)
        )
        eps = (
            stmt.loc["Diluted EPS"].dropna().astype(float).sort_index()
            if "Diluted EPS" in stmt.index
            else pd.Series(dtype=float)
        )
        if len(ni) < 5:
            return out

        # YoY growth per quarter → acceleration score for each trend
        pat_growths = _yoy_growths(ni)
        rev_growths = _yoy_growths(rev) if len(rev) >= 5 else []
        eps_growths = _yoy_growths(eps) if len(eps) >= 5 else []
        out["rev_accel"] = _trend_score(rev_growths)
        out["pat_accel"] = _trend_score(pat_growths)
        eps_accel_raw = _trend_score(eps_growths)
        # Blend: Revenue 35% · PAT 35% · EPS 30% (available parts only)
        parts = [(out["rev_accel"], 0.35), (out["pat_accel"], 0.35), (eps_accel_raw, 0.30)]
        num = den = 0.0
        for v, w in parts:
            if v is not None:
                num += v * w
                den += w
        if den > 0:
            out["eps_accel"] = max(0.0, min(100.0, num / den))
        out["eps_quarters"] = [round(g * 100, 1) if g is not None else None for g in pat_growths]
        out["rev_quarters"] = [round(g * 100, 1) if g is not None else None for g in rev_growths]

        # Net margin per quarter → margin-expansion signal (last 4Q avg vs prior 4Q avg)
        margins = []
        quarters = []
        for period_end in ni.index[-8:]:
            income = float(ni.loc[period_end]) if period_end in ni.index else None
            revenue = float(rev.loc[period_end]) if period_end in rev.index else None
            margin = (income / revenue * 100) if (income is not None and revenue and revenue > 0) else None
            margins.append(margin)
            quarters.append(
                {
                    "period_end": str(period_end.date()),
                    "quarter": f"{period_end.year}Q{(period_end.month - 1) // 3 + 1}",
                    "revenue": revenue,
                    "net_income": income,
                    "net_margin": margin,
                }
            )
        recent = [m for m in margins[-4:] if m is not None]
        prior = [m for m in margins[-8:-4] if m is not None]
        if recent and prior:
            out["margin_expansion"] = round(sum(recent) / len(recent) - sum(prior) / len(prior), 2)
        out["quarters"] = quarters
        return out
    except Exception as exc:  # pragma: no cover - network
        print(f"  ⚠️ quarterly stmt failed for {symbol}: {exc}")
        return out


def fetch_fundamentals(symbols: list[str], force: bool = False) -> dict[str, dict]:
    """Fetch fundamentals for symbols missing a fresh cached copy."""
    import yfinance as yf

    cutoff = datetime.now() - timedelta(hours=FUNDAMENTAL_TTL_HOURS)
    to_fetch = []
    for sym in symbols:
        cached = db.fundamentals_for(sym)
        fresh = cached and cached.get("last_updated") and _parse_dt(cached["last_updated"]) > cutoff
        if force or not fresh:
            to_fetch.append(sym)

    results: dict[str, dict] = {}
    for cached in db.load_fundamentals():
        if cached["symbol"] in symbols:
            results[cached["symbol"]] = cached

    def _one(sym: str) -> tuple[str, dict]:
        try:
            ticker = yf.Ticker(sym)
            info = ticker.info or {}
            metrics = _from_info(info, sym)
            metrics["symbol"] = sym
            q = _quarterly_metrics(sym)
            metrics["eps_accel"] = q["eps_accel"]
            metrics["rev_accel"] = q["rev_accel"]
            metrics["pat_accel"] = q["pat_accel"]
            metrics["eps_quarters"] = q["eps_quarters"]
            metrics["rev_quarters"] = q["rev_quarters"]
            metrics["margin_expansion"] = q["margin_expansion"]
            if q["quarters"]:
                db.upsert_quarterly_results(sym, q["quarters"])
            db.upsert_fundamentals(metrics)
            return sym, metrics
        except Exception as exc:
            print(f"  ⚠️ fundamentals failed for {sym}: {exc}")
            return sym, {}

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futs = {pool.submit(_one, sym): sym for sym in to_fetch}
        for fut in as_completed(futs):
            sym, metrics = fut.result()
            if metrics:
                results[sym] = metrics
    return results


# ── Benchmarks / regime inputs ──────────────────────────────────────────────

def fetch_benchmark(symbol: str = "^NSEI", period: str = HISTORY_PERIOD) -> Optional[pd.DataFrame]:
    import yfinance as yf

    try:
        df = yf.download(
            tickers=symbol,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            # yfinance 1.x returns (Price, Ticker) MultiIndex even for a
            # single ticker — flatten so `df["Close"]` is a plain Series.
            df.columns = df.columns.get_level_values(0)
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")
    except Exception as exc:  # pragma: no cover - network
        print(f"  ⚠️ benchmark {symbol} failed: {exc}")
        return None


def fetch_quick_quote(symbol: str) -> Optional[dict]:
    """Live 5-day quote for a single symbol (used by the portfolio screen)."""
    import yfinance as yf

    try:
        df = yf.download(
            tickers=symbol,
            period="5d",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if df is None or df.empty:
            return None
        closes = df["Close"].dropna()
        if len(closes) < 2:
            return None
        price = float(closes.iloc[-1])
        prev = float(closes.iloc[-2])
        return {
            "symbol": symbol,
            "price": price,
            "prev_close": prev,
            "change": price - prev,
            "change_pct": ((price - prev) / prev * 100) if prev else 0.0,
            "date": closes.index[-1].date().isoformat(),
        }
    except Exception:  # pragma: no cover - network
        return None


def _clean(v) -> Optional[float]:
    if v is None or pd.isna(v):
        return None
    return float(v)


def _parse_dt(s: str) -> datetime:
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return datetime.min
