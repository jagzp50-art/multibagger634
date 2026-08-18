"""
Sovereign Lite v16 — NSE daily delivery position (accumulation/distribution).

yFinance has no delivery data, so this module reads NSE's public daily
"delivery position" report (MA{DDMMYYYY}.csv under nsearchives) directly.
The official site is anti-bot protected, so the fetch is best-effort and
falls back to the ScraperAPI proxy when `SCRAPERAPI_KEY` is set (paste it
into the project's API Keys tab). Failures are recorded, never fatal.

    fetch → parse (SYMBOL, SERIES, TTL_TRADED_QTY, DELIV_QTY, DELIV_PER)
          → persist (delivery_data table) → delivery_signal per symbol
"""
from __future__ import annotations

import csv
import io
import os
from datetime import date, timedelta
from typing import Optional

import requests

from . import db

NSE_ARCHIVE_URL = "https://nsearchives.nseindia.com/archives/equities/mkt/MA{date}.csv"
SCRAPERAPI_PROXY = "http://scraperapi:{key}@proxy-server.scraperapi.com:8001"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    ),
    "Accept": "text/csv,text/plain,*/*",
    "Referer": "https://www.nseindia.com/",
}
RECENT_DAYS = 5
BASE_DAYS = 20


def _ddmmyyyy(day: str) -> str:
    y, m, d = day.split("-")
    return f"{d}{m}{y}"


def _fetch_csv(url: str, timeout: int = 20) -> Optional[str]:
    """Direct NSE archive first; ScraperAPI proxy fallback when the key is set."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code == 200 and r.text.strip():
            return r.text
    except Exception:
        pass
    key = os.getenv("SCRAPERAPI_KEY")
    if not key:
        return None
    try:
        proxies = {
            "http": SCRAPERAPI_PROXY.format(key=key),
            "https": SCRAPERAPI_PROXY.format(key=key),
        }
        r = requests.get(url, headers=HEADERS, proxies=proxies, timeout=timeout + 10)
        if r.status_code == 200 and r.text.strip():
            return r.text
    except Exception:
        return None
    return None


def _parse_delivery_csv(text: str, day: str) -> list[dict]:
    """Parse the MA report: SYMBOL / SERIES / TTL_TRADED_QTY / DELIV_QTY / DELIV_PER.

    Only EQ series rows are kept; column names are matched tolerantly so the
    parser survives NSE's occasional header tweaks.
    """
    out: list[dict] = []
    try:
        reader = csv.DictReader(io.StringIO(text))
    except Exception:
        return out
    if not reader.fieldnames:
        return out
    fn = {str(f).strip().upper(): f for f in reader.fieldnames}
    col_sym = fn.get("SYMBOL")
    col_series = fn.get("SERIES")
    col_vol = fn.get("TTL_TRADED_QTY") or fn.get("TRADED_QTY")
    col_dq = fn.get("DELIV_QTY")
    col_dp = fn.get("DELIV_PER") or fn.get("DELIV_PER%") or fn.get("DELIV_PER %")
    if not (col_sym and col_vol and col_dq):
        return out
    for row in reader:
        if col_series:
            series = str(row.get(col_series, "")).strip().upper()
            if series != "EQ":
                continue
        try:
            volume = float(row[col_vol])
            dq = float(row[col_dq])
        except (TypeError, ValueError):
            continue
        dp = None
        if col_dp:
            try:
                dp = float(row[col_dp])
            except (TypeError, ValueError):
                pass
        if dp is None and volume > 0:
            dp = dq / volume * 100.0
        sym = str(row[col_sym]).strip().upper()
        if not sym or sym in ("SYMBOL",):
            continue
        out.append(
            {
                "symbol": sym + ".NS",
                "date": day,
                "volume": volume,
                "delivery_qty": dq,
                "delivery_pct": round(dp, 2) if dp is not None else None,
                "source": "nse",
            }
        )
    return out


def refresh_delivery(max_days_back: int = 5, timeout: int = 20) -> dict:
    """Fetch + persist the most recent delivery session (best-effort).

    Walks back up to `max_days_back` calendar days looking for the latest
    published report, skipping dates already stored. Anti-bot blocks are
    recorded in failed_symbols and reported — never fatal.
    """
    latest = db.latest_delivery_date()
    for offset in range(1, max_days_back + 1):
        day = (date.today() - timedelta(days=offset)).isoformat()
        if latest and day <= latest:
            continue
        url = NSE_ARCHIVE_URL.format(date=_ddmmyyyy(day))
        text = _fetch_csv(url, timeout=timeout)
        if not text:
            db.record_failed_symbol(f"DELIVERY-{day}", "nse_delivery", "no content / blocked")
            continue
        rows = _parse_delivery_csv(text, day)
        if rows:
            n = db.upsert_delivery_rows(rows)
            return {"status": "ok", "date": day, "rows": n, "source": "nse"}
    return {
        "status": "empty",
        "message": (
            "No delivery data fetched in the last 5 sessions — the NSE archive is "
            "anti-bot protected. Paste SCRAPERAPI_KEY into API Keys to route via ScraperAPI."
        ),
    }


def delivery_signal(symbol: str) -> Optional[dict]:
    """Accumulation signal: recent delivery % vs its own base (0-100 score).

    Base 50; every +1pp of delivery-% lift over the trailing base adds 2.5
    points. Heavy, sustained delivery (e.g. 60%+ vs a 45% base) reads as
    institutional accumulation; falling delivery reads as distribution.
    """
    rows = db.load_delivery(symbol, limit=BASE_DAYS + RECENT_DAYS)
    if len(rows) < RECENT_DAYS + 3:
        return None
    recent = [r["delivery_pct"] for r in rows[:RECENT_DAYS] if r["delivery_pct"] is not None]
    base = [r["delivery_pct"] for r in rows[: RECENT_DAYS + BASE_DAYS] if r["delivery_pct"] is not None]
    if len(recent) < 3 or len(base) < 6:
        return None
    ravg = sum(recent) / len(recent)
    bavg = sum(base) / len(base)
    delta = ravg - bavg
    score = max(0.0, min(100.0, 50.0 + delta * 2.5))
    return {
        "symbol": symbol,
        "recent_avg_pct": round(ravg, 1),
        "base_avg_pct": round(bavg, 1),
        "delta_pct": round(delta, 1),
        "delivery_score": round(score, 1),
    }


def delivery_accumulators(limit: int = 10) -> list[dict]:
    """Top names by delivery signal for the latest stored session. With thin
    history (signals need ~5 days of data) it falls back to the highest
    delivery % of the latest day, honestly marked with delivery_score=None.
    """
    latest = db.latest_delivery_date()
    if not latest:
        return []
    rows = [r for r in db.load_delivery(limit=2500) if r["date"] == latest]
    ranked: list[dict] = []
    for r in rows:
        sig = delivery_signal(r["symbol"])
        if sig:
            ranked.append({**sig, "date": latest, "delivery_pct": r["delivery_pct"]})
    ranked.sort(key=lambda x: x["delivery_score"] or 0, reverse=True)
    if not ranked:
        for r in sorted(rows, key=lambda x: x["delivery_pct"] or 0, reverse=True)[:limit]:
            ranked.append(
                {
                    "symbol": r["symbol"],
                    "date": latest,
                    "delivery_pct": r["delivery_pct"],
                    "delivery_score": None,
                }
            )
    return ranked[:limit]
