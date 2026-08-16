"""
Sovereign Lite v7 — SQLite data layer (exactly 5 tables).

    stocks        universe membership (symbol, name, sector)
    prices        daily OHLCV per symbol
    fundamentals  latest point-in-time fundamentals per symbol
    scores        latest computed component + composite scores per symbol
    backtests     stored backtest runs (params / equity curve / summary)
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, date
from typing import Any, Iterable, Optional

DB_PATH = "lite.db"
_BUSY_TIMEOUT_MS = 5000

SCHEMA = """
CREATE TABLE IF NOT EXISTS stocks (
    symbol     TEXT PRIMARY KEY,
    name       TEXT,
    sector     TEXT,
    in_universe INTEGER DEFAULT 1,
    added_at   TEXT
);
CREATE TABLE IF NOT EXISTS prices (
    symbol TEXT NOT NULL,
    date   TEXT NOT NULL,
    open   REAL,
    high   REAL,
    low    REAL,
    close  REAL,
    volume INTEGER,
    PRIMARY KEY (symbol, date)
);
CREATE INDEX IF NOT EXISTS idx_prices_date ON prices (date);
CREATE TABLE IF NOT EXISTS fundamentals (
    symbol          TEXT PRIMARY KEY,
    market_cap      REAL,
    roe             REAL,
    roce            REAL,
    debt_equity     REAL,
    sales_growth    REAL,
    profit_growth   REAL,
    pe              REAL,
    pb              REAL,
    fcf_margin      REAL,
    eps_growth      REAL,
    eps_accel       REAL,
    eps_quarters    TEXT,
    promoter_holding REAL,
    sector          TEXT,
    name            TEXT,
    last_updated    TEXT
);
CREATE TABLE IF NOT EXISTS scores (
    symbol   TEXT PRIMARY KEY,
    score    REAL,
    quality  REAL,
    growth   REAL,
    momentum REAL,
    valuation REAL,
    risk     REAL,
    mb_score REAL,
    mb_bucket TEXT,
    mb_checklist TEXT,
    regime   TEXT,
    rank     INTEGER,
    rs_rank  INTEGER,
    rs_6m    INTEGER,
    rs_12m   INTEGER,
    rs_boost REAL,
    accumulation REAL,
    trend_ok INTEGER,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS backtests (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    params     TEXT,
    equity_curve TEXT,
    trades     TEXT,
    summary    TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS quarterly_results (
    symbol     TEXT NOT NULL,
    period_end TEXT NOT NULL,
    quarter    TEXT,
    revenue    REAL,
    net_income REAL,
    net_margin REAL,
    PRIMARY KEY (symbol, period_end)
);
CREATE TABLE IF NOT EXISTS score_history (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol     TEXT NOT NULL,
    scan_date  TEXT NOT NULL,
    score      REAL,
    rank       INTEGER,
    mb_score   REAL,
    mb_bucket  TEXT,
    trend_ok   INTEGER,
    regime     TEXT,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_score_history_sym ON score_history (symbol, scan_date);
CREATE TABLE IF NOT EXISTS mb_candidates (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol     TEXT NOT NULL,
    scan_date  TEXT NOT NULL,
    mb_score   REAL,
    mb_rank    INTEGER,
    mb_bucket  TEXT,
    regime     TEXT,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_mb_candidates_sym ON mb_candidates (symbol, scan_date);
"""


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=5, check_same_thread=False)
    conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(universe: Optional[Iterable[dict]] = None) -> None:
    """Create tables (idempotent) and seed the universe if empty."""
    conn = get_connection()
    try:
        conn.executescript(SCHEMA)
        # Lightweight migration: add columns added after the table existed.
        cols = {r[1] for r in conn.execute("PRAGMA table_info(scores)").fetchall()}
        if "mb_checklist" not in cols:
            conn.execute("ALTER TABLE scores ADD COLUMN mb_checklist TEXT")
        fcols = {r[1] for r in conn.execute("PRAGMA table_info(fundamentals)").fetchall()}
        if "margin_expansion" not in fcols:
            conn.execute("ALTER TABLE fundamentals ADD COLUMN margin_expansion REAL")
        if "rev_accel" not in fcols:
            conn.execute("ALTER TABLE fundamentals ADD COLUMN rev_accel REAL")
        if "pat_accel" not in fcols:
            conn.execute("ALTER TABLE fundamentals ADD COLUMN pat_accel REAL")
        for col in ("rs_6m", "rs_12m", "rs_boost", "accumulation"):
            if col not in cols:
                conn.execute(f"ALTER TABLE scores ADD COLUMN {col} REAL")
        conn.commit()
    finally:
        conn.close()
    if universe:
        seed_universe(universe)


def seed_universe(stocks: Iterable[dict]) -> None:
    conn = get_connection()
    try:
        conn.executemany(
            "INSERT OR IGNORE INTO stocks (symbol, name, sector, in_universe, added_at) "
            "VALUES (:symbol, :name, :sector, 1, :added_at)",
            [
                {
                    "symbol": s["symbol"],
                    "name": s.get("name", s["symbol"]),
                    "sector": s.get("sector", "Unknown"),
                    "added_at": date.today().isoformat(),
                }
                for s in stocks
            ],
        )
        conn.commit()
    finally:
        conn.close()


def universe_symbols() -> list[str]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT symbol FROM stocks WHERE in_universe = 1 ORDER BY symbol"
        ).fetchall()
        return [r["symbol"] for r in rows]
    finally:
        conn.close()


def add_stock(symbol: str, name: str = "", sector: str = "Unknown") -> None:
    conn = get_connection()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO stocks (symbol, name, sector, in_universe, added_at) "
            "VALUES (?, ?, ?, 1, ?)",
            (symbol, name, sector, date.today().isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


# ── Prices ──────────────────────────────────────────────────────────────────

def upsert_prices(symbol: str, rows: Iterable[dict]) -> int:
    conn = get_connection()
    n = 0
    try:
        for r in rows:
            conn.execute(
                "INSERT OR REPLACE INTO prices (symbol, date, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    symbol,
                    r["date"],
                    r.get("open"),
                    r.get("high"),
                    r.get("low"),
                    r.get("close"),
                    r.get("volume"),
                ),
            )
            n += 1
        conn.commit()
    finally:
        conn.close()
    return n


def load_prices(symbol: str, start: Optional[str] = None) -> list[dict]:
    conn = get_connection()
    try:
        if start:
            rows = conn.execute(
                "SELECT date, open, high, low, close, volume FROM prices "
                "WHERE symbol = ? AND date >= ? ORDER BY date",
                (symbol, start),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT date, open, high, low, close, volume FROM prices "
                "WHERE symbol = ? ORDER BY date",
                (symbol,),
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def latest_price(symbol: str) -> Optional[dict]:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT date, open, high, low, close, volume FROM prices "
            "WHERE symbol = ? ORDER BY date DESC LIMIT 1",
            (symbol,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ── Fundamentals ────────────────────────────────────────────────────────────

def upsert_fundamentals(f: dict) -> None:
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO fundamentals
            (symbol, market_cap, roe, roce, debt_equity, sales_growth, profit_growth,
             pe, pb, fcf_margin, eps_growth, eps_accel, eps_quarters, margin_expansion,
             rev_accel, pat_accel, promoter_holding, sector, name, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f.get("symbol"),
                _num(f.get("market_cap")),
                _num(f.get("roe")),
                _num(f.get("roce")),
                _num(f.get("debt_equity")),
                _num(f.get("sales_growth")),
                _num(f.get("profit_growth")),
                _num(f.get("pe")),
                _num(f.get("pb")),
                _num(f.get("fcf_margin")),
                _num(f.get("eps_growth")),
                _num(f.get("eps_accel")),
                json.dumps(f.get("eps_quarters") or []),
                _num(f.get("margin_expansion")),
                _num(f.get("rev_accel")),
                _num(f.get("pat_accel")),
                _num(f.get("promoter_holding")),
                f.get("sector"),
                f.get("name"),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def load_fundamentals() -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM fundamentals").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["eps_quarters"] = json.loads(d.get("eps_quarters") or "[]")
            except (TypeError, ValueError):
                d["eps_quarters"] = []
            out.append(d)
        return out
    finally:
        conn.close()


def fundamentals_for(symbol: str) -> Optional[dict]:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM fundamentals WHERE symbol = ?", (symbol,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ── Scores ──────────────────────────────────────────────────────────────────

def upsert_scores(records: Iterable[dict]) -> None:
    conn = get_connection()
    try:
        conn.executemany(
            """
            INSERT OR REPLACE INTO scores
            (symbol, score, quality, growth, momentum, valuation, risk,
             mb_score, mb_bucket, mb_checklist, regime, rank, rs_rank, rs_6m, rs_12m,
             rs_boost, accumulation, trend_ok, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r["symbol"],
                    _num(r.get("score")),
                    _num(r.get("quality")),
                    _num(r.get("growth")),
                    _num(r.get("momentum")),
                    _num(r.get("valuation")),
                    _num(r.get("risk")),
                    _num(r.get("mb_score")),
                    r.get("mb_bucket"),
                    json.dumps(r.get("mb_checklist") or []),
                    r.get("regime"),
                    r.get("rank"),
                    r.get("rs_rank"),
                    r.get("rs_6m"),
                    r.get("rs_12m"),
                    _num(r.get("rs_boost")),
                    _num(r.get("accumulation")),
                    int(bool(r.get("trend_ok"))),
                    datetime.now().isoformat(timespec="seconds"),
                )
                for r in records
            ],
        )
        conn.commit()
    finally:
        conn.close()


def load_scores() -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM scores").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["mb_checklist"] = json.loads(d.get("mb_checklist") or "[]")
            except (TypeError, ValueError):
                d["mb_checklist"] = []
            out.append(d)
        return out
    finally:
        conn.close()


def latest_scan_regime() -> Optional[str]:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT regime, updated_at FROM scores ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ── Score history (trends) ─────────────────────────────────────────────────

def latest_score_snapshot() -> dict[str, dict]:
    """Most recent completed scan snapshot: {symbol: {score, rank, mb_score}}."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT MAX(created_at) AS ts FROM score_history"
        ).fetchone()
        ts = row["ts"] if row else None
        if not ts:
            return {}
        rows = conn.execute(
            "SELECT symbol, score, rank, mb_score FROM score_history WHERE created_at = ?",
            (ts,),
        ).fetchall()
        return {r["symbol"]: dict(r) for r in rows}
    finally:
        conn.close()


def previous_score_snapshot() -> dict[str, dict]:
    """Second-newest scan snapshot — the comparison baseline for trends."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT DISTINCT created_at FROM score_history ORDER BY created_at DESC LIMIT 2"
        ).fetchall()
        if len(rows) < 2:
            return {}
        ts = rows[1]["created_at"]
        rows2 = conn.execute(
            "SELECT symbol, score, rank, mb_score FROM score_history WHERE created_at = ?",
            (ts,),
        ).fetchall()
        return {r["symbol"]: dict(r) for r in rows2}
    finally:
        conn.close()


def snapshot_scores(records: Iterable[dict], scan_date: str, regime: str) -> int:
    """Append the current scan's scores/ranks to score_history."""
    conn = get_connection()
    ts = datetime.now().isoformat(timespec="microseconds")
    try:
        conn.executemany(
            "INSERT INTO score_history "
            "(symbol, scan_date, score, rank, mb_score, mb_bucket, trend_ok, regime, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    r["symbol"],
                    scan_date,
                    _num(r.get("score")),
                    r.get("rank"),
                    _num(r.get("mb_score")),
                    r.get("mb_bucket"),
                    int(bool(r.get("trend_ok"))),
                    regime,
                    ts,
                )
                for r in records
            ],
        )
        conn.commit()
        return len(records)
    finally:
        conn.close()


def score_history_for(symbol: str, limit: int = 40) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT scan_date, score, rank, mb_score, mb_bucket, trend_ok, regime "
            "FROM score_history WHERE symbol = ? ORDER BY id DESC LIMIT ?",
            (symbol, limit),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]
    finally:
        conn.close()


# ── Multibagger candidates (tracking) ─────────────────────────────────────

def snapshot_mb_candidates(records: Iterable[dict], scan_date: str, regime: str) -> int:
    """Append the current scan's MB scores/ranks to mb_candidates."""
    conn = get_connection()
    ts = datetime.now().isoformat(timespec="microseconds")
    try:
        conn.executemany(
            "INSERT INTO mb_candidates "
            "(symbol, scan_date, mb_score, mb_rank, mb_bucket, regime, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    r["symbol"],
                    scan_date,
                    _num(r.get("mb_score")),
                    r.get("mb_rank"),
                    r.get("mb_bucket"),
                    regime,
                    ts,
                )
                for r in records
            ],
        )
        conn.commit()
        return len(records)
    finally:
        conn.close()


def _mb_snapshot_at(ts: Optional[str]) -> dict[str, dict]:
    if not ts:
        return {}
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT symbol, mb_score, mb_rank, mb_bucket FROM mb_candidates WHERE created_at = ?",
            (ts,),
        ).fetchall()
        return {r["symbol"]: dict(r) for r in rows}
    finally:
        conn.close()


def latest_mb_candidates() -> dict[str, dict]:
    """Most recent MB snapshot: {symbol: {mb_score, mb_rank, mb_bucket}}."""
    conn = get_connection()
    try:
        row = conn.execute("SELECT MAX(created_at) AS ts FROM mb_candidates").fetchone()
        return _mb_snapshot_at(row["ts"] if row else None)
    finally:
        conn.close()


def previous_mb_candidates() -> dict[str, dict]:
    """Second-newest MB snapshot — the baseline for MB trend deltas."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT DISTINCT created_at FROM mb_candidates ORDER BY created_at DESC LIMIT 2"
        ).fetchall()
        if len(rows) < 2:
            return {}
        return _mb_snapshot_at(rows[1]["created_at"])
    finally:
        conn.close()


def mb_candidates_history(symbol: str, limit: int = 40) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT scan_date, mb_score, mb_rank, mb_bucket, regime "
            "FROM mb_candidates WHERE symbol = ? ORDER BY id DESC LIMIT ?",
            (symbol, limit),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]
    finally:
        conn.close()


# ── Quarterly results ───────────────────────────────────────────────────────

def upsert_quarterly_results(symbol: str, rows: Iterable[dict]) -> int:
    conn = get_connection()
    n = 0
    try:
        for r in rows:
            conn.execute(
                "INSERT OR REPLACE INTO quarterly_results "
                "(symbol, period_end, quarter, revenue, net_income, net_margin) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    symbol,
                    r["period_end"],
                    r.get("quarter"),
                    _num(r.get("revenue")),
                    _num(r.get("net_income")),
                    _num(r.get("net_margin")),
                ),
            )
            n += 1
        conn.commit()
    finally:
        conn.close()
    return n


def load_quarterly_results(symbol: str, limit: int = 12) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT period_end, quarter, revenue, net_income, net_margin "
            "FROM quarterly_results WHERE symbol = ? ORDER BY period_end DESC LIMIT ?",
            (symbol, limit),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]
    finally:
        conn.close()


# ── Backtests ───────────────────────────────────────────────────────────────

def save_backtest(params: dict, equity_curve: list, trades: list, summary: dict) -> int:
    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO backtests (params, equity_curve, trades, summary, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                json.dumps(params, default=str),
                json.dumps(equity_curve, default=str),
                json.dumps(trades, default=str),
                json.dumps(summary, default=str),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def load_backtests(limit: int = 10) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM backtests ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            for k in ("params", "equity_curve", "trades", "summary"):
                try:
                    d[k] = json.loads(d[k] or "{}")
                except (TypeError, ValueError):
                    d[k] = {}
            out.append(d)
        return out
    finally:
        conn.close()


def _num(v: Any) -> Optional[float]:
    """Coerce to float, mapping NaN/inf/None to None (SQLite-safe)."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):  # NaN check
        return None
    return f
