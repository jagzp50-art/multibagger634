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
    tier       TEXT DEFAULT 'core',
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
    last_updated    TEXT,
    data_confidence REAL,
    roe_stability   REAL,
    roce_stability  REAL,
    profit_stability REAL,
    sales_stability REAL,
    margin_stability REAL,
    fcf_stability   REAL,
    sales_cagr_5y   REAL,
    profit_cagr_5y  REAL,
    fcf_cagr_5y     REAL,
    debt_trend      REAL,
    revenue_accel_annual REAL,
    earnings_vol    REAL,
    sales_vol       REAL,
    cfo_pat_ratio   REAL,
    cfo_growth      REAL,
    accrual_ratio   REAL
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
    rs_rank  REAL,
    rs_1m    REAL,
    rs_3m    REAL,
    rs_6m    REAL,
    rs_12m   REAL,
    rs_boost REAL,
    accumulation REAL,
    pos_score REAL,
    opp_score REAL,
    sector_boost REAL,
    trend_ok INTEGER,
    institutional_quality REAL,
    revision_score REAL,
    compounder_score REAL,
    reinvestment_score REAL,
    vol          REAL,
    max_dd       REAL,
    data_confidence REAL,
    factor_contributions TEXT,
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
    quality    REAL,
    growth     REAL,
    momentum   REAL,
    valuation  REAL,
    risk       REAL,
    rs_rank    REAL,
    sector_boost REAL,
    opp_score  REAL,
    data_confidence REAL,
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
CREATE TABLE IF NOT EXISTS financial_history (
    symbol     TEXT NOT NULL,
    year       INTEGER NOT NULL,
    revenue    REAL,
    net_income REAL,
    fcf        REAL,
    total_debt REAL,
    equity     REAL,
    roe        REAL,
    net_margin REAL,
    fcf_margin REAL,
    debt_equity REAL,
    PRIMARY KEY (symbol, year)
);
CREATE INDEX IF NOT EXISTS idx_financial_history_sym ON financial_history (symbol, year);
CREATE TABLE IF NOT EXISTS watchlist_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_date  TEXT NOT NULL,
    symbol     TEXT NOT NULL,
    event      TEXT NOT NULL,
    detail     TEXT,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_watchlist_events ON watchlist_events (scan_date, event);
CREATE TABLE IF NOT EXISTS alpha_tracking (
    scan_date     TEXT NOT NULL,
    horizon_days  INTEGER NOT NULL,
    avg_return_pct REAL,
    median_return_pct REAL,
    hit_rate_pct  REAL,
    n             INTEGER,
    regime        TEXT,
    created_at    TEXT,
    PRIMARY KEY (scan_date, horizon_days)
);
CREATE TABLE IF NOT EXISTS factor_ic (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_date   TEXT NOT NULL,
    factor      TEXT NOT NULL,
    horizon_days INTEGER NOT NULL,
    ic          REAL,
    regime      TEXT,
    n           INTEGER,
    created_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_factor_ic ON factor_ic (scan_date, factor);
CREATE TABLE IF NOT EXISTS universe_history (
    snapshot_date TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    name          TEXT,
    sector        TEXT,
    PRIMARY KEY (snapshot_date, symbol)
);
CREATE INDEX IF NOT EXISTS idx_universe_history ON universe_history (snapshot_date);
CREATE TABLE IF NOT EXISTS fundamentals_history (
    scan_date      TEXT NOT NULL,
    symbol         TEXT NOT NULL,
    roe            REAL,
    roce           REAL,
    debt_equity    REAL,
    sales_growth   REAL,
    profit_growth  REAL,
    pe             REAL,
    pb             REAL,
    fcf_margin     REAL,
    eps_growth     REAL,
    eps_accel      REAL,
    margin_expansion REAL,
    rev_accel      REAL,
    pat_accel      REAL,
    cfo_pat_ratio  REAL,
    cfo_growth     REAL,
    accrual_ratio  REAL,
    data_confidence REAL,
    PRIMARY KEY (scan_date, symbol)
);
CREATE INDEX IF NOT EXISTS idx_fundamentals_history ON fundamentals_history (symbol, scan_date);
"""


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=5, check_same_thread=False)
    conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def _migrate_columns(conn: sqlite3.Connection, table: str, columns: Iterable[str]) -> None:
    """Add any missing columns (all REAL/TEXT — SQLite is dynamically typed)."""
    existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    for col in columns:
        if col not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} REAL")


def init_db(universe: Optional[Iterable[dict]] = None) -> None:
    """Create tables (idempotent) and seed the universe if empty."""
    conn = get_connection()
    try:
        conn.executescript(SCHEMA)
        # Lightweight migration: add columns added after the table existed.
        # (stocks.tier is TEXT with a default; the generic helper adds REAL.)
        stock_cols = {r[1] for r in conn.execute("PRAGMA table_info(stocks)").fetchall()}
        if "tier" not in stock_cols:
            conn.execute("ALTER TABLE stocks ADD COLUMN tier TEXT DEFAULT 'core'")
        _migrate_columns(
            conn,
            "fundamentals",
            [
                "margin_expansion",
                "rev_accel",
                "pat_accel",
                "data_confidence",
                "roe_stability",
                "roce_stability",
                "profit_stability",
                "sales_stability",
                "margin_stability",
                "fcf_stability",
                "sales_cagr_5y",
                "profit_cagr_5y",
                "fcf_cagr_5y",
                "debt_trend",
                "revenue_accel_annual",
                "earnings_vol",
                "sales_vol",
                "cfo_pat_ratio",
                "cfo_growth",
                "accrual_ratio",
            ],
        )
        _migrate_columns(
            conn,
            "scores",
            [
                "mb_checklist",
                "rs_6m",
                "rs_12m",
                "rs_1m",
                "rs_3m",
                "rs_boost",
                "accumulation",
                "pos_score",
                "opp_score",
                "sector_boost",
                "institutional_quality",
                "revision_score",
                "compounder_score",
                "data_confidence",
                "factor_contributions",
                "reinvestment_score",
                "vol",
                "max_dd",
            ],
        )
        _migrate_columns(
            conn,
            "score_history",
            ["quality", "growth", "momentum", "valuation", "risk", "rs_rank", "sector_boost", "opp_score", "data_confidence"],
        )
        conn.commit()
    finally:
        conn.close()
    if universe:
        seed_universe(universe)


def seed_universe(stocks: Iterable[dict], tier: str = "core") -> None:
    conn = get_connection()
    try:
        conn.executemany(
            "INSERT OR IGNORE INTO stocks (symbol, name, sector, in_universe, tier, added_at) "
            "VALUES (:symbol, :name, :sector, 1, :tier, :added_at)",
            [
                {
                    "symbol": s["symbol"],
                    "name": s.get("name", s["symbol"]),
                    "sector": s.get("sector", "Unknown"),
                    "tier": tier,
                    "added_at": date.today().isoformat(),
                }
                for s in stocks
            ],
        )
        conn.commit()
    finally:
        conn.close()


def universe_symbols(tier: Optional[str] = None) -> list[str]:
    conn = get_connection()
    try:
        if tier:
            rows = conn.execute(
                "SELECT symbol FROM stocks WHERE in_universe = 1 AND tier = ? ORDER BY symbol",
                (tier,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT symbol FROM stocks WHERE in_universe = 1 ORDER BY symbol"
            ).fetchall()
        return [r["symbol"] for r in rows]
    finally:
        conn.close()


def universe_tiers() -> dict[str, int]:
    """Membership counts per tier (e.g. {'core': 155, 'discovery': 450})."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT tier, COUNT(*) AS n FROM stocks WHERE in_universe = 1 GROUP BY tier"
        ).fetchall()
        return {r["tier"] or "core": r["n"] for r in rows}
    finally:
        conn.close()


def add_stock(symbol: str, name: str = "", sector: str = "Unknown", tier: str = "core") -> None:
    conn = get_connection()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO stocks (symbol, name, sector, in_universe, tier, added_at) "
            "VALUES (?, ?, ?, 1, ?, ?)",
            (symbol, name, sector, tier, date.today().isoformat()),
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
             rev_accel, pat_accel, promoter_holding, sector, name, last_updated,
             data_confidence, roe_stability, roce_stability, profit_stability,
             sales_stability, margin_stability, fcf_stability, sales_cagr_5y,
             profit_cagr_5y, fcf_cagr_5y, debt_trend, revenue_accel_annual,
             earnings_vol, sales_vol, cfo_pat_ratio, cfo_growth, accrual_ratio)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                _num(f.get("data_confidence")),
                _num(f.get("roe_stability")),
                _num(f.get("roce_stability")),
                _num(f.get("profit_stability")),
                _num(f.get("sales_stability")),
                _num(f.get("margin_stability")),
                _num(f.get("fcf_stability")),
                _num(f.get("sales_cagr_5y")),
                _num(f.get("profit_cagr_5y")),
                _num(f.get("fcf_cagr_5y")),
                _num(f.get("debt_trend")),
                _num(f.get("revenue_accel_annual")),
                _num(f.get("earnings_vol")),
                _num(f.get("sales_vol")),
                _num(f.get("cfo_pat_ratio")),
                _num(f.get("cfo_growth")),
                _num(f.get("accrual_ratio")),
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
             mb_score, mb_bucket, mb_checklist, regime, rank, rs_rank, rs_1m, rs_3m,
             rs_6m, rs_12m, rs_boost, accumulation, pos_score, opp_score, sector_boost,
             trend_ok, institutional_quality, revision_score, compounder_score,
             reinvestment_score, vol, max_dd, data_confidence, factor_contributions, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    _num(r.get("rs_rank")),
                    _num(r.get("rs_1m")),
                    _num(r.get("rs_3m")),
                    _num(r.get("rs_6m")),
                    _num(r.get("rs_12m")),
                    _num(r.get("rs_boost")),
                    _num(r.get("accumulation")),
                    _num(r.get("pos_score")),
                    _num(r.get("opp_score")),
                    _num(r.get("sector_boost")),
                    int(bool(r.get("trend_ok"))),
                    _num(r.get("institutional_quality")),
                    _num(r.get("revision_score")),
                    _num(r.get("compounder_score")),
                    _num(r.get("reinvestment_score")),
                    _num(r.get("vol")),
                    _num(r.get("max_dd")),
                    _num(r.get("data_confidence")),
                    json.dumps(r.get("factor_contributions") or {}),
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
            try:
                d["factor_contributions"] = json.loads(d.get("factor_contributions") or "{}")
            except (TypeError, ValueError):
                d["factor_contributions"] = {}
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
    """Append the current scan's full factor breakdown to score_history."""
    conn = get_connection()
    ts = datetime.now().isoformat(timespec="microseconds")
    try:
        conn.executemany(
            "INSERT INTO score_history "
            "(symbol, scan_date, score, rank, mb_score, mb_bucket, trend_ok, regime, "
            " quality, growth, momentum, valuation, risk, rs_rank, sector_boost, opp_score, "
            " data_confidence, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                    _num(r.get("quality")),
                    _num(r.get("growth")),
                    _num(r.get("momentum")),
                    _num(r.get("valuation")),
                    _num(r.get("risk")),
                    _num(r.get("rs_rank")),
                    _num(r.get("sector_boost")),
                    _num(r.get("opp_score")),
                    _num(r.get("data_confidence")),
                    ts,
                )
                for r in records
            ],
        )
        conn.commit()
        return len(records)
    finally:
        conn.close()


def score_snapshot_times(limit: int = 30) -> list[str]:
    """Distinct scan timestamps (newest first) — one per completed scan."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT DISTINCT created_at FROM score_history ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [r["created_at"] for r in rows]
    finally:
        conn.close()


def score_history_at(created_at: str) -> list[dict]:
    """Full factor rows for one scan snapshot (alpha decay / factor IC inputs)."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT symbol, scan_date, score, rank, mb_score, mb_bucket, trend_ok, regime, quality, "
            "growth, momentum, valuation, risk, rs_rank, sector_boost, opp_score, data_confidence "
            "FROM score_history WHERE created_at = ?",
            (created_at,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def score_history_for(symbol: str, limit: int = 40) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT scan_date, score, rank, mb_score, mb_bucket, trend_ok, regime, "
            "quality, growth, momentum, valuation, risk, rs_rank, sector_boost, "
            "opp_score, data_confidence "
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


# ── Financial history (5y annual statements → stability metrics) ───────────

def upsert_financial_history(symbol: str, rows: Iterable[dict]) -> int:
    """Store per-fiscal-year financials (revenue / net income / FCF / debt / equity)."""
    conn = get_connection()
    n = 0
    try:
        for r in rows:
            conn.execute(
                "INSERT OR REPLACE INTO financial_history "
                "(symbol, year, revenue, net_income, fcf, total_debt, equity, roe, "
                " net_margin, fcf_margin, debt_equity) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    symbol,
                    int(r["year"]),
                    _num(r.get("revenue")),
                    _num(r.get("net_income")),
                    _num(r.get("fcf")),
                    _num(r.get("total_debt")),
                    _num(r.get("equity")),
                    _num(r.get("roe")),
                    _num(r.get("net_margin")),
                    _num(r.get("fcf_margin")),
                    _num(r.get("debt_equity")),
                ),
            )
            n += 1
        conn.commit()
    finally:
        conn.close()
    return n


def load_financial_history(symbol: str) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT year, revenue, net_income, fcf, total_debt, equity, roe, "
            "net_margin, fcf_margin, debt_equity "
            "FROM financial_history WHERE symbol = ? ORDER BY year",
            (symbol,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ── Watchlist intelligence / alpha decay / factor IC ────────────────────────

def save_watchlist_events(events: Iterable[dict], scan_date: str) -> int:
    conn = get_connection()
    ts = datetime.now().isoformat(timespec="seconds")
    try:
        conn.executemany(
            "INSERT INTO watchlist_events (scan_date, symbol, event, detail, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            [(scan_date, e.get("symbol"), e.get("event"), e.get("detail"), ts) for e in events],
        )
        conn.commit()
        return len(events)
    finally:
        conn.close()


def load_watchlist_events(scan_date: Optional[str] = None, limit: int = 60) -> list[dict]:
    conn = get_connection()
    try:
        if scan_date:
            rows = conn.execute(
                "SELECT scan_date, symbol, event, detail, created_at FROM watchlist_events "
                "WHERE scan_date = ? ORDER BY id DESC LIMIT ?",
                (scan_date, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT scan_date, symbol, event, detail, created_at FROM watchlist_events "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def latest_watchlist_scan() -> Optional[str]:
    conn = get_connection()
    try:
        row = conn.execute("SELECT MAX(scan_date) AS d FROM watchlist_events").fetchone()
        return row["d"] if row else None
    finally:
        conn.close()


def save_alpha_rows(rows: Iterable[dict]) -> int:
    """Upsert forward-return aggregates per (snapshot date, horizon)."""
    conn = get_connection()
    try:
        conn.executemany(
            "INSERT OR REPLACE INTO alpha_tracking "
            "(scan_date, horizon_days, avg_return_pct, median_return_pct, hit_rate_pct, n, regime, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    r["scan_date"],
                    int(r["horizon_days"]),
                    _num(r.get("avg_return_pct")),
                    _num(r.get("median_return_pct")),
                    _num(r.get("hit_rate_pct")),
                    r.get("n"),
                    r.get("regime"),
                    datetime.now().isoformat(timespec="seconds"),
                )
                for r in rows
            ],
        )
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def load_alpha_rows(limit_horizons: int = 5) -> list[dict]:
    """Most recent forward-return aggregate per horizon."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT scan_date, horizon_days, avg_return_pct, median_return_pct, hit_rate_pct, n, regime "
            "FROM alpha_tracking ORDER BY scan_date DESC, horizon_days LIMIT ?",
            (limit_horizons * 4,),
        ).fetchall()
        out = [dict(r) for r in rows]
        # one row per horizon: the latest scan_date
        seen: set = set()
        latest = []
        for r in out:
            if r["horizon_days"] not in seen:
                seen.add(r["horizon_days"])
                latest.append(r)
        return latest
    finally:
        conn.close()


def save_factor_ic(rows: Iterable[dict]) -> int:
    conn = get_connection()
    try:
        conn.executemany(
            "INSERT INTO factor_ic (scan_date, factor, horizon_days, ic, regime, n, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    r["scan_date"],
                    r["factor"],
                    int(r.get("horizon_days", 30)),
                    _num(r.get("ic")),
                    r.get("regime"),
                    r.get("n"),
                    datetime.now().isoformat(timespec="seconds"),
                )
                for r in rows
            ],
        )
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def load_factor_ic(limit: int = 200) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT scan_date, factor, horizon_days, ic, regime, n, created_at "
            "FROM factor_ic ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]
    finally:
        conn.close()


# ── Survivorship bias + point-in-time history ──────────────────────────────

def earliest_universe_snapshot() -> Optional[str]:
    """First date we have a universe-membership snapshot for (PIT reference)."""
    conn = get_connection()
    try:
        row = conn.execute("SELECT MIN(snapshot_date) FROM universe_history").fetchone()
        return row[0] if row and row[0] else None
    finally:
        conn.close()


def snapshot_universe(scan_date: str) -> int:
    """Store today's universe membership (survivorship-bias protection)."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT symbol, name, sector FROM stocks WHERE in_universe = 1"
        ).fetchall()
        conn.executemany(
            "INSERT OR REPLACE INTO universe_history (snapshot_date, symbol, name, sector) "
            "VALUES (?, ?, ?, ?)",
            [(scan_date, r["symbol"], r["name"], r["sector"]) for r in rows],
        )
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def universe_history_snapshots(limit: int = 30) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT snapshot_date, COUNT(*) AS members FROM universe_history "
            "GROUP BY snapshot_date ORDER BY snapshot_date DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def snapshot_fundamentals_history(scan_date: str) -> int:
    """Point-in-time copy of the fundamentals table for this scan.

    Backtests and factor research can later read fundamentals *as they were
    known on each scan date* — never today's ROE for a 2021 test.
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT symbol, roe, roce, debt_equity, sales_growth, profit_growth, pe, pb, "
            "fcf_margin, eps_growth, eps_accel, margin_expansion, rev_accel, pat_accel, "
            "cfo_pat_ratio, cfo_growth, accrual_ratio, data_confidence FROM fundamentals"
        ).fetchall()
        conn.executemany(
            "INSERT OR REPLACE INTO fundamentals_history "
            "(scan_date, symbol, roe, roce, debt_equity, sales_growth, profit_growth, pe, pb, "
            "fcf_margin, eps_growth, eps_accel, margin_expansion, rev_accel, pat_accel, "
            "cfo_pat_ratio, cfo_growth, accrual_ratio, data_confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(scan_date, *[r[k] for k in r.keys()]) for r in rows],
        )
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def fundamentals_history_for(symbol: str, limit: int = 20) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM fundamentals_history WHERE symbol = ? ORDER BY scan_date DESC LIMIT ?",
            (symbol, limit),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]
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
