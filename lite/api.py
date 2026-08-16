"""
Sovereign Lite v7 — FastAPI application.

Serves the 5-screen dashboard and a small JSON API. All heavy work (scan,
backtest) runs synchronously in FastAPI's threadpool — fine for one user.
"""
from __future__ import annotations

import math
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from . import backtest as bt
from . import data, db, indicators, regime as regime_mod, scoring
from .pipeline import run_scan
from .universe import default_universe

WEB_DIR = Path(__file__).resolve().parent / "web"

_regime_cache: dict = {"payload": None, "fetched_at": 0.0}
_regime_lock = threading.Lock()
REGIME_TTL_SECONDS = 15 * 60

_scan_lock = threading.Lock()


def create_app() -> FastAPI:
    app = FastAPI(title="Sovereign Lite v7", version="7.0.0")

    db.init_db(default_universe())

    # ── Pages ────────────────────────────────────────────────────────────────

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": "7.0.0", "engine": "sovereign-lite"}

    # ── Regime ───────────────────────────────────────────────────────────────

    def _fresh_regime() -> dict:
        with _regime_lock:
            if _regime_cache["payload"] and time.time() - _regime_cache["fetched_at"] < REGIME_TTL_SECONDS:
                return _regime_cache["payload"]
            nifty = data.fetch_benchmark("^NSEI")
            vix = data.fetch_benchmark("^INDIAVIX")
            nifty_adx = (
                indicators.adx(nifty["High"], nifty["Low"], nifty["Close"])
                if nifty is not None and not nifty.empty
                else None
            )
            payload = regime_mod.detect_regime(nifty, vix, nifty_adx)
            payload["timestamp"] = datetime.now().isoformat(timespec="seconds")
            payload.pop("_nifty_close", None)
            _regime_cache["payload"] = _json_safe(payload)
            _regime_cache["fetched_at"] = time.time()
            return _regime_cache["payload"]

    @app.get("/api/regime")
    def regime_endpoint():
        try:
            return _fresh_regime()
        except Exception as exc:
            cached = _regime_cache["payload"]
            if cached:
                return {**cached, "stale": True, "error": str(exc)}
            return {"error": str(exc)}

    # ── Dashboard ────────────────────────────────────────────────────────────

    @app.get("/api/dashboard")
    def dashboard():
        regime = _fresh_regime()
        scores = db.load_scores()
        fundas = {f["symbol"]: f for f in db.load_fundamentals()}
        top = _with_prev(_merge(scores, fundas))
        top.sort(key=lambda r: r.get("score") or 0, reverse=True)
        picks = []
        for r in top[:5]:
            sym = r["symbol"]
            px = db.latest_price(sym)
            last = db.load_prices(sym)
            change_pct = None
            if len(last) >= 2:
                prev = last[-2].get("close")
                cur = last[-1].get("close")
                if prev and cur:
                    change_pct = round((cur / prev - 1) * 100, 2)
            picks.append(
                {
                    "symbol": sym,
                    "name": r.get("name") or sym,
                    "score": r.get("score"),
                    "score_delta": r.get("score_delta"),
                    "rank_delta": r.get("rank_delta"),
                    "mb_bucket": r.get("mb_bucket"),
                    "price": px.get("close") if px else None,
                    "change_pct": change_pct,
                    "sector": r.get("sector"),
                }
            )
        last_scan = db.latest_scan_regime()
        return _json_safe(
            {
                "regime": regime,
                "allocation": regime.get("allocation"),
                "top_picks": picks,
                "universe_size": len(db.universe_symbols()),
                "scored": len(scores),
                "last_scan_at": last_scan.get("updated_at") if last_scan else None,
                "last_scan_regime": last_scan.get("regime") if last_scan else None,
            }
        )

    # ── Screener ─────────────────────────────────────────────────────────────

    @app.get("/api/scores")
    def screener(
        min_score: float = 0,
        min_roe: float = 0,
        min_roce: float = 0,
        min_growth: float = 0,
        max_pe: Optional[float] = None,
        bucket: Optional[str] = None,
        sort: str = "score",
        limit: int = 500,
    ):
        scores = db.load_scores()
        fundas = {f["symbol"]: f for f in db.load_fundamentals()}
        rows = _merge(scores, fundas)
        out = []
        for r in rows:
            if (r.get("score") or 0) < min_score:
                continue
            if (r.get("roe") or 0) < min_roe:
                continue
            if (r.get("roce") or 0) < min_roce:
                continue
            if (r.get("growth_metric") or r.get("sales_growth") or 0) < min_growth:
                continue
            if max_pe is not None and (r.get("pe") or 0) > max_pe:
                continue
            if bucket and (r.get("mb_bucket") or "") != bucket.upper():
                continue
            out.append(r)
        key = {"score": "score", "roe": "roe", "growth": "sales_growth", "pe": "pe"}.get(sort, "score")
        out.sort(key=lambda r: (r.get(key) if isinstance(r.get(key), (int, float)) else -1), reverse=True)
        return _json_safe(_with_prev(out)[: max(1, min(limit, 1000))])

    # ── Elite picks + multibagger detector ───────────────────────────────────

    @app.get("/api/picks")
    def elite_picks(limit: int = 20):
        scores = db.load_scores()
        fundas = {f["symbol"]: f for f in db.load_fundamentals()}
        rows = _with_prev(_merge(scores, fundas))
        rows.sort(key=lambda r: r.get("score") or 0, reverse=True)
        return _json_safe(rows[: max(1, min(limit, 200))])

    @app.get("/api/multibaggers")
    def multibaggers():
        scores = db.load_scores()
        fundas = {f["symbol"]: f for f in db.load_fundamentals()}
        rows = _merge(scores, fundas)
        rows.sort(key=lambda r: r.get("mb_score") or 0, reverse=True)
        return _json_safe([r for r in rows if r.get("mb_checklist") is not None][:100])

    # ── Score history + quarterly detail ─────────────────────────────────────

    @app.get("/api/score-history/{symbol}")
    def score_history(symbol: str, limit: int = 40):
        return _json_safe(db.score_history_for(symbol, limit=max(1, min(limit, 200))))

    @app.get("/api/quarterly/{symbol}")
    def quarterly(symbol: str):
        return _json_safe(db.load_quarterly_results(symbol, limit=12))

    # ── Scan ─────────────────────────────────────────────────────────────────

    @app.post("/api/scan")
    def scan(force: bool = False):
        if not _scan_lock.acquire(blocking=False):
            return {"status": "running", "message": "A scan is already in progress."}
        try:
            return _json_safe(run_scan(force_fundamentals=force))
        finally:
            _scan_lock.release()

    # ── Backtest ─────────────────────────────────────────────────────────────

    @app.post("/api/backtest")
    def run_backtest_endpoint(params: Optional[dict] = None):
        frames = {}
        for sym in db.universe_symbols():
            rows = db.load_prices(sym)
            if rows:
                frames[sym] = indicators.to_dataframe(rows)
        result = bt.run_backtest(frames, params or {})
        if not result.get("trades") and not result.get("equity_curve"):
            raise HTTPException(status_code=422, detail=result["summary"].get("error", "No data"))
        saved_id = db.save_backtest(result["params"], result["equity_curve"], result["trades"], result["summary"])
        return _json_safe({**result, "id": saved_id})

    @app.get("/api/backtests")
    def backtests(limit: int = 10):
        return _json_safe(db.load_backtests(limit=limit))

    # ── Portfolio helpers ────────────────────────────────────────────────────

    @app.get("/api/quote/{symbol}")
    def quote(symbol: str):
        live = data.fetch_quick_quote(symbol)
        if live:
            return _json_safe(live)
        px = db.latest_price(symbol)
        if px:
            return _json_safe({"symbol": symbol, "price": px["close"], "change": None, "change_pct": None, "date": px["date"], "cached": True})
        return {"symbol": symbol, "error": "No data for symbol"}

    @app.get("/api/universe")
    def universe():
        return _json_safe(
            [
                {"symbol": s, "name": s.replace(".NS", ""), "sector": "Large Cap" if s in _LARGE else "Mid/Small Cap"}
                for s in db.universe_symbols()
            ]
        )

    @app.post("/api/universe")
    def add_to_universe(body: dict):
        symbol = str(body.get("symbol", "")).strip().upper()
        if not symbol:
            raise HTTPException(status_code=422, detail="symbol required")
        if not symbol.endswith(".NS") and not symbol.endswith(".BO"):
            symbol += ".NS"
        db.add_stock(symbol, body.get("name", ""), body.get("sector", "Custom"))
        return {"status": "ok", "symbol": symbol}

    return app


# ── Helpers ─────────────────────────────────────────────────────────────────

_LARGE = {s["symbol"] for s in default_universe() if s["sector"] == "Large Cap"}


def _merge(scores: list[dict], fundas: dict[str, dict]) -> list[dict]:
    out = []
    for s in scores:
        row = dict(s)
        f = fundas.get(s["symbol"], {})
        row.update(f)
        row["growth_metric"] = f.get("sales_growth")
        out.append(row)
    return out


def _with_prev(rows: list[dict]) -> list[dict]:
    """Attach previous-snapshot deltas (score/rank vs last scan) to each row."""
    prev = db.previous_score_snapshot()
    for r in rows:
        old = prev.get(r.get("symbol", "")) or {}
        cur_score = r.get("score")
        prev_score = old.get("score")
        cur_rank = r.get("rank")
        prev_rank = old.get("rank")
        r["prev_score"] = prev_score
        r["prev_rank"] = prev_rank
        r["score_delta"] = (
            round(cur_score - prev_score, 1)
            if cur_score is not None and prev_score is not None
            else None
        )
        # Positive = moved up the ranking (lower number is better)
        r["rank_delta"] = (
            prev_rank - cur_rank if cur_rank is not None and prev_rank is not None else None
        )
    return rows


def _json_safe(obj):
    """Recursively convert NaN/inf and non-JSON types to JSON-safe values."""
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float)):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
        return obj
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    if hasattr(obj, "isoformat"):  # date
        return str(obj)
    try:
        float(obj)
        return _json_safe(float(obj))
    except (TypeError, ValueError):
        return str(obj)
