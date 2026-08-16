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

import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from . import alpha
from . import backtest as bt
from . import breadth, data, db, discovery, indicators, portfolio, regime as regime_mod, rotation, scoring, watchlist
from .pipeline import run_scan
from .universe import default_universe

WEB_DIR = Path(__file__).resolve().parent / "web"

_regime_cache: dict = {"payload": None, "fetched_at": 0.0}
_regime_lock = threading.Lock()
REGIME_TTL_SECONDS = 15 * 60

_breadth_cache: dict = {"payload": None, "fetched_at": 0.0}
_breadth_lock = threading.Lock()
BREADTH_TTL_SECONDS = 5 * 60

_scan_lock = threading.Lock()


def _fresh_breadth() -> dict:
    with _breadth_lock:
        if _breadth_cache["payload"] and time.time() - _breadth_cache["fetched_at"] < BREADTH_TTL_SECONDS:
            return _breadth_cache["payload"]
        frames = {}
        for sym in db.universe_symbols():
            rows = db.load_prices(sym)
            if rows:
                frames[sym] = indicators.to_dataframe(rows)
        _breadth_cache["payload"] = _json_safe(breadth.compute_breadth(frames))
        _breadth_cache["fetched_at"] = time.time()
        return _breadth_cache["payload"]


def _recent_watchlist(limit: int = 10) -> list[dict]:
    latest = db.latest_watchlist_scan()
    if not latest:
        return []
    return _json_safe(db.load_watchlist_events(latest, limit=limit))


def create_app() -> FastAPI:
    app = FastAPI(title="Sovereign Lite v11", version="11.0.0")

    db.init_db(default_universe())

    # ── Pages ────────────────────────────────────────────────────────────────

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": "8.0.0", "engine": "sovereign-lite"}

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
        try:
            watch_events = _recent_watchlist(limit=8)
        except Exception:
            watch_events = []
        try:
            risk_rows = portfolio.attach_position_scores(_merge(scores, fundas))
            risk_equity = float(regime.get("allocation", {}).get("equity", 60))
            risk_alloc = portfolio.build_allocation(risk_rows, risk_equity, top_n=8)
            risk_records = {r["symbol"]: r for r in risk_rows}
            portfolio_risk = portfolio.portfolio_risk(risk_alloc, risk_records, risk_equity)
        except Exception:
            portfolio_risk = {"n": 0, "risk_grade": None}
        return _json_safe(
            {
                "regime": regime,
                "allocation": regime.get("allocation"),
                "breadth": _fresh_breadth(),
                "watchlist": watch_events,
                "top_picks": picks,
                "universe_size": len(db.universe_symbols()),
                "universe_tiers": db.universe_tiers(),
                "scored": len(scores),
                "portfolio_risk": portfolio_risk,
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
        min_conf: float = 0,
        bucket: Optional[str] = None,
        sort: str = "opp",
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
            if (r.get("data_confidence") or 0) < min_conf:
                continue
            if bucket and (r.get("mb_bucket") or "") != bucket.upper():
                continue
            out.append(r)
        key = {"opp": "opp_score", "score": "score", "roe": "roe", "growth": "sales_growth", "pe": "pe"}.get(sort, "opp_score")
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
        rows = _with_mb_prev(_merge(scores, fundas))
        rows.sort(key=lambda r: r.get("mb_score") or 0, reverse=True)
        return _json_safe([r for r in rows if r.get("mb_checklist") is not None][:100])

    @app.get("/api/mb-candidates")
    def mb_candidates(bucket: Optional[str] = None, limit: int = 100):
        scores = db.load_scores()
        fundas = {f["symbol"]: f for f in db.load_fundamentals()}
        rows = _with_mb_prev(_merge(scores, fundas))
        rows.sort(key=lambda r: r.get("mb_score") or 0, reverse=True)
        if bucket:
            rows = [r for r in rows if (r.get("mb_bucket") or "") == bucket.upper()]
        return _json_safe(rows[: max(1, min(limit, 300))])

    @app.get("/api/mb-candidates/{symbol}")
    def mb_candidate_history(symbol: str, limit: int = 40):
        return _json_safe(db.mb_candidates_history(symbol, limit=max(1, min(limit, 200))))

    # ── Sector rotation + portfolio construction ─────────────────────────────

    @app.get("/api/breadth")
    def breadth_endpoint():
        """Universe-wide % above 20/50/200-DMA + market health."""
        return _fresh_breadth()

    @app.get("/api/watchlist")
    def watchlist_endpoint():
        """Today's idea generator: RS leaders / score surges / MB elite / top sectors."""
        latest = db.latest_watchlist_scan()
        events = db.load_watchlist_events(latest) if latest else []
        return _json_safe({"scan_date": latest, "events": events})

    @app.get("/api/alpha")
    def alpha_endpoint():
        """Alpha decay (forward returns by horizon) + factor IC summary."""
        return _json_safe({"decay": db.load_alpha_rows(), "ic": alpha.ic_summary()})

    @app.get("/api/factor-ic")
    def factor_ic_endpoint():
        """Which factor predicts forward returns, overall and per regime."""
        return _json_safe(alpha.ic_summary())

    @app.get("/api/sectors")
    def sectors():
        scores = db.load_scores()
        fundas = {f["symbol"]: f for f in db.load_fundamentals()}
        ranked = rotation.rank_sectors(scores, fundas)
        out = sorted(ranked.values(), key=lambda s: s["strength"], reverse=True)
        return _json_safe(out)

    @app.get("/api/portfolio")
    def portfolio_endpoint(top_n: int = 8, mode: str = "kelly", max_sector_weight: float = 25.0):
        regime = _fresh_regime()
        scores = db.load_scores()
        fundas = {f["symbol"]: f for f in db.load_fundamentals()}
        rows = _merge(scores, fundas)
        equity_pct = float(regime.get("allocation", {}).get("equity", 60))
        alloc = portfolio.build_allocation(rows, equity_pct, top_n=top_n, mode=mode, max_sector_weight=max_sector_weight)
        records_by_symbol = {r["symbol"]: r for r in rows}
        return _json_safe(
            {
                "regime": regime.get("regime"),
                "equity_pct": equity_pct,
                "allocation": alloc,
                "mode": mode,
                "max_sector_weight": max_sector_weight,
                "sector_weights": portfolio.sector_weights(alloc),
                "factor_exposure": portfolio.factor_exposure(alloc, records_by_symbol),
                "portfolio_risk": portfolio.portfolio_risk(alloc, records_by_symbol, equity_pct),
            }
        )

    # ── Score history + quarterly detail ─────────────────────────────────────

    @app.get("/api/score-history/{symbol}")
    def score_history(symbol: str, limit: int = 40):
        return _json_safe(db.score_history_for(symbol, limit=max(1, min(limit, 200))))

    @app.get("/api/quarterly/{symbol}")
    def quarterly(symbol: str):
        return _json_safe(db.load_quarterly_results(symbol, limit=12))

    # ── Scan ─────────────────────────────────────────────────────────────────

    @app.post("/api/scan")
    def scan(force: bool = False, tier: str = "core"):
        if not _scan_lock.acquire(blocking=False):
            return {"status": "running", "message": "A scan is already in progress."}
        try:
            return _json_safe(run_scan(force_fundamentals=force, tier=tier))
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
        # Benchmark comparison: same window, buy-and-hold NIFTY 50.
        bench = _benchmark_stats(result.get("equity_curve") or [], result["summary"].get("initial_capital"))
        if bench:
            result["summary"].update(bench["summary"])
            result["benchmark_curve"] = bench["curve"]
        saved_id = db.save_backtest(result["params"], result["equity_curve"], result["trades"], result["summary"])
        return _json_safe({**result, "id": saved_id})

    @app.post("/api/backtest/walk-forward")
    def walk_forward_endpoint(params: Optional[dict] = None):
        """Same strategy, N consecutive 12-month folds — hit rate / avg return
        / worst drawdown across folds instead of one cherry-picked window."""
        frames = {}
        for sym in db.universe_symbols():
            rows = db.load_prices(sym)
            if rows:
                frames[sym] = indicators.to_dataframe(rows)
        result = bt.walk_forward(frames, params or {})
        if not result.get("folds"):
            raise HTTPException(status_code=422, detail=result["summary"].get("error", "No data"))
        return _json_safe(result)

    @app.get("/api/backtests")
    def backtests(limit: int = 10):
        return _json_safe(db.load_backtests(limit=limit))

    # ── Portfolio helpers ────────────────────────────────────────────────────

    @app.get("/api/fundamentals-history/{symbol}")
    def fundamentals_history(symbol: str, limit: int = 20):
        """Point-in-time fundamentals as known on each past scan date."""
        return _json_safe(db.fundamentals_history_for(symbol, limit=max(1, min(limit, 100))))

    @app.get("/api/universe-history")
    def universe_history():
        """Universe membership snapshots (survivorship-bias protection)."""
        return _json_safe(db.universe_history_snapshots(limit=60))

    # ── v11: Data quality + discovery engine ─────────────────────────────────

    @app.get("/api/data-quality")
    def data_quality_endpoint():
        """Per-field yFinance fundamentals coverage — the data bottleneck made visible."""
        return _json_safe(data.field_coverage())

    @app.get("/api/discovery")
    def discovery_endpoint(limit: int = 30):
        """Emerging Leaders: discovery-tier names ranked by discovery_score
        (RS rank + RS acceleration + revision proxy + momentum)."""
        core = set(db.universe_symbols(tier="core"))
        scores = db.load_scores()
        fundas = {f["symbol"]: f for f in db.load_fundamentals()}
        rows = _with_prev(_merge(scores, fundas))
        disc = [r for r in rows if r.get("symbol") not in core]
        for r in disc:
            r["rs_accel"] = discovery.rs_acceleration(r)
            r["discovery_score"] = discovery.discovery_score(r)
        disc.sort(key=lambda r: r.get("discovery_score") or 0, reverse=True)
        out = []
        for r in disc[: max(1, min(int(limit), 200))]:
            out.append(
                {
                    "symbol": r.get("symbol"),
                    "name": r.get("name") or r.get("symbol"),
                    "sector": r.get("sector"),
                    "score": r.get("score"),
                    "score_delta": r.get("score_delta"),
                    "discovery_score": r.get("discovery_score"),
                    "rs_rank": r.get("rs_rank"),
                    "rs_accel": r.get("rs_accel"),
                    "revision_score": r.get("revision_score"),
                    "momentum": r.get("momentum"),
                    "data_confidence": r.get("data_confidence"),
                    "mb_bucket": r.get("mb_bucket"),
                    "tier": "discovery",
                }
            )
        return _json_safe({"n": len(disc), "rows": out})

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
    def universe(tier: Optional[str] = None):
        return _json_safe(
            {
                "tiers": db.universe_tiers(),
                "symbols": [
                    {"symbol": s, "name": s.replace(".NS", ""), "sector": "Large Cap" if s in _LARGE else "Mid/Small Cap"}
                    for s in db.universe_symbols(tier=tier if tier in ("core", "discovery") else None)
                ],
            }
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


def _with_mb_prev(rows: list[dict]) -> list[dict]:
    """Attach MB trend deltas (MB score/rank vs previous candidate snapshot)."""
    latest = db.latest_mb_candidates()
    prev = db.previous_mb_candidates()
    for r in rows:
        sym = r.get("symbol", "")
        cur_row = latest.get(sym) or {}
        old = prev.get(sym) or {}
        cur = r.get("mb_score")
        prev_mb = old.get("mb_score")
        r["mb_rank"] = cur_row.get("mb_rank")
        r["prev_mb_score"] = prev_mb
        r["mb_delta"] = (
            round(cur - prev_mb, 1) if cur is not None and prev_mb is not None else None
        )
        r["mb_rank_delta"] = (
            old.get("mb_rank") - cur_row.get("mb_rank")
            if cur_row.get("mb_rank") is not None and old.get("mb_rank") is not None
            else None
        )
    return rows


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


def _benchmark_stats(curve: list, initial: float) -> Optional[dict]:
    """Buy-and-hold NIFTY 50 stats over the same window as a backtest curve.

    Returns summary fields (benchmark return/CAGR/max DD + alpha vs the
    strategy) and an aligned benchmark curve for charting.
    """
    if not curve or not initial:
        return None
    nifty = data.fetch_benchmark("^NSEI", period="5y")
    if nifty is None or nifty.empty:
        return None
    closes = nifty["Close"].dropna()
    if len(closes) < 60:
        return None
    start, end = curve[0]["date"], curve[-1]["date"]
    idx = pd.DatetimeIndex(pd.to_datetime(closes.index.date))
    mask = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
    window = closes[mask]
    if len(window) < 20:
        return None
    first = float(window.iloc[0])
    last = float(window.iloc[-1])
    if first <= 0:
        return None
    total_ret = last / first - 1
    years = max(len(window) / 252, 1 / 252)
    cagr = ((last / first) ** (1 / years) - 1) * 100
    norm = window / first * initial
    mdd = float(((norm - norm.cummax()) / norm.cummax()).min() * 100)

    by_date = {d.date().isoformat(): round(v, 2) for d, v in zip(idx[mask], norm)}
    aligned, last_v = [], None
    for pt in curve:
        if pt["date"] in by_date:
            last_v = by_date[pt["date"]]
        if last_v is not None:
            aligned.append({"date": pt["date"], "value": last_v})
    strat_ret = curve[-1]["equity"] / initial - 1
    return {
        "summary": {
            "benchmark": "NIFTY 50",
            "benchmark_return_pct": round(total_ret * 100, 2),
            "benchmark_cagr_pct": round(cagr, 2),
            "benchmark_max_dd_pct": round(mdd, 2),
            "alpha_pct": round((strat_ret - total_ret) * 100, 2),
        },
        "curve": aligned,
    }


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
