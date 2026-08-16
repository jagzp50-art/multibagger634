"""
Sovereign Lite v7 — scan pipeline.

    fetch prices → persist → fetch fundamentals (cached) → regime detection
    → score all symbols → rank → multibagger detect → persist scores
"""
from __future__ import annotations

import time
from datetime import date, datetime

import pandas as pd

from . import data, db, indicators, multibagger, regime as regime_mod, scoring
from .universe import default_universe


def run_scan(force_fundamentals: bool = False) -> dict:
    t0 = time.time()
    symbols = db.universe_symbols()
    if not symbols:
        db.seed_universe(default_universe())
        symbols = db.universe_symbols()

    print(f"[lite] scan start: {len(symbols)} symbols")
    price_frames = data.fetch_prices(symbols)
    rows_persisted = data.persist_prices(price_frames)
    print(f"[lite] prices: {len(price_frames)} symbols, {rows_persisted} rows")

    fundas = data.fetch_fundamentals(symbols, force=force_fundamentals)
    fundas_list = [f for f in fundas.values() if f.get("symbol")]
    print(f"[lite] fundamentals: {len(fundas_list)} symbols")

    # Reload full history from SQLite with indicators attached
    px_frames: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        rows = db.load_prices(sym)
        if not rows:
            continue
        df = indicators.to_dataframe(rows)
        if len(df) >= 60:
            px_frames[sym] = df
    scoring.attach_indicators(px_frames)

    nifty = data.fetch_benchmark("^NSEI")
    vix = data.fetch_benchmark("^INDIAVIX")
    nifty_adx = None
    if nifty is not None and not nifty.empty:
        nifty_adx = indicators.adx(nifty["High"], nifty["Low"], nifty["Close"])
    regime = regime_mod.detect_regime(nifty, vix, nifty_adx)
    regime["timestamp"] = datetime.now().isoformat(timespec="seconds")
    if nifty is not None and not nifty.empty:
        regime["_nifty_close"] = nifty["Close"].dropna()
    else:
        regime["_nifty_close"] = pd.Series(dtype=float)
    print(f"[lite] regime: {regime['regime']} (vix={regime['vix']}, adx={regime['adx']})")

    records = scoring.compute_scores(regime, fundas_list, px_frames)
    records.sort(key=lambda r: r.get("score") or 0, reverse=True)
    for i, r in enumerate(records, start=1):
        r["rank"] = i
    print(f"[lite] scored: {len(records)} symbols")

    fundas_map = {f["symbol"]: f for f in fundas_list}
    mb_records = multibagger.detect(records, fundas_map, px_frames)

    # Score history: compare against the previous snapshot, then append this one.
    prev = db.latest_score_snapshot()
    db.upsert_scores(mb_records)
    db.snapshot_scores(mb_records, date.today().isoformat(), regime["regime"])
    for r in mb_records:
        old = prev.get(r["symbol"]) or {}
        r["prev_score"] = old.get("score")
        r["prev_rank"] = old.get("rank")

    duration = round(time.time() - t0, 1)
    print(f"[lite] scan complete in {duration}s")
    return {
        "status": "ok",
        "scanned": len(symbols),
        "priced": len(price_frames),
        "scored": len(records),
        "fundamentals": len(fundas_list),
        "rows_persisted": rows_persisted,
        "regime": regime["regime"],
        "duration_sec": duration,
        "timestamp": regime["timestamp"],
    }
