"""
Sovereign Lite v7 — scan pipeline.

    fetch prices → persist → fetch fundamentals (cached) → regime detection
    → score all symbols → rank → multibagger detect → persist scores
"""
from __future__ import annotations

import time
from datetime import date, datetime

import pandas as pd

from . import alpha, data, db, indicators, multibagger, portfolio, regime as regime_mod, rotation, scoring, watchlist
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
    # Learn from history: tilt regime weights toward factors with recent positive IC.
    ic_summary = alpha.ic_summary()
    regime["weights"] = alpha.learned_weights(dict(regime["weights"]), ic_summary, regime["regime"])
    print(f"[lite] regime: {regime['regime']} (vix={regime['vix']}, adx={regime['adx']})")

    records = scoring.compute_scores(regime, fundas_list, px_frames)
    fundas_map = {f["symbol"]: f for f in fundas_list}

    # Sector rotation: tilt scores toward stocks inside strong sectors.
    records = rotation.apply_sector_rotation(records, fundas_map)
    # Factor attribution: record what actually drove each final score.
    for r in records:
        fc = r.setdefault("factor_contributions", {})
        fc["sector_boost"] = r.get("sector_boost")

    records.sort(key=lambda r: r.get("score") or 0, reverse=True)
    for i, r in enumerate(records, start=1):
        r["rank"] = i
    print(f"[lite] scored: {len(records)} symbols")

    mb_records = multibagger.detect(records, fundas_map, px_frames)

    # Opportunity score (screener's main ranking) + portfolio position score.
    for r in mb_records:
        r["eps_accel"] = (fundas_map.get(r["symbol"], {}) or {}).get("eps_accel")
        opp = scoring.opportunity_score(r)
        r["opp_score"] = round(opp, 1) if opp is not None else None
    mb_records = portfolio.attach_position_scores(mb_records)

    # MB rank (by MB score) so the candidates table tracks movement over time.
    mb_records.sort(key=lambda r: r.get("mb_score") or 0, reverse=True)
    for i, r in enumerate(mb_records, start=1):
        r["mb_rank"] = i

    # Score history: compare against the previous snapshot, then append this one.
    prev = db.latest_score_snapshot()
    db.upsert_scores(mb_records)
    db.snapshot_scores(mb_records, date.today().isoformat(), regime["regime"])
    db.snapshot_mb_candidates(mb_records, date.today().isoformat(), regime["regime"])

    # Watchlist intelligence: daily idea generator (RS leaders / surges / MB elite / top sectors).
    try:
        events = watchlist.detect_events(mb_records, prev, fundas_map, date.today().isoformat())
        if events:
            db.save_watchlist_events(events, date.today().isoformat())
            print(f"[lite] watchlist: {len(events)} events")
    except Exception as exc:
        print(f"  ⚠️ watchlist events failed: {exc}")

    # Alpha decay + factor IC: measure whether the model actually predicted returns.
    try:
        n_alpha = alpha.update_alpha_tracking(px_frames)
        n_ic = alpha.compute_factor_ics(px_frames)
        if n_alpha or n_ic:
            print(f"[lite] alpha: {n_alpha} horizon rows, {n_ic} factor ICs")
    except Exception as exc:
        print(f"  ⚠️ alpha/IC update failed: {exc}")
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
