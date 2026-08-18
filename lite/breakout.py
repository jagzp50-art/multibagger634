"""
Sovereign Lite v17 — Sector Breakout Monitor (Phase 18 / Institutional Discovery).

Sector rotation ranks sectors by blended strength; this module answers the
complementary question: which sectors are *pressing their 52-week highs right
now, with broad participation* — the setup that tends to precede sustained
leadership, and the one place new leaders show up before the index does.

Per symbol we measure, from stored price history:

    dist_52w_high     distance below the 52-week high (0 = at the high)
    at_52w_high       within 2% of the 52-week high (new-high zone)
    above_50/200      trend participation
    ret_3m            3-month return

Per sector we aggregate the same statistics and blend them into a
0–100 breakout score (re-weighted automatically when a component is missing):

    35%  % names within 5% of their 52-week high   (proximity)
    20%  % names in the new-high zone              (fresh highs)
    20%  % names above their 200-DMA               (trend confirmation)
    15%  % names above their 50-DMA                (short-term participation)
    10%  median 3-month return percentile          (momentum)

A sector is flagged `breakout` when its score ≥ 60 with at least 3 members —
the same spirit as the sector-breadth requirement the rotation layer uses,
but aimed at the *high end of the range* instead of trend participation.
Each sector also carries its leading names (closest to their 52-week highs,
strongest 3-month return) so the monitor doubles as an idea source.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

from . import indicators

NEAR_HIGH_PCT = 5.0    # within 5% of the 52-week high counts as "near"
AT_HIGH_PCT = 2.0      # within 2% counts as the new-high zone (matches breadth)
BREAKOUT_MIN_SCORE = 60.0
BREAKOUT_MIN_MEMBERS = 3
LEADERS_PER_SECTOR = 3

# Component weights — blend is re-normalized if a part is missing.
BREAKOUT_WEIGHTS = [
    ("near_high_pct", 0.35),
    ("at_high_pct", 0.20),
    ("above_200_pct", 0.20),
    ("above_50_pct", 0.15),
    ("mom_3m_pctile", 0.10),
]


def _symbol_stats(df) -> Optional[dict]:
    """Per-symbol breakout inputs from one price frame."""
    if df is None or df.empty or "close" not in df.columns:
        return None
    close = df["close"].dropna()
    if len(close) < 63:  # need at least ~3 months to say anything
        return None
    win = close.tail(252)
    high = float(win.max())
    low = float(win.min())
    last = float(close.iloc[-1])
    if high <= 0 or low < 0 or last <= 0:
        return None
    dist_high = (high - last) / high if high > 0 else None
    pos = (last - low) / (high - low) if high > low else (1.0 if last >= high else 0.0)
    s50 = indicators.sma(close, 50)
    s200 = indicators.sma(close, 200)
    ret_3m = indicators.returns_over(close, 63)
    return {
        "dist_52w_high": dist_high,
        "position_52w": pos,
        "at_52w_high": dist_high is not None and dist_high <= AT_HIGH_PCT / 100.0,
        "near_52w_high": dist_high is not None and dist_high <= NEAR_HIGH_PCT / 100.0,
        "above_50": s50 is not None and last > s50,
        "above_200": s200 is not None and last > s200,
        "ret_3m": ret_3m,
        "close": last,
    }


def _blend(info: dict) -> Optional[float]:
    num = den = 0.0
    for key, w in BREAKOUT_WEIGHTS:
        v = info.get(key)
        if v is None:
            continue
        num += float(v) * w
        den += w
    return (num / den) if den > 0 else None


def rank_breakouts(frames: dict[str, object], fundas_map: dict[str, dict]) -> dict:
    """Rank sectors by breakout score.

    frames:      {symbol: price DataFrame with close/high/low/volume}
    fundas_map:  {symbol: fundamentals dict with a `sector` key}

    Returns a JSON-safe dict with per-sector breakout stats and leaders.
    """
    agg: dict[str, dict] = defaultdict(
        lambda: {
            "dist_sum": 0.0,
            "near": 0,
            "at_high": 0,
            "above50": 0,
            "above200": 0,
            "rets": [],
            "leaders": [],
            "n": 0,
        }
    )
    for sym, df in frames.items():
        stats = _symbol_stats(df)
        if not stats:
            continue
        sector = (fundas_map.get(sym) or {}).get("sector") or "Unknown"
        a = agg[sector]
        a["n"] += 1
        a["dist_sum"] += stats["dist_52w_high"] or 0.0
        a["near"] += int(stats["near_52w_high"])
        a["at_high"] += int(stats["at_52w_high"])
        a["above50"] += int(stats["above_50"])
        a["above200"] += int(stats["above_200"])
        if stats["ret_3m"] is not None:
            a["rets"].append(stats["ret_3m"])
        a["leaders"].append({"symbol": sym, **stats})

    sectors = []
    for sector, a in agg.items():
        if a["n"] == 0:
            continue
        rets = sorted(r for r in a["rets"] if r is not None)
        mom = (rets[len(rets) // 2] * 100) if rets else None  # median 3M return %
        sectors.append(
            {
                "sector": sector,
                "n": a["n"],
                "near_high_pct": round(a["near"] / a["n"] * 100, 1),
                "at_high_pct": round(a["at_high"] / a["n"] * 100, 1),
                "avg_dist_high_pct": round(a["dist_sum"] / a["n"] * 100, 1),
                "above_50_pct": round(a["above50"] / a["n"] * 100, 1),
                "above_200_pct": round(a["above200"] / a["n"] * 100, 1),
                "mom_3m_pct": round(mom, 1) if mom is not None else None,
            }
        )

    if not sectors:
        return {"n_sectors": 0, "in_breakout": [], "sectors": []}

    # Percentile-rank the median 3M return across sectors (0-100).
    moms = sorted(s["mom_3m_pct"] for s in sectors if s["mom_3m_pct"] is not None)
    for s in sectors:
        m = s["mom_3m_pct"]
        if m is None or not moms:
            s["mom_3m_pctile"] = 50.0
        else:
            worse = sum(1 for v in moms if v <= m)
            s["mom_3m_pctile"] = worse / len(moms) * 100.0

    for s in sectors:
        score = _blend(s)
        s["breakout_score"] = round(score, 1) if score is not None else 0.0
        s["breakout"] = bool(
            s["breakout_score"] >= BREAKOUT_MIN_SCORE and s["n"] >= BREAKOUT_MIN_MEMBERS
        )

    # Leaders: only names near their 52-week high (they are the breakout
    # candidates); if the whole sector is far from the highs, fall back to the
    # strongest members so the card is never empty. Ties: closest to the high,
    # then strongest 3M return.
    for a, s in zip(agg.values(), sectors):
        rows = a["leaders"]
        near = [r for r in rows if r["near_52w_high"]]
        candidates = near if near else rows
        candidates.sort(
            key=lambda r: (
                -float(r["position_52w"] or 0.0),
                -float(r["ret_3m"] or -99.0),
            )
        )
        s["leaders"] = [
            {
                "symbol": r["symbol"],
                "close": round(r["close"], 2),
                "dist_52w_high_pct": round(r["dist_52w_high"] * 100, 1)
                if r["dist_52w_high"] is not None else None,
                "ret_3m_pct": round(r["ret_3m"] * 100, 1) if r["ret_3m"] is not None else None,
                "position_52w": round(r["position_52w"], 2),
            }
            for r in candidates[:LEADERS_PER_SECTOR]
        ]

    sectors.sort(key=lambda s: s["breakout_score"], reverse=True)
    for i, s in enumerate(sectors, start=1):
        s["rank"] = i
    return {
        "n_sectors": len(sectors),
        "in_breakout": [s["sector"] for s in sectors if s["breakout"]],
        "sectors": sectors,
    }
