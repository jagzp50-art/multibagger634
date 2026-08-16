"""
Sovereign Lite v12 — alpha decay, factor IC, and regime-learned weights.

After each scan we can measure whether the model actually works:

  - alpha decay: forward returns 7 / 30 / 90 / 180 trading days after each
    snapshot date (a 95-scored stock should still be up 30 days later)
  - factor IC: Spearman rank correlation between each factor value at a
    snapshot and the realized forward return — which factor predicts?
  - regime conditional IC: the same, split by the regime at signal time
  - learned weights: tilt regime weights toward factors with recent positive
    IC, clamped so the base allocation is never overridden

All of it degrades to empty results until enough snapshots + elapsed price
history exist, so a fresh deployment shows nothing until scans accumulate.
"""
from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from . import db

HORIZONS = [7, 30, 90, 180]
FACTORS = ["quality", "growth", "momentum", "valuation", "risk", "mb_score", "rs_rank", "opp_score"]


# ── Forward returns ─────────────────────────────────────────────────────────

def _forward_returns(
    frames: dict[str, pd.DataFrame],
    anchor: str,
    horizons: list[int],
) -> dict[str, dict[int, Optional[float]]]:
    """Return {symbol: {horizon: realized forward return}} from `anchor` date.

    Only returns that have fully elapsed (anchor + horizon bars <= last bar)
    are computed, so nothing looks into the future.
    """
    out: dict[str, dict[int, Optional[float]]] = {}
    ts_anchor = pd.Timestamp(anchor)
    for sym, df in frames.items():
        s = df["close"].dropna()
        if s.empty:
            continue
        pos = s.index.searchsorted(ts_anchor)
        if pos >= len(s.index):
            continue
        row: dict[int, Optional[float]] = {}
        for h in horizons:
            if pos + h < len(s.index):
                p0, p1 = float(s.iloc[pos]), float(s.iloc[pos + h])
                if p0 and p0 > 0:
                    row[h] = p1 / p0 - 1
        if row:
            out[sym] = row
    return out


def update_alpha_tracking(frames: dict[str, pd.DataFrame]) -> int:
    """Fill forward-return aggregates for every stored snapshot that has
    enough elapsed history. Replaces prior values as more time passes."""
    updated = 0
    for ts in db.score_snapshot_times(limit=40):
        rows = db.score_history_at(ts)
        if not rows:
            continue
        anchors = sorted({r["scan_date"] for r in rows if r.get("scan_date")})
        if not anchors:
            continue
        anchor = anchors[0]
        regime = rows[0].get("regime")
        fr = _forward_returns(frames, anchor, HORIZONS)
        for h in HORIZONS:
            vals = [fr[s][h] for s in fr if h in fr[s] and fr[s][h] is not None]
            if len(vals) < 5:
                continue
            avg = sum(vals) / len(vals) * 100
            med = sorted(vals)[len(vals) // 2] * 100
            hits = sum(1 for v in vals if v > 0) / len(vals) * 100
            db.save_alpha_rows(
                [
                    {
                        "scan_date": ts,
                        "horizon_days": h,
                        "avg_return_pct": round(avg, 2),
                        "median_return_pct": round(med, 2),
                        "hit_rate_pct": round(hits, 1),
                        "n": len(vals),
                        "regime": regime,
                    }
                ]
            )
            updated += 1
    return updated


# ── Factor IC (which factor predicts returns?) ──────────────────────────────

def _ranks(values: list[float]) -> list[float]:
    """Average-rank a list (ties share the mean rank)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> Optional[float]:
    """Spearman rank correlation (pure Python — no scipy dependency)."""
    if len(xs) < 10 or len(xs) != len(ys):
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    n = len(xs)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return None
    ic = num / (dx * dy)
    if math.isnan(ic) or abs(ic) > 1.0:
        return None
    return ic


def compute_factor_ics(frames: dict[str, pd.DataFrame]) -> int:
    """Spearman IC between each factor at each snapshot and the realized
    30-day forward return. Stored per (snapshot, factor, regime)."""
    saved = 0
    for ts in db.score_snapshot_times(limit=20):
        rows = db.score_history_at(ts)
        if not rows:
            continue
        anchors = sorted({r["scan_date"] for r in rows if r.get("scan_date")})
        if not anchors:
            continue
        anchor = anchors[0]
        regime = rows[0].get("regime")
        fr = _forward_returns(frames, anchor, [30])
        ret = {s: fr[s][30] for s in fr if 30 in fr[s] and fr[s][30] is not None}
        ic_rows = []
        for factor in FACTORS:
            pairs = [(r.get(factor), ret.get(r["symbol"])) for r in rows]
            pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
            if len(pairs) < 10:
                continue
            ic = _spearman([p[0] for p in pairs], [p[1] for p in pairs])
            if ic is None:
                continue
            ic_rows.append(
                {
                    "scan_date": ts,
                    "factor": factor,
                    "horizon_days": 30,
                    "ic": round(ic, 4),
                    "regime": regime,
                    "n": len(pairs),
                }
            )
        if ic_rows:
            db.save_factor_ic(ic_rows)
            saved += len(ic_rows)
    return saved


def ic_summary() -> dict:
    """Aggregate stored ICs into per-factor and per-regime averages."""
    rows = db.load_factor_ic(limit=800)
    if not rows:
        return {"factors": {}, "regimes": {}, "updated": None}
    by_factor: dict[str, list[float]] = {}
    by_regime: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        by_factor.setdefault(r["factor"], []).append(r["ic"])
        rg = r.get("regime") or "?"
        by_regime.setdefault(rg, {}).setdefault(r["factor"], []).append(r["ic"])
    factors = {
        f: {"avg_ic": round(sum(v) / len(v), 4), "n": len(v), "last_ic": round(v[-1], 4)}
        for f, v in by_factor.items()
    }
    regimes = {
        rg: {f: round(sum(v) / len(v), 4) for f, v in fmap.items()}
        for rg, fmap in by_regime.items()
    }
    return {"factors": factors, "regimes": regimes, "updated": rows[-1]["created_at"] if rows else None}


def learned_weights(base: dict[str, float], summary: dict, regime: Optional[str]) -> dict[str, float]:
    """Tilt regime weights toward factors with recent positive IC.

    Bonus is clamped to ±6 pts per factor and the set is renormalized, so the
    regime's base allocation is never overridden — evidence only tilts.
    """
    w = dict(base)
    factors = summary.get("factors") or {}
    if not factors:
        return w
    for f in ("quality", "growth", "momentum", "valuation", "risk"):
        info = factors.get(f)
        if not info or info.get("avg_ic") is None:
            continue
        bonus = max(0.0, min(0.06, (info["avg_ic"] - 0.10) * 0.6))
        w[f] += bonus
    total = sum(w.values())
    if total > 0:
        w = {k: round(v / total, 4) for k, v in w.items()}
    return w
