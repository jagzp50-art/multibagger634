"""
Sovereign Lite v10 — portfolio construction layer.

Position Score = 0.40 Quality + 0.30 MB Score + 0.20 RS Rank + 0.10 Risk
(higher = better). Allocation weights are conviction-based: the #1 position
gets the largest slice of the regime's equity allocation — top conviction
gets the most capital, no equal-weighting.

v8: Kelly-Lite 2.0 adds a drawdown penalty (low vol alone isn't enough — a
stock with catastrophic drawdowns gets penalized), allocation enforces a hard
sector cap (no 45% Financials book), and factor_exposure() surfaces crowding.

All inputs degrade to None-safe math via `scoring._weighted`.
"""
from __future__ import annotations

from typing import Optional

from . import scoring

POS_WEIGHTS = {"quality": 0.40, "mb": 0.30, "rs": 0.20, "risk": 0.10}


def position_score(quality: Optional[float], mb: Optional[float],
                   rs: Optional[float], risk: Optional[float]) -> Optional[float]:
    parts = [
        (quality, POS_WEIGHTS["quality"]),
        (mb, POS_WEIGHTS["mb"]),
        (rs, POS_WEIGHTS["rs"]),
        (risk, POS_WEIGHTS["risk"]),
    ]
    return scoring._weighted(parts)


def attach_position_scores(records: list[dict]) -> list[dict]:
    """Attach pos_score to scored records (mb_score must already be set)."""
    for r in records:
        r["pos_score"] = position_score(
            r.get("quality"), r.get("mb_score"), r.get("rs_rank"), r.get("risk")
        )
        if r["pos_score"] is not None:
            r["pos_score"] = round(r["pos_score"], 1)
    return records


def _kelly_lite(r: dict) -> float:
    """Kelly-Lite 2.0: score · quality · (1/volatility) · drawdown penalty.

    Low volatility alone isn't enough — a stock with low vol but catastrophic
    drawdowns gets penalized via its max drawdown (a −50% DD halves it).
    """
    pos = r.get("pos_score") or 0.0
    q = (r.get("quality") or 50.0) / 100.0
    vol = r.get("vol")
    if vol is None or vol <= 0:
        vol = 0.30
    vol_factor = 0.15 / (0.15 + vol)
    mdd = r.get("max_dd")
    dd_factor = 1.0
    if mdd is not None:
        depth = min(1.0, max(0.0, -float(mdd)))  # 0..1 drawdown depth
        dd_factor = max(0.05, min(1.0, 1.0 - depth / 0.5))
    return max(0.0, pos * q * vol_factor * dd_factor)


def _enforce_sector_caps(alloc: list[dict], cap_pct: float) -> list[dict]:
    """Trim any sector above `cap_pct`, redistributing the excess to
    under-cap sectors (up to their headroom). A 45% Financials book never
    happens; if the cap is infeasible (e.g. two sectors, 60% total, 25% cap)
    the un-placeable excess is left in cash rather than oscillating.
    """
    if cap_pct <= 0 or cap_pct >= 100:
        return alloc
    for _ in range(50):
        by_sector: dict[str, float] = {}
        for a in alloc:
            by_sector[a["sector"]] = by_sector.get(a["sector"], 0.0) + a["weight_pct"]
        over = {s: w for s, w in by_sector.items() if w > cap_pct}
        if not over:
            break
        excess = sum(w - cap_pct for w in over.values())
        under = {s: w for s, w in by_sector.items() if w < cap_pct}
        headroom = sum(cap_pct - w for w in under.values())
        give = min(excess, headroom)
        # Trim over-cap sectors to the cap (removes `excess` in total).
        for a in alloc:
            s = a["sector"]
            if s in over:
                trim = (by_sector[s] - cap_pct) * (a["weight_pct"] / by_sector[s])
                a["weight_pct"] = round(a["weight_pct"] - trim, 2)
        # Redistribute `give` in total across under-cap names, proportional to
        # their current weights (so a sector with several names splits it).
        under_names = [a for a in alloc if a["sector"] in under]
        under_weight_total = sum(a["weight_pct"] for a in under_names)
        if under_weight_total > 0:
            for a in under_names:
                a["weight_pct"] = round(
                    a["weight_pct"] + give * (a["weight_pct"] / under_weight_total), 2
                )
        # New total = old − excess + give; anything the caps couldn't absorb
        # (give < excess) is simply left in cash — no oscillation, no double-count.
        if give < excess:
            break
    return alloc


def sector_weights(alloc: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for a in alloc:
        out[a["sector"]] = round(out.get(a["sector"], 0.0) + a["weight_pct"], 2)
    return out


def factor_exposure(alloc: list[dict], records_by_symbol: dict[str, dict]) -> dict:
    """Portfolio-weighted average factor scores + concentration readout.

    A book that is 80% momentum is a crowding risk — surface it.
    """
    factors = ("quality", "growth", "momentum", "valuation", "risk", "mb_score", "rs_rank")
    out: dict = {}
    for f in factors:
        num = den = 0.0
        for a in alloc:
            v = (records_by_symbol.get(a["symbol"]) or {}).get(f)
            if v is not None:
                num += v * a["weight_pct"]
                den += a["weight_pct"]
        out[f] = round(num / den, 1) if den > 0 else None
    comp = {k: v for k, v in out.items() if v is not None and k in ("quality", "growth", "momentum", "valuation", "risk")}
    if comp:
        top = max(comp, key=comp.get)
        out["top_factor"] = top
        out["top_factor_share"] = round(comp[top] / sum(comp.values()) * 100, 1)
    return out


def portfolio_risk(alloc: list[dict], records_by_symbol: dict[str, dict],
                   equity_pct: float) -> dict:
    """Portfolio-level risk readout: weighted vol, drawdown, concentration.

    Vol and max drawdown come from each name's stored per-stock metrics (now
    persisted with every scan), so the numbers reflect the actual book.
    `portfolio_vol` applies a diversification discount: a 10-name book with
    40% average vol does not swing like one name at 40% — effective N
    (1 / HHI) scales the average down, a standard equal-risk approximation
    that needs no covariance matrix.
    """
    n = len(alloc)
    if n == 0:
        return {"n": 0, "risk_grade": None}
    total = sum(a["weight_pct"] for a in alloc) or 1.0
    share = [a["weight_pct"] / total for a in alloc]
    hhi = round(sum(s * s for s in share), 3)

    def wavg(key: str):
        num = den = 0.0
        for a, s in zip(alloc, share):
            v = (records_by_symbol.get(a["symbol"]) or {}).get(key)
            if v is None:
                continue
            num += v * s
            den += s
        return (num / den) if den > 0 else None

    avg_vol = wavg("vol")
    avg_dd = wavg("max_dd")
    avg_quality = wavg("quality")
    avg_conf = wavg("data_confidence")

    eff_n = round(1.0 / hhi, 1) if hhi > 0 else float(n)
    port_vol = None
    if avg_vol is not None and avg_vol > 0 and eff_n > 0:
        port_vol = round(avg_vol / (eff_n ** 0.5), 4)

    top = sorted(share, reverse=True)
    top1 = round(top[0] * 100, 1) if top else 0.0
    top3 = round(sum(top[:3]) * 100, 1)

    grade = "BALANCED"
    if port_vol is not None and avg_dd is not None:
        if port_vol < 0.18 and avg_dd > -0.35:
            grade = "CONSERVATIVE"
        elif port_vol < 0.30 and avg_dd > -0.50:
            grade = "BALANCED"
        else:
            grade = "AGGRESSIVE"
    elif port_vol is not None:
        grade = "CONSERVATIVE" if port_vol < 0.18 else ("BALANCED" if port_vol < 0.30 else "AGGRESSIVE")

    return {
        "n": n,
        "equity_pct": round(equity_pct, 1),
        "cash_pct": round(max(0.0, 100 - equity_pct), 1),
        "avg_vol": round(avg_vol * 100, 1) if avg_vol is not None else None,
        "portfolio_vol": round(port_vol * 100, 1) if port_vol is not None else None,
        "avg_max_dd": round(avg_dd * 100, 1) if avg_dd is not None else None,
        "effective_n": eff_n,
        "hhi": hhi,
        "top1_share": top1,
        "top3_share": top3,
        "avg_quality": round(avg_quality, 1) if avg_quality is not None else None,
        "avg_confidence": round(avg_conf, 1) if avg_conf is not None else None,
        "risk_grade": grade,
    }


def build_allocation(
    records: list[dict],
    equity_pct: float,
    top_n: int = 8,
    mode: str = "kelly",
    max_sector_weight: float = 25.0,
) -> list[dict]:
    """Conviction-weighted allocation over the top-N by position score.

    mode="kelly" (default): weight ∝ pos_score · quality · 1/vol · DD-penalty.
    mode="conviction": weight ∝ pos_score² — pure score tapering.
    Both normalize to equity_pct, then enforce `max_sector_weight`.
    """
    ranked = sorted(
        [r for r in records if r.get("pos_score") is not None],
        key=lambda r: r.get("pos_score") or 0,
        reverse=True,
    )[: max(1, min(int(top_n), 25))]

    if mode == "conviction":
        scores = [(r.get("pos_score") or 0.0) ** 2 for r in ranked]
    else:
        scores = [_kelly_lite(r) for r in ranked]
    total = sum(scores)
    out = []
    for r, s in zip(ranked, scores):
        w = (s / total * equity_pct) if total > 0 else equity_pct / len(ranked)
        out.append(
            {
                "symbol": r.get("symbol"),
                "name": r.get("name") or r.get("symbol"),
                "sector": r.get("sector") or "Unknown",
                "pos_score": r.get("pos_score"),
                "mb_bucket": r.get("mb_bucket"),
                "weight_pct": round(w, 2),
                "mode": mode,
            }
        )
    out = _enforce_sector_caps(out, float(max_sector_weight))
    return out
