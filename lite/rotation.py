"""
Sovereign Lite v17 — sector rotation.

Ranks sectors by a blend of relative strength, earnings growth, breadth and
momentum, then applies a modest boost to stocks inside strong sectors and a
penalty inside weak ones:

  sector_rs      = mean RS rank of members
  sector_growth  = mean Growth score of members
  sector_breadth = % of members above their 200-DMA
  sector_momentum= mean Momentum score of members

  sector_strength = percentile(0.40·RS + 0.30·Growth + 0.20·Breadth + 0.10·Momentum)
  sector_boost    = (strength − 50) / 10          → range −5 … +5 points

The boost is deliberately small — it tilts, it never overrides stock-level
fundamentals. Available parts are re-weighted when a member metric is missing.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

SECTOR_WEIGHTS = [
    ("rs", 0.40),
    ("growth", 0.30),
    ("breadth", 0.20),
    ("momentum", 0.10),
]


def _blend(info: dict) -> Optional[float]:
    num = den = 0.0
    for key, w in SECTOR_WEIGHTS:
        v = info.get(key)
        if v is None:
            continue
        num += float(v) * w
        den += w
    return (num / den) if den > 0 else None


def rank_sectors(records: list[dict], fundas_map: dict[str, dict]) -> dict[str, dict]:
    """Aggregate sector metrics and return {sector: {...}} ranked by strength."""
    agg: dict[str, dict] = defaultdict(
        lambda: {
            "rs_sum": 0.0,
            "growth_sum": 0.0,
            "mom_sum": 0.0,
            "breadth": 0,
            "n": 0,
            "n_mom": 0,
            "n_breadth": 0,
        }
    )
    for r in records:
        sector = (fundas_map.get(r.get("symbol"), {}) or {}).get("sector") or "Unknown"
        a = agg[sector]
        if r.get("rs_rank") is not None:
            a["rs_sum"] += float(r["rs_rank"])
        if r.get("growth") is not None:
            a["growth_sum"] += float(r["growth"])
        if r.get("momentum") is not None:
            a["mom_sum"] += float(r["momentum"])
            a["n_mom"] += 1
        if r.get("above_200") is not None:
            a["breadth"] += int(bool(r["above_200"]))
            a["n_breadth"] += 1
        a["n"] += 1

    sectors = []
    for sector, a in agg.items():
        if a["n"] == 0:
            continue
        sectors.append(
            {
                "sector": sector,
                "count": a["n"],
                "rs": a["rs_sum"] / a["n"],
                "growth": a["growth_sum"] / a["n"],
                "momentum": (a["mom_sum"] / a["n_mom"]) if a["n_mom"] else None,
                "breadth": (a["breadth"] / a["n_breadth"] * 100) if a["n_breadth"] else None,
            }
        )

    # Percentile-rank each sector's strength blend across sectors.
    blends = [b for b in (_blend(s) for s in sectors) if b is not None]
    for s in sectors:
        blend = _blend(s)
        if blend is None:
            s["strength"] = 50.0
            s["boost"] = 0.0
            continue
        worse = sum(1 for v in blends if v <= blend)
        strength = worse / len(blends) * 100 if blends else 50.0
        s["strength"] = round(strength, 1)
        s["boost"] = round((strength - 50) / 10, 2)
    sectors.sort(key=lambda s: s["strength"], reverse=True)
    for i, s in enumerate(sectors, start=1):
        s["rank"] = i
    return {s["sector"]: s for s in sectors}


def apply_sector_rotation(records: list[dict], fundas_map: dict[str, dict]) -> list[dict]:
    """Attach sector strength/boost to each record and adjust its score."""
    sectors = rank_sectors(records, fundas_map)
    for r in records:
        sector = (fundas_map.get(r.get("symbol"), {}) or {}).get("sector") or "Unknown"
        info = sectors.get(sector) or {}
        boost = float(info.get("boost") or 0.0)
        r["sector_strength"] = info.get("strength")
        r["sector_boost"] = round(boost, 2)
        if r.get("score") is not None:
            r["score"] = round(max(0.0, min(100.0, r["score"] + boost)), 1)
    return records
