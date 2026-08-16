"""
Sovereign Lite v7 — sector rotation.

Ranks sectors by a blend of relative strength (mean RS rank of members) and
earnings growth (mean Growth score of members), then applies a modest boost
to stocks inside strong sectors and a penalty inside weak ones.

  sector_strength = percentile(0.5 * sector_rs + 0.5 * sector_growth)
  sector_boost    = (strength - 50) / 10        → range −5 … +5 points

The boost is deliberately small — it tilts, it never overrides stock-level
fundamentals.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional


def rank_sectors(records: list[dict], fundas_map: dict[str, dict]) -> dict[str, dict]:
    """Aggregate sector RS/growth and return {sector: {...}} ranked by strength."""
    agg: dict[str, dict] = defaultdict(lambda: {"rs_sum": 0.0, "growth_sum": 0.0, "n": 0})
    for r in records:
        sector = (fundas_map.get(r.get("symbol"), {}) or {}).get("sector") or "Unknown"
        if r.get("rs_rank") is None and r.get("growth") is None:
            continue
        a = agg[sector]
        if r.get("rs_rank") is not None:
            a["rs_sum"] += float(r["rs_rank"])
        if r.get("growth") is not None:
            a["growth_sum"] += float(r["growth"])
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
            }
        )

    # Percentile-rank each sector's strength blend across sectors.
    strengths = [0.5 * s["rs"] + 0.5 * s["growth"] for s in sectors]
    for s in sectors:
        blend = 0.5 * s["rs"] + 0.5 * s["growth"]
        worse = sum(1 for v in strengths if v <= blend)
        strength = worse / len(strengths) * 100 if strengths else 50.0
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
