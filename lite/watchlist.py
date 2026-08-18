"""
Sovereign Lite v17 — watchlist intelligence (the daily idea generator).

Fires events when a stock does something interesting relative to the
previous scan:

  RS_LEADER     RS rank > 95           — top 5% relative strength
  SCORE_SURGE   score jump > +10 pts   — big fundamental/technical upgrade
  MB_ELITE      MB score > 90          — multibagger candidate territory
  SECTOR_TOP3   sector enters top 3    — strongest sector rotation ranks

Events are stored per scan and surfaced on the dashboard as a "Today's
ideas" feed.
"""
from __future__ import annotations

from typing import Optional

from . import rotation


def detect_events(
    records: list[dict],
    prev_snapshot: dict[str, dict],
    fundas_map: dict[str, dict],
    scan_date: str,
) -> list[dict]:
    sector_ranks = rotation.rank_sectors(records, fundas_map)
    events: list[dict] = []
    for r in records:
        sym = r.get("symbol", "")
        rs = r.get("rs_rank")
        if rs is not None and rs > 95:
            events.append({"symbol": sym, "event": "RS_LEADER", "detail": f"RS {rs:.0f} — top 5% relative strength"})
        score = r.get("score")
        old = prev_snapshot.get(sym) or {}
        prev_score = old.get("score")
        if score is not None and prev_score is not None and (score - prev_score) > 10:
            events.append(
                {"symbol": sym, "event": "SCORE_SURGE", "detail": f"+{score - prev_score:.1f} pts vs last scan"}
            )
        mb = r.get("mb_score")
        if mb is not None and mb > 90:
            events.append({"symbol": sym, "event": "MB_ELITE", "detail": f"MB {mb:.0f} — multibagger candidate"})
        sector = (fundas_map.get(sym, {}) or {}).get("sector") or "Unknown"
        info = sector_ranks.get(sector) or {}
        rank = info.get("rank")
        if rank is not None and rank <= 3:
            events.append({"symbol": sym, "event": "SECTOR_TOP3", "detail": f"{sector} ranked #{rank}"})
    return events
