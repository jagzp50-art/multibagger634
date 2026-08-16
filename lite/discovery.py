"""
Sovereign Lite v11 — discovery engine (Emerging Leaders).

The core universe is curated and stable; the Discovery tier is where
multibaggers show up before they become obvious. This module ranks
discovery-tier names by how *now* they are:

  discovery_score = 35% RS rank + 25% RS acceleration + 20% revision proxy
                    + 20% momentum, dampened by data confidence.

RS acceleration is the key signal: a name whose 1M relative-strength
percentile has jumped past its 3M percentile is being discovered right now —
even if it is not yet a 12M leader.
"""
from __future__ import annotations

from typing import Optional

from . import scoring


def rs_acceleration(r: dict) -> Optional[float]:
    """1M RS percentile minus 3M RS percentile. Positive = accelerating:
    the stock is heating up faster than its own recent history."""
    a = r.get("rs_1m")
    b = r.get("rs_3m")
    if a is None or b is None:
        return None
    return round(a - b, 1)


def _accel_score(r: dict) -> Optional[float]:
    """Map the rs_acceleration spread (−100..100) onto 0–100 centered at 50."""
    a = rs_acceleration(r)
    if a is None:
        return None
    return max(0.0, min(100.0, 50.0 + a / 2.0))


def discovery_score(r: dict) -> Optional[float]:
    """Emerging-leader composite for a scored record (0-100)."""
    raw = scoring._weighted(
        [
            (r.get("rs_rank"), 0.35),
            (_accel_score(r), 0.25),
            (r.get("revision_score"), 0.20),
            (r.get("momentum"), 0.20),
        ]
    )
    if raw is None:
        return None
    conf = r.get("data_confidence")
    return round(scoring._clamp(raw * scoring.confidence_factor(conf)), 1)
