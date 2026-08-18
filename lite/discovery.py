"""
Sovereign Lite v14 — discovery engine (Emerging Leaders).

The core universe is curated and stable; the Discovery tier is where
multibaggers show up before they become obvious. This module ranks
discovery-tier names by how *now* they are:

  discovery_score = 25% RS rank + 25% RS acceleration + 25% revision proxy
                    + 15% margin expansion + 10% size factor, dampened by
                    data confidence.

RS rank and momentum are strongly correlated (both are trend), so momentum
was dropped from the composite to avoid double-counting the same signal —
revision proxy and margin expansion bring the orthogonal fundamentals view.
RS acceleration is the key timing signal: a name whose 1M relative-strength
percentile has jumped past its 3M percentile is being discovered right now —
even if it is not yet a 12M leader.
"""
from __future__ import annotations

import math
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


def _margin_score(r: dict) -> Optional[float]:
    """Map margin expansion (pp, last 4Q vs prior 4Q) onto 0-100: +1pp → 50,
    +4pp → 82, −2pp → 18. Orthogonal to the trend-heavy RS components."""
    v = r.get("margin_expansion")
    if v is None:
        return None
    return scoring.sigmoid(v, 1.0, 2.0)


def _size_score(r: dict) -> Optional[float]:
    """Log-scale size factor from market cap (₹ Cr): 10Cr → 0, 1,000Cr → 50,
    1L Cr → 100. Discovery favors names big enough to actually trade but
    small enough to run — it's a tradability gate, not a large-cap bias."""
    mc = r.get("market_cap")
    if mc is None or mc <= 0:
        return None
    return round(max(0.0, min(100.0, 100.0 * (math.log10(mc) - 1.0) / 4.0)), 1)


def discovery_score(r: dict) -> Optional[float]:
    """Emerging-leader composite for a scored record (0-100).

    Orthogonal by design: trend (RS rank) carries 25% not 55%, with revision
    proxy (25%) and margin expansion (15%) as the fundamentals view and a
    10% size gate — a 95-scoring microcap with no traded value can't dominate.
    """
    raw = scoring._weighted(
        [
            (r.get("rs_rank"), 0.25),
            (_accel_score(r), 0.25),
            (r.get("revision_score"), 0.25),
            (_margin_score(r), 0.15),
            (_size_score(r), 0.10),
        ]
    )
    if raw is None:
        return None
    conf = r.get("data_confidence")
    return round(scoring._clamp(raw * scoring.confidence_factor(conf)), 1)
