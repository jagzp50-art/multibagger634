"""
Sovereign Lite v7 — portfolio construction layer.

Position Score = 0.40 Quality + 0.30 MB Score + 0.20 RS Rank + 0.10 Risk
(higher = better). Allocation weights are rank-based: the #1 position gets
the largest slice of the regime's equity allocation, tapering linearly —
top conviction gets the most capital, no equal-weighting.

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
    """Kelly-Lite conviction: score · quality · (1/volatility), normalized.

    High quality + low volatility + strong RS deserve more capital than an
    equally-scored but volatile, low-quality name.
    """
    pos = r.get("pos_score") or 0.0
    q = (r.get("quality") or 50.0) / 100.0
    vol = r.get("vol")
    if vol is None or vol <= 0:
        vol = 0.30
    vol_factor = 0.15 / (0.15 + vol)
    return max(0.0, pos * q * vol_factor)


def build_allocation(
    records: list[dict],
    equity_pct: float,
    top_n: int = 8,
    mode: str = "kelly",
) -> list[dict]:
    """Conviction-weighted allocation over the top-N by position score.

    mode="kelly" (default): weight ∝ pos_score · quality · 1/vol — high
    quality, low volatility, strong RS get the largest slices.
    mode="conviction": weight ∝ pos_score² — pure score tapering.
    Both normalize so the book sums to equity_pct.
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
                "sector": r.get("sector"),
                "pos_score": r.get("pos_score"),
                "mb_bucket": r.get("mb_bucket"),
                "weight_pct": round(w, 2),
                "mode": mode,
            }
        )
    return out
