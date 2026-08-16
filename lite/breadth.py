"""
Sovereign Lite v7 — market breadth engine.

Measures how many universe members are above their 20 / 50 / 200-day moving
averages and blends them into a single market-health score:

  market_health = 0.20·above20 + 0.30·above50 + 0.50·above200

Breadth confirms (or contradicts) the index-level regime: an index above its
200-DMA with only 30% of stocks above theirs is a weak bull, not a strong one.
"""
from __future__ import annotations

from typing import Optional

from . import indicators

WEIGHTS = {"above_20": 0.20, "above_50": 0.30, "above_200": 0.50}


def compute_breadth(frames: dict[str, object]) -> dict:
    above = {"above_20": 0, "above_50": 0, "above_200": 0}
    n = 0
    for sym, df in frames.items():
        if df is None or df.empty or "close" not in df.columns:
            continue
        close = df["close"].dropna()
        if len(close) < 200:
            continue
        n += 1
        last = float(close.iloc[-1])
        if last > indicators.sma(close, 20):
            above["above_20"] += 1
        if last > indicators.sma(close, 50):
            above["above_50"] += 1
        if last > indicators.sma(close, 200):
            above["above_200"] += 1
    if n == 0:
        return {"above_20": None, "above_50": None, "above_200": None, "market_health": None, "n": 0}
    out = {k: round(v / n * 100, 1) for k, v in above.items()}
    health = sum(out[k] * w for k, w in WEIGHTS.items())
    out["market_health"] = round(health, 1)
    out["n"] = n
    return out
