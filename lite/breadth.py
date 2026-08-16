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

WEIGHTS = {"above_20": 0.10, "above_50": 0.25, "above_200": 0.40, "hl": 0.25}


def compute_breadth(frames: dict[str, object]) -> dict:
    above = {"above_20": 0, "above_50": 0, "above_200": 0}
    highs = 0
    lows = 0
    n = 0
    for sym, df in frames.items():
        if df is None or df.empty or "close" not in df.columns:
            continue
        close = df["close"].dropna()
        if len(close) < 252:
            continue
        n += 1
        last = float(close.iloc[-1])
        if last > indicators.sma(close, 20):
            above["above_20"] += 1
        if last > indicators.sma(close, 50):
            above["above_50"] += 1
        if last > indicators.sma(close, 200):
            above["above_200"] += 1
        win = close.tail(252)
        if last >= float(win.max()) * 0.98:  # within 2% of the 52-week high
            highs += 1
        if last <= float(win.min()) * 1.02:  # within 2% of the 52-week low
            lows += 1
    if n == 0:
        return {
            "above_20": None, "above_50": None, "above_200": None,
            "new_highs": None, "new_lows": None, "market_health": None, "n": 0,
        }
    out = {k: round(v / n * 100, 1) for k, v in above.items()}
    out["new_highs"] = round(highs / n * 100, 1)
    out["new_lows"] = round(lows / n * 100, 1)
    # High/low ratio → 0-100 (more highs than lows = healthier tape)
    hl = (highs - lows) / (highs + lows) if (highs + lows) else 0.0
    hl_score = (hl + 1) / 2 * 100
    health = out["above_20"] * WEIGHTS["above_20"] + out["above_50"] * WEIGHTS["above_50"] + out["above_200"] * WEIGHTS["above_200"] + hl_score * WEIGHTS["hl"]
    out["market_health"] = round(health, 1)
    out["n"] = n
    return out
