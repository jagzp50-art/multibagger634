"""
Sovereign Lite v16 — market regime (Phase 3).

Rules (in priority order):
  HIGH_VOLATILITY  India VIX > 20
  BEAR             NIFTY < 200-DMA
  SIDEWAYS         ADX < 15
  BULL             NIFTY > 200-DMA and ADX >= 20

Each regime carries its own scoring weights (all sum to 1.0) and a suggested
equity / cash allocation for the dashboard.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from . import indicators

# Base weights (Phase 2 final formula): Quality/Growth/Momentum/Valuation/Risk
BASE_WEIGHTS = {"quality": 0.30, "growth": 0.25, "momentum": 0.20, "valuation": 0.15, "risk": 0.10}

REGIME_WEIGHTS = {
    "BULL": {"quality": 0.20, "growth": 0.30, "momentum": 0.35, "valuation": 0.10, "risk": 0.05},
    "BEAR": {"quality": 0.40, "growth": 0.15, "momentum": 0.05, "valuation": 0.30, "risk": 0.10},
    "SIDEWAYS": BASE_WEIGHTS,
    "HIGH_VOLATILITY": {"quality": 0.30, "growth": 0.20, "momentum": 0.15, "valuation": 0.15, "risk": 0.20},
}

ALLOCATION = {
    "BULL": {"equity": 90, "cash": 10, "note": "Trend + momentum regime — stay invested, ride winners."},
    "SIDEWAYS": {"equity": 60, "cash": 40, "note": "Choppy tape — wait for breakouts, keep dry powder."},
    "BEAR": {"equity": 25, "cash": 75, "note": "Index below 200-DMA — capital preservation first."},
    "HIGH_VOLATILITY": {"equity": 40, "cash": 60, "note": "VIX elevated — smaller size, wider stops."},
}

REGIME_REASONS = {
    "BULL": "NIFTY above 200-DMA with ADX ≥ 20",
    "BEAR": "NIFTY below 200-DMA",
    "SIDEWAYS": "ADX < 15 — no directional trend",
    "HIGH_VOLATILITY": "India VIX > 20",
}


def detect_regime(
    nifty: Optional[pd.DataFrame],
    vix: Optional[pd.DataFrame],
    nifty_adx: Optional[float],
) -> dict:
    """Compute the current regime + weights + allocation from index data."""
    nifty_close = nifty["Close"].dropna() if nifty is not None and not nifty.empty else pd.Series(dtype=float)
    vix_close = vix["Close"].dropna() if vix is not None and not vix.empty else pd.Series(dtype=float)

    last_nifty = float(nifty_close.iloc[-1]) if len(nifty_close) else None
    sma200 = indicators.sma(nifty_close, 200)
    if last_nifty is None or sma200 is None:
        above_200 = None
    else:
        above_200 = last_nifty > sma200
    vix_value = float(vix_close.iloc[-1]) if len(vix_close) else None

    if vix_value is not None and vix_value > 20:
        regime = "HIGH_VOLATILITY"
    elif above_200 is False:
        regime = "BEAR"
    elif nifty_adx is not None and nifty_adx < 15:
        regime = "SIDEWAYS"
    else:
        regime = "BULL"

    return {
        "regime": regime,
        "weights": REGIME_WEIGHTS[regime],
        "allocation": ALLOCATION[regime],
        "reason": REGIME_REASONS[regime],
        "vix": vix_value,
        "nifty": last_nifty,
        "nifty_200dma": sma200,
        "above_200dma": above_200,
        "adx": nifty_adx,
        "timestamp": None,  # filled by caller
    }
