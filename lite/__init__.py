"""
Sovereign Lite v12 — a single-user, SQLite + yFinance-only NSE stock platform.

Phases:
  1. Data layer: 5 SQLite tables, one data source (yfinance)
  2. Scoring: Quality / Growth / Momentum / Valuation / Risk (regime-weighted)
  3. Regime: NIFTY 200-DMA + ADX + India VIX
  4. Multibagger: 100-Bagger Detector (7 rules + MB score)
  5. Dashboard: 5 screens (Dashboard, Screener, Elite Picks, Portfolio, Backtest)
"""

# Single source of truth — import VERSION from this module everywhere.
VERSION = "12.0.0"
__version__ = VERSION
