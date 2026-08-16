"""
Sovereign Lite v7 — scoring engine (Phase 2).

Every raw metric is sigmoid-normalized to 0-100 (no binary cliff thresholds).

  Quality   = ROE · ROCE · FCF margin · debt penalty
  Growth    = sales growth · profit growth · earnings acceleration
  Momentum  = 6M return · 12M return · volume expansion · RS rank · trend template
  Valuation = PE · PB (lower is better)
  Risk      = volatility · max drawdown · debt · beta (higher = safer)

Final score = w_q·Quality + w_g·Growth + w_m·Momentum + w_v·Valuation + w_r·Risk
with regime-aware weights from `lite.regime`.
"""
from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from . import db, indicators, regime as regime_mod

FINANCIAL_SECTORS = ("BANK", "FINANCIAL", "INSURANCE", "NBFC", "MUTUAL")


# ── Normalization ───────────────────────────────────────────────────────────

def sigmoid(x: Optional[float], midpoint: float, scale: float) -> Optional[float]:
    """Sigmoid-normalize a raw value to 0-100 centered at `midpoint`."""
    if x is None:
        return None
    try:
        z = (x - midpoint) / scale
        return 100.0 / (1.0 + math.exp(-z))
    except (OverflowError, ValueError, TypeError):
        return None


def _clamp(v: Optional[float], lo: float = 0.0, hi: float = 100.0) -> Optional[float]:
    if v is None:
        return None
    return max(lo, min(hi, v))


def _weighted(parts: list[tuple[Optional[float], float]]) -> Optional[float]:
    """Weighted average of available parts; None parts are skipped."""
    num = den = 0.0
    for value, w in parts:
        if value is None:
            continue
        num += value * w
        den += w
    if den <= 0:
        return None
    return _clamp(num / den)


def is_financial(sector: Optional[str]) -> bool:
    if not sector:
        return False
    s = sector.upper()
    return any(t in s for t in FINANCIAL_SECTORS)


# ── Component scores ────────────────────────────────────────────────────────

def quality_score(f: dict) -> Optional[float]:
    roe = sigmoid(f.get("roe"), 15, 8)
    roce = sigmoid(f.get("roce"), 18, 10)
    fcf = sigmoid(f.get("fcf_margin"), 5, 8)
    de = f.get("debt_equity")
    if is_financial(f.get("sector")):
        debt = 60.0  # banks/financials carry structurally high leverage
    elif de is not None:
        debt = 100 - sigmoid(de, 0.6, 0.5)
    else:
        debt = None
    return _weighted([(roe, 0.35), (roce, 0.35), (fcf, 0.15), (debt, 0.15)])


def margin_expansion_score(f: dict) -> Optional[float]:
    """Net-margin acceleration in percentage points (positive = expanding)."""
    me = f.get("margin_expansion")
    if me is None:
        return None
    return sigmoid(me, 2.0, 3.0)


def growth_score(f: dict) -> Optional[float]:
    sales = sigmoid(f.get("sales_growth"), 15, 10)
    profit = sigmoid(f.get("profit_growth"), 15, 15)
    accel = f.get("eps_accel")
    if accel is None:
        accel = sigmoid(f.get("eps_growth"), 20, 15)
    else:
        accel = _clamp(accel)
    margin = margin_expansion_score(f)
    return _weighted([(sales, 0.35), (profit, 0.30), (accel, 0.20), (margin, 0.15)])


def momentum_score(px: dict, benchmark_ret6: Optional[float]) -> Optional[float]:
    """px holds return/volume/trend metrics computed by `compute_metrics`."""
    ret6 = px.get("ret_6m")
    ret12 = px.get("ret_12m")
    vr = px.get("volume_ratio")
    ret6_score = sigmoid(ret6 * 100 if ret6 is not None else None, 10, 15)
    ret12_score = sigmoid(ret12 * 100 if ret12 is not None else None, 20, 25)
    vol_score = sigmoid((vr - 1) * 100 if vr is not None else None, 30, 50)
    raw = _weighted([(ret6_score, 0.40), (ret12_score, 0.40), (vol_score, 0.20)])

    rs_rank_score = px.get("rs_rank_score")  # 0-100 percentile vs universe
    trend_score = 100.0 if px.get("trend_ok") else (25.0 if px.get("above_200") else 0.0)
    return _weighted([(raw, 0.35), (rs_rank_score, 0.35), (trend_score, 0.30)])


def valuation_score(f: dict) -> Optional[float]:
    pe = f.get("pe")
    pb = f.get("pb")
    pe_score = None if pe is None or pe <= 0 else 100 - sigmoid(pe, 25, 20)
    pb_score = None if pb is None or pb <= 0 else 100 - sigmoid(pb, 4, 3)
    if pe_score is None and pb_score is None:
        return None
    return _weighted([(pe_score, 0.6), (pb_score, 0.4)])


def risk_score(f: dict, px: dict) -> Optional[float]:
    vol = px.get("vol")
    mdd = px.get("max_dd")
    de = f.get("debt_equity")
    beta = f.get("beta")

    vol_safety = None if vol is None else 100 - sigmoid(vol * 100, 45, 20)
    dd_safety = None if mdd is None else 100 - sigmoid(abs(mdd) * 100, 25, 15)
    if is_financial(f.get("sector")):
        debt_safety = 60.0
    elif de is not None:
        debt_safety = 100 - sigmoid(de, 0.6, 0.5)
    else:
        debt_safety = None
    beta_safety = None if beta is None else 100 - sigmoid(beta, 1.2, 0.5)
    return _weighted([(vol_safety, 0.30), (dd_safety, 0.25), (debt_safety, 0.30), (beta_safety, 0.15)])


def accumulation_score(f: dict, px: dict) -> Optional[float]:
    """Institutional-accumulation proxy from data yFinance actually provides.

    Volume expansion (35%) · market-cap growth via 12M return (35%) ·
    earnings acceleration (30%). Delivery % isn't in the Yahoo feed for NSE
    names, so heavy volume + price growth + accelerating earnings stand in
    for it.
    """
    vr = px.get("volume_ratio")
    vol = sigmoid((vr - 1) * 100, 30, 50) if vr is not None else None
    ret12 = px.get("ret_12m")
    mcap_growth = sigmoid(ret12 * 100, 15, 25) if ret12 is not None else None
    accel = f.get("eps_accel")
    accel = _clamp(accel, 0, 100) if accel is not None else None
    return _weighted([(vol, 0.35), (mcap_growth, 0.35), (accel, 0.30)])


def rs_boost_for(rs: Optional[float]) -> float:
    """Explicit relative-strength boost tiers (Minervini/O'Neil style)."""
    if rs is None:
        return 0.0
    if rs >= 95:
        return 10.0
    if rs >= 90:
        return 7.0
    if rs >= 80:
        return 4.0
    return 0.0


# ── Price-derived metrics ───────────────────────────────────────────────────

def compute_price_metrics(close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series) -> dict:
    pos = indicators.price_position(close)
    trend = indicators.trend_template(close)
    return {
        "ret_6m": indicators.returns_over(close, 126),
        "ret_12m": indicators.returns_over(close, 252),
        "ret_1m": indicators.returns_over(close, 21),
        "volume_ratio": indicators.volume_ratio(volume),
        "vol": indicators.annualized_vol(close),
        "max_dd": indicators.max_drawdown(close),
        "rsi": indicators.rsi(close),
        "adx": indicators.adx(high, low, close),
        "dist_52w_high": pos["dist_52w_high"],
        "position_52w": pos["position_52w"],
        "trend_ok": trend["ok"],
        "above_200": float(close.iloc[-1]) > trend["sma200"] if (len(close) and trend["sma200"] is not None) else False,
        "sma50": trend["sma50"],
        "sma200": trend["sma200"],
        "high_52w": pos["high_52w"],
        "price": float(close.iloc[-1]) if len(close) else None,
    }


# ── Full scoring pipeline ───────────────────────────────────────────────────

def score_symbol(symbol: str, f: dict, px: dict, benchmark_ret6: Optional[float]) -> dict:
    """Component + composite score for one symbol (weights applied by caller)."""
    return {
        "symbol": symbol,
        "quality": quality_score(f),
        "growth": growth_score(f),
        "momentum": momentum_score(px, benchmark_ret6),
        "valuation": valuation_score(f),
        "risk": risk_score(f, px),
        "accumulation": accumulation_score(f, px),
    }


def compute_scores(
    regime: dict,
    fundamentals: list[dict],
    prices: dict[str, pd.DataFrame],
) -> list[dict]:
    """
    Score the universe. `regime` is the output of `regime.detect_regime`.
    Prices are expected to have indicator columns already (see `attach_indicators`).
    Returns scored records (not yet persisted/ranked).
    """
    weights = regime["weights"]
    benchmark_ret6 = indicators.returns_over(
        regime.get("_nifty_close", pd.Series(dtype=float)), 126
    )

    # RS rank: percentiles of 6M and 12M returns within this universe,
    # blended 50/50 (a stock strong in both is a true relative leader).
    ret6_by_symbol: dict[str, Optional[float]] = {}
    ret12_by_symbol: dict[str, Optional[float]] = {}
    for symbol, df in prices.items():
        row = df.iloc[-1] if len(df) else None
        ret6_by_symbol[symbol] = row.get("ret_6m") if row is not None else None
        ret12_by_symbol[symbol] = row.get("ret_12m") if row is not None else None
    valid_ret6 = sorted(
        (r for r in ret6_by_symbol.values() if r is not None), reverse=True
    )
    valid_ret12 = sorted(
        (r for r in ret12_by_symbol.values() if r is not None), reverse=True
    )
    n6, n12 = len(valid_ret6), len(valid_ret12)

    records = []
    for f in fundamentals:
        symbol = f.get("symbol")
        df = prices.get(symbol)
        if df is None or df.empty:
            continue
        row = df.iloc[-1].to_dict()
        rs_6m = _rs_rank(ret6_by_symbol.get(symbol), valid_ret6, n6)
        rs_12m = _rs_rank(ret12_by_symbol.get(symbol), valid_ret12, n12)
        if rs_6m is None and rs_12m is None:
            rs_rank = None
        elif rs_6m is None:
            rs_rank = rs_12m
        elif rs_12m is None:
            rs_rank = rs_6m
        else:
            rs_rank = 0.5 * rs_6m + 0.5 * rs_12m
        row["rs_rank_score"] = rs_rank
        parts = score_symbol(symbol, f, row, benchmark_ret6)

        total = 0.0
        for key in ("quality", "growth", "momentum", "valuation", "risk"):
            v = parts.get(key)
            if v is not None:
                total += v * weights[key]
        total = _clamp(total + rs_boost_for(rs_rank))
        parts["score"] = round(total, 1)
        parts["regime"] = regime["regime"]
        parts["trend_ok"] = bool(row.get("trend_ok"))
        parts["above_200"] = bool(row.get("above_200"))
        parts["rs_rank"] = rs_rank
        parts["rs_6m"] = rs_6m
        parts["rs_12m"] = rs_12m
        parts["rs_boost"] = rs_boost_for(rs_rank)
        records.append(parts)
    return records


def _rs_rank(value: Optional[float], sorted_desc: list[float], n: int) -> Optional[float]:
    """Percentile rank (0-100) of a 6M return within the universe."""
    if value is None or n == 0:
        return None
    worse = sum(1 for v in sorted_desc if v <= value)
    return _clamp(worse / n * 100)


def attach_indicators(prices: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Add computed indicator columns to each price frame (in place)."""
    for symbol, df in prices.items():
        if df.empty or "close" not in df.columns:
            continue
        close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
        row = compute_price_metrics(close, high, low, volume)
        for k, v in row.items():
            df.loc[df.index[-1], k] = v
    return prices


def rank_and_persist(records: list[dict]) -> list[dict]:
    """Sort by composite score, assign ranks, persist to SQLite."""
    records.sort(key=lambda r: r.get("score") or 0, reverse=True)
    for i, r in enumerate(records, start=1):
        r["rank"] = i
    db.upsert_scores(records)
    return records
