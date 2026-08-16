"""
Sovereign Lite v12 — scoring engine (Phase 2).

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


def confidence_factor(confidence: Optional[float]) -> float:
    """Multiplier that punishes incomplete data, convex in coverage:
    confidence = (coverage/100) ** 1.5. A stock with zero fundamentals scores
    0 (not 50%), 50% coverage → 0.35, 80% → 0.72, 100% → 1.0."""
    if confidence is None:
        return 1.0
    c = max(0.0, min(100.0, float(confidence))) / 100.0
    return round(c ** 1.5, 4)


def apply_confidence(score: Optional[float], confidence: Optional[float]) -> Optional[float]:
    if score is None:
        return None
    return _clamp(score * confidence_factor(confidence))


def is_financial(sector: Optional[str]) -> bool:
    if not sector:
        return False
    s = sector.upper()
    return any(t in s for t in FINANCIAL_SECTORS)


# ── Component scores ────────────────────────────────────────────────────────

def institutional_quality_score(f: dict) -> Optional[float]:
    """Consistency across the last ~5 fiscal years — what institutions actually
    pay for. ROE 22/23/21/24/22 beats ROE 8/35/12/40/10 even at the same mean."""
    parts = [
        (f.get("roe_stability"), 0.30),
        (f.get("profit_stability"), 0.25),
        (f.get("sales_stability"), 0.15),
        (f.get("margin_stability"), 0.15),
        (f.get("fcf_stability"), 0.15),
    ]
    return _weighted(parts)


def quality_score(f: dict) -> Optional[float]:
    roe = sigmoid(f.get("roe"), 15, 8)
    roce = sigmoid(f.get("roce"), 18, 10)
    fcf = sigmoid(f.get("fcf_margin"), 5, 8)
    stability = institutional_quality_score(f)
    # Quality of earnings: CFO/PAT ≈ 1 is healthy; profits without cash is a red flag.
    cfo_pat = sigmoid(f.get("cfo_pat_ratio"), 1.0, 0.5)
    # Accrual ratio (Sloan): (NI − CFO) / Total Assets. High positive accruals
    # mean earnings quality is low — a proven fraud predictor.
    acc = f.get("accrual_ratio")
    accrual = (100 - sigmoid(acc * 100, 6, 4)) if acc is not None else None
    de = f.get("debt_equity")
    if is_financial(f.get("sector")):
        debt = 60.0  # banks/financials carry structurally high leverage
    elif de is not None:
        debt = 100 - sigmoid(de, 0.6, 0.5)
    else:
        debt = None
    return _weighted(
        [(roe, 0.26), (roce, 0.26), (fcf, 0.09), (cfo_pat, 0.08), (accrual, 0.07), (debt, 0.10), (stability, 0.14)]
    )


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
    # CFO growth — earnings backed by growing operating cash flow.
    cfo_growth = sigmoid(f.get("cfo_growth") * 100, 10, 15) if f.get("cfo_growth") is not None else None
    return _weighted(
        [(sales, 0.32), (profit, 0.28), (accel, 0.18), (margin, 0.12), (cfo_growth, 0.10)]
    )


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
    # Earnings / sales volatility (std of YoY growth, 5y) → stability of the franchise
    ev = f.get("earnings_vol")
    sv = f.get("sales_vol")
    earnings_safety = None if ev is None else 100 - sigmoid(ev * 100, 40, 20)
    sales_safety = None if sv is None else 100 - sigmoid(sv * 100, 40, 20)
    return _weighted(
        [
            (vol_safety, 0.25),
            (dd_safety, 0.20),
            (debt_safety, 0.25),
            (beta_safety, 0.10),
            (earnings_safety, 0.10),
            (sales_safety, 0.10),
        ]
    )


def accumulation_score(f: dict, px: dict) -> Optional[float]:
    """Institutional-accumulation proxy from data yFinance actually provides.

    Volume expansion (35%) · price strength via 12M return (35%) · proximity
    to the 52-week high (30%). Delivery % isn't in the Yahoo feed for NSE
    names, so heavy volume + price strength + making new highs stand in for
    institutional buying.
    """
    vr = px.get("volume_ratio")
    vol = sigmoid((vr - 1) * 100, 30, 50) if vr is not None else None
    ret12 = px.get("ret_12m")
    price_strength = sigmoid(ret12 * 100, 15, 25) if ret12 is not None else None
    dist = px.get("dist_52w_high")
    proximity = None
    if dist is not None:
        proximity = 100 - sigmoid(dist * 100, 10, 8)
    return _weighted([(vol, 0.35), (price_strength, 0.35), (proximity, 0.30)])


def revision_score(f: dict) -> Optional[float]:
    """Earnings-revision proxy built from free data — institutions buy future
    earnings, so acceleration signals carry alpha:

    35% EPS acceleration · 30% revenue acceleration · 20% margin expansion ·
    15% CFO growth.

    CFO growth is the confirmation layer: earnings backed by growing
    operating cash flow are real, not accrual-driven.
    """
    accel = f.get("eps_accel")
    accel = _clamp(accel, 0, 100) if accel is not None else None
    rev = f.get("rev_accel")
    rev = _clamp(rev, 0, 100) if rev is not None else None
    margin = margin_expansion_score(f)
    cfo = f.get("cfo_growth")
    cfo = sigmoid(cfo * 100, 10, 15) if cfo is not None else None
    return _weighted([(accel, 0.35), (rev, 0.30), (margin, 0.20), (cfo, 0.15)])


def opportunity_score(row: dict) -> Optional[float]:
    """Screener's primary ranking (Opportunity 2.0): a strong idea that is also
    cheap-ish, consistent, and inside a strong sector:

    30% MB score · 25% RS rank · 20% earnings acceleration · 15% Quality ·
    10% sector strength.
    """
    mb = row.get("mb_score")
    rs = row.get("rs_rank")
    accel = row.get("eps_accel")
    accel = _clamp(accel, 0, 100) if accel is not None else None
    quality = row.get("quality")
    sector = row.get("sector_strength")
    sector = _clamp(sector, 0, 100) if sector is not None else None
    return _weighted([(mb, 0.30), (rs, 0.25), (accel, 0.20), (quality, 0.15), (sector, 0.10)])


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
        "ret_3m": indicators.returns_over(close, 63),
        "volume_ratio": indicators.volume_ratio(volume),
        "vol": indicators.annualized_vol(close),
        "max_dd": indicators.max_drawdown(close),
        "avg_traded_value": indicators.avg_traded_value(close, volume),
        "liquidity": indicators.liquidity_factor(indicators.avg_traded_value(close, volume)),
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

    # Multi-factor relative strength: percentile rank the universe on 1M, 3M,
    # 6M and 12M returns, blended 20/20/30/30 — recent acceleration and
    # longer leadership both count (Minervini / O'Neil style).
    HORIZONS = [
        ("ret_1m", 0.20),
        ("ret_3m", 0.20),
        ("ret_6m", 0.30),
        ("ret_12m", 0.30),
    ]
    ret_by_symbol: dict[str, dict[str, Optional[float]]] = {}
    for symbol, df in prices.items():
        row = df.iloc[-1] if len(df) else None
        ret_by_symbol[symbol] = {
            key: (row.get(key) if row is not None else None) for key, _ in HORIZONS
        }
    valid = {
        key: sorted(
            (r[key] for r in ret_by_symbol.values() if r[key] is not None), reverse=True
        )
        for key, _ in HORIZONS
    }

    records = []
    for f in fundamentals:
        symbol = f.get("symbol")
        df = prices.get(symbol)
        if df is None or df.empty:
            continue
        row = df.iloc[-1].to_dict()
        rs_parts: list[tuple[Optional[float], float]] = []
        for key, w in HORIZONS:
            val = ret_by_symbol.get(symbol, {}).get(key)
            rs_h = _rs_rank(val, valid[key], len(valid[key]))
            row[f"rs_{key}".replace("ret_", "")] = rs_h  # rs_1m / rs_3m / rs_6m / rs_12m
            if rs_h is not None:
                rs_parts.append((rs_h, w))
        rs_rank = _weighted(rs_parts)
        # RS stability: a flat, consistent relative-strength profile (90/91/92/93)
        # beats an erratic one (20/10/99/15) even at the same blended rank.
        # Consistency = 100 − 2·σ of the four horizon percentiles; blended 15%
        # into the RS rank when ≥3 horizons are present.
        cons = [v for _, v in rs_parts if v is not None]
        if len(cons) >= 3:
            mean = sum(cons) / len(cons)
            var = sum((v - mean) ** 2 for v in cons) / len(cons)
            row["rs_consistency"] = round(max(0.0, min(100.0, 100.0 - 2.0 * math.sqrt(var))), 1)
            rs_rank = _weighted([(rs_rank, 0.85), (row["rs_consistency"], 0.15)])
        row["rs_rank_score"] = rs_rank
        parts = score_symbol(symbol, f, row, benchmark_ret6)

        total = 0.0
        for key in ("quality", "growth", "momentum", "valuation", "risk"):
            v = parts.get(key)
            if v is not None:
                total += v * weights[key]
        revision = revision_score(f)
        if revision is not None:
            # Revision proxy (future-earnings view) earns a dedicated 10%
            # slice of the composite — the five regime factors share the
            # remaining 90% proportionally.
            total = 0.90 * total + 0.10 * revision
        total = _clamp(total + rs_boost_for(rs_rank))
        # Data-confidence dampener: partial fundamentals can't rank as highly.
        confidence = f.get("data_confidence")
        conf_factor = confidence_factor(confidence)
        total = _clamp(total * conf_factor)
        # Factor attribution: each factor's contribution to the final score.
        # When revision exists it owns a 10% slice, so the five regime
        # factors scale by 0.90 — the bars always sum to the score shown.
        contrib: dict[str, Optional[float]] = {}
        five_scale = 0.90 if revision is not None else 1.0
        for key in ("quality", "growth", "momentum", "valuation", "risk"):
            v = parts.get(key)
            contrib[key] = round(v * weights[key] * five_scale * conf_factor, 1) if v is not None else None
        rs_boost = rs_boost_for(rs_rank)
        contrib["rs_boost"] = round(rs_boost * conf_factor, 1) if rs_boost else None
        if revision is not None:
            contrib["revision"] = round(revision * 0.10 * conf_factor, 1)
        contrib["sector_boost"] = None  # attached post-rotation by the pipeline
        parts["score"] = round(total, 1)
        parts["regime"] = regime["regime"]
        parts["trend_ok"] = bool(row.get("trend_ok"))
        parts["above_200"] = bool(row.get("above_200"))
        parts["rs_rank"] = rs_rank
        parts["rs_1m"] = row.get("rs_1m")
        parts["rs_3m"] = row.get("rs_3m")
        parts["rs_6m"] = row.get("rs_6m")
        parts["rs_12m"] = row.get("rs_12m")
        parts["rs_boost"] = rs_boost
        parts["data_confidence"] = confidence
        parts["institutional_quality"] = institutional_quality_score(f)
        parts["revision_score"] = revision
        parts["factor_contributions"] = contrib
        parts["vol"] = row.get("vol")
        parts["max_dd"] = row.get("max_dd")
        parts["liquidity"] = row.get("liquidity")
        parts["rs_consistency"] = row.get("rs_consistency")
        parts["margin_expansion"] = f.get("margin_expansion")
        parts["market_cap"] = f.get("market_cap")
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
