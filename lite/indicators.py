"""
Sovereign Lite v17 — lightweight technical indicators (pandas only).

Implements exactly what the scoring/regime layers need:
  SMA, Wilder's ADX, RSI, realized volatility, max drawdown,
  volume expansion, 52-week range, and the Minervini trend template.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd


def sma(series: pd.Series, n: int) -> Optional[float]:
    if len(series) < n:
        return None
    return float(series.tail(n).mean())


def _wilder_smooth(values: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    if len(values) < n:
        return out
    out[n - 1] = values[:n].mean()
    for i in range(n, len(values)):
        out[i] = (out[i - 1] * (n - 1) + values[i]) / n
    return out


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> Optional[float]:
    """Wilder's Average Directional Index. Returns the latest value."""
    if len(close) < n * 2 + 1:
        return None
    h = high.values.astype(float)
    l = low.values.astype(float)
    c = close.values.astype(float)

    # Directional moves between consecutive bars (length n-1, aligned with bars 1..n-1)
    up_move = h[1:] - h[:-1]
    dn_move = l[:-1] - l[1:]
    up = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    dn = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))

    atr = _wilder_smooth(tr, n)
    up_s = _wilder_smooth(up, n)
    dn_s = _wilder_smooth(dn, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = 100 * up_s / atr
        mdi = 100 * dn_s / atr
        dx = 100 * np.abs(pdi - mdi) / np.where((pdi + mdi) == 0, np.nan, pdi + mdi)
    # Smooth DX only over its valid (non-NaN) segment
    seg = dx[~np.isnan(dx)]
    if len(seg) < n:
        return None
    adx_series = _wilder_smooth(seg, n)
    vals = adx_series[~np.isnan(adx_series)]
    if len(vals) == 0:
        return None
    return float(np.clip(vals[-1], 0, 100))


def rsi(close: pd.Series, n: int = 14) -> Optional[float]:
    if len(close) < n + 1:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, min_periods=n).mean().iloc[-1]
    avg_loss = loss.ewm(alpha=1 / n, min_periods=n).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - 100 / (1 + rs))


def returns_over(close: pd.Series, days: int) -> Optional[float]:
    """Total return over the trailing `days` trading days (as a fraction)."""
    if len(close) < days + 1:
        return None
    start = close.iloc[-days - 1]
    end = close.iloc[-1]
    if start is None or pd.isna(start) or start <= 0 or pd.isna(end):
        return None
    return float(end / start - 1)


def annualized_vol(close: pd.Series, n: int = 60) -> Optional[float]:
    if len(close) < 2:
        return None
    rets = close.pct_change().dropna().tail(n)
    if len(rets) < 2:
        return None
    return float(rets.std(ddof=1) * math.sqrt(252))


def max_drawdown(close: pd.Series) -> Optional[float]:
    if len(close) < 2:
        return None
    peak = close.cummax()
    dd = (close - peak) / peak
    return float(dd.min())


def volume_ratio(volume: pd.Series, short: int = 5, long: int = 60) -> Optional[float]:
    """Recent (short) avg volume vs longer (long) avg volume, in multiples."""
    if len(volume) < long + 1:
        return None
    short_avg = volume.tail(short).mean()
    long_avg = volume.iloc[-long - short : -short].mean() if len(volume) > long + short else volume.iloc[:-short].tail(long).mean()
    if long_avg is None or pd.isna(long_avg) or long_avg <= 0:
        return None
    return float(short_avg / long_avg)


def price_position(close: pd.Series, window: int = 252) -> dict:
    """52-week high / low distance and percentile position."""
    if len(close) < 2:
        return {"high_52w": None, "low_52w": None, "dist_52w_high": None, "position_52w": None}
    win = close.tail(window)
    high = float(win.max())
    low = float(win.min())
    last = float(close.iloc[-1])
    dist_high = (high - last) / high if high > 0 else None
    pos = (last - low) / (high - low) if high > low else None
    return {
        "high_52w": high,
        "low_52w": low,
        "dist_52w_high": dist_high,
        "position_52w": pos,
    }


def avg_traded_value(close: pd.Series, volume: pd.Series, n: int = 20) -> Optional[float]:
    """Average daily traded value in ₹ over the trailing `n` sessions
    (close × volume). This is the liquidity proxy for Kelly sizing —
    microcaps with tiny traded value get penalized even if their score is 95."""
    if len(close) < 2 or len(volume) < 2:
        return None
    c = close.tail(n)
    v = volume.tail(n)
    traded = (c * v).dropna()
    if len(traded) < 5:
        return None
    return float(traded.mean())


def liquidity_factor(avg_traded_value: Optional[float]) -> Optional[float]:
    """Map avg daily traded value (₹) onto a 0.2–1.0 factor on a log scale.

    ₹1L/day → 0.20 (untradeable), ₹10Cr/day → ~0.73, ₹100Cr+/day → 1.0.
    A name scoring 95 with ₹50L of daily traded value can no longer be
    oversized in the book.
    """
    if avg_traded_value is None or avg_traded_value <= 0:
        return None
    log_v = math.log10(avg_traded_value)
    # log10(1e5)=5 → 0.20 · log10(1e8)=8 → 1.0, linear in log-space
    return round(max(0.2, min(1.0, 0.2 + 0.8 * (log_v - 5.0) / 3.0)), 3)


def trend_template(close: pd.Series) -> dict:
    """Minervini trend template: price > 50SMA > 150SMA > 200SMA."""
    s50 = sma(close, 50)
    s150 = sma(close, 150)
    s200 = sma(close, 200)
    last = float(close.iloc[-1]) if len(close) else None
    ok = (
        last is not None
        and all(v is not None for v in (s50, s150, s200))
        and last > s50 > s150 > s200
    )
    return {"ok": bool(ok), "sma50": s50, "sma150": s150, "sma200": s200}


def to_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Build a price DataFrame from DB rows (dicts with date/open/high/low/close/volume)."""
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
