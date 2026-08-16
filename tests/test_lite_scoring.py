"""
Unit tests for Sovereign Lite v7 (lite/scoring, lite/regime, lite/multibagger).

Run:  python3 -m pytest tests/test_lite_scoring.py -q
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lite import indicators, multibagger, regime as regime_mod, scoring  # noqa: E402


# ── Normalization ───────────────────────────────────────────────────────────

def test_sigmoid_midpoint_is_50():
    assert scoring.sigmoid(15, 15, 8) == pytest.approx(50.0, abs=1.0)
    assert scoring.sigmoid(0, 15, 8) < 20
    assert scoring.sigmoid(30, 15, 8) > 80


def test_sigmoid_monotonic_and_clamped():
    xs = [-100, -10, 0, 10, 50, 500]
    ys = [scoring.sigmoid(x, 15, 8) for x in xs]
    assert ys == sorted(ys)
    assert all(0 <= y <= 100 for y in ys)


def test_weighted_skips_none():
    assert scoring._weighted([(None, 1.0), (50.0, 1.0)]) == pytest.approx(50.0)
    assert scoring._weighted([(None, 1.0)]) is None
    assert scoring._weighted([(20.0, 0.5), (40.0, 0.5)]) == pytest.approx(30.0)


# ── Components ──────────────────────────────────────────────────────────────

def test_quality_score_ranks_good_fundamentals_higher():
    good = {"roe": 25, "roce": 30, "fcf_margin": 12, "debt_equity": 0.2, "sector": "Technology"}
    bad = {"roe": 5, "roce": 8, "fcf_margin": -5, "debt_equity": 2.5, "sector": "Technology"}
    assert scoring.quality_score(good) > scoring.quality_score(bad) + 30


def test_financial_sector_debt_neutral():
    fin = {"roe": 20, "roce": 15, "fcf_margin": 5, "debt_equity": 8.0, "sector": "Banks"}
    nonfin = {"roe": 20, "roce": 15, "fcf_margin": 5, "debt_equity": 8.0, "sector": "Technology"}
    assert scoring.quality_score(fin) > scoring.quality_score(nonfin)


def test_valuation_lower_pe_scores_higher():
    cheap = {"pe": 10, "pb": 1.5}
    dear = {"pe": 80, "pb": 12}
    assert scoring.valuation_score(cheap) > scoring.valuation_score(dear) + 20


def test_margin_expansion_boosts_growth():
    expanding = {"sales_growth": 15, "profit_growth": 15, "eps_accel": 50, "margin_expansion": 8.0}
    contracting = {"sales_growth": 15, "profit_growth": 15, "eps_accel": 50, "margin_expansion": -8.0}
    assert scoring.growth_score(expanding) > scoring.growth_score(contracting) + 10


def test_momentum_trend_boost():
    up_trend = {"ret_6m": 0.3, "ret_12m": 0.5, "volume_ratio": 1.8, "rs_rank_score": 90.0, "trend_ok": True, "above_200": True}
    down = {"ret_6m": -0.2, "ret_12m": -0.1, "volume_ratio": 0.6, "rs_rank_score": 10.0, "trend_ok": False, "above_200": False}
    assert scoring.momentum_score(up_trend, None) > scoring.momentum_score(down, None) + 25


# ── Regime ──────────────────────────────────────────────────────────────────

def _mk_index(closes, highs=None, lows=None):
    import pandas as pd

    idx = pd.date_range("2023-01-01", periods=len(closes), freq="B")
    df = pd.DataFrame(
        {
            "Open": closes,
            "High": highs or closes,
            "Low": lows or closes,
            "Close": closes,
            "Volume": [1_000_000] * len(closes),
        },
        index=idx,
    )
    return df


def test_regime_bear_when_below_200dma():
    # Flat 100 → crash to 50: price below its 200-SMA
    closes = [100.0] * 250 + [50.0] * 50
    nifty = _mk_index(closes)
    import pandas as pd

    r = regime_mod.detect_regime(nifty, pd.DataFrame(), None)
    assert r["regime"] == "BEAR"
    assert r["above_200dma"] is False


def test_regime_bull_when_above_with_high_adx():
    closes = [100.0 + i * 0.5 for i in range(260)]  # sustained uptrend, 200+ bars
    nifty = _mk_index(closes)
    r = regime_mod.detect_regime(nifty, None, 25.0)
    assert r["regime"] == "BULL"
    assert r["above_200dma"] is True


def test_regime_high_vol_when_vix_high():
    import pandas as pd

    closes = list(range(100, 200))
    nifty = _mk_index(closes)
    vix = _mk_index([25.0] * 10, [26.0] * 10, [24.0] * 10)
    r = regime_mod.detect_regime(nifty, vix, 30.0)
    assert r["regime"] == "HIGH_VOLATILITY"


def test_regime_weights_sum_to_one():
    for name, w in regime_mod.REGIME_WEIGHTS.items():
        assert sum(w.values()) == pytest.approx(1.0), name


# ── Indicators ──────────────────────────────────────────────────────────────

def test_adx_on_trending_series_is_high():
    import pandas as pd

    closes = [float(i) for i in range(120)]
    high = [c + 1 for c in closes]
    low = [c - 1 for c in closes]
    s = pd.Series(closes)
    a = indicators.adx(pd.Series(high), pd.Series(low), s)
    assert a is not None and a > 20


def test_trend_template_requires_200_bars():
    import pandas as pd

    short = pd.Series([float(i) for i in range(50)])
    assert indicators.trend_template(short)["ok"] is False  # not enough history


# ── Multibagger ─────────────────────────────────────────────────────────────

def test_bucket_boundaries():
    assert multibagger.bucket_for(95) == "MULTIBAGGER"
    assert multibagger.bucket_for(80) == "STRONG"
    assert multibagger.bucket_for(70) == "EMERGING"
    assert multibagger.bucket_for(30) == "WATCHLIST"


def test_mb_weights_sum_to_one():
    assert sum(multibagger.MB_WEIGHTS.values()) == pytest.approx(1.0)


def test_detect_attaches_rules_and_bucket():
    rec = {
        "symbol": "TEST.NS",
        "score": 80.0,
        "growth": 90.0,
        "quality": 95.0,
        "momentum": 85.0,
        "mb_ownership": 70.0,
        "rs_rank": 99.0,
    }
    f = {
        "sales_growth": 40.0,
        "roce": 30.0,
        "debt_equity": 0.1,
        "profit_growth": 30.0,
        "eps_accel": 95.0,
        "margin_expansion": 10.0,
        "pe": 12.0,
        "pb": 1.0,
        "market_cap": 3000.0,
    }
    px = {"dist_52w_high": 0.05, "volume_ratio": 2.0}
    out = multibagger.detect([rec], {"TEST.NS": f}, {"TEST.NS": px})
    assert len(out) == 1
    row = out[0]
    assert row["mb_rules_passed"] == 7
    assert row["mb_score"] > 75
    assert row["mb_bucket"] in ("STRONG", "MULTIBAGGER")
    assert len(row["mb_checklist"]) == 7


def test_mb_uses_accel_over_single_quarter_growth():
    # Weak single-quarter growth but strong acceleration → accel path wins
    rec = {"symbol": "T.NS", "score": 60.0, "growth": 60.0, "quality": 60.0, "momentum": 60.0, "mb_ownership": 60.0, "rs_rank": 60.0}
    f = {"eps_accel": 90.0, "eps_growth": 5.0, "sales_growth": 20.0, "roce": 20.0, "margin_expansion": 5.0, "debt_equity": 0.3, "pe": 15.0, "pb": 2.0}
    out = multibagger.detect([rec], {"T.NS": f}, {"T.NS": {}})
    assert out[0]["mb_score"] > 60


# ── Score history (trends) ──────────────────────────────────────────────────

def test_score_history_snapshot_and_previous(tmp_path, monkeypatch):
    import lite.db as db_mod

    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "trends.db"))
    db_mod.init_db()

    recs = [
        {"symbol": "A.NS", "score": 70.0, "rank": 1, "mb_score": 80.0, "mb_bucket": "STRONG", "trend_ok": True},
        {"symbol": "B.NS", "score": 50.0, "rank": 2, "mb_score": 40.0, "mb_bucket": "WATCHLIST", "trend_ok": False},
    ]
    db_mod.snapshot_scores(recs, "2026-08-15", "BULL")
    assert db_mod.latest_score_snapshot()["A.NS"]["score"] == 70.0
    assert db_mod.previous_score_snapshot() == {}  # only one snapshot yet

    recs2 = [
        {"symbol": "A.NS", "score": 74.0, "rank": 1, "mb_score": 84.0, "mb_bucket": "STRONG", "trend_ok": True},
        {"symbol": "B.NS", "score": 55.0, "rank": 2, "mb_score": 45.0, "mb_bucket": "WATCHLIST", "trend_ok": False},
    ]
    db_mod.snapshot_scores(recs2, "2026-08-16", "BULL")
    assert db_mod.previous_score_snapshot()["A.NS"]["score"] == 70.0
    hist = db_mod.score_history_for("A.NS")
    assert [h["score"] for h in hist] == [70.0, 74.0]
