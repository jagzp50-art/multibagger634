"""
Unit tests for Sovereign Lite v9 (lite/scoring, lite/regime, lite/multibagger,
lite/portfolio, lite/backtest, lite/breadth, lite/db, lite/universe).

Run:  python3 -m pytest tests/test_lite_scoring.py -q
"""
import math
import os
import sys

import pandas as pd
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

# ── Relative strength (6M/12M blend + boost) ───────────────────────────────

def test_rs_boost_tiers():
    assert scoring.rs_boost_for(96) == 10.0
    assert scoring.rs_boost_for(95) == 10.0
    assert scoring.rs_boost_for(92) == 7.0
    assert scoring.rs_boost_for(85) == 4.0
    assert scoring.rs_boost_for(79) == 0.0
    assert scoring.rs_boost_for(None) == 0.0


def test_accumulation_score_ranks_accumulating_stocks_higher():
    accumulating = {"volume_ratio": 2.5, "ret_12m": 0.5, "dist_52w_high": 0.03, "eps_accel": 90.0}
    fading = {"volume_ratio": 0.5, "ret_12m": -0.3, "dist_52w_high": 0.45, "eps_accel": 10.0}
    assert scoring.accumulation_score(accumulating, accumulating) > scoring.accumulation_score(fading, fading) + 30


# ── Portfolio construction + opportunity score ──────────────────────────────

def test_position_score_weights():
    from lite import portfolio

    # 0.4*90 + 0.3*85 + 0.2*95 + 0.1*80 = 88.5
    assert portfolio.position_score(90, 85, 95, 80) == pytest.approx(88.5, abs=0.01)
    weak = portfolio.position_score(30, 20, 10, 30)
    assert portfolio.position_score(90, 85, 95, 80) > weak + 30
    assert portfolio.position_score(None, 85, 95, 80) is not None  # None-safe
    assert portfolio.position_score(None, None, None, None) is None


def test_build_allocation_rank_weighted():
    from lite import portfolio

    recs = [
        {"symbol": f"S{i}.NS", "name": f"S{i}", "sector": f"SECT{i % 3}", "pos_score": 90 - i, "mb_bucket": "STRONG"}
        for i in range(5)
    ]
    # Cap disabled so this test isolates the rank tapering, not sector caps.
    alloc = portfolio.build_allocation(recs, equity_pct=60, top_n=5, max_sector_weight=100)
    assert len(alloc) == 5
    assert abs(sum(a["weight_pct"] for a in alloc) - 60.0) < 0.01
    assert alloc[0]["weight_pct"] > alloc[-1]["weight_pct"]  # top gets the most
    assert alloc[0]["symbol"] == "S0.NS"


def test_opportunity_score_formula():
    # Opportunity 2.0: 30% MB · 25% RS · 20% accel · 15% Quality · 10% sector strength
    row = {"mb_score": 90, "rs_rank": 95, "eps_accel": 80, "quality": 85, "sector_strength": 70}
    assert scoring.opportunity_score(row) == pytest.approx(0.30 * 90 + 0.25 * 95 + 0.20 * 80 + 0.15 * 85 + 0.10 * 70, abs=0.01)
    weak = {"mb_score": 30, "rs_rank": 20, "eps_accel": 10, "quality": 20, "sector_strength": 30}
    assert scoring.opportunity_score(weak) < 30
    assert scoring.opportunity_score({"mb_score": 50}) is not None  # None-safe
    assert scoring.opportunity_score({}) is None


# ── Sector rotation ─────────────────────────────────────────────────────────

def test_sector_rotation_boosts_strong_sectors():
    from lite import rotation

    fundas = {"A.NS": {"sector": "IT"}, "B.NS": {"sector": "IT"}, "C.NS": {"sector": "CEMENT"}, "D.NS": {"sector": "FMCG"}}
    records = [
        {"symbol": "A.NS", "rs_rank": 95.0, "growth": 90.0, "score": 60.0},
        {"symbol": "B.NS", "rs_rank": 90.0, "growth": 85.0, "score": 55.0},
        {"symbol": "C.NS", "rs_rank": 10.0, "growth": 20.0, "score": 50.0},
        {"symbol": "D.NS", "rs_rank": 50.0, "growth": 55.0, "score": 52.0},
    ]
    out = rotation.apply_sector_rotation(records, fundas)
    by = {r["symbol"]: r for r in out}
    assert by["A.NS"]["sector_boost"] > 0
    assert by["C.NS"]["sector_boost"] < 0
    assert by["A.NS"]["score"] > 60.0  # boosted
    assert by["C.NS"]["score"] < 50.0  # penalized
    secs = rotation.rank_sectors(records, fundas)
    assert secs["IT"]["rank"] == 1
    assert secs["CEMENT"]["rank"] == 3


def test_compute_scores_sets_rs_blend_boost_and_ranking():
    import pandas as pd

    idx = pd.date_range("2024-01-01", periods=400, freq="B")

    def frame(closes):
        s = pd.Series(closes, index=idx)
        return pd.DataFrame({"close": s, "high": s + 1, "low": s - 1, "volume": pd.Series([1_000_000] * 400, index=idx)})

    rising = frame([100 + i * 0.2 for i in range(400)])
    falling = frame([200 - i * 0.2 for i in range(400)])
    prices = {"WIN.NS": rising, "LOSE.NS": falling}
    scoring.attach_indicators(prices)
    fundas = [
        {"symbol": "WIN.NS", "roe": 20, "roce": 25, "debt_equity": 0.3, "sales_growth": 15, "profit_growth": 15, "pe": 20, "pb": 3, "sector": "Technology"},
        {"symbol": "LOSE.NS", "roe": 10, "roce": 10, "debt_equity": 1.5, "sales_growth": 5, "profit_growth": 5, "pe": 40, "pb": 6, "sector": "Technology"},
    ]
    regime = {"regime": "BULL", "weights": {"quality": 0.25, "growth": 0.30, "momentum": 0.35, "valuation": 0.05, "risk": 0.05}}
    recs = scoring.compute_scores(regime, fundas, prices)
    by = {r["symbol"]: r for r in recs}
    win, lose = by["WIN.NS"], by["LOSE.NS"]
    assert win["rs_6m"] is not None and win["rs_12m"] is not None
    assert win["rs_rank"] > lose["rs_rank"]
    assert win["rs_boost"] >= lose["rs_boost"]
    assert win["accumulation"] is not None
    assert win["score"] > lose["score"]


# ── Earnings acceleration (Revenue/EPS/PAT trends) ──────────────────────────

def test_trend_score_rewards_acceleration():
    from lite import data

    accelerating = [0.10, 0.18, 0.25, 0.38]
    flat = [0.20, 0.20, 0.20, 0.20]
    decelerating = [0.38, 0.25, 0.18, 0.10]
    assert data._trend_score(accelerating) > data._trend_score(flat) > data._trend_score(decelerating)
    assert data._trend_score([0.20]) is None  # needs >= 2 points


# ── Data confidence + institutional quality (Phases 1-2) ────────────────────

def test_confidence_factor_penalizes_partial_data():
    assert scoring.confidence_factor(100) == pytest.approx(1.0)
    assert scoring.confidence_factor(50) == pytest.approx(0.75)
    assert scoring.confidence_factor(0) == pytest.approx(0.5)
    assert scoring.confidence_factor(None) == 1.0
    assert scoring.apply_confidence(80.0, 50) == pytest.approx(60.0)
    assert scoring.apply_confidence(80.0, None) == pytest.approx(80.0)


def test_institutional_quality_prefers_stability():
    stable = {"roe_stability": 95, "profit_stability": 92, "sales_stability": 90, "margin_stability": 88, "fcf_stability": 85}
    volatile = {"roe_stability": 40, "profit_stability": 35, "sales_stability": 30, "margin_stability": 25, "fcf_stability": 20}
    assert scoring.institutional_quality_score(stable) > scoring.institutional_quality_score(volatile) + 30
    assert scoring.institutional_quality_score({}) is None


def test_quality_includes_stability():
    stable = {
        "roe": 20, "roce": 25, "fcf_margin": 8, "debt_equity": 0.3, "sector": "Technology",
        "roe_stability": 95, "profit_stability": 90, "sales_stability": 90, "margin_stability": 90, "fcf_stability": 90,
    }
    unstable = {"roe": 20, "roce": 25, "fcf_margin": 8, "debt_equity": 0.3, "sector": "Technology"}
    assert scoring.quality_score(stable) > scoring.quality_score(unstable)


def test_revision_score_rewards_acceleration():
    strong = {"eps_accel": 90, "rev_accel": 85, "margin_expansion": 6.0, "revenue_accel_annual": 80}
    weak = {"eps_accel": 15, "rev_accel": 10, "margin_expansion": -5.0, "revenue_accel_annual": 10}
    assert scoring.revision_score(strong) > scoring.revision_score(weak) + 30
    assert scoring.revision_score({}) is None


def test_stability_and_cagr_helpers():
    from lite import data

    assert data._stability_score([22, 23, 21, 24, 22]) > data._stability_score([8, 35, 12, 40, 10])
    assert data._stability_score([10, 20]) is None  # needs >= 3 points
    assert data._cagr([100, 121]) == pytest.approx(0.21, abs=0.01)
    assert data._cagr([-5, 10]) is None  # negative base -> None
    assert data._growth_vol([100, 110, 121, 133]) is not None


# ── Compounder (Phase 10) ───────────────────────────────────────────────────

def test_compounder_prefers_consistent_5y_compounding():
    from lite import multibagger

    good = {"roce_stability": 90, "sales_cagr_5y": 0.20, "profit_cagr_5y": 0.25, "fcf_cagr_5y": 0.20, "debt_trend": 80}
    weak = {"roce_stability": 30, "sales_cagr_5y": 0.02, "profit_cagr_5y": 0.0, "fcf_cagr_5y": -0.05, "debt_trend": 20}
    assert multibagger.compounder_score(good) > multibagger.compounder_score(weak) + 20
    assert multibagger.compounder_score({}) is None


def test_mb_v3_attaches_compounder_and_uses_it():
    from lite import multibagger

    rec = {"symbol": "C.NS", "score": 70.0, "growth": 70.0, "quality": 70.0, "momentum": 70.0, "mb_ownership": 70.0, "rs_rank": 70.0}
    f = {
        "eps_accel": 80.0, "sales_growth": 25.0, "roce": 22.0, "margin_expansion": 6.0,
        "debt_equity": 0.2, "pe": 15.0, "pb": 2.0, "roce_stability": 85.0, "sales_cagr_5y": 0.20,
        "profit_cagr_5y": 0.22, "fcf_cagr_5y": 0.18, "debt_trend": 75.0,
    }
    out = multibagger.detect([rec], {"C.NS": f}, {"C.NS": {}})
    row = out[0]
    assert row["compounder_score"] is not None
    assert row["compounder_score"] > 50
    assert row["mb_score"] > 60


# ── Walk-forward (Phase 8) ──────────────────────────────────────────────────

def test_walk_forward_returns_folds_and_aggregate():
    import pandas as pd

    from lite import backtest

    idx = pd.date_range("2021-01-01", periods=1150, freq="B")
    n = len(idx)
    winner = pd.Series([100 * (1.0009 ** i) for i in range(n)], index=idx)
    flat = pd.Series([100.0] * n, index=idx)
    frames = {
        "WIN.NS": pd.DataFrame({"close": winner}),
        "FLAT.NS": pd.DataFrame({"close": flat}),
    }
    res = backtest.walk_forward(frames, {"folds": 3, "fold_months": 12, "top_n": 1})
    assert len(res["folds"]) == 3
    assert res["summary"]["folds_evaluated"] == 3
    assert res["summary"]["hit_rate_pct"] is not None
    assert res["summary"]["avg_return_pct"] is not None
    assert res["summary"]["worst_max_drawdown_pct"] is not None
    for f in res["folds"]:
        assert f["window"] and f["net_return_pct"] is not None


def test_walk_forward_requires_history():
    import pandas as pd

    from lite import backtest

    idx = pd.date_range("2024-01-01", periods=300, freq="B")
    frames = {"A.NS": pd.DataFrame({"close": pd.Series([100.0] * 300, index=idx)})}
    res = backtest.walk_forward(frames, {"folds": 3})
    assert res["summary"].get("error")  # too little data
    assert not res.get("folds")


# ── Quality of earnings (Phase 6) ───────────────────────────────────────────

def test_cfo_pat_ratio_in_quality():
    good = {"roe": 20, "roce": 25, "fcf_margin": 8, "debt_equity": 0.3, "sector": "Technology", "cfo_pat_ratio": 1.4}
    red = {"roe": 20, "roce": 25, "fcf_margin": 8, "debt_equity": 0.3, "sector": "Technology", "cfo_pat_ratio": -0.5}
    assert scoring.quality_score(good) > scoring.quality_score(red) + 5


def test_cfo_growth_in_growth():
    good = {"sales_growth": 15, "profit_growth": 15, "eps_accel": 50, "cfo_growth": 0.25}
    weak = {"sales_growth": 15, "profit_growth": 15, "eps_accel": 50, "cfo_growth": -0.2}
    assert scoring.growth_score(good) > scoring.growth_score(weak) + 5


# ── Factor attribution (Phase 1) ────────────────────────────────────────────

def test_factor_contributions_explain_score():
    import pandas as pd

    idx = pd.date_range("2024-01-01", periods=300, freq="B")

    def frame(closes):
        s = pd.Series(closes, index=idx)
        return pd.DataFrame({"close": s, "high": s + 1, "low": s - 1, "volume": pd.Series([1_000_000] * 300, index=idx)})

    prices = {"X.NS": frame([100 + i * 0.2 for i in range(300)])}
    scoring.attach_indicators(prices)
    fundas = [
        {
            "symbol": "X.NS", "roe": 22, "roce": 26, "debt_equity": 0.2, "sales_growth": 18,
            "profit_growth": 20, "pe": 20, "pb": 3, "sector": "Technology", "data_confidence": 100,
        }
    ]
    regime = {"regime": "BULL", "weights": {"quality": 0.2, "growth": 0.3, "momentum": 0.35, "valuation": 0.1, "risk": 0.05}}
    recs = scoring.compute_scores(regime, fundas, prices)
    r = recs[0]
    fc = r["factor_contributions"]
    assert set(fc) >= {"quality", "growth", "momentum", "valuation", "risk", "rs_boost"}
    contrib_sum = sum(v for v in fc.values() if v is not None)
    assert abs(contrib_sum - r["score"]) < 2.5  # contributions add up to the score


# ── Kelly-Lite sizing (Phase 3) ─────────────────────────────────────────────

def test_kelly_lite_prefers_low_vol_quality():
    from lite import portfolio

    recs = [
            {"symbol": "A.NS", "pos_score": 80, "quality": 95, "vol": 0.18, "mb_bucket": "STRONG", "sector": "IT"},
            {"symbol": "B.NS", "pos_score": 80, "quality": 40, "vol": 0.60, "mb_bucket": "WATCHLIST", "sector": "Healthcare"},
        ]
    # Cap disabled so this test isolates the Kelly-Lite preference, not sector caps.
    alloc = portfolio.build_allocation(recs, equity_pct=50, top_n=2, mode="kelly", max_sector_weight=100)
    by = {a["symbol"]: a for a in alloc}
    assert by["A.NS"]["weight_pct"] > by["B.NS"]["weight_pct"] * 2
    assert abs(sum(a["weight_pct"] for a in alloc) - 50.0) < 0.01


# ── Market breadth (Phase 7) ────────────────────────────────────────────────

def test_breadth_computes_percentages():
    import pandas as pd

    from lite import breadth

    idx = pd.date_range("2024-01-01", periods=260, freq="B")
    up = pd.Series([100 + i * 0.5 for i in range(260)], index=idx)  # above every SMA
    flat = pd.Series([200.0] * 260, index=idx)  # exactly at the 200-SMA → not above
    b = breadth.compute_breadth({"UP.NS": pd.DataFrame({"close": up}), "FLAT.NS": pd.DataFrame({"close": flat})})
    assert b["n"] == 2
    assert b["above_20"] == 50.0 and b["above_50"] == 50.0 and b["above_200"] == 50.0
    assert b["new_highs"] == 100.0  # both names sit at/near their 52-week high
    # 0.10·50 + 0.25·50 + 0.40·50 + 0.25·66.7 (high/low ratio component)
    assert b["market_health"] == pytest.approx(54.2, abs=0.5)


# ── Watchlist intelligence (Phase 8) ────────────────────────────────────────

def test_watchlist_events_fire():
    from lite import watchlist

    fundas = {"A.NS": {"sector": "IT"}, "B.NS": {"sector": "IT"}, "C.NS": {"sector": "CEMENT"}}
    records = [
        {"symbol": "A.NS", "rs_rank": 98.0, "score": 85.0, "mb_score": 92.0},
        {"symbol": "B.NS", "rs_rank": 50.0, "score": 60.0, "mb_score": 55.0},
        {"symbol": "C.NS", "rs_rank": 30.0, "score": 45.0, "mb_score": 40.0},
    ]
    events = watchlist.detect_events(records, {"A.NS": {"score": 70.0}}, fundas, "2026-08-16")
    ev = {(e["symbol"], e["event"]) for e in events}
    assert ("A.NS", "RS_LEADER") in ev
    assert ("A.NS", "SCORE_SURGE") in ev
    assert ("A.NS", "MB_ELITE") in ev
    assert ("A.NS", "SECTOR_TOP3") in ev


# ── Alpha decay + factor IC (Phases 2/4/5) ──────────────────────────────────

def test_alpha_forward_returns_and_factor_ic():
    import os
    import tempfile

    import pandas as pd

    from lite import alpha, db

    tmp = tempfile.mkdtemp()
    old_path = db.DB_PATH
    db.DB_PATH = os.path.join(tmp, "t.db")
    try:
        db.init_db([])
        n = 12
        idx = pd.date_range("2025-01-01", periods=400, freq="B")
        p_anchor = 200
        anchor = idx[p_anchor]
        frames = {}
        records = []
        for i in range(n):
            growth = 0.001 * (i + 1)
            closes = [100.0] * p_anchor + [100 * (1 + growth * j) for j in range(1, 400 - p_anchor + 1)]
            frames[f"S{i}.NS"] = pd.DataFrame({"close": pd.Series(closes, index=idx)})
            records.append(
                {
                    "symbol": f"S{i}.NS", "score": 50 + i, "rank": i + 1, "mb_score": 60.0,
                    "mb_bucket": "WATCHLIST", "trend_ok": False, "regime": "BULL",
                    "quality": 40 + i, "growth": 40 + i, "momentum": 40 + i * 5, "valuation": 50.0,
                    "risk": 50.0, "rs_rank": 30 + i * 5, "opp_score": 50 + i, "data_confidence": 100,
                }
            )
        db.snapshot_scores(records, anchor.date().isoformat(), "BULL")

        assert alpha.update_alpha_tracking(frames) > 0
        by_h = {r["horizon_days"]: r for r in db.load_alpha_rows()}
        assert 30 in by_h and by_h[30]["n"] == n
        assert by_h[30]["avg_return_pct"] > 0

        assert alpha.compute_factor_ics(frames) > 0
        summary = alpha.ic_summary()
        assert "momentum" in summary["factors"]
        assert summary["factors"]["momentum"]["avg_ic"] > 0.5  # momentum predicts returns here

        base = {"quality": 0.3, "growth": 0.25, "momentum": 0.2, "valuation": 0.15, "risk": 0.1}
        w = alpha.learned_weights(base, summary, "BULL")
        assert w["momentum"] > base["momentum"] + 0.01
        assert abs(sum(w.values()) - 1.0) < 0.02
    finally:
        db.DB_PATH = old_path


# ── Backtest benchmarking (Phase 9) ─────────────────────────────────────────

def test_backtest_has_sortino_turnover_and_universe_bench():
    import pandas as pd

    from lite import backtest

    idx = pd.date_range("2022-01-01", periods=700, freq="B")
    n = len(idx)
    up = pd.Series([100 * (1.0008 ** i) for i in range(n)], index=idx)
    flat = pd.Series([100.0] * n, index=idx)
    res = backtest.run_backtest({"UP.NS": pd.DataFrame({"close": up}), "FLAT.NS": pd.DataFrame({"close": flat})}, {"years": 2, "top_n": 1})
    s = res["summary"]
    assert "sortino" in s and "turnover_annual" in s
    assert res["universe_curve"]
    assert s.get("universe_cagr_pct") is not None
    assert s.get("universe_alpha_pct") is not None


# ── MB candidates tracking ──────────────────────────────────────────────────

def test_mb_candidates_snapshot_and_history(tmp_path, monkeypatch):
    import lite.db as db_mod

    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "mb.db"))
    db_mod.init_db()

    recs = [
        {"symbol": "A.NS", "mb_score": 82.0, "mb_rank": 1, "mb_bucket": "STRONG"},
        {"symbol": "B.NS", "mb_score": 45.0, "mb_rank": 2, "mb_bucket": "WATCHLIST"},
    ]
    db_mod.snapshot_mb_candidates(recs, "2026-08-15", "BULL")
    assert db_mod.latest_mb_candidates()["A.NS"]["mb_rank"] == 1
    assert db_mod.previous_mb_candidates() == {}  # only one snapshot yet

    recs2 = [
        {"symbol": "A.NS", "mb_score": 88.0, "mb_rank": 1, "mb_bucket": "STRONG"},
        {"symbol": "B.NS", "mb_score": 50.0, "mb_rank": 2, "mb_bucket": "WATCHLIST"},
    ]
    db_mod.snapshot_mb_candidates(recs2, "2026-08-16", "BULL")
    assert db_mod.previous_mb_candidates()["A.NS"]["mb_score"] == 82.0
    hist = db_mod.mb_candidates_history("A.NS")
    assert [h["mb_score"] for h in hist] == [82.0, 88.0]


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


# ── v8: Quality of earnings (accrual ratio) ─────────────────────────────────

def test_quality_penalizes_high_accruals():
    base = {"roe": 20, "roce": 24, "fcf_margin": 10, "debt_equity": 0.3, "sector": "Technology"}
    clean = {**base, "accrual_ratio": 0.04}    # earnings backed by cash
    risky = {**base, "accrual_ratio": 0.6}     # Sloan red flag: profits not in CFO
    assert scoring.quality_score(clean) > scoring.quality_score(risky) + 5


# ── v8: Transaction cost model ──────────────────────────────────────────────

def test_cost_breakdown_components_and_floor():
    from lite import backtest

    low = backtest._cost_breakdown(0.25)
    assert low["stt_pct"] == 0.10
    assert low["brokerage_pct"] == 0.03
    assert low["gst_pct"] == pytest.approx(0.0054, abs=0.001)
    assert low["sebi_pct"] == pytest.approx(0.0001, abs=0.0001)
    assert low["total_per_side_pct"] == 0.25  # floored at the configured cost

    high = backtest._cost_breakdown(0.6)
    assert high["total_per_side_pct"] == 0.6
    assert high["slippage_pct"] > 0.3  # extra slippage beyond the regulatory stack


def test_trade_net_return_includes_costs():
    from lite import backtest

    pos = {"shares": 10, "entry": 100.0, "peak": 100.0, "entry_date": "2024-01-01"}
    t = backtest._trade(pos, "A.NS", pd.Timestamp("2024-02-01"), 110.0, "test", cost=0.0025)
    gross = 110.0 / 100.0 - 1  # +10%
    expected = ((110.0 * (1 - 0.0025)) / (100.0 * (1 + 0.0025)) - 1) * 100
    assert t["return_pct"] == pytest.approx(expected, abs=0.01)
    assert t["return_pct"] < gross * 100 - 0.4  # costs eat ~0.5% round trip
    assert t["cost_pct"] == 0.25


def test_backtest_reports_cost_drag_and_survivorship_guard():
    from lite import backtest

    idx = pd.date_range("2022-01-01", periods=700, freq="B")
    n = len(idx)
    early = pd.Series([100 * (1.0008 ** i) for i in range(n)], index=idx)
    # Late starter: plenty of bars, but no history before the window begins →
    # it must be excluded rather than silently backfilled into 2022.
    late_idx = idx[350:]
    late = pd.Series([100 * (1.001 ** i) for i in range(len(late_idx))], index=late_idx)
    res = backtest.run_backtest(
        {"EARLY.NS": pd.DataFrame({"close": early}), "LATE.NS": pd.DataFrame({"close": late})},
        {"years": 2, "top_n": 1},
    )
    s = res["summary"]
    assert "cost_drag_pct" in s and s["cost_drag_pct"] > 0
    assert s.get("cost_model", {}).get("total_per_side_pct") == 0.25
    assert any("survivorship" in w for w in res["warnings"])
    # Only the early survivor should trade
    assert all(t["symbol"] == "EARLY.NS" for t in res["trades"])


# ── v8: Drawdown-aware Kelly-Lite + sector caps + factor exposure ───────────

def test_kelly_lite_penalizes_deep_drawdowns():
    from lite import portfolio

    calm = {"pos_score": 80.0, "quality": 60.0, "vol": 0.25, "max_dd": -0.15}
    scary = {**calm, "max_dd": -0.55}  # low vol but catastrophic drawdown
    assert portfolio._kelly_lite(calm) > portfolio._kelly_lite(scary) * 2


def test_sector_caps_trim_concentration():
    from lite import portfolio

    # Infeasible case: 6 Financials + 2 Tech, 60% total, 25% cap each →
    # Financials 45% is trimmed to 25%, Tech fills to 25%, the un-placeable
    # 10% (only two sectors can hold 50% max) leaves the book as cash.
    records = []
    for i in range(6):
        records.append({"symbol": f"FIN{i}.NS", "sector": "Financials", "pos_score": 80.0, "name": f"FIN{i}"})
    for i in range(2):
        records.append({"symbol": f"TECH{i}.NS", "sector": "Technology", "pos_score": 80.0, "name": f"TECH{i}"})
    alloc = portfolio.build_allocation(records, equity_pct=60.0, top_n=8, mode="conviction", max_sector_weight=25.0)
    sw = portfolio.sector_weights(alloc)
    # Rounding tolerance: per-name 2-decimal weights can overshoot by ~0.02.
    assert sw["Financials"] <= 25.1
    assert sw["Technology"] <= 25.1
    assert abs(sum(sw.values()) - 50.0) < 0.5  # excess 10% → cash, nothing lost

    # Feasible case: 3 sectors, 60% total, 25% cap → no cash loss.
    records2 = []
    for i in range(4):
        records2.append({"symbol": f"FIN{i}.NS", "sector": "Financials", "pos_score": 80.0, "name": f"FIN{i}"})
    for i in range(2):
        records2.append({"symbol": f"TECH{i}.NS", "sector": "Technology", "pos_score": 80.0, "name": f"TECH{i}"})
    for i in range(2):
        records2.append({"symbol": f"HEALTH{i}.NS", "sector": "Healthcare", "pos_score": 80.0, "name": f"HEALTH{i}"})
    alloc2 = portfolio.build_allocation(records2, equity_pct=60.0, top_n=8, mode="conviction", max_sector_weight=25.0)
    sw2 = portfolio.sector_weights(alloc2)
    assert sw2["Financials"] <= 25.01
    assert abs(sum(sw2.values()) - 60.0) < 0.5  # fully redistributed, total intact


def test_factor_exposure_reports_crowding():
    from lite import portfolio

    alloc = [
        {"symbol": "A.NS", "sector": "Financials", "weight_pct": 20.0},
        {"symbol": "B.NS", "sector": "Technology", "weight_pct": 10.0},
    ]
    recs = {
        "A.NS": {"quality": 90, "growth": 40, "momentum": 30, "valuation": 20, "risk": 10, "mb_score": 70, "rs_rank": 60},
        "B.NS": {"quality": 30, "growth": 30, "momentum": 90, "valuation": 20, "risk": 10, "mb_score": 70, "rs_rank": 90},
    }
    exp = portfolio.factor_exposure(alloc, recs)
    assert exp["quality"] == pytest.approx(70.0, abs=0.1)  # (90*20 + 30*10)/30
    assert exp["momentum"] == pytest.approx(50.0, abs=0.1)
    assert exp["top_factor"] == "quality"


# ── v8: Market breadth with new highs / new lows ────────────────────────────

def test_breadth_new_highs_lows_and_health():
    from lite import breadth

    idx = pd.date_range("2024-01-01", periods=300, freq="B")
    rising = pd.DataFrame({"close": [100 * (1.001 ** i) for i in range(300)]}, index=idx)
    falling = pd.DataFrame({"close": [100 * (0.999 ** i) for i in range(300)]}, index=idx)

    only_up = breadth.compute_breadth({"A.NS": rising})
    assert only_up["above_200"] == 100.0
    assert only_up["new_highs"] == 100.0
    assert only_up["new_lows"] == 0.0
    assert only_up["market_health"] == pytest.approx(100.0, abs=0.1)

    mixed = breadth.compute_breadth({"A.NS": rising, "B.NS": falling})
    assert mixed["above_200"] == 50.0
    assert mixed["new_lows"] == 50.0
    assert mixed["market_health"] == pytest.approx(50.0, abs=0.1)


# ── v8: Point-in-time snapshots (survivorship + PIT fundamentals) ───────────

def test_pit_universe_and_fundamentals_snapshots(tmp_path, monkeypatch):
    import lite.db as db_mod

    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "pit.db"))
    db_mod.init_db()
    db_mod.seed_universe([{"symbol": "A.NS", "name": "A", "sector": "Tech"}])
    db_mod.upsert_fundamentals(
        {"symbol": "A.NS", "roe": 25.0, "roce": 30.0, "debt_equity": 0.2,
         "sales_growth": 18.0, "profit_growth": 22.0, "pe": 28.0, "pb": 5.0,
         "fcf_margin": 10.0, "accrual_ratio": 0.05, "data_confidence": 92.0}
    )

    assert db_mod.snapshot_universe("2026-08-15") == 1
    assert db_mod.universe_history_snapshots()[0]["members"] == 1

    assert db_mod.snapshot_fundamentals_history("2026-08-15") == 1
    hist = db_mod.fundamentals_history_for("A.NS")
    assert len(hist) == 1
    assert hist[0]["roe"] == 25.0
    assert hist[0]["accrual_ratio"] == pytest.approx(0.05, abs=1e-9)
    assert hist[0]["data_confidence"] == 92.0


# ── v9: Two-tier universe (Core + Discovery) ────────────────────────────────

def test_discovery_universe_mines_broad_list():
    from lite.universe import default_universe, discovery_universe

    core = default_universe()
    disc = discovery_universe()
    assert len(core) > 100
    assert len(disc) > 300  # NIFTY-500-style breadth beyond the curated core
    core_syms = {s["symbol"] for s in core}
    disc_syms = {s["symbol"] for s in disc}
    assert not (core_syms & disc_syms)  # no overlap — deduped against core
    assert all(s.endswith(".NS") for s in disc_syms)


def test_universe_tiers_roundtrip(tmp_path, monkeypatch):
    import lite.db as db_mod

    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "tiers.db"))
    db_mod.init_db()
    db_mod.seed_universe([{"symbol": "A.NS", "name": "A", "sector": "Tech"}], tier="core")
    db_mod.seed_universe(
        [{"symbol": "Z1.NS", "name": "Z1", "sector": "Discovery"},
         {"symbol": "Z2.NS", "name": "Z2", "sector": "Discovery"}],
        tier="discovery",
    )
    assert db_mod.universe_symbols(tier="core") == ["A.NS"]
    assert sorted(db_mod.universe_symbols(tier="discovery")) == ["Z1.NS", "Z2.NS"]
    assert sorted(db_mod.universe_symbols()) == ["A.NS", "Z1.NS", "Z2.NS"]
    tiers = db_mod.universe_tiers()
    assert tiers["core"] == 1 and tiers["discovery"] == 2
    db_mod.add_stock("Z3.NS", tier="discovery")
    assert db_mod.universe_tiers()["discovery"] == 3


def test_earliest_universe_snapshot(tmp_path, monkeypatch):
    import lite.db as db_mod

    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "pit2.db"))
    db_mod.init_db()
    assert db_mod.earliest_universe_snapshot() is None
    db_mod.seed_universe([{"symbol": "A.NS", "name": "A", "sector": "Tech"}])
    db_mod.snapshot_universe("2026-08-15")
    db_mod.snapshot_universe("2026-08-16")
    assert db_mod.earliest_universe_snapshot() == "2026-08-15"


# ── v9: Reinvestment score ──────────────────────────────────────────────────

def test_reinvestment_score_ranks_compounders():
    from lite import multibagger

    good = {"sales_cagr_5y": 0.22, "profit_cagr_5y": 0.25, "roce_stability": 88.0}
    bad = {"sales_cagr_5y": 0.03, "profit_cagr_5y": 0.02, "roce_stability": 30.0}
    assert multibagger.reinvestment_score(good) > multibagger.reinvestment_score(bad) + 30
    assert multibagger.reinvestment_score({"roce_stability": 80.0}) is not None  # None-safe
    assert multibagger.reinvestment_score({}) is None


def test_detect_attaches_reinvestment_score():
    from lite import multibagger

    out = multibagger.detect(
        [{"symbol": "A.NS", "score": 70.0, "rs_rank": 80.0}],
        {"A.NS": {"sales_cagr_5y": 0.2, "profit_cagr_5y": 0.24, "roce_stability": 85.0, "sector": "Tech"}},
        {},
    )
    assert out[0]["reinvestment_score"] is not None
    assert "reinvestment_score" in out[0]


# ── v9: PIT-universe stats in walk-forward ──────────────────────────────────

def test_walk_forward_reports_pit_and_universe(tmp_path, monkeypatch):
    import lite.db as db_mod
    from lite import backtest

    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "wfp.db"))
    db_mod.init_db()
    db_mod.seed_universe([{"symbol": "A.NS", "name": "A", "sector": "Tech"}])
    db_mod.snapshot_universe("2026-01-15")  # PIT snapshots only exist from 2026

    idx = pd.date_range("2022-01-01", periods=1100, freq="B")
    n = len(idx)
    up = pd.Series([100 * (1.0008 ** i) for i in range(n)], index=idx)
    flat = pd.Series([100.0] * n, index=idx)
    res = backtest.walk_forward(
        {"UP.NS": pd.DataFrame({"close": up}), "FLAT.NS": pd.DataFrame({"close": flat})},
        {"folds": 3, "fold_months": 12, "top_n": 1},
    )
    assert res["folds"], res.get("summary")
    assert res["summary"]["pit_snapshots_from"] == "2026-01-15"
    assert res["summary"]["pre_pit_folds"] == len(res["folds"])  # all folds precede 2026
    for f in res["folds"]:
        assert f["universe_size"] == 2
        assert f["pre_pit"] is True


# ── v10: Portfolio Risk readout + vol/max_dd persistence ─────────────────────

def test_portfolio_risk_readout():
    from lite import portfolio
    recs = [
        {"symbol": "A.NS", "pos_score": 90, "quality": 85, "vol": 0.35, "max_dd": -0.45, "data_confidence": 92},
        {"symbol": "B.NS", "pos_score": 80, "quality": 75, "vol": 0.25, "max_dd": -0.30, "data_confidence": 88},
        {"symbol": "C.NS", "pos_score": 70, "quality": 65, "vol": 0.20, "max_dd": -0.20, "data_confidence": 80},
    ]
    alloc = portfolio.build_allocation(recs, equity_pct=60, top_n=3)
    r = portfolio.portfolio_risk(alloc, {x["symbol"]: x for x in recs}, 60)
    assert r["n"] == 3
    assert r["equity_pct"] == 60.0 and r["cash_pct"] == 40.0
    assert r["avg_vol"] is not None and r["portfolio_vol"] < r["avg_vol"]  # diversification discount
    assert r["avg_max_dd"] is not None and r["avg_max_dd"] < 0
    assert r["hhi"] > 0 and r["top1_share"] >= r["top3_share"] / 3
    assert r["risk_grade"] in ("CONSERVATIVE", "BALANCED", "AGGRESSIVE")
    # single-name book → HHI ~1 and no diversification discount
    solo = portfolio.build_allocation([recs[0]], equity_pct=50, top_n=1)
    rs = portfolio.portfolio_risk(solo, {"A.NS": recs[0]}, 50)
    assert rs["hhi"] == 1.0
    assert abs(rs["portfolio_vol"] - rs["avg_vol"]) < 0.5


def test_vol_maxdd_persisted(tmp_path, monkeypatch):
    import lite.db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "vm.db"))
    db_mod.init_db()
    db_mod.upsert_scores([
        {
            "symbol": "A.NS", "score": 80.0, "quality": 80.0, "growth": 70.0,
            "momentum": 75.0, "valuation": 60.0, "risk": 72.0, "mb_score": 78.0,
            "pos_score": 81.0, "opp_score": 77.0, "rs_rank": 85.0, "rank": 1,
            "regime": "bull", "vol": 0.34, "max_dd": -0.42, "data_confidence": 90.0,
        }
    ])
    rows = db_mod.load_scores()
    assert len(rows) == 1
    assert rows[0]["vol"] == pytest.approx(0.34)
    assert rows[0]["max_dd"] == pytest.approx(-0.42)


# ── v11: Revision proxy 2.0 · data quality · discovery engine ───────────────

def test_revision_score_includes_cfo_growth():
    base = {"eps_accel": 90, "rev_accel": 85, "margin_expansion": 6.0}
    strong = dict(base, cfo_growth=0.30)
    weak = dict(base, cfo_growth=-0.10)
    assert scoring.revision_score(strong) > scoring.revision_score(weak) + 5


def test_revision_factor_in_composite_and_attribution():
    import pandas as pd

    idx = pd.date_range("2024-01-01", periods=400, freq="B")
    s = pd.Series([100 + i * 0.15 for i in range(400)], index=idx)
    prices = {"REV.NS": pd.DataFrame({"close": s, "high": s + 1, "low": s - 1, "volume": pd.Series([1_000_000] * 400, index=idx)})}
    scoring.attach_indicators(prices)
    fundas = [{
        "symbol": "REV.NS", "roe": 20, "roce": 25, "debt_equity": 0.3,
        "sales_growth": 15, "profit_growth": 15, "pe": 20, "pb": 3,
        "fcf_margin": 8, "sector": "Technology",
        "eps_accel": 90, "rev_accel": 85, "margin_expansion": 6.0, "cfo_growth": 0.30,
    }]
    regime = {"regime": "BULL", "weights": {"quality": 0.25, "growth": 0.30, "momentum": 0.35, "valuation": 0.05, "risk": 0.05}}
    rec = scoring.compute_scores(regime, fundas, prices)[0]
    fc = rec["factor_contributions"]
    assert fc.get("revision") is not None
    # attribution bars sum back to the displayed score (rounding tolerance)
    total = sum(v for v in fc.values() if v is not None)
    assert abs(total - rec["score"]) < 0.6


def test_field_coverage_reports_weak_fields(tmp_path, monkeypatch):
    import lite.db as db_mod
    from lite import data

    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "dq.db"))
    db_mod.init_db()
    full = {"symbol": "A.NS", "roe": 20, "roce": 25, "debt_equity": 0.3, "sales_growth": 15,
            "profit_growth": 15, "pe": 20, "pb": 3, "fcf_margin": 8, "eps_accel": 60,
            "margin_expansion": 2.0, "cfo_growth": 0.2, "cfo_pat_ratio": 1.1, "accrual_ratio": -0.02,
            "data_confidence": 92}
    partial = {"symbol": "B.NS", "roe": 12, "roce": None, "debt_equity": 0.8, "sales_growth": 8,
               "profit_growth": 8, "pe": 30, "pb": 4, "fcf_margin": None, "eps_accel": None,
               "margin_expansion": None, "cfo_growth": None, "cfo_pat_ratio": None, "accrual_ratio": None,
               "data_confidence": 55}
    db_mod.upsert_fundamentals(full)
    db_mod.upsert_fundamentals(partial)
    cov = data.field_coverage()
    assert cov["n"] == 2
    by = {f["key"]: f["coverage"] for f in cov["fields"]}
    assert by["roce"] == 50.0
    assert by["cfo_growth"] == 50.0
    assert by["roe"] == 100.0
    assert cov["avg_confidence"] == pytest.approx(73.5)


def test_discovery_score_ranks_accelerating_names():
    from lite import discovery

    accel = {"rs_rank": 80, "rs_1m": 92, "rs_3m": 60, "revision_score": 70, "momentum": 70, "data_confidence": 90}
    decel = {"rs_rank": 80, "rs_1m": 60, "rs_3m": 92, "revision_score": 70, "momentum": 70, "data_confidence": 90}
    assert discovery.discovery_score(accel) > discovery.discovery_score(decel)
    assert discovery.rs_acceleration(accel) == pytest.approx(32.0)
    assert discovery.rs_acceleration(decel) == pytest.approx(-32.0)
    assert discovery.discovery_score({"rs_rank": None}) is None
