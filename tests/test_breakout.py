"""
Tests for lite/breakout.py — the Sector Breakout Monitor (Phase 18).

Run:  python3 -m pytest tests/test_breakout.py -q
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lite import breakout  # noqa: E402


# ── Frame helpers ───────────────────────────────────────────────────────────

def _frame(closes):
    idx = pd.date_range("2023-01-02", periods=len(closes), freq="B")
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1_000_000] * len(closes),
        },
        index=idx,
    )


def _near_high_closes():
    """Every bar ends within 1% of the 52-week high (ramp to 200, hold there)."""
    return [100.0] * 200 + [100.0 + i * 4.0 for i in range(1, 26)] + [200.0] * 25


def _crashed_closes():
    """Same ramp, then a 40% collapse away from the high (120 vs 200)."""
    return [100.0] * 200 + [100.0 + i * 4.0 for i in range(1, 26)] + [120.0] * 25


# ── Module tests ────────────────────────────────────────────────────────────

def test_sector_near_highs_is_flagged_breakout():
    frames = {
        "H1.NS": _frame(_near_high_closes()),
        "H2.NS": _frame(_near_high_closes()),
        "H3.NS": _frame(_near_high_closes()),
        "C1.NS": _frame(_crashed_closes()),
        "C2.NS": _frame(_crashed_closes()),
        "C3.NS": _frame(_crashed_closes()),
    }
    fundas = {
        "H1.NS": {"sector": "Hot"},
        "H2.NS": {"sector": "Hot"},
        "H3.NS": {"sector": "Hot"},
        "C1.NS": {"sector": "Cold"},
        "C2.NS": {"sector": "Cold"},
        "C3.NS": {"sector": "Cold"},
    }
    out = breakout.rank_breakouts(frames, fundas)
    assert out["n_sectors"] == 2
    hot = next(s for s in out["sectors"] if s["sector"] == "Hot")
    cold = next(s for s in out["sectors"] if s["sector"] == "Cold")
    assert hot["breakout"] is True
    assert hot["near_high_pct"] == 100.0
    assert hot["at_high_pct"] == 100.0
    assert hot["breakout_score"] >= 90
    assert cold["breakout"] is False
    assert cold["near_high_pct"] == 0.0
    assert hot["rank"] < cold["rank"]
    assert out["in_breakout"] == ["Hot"]


def test_leaders_exclude_names_far_below_high():
    frames = {
        "A1.NS": _frame(_near_high_closes()),
        "A2.NS": _frame(_crashed_closes()),  # crashed member of an otherwise hot sector
        "A3.NS": _frame(_near_high_closes()),
    }
    fundas = {k: {"sector": "Mixed"} for k in frames}
    out = breakout.rank_breakouts(frames, fundas)
    sec = out["sectors"][0]
    assert sec["near_high_pct"] == pytest.approx(66.7, abs=0.1)
    assert sec["breakout"] is True  # 66.7% near-high with 3 members clears the bar
    leader_syms = {l["symbol"] for l in sec["leaders"]}
    assert "A2.NS" not in leader_syms
    assert all(l["dist_52w_high_pct"] <= 5.0 for l in sec["leaders"])


def test_empty_input_returns_empty():
    out = breakout.rank_breakouts({}, {})
    assert out == {"n_sectors": 0, "in_breakout": [], "sectors": []}


def test_unknown_sector_bucket():
    frames = {"X1.NS": _frame(_near_high_closes())}
    out = breakout.rank_breakouts(frames, {})  # no fundamentals → "Unknown" bucket
    assert out["n_sectors"] == 1
    assert out["sectors"][0]["sector"] == "Unknown"
    assert out["sectors"][0]["breakout"] is False  # 1 member < 3 minimum


def test_short_history_is_skipped():
    frames = {"Y1.NS": _frame([100.0] * 30)}  # < 63 bars — not enough to say anything
    out = breakout.rank_breakouts(frames, {"Y1.NS": {"sector": "Tiny"}})
    assert out["n_sectors"] == 0


# ── API endpoint test ───────────────────────────────────────────────────────

def test_research_breakouts_endpoint(tmp_path, monkeypatch):
    from lite import api
    import lite.db as db_mod

    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "breakout.db"))
    db_mod.init_db()
    for sym, sector in (("H1.NS", "Hot"), ("H2.NS", "Hot"), ("H3.NS", "Hot"),
                        ("C1.NS", "Cold"), ("C2.NS", "Cold"), ("C3.NS", "Cold")):
        db_mod.add_stock(sym, tier="core")
        closes = _near_high_closes() if sector == "Hot" else _crashed_closes()
        idx = pd.date_range("2023-01-02", periods=len(closes), freq="B")
        rows = [
            {
                "date": d.strftime("%Y-%m-%d"),
                "open": c, "high": c, "low": c, "close": c, "volume": 1_000_000,
            }
            for d, c in zip(idx, closes)
        ]
        db_mod.upsert_prices(sym, rows)
        db_mod.upsert_fundamentals({"symbol": sym, "sector": sector, "name": sym})

    app = api.create_app()
    ep = next(r.endpoint for r in app.routes if getattr(r, "path", "") == "/api/research/breakouts")
    out = ep()
    assert out["n_sectors"] == 2
    hot = next(s for s in out["sectors"] if s["sector"] == "Hot")
    assert hot["breakout"] is True
    assert len(hot["leaders"]) == 3
    assert out["in_breakout"] == ["Hot"]
