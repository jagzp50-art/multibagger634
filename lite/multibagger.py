"""
Sovereign Lite v7 — 100-Bagger Detector (Phase 4).

Seven Lynch/Minervini/O'Neil style rules with smooth scoring:

  R1  Sales CAGR > 15%          (YoY revenue growth as proxy)
  R2  ROCE > 18%
  R3  Debt/Equity < 0.5
  R4  Distance from 52W high < 15%
  R5  Volume expansion >= 1.5x average
  R6  EPS growth > 20%
  R7  Market cap 500 Cr - 25,000 Cr

MB Score = Growth + Quality + Momentum + Ownership (0-100), bucketed:
  0-60 Watchlist | 60-80 Strong | 80-90 Elite | 90+ Multibagger Candidate
"""
from __future__ import annotations

from typing import Optional

from . import scoring

RULES = [
    ("R1", "Sales growth > 15%", lambda f, px: f.get("sales_growth") is not None and (f.get("sales_growth") or 0) > 15),
    ("R2", "ROCE > 18%", lambda f, px: f.get("roce") is not None and (f.get("roce") or 0) > 18),
    ("R3", "Debt/Equity < 0.5", lambda f, px: f.get("debt_equity") is not None and (f.get("debt_equity") or 0) < 0.5),
    ("R4", "Near 52W high (<15%)", lambda f, px: px.get("dist_52w_high") is not None and (px.get("dist_52w_high") or 1) < 0.15),
    ("R5", "Volume >= 1.5x avg", lambda f, px: px.get("volume_ratio") is not None and (px.get("volume_ratio") or 0) >= 1.5),
    ("R6", "EPS growth > 20%", lambda f, px: f.get("profit_growth") is not None and (f.get("profit_growth") or 0) > 20),
    ("R7", "Market cap 500–25,000 Cr", lambda f, px: _mc_ok(f.get("market_cap"))),
]


def _mc_ok(mc: Optional[float]) -> bool:
    return mc is not None and 500 <= mc <= 25_000


def bucket_for(score: float) -> str:
    if score >= 90:
        return "MULTIBAGGER"
    if score >= 80:
        return "ELITE"
    if score >= 60:
        return "STRONG"
    return "WATCHLIST"


def detect(records: list[dict], fundamentals: dict[str, dict], prices: dict[str, object]) -> list[dict]:
    """Attach rule checklist + MB score to scored records."""
    out = []
    for r in records:
        symbol = r["symbol"]
        f = fundamentals.get(symbol, {})
        px = prices.get(symbol, {})
        if isinstance(px, dict):
            px = px
        else:  # DataFrame → last row dict
            px = px.iloc[-1].to_dict() if len(px) else {}

        checklist = []
        passed = 0
        for code, label, fn in RULES:
            try:
                ok = bool(fn(f, px))
            except Exception:
                ok = False
            if ok:
                passed += 1
            checklist.append({"code": code, "label": label, "pass": ok})

        growth = r.get("growth")
        quality = r.get("quality")
        momentum = r.get("momentum")
        ownership = r.get("mb_ownership")
        parts = [(growth, 0.30), (quality, 0.30), (momentum, 0.25), (ownership, 0.15)]
        mb = scoring._weighted(parts)
        mb = round(mb, 1) if mb is not None else 0.0

        row = dict(r)
        row["mb_score"] = mb
        row["mb_bucket"] = bucket_for(mb)
        row["mb_rules_passed"] = passed
        row["mb_rules_total"] = len(RULES)
        row["mb_checklist"] = checklist
        out.append(row)
    return out
