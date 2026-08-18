"""
Sovereign Lite v17 — 100-Bagger Detector (Phase 4).

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


# Multibagger Score v3 — dedicated formula (weights sum to 1.0):
#   25% Compounder (5y consistency) · 18% earnings acceleration ·
#   14% sales growth · 12% ROCE · 12% relative strength · 8% margin expansion
#   · 8% debt reduction · 3% valuation
MB_WEIGHTS = {
    "compounder": 0.25,
    "accel": 0.18,
    "sales": 0.14,
    "roce": 0.12,
    "rs": 0.12,
    "margin": 0.08,
    "debt": 0.08,
    "valuation": 0.03,
}


def reinvestment_score(f: dict) -> Optional[float]:
    """Reinvestment quality over 5 years — the plough-back engine behind
    compounders: 40% sales CAGR · 40% profit CAGR · 20% ROCE consistency.
    Many 50-baggers score highly here long before the price chart says so.
    """
    sales_cagr = scoring.sigmoid(f.get("sales_cagr_5y") * 100, 15, 10) if f.get("sales_cagr_5y") is not None else None
    profit_cagr = scoring.sigmoid(f.get("profit_cagr_5y") * 100, 15, 12) if f.get("profit_cagr_5y") is not None else None
    parts = [
        (sales_cagr, 0.40),
        (profit_cagr, 0.40),
        (f.get("roce_stability"), 0.20),
    ]
    return scoring._weighted(parts)


def compounder_score(f: dict) -> Optional[float]:
    """5-year compounding quality — the longevity a true multibagger needs:

    25% ROCE consistency · 20% sales CAGR · 20% profit CAGR · 20% FCF CAGR ·
    15% debt reduction trend.
    """
    roce_stab = f.get("roce_stability")
    sales_cagr = scoring.sigmoid(f.get("sales_cagr_5y") * 100, 15, 10) if f.get("sales_cagr_5y") is not None else None
    profit_cagr = scoring.sigmoid(f.get("profit_cagr_5y") * 100, 15, 12) if f.get("profit_cagr_5y") is not None else None
    fcf_cagr = scoring.sigmoid(f.get("fcf_cagr_5y") * 100, 15, 15) if f.get("fcf_cagr_5y") is not None else None
    parts = [
        (roce_stab, 0.25),
        (sales_cagr, 0.20),
        (profit_cagr, 0.20),
        (fcf_cagr, 0.20),
        (f.get("debt_trend"), 0.15),
    ]
    return scoring._weighted(parts)

MB_BUCKETS = [
    (90.0, "MULTIBAGGER"),
    (75.0, "STRONG"),
    (60.0, "EMERGING"),
    (0.0, "WATCHLIST"),
]


def bucket_for(score: float) -> str:
    for threshold, name in MB_BUCKETS:
        if score >= threshold:
            return name
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

        # MB v3 — dedicated multibagger formula (see MB_WEIGHTS).
        accel = f.get("eps_accel")
        if accel is None:
            accel = scoring.sigmoid(f.get("eps_growth"), 20, 15)
        else:
            accel = scoring._clamp(accel, 0, 100)
        sales = scoring.sigmoid(f.get("sales_growth"), 15, 10)
        roce = scoring.sigmoid(f.get("roce"), 18, 10)
        rs = r.get("rs_rank")
        if rs is not None:
            rs = scoring._clamp(rs, 0, 100)
        margin = scoring.margin_expansion_score(f)
        de = f.get("debt_equity")
        if scoring.is_financial(f.get("sector")):
            debt = 60.0
        elif de is not None:
            debt = 100 - scoring.sigmoid(de, 0.6, 0.5)
        else:
            debt = None
        valuation = scoring.valuation_score(f)
        compounder = compounder_score(f)
        reinvest = reinvestment_score(f)
        parts = [
            (compounder, MB_WEIGHTS["compounder"]),
            (accel, MB_WEIGHTS["accel"]),
            (sales, MB_WEIGHTS["sales"]),
            (roce, MB_WEIGHTS["roce"]),
            (rs, MB_WEIGHTS["rs"]),
            (margin, MB_WEIGHTS["margin"]),
            (debt, MB_WEIGHTS["debt"]),
            (valuation, MB_WEIGHTS["valuation"]),
        ]
        mb = scoring._weighted(parts)
        mb = round(mb, 1) if mb is not None else 0.0

        row = dict(r)
        row["mb_score"] = mb
        row["compounder_score"] = round(compounder, 1) if compounder is not None else None
        row["reinvestment_score"] = round(reinvest, 1) if reinvest is not None else None
        row["mb_bucket"] = bucket_for(mb)
        row["mb_rules_passed"] = passed
        row["mb_rules_total"] = len(RULES)
        row["mb_checklist"] = checklist
        out.append(row)
    return out
