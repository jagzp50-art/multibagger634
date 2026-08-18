"""
Sovereign Lite v16 — portfolio construction layer.

Position Score = 0.40 Quality + 0.30 MB Score + 0.20 RS Rank + 0.10 Risk
(higher = better). Allocation weights are conviction-based: the #1 position
gets the largest slice of the regime's equity allocation — top conviction
gets the most capital, no equal-weighting.

v8: Kelly-Lite 2.0 adds a drawdown penalty (low vol alone isn't enough — a
stock with catastrophic drawdowns gets penalized), allocation enforces a hard
sector cap (no 45% Financials book), and factor_exposure() surfaces crowding.

v14: rebalance_plan() — the tax-aware rebalancing engine. Compares the
previous saved allocation to the proposed target, classifies every position
as HOLD / ADD / TRIM / SELL / BUY, and estimates the tax hit on realized
gains (STCG vs LTCG by holding period) plus trading costs, so a monthly
rebalance can be judged by its after-tax value instead of executed blindly.

All inputs degrade to None-safe math via `scoring._weighted`.
"""
from __future__ import annotations

from typing import Optional

from datetime import date
from typing import Optional

from . import scoring

POS_WEIGHTS = {"quality": 0.40, "mb": 0.30, "rs": 0.20, "risk": 0.10}


def position_score(quality: Optional[float], mb: Optional[float],
                   rs: Optional[float], risk: Optional[float]) -> Optional[float]:
    parts = [
        (quality, POS_WEIGHTS["quality"]),
        (mb, POS_WEIGHTS["mb"]),
        (rs, POS_WEIGHTS["rs"]),
        (risk, POS_WEIGHTS["risk"]),
    ]
    return scoring._weighted(parts)


def attach_position_scores(records: list[dict]) -> list[dict]:
    """Attach pos_score to scored records (mb_score must already be set)."""
    for r in records:
        r["pos_score"] = position_score(
            r.get("quality"), r.get("mb_score"), r.get("rs_rank"), r.get("risk")
        )
        if r["pos_score"] is not None:
            r["pos_score"] = round(r["pos_score"], 1)
    return records


def _kelly_lite(r: dict) -> float:
    """Kelly-Lite 2.1: score · quality · (1/volatility) · drawdown penalty ·
    liquidity factor.

    Low volatility alone isn't enough — a stock with low vol but catastrophic
    drawdowns gets penalized via its max drawdown (a −50% DD halves it). And
    a 95-scoring microcap with no daily traded value can't be oversized:
    the liquidity factor (avg traded value, log-scaled 0.2–1.0) is now part
    of the sizing. Names without a stored liquidity default to no penalty.
    """
    pos = r.get("pos_score") or 0.0
    q = (r.get("quality") or 50.0) / 100.0
    vol = r.get("vol")
    if vol is None or vol <= 0:
        vol = 0.30
    vol_factor = 0.15 / (0.15 + vol)
    mdd = r.get("max_dd")
    dd_factor = 1.0
    if mdd is not None:
        depth = min(1.0, max(0.0, -float(mdd)))  # 0..1 drawdown depth
        dd_factor = max(0.05, min(1.0, 1.0 - depth / 0.5))
    liq = r.get("liquidity")
    liq_factor = float(liq) if liq is not None and float(liq) > 0 else 1.0
    return max(0.0, pos * q * vol_factor * dd_factor * liq_factor)


def _enforce_sector_caps(alloc: list[dict], cap_pct: float) -> list[dict]:
    """Trim any sector above `cap_pct`, redistributing the excess to
    under-cap sectors (up to their headroom). A 45% Financials book never
    happens; if the cap is infeasible (e.g. two sectors, 60% total, 25% cap)
    the un-placeable excess is left in cash rather than oscillating.
    """
    if cap_pct <= 0 or cap_pct >= 100:
        return alloc
    for _ in range(50):
        by_sector: dict[str, float] = {}
        for a in alloc:
            by_sector[a["sector"]] = by_sector.get(a["sector"], 0.0) + a["weight_pct"]
        over = {s: w for s, w in by_sector.items() if w > cap_pct}
        if not over:
            break
        excess = sum(w - cap_pct for w in over.values())
        under = {s: w for s, w in by_sector.items() if w < cap_pct}
        headroom = sum(cap_pct - w for w in under.values())
        give = min(excess, headroom)
        # Trim over-cap sectors to the cap (removes `excess` in total).
        for a in alloc:
            s = a["sector"]
            if s in over:
                trim = (by_sector[s] - cap_pct) * (a["weight_pct"] / by_sector[s])
                a["weight_pct"] = round(a["weight_pct"] - trim, 2)
        # Redistribute `give` in total across under-cap names, proportional to
        # their current weights (so a sector with several names splits it).
        under_names = [a for a in alloc if a["sector"] in under]
        under_weight_total = sum(a["weight_pct"] for a in under_names)
        if under_weight_total > 0:
            for a in under_names:
                a["weight_pct"] = round(
                    a["weight_pct"] + give * (a["weight_pct"] / under_weight_total), 2
                )
        # New total = old − excess + give; anything the caps couldn't absorb
        # (give < excess) is simply left in cash — no oscillation, no double-count.
        if give < excess:
            break
    return alloc


def _enforce_budgets(alloc: list[dict], budgets: dict[str, float], global_cap: float = 25.0) -> list[dict]:
    """Per-sector budget enforcement. Sectors without an explicit budget fall
    back to `global_cap`. Same redistribution-to-cash behavior as the global
    cap: excess that no sector can absorb is left unplaced (cash), never
    oscillated."""
    if not budgets:
        return alloc
    for _ in range(50):
        by_sector: dict[str, float] = {}
        for a in alloc:
            by_sector[a["sector"]] = by_sector.get(a["sector"], 0.0) + a["weight_pct"]
        over = {s: w for s, w in by_sector.items() if w > budgets.get(s, global_cap)}
        if not over:
            break
        excess = sum(w - budgets.get(s, global_cap) for s, w in over.items())
        under = {s: w for s, w in by_sector.items() if w < budgets.get(s, global_cap)}
        headroom = sum(budgets.get(s, global_cap) - w for s, w in under.items())
        give = min(excess, headroom)
        for a in alloc:
            s = a["sector"]
            if s in over:
                cap = budgets.get(s, global_cap)
                trim = (by_sector[s] - cap) * (a["weight_pct"] / by_sector[s])
                a["weight_pct"] = round(a["weight_pct"] - trim, 2)
        under_names = [a for a in alloc if a["sector"] in under]
        under_total = sum(a["weight_pct"] for a in under_names)
        if under_total > 0:
            for a in under_names:
                a["weight_pct"] = round(a["weight_pct"] + give * (a["weight_pct"] / under_total), 2)
        if give < excess:
            break
    return alloc


def cash_plan(
    alloc: list[dict],
    records_by_symbol: dict[str, dict],
    regime: dict,
    breadth: Optional[dict] = None,
) -> dict:
    """Advisory cash management: how much cash the regime/breadth call for,
    which names should be staged instead of bought at once, and the
    deployment schedule for the buffer. Complements (never overrides) the
    Kelly-Lite target — the Rebalancer shows it so you can judge a plan by
    its cash efficiency too."""
    equity_pct = float((regime.get("allocation") or {}).get("equity", 60))
    base_cash = round(100.0 - equity_pct, 1)
    extra = 0.0
    reasons: list[str] = []
    rg = regime.get("regime", "SIDEWAYS")
    if rg in ("BEAR", "HIGH_VOLATILITY"):
        extra += 8.0
        reasons.append(f"{rg.replace('_', ' ')} regime → +8% storm buffer")
    if breadth and breadth.get("market_health") is not None and breadth["market_health"] < 40:
        extra += 8.0
        reasons.append(f"breadth {breadth['market_health']}/100 → +8% until the tape recovers")

    wait_for_pullback = []
    capped_liquidity = []
    for a in alloc:
        r = records_by_symbol.get(a["symbol"]) or {}
        rs = r.get("rs_rank")
        score = r.get("score")
        if rs is not None and score is not None and rs >= 95 and score < 72:
            wait_for_pullback.append(a["symbol"])
        liq = r.get("liquidity")
        if liq is not None and float(liq) < 0.3:
            capped_liquidity.append(a["symbol"])

    target_cash = round(min(50.0, base_cash + extra), 1)
    deploy = []
    if extra > 0:
        half = round(extra / 2, 1)
        deploy.append({"phase": "on_improvement", "pct": half, "trigger": "breadth > 40 and regime normalizes"})
        deploy.append({"phase": "reserve", "pct": round(extra - half, 1), "trigger": "held until the next scan confirms the improvement"})
    else:
        deploy.append({"phase": "now", "pct": 0.0, "trigger": "no buffer — deploy per the Kelly-Lite target"})
    return {
        "base_cash_pct": base_cash,
        "target_cash_pct": target_cash,
        "extra_buffer": round(extra, 1),
        "reasons": reasons,
        "wait_for_pullback": wait_for_pullback,
        "capped_liquidity": capped_liquidity,
        "deploy_schedule": deploy,
    }


def sector_weights(alloc: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for a in alloc:
        out[a["sector"]] = round(out.get(a["sector"], 0.0) + a["weight_pct"], 2)
    return out


def factor_exposure(alloc: list[dict], records_by_symbol: dict[str, dict]) -> dict:
    """Portfolio-weighted average factor scores + concentration readout.

    A book that is 80% momentum is a crowding risk — surface it.
    """
    factors = ("quality", "growth", "momentum", "valuation", "risk", "mb_score", "rs_rank")
    out: dict = {}
    for f in factors:
        num = den = 0.0
        for a in alloc:
            v = (records_by_symbol.get(a["symbol"]) or {}).get(f)
            if v is not None:
                num += v * a["weight_pct"]
                den += a["weight_pct"]
        out[f] = round(num / den, 1) if den > 0 else None
    comp = {k: v for k, v in out.items() if v is not None and k in ("quality", "growth", "momentum", "valuation", "risk")}
    if comp:
        top = max(comp, key=comp.get)
        out["top_factor"] = top
        out["top_factor_share"] = round(comp[top] / sum(comp.values()) * 100, 1)
    return out


def portfolio_risk(alloc: list[dict], records_by_symbol: dict[str, dict],
                   equity_pct: float) -> dict:
    """Portfolio-level risk readout: weighted vol, drawdown, concentration.

    Vol and max drawdown come from each name's stored per-stock metrics (now
    persisted with every scan), so the numbers reflect the actual book.
    `portfolio_vol` applies a diversification discount: a 10-name book with
    40% average vol does not swing like one name at 40% — effective N
    (1 / HHI) scales the average down, a standard equal-risk approximation
    that needs no covariance matrix.
    """
    n = len(alloc)
    if n == 0:
        return {"n": 0, "risk_grade": None}
    total = sum(a["weight_pct"] for a in alloc) or 1.0
    share = [a["weight_pct"] / total for a in alloc]
    hhi = round(sum(s * s for s in share), 3)

    def wavg(key: str):
        num = den = 0.0
        for a, s in zip(alloc, share):
            v = (records_by_symbol.get(a["symbol"]) or {}).get(key)
            if v is None:
                continue
            num += v * s
            den += s
        return (num / den) if den > 0 else None

    avg_vol = wavg("vol")
    avg_dd = wavg("max_dd")
    avg_quality = wavg("quality")
    avg_conf = wavg("data_confidence")

    eff_n = round(1.0 / hhi, 1) if hhi > 0 else float(n)
    port_vol = None
    if avg_vol is not None and avg_vol > 0 and eff_n > 0:
        port_vol = round(avg_vol / (eff_n ** 0.5), 4)

    top = sorted(share, reverse=True)
    top1 = round(top[0] * 100, 1) if top else 0.0
    top3 = round(sum(top[:3]) * 100, 1)

    grade = "BALANCED"
    if port_vol is not None and avg_dd is not None:
        if port_vol < 0.18 and avg_dd > -0.35:
            grade = "CONSERVATIVE"
        elif port_vol < 0.30 and avg_dd > -0.50:
            grade = "BALANCED"
        else:
            grade = "AGGRESSIVE"
    elif port_vol is not None:
        grade = "CONSERVATIVE" if port_vol < 0.18 else ("BALANCED" if port_vol < 0.30 else "AGGRESSIVE")

    return {
        "n": n,
        "equity_pct": round(equity_pct, 1),
        "cash_pct": round(max(0.0, 100 - equity_pct), 1),
        "avg_vol": round(avg_vol * 100, 1) if avg_vol is not None else None,
        "portfolio_vol": round(port_vol * 100, 1) if port_vol is not None else None,
        "avg_max_dd": round(avg_dd * 100, 1) if avg_dd is not None else None,
        "effective_n": eff_n,
        "hhi": hhi,
        "top1_share": top1,
        "top3_share": top3,
        "avg_quality": round(avg_quality, 1) if avg_quality is not None else None,
        "avg_confidence": round(avg_conf, 1) if avg_conf is not None else None,
        "risk_grade": grade,
    }


# ── Tax-aware rebalancing ─────────────────────────────────────────────────

REBALANCE_PARAMS = {
    # Flat Indian equity tax assumptions (delivery): 30% STCG within a year,
    # 10% LTCG beyond it. Gains on sold names are assumed at `assumed_gain_pct`
    # — the plan is advisory; replace with real cost basis when available.
    "tax_stcg_rate": 0.30,
    "tax_ltcg_rate": 0.10,
    "ltcg_days": 365,
    "assumed_gain_pct": 20.0,
    "round_trip_cost_pct": 0.50,  # two sides × ~0.25% (the backtest cost stack)
    "min_delta_pct": 2.0,         # ignore weight drift below this (no churn)
    "capital": 1_000_000,         # ₹10L book used to translate % → notional
}


def rebalance_plan(
    target: list[dict],
    prev: Optional[list[dict]],
    params: Optional[dict] = None,
    first_seen: Optional[dict] = None,
    price_pairs: Optional[dict] = None,
) -> dict:
    """Tax-aware rebalance from the previous saved allocation to `target`.

    The core anti-churn rule: a name still in the top-N with drift under
    `min_delta_pct` is HOLD — the engine will NOT sell a winner just to
    re-buy it (that's how turnover and taxes destroy returns). Only names
    that drop out of the target, or drift beyond tolerance, realize gains,
    taxed as STCG or LTCG by how long the name has been held (first-seen
    scan date as the holding-start proxy).

    `price_pairs` is an optional {symbol: (entry_price, current_price)} map;
    when present, realized gains use the real entry vs current price instead
    of the `assumed_gain_pct` fallback.
    """
    p = {**REBALANCE_PARAMS, **(params or {})}
    first_seen = first_seen or {}
    price_pairs = price_pairs or {}
    prev = prev or []
    today = date.today()
    capital = float(p.get("capital", 1_000_000))
    min_delta = float(p.get("min_delta_pct", 2.0))

    prev_map = {r["symbol"]: r for r in prev}
    target_map = {a["symbol"]: a for a in target}
    trades: list[dict] = []
    realized = 0.0
    buy_notional = 0.0
    stcg_tax = 0.0
    ltcg_tax = 0.0

    for sym, pr in prev_map.items():
        pw = float(pr.get("weight_pct") or 0.0)
        tw = float((target_map.get(sym) or {}).get("weight_pct") or 0.0)
        delta = tw - pw
        if abs(delta) <= min_delta and sym in target_map:
            side, notional, reason = "HOLD", 0.0, "within tolerance — no churn"
        elif sym not in target_map:
            side, notional, reason = "SELL", pw / 100.0 * capital, "dropped out of top-N"
        elif delta > 0:
            side, notional, reason = "ADD", delta / 100.0 * capital, "conviction up"
        else:
            side, notional, reason = "TRIM", -delta / 100.0 * capital, "conviction down"

        tax, kind = 0.0, None
        est_gain_pct = None
        if side in ("SELL", "TRIM") and notional > 0:
            pair = price_pairs.get(sym)
            if pair and pair[0] and pair[1] and pair[0] > 0:
                est_gain_pct = (pair[1] / pair[0] - 1.0) * 100.0
            else:
                est_gain_pct = float(p.get("assumed_gain_pct", 20.0))
            gain = notional * est_gain_pct / 100.0
            fs = first_seen.get(sym)
            held_days = (today - date.fromisoformat(fs)).days if fs else 0
            if held_days > float(p.get("ltcg_days", 365)):
                kind = "LTCG"
                tax = gain * float(p.get("tax_ltcg_rate", 0.10))
            else:
                kind = "STCG"
                tax = gain * float(p.get("tax_stcg_rate", 0.30))
            if kind == "LTCG":
                ltcg_tax += tax
            else:
                stcg_tax += tax
            realized += notional

        trades.append(
            {
                "symbol": sym,
                "side": side,
                "prev_pct": round(pw, 2),
                "target_pct": round(tw, 2),
                "notional": round(notional, 2),
                "est_tax": round(tax, 2),
                "est_gain_pct": round(est_gain_pct, 2) if est_gain_pct is not None else None,
                "tax_kind": kind,
                "reason": reason,
            }
        )

    for sym, a in target_map.items():
        if sym not in prev_map:
            notional = float(a.get("weight_pct") or 0.0) / 100.0 * capital
            buy_notional += notional
            trades.append(
                {
                    "symbol": sym,
                    "side": "BUY",
                    "prev_pct": 0.0,
                    "target_pct": round(float(a.get("weight_pct") or 0.0), 2),
                    "notional": round(notional, 2),
                    "est_tax": 0.0,
                    "est_gain_pct": None,
                    "tax_kind": None,
                    "reason": "new in top-N",
                }
            )

    order = {"SELL": 0, "TRIM": 1, "BUY": 2, "ADD": 3, "HOLD": 4}
    trades.sort(key=lambda t: order.get(t["side"], 9))

    total_tax = stcg_tax + ltcg_tax
    cost_est = (realized + buy_notional) * float(p.get("round_trip_cost_pct", 0.5)) / 100.0
    traded = realized + buy_notional
    return {
        "n_prev": len(prev),
        "n_target": len(target),
        "trades": trades,
        "realized_notional": round(realized, 2),
        "buy_notional": round(buy_notional, 2),
        "cost_est": round(cost_est, 2),
        "stcg_tax": round(stcg_tax, 2),
        "ltcg_tax": round(ltcg_tax, 2),
        "total_tax": round(total_tax, 2),
        "total_drag_pct": round((total_tax + cost_est) / capital * 100, 2) if capital else 0.0,
        "turnover_pct": round(traded / capital * 100, 1) if capital else 0.0,
        "assumptions": {
            "gain_pct": p.get("assumed_gain_pct"),
            "stcg_rate": p.get("tax_stcg_rate"),
            "ltcg_rate": p.get("tax_ltcg_rate"),
            "capital": capital,
        },
    }


def build_allocation(
    records: list[dict],
    equity_pct: float,
    top_n: int = 8,
    mode: str = "kelly",
    max_sector_weight: float = 25.0,
    budgets: Optional[dict] = None,
) -> list[dict]:
    """Conviction-weighted allocation over the top-N by position score.

    mode="kelly" (default): weight ∝ pos_score · quality · 1/vol · DD-penalty.
    mode="conviction": weight ∝ pos_score² — pure score tapering.
    Both normalize to equity_pct, then enforce `max_sector_weight`.
    """
    ranked = sorted(
        [r for r in records if r.get("pos_score") is not None],
        key=lambda r: r.get("pos_score") or 0,
        reverse=True,
    )[: max(1, min(int(top_n), 25))]

    if mode == "conviction":
        scores = [(r.get("pos_score") or 0.0) ** 2 for r in ranked]
    else:
        scores = [_kelly_lite(r) for r in ranked]
    total = sum(scores)
    out = []
    for r, s in zip(ranked, scores):
        w = (s / total * equity_pct) if total > 0 else equity_pct / len(ranked)
        out.append(
            {
                "symbol": r.get("symbol"),
                "name": r.get("name") or r.get("symbol"),
                "sector": r.get("sector") or "Unknown",
                "pos_score": r.get("pos_score"),
                "mb_bucket": r.get("mb_bucket"),
                "weight_pct": round(w, 2),
                "mode": mode,
            }
        )
    out = _enforce_sector_caps(out, float(max_sector_weight))
    if budgets:
        out = _enforce_budgets(out, budgets, float(max_sector_weight))
    return out
