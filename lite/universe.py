"""
Sovereign Lite v16 — two-tier curated NSE universe.

  CORE      — ~155 curated large/mid/small caps (fast daily scan).
  DISCOVERY — ~450 more names mined from the repo's broader NSE symbol list
              (pro/ticker_list.py): NIFTY-500-style breadth for a slower
              weekly scan, so multibaggers get found before they are obvious.

Both tiers are fetched through yfinance using the `.NS` suffix. Add symbols
via the dashboard (or the `add_stock` helper).
"""
from __future__ import annotations

import os
import re

# Liquid large + mid caps (NIFTY 50 core)
LARGE = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS", "AXISBANK.NS",
    "KOTAKBANK.NS", "HINDUNILVR.NS", "BAJFINANCE.NS", "M&M.NS", "TATAMOTORS.NS",
    "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "NTPC.NS",
    "POWERGRID.NS", "ADANIENT.NS", "ADANIPORTS.NS", "WIPRO.NS", "TECHM.NS",
    "HCLTECH.NS", "ASIANPAINT.NS", "NESTLEIND.NS", "JSWSTEEL.NS", "TATASTEEL.NS",
    "BAJAJFINSV.NS", "COALINDIA.NS", "ONGC.NS", "BPCL.NS", "IOC.NS",
    "GRASIM.NS", "HEROMOTOCO.NS", "EICHERMOT.NS", "DRREDDY.NS", "CIPLA.NS",
    "APOLLOHOSP.NS", "DIVISLAB.NS", "BRITANNIA.NS", "HINDALCO.NS", "DLF.NS",
    "INDUSINDBK.NS", "HDFCLIFE.NS", "SBILIFE.NS", "BAJAJ-AUTO.NS", "TATACONSUM.NS",
]

# Quality mid / small caps — the natural multibagger hunting ground
MID = [
    "TRENT.NS", "PIDILITIND.NS", "POLYCAB.NS", "PERSISTENT.NS", "COFORGE.NS",
    "BSE.NS", "CDSL.NS", "CAMS.NS", "MCX.NS", "IRCTC.NS",
    "INDIAMART.NS", "LALPATHLAB.NS", "METROPOLIS.NS", "VIJAYA.NS", "NH.NS",
    "KIMS.NS", "MAXHEALTH.NS", "FORTIS.NS", "NATCOPHARM.NS", "AJANTPHARM.NS",
    "GLAXO.NS", "GLENMARK.NS", "LUPIN.NS", "ZYDUSLIFE.NS", "TORNTPHARM.NS",
    "DIXON.NS", "KAYNES.NS", "AMBER.NS", "SYRMA.NS", "CYIENT.NS",
    "KPITTECH.NS", "TANLA.NS", "ZENSARTECH.NS", "LTTS.NS", "MPHASIS.NS",
    "DEEPAKNTR.NS", "NAVINFLUOR.NS", "VINATIORGA.NS", "SRF.NS", "AARTIIND.NS",
    "CLEAN.NS", "SUMICHEM.NS", "PIIND.NS", "UPL.NS", "BAYERCROP.NS",
    "GRINDWELL.NS", "CARBORUNIV.NS", "AIAENG.NS", "SUPREMEIND.NS", "ASTRAL.NS",
    "ATUL.NS", "CROMPTON.NS", "HAVELLS.NS", "SIEMENS.NS", "ABB.NS",
    "BEL.NS", "DATAPATTNS.NS", "HAL.NS", "ZENTEC.NS", "PARAS.NS",
    "JYOTHYLAB.NS", "RADICO.NS", "BIKAJI.NS", "DODLA.NS", "GODREJPROP.NS",
    "OBEROIRLTY.NS", "PRESTIGE.NS", "BRIGADE.NS", "SONATSOFTW.NS", "TECHNOE.NS",
    "RAMCOCEM.NS", "SHREECEM.NS", "ACC.NS", "AMBUJACEM.NS", "DALBHARAT.NS",
    "ICICIGI.NS", "ICICIPRULI.NS", "STARHEALTH.NS", "POLICYBZR.NS", "PAYTM.NS",
    "TATAPOWER.NS", "ADANIGREEN.NS", "ADANIPOWER.NS", "SUZLON.NS", "INOXWIND.NS",
    "WAAREEENER.NS", "PREMIERENE.NS", "NHPC.NS", "SJVN.NS", "PFC.NS",
    "RECLTD.NS", "LICI.NS", "HUDCO.NS", "IRFC.NS", "RVNL.NS",
    "TIINDIA.NS", "UNOMINDA.NS", "SUNDRMFAST.NS", "BHARATFORG.NS", "CUMMINSIND.NS",
    "THERMAX.NS", "KIRLOSENG.NS", "BHEL.NS", "LTIM.NS", "OFSS.NS",
]

# Benchmarks / indices used for regime + relative strength
BENCHMARK = "^NSEI"          # NIFTY 50
VIX_INDEX = "^INDIAVIX"      # India VIX


def default_universe() -> list[dict]:
    """Merged, de-duplicated CORE universe of {symbol, name, sector} dicts."""
    seen: set[str] = set()
    out: list[dict] = []
    for symbol in LARGE + MID:
        if symbol in seen:
            continue
        seen.add(symbol)
        sector = "Large Cap" if symbol in LARGE else "Mid/Small Cap"
        out.append({"symbol": symbol, "name": symbol.replace(".NS", ""), "sector": sector})
    return out


def discovery_universe() -> list[dict]:
    """DISCOVERY tier: broader NSE names (NIFTY-500 style breadth) not already
    in the core universe, mined from the repo's `pro/ticker_list.py`.

    Read at runtime (a plain data file, no pro/ imports) so Lite stays free of
    the legacy stack; returns [] if the file is absent.
    """
    core = {s["symbol"] for s in default_universe()}
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pro", "ticker_list.py")
    try:
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
    except OSError:
        return []
    m = re.search(r"TICKERS\s*=\s*\[(.*?)\]", src, re.S)
    if not m:
        return []
    symbols = re.findall(r'"([A-Z0-9.\-]+\.(?:NS|BO))"', m.group(1))
    out: list[dict] = []
    seen: set[str] = set()
    for symbol in symbols:
        if symbol in core or symbol in seen:
            continue
        seen.add(symbol)
        out.append({"symbol": symbol, "name": symbol.replace(".NS", "").replace(".BO", ""), "sector": "Discovery"})
    return out
