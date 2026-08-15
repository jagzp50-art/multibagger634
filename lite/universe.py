"""
Sovereign Lite v7 — curated NSE universe.

A single-user scanner stays fast and rate-limit friendly, so we screen a
curated ~100 name universe of large, mid and small caps instead of the full
exchange. Add symbols via the dashboard (or the `add_stock` helper); every
symbol is fetched through yfinance using its `.NS` suffix.
"""
from __future__ import annotations

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
    """Merged, de-duplicated universe of {symbol, name, sector} dicts."""
    seen: set[str] = set()
    out: list[dict] = []
    for symbol in LARGE + MID:
        if symbol in seen:
            continue
        seen.add(symbol)
        sector = "Large Cap" if symbol in LARGE else "Mid/Small Cap"
        out.append({"symbol": symbol, "name": symbol.replace(".NS", ""), "sector": sector})
    return out
