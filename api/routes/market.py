"""Market data endpoints: price, TMC config, metagraph."""

import traceback

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..config import NETUID, CACHE_TTL, TMC_BASE
from ..helpers.cache import _get_cached, _get_stale, _set_cached, _bg_refresh
from ..helpers.fetch import _fetch_metagraph, _fetch_price

router = APIRouter()


@router.get("/api/metagraph", tags=["Metagraph"], summary="Full subnet metagraph",
         description="""Returns all 256 UIDs with on-chain data: hotkey, coldkey, stake, trust, consensus, incentive, emission, and dividends.

**Cached for 60s** - background refreshes keep data fresh without blocking requests.

Response includes:
- `block`: Current Bittensor block number
- `n`: Number of UIDs in the subnet (256)
- `neurons[]`: Array of all UIDs with their on-chain metrics
""",
         response_description="Metagraph with all 256 UIDs and their on-chain metrics")
def get_metagraph():
    # Fast: return cache immediately, refresh in background if stale
    cached = _get_cached("metagraph", CACHE_TTL)
    if cached:
        return JSONResponse(content=cached, headers={"Cache-Control": "public, max-age=30, stale-while-revalidate=60"})
    # No fresh cache - return stale if available, and refresh in background
    stale = _get_stale("metagraph")
    if stale:
        _bg_refresh("metagraph", _fetch_metagraph)
        return JSONResponse(content=stale, headers={"Cache-Control": "public, max-age=30, stale-while-revalidate=60"})
    # No cache at all - must block (first ever request)
    try:
        result = _fetch_metagraph()
        _set_cached("metagraph", result)
        return JSONResponse(content=result, headers={"Cache-Control": "public, max-age=30, stale-while-revalidate=60"})
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


@router.get("/api/price", tags=["Market"], summary="Token price and market data",
         description="""Returns SN97 alpha token pricing, TAO/USD rate, pool liquidity, emission, and volume.

Response includes:
- `alpha_price_tao` / `alpha_price_usd`: Current alpha token price
- `tao_usd`: TAO/USD exchange rate (via CoinGecko)
- `alpha_in_pool` / `tao_in_pool`: DEX pool liquidity
- `marketcap_tao`: Total market cap in TAO
- `emission_pct`: Current emission allocation percentage
- `price_change_1h`, `_24h`, `_7d`: Price change percentages
- `miners_tao_per_day`: Total TAO earned by miners per day

**Cached for 30s.**
""")
def get_price():
    cached = _get_cached("price", 30)
    if cached:
        return JSONResponse(content=cached, headers={"Cache-Control": "public, max-age=10, stale-while-revalidate=30"})
    stale = _get_stale("price")
    if stale:
        _bg_refresh("price", _fetch_price)
        return JSONResponse(content=stale, headers={"Cache-Control": "public, max-age=10, stale-while-revalidate=30"})
    try:
        result = _fetch_price()
        _set_cached("price", result)
        return JSONResponse(content=result, headers={"Cache-Control": "public, max-age=10, stale-while-revalidate=30"})
    except Exception as e:
        return {"error": str(e)}


@router.get("/api/tmc-config", tags=["Market"], summary="TaoMarketCap SSE config",
         description="Returns SSE (Server-Sent Events) URLs for real-time price and subnet data from TaoMarketCap. Used by the dashboard for live price updates.")
def get_tmc_config():
    return {
        "sse_price_url": f"{TMC_BASE}/public/v1/sse/subnets/prices/",
        "sse_subnet_url": f"{TMC_BASE}/public/v1/sse/subnets/{NETUID}/",
        "netuid": NETUID,
    }
