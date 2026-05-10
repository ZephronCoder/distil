"""Distil - Subnet 97 API. App creation, middleware, startup."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from config import ALLOWED_ORIGINS, API_DESCRIPTION
from helpers.rate_limit import _PROXY_LOCAL, _rate_limiter, client_real_ip
from helpers.cache import _bg_refresh
from helpers.fetch import _fetch_metagraph, _fetch_commitments, _fetch_price

# Import routers
from routes.health import router as health_router
from routes.miners import router as miners_router
from routes.evaluation import router as evaluation_router
from routes.market import router as market_router
from routes.chat import router as chat_router
from routes.debugging import router as debugging_router
from routes.telemetry import router as telemetry_router


app = FastAPI(
    title="Distil - Subnet 97 API",
    description=API_DESCRIPTION,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Overview", "description": "API info and health checks"},
        {"name": "Metagraph", "description": "On-chain subnet data - UIDs, stakes, weights, incentive"},
        {"name": "Miners", "description": "Miner model commitments and scores"},
        {"name": "Evaluation", "description": "Live eval progress, head-to-head rounds, and score history"},
        {"name": "Market", "description": "Token pricing, emission, and market data"},
        {"name": "Chat", "description": "Chat with the current king model (when GPU is available)"},
        {"name": "Telemetry", "description": "Dashboard telemetry — composite axes, DQs, validator events, pod health"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Rate limiting middleware for all endpoints ────────────────────────────────
#
# 2026-05-04: every external request lands on uvicorn from 127.0.0.1 because
# Caddy is the sole upstream client. The previous middleware blanket-exempted
# 127.0.0.1 to keep the dashboard SSR loose, which inadvertently made the
# API limiter a no-op for *all* traffic — a single misbehaving scraper was
# pushing ~1100 /api/eval-data requests/sec through Caddy and starving the
# uvicorn worker pool, surfacing as 503s on chat (api/routes/chat.py) and
# every other endpoint.
#
# 2026-05-09: the canonical extraction logic now lives in
# ``helpers.rate_limit.client_real_ip`` so chat.py's per-handler
# limiters key on the same identity (cf-connecting-ip → xff → x-real-ip
# → peer). Pre-fix the chat handlers used ``request.client.host`` which
# was always ``127.0.0.1`` behind Caddy, so a single abuser filled the
# bucket for *all* clients and the OpenAI surface got a flood + 200%
# CPU pile-up.


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        path = request.url.path
        if path in ("/docs", "/redoc", "/openapi.json"):
            return await call_next(request)
        if path in ("/api/chat", "/v1/chat/completions", "/v1/models"):
            # These endpoints carry their own (stricter) limiter in-handler
            return await call_next(request)
        key, is_internal = client_real_ip(request)
        if is_internal:
            return await call_next(request)
        if not _rate_limiter.is_allowed(key):
            return JSONResponse(
                status_code=429,
                content={"error": "rate limit exceeded"},
                headers={"Retry-After": "30"},
            )
        return await call_next(request)


app.add_middleware(RateLimitMiddleware)


# ── Include routers ──────────────────────────────────────────────────────────

app.include_router(health_router)
app.include_router(miners_router)
app.include_router(evaluation_router)
app.include_router(market_router)
app.include_router(chat_router)
app.include_router(debugging_router)
app.include_router(telemetry_router)


# ── Startup: prime caches ────────────────────────────────────────────────────

@app.on_event("startup")
def prime_caches():
    """On startup, kick off background refreshes so first request is fast."""
    _bg_refresh("metagraph", _fetch_metagraph)
    _bg_refresh("commitments", _fetch_commitments)
    _bg_refresh("price", _fetch_price)
