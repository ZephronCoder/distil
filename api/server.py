"""Distil - Subnet 97 API. App creation, middleware, startup."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from config import ALLOWED_ORIGINS, API_DESCRIPTION
from helpers.rate_limit import _rate_limiter
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
# Fix: prefer X-Real-Ip (set by Caddy via ``header_up X-Real-Ip {remote_host}``)
# and fall back to the leftmost X-Forwarded-For for Cloudflare hops; only
# treat ``127.0.0.1`` as "internal SSR" when no proxy header is present
# (the dashboard's local Next.js loop hits us without those headers).

_PROXY_LOCAL = {"127.0.0.1", "::1", "localhost"}


def _client_key(request) -> tuple[str, bool]:
    """Return (key, is_internal) for rate limiting.

    Header preference: ``CF-Connecting-IP`` (Cloudflare's verified
    original client IP — this is the only way to distinguish the real
    abuser from a flood across Cloudflare edges) → leftmost
    ``X-Forwarded-For`` (any RFC 7239-shaped proxy chain) → ``X-Real-Ip``
    (Caddy fallback for non-CF clients) → socket peer.

    ``is_internal`` is True only when the request came from a process on
    the same host without traversing Caddy — e.g. the Next.js SSR loop
    or a local curl probe. Those should not be rate-limited.
    """
    cf_ip = request.headers.get("cf-connecting-ip")
    if cf_ip:
        return cf_ip.strip(), False
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        first = fwd.split(",", 1)[0].strip()
        if first:
            return first, False
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip(), False
    raw_host = request.client.host if request.client else "unknown"
    return raw_host, raw_host in _PROXY_LOCAL


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        path = request.url.path
        if path in ("/docs", "/redoc", "/openapi.json"):
            return await call_next(request)
        if path in ("/api/chat", "/v1/chat/completions", "/v1/models"):
            # These endpoints carry their own (stricter) limiter in-handler
            return await call_next(request)
        key, is_internal = _client_key(request)
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
