"""Rate limiting utilities."""

import time as _rate_time
from collections import defaultdict


class RateLimiter:
    def __init__(self, max_requests: int = 60, window_sec: int = 60):
        self.max_requests = max_requests
        self.window_sec = window_sec
        self._requests = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        now = _rate_time.time()
        window_start = now - self.window_sec
        self._requests[key] = [t for t in self._requests[key] if t > window_start]
        if len(self._requests[key]) >= self.max_requests:
            return False
        self._requests[key].append(now)
        return True


_rate_limiter = RateLimiter(max_requests=60, window_sec=60)
_chat_rate_limiter = RateLimiter(max_requests=10, window_sec=60)  # Stricter for chat
# 2026-05-02 (v30.5): the OpenAI-compatible endpoint at /v1/chat/completions
# is used by agent harnesses (Flue, OpenAI Agents SDK, Vercel AI SDK,
# LangChain, …) that fire many requests per second during a single
# tool-calling loop. The 10/min cap on _chat_rate_limiter dethrones any
# real agent within seconds. Agent traffic is (a) much cheaper than
# Open-WebUI human chat (no markdown rendering, no streaming UX) and
# (b) self-throttling via tool execution time, so a more generous cap is
# appropriate. 240/min ≈ 4/sec sustained, with 5×-burst headroom — fine
# for any realistic single-agent loop, still tight enough that a runaway
# would be visible in a minute.
_openai_api_rate_limiter = RateLimiter(max_requests=240, window_sec=60)


# ── Real-client-IP extraction ────────────────────────────────────────────────
#
# 2026-05-09: every external request lands on uvicorn from 127.0.0.1
# because Caddy is the sole upstream proxy. ``request.client.host`` is
# therefore always ``127.0.0.1`` for traffic crossing Caddy, which means
# the in-handler chat rate limiters (keyed on ``request.client.host``)
# were treating EVERY external client as the same bucket. A single
# Hetzner-FI client (135.181.213.189 / 2a01:4f9:3a:2b93::2) flooded
# /v1/chat/completions at ~33 req/s of 28 KB prompts, saturating the
# connection pool and starving every other endpoint. The 240 req/min cap
# on the OpenAI surface looked like global instead of per-client, so the
# limiter would gate the dashboard's polite polling alongside the
# attacker rather than gating just the attacker. Fix: make every chat
# handler key on the same canonical client IP the global middleware
# uses (``cf-connecting-ip`` → ``x-forwarded-for`` → ``x-real-ip`` →
# socket peer), so a single abuser fills only their own bucket.
_PROXY_LOCAL = {"127.0.0.1", "::1", "localhost"}


def client_real_ip(request) -> tuple[str, bool]:
    """Return ``(key, is_internal)`` for rate-limit / per-client keying.

    Header preference matches ``api.server._client_key``:

    1. ``CF-Connecting-IP`` — Cloudflare's verified original client IP
       (the only way to distinguish a real abuser from a flood across
       multiple Cloudflare edge IPs).
    2. Leftmost ``X-Forwarded-For`` token — any RFC 7239 proxy chain.
    3. ``X-Real-Ip`` — Caddy fallback when CF isn't in front.
    4. ``request.client.host`` — direct socket peer.

    ``is_internal`` is ``True`` only when the request came from a local
    process that didn't go through Caddy (e.g. the Next.js SSR loop, a
    local healthcheck, or chat-keeper probing). Internal callers MUST
    NOT be rate-limited or the dashboard's tight polling self-DoSes.
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
