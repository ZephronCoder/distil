"""OpenAI-compatible chat endpoints; proxies to king vLLM via chat-tunnel.

This module is a *thin transport layer*. All multi-turn orchestration,
tool calling, sandboxed code execution, web search, SN97 lookup, etc.
live in :mod:`api.agent_runner` (built on the OpenAI Agents SDK).

What this module is responsible for:
  * Picking the live king (``_get_king_info``) and pinning the served
    model name to ``sn97-king`` when forwarding to vLLM.
  * Maintaining a single pooled httpx client to the local chat tunnel.
  * Mapping every transport-level vLLM failure (incl. PoolTimeout,
    RemoteProtocolError, 5xx) to ``_ChatPodUnavailable`` -> 503 so the
    public surface NEVER returns 500.
  * Per-turn JSONL audit logging + a 10-second status cache.
  * Two public surfaces:
      - ``/api/chat`` + ``/api/chat/status``       (dashboard)
      - ``/v1/models`` + ``/v1/chat/completions``  (OpenAI-compatible)

Everything tool-call / reasoning / tool-output-parsing related is in
``api.agent_runner``. There is a single feature flag,
``DISTIL_CHAT_AGENT_HARNESS=0`` (default ON), that drops back to the bare
proxy passthrough for emergency rollback. There is no other path."""

import json
import os
import threading
import time

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from config import (
    CHAT_POD_HOST,
    CHAT_POD_PORT,
    STATE_DIR,
)
from helpers.rate_limit import (
    _chat_rate_limiter,
    _openai_api_rate_limiter,
    client_real_ip,
)
from helpers.sanitize import _safe_json_load
from state_store import (
    eval_progress,
    h2h_latest,
    read_cache,
    read_state,
    uid_hotkey_map,
)

router = APIRouter()


# ── King info helper ──────────────────────────────────────────────────────────

def _get_king_info():
    h2h = h2h_latest()
    king_uid = h2h.get("king_uid")
    if king_uid is not None:
        for r in h2h.get("results", []):
            if r.get("is_king") or r.get("uid") == king_uid:
                return king_uid, r.get("model")
        commitments_data = read_cache("commitments", {})
        commitments = (
            commitments_data.get("commitments", commitments_data)
            if isinstance(commitments_data, dict)
            else {}
        )
        king_hotkey = uid_hotkey_map().get(str(king_uid))
        if king_hotkey and king_hotkey in commitments:
            info = commitments[king_hotkey]
            return king_uid, info.get("model") if isinstance(info, dict) else info
        return king_uid, None

    # h2h_latest can sit at king_uid=None for an eval generation after
    # a hard arch cutover. chat-keeper's vLLM still serves the prior
    # king and we surface that here so chat stays live across the gap.
    chat_pod_state = read_state("chat_pod.json", {}) or {}
    fallback_model = chat_pod_state.get("model")
    if fallback_model:
        return -1, fallback_model
    return None, None


# ── Chat pod transport ───────────────────────────────────────────────────────

# chat_server.py serves the king under the fixed name "sn97-king" so
# we rewrite client-supplied model names below.
CHAT_POD_SERVED_MODEL = "sn97-king"

# chat-tunnel.service forwards 127.0.0.1:8100 -> chat-pod:8100; using
# localhost reuses a TCP keep-alive and detects tunnel-down in <2s.
_LOCAL_CHAT_BASE = f"http://127.0.0.1:{CHAT_POD_PORT}"

_chat_http_client: httpx.AsyncClient | None = None
_chat_http_lock = threading.Lock()


_CHAT_MODEL_MAX_LEN = int(os.environ.get("DISTIL_CHAT_MODEL_MAX_LEN", "32768"))
_CHAT_AGENT_OVERHEAD_TOKENS = int(
    os.environ.get("DISTIL_CHAT_AGENT_OVERHEAD_TOKENS", "2048")
)
_CHAT_MIN_OUTPUT_TOKENS = int(
    os.environ.get("DISTIL_CHAT_MIN_OUTPUT_TOKENS", "256")
)


def _estimate_input_tokens(messages):
    """Cheap upper-bound token estimate for the chat-king context budget.

    Assumes ~3 chars/token (correct order of magnitude for English; a
    hair pessimistic for code-heavy text where BPE splits punctuation
    aggressively) and adds 8 tokens of per-message chat-template
    overhead. We deliberately avoid shipping a tokenizer in the API
    process -- the clamp's job is to keep us under the hard 32 K vLLM
    limit, not to be exact.
    """
    total = 0
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            total += max(1, len(content) // 3)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text", "") or ""
                    if isinstance(text, str):
                        total += max(1, len(text) // 3)
        total += 8
    return total


def _clamp_for_context_budget(
    messages,
    max_tokens,
    *,
    model_max_len=_CHAT_MODEL_MAX_LEN,
    overhead=_CHAT_AGENT_OVERHEAD_TOKENS,
    floor=_CHAT_MIN_OUTPUT_TOKENS,
):
    """Trim oldest non-system messages and lower ``max_tokens`` so
    ``input + output + overhead`` stays under ``model_max_len``.

    The chat pod runs vLLM with ``max_model_len=32768`` and the agent
    harness layers ~1.5 K tokens of system-prompt + tool-definition
    overhead on top of the raw messages. Without this clamp, a
    multi-turn dashboard session that grew past ~30 K input tokens
    would 400 with ``maximum context length is 32768 tokens``. We
    preserve every system message and the most recent non-system turn
    (user's current question) and only drop history in between.

    Returns ``(trimmed_messages, clamped_max_tokens)``.
    """
    msgs = list(messages or [])
    available = max(floor + 1, model_max_len - overhead)
    estimated = _estimate_input_tokens(msgs)
    while estimated + floor > available and len(msgs) > 1:
        idx = next(
            (
                i
                for i, m in enumerate(msgs)
                if isinstance(m, dict) and m.get("role") != "system"
            ),
            None,
        )
        if idx is None or idx == len(msgs) - 1:
            break
        msgs.pop(idx)
        estimated = _estimate_input_tokens(msgs)
    budget = max(floor, available - estimated)
    try:
        requested = int(max_tokens)
    except (TypeError, ValueError):
        requested = budget
    return msgs, max(floor, min(requested, budget))


def _get_http_client() -> httpx.AsyncClient:
    """Return the module-level pooled httpx client, creating it on first use."""
    global _chat_http_client
    if _chat_http_client is None:
        with _chat_http_lock:
            if _chat_http_client is None:
                _chat_http_client = httpx.AsyncClient(
                    base_url=_LOCAL_CHAT_BASE,
                    timeout=httpx.Timeout(connect=3.0, read=90.0, write=10.0, pool=5.0),
                    limits=httpx.Limits(
                        max_connections=64,
                        max_keepalive_connections=32,
                        keepalive_expiry=30.0,
                    ),
                )
    return _chat_http_client


def _normalize_chat_payload(payload: dict) -> dict:
    """Pin served-model to sn97-king and apply orchestrator defaults.

    * ``enable_thinking`` is forced ON. The king is a reasoning model and
      we want the public chat to use the same think → answer pipeline
      that the held-out benchmarks now use. The agent runner already
      routes ``<think>...</think>`` content through the reasoning
      channel so the user sees the trace, not the raw thoughts.
    * ``max_tokens`` defaults to 16 K and the agent ceiling is 30 K
      (chat pod runs vLLM with ``max_model_len=32768``; we leave 2 K of
      headroom for the prompt). The whole point: the king is a
      reasoning model and we want the chain-of-thought to never
      truncate so the answer reflects the model's real capability —
      the same uncap-the-think rationale that drove the v32-think-budget
      bump on the held-out benchmarks and capability axes.
    * A formatting/system prompt is injected if the client didn't ship
      one (math-rendering hint that survives the LaTeX layout).
    """
    payload = dict(payload)
    payload["model"] = CHAT_POD_SERVED_MODEL
    kwargs = dict(payload.get("chat_template_kwargs") or {})
    # Default ON. Callers (eg. clients that *want* a fast no-think
    # response) can still pass ``chat_template_kwargs.enable_thinking
    # = False`` explicitly and we'll honour it.
    kwargs.setdefault("enable_thinking", True)
    payload["chat_template_kwargs"] = kwargs
    if payload.get("max_tokens") is None:
        payload["max_tokens"] = 16384
    msgs = list(payload.get("messages") or [])
    has_system = any(
        (isinstance(m, dict) and m.get("role") == "system") for m in msgs
    )
    if not has_system:
        formatting_guide = (
            "You are a helpful assistant. Take your time to think before "
            "answering: use the ``<think>...</think>`` block (the runtime "
            "renders this as 'thinking' in the dashboard) for chain-of-"
            "thought, then write the user-visible answer outside the "
            "think block. When you write math, ALWAYS use LaTeX with "
            "consistent delimiters: ``$...$`` for inline math (e.g., "
            "the speed $v=d/t$) and ``$$...$$`` on their OWN lines for "
            "block math. Never emit bare LaTeX commands (``\\text``, "
            "``\\frac``, ``\\times``, ``\\approx``) outside of these "
            "delimiters. For simple arithmetic prefer plain text "
            "(``2.36 × 10^22`` is fine) — reserve LaTeX for multi-line "
            "derivations and equations with fractions, integrals, or "
            "sums. Use Markdown for headers, lists, and code blocks."
        )
        payload["messages"] = [
            {"role": "system", "content": formatting_guide}
        ] + msgs
    return payload


# Default ON: drive chat through the OpenAI Agents SDK harness in
# api.agent_runner. Set DISTIL_CHAT_AGENT_HARNESS=0 to drop to bare
# proxy (emergency rollback only — no tool-calling, no python_exec,
# no web_search, no SN97 lookup, no model_info).
_AGENT_HARNESS_ENABLED = (
    os.environ.get("DISTIL_CHAT_AGENT_HARNESS", "1") != "0"
)


class _ChatPodUnavailable(RuntimeError):
    """Raised when the local tunnel to the chat pod is not reachable."""


async def _local_chat_post(payload: dict, *, timeout: float = 90.0) -> dict:
    """Async POST to the chat tunnel; raises _ChatPodUnavailable for
    transport-level failures so the caller maps cleanly to a 503."""
    client = _get_http_client()
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json=_normalize_chat_payload(payload),
            timeout=httpx.Timeout(connect=3.0, read=timeout, write=10.0, pool=5.0),
        )
    except (
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.PoolTimeout,
        httpx.RemoteProtocolError,
        httpx.WriteTimeout,
    ) as e:
        raise _ChatPodUnavailable(f"{type(e).__name__}: {e}") from e
    except httpx.HTTPError as e:
        # Catch-all for any remaining transport-level error so an
        # unexpected httpx variant cannot 500 the public chat surface.
        raise _ChatPodUnavailable(f"{type(e).__name__}: {e}") from e
    if resp.status_code >= 500:
        # vLLM crashed or the tunnel is half-open; surface as unavailable
        # so chat-keeper picks it up on the next tick.
        raise _ChatPodUnavailable(f"vLLM returned {resp.status_code}")
    return resp.json()


async def _local_models_probe(timeout: float = 2.5) -> str | None:
    """Cheap async probe: returns the served model id, or None on failure."""
    client = _get_http_client()
    try:
        resp = await client.get(
            "/v1/models",
            timeout=httpx.Timeout(connect=1.5, read=timeout, write=2.0, pool=2.0),
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        for m in (data.get("data") or []):
            mid = m.get("id")
            if mid:
                return mid
    except Exception:
        return None
    return None


# ── Streaming response helpers ───────────────────────────────────────────────

_SSE_RESPONSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    # Disable Cloudflare/Caddy buffering so SSE deltas reach the
    # client without a multi-second batched flush at the proxy.
    "X-Accel-Buffering": "no",
}


def _sse_response(generator) -> StreamingResponse:
    """Wrap an SSE async generator with the standard headers used by both
    the dashboard chat proxy and the OpenAI-compatible passthrough."""
    return StreamingResponse(
        generator, media_type="text/event-stream",
        headers=_SSE_RESPONSE_HEADERS,
    )


def _extract_message_content(message: dict) -> tuple[str, str | None]:
    """Pull (content, thinking) from a vLLM choices[0].message."""
    content = message.get("content") or ""
    thinking = message.get("reasoning") or message.get("thinking")
    if not content and thinking:
        content = thinking
        thinking = None
    return content, thinking


# ── Sync / streaming chat dispatch ───────────────────────────────────────────

async def _sync_chat(payload, king_uid, king_model):
    """Run a non-streaming chat. Routes through the agent harness by
    default; falls through to a bare vLLM proxy when
    DISTIL_CHAT_AGENT_HARNESS=0 (emergency rollback)."""
    payload["stream"] = False
    if _AGENT_HARNESS_ENABLED:
        from agent_runner import run_agent_chat  # lazy import keeps tests cheap
        try:
            data = await run_agent_chat(payload, king_uid, king_model)
        except Exception as exc:  # noqa: BLE001 — never 500 the public surface
            return JSONResponse(
                status_code=503,
                content={
                    "error": "chat agent failed",
                    "detail": f"{type(exc).__name__}: {str(exc)[:200]}",
                    "king_uid": king_uid,
                    "king_model": king_model,
                },
            )
        message = data["choices"][0]["message"]
        resp = {
            "response": message.get("content") or "",
            "model": king_model,
            "king_uid": king_uid,
            "thinking": message.get("reasoning") or "",
            "usage": data.get("usage"),
        }
        _log_chat_turn(payload, resp["response"], king_uid, king_model, data)
        return resp

    # Emergency rollback path: raw vLLM proxy, no tool calling.
    try:
        data = await _local_chat_post(payload, timeout=90.0)
    except _ChatPodUnavailable as e:
        return JSONResponse(
            status_code=503,
            content={
                "error": "chat server unavailable",
                "detail": str(e)[:200],
                "king_uid": king_uid,
                "king_model": king_model,
            },
        )
    if "choices" in data:
        message = data["choices"][0].get("message") or {}
        content, thinking = _extract_message_content(message)
        resp = {
            "response": content,
            "model": king_model,
            "king_uid": king_uid,
        }
        if thinking:
            resp["thinking"] = thinking
        if "usage" in data:
            resp["usage"] = data["usage"]
        _log_chat_turn(
            _normalize_chat_payload(payload),
            content, king_uid, king_model, data,
        )
        return resp
    return {"error": "unexpected response from chat server"}


def _stream_chat(payload, king_uid, king_model):
    """Run a streaming chat. Routes through the agent harness by default;
    falls through to a bare vLLM SSE proxy when the harness is disabled."""
    payload["stream"] = True
    if _AGENT_HARNESS_ENABLED:
        from agent_runner import stream_agent_chat_legacy
        gen = stream_agent_chat_legacy(payload, king_uid, king_model)

        async def wrap():
            acc = ""
            try:
                async for chunk in gen():
                    yield chunk
                    if isinstance(chunk, str) and chunk.startswith("data: "):
                        body_str = chunk[6:].strip()
                        if body_str and body_str != "[DONE]":
                            try:
                                obj = json.loads(body_str)
                                if isinstance(obj, dict) and obj.get("response"):
                                    acc += str(obj["response"])
                            except Exception:
                                pass
            finally:
                try:
                    _log_chat_turn(payload, acc, king_uid, king_model, None)
                except Exception:
                    pass

        return _sse_response(wrap())

    # Emergency rollback: raw vLLM SSE forwarding.
    norm = _normalize_chat_payload(payload)

    async def generate():
        acc = ""
        client = _get_http_client()
        try:
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json=norm,
                # 5 min read cap against runaway generations.
                timeout=httpx.Timeout(connect=3.0, read=300.0, write=10.0, pool=5.0),
            ) as resp:
                if resp.status_code >= 500:
                    yield (
                        f"data: {json.dumps({'error': f'chat server returned {resp.status_code}'})}\n\n"
                    )
                    return
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    line = line.strip()
                    if not line.startswith("data: "):
                        continue
                    raw = line[6:]
                    if raw == "[DONE]":
                        yield "data: [DONE]\n\n"
                        break
                    try:
                        parsed = json.loads(raw)
                    except json.JSONDecodeError:
                        yield f"data: {raw}\n\n"
                        continue
                    choices = parsed.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        msg = choices[0].get("message") or {}
                        delta_content = (
                            delta.get("content")
                            or msg.get("content")
                            or ""
                        )
                        if delta_content:
                            acc += delta_content
                    parsed["king_uid"] = king_uid
                    parsed["king_model"] = king_model
                    yield f"data: {json.dumps(parsed)}\n\n"
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            yield (
                f"data: {json.dumps({'error': 'chat server unavailable', 'detail': str(e)[:200]})}\n\n"
            )
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)[:200]})}\n\n"
        finally:
            try:
                _log_chat_turn(norm, acc, king_uid, king_model, None)
            except Exception:
                pass

    return _sse_response(generate())


# ── Per-turn audit log (JSONL): metadata + 200-char preview only ─────────────

_CHAT_LOG_PATH = os.path.join(STATE_DIR, "chat_turns.jsonl")
_chat_log_lock = threading.Lock()
_CHAT_LOG_MAX_BYTES = 50 * 1024 * 1024  # 50MB rotation


def _detect_repeated_substring(text: str, win: int = 50, step: int = 25) -> int:
    """Count repeated ``win``-char windows (stride ``step``) in ``text``."""
    seen = set()
    repeats = 0
    if not text or len(text) < win * 2:
        return 0
    for i in range(0, len(text) - win, step):
        s = text[i:i + win]
        if s in seen:
            repeats += 1
        seen.add(s)
    return repeats


def _log_chat_turn(payload, response_text, king_uid, king_model, raw_data=None):
    """Append a one-line JSON record of a completed chat turn.

    Best-effort and non-blocking on errors — never let logging take down a
    user-facing request.
    """
    try:
        prompt_text = ""
        for msg in (payload.get("messages") or []):
            if isinstance(msg, dict):
                c = msg.get("content")
                if isinstance(c, str):
                    prompt_text += c + "\n"
        response_text = response_text or ""
        non_ascii = sum(1 for c in response_text if ord(c) > 127)
        non_ascii_frac = non_ascii / len(response_text) if response_text else 0.0
        repeats = _detect_repeated_substring(response_text)
        usage = (raw_data or {}).get("usage") or {}
        rec = {
            "ts": time.time(),
            "king_uid": king_uid,
            "king_model": king_model,
            "prompt_chars": len(prompt_text),
            "response_chars": len(response_text),
            "non_ascii_frac": round(non_ascii_frac, 4),
            "repeats_50char": repeats,
            "completion_tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "temperature": payload.get("temperature"),
            "top_p": payload.get("top_p"),
            "repetition_penalty": payload.get("repetition_penalty"),
            "frequency_penalty": payload.get("frequency_penalty"),
            "max_tokens": payload.get("max_tokens"),
            "prompt_preview": prompt_text[-400:],
            "response_head": response_text[:200],
            "response_tail": response_text[-200:],
        }
        with _chat_log_lock:
            try:
                if (
                    os.path.exists(_CHAT_LOG_PATH)
                    and os.path.getsize(_CHAT_LOG_PATH) > _CHAT_LOG_MAX_BYTES
                ):
                    bak = _CHAT_LOG_PATH + ".1"
                    if os.path.exists(bak):
                        os.remove(bak)
                    os.rename(_CHAT_LOG_PATH, bak)
            except OSError:
                pass
            try:
                with open(_CHAT_LOG_PATH, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            except OSError:
                pass
    except Exception:
        # Logging is strictly best-effort.
        pass


# /api/chat/status is polled every 30s by every dashboard tab; cache
# the live vLLM probe for 10s.
_status_cache: dict | None = None
_status_cache_ts: float = 0.0
_STATUS_CACHE_TTL = 10.0
_status_cache_lock = threading.Lock()


def _cached_status_lookup() -> dict | None:
    now = time.time()
    with _status_cache_lock:
        if _status_cache is not None and now - _status_cache_ts < _STATUS_CACHE_TTL:
            return _status_cache
    return None


def _store_status_cache(snapshot: dict) -> None:
    global _status_cache, _status_cache_ts
    with _status_cache_lock:
        _status_cache = snapshot
        _status_cache_ts = time.time()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/api/chat",
    tags=["Chat"],
    summary="Chat with the king model (dashboard surface)",
    description="Dashboard chat surface; supports stream=true/false. 10/min per IP. 503 when chat pod unavailable.",
)
async def chat_with_king(request: Request):
    """Proxy chat to the king model running on the GPU pod."""
    client_ip, is_internal = client_real_ip(request)
    if not is_internal and not _chat_rate_limiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=429,
            content={"error": "rate limit exceeded"},
            headers={"Retry-After": "30"},
        )

    body = await request.json()
    messages = body.get("messages", [])
    max_tokens = body.get("max_tokens", 16384)
    try:
        max_tokens = min(int(max_tokens), 30720)
    except (TypeError, ValueError):
        max_tokens = 16384
    stream = body.get("stream", False)

    if not messages:
        return {"error": "messages required"}
    if not isinstance(messages, list) or len(messages) > 50:
        return JSONResponse(
            status_code=400,
            content={"error": "messages must be an array with at most 50 entries"},
        )
    for msg in messages:
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        if isinstance(content, str) and len(content) > 10000:
            return JSONResponse(
                status_code=400,
                content={"error": "message content too long (max 10000 chars)"},
            )
    if not isinstance(max_tokens, (int, float)) or max_tokens < 1:
        max_tokens = 16384
    messages, max_tokens = _clamp_for_context_budget(messages, max_tokens)
    temperature = body.get("temperature", 0.7)
    if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2:
        temperature = 0.7
    top_p = body.get("top_p", 0.9)
    if not isinstance(top_p, (int, float)) or top_p < 0 or top_p > 1:
        top_p = 0.9

    king_uid, king_model = _get_king_info()
    if king_uid is None:
        return {"error": "no king model available"}

    body_rep = body.get("repetition_penalty")
    body_freq = body.get("frequency_penalty")
    body_pres = body.get("presence_penalty")
    pod_payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": stream,
    }
    if isinstance(body_rep, (int, float)) and 1.0 <= body_rep <= 2.0:
        pod_payload["repetition_penalty"] = body_rep
    if isinstance(body_freq, (int, float)) and -2.0 <= body_freq <= 2.0:
        pod_payload["frequency_penalty"] = body_freq
    if isinstance(body_pres, (int, float)) and -2.0 <= body_pres <= 2.0:
        pod_payload["presence_penalty"] = body_pres

    if stream:
        return _stream_chat(pod_payload, king_uid, king_model)
    try:
        return await _sync_chat(pod_payload, king_uid, king_model)
    except _ChatPodUnavailable as e:
        return JSONResponse(
            status_code=503,
            content={"error": "chat server unavailable", "detail": str(e)[:200]},
        )
    except Exception as e:  # pragma: no cover
        # Top-level safety net: never 500 the public chat surface.
        return JSONResponse(
            status_code=503,
            content={
                "error": "chat orchestration failed",
                "detail": f"{type(e).__name__}: {str(e)[:200]}",
            },
        )


@router.get(
    "/api/chat/status",
    tags=["Chat"],
    summary="King chat server availability + quality snapshot",
    description="Live king UID/model + eval-holding-GPU flag + latest judge axes. Cached 10s.",
)
async def chat_status():
    """Check if the king chat server is available (cached 10s)."""
    cached = _cached_status_lookup()
    if cached is not None:
        return cached

    king_uid, king_model = _get_king_info()
    progress = _safe_json_load(os.path.join(STATE_DIR, "eval_progress.json"), {})
    eval_active = progress.get("active", False)

    server_ok = False
    served_model: str | None = None
    if CHAT_POD_HOST:
        served_model = await _local_models_probe(timeout=2.5)
        if served_model:
            # vLLM always serves under "sn97-king"; any probe success = healthy.
            server_ok = True

    quality = {
        "long_form_judge": None,
        "long_gen_coherence": None,
        "judge_probe": None,
        "composite_final": None,
    }
    if king_uid is not None and king_uid >= 0:
        try:
            h2h = h2h_latest()
            for r in (h2h.get("results") or []):
                if r.get("uid") == king_uid:
                    comp = r.get("composite") or {}
                    axes = comp.get("axes") or {}
                    quality["long_form_judge"] = axes.get("long_form_judge")
                    quality["long_gen_coherence"] = axes.get("long_gen_coherence")
                    quality["judge_probe"] = axes.get("judge_probe")
                    quality["composite_final"] = comp.get("final")
                    break
        except Exception:
            pass

    snapshot = {
        "available": server_ok and king_uid is not None,
        "king_uid": king_uid,
        "king_model": king_model,
        "served_model": served_model,
        "eval_active": eval_active,
        "server_running": server_ok,
        "quality": quality,
        "note": (
            "King model is loaded on GPU and ready for chat."
            if server_ok
            else (
                "Chat pod is not configured."
                if not CHAT_POD_HOST
                else (
                    "Chat paused while the eval pipeline holds the GPU."
                    if eval_active
                    else "Chat server is starting or unavailable."
                )
            )
        ),
    }
    _store_status_cache(snapshot)
    return snapshot


# ── OpenAI-compatible endpoints (for Open WebUI etc.) ─────────────────────────

@router.get(
    "/v1/models",
    tags=["Chat"],
    summary="OpenAI-compatible model list",
    description="Returns `sn97-king` plus the live king's HF repo id; both resolve to the same vLLM.",
)
def openai_models():
    """OpenAI-compatible models list. Returns the current king model."""
    king_uid, king_model = _get_king_info()
    model_id = king_model or "distil-king"
    models = [{
        "id": CHAT_POD_SERVED_MODEL,
        "object": "model",
        "created": int(time.time()),
        "owned_by": f"distil-sn97-uid{king_uid}" if king_uid else "distil-sn97",
    }]
    if model_id != CHAT_POD_SERVED_MODEL:
        models.append({
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": f"distil-sn97-uid{king_uid}" if king_uid else "distil-sn97",
        })
    return {
        "object": "list",
        "data": models,
    }


@router.post(
    "/v1/chat/completions",
    tags=["Chat"],
    summary="OpenAI-compatible chat completions",
    description="Drop-in OpenAI completions for the SN97 king. 240/min per IP. 503 when chat pod unavailable.",
)
async def openai_chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint. Proxies to the king model."""
    client_ip, is_internal = client_real_ip(request)
    if not is_internal and not _openai_api_rate_limiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": "30"},
            content={"error": {"message": "rate limit exceeded", "type": "rate_limit_error"}},
        )

    body = await request.json()
    messages = body.get("messages", [])
    if not messages:
        return JSONResponse(status_code=400, content={"error": {"message": "messages required"}})

    king_uid, king_model = _get_king_info()
    if king_uid is None:
        return JSONResponse(status_code=503, content={"error": {"message": "no king model available"}})

    requested_max = body.get("max_tokens", 16384)
    trimmed_msgs, clamped_max = _clamp_for_context_budget(messages, requested_max)
    body["messages"] = trimmed_msgs
    body["max_tokens"] = clamped_max
    messages = trimmed_msgs

    stream = body.get("stream", False)
    try:
        if _AGENT_HARNESS_ENABLED:
            from agent_runner import (
                run_agent_chat as _agent_run,
                stream_agent_chat_openai as _agent_stream,
            )
            if stream:
                gen = _agent_stream(body, king_uid, king_model)

                async def wrap():
                    acc = ""
                    try:
                        async for chunk in gen():
                            yield chunk
                            if isinstance(chunk, str) and chunk.startswith("data: "):
                                body_str = chunk[6:].strip()
                                if body_str and body_str != "[DONE]":
                                    try:
                                        obj = json.loads(body_str)
                                        choices = obj.get("choices") or []
                                        if choices:
                                            d = choices[0].get("delta") or {}
                                            piece = d.get("content")
                                            if piece:
                                                acc += str(piece)
                                    except Exception:
                                        pass
                    finally:
                        try:
                            _log_chat_turn(body, acc, king_uid, king_model, None)
                        except Exception:
                            pass

                return _sse_response(wrap())
            data = await _agent_run(body, king_uid, king_model)
            try:
                msg_content = (
                    (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
                )
                _log_chat_turn(body, msg_content, king_uid, king_model, data)
            except Exception:
                pass
            return JSONResponse(content=data)

        # Emergency rollback path: raw vLLM proxy.
        if stream:
            norm = _normalize_chat_payload(body)

            async def generate():
                client = _get_http_client()
                try:
                    async with client.stream(
                        "POST",
                        "/v1/chat/completions",
                        json=norm,
                        timeout=httpx.Timeout(connect=3.0, read=300.0, write=10.0, pool=5.0),
                    ) as resp:
                        if resp.status_code >= 500:
                            yield (
                                f"data: {json.dumps({'error': {'message': f'chat server returned {resp.status_code}'}})}\n\n"
                            )
                            return
                        async for line in resp.aiter_lines():
                            if not line:
                                continue
                            line = line.strip()
                            if line.startswith("data: "):
                                yield f"{line}\n\n"
                                if line == "data: [DONE]":
                                    break
                except (
                    httpx.ConnectError,
                    httpx.ConnectTimeout,
                    httpx.PoolTimeout,
                    httpx.RemoteProtocolError,
                ):
                    yield 'data: {"error": {"message": "chat server unavailable"}}\n\n'
                except Exception:
                    yield 'data: {"error": {"message": "stream interrupted"}}\n\n'

            return _sse_response(generate())

        try:
            data = await _local_chat_post(body, timeout=120.0)
        except _ChatPodUnavailable as e:
            return JSONResponse(
                status_code=503,
                content={"error": {"message": "chat server unavailable", "detail": str(e)[:200]}},
            )
    except _ChatPodUnavailable as e:
        return JSONResponse(
            status_code=503,
            content={"error": {"message": "chat server unavailable", "detail": str(e)[:200]}},
        )
    except Exception as e:  # pragma: no cover
        # Top-level safety net: never 500 the public OpenAI surface.
        return JSONResponse(
            status_code=503,
            content={"error": {
                "message": "chat orchestration failed",
                "detail": f"{type(e).__name__}: {str(e)[:200]}",
            }},
        )
    if isinstance(data, dict) and king_model:
        # Stamp with the live king's HF repo id so OpenAI clients show
        # the correct lineage despite the vLLM-side "sn97-king" alias.
        data["model"] = king_model
        data["king_uid"] = king_uid
    return JSONResponse(content=data)
