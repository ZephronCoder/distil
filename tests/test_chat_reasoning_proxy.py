"""Tests for the thin transport layer in ``api/routes/chat.py``.

The orchestration / tool-calling / Python-execution stack now lives in
``api.agent_runner`` and is exercised by ``tests/test_agent_runner.py``.
This file only verifies the bits ``chat.py`` is still responsible for:
the per-turn audit-log helper, the chat-pod transport error mapping,
and the request-payload normalizer. Anything more is a misplaced test."""

import asyncio
import os
import sys


ROOT = os.path.dirname(os.path.dirname(__file__))
API = os.path.join(ROOT, "api")
ROUTES = os.path.join(API, "routes")
for path in (ROUTES, API, ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import chat as chat_route  # noqa: E402


# ── Transport-error mapping ──────────────────────────────────────────────────

def test_local_chat_post_maps_pool_timeout_to_unavailable(monkeypatch):
    """Regression for the 2026-05-09 Internal Server Error bug.

    When the chat tunnel is up but the upstream vLLM is dead, dozens of
    concurrent requests pile up on the 64-slot connection pool and the
    65th hits ``httpx.PoolTimeout`` after 5 s. Pre-fix that leaked as
    a bare 500. The contract is: any transport-level failure (incl.
    PoolTimeout, RemoteProtocolError, WriteTimeout) must be mapped to
    :class:`_ChatPodUnavailable` so callers can return the documented
    503.
    """
    import httpx

    class _FakeClient:
        def __init__(self, exc):
            self._exc = exc

        async def post(self, *args, **kwargs):
            raise self._exc

    cases = [
        httpx.PoolTimeout("pool full"),
        httpx.RemoteProtocolError("upstream RST"),
        httpx.WriteTimeout("write stalled"),
        httpx.ConnectError("refused"),
    ]
    for exc in cases:
        monkeypatch.setattr(chat_route, "_get_http_client", lambda exc=exc: _FakeClient(exc))
        try:
            asyncio.run(chat_route._local_chat_post({"messages": []}, timeout=1.0))
        except chat_route._ChatPodUnavailable as e:
            assert exc.__class__.__name__ in str(e)
        else:
            raise AssertionError(
                f"_local_chat_post did not convert {exc!r} to _ChatPodUnavailable"
            )


def test_local_chat_post_maps_5xx_response_to_unavailable(monkeypatch):
    """vLLM 5xx (or a half-open tunnel returning 502) must surface as
    503, not propagate as a 500. ``chat-keeper`` then notices the pod
    is down on the next tick and triggers a recovery."""
    class _Resp:
        status_code = 503

        def json(self):  # pragma: no cover — should not be called
            return {}

    class _FakeClient:
        async def post(self, *args, **kwargs):
            return _Resp()

    monkeypatch.setattr(chat_route, "_get_http_client", lambda: _FakeClient())
    try:
        asyncio.run(chat_route._local_chat_post({"messages": []}, timeout=1.0))
    except chat_route._ChatPodUnavailable as e:
        assert "503" in str(e)
    else:
        raise AssertionError(
            "_local_chat_post did not surface 503 as _ChatPodUnavailable"
        )


# ── Payload normalizer ───────────────────────────────────────────────────────

def test_normalize_chat_payload_pins_served_model_and_enables_thinking():
    """Thinking must default ON so the chat reflects the king's actual
    reasoning capability (matches the held-out benchmark and bench
    axes). ``max_tokens`` defaults to 16 384 (16 K) — the v32.1
    reasoning-uncap default — so the king's <think>...</think> chain
    has full room to reason without truncation. The chat pod's vLLM
    runs with max_model_len=32 768, so 16 K leaves ample headroom for
    the prompt + thinking + answer. The served-model is pinned and a
    formatting system prompt is injected if the client didn't supply
    one."""
    payload = {
        "model": "client-supplied-model",
        "messages": [{"role": "user", "content": "hi"}],
    }
    out = chat_route._normalize_chat_payload(payload)
    assert out["model"] == chat_route.CHAT_POD_SERVED_MODEL
    assert out["chat_template_kwargs"]["enable_thinking"] is True
    assert out["max_tokens"] == 16384
    sys_msgs = [m for m in out["messages"] if m["role"] == "system"]
    assert sys_msgs, "missing system formatting prompt was not injected"


def test_normalize_chat_payload_honours_explicit_thinking_off():
    """A client that explicitly opts out of thinking (low-latency
    use cases) must still get the no-think rendering."""
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    out = chat_route._normalize_chat_payload(payload)
    assert out["chat_template_kwargs"]["enable_thinking"] is False


def test_normalize_chat_payload_preserves_existing_system_prompt():
    """If the client supplies a system prompt we must not stomp on it."""
    payload = {
        "messages": [
            {"role": "system", "content": "BE TERSE."},
            {"role": "user", "content": "hi"},
        ],
        "max_tokens": 256,
    }
    out = chat_route._normalize_chat_payload(payload)
    sys_msgs = [m for m in out["messages"] if m["role"] == "system"]
    assert len(sys_msgs) == 1
    assert sys_msgs[0]["content"] == "BE TERSE."
    assert out["max_tokens"] == 256


# ── Audit log: repeated-substring detector ───────────────────────────────────

def test_detect_repeated_substring_finds_duplicates():
    repeats = chat_route._detect_repeated_substring("abc" * 200)
    assert repeats > 0


def test_detect_repeated_substring_handles_short_input():
    assert chat_route._detect_repeated_substring("") == 0
    assert chat_route._detect_repeated_substring("short") == 0


def test_extract_message_content_promotes_thinking_when_content_empty():
    """Some vLLM builds emit an empty `content` and stuff the answer
    into `reasoning` instead. The transport helper must surface that
    as the visible content so the user is not left with nothing."""
    content, thinking = chat_route._extract_message_content(
        {"content": "", "reasoning": "the answer is 42"}
    )
    assert content == "the answer is 42"
    assert thinking is None


def test_extract_message_content_keeps_thinking_when_content_present():
    content, thinking = chat_route._extract_message_content(
        {"content": "the answer is 42", "reasoning": "step-by-step trace"}
    )
    assert content == "the answer is 42"
    assert thinking == "step-by-step trace"
