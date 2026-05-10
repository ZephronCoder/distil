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


def test_clean_client_messages_strips_displayed_thinking():
    cleaned = chat_route._clean_client_messages(
        [
            {"role": "user", "content": "first"},
            {
                "role": "assistant",
                "content": (
                    "<think>Runtime trace, not hidden model reasoning: inspected "
                    "the latest user request.</think>\nActual answer."
                ),
            },
        ],
        system="system prompt",
    )

    assert cleaned[-1] == {"role": "assistant", "content": "Actual answer."}


def test_orchestrated_chat_returns_only_model_reasoning(monkeypatch):
    async def fake_local_chat_post(_payload, *, timeout):
        return {
            "choices": [
                {
                    "message": {
                        "content": "final answer",
                        "reasoning_content": "model-supplied thinking",
                    }
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }

    monkeypatch.setattr(chat_route, "_local_chat_post", fake_local_chat_post)

    data = asyncio.run(
        chat_route._orchestrated_chat_completion(
            {"messages": [{"role": "user", "content": "hello"}]},
            king_uid=1,
            king_model="king/model",
        )
    )

    msg = data["choices"][0]["message"]
    assert msg["content"] == "final answer"
    assert msg["reasoning"] == "model-supplied thinking"
    assert "Runtime trace" not in msg["reasoning"]


def test_orchestrated_chat_omits_fake_reasoning_when_model_has_none(monkeypatch):
    async def fake_local_chat_post(_payload, *, timeout):
        return {"choices": [{"message": {"content": "final answer"}}]}

    monkeypatch.setattr(chat_route, "_local_chat_post", fake_local_chat_post)

    data = asyncio.run(
        chat_route._orchestrated_chat_completion(
            {"messages": [{"role": "user", "content": "hello"}]},
            king_uid=1,
            king_model="king/model",
        )
    )

    msg = data["choices"][0]["message"]
    assert msg["content"] == "final answer"
    assert "reasoning" not in msg
    assert "reasoning_content" not in msg


def test_orchestrated_chat_passes_tool_results_to_model(monkeypatch):
    captured = {}

    async def fake_local_chat_post(payload, *, timeout):
        captured["payload"] = payload
        return {
            "choices": [
                {
                    "message": {
                        "content": "The 32nd Fibonacci number is 2178309.",
                    }
                }
            ]
        }

    monkeypatch.setattr(chat_route, "_local_chat_post", fake_local_chat_post)

    data = asyncio.run(
        chat_route._orchestrated_chat_completion(
            {"messages": [{"role": "user", "content": "what is the 2**5th fibonacci number"}]},
            king_uid=1,
            king_model="king/model",
        )
    )

    msg = data["choices"][0]["message"]
    assert "2178309" in msg["content"]
    assert msg["content"] == "The 32nd Fibonacci number is 2178309."
    payload_msgs = captured["payload"]["messages"]
    sys_msgs = [m for m in payload_msgs if m["role"] == "system"]
    assert any("PYTHON_EXECUTION_RESULT" in m["content"] for m in sys_msgs), (
        "tool results must be supplied to the model as authoritative context"
    )
    assert any("web search" in m["content"].lower() for m in sys_msgs), (
        "system prompt must advertise the runtime tool capabilities"
    )
    assert any("internet access" in m["content"].lower() for m in sys_msgs)
    assert "Tool trace:" in msg["reasoning"]


def test_orchestrated_chat_falls_back_to_direct_answer_when_model_empty(monkeypatch):
    async def empty_model(_payload, *, timeout):
        return {"choices": [{"message": {"content": "   "}}]}

    monkeypatch.setattr(chat_route, "_local_chat_post", empty_model)

    data = asyncio.run(
        chat_route._orchestrated_chat_completion(
            {"messages": [{"role": "user", "content": "run python: print(4**8)"}]},
            king_uid=1,
            king_model="king/model",
        )
    )

    msg = data["choices"][0]["message"]
    assert "65536" in msg["content"]
    assert "Tool trace:" in msg["reasoning"]
    assert "deterministic" in msg["reasoning"]


def test_orchestrated_chat_falls_back_when_model_says_no_internet(monkeypatch):
    async def derailed_model(_payload, *, timeout):
        return {
            "choices": [
                {
                    "message": {
                        "content": "I'm sorry, I cannot do that.",
                    }
                }
            ]
        }

    monkeypatch.setattr(chat_route, "_local_chat_post", derailed_model)

    data = asyncio.run(
        chat_route._orchestrated_chat_completion(
            {"messages": [{"role": "user", "content": "run python: print(4**8)"}]},
            king_uid=1,
            king_model="king/model",
        )
    )

    msg = data["choices"][0]["message"]
    assert "65536" in msg["content"]
    assert "deterministic" in msg["reasoning"]


def test_orchestrated_chat_falls_back_when_model_hallucinates_fibonacci(monkeypatch):
    async def hallucinating_model(_payload, *, timeout):
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            "The 4^8th Fibonacci number is 1,024,000,000,000,000,000,000."
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(chat_route, "_local_chat_post", hallucinating_model)

    data = asyncio.run(
        chat_route._orchestrated_chat_completion(
            {"messages": [{"role": "user", "content": "what is the 2**5th fibonacci number"}]},
            king_uid=1,
            king_model="king/model",
        )
    )

    msg = data["choices"][0]["message"]
    assert "2178309" in msg["content"]
    assert "did not quote" in msg["reasoning"]


def test_orchestrated_chat_executes_contextual_fibonacci_followup(monkeypatch):
    captured = {}

    async def hallucinating_model(payload, *, timeout):
        captured["payload"] = payload
        return {
            "choices": [
                {
                    "message": {
                        "content": "[boxed{832}]",
                    }
                }
            ]
        }

    monkeypatch.setattr(chat_route, "_local_chat_post", hallucinating_model)

    data = asyncio.run(
        chat_route._orchestrated_chat_completion(
            {
                "messages": [
                    {"role": "user", "content": "no 4^6th number"},
                    {
                        "role": "assistant",
                        "content": (
                            "I'll interpret that as the (4^6)th number in the "
                            "Fibonacci sequence. 4^6 = 4096."
                        ),
                    },
                    {"role": "user", "content": "yes the 4096th number"},
                ]
            },
            king_uid=1,
            king_model="king/model",
        )
    )

    msg = data["choices"][0]["message"]
    assert "index 4096" in msg["content"]
    assert "It has 856 digits" in msg["content"]
    assert "[boxed{832}]" not in msg["content"]
    assert "did not quote" in msg["reasoning"]
    sys_msgs = [m for m in captured["payload"]["messages"] if m["role"] == "system"]
    assert any("PYTHON_EXECUTION_RESULT" in m["content"] for m in sys_msgs)


def test_orchestrated_chat_executes_plain_arithmetic_without_keyword(monkeypatch):
    async def empty_model(_payload, *, timeout):
        return {"choices": [{"message": {"content": ""}}]}

    monkeypatch.setattr(chat_route, "_local_chat_post", empty_model)

    data = asyncio.run(
        chat_route._orchestrated_chat_completion(
            {"messages": [{"role": "user", "content": "what is 4^6?"}]},
            king_uid=1,
            king_model="king/model",
        )
    )

    msg = data["choices"][0]["message"]
    assert "4096" in msg["content"]
    assert "Executed a arithmetic expression" in msg["reasoning"]


def test_model_quoted_numeric_answer_helper():
    assert chat_route._model_quoted_numeric_answer("F_32 is 2178309.", "Result: 2178309")
    assert chat_route._model_quoted_numeric_answer(
        "Result: 2178309",
        "The Fibonacci index 32 result is 2,178,309.",
    )
    assert not chat_route._model_quoted_numeric_answer(
        "Python stdout:\n285",
        "The sum is 385. The runtime printed 285.",
    )
    assert not chat_route._model_quoted_numeric_answer(
        "Result: 2178309",
        "The number is 1,024,000,000,000.",
    )
    big_value = "26" + "0" * 13700 + "57"
    assert chat_route._model_quoted_numeric_answer(
        f"Result: {big_value}",
        f"Approximately {big_value[:8]}…{big_value[-8:]}",
    )
    assert not chat_route._model_quoted_numeric_answer(
        f"Result: {big_value}",
        "Approximately 102400000000000000000.",
    )


def test_orchestrated_chat_promotes_inline_think_blocks(monkeypatch):
    async def think_model(_payload, *, timeout):
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            "<think>Let me compute step-by-step. 4 ** 8 = 65536.</think>"
                            "The answer is 65536."
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(chat_route, "_local_chat_post", think_model)

    data = asyncio.run(
        chat_route._orchestrated_chat_completion(
            {"messages": [{"role": "user", "content": "what is 4**8?"}]},
            king_uid=1,
            king_model="king/model",
        )
    )

    msg = data["choices"][0]["message"]
    assert msg["content"] == "The answer is 65536."
    assert "Let me compute step-by-step" in msg["reasoning"]
    assert "<think>" not in msg["content"]


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
            raise AssertionError(f"_local_chat_post did not convert {exc!r} to _ChatPodUnavailable")


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
        raise AssertionError("_local_chat_post did not surface 503 as _ChatPodUnavailable")


def test_web_search_regex_triggers_on_time_sensitive_questions():
    cases = [
        "what's the bitcoin price right now?",
        "current bitcoin price please",
        "today's headlines on AI",
        "latest news about openai",
        "current weather in tokyo",
        "btc price",
        "what's happening today in the markets",
        "what are today's top tech headlines?",
        "find me a list of recent space launches",
        "what's the price of ethereum right now?",
    ]
    for case in cases:
        assert chat_route._WEB_SEARCH_RE.search(case), case


def test_web_search_regex_skips_general_questions():
    not_search = [
        "tell me about bittensor subnet 97",
        "explain transformers in simple terms",
        "write me a haiku about clouds",
    ]
    for case in not_search:
        assert not chat_route._WEB_SEARCH_RE.search(case), case


def test_parse_duckduckgo_html_extracts_titles_and_snippets():
    body = """
    <div class="result"><h2 class="result__title">
      <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fa.example%2Fbtc">
        Bitcoin price today
      </a>
    </h2>
      <a class="result__snippet" href="...">Bitcoin is currently $81,491.41 USD.</a>
    </div>
    <div class="result"><h2 class="result__title">
      <a class="result__a" href="https://b.example/news">Tech news</a>
    </h2>
    <div class="result__snippet">Top tech stories today.</div>
    </div>
    """
    rendered = chat_route._parse_duckduckgo_html(body, query="btc", limit=5)
    assert "Bitcoin price today" in rendered
    assert "https://a.example/btc" in rendered
    assert "$81,491.41" in rendered
    assert "Tech news" in rendered
    assert "Top tech stories today" in rendered


def test_think_stream_splitter_handles_split_chunks():
    splitter = chat_route._ThinkStreamSplitter()
    visibles, reasonings = [], []
    for chunk in ["Hello <thi", "nk>step one ", "step two</think>final ", "answer"]:
        v, r = splitter.feed(chunk)
        visibles.append(v)
        reasonings.append(r)
    v_tail, r_tail = splitter.flush()
    visible_total = "".join(visibles) + v_tail
    reasoning_total = "".join(reasonings) + r_tail
    assert visible_total.strip() == "Hello final answer"
    assert "step one" in reasoning_total
    assert "step two" in reasoning_total
    # The splitter must not leak the literal tags into either stream.
    assert "<think" not in visible_total
    assert "</think>" not in reasoning_total
    assert "<think" not in reasoning_total
