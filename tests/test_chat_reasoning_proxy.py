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


def test_orchestrated_chat_answers_fibonacci_from_python_tool(monkeypatch):
    async def fail_if_model_called(_payload, *, timeout):
        raise AssertionError("deterministic Fibonacci tool should not call the model")

    monkeypatch.setattr(chat_route, "_local_chat_post", fail_if_model_called)

    data = asyncio.run(
        chat_route._orchestrated_chat_completion(
            {"messages": [{"role": "user", "content": "what is the 2**5th fibonacci number"}]},
            king_uid=1,
            king_model="king/model",
        )
    )

    msg = data["choices"][0]["message"]
    assert "2178309" in msg["content"]
    assert "index 32" in msg["content"]
    assert "Tool trace:" in msg["reasoning"]
    assert "Executed Python" in msg["reasoning"]


def test_orchestrated_chat_answers_inline_python_from_tool(monkeypatch):
    async def fail_if_model_called(_payload, *, timeout):
        raise AssertionError("deterministic Python tool should not call the model")

    monkeypatch.setattr(chat_route, "_local_chat_post", fail_if_model_called)

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
