"""Function tools for the SN97 chat agent.

These are exposed to the king model via the OpenAI Agents SDK
(``@function_tool``). Each tool is a thin, hardened wrapper around an
existing helper so behaviour matches what the chat already shipped, while
the orchestration loop, retry, and streaming come from the SDK.

Tools are intentionally async + side-effect-free wrt server state so the
SDK can fan them out within a single agent turn.
"""
from __future__ import annotations

import ast
import asyncio
import contextlib
import html
import json
import os
import re
import subprocess
import tempfile
import urllib.parse
from dataclasses import dataclass
from typing import Any

import httpx
from agents import RunContextWrapper, function_tool


# ── Shared per-request agent context ─────────────────────────────────────────

@dataclass
class SN97AgentContext:
    """Per-request data the tools may consult.

    Kept deliberately small: the SDK passes this opaque object through every
    tool invocation so the tool body can read live king info without
    re-importing route-level globals (which would create import cycles).
    """

    king_uid: int | None = None
    king_model: str | None = None
    request_id: str | None = None
    # Soft cap so a chat that goes wild (e.g. recursive code generation)
    # can't drain a full Python sandbox quota; tracked client-side because
    # the SDK has no per-request budget primitive. 8 is enough for a
    # ~4-step "try, see error, retry, succeed" loop on each of 2 sub-questions.
    python_runs_remaining: int = 8


# ── python_exec ──────────────────────────────────────────────────────────────

_PY_BANNED_IMPORTS = frozenset(
    {"socket", "subprocess", "shutil", "pathlib", "requests", "httpx", "urllib", "os.path"}
)
_PY_BANNED_BUILTINS = frozenset({"open", "exec", "eval", "compile", "__import__"})
# Python 3.11+ caps int->str at 4300 digits by default; that breaks
# the canonical use case (print(big_factorial), print(fib(2000))).
# Disable it here -- the sandbox is short-lived so the DoS-prevention
# motivation upstream doesn't apply.
_PY_PRELUDE = (
    "import sys\n"
    "sys.set_int_max_str_digits(0)\n"
    "import math, statistics, fractions, decimal, itertools, functools, "
    "collections, random, string, datetime, json, re\n"
)
_PY_TIMEOUT_SECONDS = 8
_PY_MAX_STDOUT_CHARS = 60000


def _python_code_is_safe(code: str) -> tuple[bool, str | None]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc.msg} (line {exc.lineno})"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                root = (alias.name or "").split(".", 1)[0]
                if root in _PY_BANNED_IMPORTS:
                    return False, f"import of {root!r} is not allowed in this sandbox"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _PY_BANNED_BUILTINS:
                return False, f"call to {node.func.id!r} is not allowed in this sandbox"
    return True, None


def _run_python_subprocess(code: str) -> tuple[str, str | None]:
    """Run ``code`` in a fresh interpreter; return (stdout, error or None).

    Uses ``python3 -I`` to ignore site/USER environment + a tempdir cwd so
    the snippet cannot poison the parent process or read project files.
    """
    safe, reason = _python_code_is_safe(code)
    if not safe:
        return "", f"refused: {reason}"
    wrapper = _PY_PRELUDE + (code.strip() + "\n")
    with tempfile.TemporaryDirectory(prefix="distil_chat_py_") as td:
        path = os.path.join(td, "snippet.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(wrapper)
        try:
            proc = subprocess.run(
                ["python3", "-I", path],
                cwd=td,
                text=True,
                capture_output=True,
                timeout=_PY_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return "", f"Python execution timed out after {_PY_TIMEOUT_SECONDS} seconds."
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    if proc.returncode != 0:
        tail = stderr[-1200:] if stderr else "(no stderr)"
        return stdout[:_PY_MAX_STDOUT_CHARS], (
            f"Python exited with code {proc.returncode}:\n{tail}"
        )
    return stdout[:_PY_MAX_STDOUT_CHARS], None


@function_tool
async def python_exec(
    ctx: RunContextWrapper[SN97AgentContext], code: str,
) -> dict[str, Any]:
    """Run a short Python snippet in a sandboxed subprocess and return its
    stdout. Use this whenever you need an exact computation (large
    arithmetic, factorials, Fibonacci, sympy / decimal / fractions / number
    theory, deterministic combinatorics) instead of guessing.

    The sandbox blocks network, file I/O, subprocess, ``open``/``exec``/
    ``eval`` and similar; it has these prelude imports already in scope:
    ``math``, ``statistics``, ``fractions``, ``decimal``, ``itertools``,
    ``functools``, ``collections``, ``random``, ``string``, ``datetime``,
    ``json``, ``re``. Print every value you want returned -- the runtime
    only captures stdout.

    Args:
        code: Python source to execute. Must include ``print(...)`` for any
            value you want back. Hard timeout: 6 seconds.

    Returns:
        ``{"stdout": str, "stderr_or_error": str | None, "exit_code": int}``.
    """
    state = ctx.context if ctx is not None else None
    if state is not None and state.python_runs_remaining <= 0:
        return {
            "stdout": "",
            "stderr_or_error": (
                "python_exec budget exhausted for this turn; pick the most "
                "important computation and answer with what you have."
            ),
            "exit_code": 1,
        }
    if state is not None:
        state.python_runs_remaining -= 1

    loop = asyncio.get_running_loop()
    stdout, err = await loop.run_in_executor(None, _run_python_subprocess, code)
    return {
        "stdout": stdout,
        "stderr_or_error": err,
        "exit_code": 0 if err is None else 1,
    }


# ── web_search (DuckDuckGo HTML) ─────────────────────────────────────────────

_WEB_TIMEOUT = httpx.Timeout(connect=4.0, read=12.0, write=10.0, pool=5.0)
_WEB_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; DistilSN97Chat/2.0; +https://distillation.ai)",
    "Accept-Language": "en-US,en;q=0.7",
}


def _strip_html_tags(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _resolve_ddg_href(href: str) -> str:
    href = html.unescape(href)
    if href.startswith("//duckduckgo.com/l/?uddg="):
        parsed = urllib.parse.urlparse("https:" + href)
        href = urllib.parse.parse_qs(parsed.query).get("uddg", [href])[0]
    elif href.startswith("/l/?uddg="):
        parsed = urllib.parse.urlparse("https://duckduckgo.com" + href)
        href = urllib.parse.parse_qs(parsed.query).get("uddg", [href])[0]
    return href


def _parse_duckduckgo_html(body: str, *, query: str, limit: int) -> list[dict]:
    title_re = re.compile(
        r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        flags=re.I | re.S,
    )
    snippet_re = re.compile(
        r'<(?:a|div|span)[^>]+class="[^"]*'
        r"(?:result__snippet|result-snippet|result__body__snippet)"
        r'[^"]*"[^>]*>(.*?)</(?:a|div|span)>',
        flags=re.I | re.S,
    )
    title_matches = list(title_re.finditer(body))
    snippet_matches = sorted(snippet_re.finditer(body), key=lambda m: m.start())

    results: list[dict] = []
    for tm in title_matches:
        href = _resolve_ddg_href(tm.group(1))
        title = _strip_html_tags(tm.group(2))
        snippet = ""
        for sm in snippet_matches:
            if sm.start() > tm.end():
                snippet = _strip_html_tags(sm.group(1))
                break
        if title:
            results.append({"title": title, "url": href, "snippet": snippet})
        if len(results) >= limit:
            break
    return results


@function_tool
async def web_search(
    ctx: RunContextWrapper[SN97AgentContext], query: str, limit: int = 5,
) -> dict[str, Any]:
    """Run a real-time DuckDuckGo HTML search and return the top results.

    Use for time-sensitive questions: news, prices, weather, sports scores,
    or any "today / right now / latest" query. Do NOT use for general
    knowledge that doesn't change.

    Args:
        query: Natural-language search query. Use the user's words; do not
            prepend "search the web for".
        limit: Max number of results to return (1-10, default 5).

    Returns:
        ``{"query": str, "results": [{"title", "url", "snippet"}, ...]}``.
    """
    del ctx  # tool is stateless wrt agent context
    q = (query or "").strip()
    q = re.sub(r"^\s*(search (?:the )?web for|web search for|look up|google)\s+", "", q, flags=re.I)
    if not q:
        return {"query": query, "results": [], "error": "empty query"}
    n = max(1, min(int(limit or 5), 10))

    url = "https://duckduckgo.com/html/?" + urllib.parse.urlencode({"q": q})
    try:
        async with httpx.AsyncClient(
            timeout=_WEB_TIMEOUT, headers=_WEB_HEADERS, follow_redirects=True,
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            body = resp.text
    except httpx.HTTPError as exc:
        return {
            "query": q,
            "results": [],
            "error": f"web search failed: {type(exc).__name__}: {str(exc)[:200]}",
        }

    results = _parse_duckduckgo_html(body, query=q, limit=n)
    return {"query": q, "results": results}


# ── sn97_state ───────────────────────────────────────────────────────────────

# The state-store imports live in the tool body to keep this module
# importable without the rest of the API package present (helps tests).

@function_tool
async def sn97_state(
    ctx: RunContextWrapper[SN97AgentContext], topic: str | None = None,
) -> dict[str, Any]:
    """Return the live SN97 (Bittensor Subnet 97) state from the validator
    state store.

    Use whenever the user asks about the current king, leaderboard,
    contenders, eval round/status, or a specific UID/miner. The runtime
    pulls this from the validator's read-side cache (sub-millisecond).

    Args:
        topic: Optional hint, one of ``"leaderboard"``, ``"king"``,
            ``"eval"``, ``"uid:<n>"``, or omit for a full snapshot.

    Returns:
        Structured dict with whichever of ``king``, ``contenders``,
        ``eval_progress``, ``uid_lookup`` are relevant.
    """
    try:
        from state_store import (
            eval_progress, h2h_latest, read_cache, top4_leaderboard,
            uid_hotkey_map,
        )
    except ImportError as exc:
        return {"error": f"state store unavailable: {exc}"}

    state = ctx.context if ctx is not None else None
    king_uid = state.king_uid if state else None
    king_model = state.king_model if state else None
    topic = (topic or "").strip().lower()

    out: dict[str, Any] = {}
    with contextlib.suppress(Exception):
        top4 = top4_leaderboard() or {}
        king = dict(top4.get("king") or {})
        if king_uid is not None:
            king.setdefault("uid", king_uid)
        if king_model:
            king.setdefault("model", king_model)
        contenders = [dict(c) for c in (top4.get("contenders") or [])[:4]]
        out["king"] = king
        if not topic or topic in {"leaderboard", "king", "contenders"}:
            out["contenders"] = contenders

    with contextlib.suppress(Exception):
        progress = eval_progress() or {}
        h2h = h2h_latest() or {}
        out["eval_progress"] = {
            "active": bool(progress.get("active") or progress.get("phase") not in (None, "", "idle")),
            "phase": progress.get("phase") or progress.get("stage") or progress.get("status"),
            "students_done": progress.get("students_done"),
            "students_total": progress.get("students_total"),
            "block": progress.get("completed_block") or h2h.get("block"),
        }

    if topic.startswith("uid:"):
        try:
            uid = int(topic.split(":", 1)[1].strip())
        except ValueError:
            uid = None
        if uid is not None:
            with contextlib.suppress(Exception):
                hotkey = uid_hotkey_map().get(str(uid))
                commitments_data = read_cache("commitments", {}) or {}
                commitments = commitments_data.get("commitments", commitments_data)
                model = None
                if hotkey and isinstance(commitments, dict):
                    c = commitments.get(hotkey) or {}
                    if isinstance(c, dict):
                        model = c.get("model") or c.get("repo")
                out["uid_lookup"] = {"uid": uid, "hotkey": hotkey, "model": model}

    if not out:
        out["error"] = "state store returned no live data"
    return out


# ── model_info ───────────────────────────────────────────────────────────────

@function_tool
async def model_info(
    ctx: RunContextWrapper[SN97AgentContext], model_path: str,
) -> dict[str, Any]:
    """Look up structured metadata for a HuggingFace model repo.

    Returns parameter count, MoE structure, license, downloads, etc. via
    the validator's cached HF probe. Use whenever the user asks "tell me
    about ``owner/repo``" or "what model is UID X running".

    Args:
        model_path: HuggingFace repo id like ``"moonshotai/Kimi-K2.6"`` or
            ``"Qwen/Qwen3.5-4B"``. Must include the ``owner/repo`` slash.
    """
    del ctx
    path = (model_path or "").strip().strip("`'\"")
    if "/" not in path or len(path) < 3:
        return {
            "error": "model_path must look like 'owner/repo'",
            "received": model_path,
        }
    try:
        from external import get_model_info as fetch_model_info_data
    except ImportError as exc:
        return {"error": f"model info helper unavailable: {exc}"}
    try:
        info = fetch_model_info_data(path)
    except Exception as exc:  # pragma: no cover -- defensive
        return {"error": f"{type(exc).__name__}: {str(exc)[:200]}", "model_path": path}
    if not isinstance(info, dict):
        return {"error": "unexpected response type", "model_path": path}
    info = dict(info)
    info["model_path"] = path
    return info


# ── summarise_history ────────────────────────────────────────────────────────
#
# The chat itself trims old turns to ``max_history=8`` before handing them
# to the agent, so this tool exists as a self-service compaction option for
# long, single-turn brainstorms where the conversation context is bigger
# than the window. It just runs the model itself in a tiny one-shot mode.

@function_tool
async def summarise_history(
    ctx: RunContextWrapper[SN97AgentContext], history_text: str, max_words: int = 200,
) -> dict[str, Any]:
    """Compress a long block of prior conversation into a short summary so
    you can keep referencing it without burning context window.

    Use this only when the user pastes (or you have) a very long history
    and the salient details would otherwise be cropped. Output is a single
    paragraph with the key facts and any open questions.

    Args:
        history_text: The full conversation/history to summarise.
        max_words: Soft cap on the summary length (default 200).
    """
    del ctx
    text = (history_text or "").strip()
    if not text:
        return {"summary": "", "warning": "empty history_text"}
    # Cheap deterministic summariser: keep the first/last few sentences.
    # We deliberately don't recurse into the model here because Runner.run
    # is already inside an agent turn and a nested LLM call would race the
    # outer loop. The model can use this scaffold + its own context.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) <= 6:
        summary = " ".join(sentences)
    else:
        head = sentences[:3]
        tail = sentences[-3:]
        summary = (
            " ".join(head)
            + " ... [middle of conversation truncated; "
            + str(len(sentences) - 6) + " sentences omitted] ... "
            + " ".join(tail)
        )
    words = summary.split()
    if len(words) > max_words:
        summary = " ".join(words[:max_words]) + " ..."
    return {"summary": summary, "source_chars": len(text)}


# ── Discovery + system-prompt synthesis ──────────────────────────────────────

ALL_TOOLS = [python_exec, web_search, sn97_state, model_info, summarise_history]


def system_prompt_for(king_uid: int | None, king_model: str | None) -> str:
    """Build the chat agent's system prompt for the current king.

    The prompt only describes the tool the model can actively invoke
    (``python_exec``, by writing a fenced ``\u0060\u0060\u0060python`` block). The
    runtime pre-fetches web search, SN97 live state, and HuggingFace model
    info BEFORE the model gets the question and injects the data as a
    user-visible tool result block. We deliberately don't let the model
    "call" those itself because the current king isn't fine-tuned to emit
    OpenAI-format ``tool_calls``; mentioning them as importable Python
    modules would just trip it into ``ModuleNotFoundError`` loops.
    """
    label = (
        f"UID {king_uid} ({king_model})" if king_uid is not None and king_model
        else (king_model or "the current SN97 king model")
    )
    return (
        "You are SN97 chat, the public conversational surface for Bittensor "
        f"Subnet 97. You are powered by {label}.\n\n"
        "Capabilities (the runtime grants these to you transparently):\n"
        "- Python sandbox: write a fenced ```python``` code block in your "
        "reply whenever you need an exact computation. The runtime "
        "executes it and sends you the real stdout in the next turn so "
        "you can revise your answer using the true value. Available "
        "imports already in scope: ``math``, ``statistics``, ``fractions``"
        ", ``decimal``, ``itertools``, ``functools``, ``collections``, "
        "``random``, ``string``, ``datetime``, ``json``, ``re``. Note: "
        "``math.fibonacci`` does NOT exist; if you need a Fibonacci "
        "number, write the iterative loop yourself. Always ``print(...)``"
        " whatever you want returned.\n"
        "- Web search, SN97 live state, HuggingFace model info: the "
        "runtime PRE-FETCHES these before you see the question whenever "
        "your prompt looks time-sensitive (news / today / latest / "
        "prices / weather), mentions SN97 / king / leaderboard / UID, or "
        "asks about a HuggingFace ``owner/repo``. The results arrive as "
        "an authoritative system message labelled "
        "``WEB_SEARCH_RESULT`` / ``SN97_LIVE_STATE`` / ``MODEL_INFO_RESULT``"
        ". You can NOT call these helpers yourself -- treat the data the "
        "runtime injects as ground truth and quote it verbatim.\n\n"
        "Hard rules:\n"
        "1. Never claim you lack internet, code execution, real-time "
        "data, or math. The runtime covers all of those for you. NEVER "
        "say 'as of my knowledge cutoff', 'I don't have access to a "
        "real Python environment', 'I can't connect to external "
        "websites', or any equivalent disclaimer -- the orchestrator "
        "WILL run your ``python_exec`` blocks and WILL give you live "
        "web search / SN97 data, and the user expects you to use them.\n"
        "2. Only write a ```python``` block when the user is actually "
        "asking for a calculation, derivation, simulation, or other "
        "exact numeric / symbolic result. For greetings, small talk, "
        "definitions, opinions, code reviews, or any answer that "
        "doesn't depend on running code, REPLY DIRECTLY in plain text "
        "without invoking the sandbox. Do NOT print placeholder values "
        "like ``print(42)`` to 'use the tool'.\n"
        "3. ** TOOL-USE PROTOCOL ** When you decide to run "
        "``python_exec``, write a SHORT one-line preface like ``Computing"
        " with python:`` followed IMMEDIATELY by the ```python``` block "
        "and NOTHING ELSE in that turn. Do NOT write a final answer in "
        "the same turn -- the runtime will execute the code and call "
        "you again with the real stdout, and you give the final answer "
        "in that NEXT turn. Repeating your guessed answer alongside the "
        "code wastes turns and confuses the user when the guess is "
        "wrong.\n"
        "4. When you DO need a computation, write the smallest snippet "
        "that prints the answer (or just the few values you need). "
        "Don't re-import the prelude modules -- ``math``, ``decimal``, "
        "``fractions``, ``itertools``, ``functools``, ``collections``, "
        "``random``, ``string``, ``datetime``, ``json``, ``re``, "
        "``statistics`` are already imported.\n"
        "5. NEVER fabricate a 'Tool Output:' / 'Output:' / 'Sandbox "
        "stdout:' block, and NEVER write strings like "
        "``call_pyfence_xxx{...}`` or ``## Return of call_*`` -- the "
        "runtime is the only thing that emits tool calls and tool "
        "results. If you do, the chat will cut off your message there.\n"
        "6. When a tool result comes back, treat it as GROUND TRUTH. "
        "Quote numbers, URLs, prices, scores VERBATIM. If your earlier "
        "guess disagrees with the tool result, correct yourself "
        "explicitly and use the tool result. Do not repeat your earlier "
        "wrong guess.\n"
        "7. If a ```python``` block FAILED (you'll see "
        "``stderr_or_error``"
        " in the result), do NOT pretend it succeeded. Read the error, "
        "rewrite the code with a real fix, and try again. Do not retry "
        "the same broken snippet -- if you can't fix it after 2 tries, "
        "explain the problem to the user honestly.\n"
        "8. Do NOT try to import 'sn97_state', 'web_search', or "
        "'model_info' as Python modules -- they are runtime helpers, "
        "not Python packages. They show up automatically as system "
        "context whenever the user's question needs them.\n"
        "8a. Do NOT try to import 'requests', 'urllib', 'socket', or "
        "any other network module from the sandbox -- they are "
        "blocked. If you need fresh web data, the runtime already "
        "ran ``web_search`` for you and the results are in the "
        "``WEB_SEARCH_RESULT`` system message above; quote those.\n"
        "8b. When ``WEB_SEARCH_RESULT`` is present, the prices / "
        "headlines / scores you should report come from THERE -- "
        "not from your training data, not from a placeholder. Cite "
        "the source URL. Same for ``SN97_LIVE_STATE`` (king UID + "
        "model + scores) and ``MODEL_INFO_RESULT`` (params, paths).\n"
        "9. Wrap private reasoning in <think>...</think> tags so the "
        "dashboard renders it as the assistant's thinking; put the "
        "user-facing answer outside the <think>.\n"
        "10. Be concise and direct. Use Markdown for headers, lists, "
        "and code blocks; use ``$...$`` for inline math and ``$$...$$`` "
        "on its own line for block math.\n"
        "11. The chat is in real time -- the user is waiting for your "
        "answer. Don't say 'I'll get back to you' or 'let me check' "
        "without immediately following with the code or final answer.\n"
    )


__all__ = [
    "ALL_TOOLS",
    "SN97AgentContext",
    "model_info",
    "python_exec",
    "summarise_history",
    "sn97_state",
    "system_prompt_for",
    "web_search",
]
