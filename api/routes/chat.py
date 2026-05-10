"""OpenAI-compatible chat endpoints; proxies to king vLLM via chat-tunnel."""

import ast
import html
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import urllib.parse
import uuid

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from config import (
    CHAT_POD_HOST,
    CHAT_POD_PORT,
    STATE_DIR,
)
from external import get_model_info as fetch_model_info_data
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
    top4_leaderboard,
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


# chat_server.py serves the king under the fixed name "sn97-king" so
# we rewrite client-supplied model names below.
CHAT_POD_SERVED_MODEL = "sn97-king"

# chat-tunnel.service forwards 127.0.0.1:8100 -> chat-pod:8100; using
# localhost reuses a TCP keep-alive and detects tunnel-down in <2s.
_LOCAL_CHAT_BASE = f"http://127.0.0.1:{CHAT_POD_PORT}"

_chat_http_client: httpx.AsyncClient | None = None
_chat_http_lock = threading.Lock()


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
    """Pin served-model to sn97-king, default thinking off + max_tokens=1024,
    and inject a math-formatting system prompt if the client didn't."""
    payload = dict(payload)
    payload["model"] = CHAT_POD_SERVED_MODEL
    kwargs = dict(payload.get("chat_template_kwargs") or {})
    kwargs.setdefault("enable_thinking", False)
    payload["chat_template_kwargs"] = kwargs
    if payload.get("max_tokens") is None:
        payload["max_tokens"] = 1024
    msgs = list(payload.get("messages") or [])
    has_system = any(
        (isinstance(m, dict) and m.get("role") == "system") for m in msgs
    )
    if not has_system:
        formatting_guide = (
            "You are a helpful, concise assistant. When you write math, "
            "ALWAYS use LaTeX with consistent delimiters: ``$...$`` for "
            "inline math (e.g., the speed $v=d/t$) and ``$$...$$`` on "
            "their OWN lines for block math. Never emit bare LaTeX "
            "commands (``\\text``, ``\\frac``, ``\\times``, "
            "``\\approx``) outside of these delimiters. For simple "
            "arithmetic prefer plain text (``2.36 × 10^22`` is fine) — "
            "reserve LaTeX for multi-line derivations and equations "
            "with fractions, integrals, or sums. Use Markdown for "
            "headers, lists, and code blocks."
        )
        payload["messages"] = [
            {"role": "system", "content": formatting_guide}
        ] + msgs
    return payload


# ── Product-grade chat orchestration ─────────────────────────────────────────

_CHAT_ORCHESTRATION_ENABLED = (
    os.environ.get("DISTIL_CHAT_ORCHESTRATION", "1") != "0"
)

_SN97_RE = re.compile(
    r"\b(sn\s?97|sn-97|subnet|king|leaderboard|miner|uid|eval|round|"
    r"validator|arbos|distil)\b",
    re.IGNORECASE,
)
_CODE_RE = re.compile(
    r"\b(run|execute|python|script|code|calculate|compute|evaluate|math|"
    r"power|fibonacci|fib(?:onacci)?_?sequence)\b",
    re.IGNORECASE,
)
_UID_RE = re.compile(r"\buid\s*[:=#]?\s*(\d{1,3})\b", re.IGNORECASE)
_MODEL_PATH_RE = re.compile(
    r'"model_path"\s*:\s*"([^"]+)"|'
    r"\b([A-Za-z0-9._-]{2,40}/[A-Za-z0-9._-]{2,100})\b"
)
_WEB_SEARCH_RE = re.compile(
    r"(?:"
    # Explicit search verbs — always trigger.
    r"\b(?:search\s+(?:the\s+)?web|web\s+search|look\s+up|google\s+for|google\s+the|"
    r"duckduckgo|find\s+me|fetch\s+(?:the\s+)?(?:latest|news|results?))\b"
    # Time-sensitive markers anywhere in the question.
    r"|\b(?:right\s+now|today|tonight|this\s+(?:morning|afternoon|evening|week|month|year))\b"
    r"|\b(?:latest|breaking|current|today'?s|recent|live|real[-\s]?time)\b"
    # Categories that are always fresh-info questions.
    r"|\b(?:weather|forecast|temperature|stock\s+price|share\s+price|exchange\s+rate|"
    r"headline[s]?|news|score)\b"
    # Crypto/asset price patterns.
    r"|\b(?:bitcoin|btc|ethereum|eth|tesla|gold|silver|oil|sp500|s&p\s*500)\b"
    # "price of X" / "X price" prompts.
    r"|\bprice\s+of\s+\w+\b|\b\w+\s+price\b"
    r")",
    re.IGNORECASE,
)
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>(.*?)</think>", re.IGNORECASE | re.DOTALL)
_FIB_INDEX_RE = re.compile(
    r"(?:(\d+)\s*(?:\^|\*\*)\s*(\d+)(?:st|nd|rd|th)?|"
    r"(\d+)\s*(?:to\s+the\s+power\s+of|power\s+of)\s*(?:the\s*)?(\d+)(?:st|nd|rd|th)?|"
    r"\b(\d{1,8})(?:st|nd|rd|th)?)"
    r".{0,40}\b(?:fib(?:onacci)?|fibonacci)\b",
    re.IGNORECASE,
)
_POWER_PHRASE_RE = re.compile(
    r"\b(\d{1,9})\s*(?:to\s+the\s+power\s+of|power\s+of)\s*(?:the\s*)?"
    r"(\d{1,6})(?:st|nd|rd|th)?\b",
    re.IGNORECASE,
)
_FACTORIAL_RE = re.compile(
    r"(?:factorial\s*(?:of)?|(\d{1,6})\s*!)\s*(\d{1,6})?",
    re.IGNORECASE,
)
_DERAIL_MARKERS = (
    "Use the tool ",
    "get_model_info",
    "get_subnet_overview",
    "get_leaderboard",
    "knowledge base",
    "Update knowledge base",
    "outlook steady",
    "Outlook steady",
    "steady demand for clear",
    "Persistent File System",
    "Pyodide",
)


def _latest_user_text(messages: list[dict]) -> str:
    for msg in reversed(messages or []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content") or ""
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(str(part.get("text") or ""))
                return "\n".join(parts)
    return ""


def _recent_chat_text(messages: list[dict], *, max_messages: int = 6) -> str:
    parts: list[str] = []
    for msg in (messages or [])[-max_messages:]:
        if not isinstance(msg, dict) or msg.get("role") not in {"user", "assistant"}:
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        if msg.get("role") == "assistant":
            content = _strip_client_thinking(content)
        if content:
            parts.append(content)
    return "\n".join(parts)


def _contextual_tool_user_text(messages: list[dict], user_text: str) -> str:
    """Synthesize a local tool query when a numeric follow-up references
    a Fibonacci context already established earlier in the chat."""
    text = user_text or ""
    if "fib" in text.lower():
        return text
    recent = _recent_chat_text(messages)
    if "fib" not in recent.lower():
        return text
    followup = re.search(
        r"(?:"
        r"\b(?:yes|yeah|yep|correct|right|no|not)\b.{0,30})?"
        r"(\d{1,8}\s*(?:\^|\*\*)\s*\d{1,6}|\d{1,8})(?:st|nd|rd|th)?"
        r"(?:\s+(?:number|term|index))?\b",
        text,
        re.IGNORECASE,
    )
    if followup and re.search(r"\b(?:number|term|index|yes|yeah|yep|correct|right|no|not)\b", text, re.IGNORECASE):
        return f"{text} fibonacci"
    return text


_CLIENT_THINK_RE = re.compile(r"<think\b[^>]*>.*?</think>\s*", re.IGNORECASE | re.DOTALL)
_RUNTIME_TRACE_LINE_RE = re.compile(
    r"(?im)^Runtime trace, not hidden model reasoning:.*(?:\n|$)"
)


def _strip_client_thinking(content: str) -> str:
    """Drop previously displayed <think>/runtime-trace blocks from history."""
    content = _CLIENT_THINK_RE.sub("", content or "")
    content = _RUNTIME_TRACE_LINE_RE.sub("", content)
    return content.strip()


def _clean_client_messages(messages: list[dict], *, system: str, max_history: int = 8) -> list[dict]:
    """Keep recent user/assistant text only; tools are handled by the proxy."""
    cleaned = [{"role": "system", "content": system}]
    for msg in (messages or [])[-max_history:]:
        if not isinstance(msg, dict) or msg.get("role") not in {"user", "assistant"}:
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        if msg["role"] == "assistant":
            content = _strip_client_thinking(content)
            if not content:
                continue
        if any(marker in content for marker in _DERAIL_MARKERS):
            continue
        cleaned.append({"role": msg["role"], "content": content[:4000]})
    return cleaned


def _live_sn97_context(user_text: str, king_uid: int | None, king_model: str | None) -> tuple[str, str | None]:
    """Return (thinking, deterministic answer if obvious)."""
    top4 = top4_leaderboard() or {}
    progress = eval_progress() or {}
    h2h = h2h_latest() or {}
    king = dict(top4.get("king") or {})
    if king_uid is not None:
        king.setdefault("uid", king_uid)
    if king_model:
        king.setdefault("model", king_model)
    contenders = [dict(c) for c in (top4.get("contenders") or [])[:4]]
    eval_active = bool(progress.get("active") or progress.get("phase") not in (None, "", "idle"))

    thought_lines = [
        "I checked the live SN97 state before answering.",
        f"Current king: UID {king.get('uid', king_uid)}, model {king.get('model') or king_model or 'unknown'}.",
    ]
    if progress:
        stage = progress.get("phase") or progress.get("stage") or progress.get("status")
        thought_lines.append(f"Eval status: active={eval_active}, stage={stage or 'unknown'}.")

    lower = user_text.lower()
    answer = None
    if "leaderboard" in lower or "top" in lower:
        rows = [king] + contenders
        lines = ["Current SN97 leaderboard:"]
        for i, row in enumerate(rows[:5], start=1):
            comp = row.get("composite") or {}
            score = comp.get("final") or row.get("score") or row.get("h2h_kl")
            score_txt = f", score {score:.4f}" if isinstance(score, (int, float)) else ""
            label = "king" if i == 1 else "contender"
            lines.append(
                f"{i}. UID {row.get('uid')} ({label}): {row.get('model') or 'unknown'}{score_txt}"
            )
        answer = "\n".join(lines)
    elif "king" in lower or "who" in lower:
        answer = (
            f"The current SN97 king is UID {king.get('uid', king_uid)}"
            f" running `{king.get('model') or king_model or 'unknown'}`."
        )
    elif "eval" in lower or "round" in lower or "status" in lower:
        stage = progress.get("phase") or progress.get("stage") or progress.get("status") or "unknown"
        done = progress.get("students_done")
        total = progress.get("students_total")
        block = progress.get("completed_block") or h2h.get("block")
        bits = [f"Eval active: {eval_active}", f"stage: {stage}"]
        if done is not None and total is not None:
            bits.append(f"students: {done}/{total}")
        if block is not None:
            bits.append(f"block: {block}")
        answer = "Current SN97 eval status: " + ", ".join(bits) + "."
    else:
        m = _UID_RE.search(user_text)
        if m:
            uid = int(m.group(1))
            commitments_data = read_cache("commitments", {}) or {}
            commitments = commitments_data.get("commitments", commitments_data)
            hotkey = uid_hotkey_map().get(str(uid))
            model = None
            if hotkey and isinstance(commitments, dict):
                c = commitments.get(hotkey) or {}
                if isinstance(c, dict):
                    model = c.get("model") or c.get("repo")
            answer = (
                f"UID {uid} is registered as `{model or 'unknown model'}`. "
                "For the full score/history view, use the dashboard miner page."
            )

    return "\n".join(thought_lines), answer


def _model_info_answer(user_text: str) -> tuple[str, str] | None:
    if "get_model_info" not in user_text and "model_path" not in user_text:
        return None
    m = _MODEL_PATH_RE.search(user_text)
    if not m:
        return (
            "I recognized a model-info request but did not find a valid HuggingFace repo path.",
            "I need a HuggingFace repo path like `owner/repo` to look up model info.",
        )
    model_path = (m.group(1) or m.group(2) or "").strip()
    info = fetch_model_info_data(model_path)
    thought = f"I executed the model-info lookup for `{model_path}` via the Distil API helper."
    if not isinstance(info, dict) or info.get("error"):
        return thought, f"I could not fetch reliable model info for `{model_path}`: {info.get('error') if isinstance(info, dict) else 'unknown error'}"
    fields = []
    for key in ("params_b", "is_moe", "num_experts", "num_active_experts", "pipeline_tag", "license", "downloads", "likes"):
        if info.get(key) is not None:
            fields.append(f"- `{key}`: {info.get(key)}")
    if not fields:
        fields.append("- No detailed metadata was available in the cache/provider response.")
    return thought, f"Model info for `{model_path}`:\n" + "\n".join(fields)


def _strip_html_tags(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


async def _web_search_tool(query: str, *, limit: int = 5) -> str:
    q = query.strip()
    q = re.sub(r"^\s*(search (?:the )?web for|web search for|look up|google)\s+", "", q, flags=re.I)
    if not q:
        return "No search query detected."
    url = "https://duckduckgo.com/html/?" + urllib.parse.urlencode({"q": q})
    headers = {"User-Agent": "Mozilla/5.0 (compatible; DistilSN97Chat/1.0)"}
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(12.0, connect=4.0),
            headers=headers,
            follow_redirects=True,
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            body = resp.text
    except Exception as exc:
        return f"Web search failed: {type(exc).__name__}: {str(exc)[:200]}"

    return _parse_duckduckgo_html(body, query=q, limit=limit)


def _resolve_ddg_href(href: str) -> str:
    href = html.unescape(href)
    if href.startswith("//duckduckgo.com/l/?uddg="):
        parsed = urllib.parse.urlparse("https:" + href)
        href = urllib.parse.parse_qs(parsed.query).get("uddg", [href])[0]
    elif href.startswith("/l/?uddg="):
        parsed = urllib.parse.urlparse("https://duckduckgo.com" + href)
        href = urllib.parse.parse_qs(parsed.query).get("uddg", [href])[0]
    return href


def _parse_duckduckgo_html(body: str, *, query: str, limit: int = 5) -> str:
    """Pair (title, href, snippet) from DDG HTML for the model context."""
    title_re = re.compile(
        r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        flags=re.I | re.S,
    )
    snippet_re = re.compile(
        r'<(?:a|div|span)[^>]+class="[^"]*(?:result__snippet|result-snippet|result__body__snippet)[^"]*"[^>]*>(.*?)</(?:a|div|span)>',
        flags=re.I | re.S,
    )
    title_matches = list(title_re.finditer(body))
    snippet_matches = list(snippet_re.finditer(body))
    snippets_by_pos = sorted(snippet_matches, key=lambda m: m.start())

    results: list[tuple[str, str, str]] = []
    for tm in title_matches:
        href = _resolve_ddg_href(tm.group(1))
        title = _strip_html_tags(tm.group(2))
        snippet = ""
        for sm in snippets_by_pos:
            if sm.start() > tm.end():
                snippet = _strip_html_tags(sm.group(1))
                break
        if title:
            results.append((title, href, snippet))
        if len(results) >= limit:
            break

    if not results:
        return f"No web results parsed for query: {query}"
    lines = [f"query = {query}"]
    for i, (title, href, snippet) in enumerate(results, 1):
        if snippet:
            lines.append(f"{i}. {title}\n   {href}\n   snippet: {snippet}")
        else:
            lines.append(f"{i}. {title}\n   {href}")
    return "\n".join(lines)


def _extract_python_code(user_text: str) -> tuple[str | None, str]:
    fenced = re.search(r"```(?:python|py)?\s*(.*?)```", user_text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip(), "python code block"

    run_py = re.search(
        r"\b(?:run|execute)\s+(?:this\s+)?(?:python|py)\s*:?\s*(.+)$",
        user_text,
        re.IGNORECASE | re.DOTALL,
    )
    if run_py:
        return run_py.group(1).strip(), "inline Python request"

    inline = re.search(r"`([^`]{3,400})`", user_text)
    if inline and any(ch in inline.group(1) for ch in "+-*/()%^"):
        expr = inline.group(1).strip().replace("^", "**")
        return f"print({expr})", "inline arithmetic expression"

    power = _POWER_PHRASE_RE.search(user_text)
    if power:
        base = int(power.group(1))
        exp = int(power.group(2))
        return f"print({base} ** {exp})", f"power expression {base}^{exp}"

    factorial = _FACTORIAL_RE.search(user_text)
    if factorial:
        n_txt = factorial.group(1) or factorial.group(2)
        try:
            n = int(n_txt)
        except (TypeError, ValueError):
            n = -1
        if 0 <= n <= 100_000:
            return (
                "import math, sys\n"
                "if hasattr(sys, 'set_int_max_str_digits'):\n"
                "    sys.set_int_max_str_digits(0)\n"
                f"n = {n}\n"
                "value = math.factorial(n)\n"
                "s = str(value)\n"
                "print('n =', n)\n"
                "print('digits =', len(s))\n"
                "if len(s) <= 1000:\n"
                "    print('value =', s)\n"
                "else:\n"
                "    print('value_head_80 =', s[:80])\n"
                "    print('value_tail_80 =', s[-80:])",
                "factorial runtime",
            )

    # Conservative expression extraction for requests like
    # "calculate ((88+43)**(2-2))//9".
    expr_match = re.search(
        r"(?:calculate|compute|evaluate|what is|what's|solve|result of|math)\s+([0-9\s+\-*/%().,_<>=!&|^~]+)",
        user_text,
        re.IGNORECASE,
    )
    if expr_match:
        expr = expr_match.group(1).strip(" .?\n\t")
        if expr:
            expr = expr.replace("^", "**")
            return f"print({expr})", "arithmetic expression"
    bare_expr = re.fullmatch(r"\s*(?:yes|yeah|yep|no|not)?\s*([0-9][0-9\s+\-*/%().,_^]{1,200})\s*\??\s*", user_text, re.I)
    if bare_expr and any(ch in bare_expr.group(1) for ch in "+-*/%^"):
        expr = bare_expr.group(1).strip(" .?\n\t").replace("^", "**")
        return f"print({expr})", "bare arithmetic expression"
    return None, ""


def _python_code_is_reasonable(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    banned = {"socket", "subprocess", "shutil", "pathlib", "requests", "httpx", "urllib"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                root = (alias.name or "").split(".", 1)[0]
                if root in banned:
                    return False
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"open", "exec", "eval", "compile", "__import__"}:
                return False
    return True


def _run_python_code(code: str, *, max_stdout_chars: int = 4000) -> tuple[str, str | None]:
    if not _python_code_is_reasonable(code):
        return "", "I refused to run that code because it uses unsafe imports or dynamic execution."
    wrapper = (
        "import math, statistics, fractions, decimal, itertools, functools, collections\n"
        + code.strip()
        + "\n"
    )
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
                timeout=4,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return "", "Python execution timed out after 4 seconds."
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if proc.returncode != 0:
        return out, f"Python exited with code {proc.returncode}: {err[-800:]}"
    return out[:max_stdout_chars], None


def _parse_key_value_lines(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in (text or "").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _python_direct_answer(desc: str, out: str, err: str | None) -> str:
    if err:
        body = err
        if out:
            body += f"\n\nPartial stdout:\n{out}"
        return f"I ran Python for {desc}, but it failed:\n\n```text\n{body}\n```"
    return f"I executed Python for {desc}:\n\n```text\n{out or '(no stdout)'}\n```"


def _fibonacci_direct_answer(desc: str, out: str, err: str | None) -> str:
    if err:
        return _python_direct_answer(desc, out, err)
    values = _parse_key_value_lines(out)
    index = values.get("index", "?")
    digits = values.get("digits")
    value = values.get("value")
    if value:
        digit_note = f"\n\nIt has {digits} digits." if digits else ""
        return (
            f"The Fibonacci number at index {index} ({desc}) is:\n\n"
            f"```text\n{value}\n```"
            f"{digit_note}"
        )
    head = values.get("value_head_80")
    tail = values.get("value_tail_80")
    if head and tail:
        return (
            f"The Fibonacci number at index {index} ({desc}) has {digits or 'many'} digits.\n\n"
            f"It is too long to print in full here, but I computed it exactly with Python fast doubling.\n\n"
            f"```text\nfirst 80 digits: {head}\nlast 80 digits:  {tail}\n```"
        )
    return _python_direct_answer(desc, out, None)


def _tool_reasoning(runtime_trace: list[str], model_reasoning: str = "") -> str:
    parts = []
    if model_reasoning:
        parts.append(model_reasoning.strip())
    if runtime_trace:
        trace = "\n".join(f"- {line}" for line in runtime_trace)
        parts.append("Tool trace:\n" + trace)
    return "\n\n".join(p for p in parts if p)


class _ThinkStreamSplitter:
    """Streaming splitter: routes <think>...</think> content to reasoning channel.

    Buffers trailing text so partial tags spanning chunk boundaries still split
    correctly. Use ``feed`` for streamed deltas and ``flush`` at end of stream.
    """

    _OPEN = "<think"
    _OPEN_FULL = ">"
    _CLOSE = "</think>"

    def __init__(self) -> None:
        self._buf = ""
        self._inside = False

    def _trailing_partial_len(self, target: str) -> int:
        """Return the length of the longest suffix of ``self._buf`` that is
        also a non-empty prefix of ``target`` (case-insensitive).
        """
        max_len = min(len(self._buf), len(target))
        lower_buf = self._buf.lower()
        lower_target = target.lower()
        for k in range(max_len, 0, -1):
            if lower_buf.endswith(lower_target[:k]):
                return k
        return 0

    def feed(self, chunk: str) -> tuple[str, str]:
        if not chunk:
            return "", ""
        self._buf += chunk
        visible = ""
        reasoning = ""
        while self._buf:
            if not self._inside:
                idx = self._buf.lower().find(self._OPEN)
                if idx < 0:
                    keep = self._trailing_partial_len(self._OPEN)
                    if keep:
                        visible += self._buf[:-keep]
                        self._buf = self._buf[-keep:]
                    else:
                        visible += self._buf
                        self._buf = ""
                    break
                visible += self._buf[:idx]
                rest = self._buf[idx:]
                close_open = rest.find(self._OPEN_FULL)
                if close_open < 0:
                    self._buf = rest
                    break
                self._buf = rest[close_open + 1:]
                self._inside = True
            else:
                idx = self._buf.lower().find(self._CLOSE)
                if idx < 0:
                    keep = self._trailing_partial_len(self._CLOSE)
                    if keep:
                        reasoning += self._buf[:-keep]
                        self._buf = self._buf[-keep:]
                    else:
                        reasoning += self._buf
                        self._buf = ""
                    break
                reasoning += self._buf[:idx]
                self._buf = self._buf[idx + len(self._CLOSE):]
                self._inside = False
        return visible, reasoning

    def flush(self) -> tuple[str, str]:
        visible = ""
        reasoning = ""
        if self._buf:
            if self._inside:
                reasoning = self._buf
            else:
                visible = self._buf
            self._buf = ""
        self._inside = False
        return visible, reasoning


def _split_think_blocks(content: str) -> tuple[str, str]:
    """Move <think>...</think> segments out of content into a reasoning string."""
    if not content or "<think" not in content.lower():
        return content, ""
    think_chunks = [m.group(1).strip() for m in _THINK_BLOCK_RE.finditer(content)]
    cleaned = _THINK_BLOCK_RE.sub("", content).strip()
    reasoning = "\n\n".join(c for c in think_chunks if c).strip()
    return cleaned, reasoning


_REFUSAL_PATTERNS = (
    re.compile(r"\bI (?:cannot|can't|am unable|do not have the ability) (?:do that|help|browse|access)", re.IGNORECASE),
    re.compile(r"\bI (?:do not|don't) have (?:real[-\s]?time|internet|web) access", re.IGNORECASE),
    re.compile(r"\bI (?:do not|don't) have (?:the ability to|access to) (?:browse|search|run code|execute)", re.IGNORECASE),
    re.compile(r"\bI'?m sorry,? (?:but )?I (?:can'?t|cannot)", re.IGNORECASE),
    re.compile(r"\bAs an AI(?: language)? model,? I (?:cannot|can't|do not|don't)", re.IGNORECASE),
    re.compile(r"\bI'?m unable to (?:do that|help with that|browse|access|search|run|execute)", re.IGNORECASE),
)


def _looks_empty_or_derailed(content: str) -> bool:
    if not content:
        return True
    stripped = content.strip()
    if not stripped:
        return True
    if any(marker in stripped for marker in _DERAIL_MARKERS):
        return True
    if any(pat.search(stripped) for pat in _REFUSAL_PATTERNS):
        return True
    return False


def _model_quoted_numeric_answer(direct_answer: str, model_content: str) -> bool:
    """True if ``model_content`` faithfully reproduces the numeric tool output."""
    if not direct_answer or not model_content:
        return False
    candidates = re.findall(r"[0-9][0-9,_\s]{2,}", direct_answer)
    if not candidates:
        return True
    longest = max(candidates, key=len)
    digits = re.sub(r"[^0-9]", "", longest)
    model_digits = re.sub(r"[^0-9]", "", model_content)
    if not model_digits:
        return False
    if len(digits) <= 24:
        if digits not in model_digits:
            return False
        if len(digits) >= 3:
            # If stdout says "285", an answer that says "385 ... 285" is
            # still bad. Ignore shorter prompt/index values (like the "10" in
            # "first 10 integers") but reject same-width contradictory values.
            for token in re.findall(r"\d[\d,]*", model_content):
                token_digits = re.sub(r"[^0-9]", "", token)
                if len(token_digits) >= len(digits) and token_digits != digits:
                    return False
        return True
    return digits[:8] in model_digits and digits[-8:] in model_digits


def _extract_fibonacci_tool(user_text: str) -> tuple[str | None, str]:
    if "fib" not in user_text.lower():
        return None, ""
    m = _FIB_INDEX_RE.search(user_text)
    if not m:
        return None, ""
    if m.group(1) and m.group(2):
        n = int(m.group(1)) ** int(m.group(2))
        desc = f"Fibonacci index {m.group(1)}^{m.group(2)} = {n}"
    elif m.group(3) and m.group(4):
        n = int(m.group(3)) ** int(m.group(4))
        desc = f"Fibonacci index {m.group(3)}^{m.group(4)} = {n}"
    else:
        n = int(m.group(5))
        desc = f"Fibonacci index {n}"
    if n < 0 or n > 1_000_000:
        return None, f"Requested Fibonacci index {n} is outside the runtime limit (0..1,000,000)."
    code = f"""
import sys
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

def fib_pair(n):
    if n == 0:
        return (0, 1)
    a, b = fib_pair(n // 2)
    c = a * (2 * b - a)
    d = a * a + b * b
    if n % 2:
        return (d, c + d)
    return (c, d)

n = {n}
value = fib_pair(n)[0]
s = str(value)
print("index =", n)
print("digits =", len(s))
if len(s) <= 20000:
    print("value =", s)
else:
    print("value_head_80 =", s[:80])
    print("value_tail_80 =", s[-80:])
"""
    return code.strip(), desc


async def _prepare_orchestrated_chat(
    body: dict,
    king_uid: int | None,
    king_model: str | None,
    *,
    stream: bool = False,
) -> tuple[dict, list[str], list[str], str | None]:
    """Build the vLLM request plus deterministic runtime tool context.

    Tool execution happens before the model call. The streaming path uses this
    to emit tool output immediately, then streams the raw vLLM answer.
    """
    messages = list(body.get("messages") or [])
    user_text = _latest_user_text(messages)
    tool_user_text = _contextual_tool_user_text(messages, user_text)
    runtime_trace: list[str] = []
    tool_context: list[str] = []
    direct_answer: str | None = None

    model_info = _model_info_answer(user_text or "")
    if model_info:
        t, answer = model_info
        runtime_trace.append(t)
        tool_context.append(f"MODEL_INFO_RESULT:\n{answer}")

    if _WEB_SEARCH_RE.search(user_text or ""):
        result = await _web_search_tool(user_text or "")
        runtime_trace.append("Executed a DuckDuckGo web search from the chat runtime.")
        tool_context.append(f"WEB_SEARCH_RESULT:\n{result}")

    # Tool execution: SN97 live state.
    if _SN97_RE.search(user_text or ""):
        t, answer = _live_sn97_context(user_text, king_uid, king_model)
        runtime_trace.append(t)
        if answer:
            tool_context.append(f"SN97_LIVE_RESULT:\n{answer}")

    fib_code, fib_desc = _extract_fibonacci_tool(tool_user_text or "")
    if fib_desc and not fib_code:
        runtime_trace.append(f"Fibonacci runtime rejected the request: {fib_desc}")
        tool_context.append(f"PYTHON_EXECUTION_RESULT:\n{fib_desc}")
        direct_answer = fib_desc
    elif fib_code:
        out, err = _run_python_code(fib_code, max_stdout_chars=25000)
        runtime_trace.append(f"Executed Python for {fib_desc}.")
        if err:
            result = f"Python failed:\n{err}"
            if out:
                result += f"\nPartial stdout:\n{out}"
        else:
            result = f"Python stdout:\n{out or '(no stdout)'}"
        tool_context.append(f"PYTHON_EXECUTION_RESULT:\n{result}")
        direct_answer = _fibonacci_direct_answer(fib_desc, out, err)

    # Tool execution: Python code / arithmetic.
    code, code_kind = _extract_python_code(user_text or "")
    if code and not fib_code:
        out, err = _run_python_code(code)
        runtime_trace.append(f"Executed a {code_kind} in a short-lived Python sandbox.")
        if err:
            result = f"Python failed:\n{err}"
            if out:
                result += f"\nPartial stdout:\n{out}"
        else:
            result = f"Python stdout:\n{out or '(no stdout)'}"
        tool_context.append(f"PYTHON_EXECUTION_RESULT:\n{result}")
        direct_answer = _python_direct_answer(code_kind, out, err)
    elif _CODE_RE.search(user_text or "") and "python" in (user_text or "").lower():
        runtime_trace.append("The user asked for Python execution but no runnable code/expression was detected.")
        tool_context.append(
            "PYTHON_EXECUTION_RESULT:\nNo runnable Python code or arithmetic expression was detected."
        )

    king_label = (
        f"UID {king_uid} ({king_model})" if king_uid is not None and king_model
        else (king_model or "the current SN97 king model")
    )
    system = (
        "You are SN97 chat, the public chat surface for Bittensor Subnet 97. "
        f"You are powered by {king_label}.\n\n"
        "The chat runtime gives you these capabilities transparently:\n"
        "- Web search: when the runtime detects a time-sensitive query it will "
        "fetch DuckDuckGo results for you.\n"
        "- Python execution: short Python snippets and arithmetic expressions "
        "run in a sandbox; you see the stdout.\n"
        "- Math helpers: large powers, factorials, and Fibonacci numbers are "
        "computed exactly by the runtime.\n"
        "- SN97 live state: the runtime can quote the current king, leaderboard, "
        "and eval round.\n\n"
        "Important rules for you, the model:\n"
        "1. Do NOT claim you lack internet access, code execution, math, or "
        "real-time data — the runtime handles those for you.\n"
        "2. Whenever a system message contains tool results, treat them as "
        "the authoritative ground truth and weave them into your answer in "
        "natural prose.\n"
        "3. Think step-by-step before answering. Wrap private reasoning in "
        "<think>...</think> tags so it is shown as the assistant's thinking. "
        "Put the user-facing answer outside the <think> block.\n"
        "4. Be concise, specific, and quote tool results verbatim when "
        "exactness matters (numbers, prices, URLs, code output)."
    )
    clean_messages = _clean_client_messages(messages, system=system)
    if tool_context:
        clean_messages.insert(
            1,
            {
                "role": "system",
                "content": (
                    "Authoritative runtime tool results — treat these as ground "
                    "truth and integrate them naturally into your answer; do not "
                    "contradict them or claim you cannot use them.\n\n"
                    + "\n\n".join(tool_context)
                ),
            },
        )
    vllm_body = {
        "model": CHAT_POD_SERVED_MODEL,
        "messages": clean_messages,
        "stream": stream,
        "max_tokens": body.get("max_tokens") or 700,
        "temperature": body.get("temperature", 0.6),
        "top_p": body.get("top_p", 0.9),
        "frequency_penalty": body.get("frequency_penalty", 0.0),
        "presence_penalty": body.get("presence_penalty", 0.0),
        "repetition_penalty": body.get("repetition_penalty", 1.0),
        "chat_template_kwargs": {"thinking": True, "enable_thinking": True},
    }
    return vllm_body, tool_context, runtime_trace, direct_answer


async def _orchestrated_chat_completion(body: dict, king_uid: int | None, king_model: str | None) -> dict:
    """Transparent tool runtime around vLLM (do not rewrite the model's text)."""
    vllm_body, tool_context, runtime_trace, direct_answer = await _prepare_orchestrated_chat(
        body, king_uid, king_model, stream=False,
    )
    pod_unavailable_msg: str | None = None
    try:
        raw = await _local_chat_post(vllm_body, timeout=120.0)
        msg = (raw.get("choices") or [{}])[0].get("message") or {}
        content = msg.get("content") or ""
        model_reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
        usage = raw.get("usage")
    except _ChatPodUnavailable as exc:
        pod_unavailable_msg = f"chat server unavailable: {str(exc)[:200]}"
        content = ""
        model_reasoning = ""
        usage = None

    # Promote inline <think> blocks into the reasoning channel so Open WebUI
    # renders the model's chain of thought instead of literal tags.
    content, think_reasoning = _split_think_blocks(content)
    if think_reasoning:
        model_reasoning = (
            (model_reasoning + "\n\n" + think_reasoning).strip()
            if model_reasoning else think_reasoning
        )

    used_direct_fallback = False
    if pod_unavailable_msg and direct_answer is not None:
        content = direct_answer
        runtime_trace.append("Chat pod unavailable; served deterministic tool result.")
        used_direct_fallback = True
    elif pod_unavailable_msg:
        # Re-raise so chat_with_king / openai_chat_completions map
        # this to a documented 503 — never stuff the error into the
        # assistant's content (which would 200 OK with the error
        # surfaced as if the model said it).
        raise _ChatPodUnavailable(
            pod_unavailable_msg.removeprefix("chat server unavailable: ")
            or "pod unreachable"
        )
    elif direct_answer is not None and _looks_empty_or_derailed(content):
        content = direct_answer
        runtime_trace.append(
            "Model returned no usable answer; served deterministic tool result instead."
        )
        used_direct_fallback = True
    elif direct_answer is not None and not _model_quoted_numeric_answer(direct_answer, content):
        content = direct_answer
        runtime_trace.append(
            "Model did not quote the deterministic tool result; served the runtime result instead."
        )
        used_direct_fallback = True
    now = int(time.time())
    response_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    usage = usage or {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }
    message = {
        "role": "assistant",
        "content": content,
        "tool_calls": [],
    }
    if model_reasoning:
        reasoning = _tool_reasoning(runtime_trace, model_reasoning)
        message["reasoning"] = reasoning
        message["reasoning_content"] = reasoning
    elif runtime_trace:
        reasoning = _tool_reasoning(runtime_trace)
        message["reasoning"] = reasoning
        message["reasoning_content"] = reasoning
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": now,
        "model": king_model or CHAT_POD_SERVED_MODEL,
        "king_uid": king_uid,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
    }


def _openai_sse_chunk(base: dict, delta: dict, *, finish_reason=None) -> str:
    return "data: " + json.dumps({
        **base,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    }) + "\n\n"


def _delta_reasoning(delta: dict, msg: dict | None = None) -> str:
    msg = msg or {}
    return (
        delta.get("reasoning")
        or delta.get("reasoning_content")
        or delta.get("thinking")
        or msg.get("reasoning")
        or msg.get("reasoning_content")
        or msg.get("thinking")
        or ""
    )


def _stream_orchestrated_openai_response(body: dict, king_uid: int | None, king_model: str | None):
    """True SSE streaming: tool output first, then vLLM tokens as deltas."""
    async def generate():
        response_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
        created = int(time.time())
        base = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": king_model or CHAT_POD_SERVED_MODEL,
            "king_uid": king_uid,
        }
        acc = ""
        reasoning_acc: list[str] = []
        finish_sent = False
        vllm_body: dict | None = None

        try:
            vllm_body, tool_context, runtime_trace, direct_answer = await _prepare_orchestrated_chat(
                body, king_uid, king_model, stream=True,
            )
        except Exception as exc:
            yield "data: " + json.dumps({
                "error": {"message": f"chat orchestration failed: {str(exc)[:200]}"},
            }) + "\n\n"
            yield "data: [DONE]\n\n"
            return

        yield _openai_sse_chunk(base, {"role": "assistant"})
        trace = _tool_reasoning(runtime_trace)
        if trace:
            reasoning_acc.append(trace)
            yield _openai_sse_chunk(
                base,
                {"reasoning": trace, "reasoning_content": trace},
            )

        think_state = _ThinkStreamSplitter()
        client = _get_http_client()
        pod_failed = False
        buffer_model_content = direct_answer is not None
        try:
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json=_normalize_chat_payload(vllm_body),
                timeout=httpx.Timeout(connect=3.0, read=300.0, write=10.0, pool=5.0),
            ) as resp:
                if resp.status_code >= 400:
                    detail = (await resp.aread()).decode("utf-8", "replace")[:300]
                    pod_failed = True
                    if direct_answer is None:
                        yield "data: " + json.dumps({
                            "error": {
                                "message": f"chat server returned {resp.status_code}",
                                "detail": detail,
                            },
                        }) + "\n\n"
                        yield "data: [DONE]\n\n"
                        return
                else:
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        line = line.strip()
                        if not line.startswith("data: "):
                            continue
                        raw = line[6:]
                        if raw == "[DONE]":
                            break
                        try:
                            parsed = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        choice = (parsed.get("choices") or [{}])[0]
                        delta = choice.get("delta") or {}
                        msg = choice.get("message") or {}
                        out_delta = {}
                        content = delta.get("content") or msg.get("content") or ""
                        if content:
                            visible, reasoning_chunk = think_state.feed(content)
                            if reasoning_chunk:
                                reasoning_acc.append(reasoning_chunk)
                                out_delta["reasoning"] = reasoning_chunk
                                out_delta["reasoning_content"] = reasoning_chunk
                            if visible:
                                acc += visible
                                if not buffer_model_content:
                                    out_delta["content"] = visible
                        reasoning = _delta_reasoning(delta, msg)
                        if reasoning:
                            reasoning_acc.append(reasoning)
                            out_delta["reasoning"] = (
                                (out_delta.get("reasoning", "") + reasoning) if out_delta.get("reasoning") else reasoning
                            )
                            out_delta["reasoning_content"] = out_delta["reasoning"]
                        if delta.get("tool_calls"):
                            out_delta["tool_calls"] = delta.get("tool_calls")
                        finish_reason = choice.get("finish_reason")
                        if out_delta:
                            yield _openai_sse_chunk(base, out_delta)
                        if finish_reason and not pod_failed:
                            visible_tail, reasoning_tail = think_state.flush()
                            if reasoning_tail:
                                reasoning_acc.append(reasoning_tail)
                                yield _openai_sse_chunk(base, {"reasoning": reasoning_tail, "reasoning_content": reasoning_tail})
                            if visible_tail:
                                acc += visible_tail
                                if not buffer_model_content:
                                    yield _openai_sse_chunk(base, {"content": visible_tail})
                            if not buffer_model_content:
                                yield _openai_sse_chunk(base, {}, finish_reason=finish_reason)
                                finish_sent = True
        except (httpx.ConnectError, httpx.ConnectTimeout):
            pod_failed = True
            if direct_answer is None:
                yield 'data: {"error": {"message": "chat server unavailable"}}\n\n'
                yield "data: [DONE]\n\n"
                return
        except Exception as exc:
            pod_failed = True
            if direct_answer is None:
                yield "data: " + json.dumps({
                    "error": {"message": f"stream interrupted: {str(exc)[:200]}"},
                }) + "\n\n"
                yield "data: [DONE]\n\n"
                return

        # Final fallback: model returned nothing usable, refused, or hallucinated
        # past a deterministic numeric result.
        needs_fallback = direct_answer is not None and (
            pod_failed
            or _looks_empty_or_derailed(acc)
            or not _model_quoted_numeric_answer(direct_answer, acc)
        )
        if needs_fallback:
            visible_tail, reasoning_tail = think_state.flush()
            if reasoning_tail:
                yield _openai_sse_chunk(base, {"reasoning": reasoning_tail, "reasoning_content": reasoning_tail})
            if pod_failed:
                note = "(chat pod unavailable; falling back to deterministic tool result)\n\n"
            elif _looks_empty_or_derailed(acc):
                note = "(model returned no usable answer; falling back to deterministic tool result)\n\n"
            else:
                note = "(model did not quote the deterministic tool result; serving runtime answer)\n\n"
            yield _openai_sse_chunk(base, {"reasoning": note, "reasoning_content": note})
            acc = direct_answer
            yield _openai_sse_chunk(base, {"content": direct_answer}, finish_reason="stop")
            finish_sent = True
        elif not finish_sent:
            visible_tail, reasoning_tail = think_state.flush()
            if reasoning_tail:
                yield _openai_sse_chunk(base, {"reasoning": reasoning_tail, "reasoning_content": reasoning_tail})
            if visible_tail:
                acc += visible_tail
                if not buffer_model_content:
                    yield _openai_sse_chunk(base, {"content": visible_tail})
            if buffer_model_content and acc:
                yield _openai_sse_chunk(base, {"content": acc})
            yield _openai_sse_chunk(base, {}, finish_reason="stop")

        yield "data: [DONE]\n\n"
        try:
            _log_chat_turn(vllm_body or body, acc, king_uid, king_model, {
                "choices": [{
                    "message": {
                        "content": acc,
                        "reasoning": "\n".join(reasoning_acc),
                    }
                }]
            })
        except Exception:
            pass

    return _sse_response(generate())


def _stream_orchestrated_chat_response(payload: dict, king_uid: int | None, king_model: str | None):
    """Streaming variant for the legacy `/api/chat` endpoint."""
    async def generate():
        acc = ""
        vllm_body: dict | None = None
        try:
            vllm_body, tool_context, runtime_trace, direct_answer = await _prepare_orchestrated_chat(
                payload, king_uid, king_model, stream=True,
            )
            trace = _tool_reasoning(runtime_trace)
            if trace:
                yield f"data: {json.dumps({'thinking': trace, 'delta': True})}\n\n"
            think_state = _ThinkStreamSplitter()
            pod_failed = False
            buffer_model_content = direct_answer is not None
            client = _get_http_client()
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json=_normalize_chat_payload(vllm_body),
                timeout=httpx.Timeout(connect=3.0, read=300.0, write=10.0, pool=5.0),
            ) as resp:
                if resp.status_code >= 400:
                    pod_failed = True
                    if direct_answer is None:
                        yield f"data: {json.dumps({'error': f'chat server returned {resp.status_code}'})}\n\n"
                        return
                else:
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        line = line.strip()
                        if not line.startswith("data: "):
                            continue
                        raw = line[6:]
                        if raw == "[DONE]":
                            break
                        try:
                            parsed = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        choice = (parsed.get("choices") or [{}])[0]
                        delta = choice.get("delta") or {}
                        msg = choice.get("message") or {}
                        reasoning = _delta_reasoning(delta, msg)
                        if reasoning:
                            yield f"data: {json.dumps({'thinking': reasoning, 'delta': True})}\n\n"
                        content = delta.get("content") or msg.get("content") or ""
                        if content:
                            visible, reasoning_chunk = think_state.feed(content)
                            if reasoning_chunk:
                                yield f"data: {json.dumps({'thinking': reasoning_chunk, 'delta': True})}\n\n"
                            if visible:
                                acc += visible
                                if not buffer_model_content:
                                    yield f"data: {json.dumps({'response': visible, 'delta': True, 'king_uid': king_uid, 'king_model': king_model})}\n\n"
                    visible_tail, reasoning_tail = think_state.flush()
                    if reasoning_tail:
                        yield f"data: {json.dumps({'thinking': reasoning_tail, 'delta': True})}\n\n"
                    if visible_tail:
                        acc += visible_tail
                        if not buffer_model_content:
                            yield f"data: {json.dumps({'response': visible_tail, 'delta': True, 'king_uid': king_uid, 'king_model': king_model})}\n\n"

            needs_fallback = direct_answer is not None and (
                pod_failed
                or _looks_empty_or_derailed(acc)
                or not _model_quoted_numeric_answer(direct_answer, acc)
            )
            if needs_fallback:
                if pod_failed:
                    note = "(chat pod unavailable; falling back to deterministic tool result)"
                elif _looks_empty_or_derailed(acc):
                    note = "(model returned no usable answer; falling back to deterministic tool result)"
                else:
                    note = "(model did not quote the deterministic tool result; serving runtime answer)"
                yield f"data: {json.dumps({'thinking': note, 'delta': True})}\n\n"
                acc = direct_answer
                yield f"data: {json.dumps({'response': direct_answer, 'delta': True, 'king_uid': king_uid, 'king_model': king_model})}\n\n"
            elif buffer_model_content and acc:
                yield f"data: {json.dumps({'response': acc, 'delta': True, 'king_uid': king_uid, 'king_model': king_model})}\n\n"
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            if direct_answer is not None:
                yield f"data: {json.dumps({'thinking': 'chat pod unavailable; falling back to deterministic tool result', 'delta': True})}\n\n"
                acc = direct_answer
                yield f"data: {json.dumps({'response': direct_answer, 'delta': True, 'king_uid': king_uid, 'king_model': king_model})}\n\n"
            else:
                yield f"data: {json.dumps({'error': 'chat server unavailable', 'detail': str(exc)[:200]})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)[:200]})}\n\n"
        finally:
            yield "data: [DONE]\n\n"
            try:
                _log_chat_turn(vllm_body or payload, acc, king_uid, king_model, None)
            except Exception:
                pass

    return _sse_response(generate())


# ── Local chat helpers ───────────────────────────────────────────────────────

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


# ── Chat helpers ──────────────────────────────────────────────────────────────

def _extract_message_content(message: dict) -> tuple[str, str | None]:
    """Pull (content, thinking) from a vLLM choices[0].message."""
    content = message.get("content") or ""
    thinking = message.get("reasoning") or message.get("thinking")
    if not content and thinking:
        content = thinking
        thinking = None
    return content, thinking


async def _sync_chat(payload, king_uid, king_model):
    payload["stream"] = False
    if _CHAT_ORCHESTRATION_ENABLED:
        data = await _orchestrated_chat_completion(payload, king_uid, king_model)
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
    payload["stream"] = True
    if _CHAT_ORCHESTRATION_ENABLED:
        return _stream_orchestrated_chat_response(payload, king_uid, king_model)
    norm = _normalize_chat_payload(payload)

    async def generate():
        # Streaming via httpx async; forward every SSE delta as-is,
        # accumulate ``acc`` only for the chat_turns.jsonl audit log.
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


# Per-turn audit log (JSONL): metadata + 200-char preview only.
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
    max_tokens = body.get("max_tokens", 4096)
    try:
        max_tokens = min(int(max_tokens), 6144)
    except (TypeError, ValueError):
        max_tokens = 4096
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
        max_tokens = 4096
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

    stream = body.get("stream", False)
    try:
        if _CHAT_ORCHESTRATION_ENABLED:
            if stream:
                return _stream_orchestrated_openai_response(body, king_uid, king_model)
            data = await _orchestrated_chat_completion(body, king_uid, king_model)
            return JSONResponse(content=data)

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
