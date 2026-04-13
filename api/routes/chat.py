"""Chat endpoints: proxy to king model on GPU pod, OpenAI-compatible endpoints."""

import json
import os
import subprocess
import threading
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..config import STATE_DIR, DISK_CACHE_DIR, CHAT_POD_PORT, CHAT_POD_HOST, CHAT_POD_SSH_PORT, CHAT_POD_SSH_KEY
from ..helpers.sanitize import _safe_json_load
from ..helpers.ssh import _ssh_exec
from ..helpers.rate_limit import _chat_rate_limiter

router = APIRouter()


# ── King info helper ──────────────────────────────────────────────────────────

def _get_king_info():
    """Get king UID and model name."""
    h2h = _safe_json_load(os.path.join(STATE_DIR, "h2h_latest.json"), {})
    king_uid = h2h.get("king_uid")
    if king_uid is None:
        return None, None

    # Try h2h results first (has model directly)
    for r in h2h.get("results", []):
        if r.get("is_king") or r.get("uid") == king_uid:
            return king_uid, r.get("model")

    # Fallback: commitments cache (hotkey-keyed, need metagraph for UID→hotkey)
    metagraph = _safe_json_load(os.path.join(DISK_CACHE_DIR, "metagraph.json"), {})
    commitments_data = _safe_json_load(os.path.join(DISK_CACHE_DIR, "commitments.json"), {})
    commitments = commitments_data.get("commitments", commitments_data) if isinstance(commitments_data, dict) else {}

    # Build UID→hotkey from metagraph
    uids = metagraph.get("uids", [])
    hotkeys = metagraph.get("hotkeys", [])
    king_hotkey = None
    for i, uid in enumerate(uids):
        if uid == king_uid and i < len(hotkeys):
            king_hotkey = hotkeys[i]
            break

    if king_hotkey and king_hotkey in commitments:
        info = commitments[king_hotkey]
        return king_uid, info.get("model") if isinstance(info, dict) else info

    return king_uid, None


# ── Chat helpers ──────────────────────────────────────────────────────────────

def _sync_chat(payload, king_uid, king_model):
    """Non-streaming chat proxy via SSH + curl."""
    import base64
    payload["stream"] = False
    payload_b64 = base64.b64encode(json.dumps(payload).encode()).decode()
    cmd = f"echo '{payload_b64}' | base64 -d | curl -s -X POST http://localhost:{CHAT_POD_PORT}/v1/chat/completions -H 'Content-Type: application/json' -d @-"
    stdout = _ssh_exec(cmd, timeout=60)

    try:
        data = json.loads(stdout)
        if "choices" in data:
            resp = {
                "response": data["choices"][0]["message"]["content"],
                "model": king_model,
                "king_uid": king_uid,
            }
            if "thinking" in data:
                resp["thinking"] = data["thinking"]
            if "usage" in data:
                resp["usage"] = data["usage"]
            return resp
        return {"error": "unexpected response from chat server"}
    except json.JSONDecodeError:
        return {"error": "chat server not responding - may be starting up"}


def _stream_chat(payload, king_uid, king_model):
    """Streaming chat proxy via SSE. SSH + curl -N pipes pod SSE → client."""
    import base64
    payload["stream"] = True
    payload_b64 = base64.b64encode(json.dumps(payload).encode()).decode()
    cmd = f"echo '{payload_b64}' | base64 -d | curl -sN -X POST http://localhost:{CHAT_POD_PORT}/v1/chat/completions -H 'Content-Type: application/json' -d @-"

    def generate():
        try:
            ssh_cmd = [
                "ssh", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no",
                "-i", CHAT_POD_SSH_KEY, "-p", str(CHAT_POD_SSH_PORT),
                f"root@{CHAT_POD_HOST}", cmd,
            ]
            proc = subprocess.Popen(ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            for line in proc.stdout:
                line = line.strip()
                if line.startswith("data: "):
                    raw = line[6:]
                    if raw == "[DONE]":
                        yield "data: [DONE]\n\n"
                        break
                    try:
                        parsed = json.loads(raw)
                        parsed["king_uid"] = king_uid
                        parsed["king_model"] = king_model
                        yield f"data: {json.dumps(parsed)}\n\n"
                    except json.JSONDecodeError:
                        yield f"data: {raw}\n\n"
            proc.wait(timeout=5)
        except Exception as e:
            err = str(e)
            if "ssh" in err.lower() or "root@" in err or ".ssh/" in err:
                err = "chat server connection failed"
            yield f"data: {json.dumps({'error': err[:200]})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


_chat_restart_lock = threading.Lock()
_last_chat_restart = 0.0


def _ensure_chat_server(king_model=None):
    """Auto-start chat server if not running or running wrong model. Rate-limited to once per 2 min."""
    global _last_chat_restart
    with _chat_restart_lock:
        if time.time() - _last_chat_restart < 120:
            return  # Already tried recently
        _last_chat_restart = time.time()

    model_name = king_model or "unknown"
    try:
        stdout = _ssh_exec(f"curl -fsS http://localhost:{CHAT_POD_PORT}/v1/models || echo not_running")
        if "not_running" in stdout:
            print(f"[chat] Auto-starting chat server for {model_name}", flush=True)
            _ssh_exec(f"nohup python3 /root/chat_server.py '{model_name}' {CHAT_POD_PORT} > /root/chat.log 2>&1 &", timeout=10)
        elif model_name != "unknown" and model_name not in stdout:
            # Server running but wrong model - restart
            print(f"[chat] Chat server running wrong model, restarting for {model_name}", flush=True)
            _ssh_exec("pkill -f 'vllm.entrypoints.openai.api_server|chat_server.py' || true", timeout=10)
            import time as _t; _t.sleep(2)
            _ssh_exec(f"nohup python3 /root/chat_server.py '{model_name}' {CHAT_POD_PORT} > /root/chat.log 2>&1 &", timeout=10)
    except Exception as e:
        print(f"[chat] Auto-restart failed: {e}", flush=True)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/api/chat")
async def chat_with_king(request: Request):
    """Proxy chat to the king model running on the GPU pod. Supports streaming via stream=true."""
    # Rate limit: 10 req/min per IP for chat
    client_ip = request.client.host if request.client else "unknown"
    if not _chat_rate_limiter.is_allowed(client_ip):
        return JSONResponse(status_code=429, content={"error": "rate limit exceeded"})

    body = await request.json()
    messages = body.get("messages", [])
    max_tokens = body.get("max_tokens", 8192)
    stream = body.get("stream", False)

    if not messages:
        return {"error": "messages required"}

    # Input validation
    if not isinstance(messages, list) or len(messages) > 50:
        return JSONResponse(status_code=400, content={"error": "messages must be an array with at most 50 entries"})
    for msg in messages:
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        if isinstance(content, str) and len(content) > 10000:
            return JSONResponse(status_code=400, content={"error": "message content too long (max 10000 chars)"})
    if not isinstance(max_tokens, (int, float)) or max_tokens < 1:
        max_tokens = 8192
    temperature = body.get("temperature", 0.7)
    if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2:
        temperature = 0.7
    top_p = body.get("top_p", 0.9)
    if not isinstance(top_p, (int, float)) or top_p < 0 or top_p > 1:
        top_p = 0.9

    king_uid, king_model = _get_king_info()
    if king_uid is None:
        return {"error": "no king model available"}

    try:
        pod_payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }

        if stream:
            return _stream_chat(pod_payload, king_uid, king_model)
        else:
            return _sync_chat(pod_payload, king_uid, king_model)

    except Exception as e:
        # Sanitize error - don't leak SSH commands/paths to frontend
        err = str(e)
        if "ssh" in err.lower() or "root@" in err or ".ssh/" in err:
            return {"error": "chat server connection failed - try again in a moment"}
        return {"error": f"chat error: {err[:200]}"}


@router.get("/api/chat/status")
def chat_status():
    """Check if the king chat server is available. Auto-starts if down."""
    king_uid, king_model = _get_king_info()
    progress = _safe_json_load(os.path.join(STATE_DIR, "eval_progress.json"), {})
    eval_active = progress.get("active", False)

    # Try health check on pod
    server_ok = False
    try:
        stdout = _ssh_exec(f"curl -fsS http://localhost:{CHAT_POD_PORT}/v1/models")
        if stdout and (king_model is None or king_model in stdout):
            server_ok = True
        elif not eval_active:
            # Server not responding and no eval in progress - auto-restart
            _ensure_chat_server(king_model)
    except Exception:
        pass

    return {
        "available": server_ok and king_uid is not None,
        "king_uid": king_uid,
        "king_model": king_model,
        "eval_active": eval_active,
        "server_running": server_ok,
        "note": "King model is loaded on GPU and ready for chat." if server_ok else "Chat server is starting or unavailable.",
    }


# ── OpenAI-compatible endpoints (for Open WebUI etc.) ─────────────────────────


@router.get("/v1/models")
def openai_models():
    """OpenAI-compatible models list. Returns the current king model."""
    king_uid, king_model = _get_king_info()
    model_id = king_model or "distil-king"
    return {
        "object": "list",
        "data": [{
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": f"distil-sn97-uid{king_uid}" if king_uid else "distil-sn97",
        }],
    }


@router.post("/v1/chat/completions")
async def openai_chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint. Proxies to the king model.
    Used by Open WebUI and other OpenAI-compatible clients."""
    client_ip = request.client.host if request.client else "unknown"
    if not _chat_rate_limiter.is_allowed(client_ip):
        return JSONResponse(status_code=429, content={"error": {"message": "rate limit exceeded", "type": "rate_limit_error"}})

    body = await request.json()
    messages = body.get("messages", [])
    if not messages:
        return JSONResponse(status_code=400, content={"error": {"message": "messages required"}})

    # Default to non-thinking chat output for Open WebUI / normal chat UX.
    body.setdefault("chat_template_kwargs", {})
    body["chat_template_kwargs"].setdefault("enable_thinking", False)

    king_uid, king_model = _get_king_info()
    if king_uid is None:
        return JSONResponse(status_code=503, content={"error": {"message": "no king model available"}})

    # Forward the request as-is to the chat server (it already speaks OpenAI format)
    import base64
    stream = body.get("stream", False)
    payload_b64 = base64.b64encode(json.dumps(body).encode()).decode()

    try:
        if stream:
            # Stream SSE directly from chat server
            cmd = f"echo '{payload_b64}' | base64 -d | curl -sN -X POST http://localhost:{CHAT_POD_PORT}/v1/chat/completions -H 'Content-Type: application/json' -d @-"
            ssh_cmd = [
                "ssh", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no",
                "-i", CHAT_POD_SSH_KEY, "-p", str(CHAT_POD_SSH_PORT),
                f"root@{CHAT_POD_HOST}", cmd,
            ]

            def generate():
                try:
                    proc = subprocess.Popen(ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    for line in proc.stdout:
                        line = line.strip()
                        if line.startswith("data: "):
                            yield f"{line}\n\n"
                            if line == "data: [DONE]":
                                break
                    proc.wait(timeout=5)
                except Exception:
                    yield 'data: {"error": "stream interrupted"}\n\n'

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
            )
        else:
            # Non-streaming: return raw OpenAI response
            cmd = f"echo '{payload_b64}' | base64 -d | curl -s -X POST http://localhost:{CHAT_POD_PORT}/v1/chat/completions -H 'Content-Type: application/json' -d @-"
            stdout = _ssh_exec(cmd, timeout=120)
            try:
                data = json.loads(stdout)
                return JSONResponse(content=data)
            except json.JSONDecodeError:
                return JSONResponse(status_code=502, content={"error": {"message": "chat server not responding"}})
    except Exception:
        return JSONResponse(status_code=502, content={"error": {"message": "chat server connection failed"}})
