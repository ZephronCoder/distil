"""Chat-pod state administration.

``state/chat_pod.json`` is the single source of truth for the king-serving
Lium pod. This module exposes read/write/clear helpers plus a CLI
(``python -m scripts.validator.chat_pod_admin {get,set,clear,heal,probe}``).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from eval.state import atomic_json_write

logger = logging.getLogger("distillation.chat_pod_admin")

# Resolved fresh on every call so DISTIL_STATE_DIR overrides take effect.
def _state_dir() -> Path:
    return Path(
        os.environ.get("DISTIL_STATE_DIR")
        or os.environ.get("DISTIL_REPO_ROOT", "/opt/distil/repo") + "/state"
    )


def _state_path() -> Path:
    return _state_dir() / "chat_pod.json"


def read_chat_pod_state() -> dict[str, Any]:
    path = _state_path()
    try:
        with open(path) as f:
            data = json.load(f) or {}
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, ValueError, OSError):
        return {}


def write_chat_pod_state(updates: dict[str, Any], *, source: str = "validator") -> dict[str, Any]:
    """Atomic merge of ``updates`` into the persisted chat-pod state."""
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    state = read_chat_pod_state()
    state.update({k: v for k, v in updates.items() if v is not None})
    state["updated_at"] = time.time()
    state["updated_by"] = source
    atomic_json_write(path, state, indent=2)
    return state


def clear_chat_pod_state(*, source: str = "validator") -> None:
    """Mark the chat pod as undefined (empty host == 'no chat pod')."""
    write_chat_pod_state({"host": "", "ssh_port": 0, "model": ""}, source=source)


# Tolerate host-key rotation across Lium pod reprovisions.
_BASE_SSH_OPTS = [
    "-o", "ConnectTimeout=10",
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "GlobalKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
    "-o", "BatchMode=yes",
]


def _ssh_args(state: dict[str, Any]) -> list[str]:
    host = state.get("host") or ""
    port = int(state.get("ssh_port") or 0)
    key = state.get("ssh_key") or os.path.expanduser("~/.ssh/id_ed25519")
    if not host or not port:
        raise RuntimeError("chat pod is not configured (host/ssh_port missing)")
    return [
        "ssh",
        *_BASE_SSH_OPTS,
        "-i", key,
        "-p", str(port),
        f"root@{host}",
    ]


def _scp_args(state: dict[str, Any], src: str, dst: str) -> list[str]:
    host = state.get("host") or ""
    port = int(state.get("ssh_port") or 0)
    key = state.get("ssh_key") or os.path.expanduser("~/.ssh/id_ed25519")
    if not host or not port:
        raise RuntimeError("chat pod is not configured (host/ssh_port missing)")
    return [
        "scp",
        *_BASE_SSH_OPTS,
        "-i", key,
        "-P", str(port),
        src,
        f"root@{host}:{dst}",
    ]


def probe(state: dict[str, Any] | None = None, timeout: int = 12) -> dict[str, Any]:
    """SSH-curl /v1/models; returns ok/model/raw/error dict."""
    state = state or read_chat_pod_state()
    if not state.get("host") or not state.get("ssh_port"):
        return {"ok": False, "model": None, "raw": None, "error": "no chat pod configured"}
    app_port = int(state.get("app_port") or 8100)
    cmd = (
        f"curl -fsS http://localhost:{app_port}/v1/models "
        "&& echo '---' "
        "&& cat /root/model_name.txt 2>/dev/null"
    )
    try:
        out = subprocess.run(
            _ssh_args(state) + [cmd],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"ok": False, "model": None, "raw": None, "error": str(exc)}
    if out.returncode != 0 and "---" not in out.stdout:
        err = (out.stderr or out.stdout or "").strip()[:240]
        return {"ok": False, "model": None, "raw": out.stdout[:240], "error": err}
    parts = out.stdout.split("---", 1)
    head = parts[0].strip()
    model = parts[1].strip() if len(parts) > 1 else ""
    if "data" not in head:
        return {"ok": False, "model": model or None, "raw": head[:240], "error": "no /v1/models response"}
    return {"ok": True, "model": model or None, "raw": head[:240], "error": None}


_ENV_KEY_RE = __import__("re").compile(r"^[A-Z_][A-Z0-9_]*$")


def _format_env_exports(env: dict[str, Any] | None) -> str:
    """Return ``KEY=VALUE`` exports prefix for a remote bash command.
    Keys must match ``[A-Z_][A-Z0-9_]*``; values are single-quoted."""
    if not env:
        return ""
    parts: list[str] = []
    for k, v in env.items():
        key = str(k).strip()
        if not _ENV_KEY_RE.match(key):
            logger.warning(f"chat_pod env: ignoring invalid key {k!r}")
            continue
        if v is None:
            continue
        val = str(v).replace("'", "'\"'\"'")
        parts.append(f"export {key}='{val}'")
    return ("; ".join(parts) + "; ") if parts else ""


def _is_download_in_progress(state: dict[str, Any], grace_sec: int = 90) -> tuple[bool, str]:
    """Return (in_progress, reason). True if chat_server.py is running,
    log is fresh (<= grace_sec), and tail has no traceback."""
    if not state.get("host") or not state.get("ssh_port"):
        return False, "pod not configured"
    probe_cmd = (
        "set -u; "
        "ts=$(stat -c %Y /root/chat_server.log 2>/dev/null || echo 0); "
        "now=$(date +%s); "
        "age=$((now - ts)); "
        "running=$(pgrep -fa '^python.* /root/chat_server.py' | grep -v pgrep | wc -l); "
        "tail_lines=$(tail -n 8 /root/chat_server.log 2>/dev/null | tr -d '\\0'); "
        "echo \"age=$age running=$running\"; "
        "echo '---'; "
        "echo \"$tail_lines\""
    )
    try:
        out = subprocess.run(
            _ssh_args(state) + [probe_cmd],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return False, f"probe error: {exc}"
    if out.returncode != 0:
        return False, f"probe rc={out.returncode}"
    head, _, tail = out.stdout.partition("---")
    age = -1
    running = 0
    for tok in head.split():
        if tok.startswith("age="):
            try:
                age = int(tok.split("=", 1)[1])
            except ValueError:
                age = -1
        elif tok.startswith("running="):
            try:
                running = int(tok.split("=", 1)[1])
            except ValueError:
                running = 0
    tail_low = tail.lower()
    if "traceback" in tail_low or "calledprocesserror" in tail_low or "errno" in tail_low:
        return False, f"chat_server.py crashed (running={running}, age={age}s)"
    if running >= 1 and 0 <= age <= grace_sec:
        return True, f"running={running}, log_age={age}s"
    return False, f"running={running}, log_age={age}s"


def heal(model_name: str | None = None, *, source: str = "cli", force: bool = False) -> dict[str, Any]:
    """Sync chat_server.py + relaunch vLLM on the configured chat pod.

    Falls back to the last persisted model if ``model_name`` is None.
    Noop's if a chat_server.py is making progress (unless ``force=True``).
    Exports any ``env`` dict from chat_pod.json into the launch shell."""
    state = read_chat_pod_state()
    if not state.get("host"):
        raise RuntimeError("chat pod is not configured; run `set --host ... --ssh-port ...` first")
    target = model_name or state.get("model")
    if not target:
        raise RuntimeError("no model_name provided and state has no persisted model")

    if not force:
        in_flight, why = _is_download_in_progress(state)
        if in_flight:
            logger.info(f"heal skipped: chat_server.py is making progress ({why})")
            return {
                "model": target,
                "host": state.get("host"),
                "ssh_port": state.get("ssh_port"),
                "skipped": True,
                "reason": why,
            }

    repo_root = Path(os.environ.get("DISTIL_REPO_ROOT", "/opt/distil/repo"))
    chat_src = repo_root / "scripts" / "chat_pod" / "chat_server.py"
    if not chat_src.exists():
        raise RuntimeError(f"chat_server.py source missing at {chat_src}")

    subprocess.run(
        _scp_args(state, str(chat_src), "/root/chat_server.py"),
        check=True, capture_output=True, text=True, timeout=30,
    )
    # Sync distil_kimi parser so a fresh pod has it before vLLM boots.
    parser_src = repo_root / "scripts" / "chat_pod" / "distil_kimi_reasoning_parser.py"
    if parser_src.exists():
        try:
            subprocess.run(
                _scp_args(state, str(parser_src), "/root/distil_kimi_reasoning_parser.py"),
                check=True, capture_output=True, text=True, timeout=30,
            )
        except subprocess.CalledProcessError as exc:
            logger.warning(
                "scp of distil_kimi_reasoning_parser.py failed: "
                f"{(exc.stderr or exc.stdout)[:200]}"
            )
    else:
        logger.warning(f"distil_kimi_reasoning_parser.py not found at {parser_src}; skipping sync")
    app_port = int(state.get("app_port") or 8100)
    env_prefix = _format_env_exports(state.get("env") if isinstance(state.get("env"), dict) else None)
    # Single SSH kills the VLLM::EngineCore worker + chat_server.py,
    # then relaunches. ``^python`` anchor avoids the kill matching itself.
    cmd = (
        "set -e; "
        + env_prefix
        + "pkill -9 -f '^python.* /root/chat_server.py' 2>/dev/null || true; "
        "pkill -9 -f '^python.* vllm.entrypoints.openai.api_server' 2>/dev/null || true; "
        "pkill -9 -x 'VLLM::EngineCore' 2>/dev/null || true; "
        "pkill -9 -x 'VLLM::EngineCor' 2>/dev/null || true; "
        "sleep 2; "
        f"( nohup python3 -u /root/chat_server.py {target!r} {app_port} "
        f"  >> /root/chat_server.log 2>&1 < /dev/null & ); "
        "sleep 1; pgrep -fa '^python.* /root/chat_server.py' | head -3"
    )
    out = subprocess.run(
        _ssh_args(state) + [cmd],
        capture_output=True, text=True, timeout=30, check=False,
    )
    write_chat_pod_state({"model": target}, source=source)
    return {
        "model": target,
        "host": state.get("host"),
        "ssh_port": state.get("ssh_port"),
        "ssh_stdout": out.stdout,
        "ssh_stderr": out.stderr,
        "ssh_rc": out.returncode,
    }


def _cmd_get(_args: argparse.Namespace) -> int:
    state = read_chat_pod_state()
    if not state:
        print("chat pod is not configured", file=sys.stderr)
        return 1
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


def _cmd_set(args: argparse.Namespace) -> int:
    updates: dict[str, Any] = {}
    if args.host is not None:
        updates["host"] = args.host
    if args.ssh_port is not None:
        updates["ssh_port"] = int(args.ssh_port)
    if args.app_port is not None:
        updates["app_port"] = int(args.app_port)
    if args.ssh_key is not None:
        updates["ssh_key"] = args.ssh_key
    if args.model is not None:
        updates["model"] = args.model
    if args.note is not None:
        updates["note"] = args.note
    if not updates:
        print("nothing to set; pass at least one of --host/--ssh-port/...", file=sys.stderr)
        return 2
    state = write_chat_pod_state(updates, source="cli")
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


def _cmd_clear(_args: argparse.Namespace) -> int:
    clear_chat_pod_state(source="cli")
    print("chat pod state cleared")
    return 0


def _cmd_heal(args: argparse.Namespace) -> int:
    try:
        result = heal(model_name=args.model, force=getattr(args, "force", False))
    except Exception as exc:  # noqa: BLE001
        print(f"heal failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_probe(_args: argparse.Namespace) -> int:
    result = probe()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chat_pod_admin", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("get", help="print persisted chat pod state").set_defaults(fn=_cmd_get)

    p_set = sub.add_parser("set", help="update chat pod state (atomic merge)")
    p_set.add_argument("--host", help="chat pod public IP / hostname")
    p_set.add_argument("--ssh-port", type=int, help="chat pod SSH port")
    p_set.add_argument("--app-port", type=int, help="chat pod vLLM port (default 8100)")
    p_set.add_argument("--ssh-key", help="path to SSH private key")
    p_set.add_argument("--model", help="HF repo id of the model currently served")
    p_set.add_argument("--note", help="free-form note for ops handoff")
    p_set.set_defaults(fn=_cmd_set)

    sub.add_parser("clear", help="mark chat pod as undefined").set_defaults(fn=_cmd_clear)

    p_heal = sub.add_parser("heal", help="rsync + relaunch chat_server.py on the pod")
    p_heal.add_argument("--model", help="HF repo id; defaults to persisted model")
    p_heal.add_argument(
        "--force",
        action="store_true",
        help="kill+restart even if a chat_server.py is already making "
        "download/load progress (default: noop in that case)",
    )
    p_heal.set_defaults(fn=_cmd_heal)

    sub.add_parser("probe", help="ssh-curl /v1/models").set_defaults(fn=_cmd_probe)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
