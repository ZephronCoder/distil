"""Chat-pod state administration.

Single source of truth for the king-serving Lium pod coordinates lives in
``state/chat_pod.json``. This module exposes:

* ``read_chat_pod_state()`` — programmatic access used by the validator's
  side-effects loop and the API.
* ``write_chat_pod_state()`` — atomic write used whenever ``side_effects``
  re-deploys the king to a new pod.
* A CLI (``python -m scripts.validator.chat_pod_admin``) for manual ops.

Why a dedicated module:
    Pre-2026-04-26, the chat pod host was wired through ``CHAT_POD_HOST`` env,
    set by hand in the systemd unit. Lium reprovisioning wiped the host
    every few days and ``chat.arbos.life`` went dark until ops noticed.
    The state file + this admin layer make the rotation a one-line
    operation that also nudges the systemd ``chat-tunnel.path`` watcher to
    rebind the SSH tunnel without a service restart.

CLI usage::

    python -m scripts.validator.chat_pod_admin get
    python -m scripts.validator.chat_pod_admin set \
        --host 213.13.7.110 --ssh-port 6039 --app-port 8100 \
        --ssh-key /root/.ssh/eval_pod_key --model ivangrapher/distilman3
    python -m scripts.validator.chat_pod_admin clear
    python -m scripts.validator.chat_pod_admin heal           # re-launch chat_server.py
    python -m scripts.validator.chat_pod_admin probe          # ssh-curl /v1/models
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

# Resolved fresh on every import / call so the CLI works even when STATE_DIR
# is overridden from the shell (tests, multi-validator hosts).
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
    """Merge ``updates`` into the persisted chat-pod state.

    Atomic via tmp-file + rename so the systemd path watcher only fires once
    per change. Touch is OK if no fields changed (callers can detect via
    ``updated_at`` skew if they need to throttle).
    """
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    state = read_chat_pod_state()
    state.update({k: v for k, v in updates.items() if v is not None})
    state["updated_at"] = time.time()
    state["updated_by"] = source
    atomic_json_write(path, state, indent=2)
    return state


def clear_chat_pod_state(*, source: str = "validator") -> None:
    """Mark the chat pod as undefined.

    The wrapper script + systemd unit treat an empty ``host`` as 'no chat
    pod available' and exit cleanly; healthchecks then report the chat
    surface as unavailable instead of pounding a dead host.
    """
    write_chat_pod_state({"host": "", "ssh_port": 0, "model": ""}, source=source)


# Tolerate host-key rotation. Lium pods sometimes get reprovisioned with the
# same IP+port but a fresh host key; without ``UserKnownHostsFile=/dev/null``
# the keeper hits ``REMOTE HOST IDENTIFICATION HAS CHANGED`` and refuses to
# connect even with ``StrictHostKeyChecking=no``. The chat-tunnel already
# ignored known_hosts; this brings probe/heal in line so chat.arbos.life
# survives a rebuild without manual ``ssh-keygen -R``.
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
    """Probe the chat pod for vLLM health + the served model name.

    Returns ``{"ok": bool, "model": str|None, "raw": str|None, "error": str|None}``.
    Used by the validator self-heal loop and the admin CLI.
    """
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

    2026-05-09: when chat moved off the eval pod onto a dedicated GPU
    (``provision_chat_pod.py`` flow), the chat_server.py defaults — tuned
    for co-location at ``CHAT_VLLM_GPU_UTIL=0.30`` — left ~70% of the
    chat-pod GPU idle and capped the king's KV cache short. We now allow
    chat_pod.json to declare an ``env`` dict (e.g. ``{"CHAT_VLLM_GPU_UTIL":
    "0.85", "CHAT_VLLM_MAX_MODEL_LEN": "32768"}``) which is exported into
    the heal-launch shell. Keys are validated against
    ``[A-Z_][A-Z0-9_]*`` so a fat-fingered chat_pod.json can't smuggle
    ``LD_PRELOAD`` or ``rm -rf /`` past the export. Values are
    single-quoted and any literal single-quote is escaped via the bash
    ``'"'"'`` idiom — the only way to embed ``'`` inside a single-quoted
    bash string.
    """
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
    """Return ``(in_progress, reason)`` for an existing chat_server.py on the pod.

    A chat_server.py process is "downloading" if:

    1. ``pgrep -fa chat_server.py`` finds a python interpreter, AND
    2. ``/root/chat_server.log`` was modified within ``grace_sec`` seconds, AND
    3. The log's tail does NOT contain a fatal traceback (so we can
       distinguish an active download from a recently-crashed one).

    2026-05-09 motivation: chat-keeper.timer ticks every 3 min. Before
    this guard, a 5-10 min HF model download would be killed every 3
    min by a fresh ``heal`` invocation, restarting from 0 % and never
    finishing. Result: chat stayed dark for the entire model swap. The
    validator's ``sync_king_runtime`` exhibits the same bug whenever
    a king's model isn't in MODEL_DIR yet. We now noop when the
    bootstrapper is making progress on its own.
    """
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
    """Sync ``chat_server.py`` + relaunch vLLM on the configured chat pod.

    If ``model_name`` is omitted, falls back to the last persisted model. We
    don't pick up the live king from h2h_latest.json here on purpose: the
    validator's side-effects loop is the authority for who's reigning, and
    the CLI should be a manual override that doesn't second-guess it.

    If ``chat_pod.json`` contains an ``env`` dict, those ``KEY=VALUE``
    pairs are exported into the heal launch shell so per-pod tuning
    (e.g. ``CHAT_VLLM_GPU_UTIL`` for a dedicated chat pod) propagates
    without forking the script.

    If a chat_server.py is already running with a recently-touched log
    and no traceback (i.e. it's making download/load progress), this
    function noop's so chat-keeper's 3-min ticks can't kill an
    in-flight 5-10 min model pull. Pass ``force=True`` to override
    (used by the king-rotation path which DOES need to interrupt
    a stale download to switch model).
    """
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
    # 2026-05-09: chat_server.py invokes vLLM with
    # ``--reasoning-parser distil_kimi``. That parser is registered by
    # ``_install_custom_reasoning_parser`` in chat_server.py — but
    # ONLY if the parser source file exists at
    # ``/root/distil_kimi_reasoning_parser.py`` when chat_server boots.
    # On a freshly-provisioned pod the file isn't there, vLLM then dies
    # with ``KeyError: Reasoning parser 'distil_kimi' not found.`` Sync
    # it alongside chat_server.py so heal is self-contained on a fresh
    # pod.
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
    # Single SSH does the kill / launch / probe so we don't get caught by
    # split-brain (kill-but-no-launch) if the second SSH fails. The kill is
    # broad because vLLM v1 spawns a child that renames itself to
    # ``VLLM::EngineCore`` and holds the GPU; only killing chat_server.py
    # leaves the engine running and the next launch starves on memory.
    # ``pkill -f`` matches against the *entire* command line, including its own
    # ``bash -c`` invocation. A bare pattern like ``chat_server.py`` would
    # match the SSH shell that's running this kill (because the pattern itself
    # appears in argv), self-terminate, and SSH returns rc=255 with empty
    # stderr — the chat king then never gets relaunched. Anchoring on
    # ``^python`` filters to the actual python interpreter holding the model.
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
