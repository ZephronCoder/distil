#!/usr/bin/env python3
"""Detect (and optionally repair) regressions in ``/root/.openclaw/openclaw.json``.

Background: on 2026-04-28 someone (or a CLI flow) dropped ``"message"`` from
the ``sn97-bot`` agent's ``tools.allow`` array. The bot kept running but
silently lost the ability to post replies in threads or to fan out king-
crown announcements via the cross-channel ``message`` tool. The drift sat
undetected for ~9 days until users reported the bot had gone quiet in
threads (#distil-97 on 2026-05-07).

This script encodes the invariants every backup ``openclaw.json.bak.*``
shared:

* ``agents.list[id="sn97-bot"].tools.allow`` must contain ``"message"``.
* ``channels.discord.accounts.arbos.guilds["799672011265015819"].channels``
  must include ``"1482026267392868583"`` with ``enabled = true`` and
  ``autoThread = true``.
* ``channels.discord.accounts.arbos.threadBindings.enabled`` must be true
  (without this the bot ignores all thread events).
* ``channels.discord.enabled`` must be true.
* The ``arbos`` account binding must point at the correct guild
  (``"799672011265015819"`` — Bittensor) so we don't accidentally route
  the bot to the wrong server.

Modes:

* ``--check`` (default) — print findings as JSON to stdout, exit 0 on
  ``ok``, exit 2 on ``regression`` (so a systemd timer's ``OnFailure``
  hook can alert).
* ``--repair`` — additionally rewrite the file in place, preserving 0600
  permissions and root ownership, and writing a backup to
  ``openclaw.json.bak.<UTC-timestamp>-guard``. Repair is idempotent: a
  clean file is left untouched.
* ``--quiet`` — suppress stdout when no regression; still write to a
  state file (when ``--state-file`` is given) and exit 0/2 as above.

The script never reads or echoes Discord tokens. The JSON edit only
touches the small invariants listed above.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path("/root/.openclaw/openclaw.json")

EXPECTED_BOT_ID = "sn97-bot"
EXPECTED_TOOL = "message"
EXPECTED_GUILD_ID = "799672011265015819"  # Bittensor guild
EXPECTED_CHANNEL_ID = "1482026267392868583"  # ა・distil・97
EXPECTED_ACCOUNT_ID = "arbos"


def _get(d: Any, *keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def _find_agent(cfg: dict, agent_id: str) -> dict | None:
    agents = _get(cfg, "agents", "list", default=[]) or []
    if not isinstance(agents, list):
        return None
    for entry in agents:
        if isinstance(entry, dict) and entry.get("id") == agent_id:
            return entry
    return None


def audit(cfg: dict) -> dict:
    findings: list[dict] = []
    bot = _find_agent(cfg, EXPECTED_BOT_ID)
    if bot is None:
        findings.append({
            "severity": "regression",
            "code": "missing_agent",
            "detail": f"agents.list does not contain id={EXPECTED_BOT_ID!r}",
        })
        return {"status": "regression", "findings": findings}

    allow = _get(bot, "tools", "allow", default=[]) or []
    if not isinstance(allow, list):
        findings.append({
            "severity": "regression",
            "code": "tools_allow_not_list",
            "detail": f"tools.allow for {EXPECTED_BOT_ID!r} is not a JSON array (got {type(allow).__name__})",
        })
    elif EXPECTED_TOOL not in allow:
        findings.append({
            "severity": "regression",
            "code": "missing_message_tool",
            "detail": (
                f"tools.allow for {EXPECTED_BOT_ID!r} is missing {EXPECTED_TOOL!r}; "
                f"current={allow!r}. Without this the bot cannot post king "
                f"announcements or thread replies via the cross-channel "
                f"`message` tool."
            ),
        })

    discord_enabled = bool(_get(cfg, "channels", "discord", "enabled"))
    if not discord_enabled:
        findings.append({
            "severity": "regression",
            "code": "discord_channel_disabled",
            "detail": "channels.discord.enabled is not true",
        })

    arbos = _get(cfg, "channels", "discord", "accounts", EXPECTED_ACCOUNT_ID, default=None)
    if not isinstance(arbos, dict):
        findings.append({
            "severity": "regression",
            "code": "missing_arbos_account",
            "detail": f"channels.discord.accounts.{EXPECTED_ACCOUNT_ID!r} is missing",
        })
    else:
        guilds = _get(arbos, "guilds", default={}) or {}
        guild = guilds.get(EXPECTED_GUILD_ID) if isinstance(guilds, dict) else None
        if not isinstance(guild, dict):
            findings.append({
                "severity": "regression",
                "code": "missing_guild_binding",
                "detail": f"arbos.guilds is missing {EXPECTED_GUILD_ID!r}",
            })
        else:
            chans = guild.get("channels") if isinstance(guild.get("channels"), dict) else {}
            chan = chans.get(EXPECTED_CHANNEL_ID)
            if not isinstance(chan, dict):
                findings.append({
                    "severity": "regression",
                    "code": "missing_channel_binding",
                    "detail": f"arbos.guilds[{EXPECTED_GUILD_ID!r}].channels is missing {EXPECTED_CHANNEL_ID!r}",
                })
            else:
                if chan.get("enabled") is not True:
                    findings.append({
                        "severity": "regression",
                        "code": "channel_disabled",
                        "detail": f"channel {EXPECTED_CHANNEL_ID!r} is not enabled",
                    })
                if chan.get("autoThread") is not True:
                    findings.append({
                        "severity": "drift",
                        "code": "auto_thread_disabled",
                        "detail": (
                            f"channel {EXPECTED_CHANNEL_ID!r} has autoThread != true; "
                            f"the bot may stop creating threads for new mentions."
                        ),
                    })
        thread_bindings = _get(arbos, "threadBindings", default={}) or {}
        if thread_bindings.get("enabled") is not True:
            findings.append({
                "severity": "regression",
                "code": "thread_bindings_disabled",
                "detail": (
                    f"channels.discord.accounts.{EXPECTED_ACCOUNT_ID!r}."
                    f"threadBindings.enabled is not true — the bot will ignore "
                    f"thread message events entirely."
                ),
            })

    if not findings:
        return {"status": "ok", "findings": []}

    severities = {f["severity"] for f in findings}
    status = "regression" if "regression" in severities else "drift"
    return {"status": status, "findings": findings}


def repair(cfg: dict) -> tuple[dict, list[str]]:
    """Apply the minimum-mutation set of fixes to restore invariants.

    Returns ``(new_cfg, applied_actions)``. Idempotent: a clean cfg
    yields the same object and an empty action list. Never adds tokens
    or any field beyond the documented invariants.
    """
    actions: list[str] = []
    bot = _find_agent(cfg, EXPECTED_BOT_ID)
    if bot is not None:
        tools = bot.setdefault("tools", {})
        if not isinstance(tools, dict):
            return cfg, ["abort:tools_not_dict"]
        allow = tools.get("allow")
        if not isinstance(allow, list):
            tools["allow"] = [EXPECTED_TOOL]
            actions.append(f"reset tools.allow to [{EXPECTED_TOOL!r}]")
        elif EXPECTED_TOOL not in allow:
            allow.append(EXPECTED_TOOL)
            actions.append(f"appended {EXPECTED_TOOL!r} to tools.allow")
    arbos = _get(cfg, "channels", "discord", "accounts", EXPECTED_ACCOUNT_ID, default=None)
    if isinstance(arbos, dict):
        thread_bindings = arbos.get("threadBindings")
        if not isinstance(thread_bindings, dict):
            arbos["threadBindings"] = {"enabled": True, "spawnSubagentSessions": True}
            actions.append("created arbos.threadBindings")
        elif thread_bindings.get("enabled") is not True:
            thread_bindings["enabled"] = True
            actions.append("set arbos.threadBindings.enabled = true")
    return cfg, actions


def _atomic_write(path: Path, payload: str, *, mode: int = 0o600) -> None:
    tmp = path.with_suffix(path.suffix + ".guard.tmp")
    tmp.write_text(payload)
    try:
        os.chmod(tmp, mode)
    except OSError:
        pass
    os.replace(tmp, path)
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def _backup(path: Path) -> Path:
    stamp = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bak = path.with_name(f"{path.name}.bak.{stamp}-guard")
    bak.write_bytes(path.read_bytes())
    try:
        os.chmod(bak, 0o600)
    except OSError:
        pass
    return bak


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--repair", action="store_true",
                        help="Apply the minimal repairs in place (with backup).")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress stdout when no regression; still exits 2 on regression.")
    parser.add_argument("--state-file", type=Path, default=None,
                        help="Optional JSON path to write the latest health snapshot.")
    args = parser.parse_args(argv)

    if not args.config.exists():
        report = {
            "status": "missing",
            "config": str(args.config),
            "checked_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        }
        if not args.quiet:
            print(json.dumps(report, indent=2))
        if args.state_file:
            args.state_file.parent.mkdir(parents=True, exist_ok=True)
            args.state_file.write_text(json.dumps(report, indent=2))
        return 2

    raw = args.config.read_text()
    try:
        cfg = json.loads(raw)
    except json.JSONDecodeError as exc:
        report = {
            "status": "invalid_json",
            "config": str(args.config),
            "error": str(exc),
            "checked_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        }
        if not args.quiet:
            print(json.dumps(report, indent=2))
        if args.state_file:
            args.state_file.parent.mkdir(parents=True, exist_ok=True)
            args.state_file.write_text(json.dumps(report, indent=2))
        return 2

    report = audit(cfg)
    report["config"] = str(args.config)
    report["checked_at"] = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()

    if args.repair and report["status"] != "ok":
        bak = _backup(args.config)
        new_cfg, actions = repair(cfg)
        if actions:
            payload = json.dumps(new_cfg, indent=2) + "\n"
            _atomic_write(args.config, payload, mode=0o600)
            report["repair"] = {"backup": str(bak), "actions": actions}
            # Re-audit after writing
            report["post_repair_status"] = audit(new_cfg)["status"]
        else:
            # Repair was a no-op despite findings (e.g. drift only)
            report["repair"] = {"backup": str(bak), "actions": []}

    if args.state_file:
        args.state_file.parent.mkdir(parents=True, exist_ok=True)
        args.state_file.write_text(json.dumps(report, indent=2))

    # Effective status: if --repair turned a regression into ok, exit 0.
    effective_status = report.get("post_repair_status") or report["status"]
    if effective_status == "ok":
        if not args.quiet:
            print(json.dumps(report, indent=2))
        return 0
    if not args.quiet:
        print(json.dumps(report, indent=2))
    return 2


if __name__ == "__main__":
    sys.exit(main())
