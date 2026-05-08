"""Regression tests for ``scripts/openclaw_config_guard.py``.

The 2026-04-28 regression that dropped ``"message"`` from the
``sn97-bot`` agent's ``tools.allow`` array silently disabled all
king-crown announcements and thread replies for ~9 days. These tests
pin the audit's invariants and the repair's idempotency so the same
shape can never silently slip past code review again.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402

from scripts import openclaw_config_guard as guard  # noqa: E402


def _good_config() -> dict:
    """Mirror of the on-disk shape that passes audit cleanly. Token
    placeholders are obvious garbage so the test never accidentally
    references a real secret."""
    return {
        "agents": {
            "list": [
                {
                    "id": "sn97-bot",
                    "name": "Arbos - Distil SN97",
                    "tools": {
                        "allow": ["web_fetch", "web_search", "read", "message"],
                    },
                },
                {
                    "id": "distil",
                    "name": "Distil Channel",
                    "tools": {"allow": ["read", "message"]},
                },
            ]
        },
        "channels": {
            "discord": {
                "enabled": True,
                "accounts": {
                    "arbos": {
                        "token": "FAKE_TOKEN_NOT_REAL",
                        "guilds": {
                            "799672011265015819": {
                                "channels": {
                                    "1482026267392868583": {
                                        "enabled": True,
                                        "autoThread": True,
                                    }
                                }
                            }
                        },
                        "threadBindings": {
                            "enabled": True,
                            "spawnSubagentSessions": True,
                        },
                    }
                },
            }
        },
    }


def test_audit_passes_on_good_config():
    report = guard.audit(_good_config())
    assert report["status"] == "ok"
    assert report["findings"] == []


def test_audit_flags_missing_message_tool():
    cfg = _good_config()
    cfg["agents"]["list"][0]["tools"]["allow"] = ["web_fetch", "web_search", "read"]
    report = guard.audit(cfg)
    assert report["status"] == "regression"
    codes = [f["code"] for f in report["findings"]]
    assert "missing_message_tool" in codes


def test_audit_flags_missing_agent_entirely():
    cfg = _good_config()
    cfg["agents"]["list"] = [
        e for e in cfg["agents"]["list"] if e.get("id") != "sn97-bot"
    ]
    report = guard.audit(cfg)
    assert report["status"] == "regression"
    assert any(f["code"] == "missing_agent" for f in report["findings"])


def test_audit_flags_disabled_thread_bindings():
    cfg = _good_config()
    cfg["channels"]["discord"]["accounts"]["arbos"]["threadBindings"]["enabled"] = False
    report = guard.audit(cfg)
    assert report["status"] == "regression"
    assert any(f["code"] == "thread_bindings_disabled" for f in report["findings"])


def test_audit_flags_disabled_channel():
    cfg = _good_config()
    cfg["channels"]["discord"]["enabled"] = False
    report = guard.audit(cfg)
    assert report["status"] == "regression"
    assert any(f["code"] == "discord_channel_disabled" for f in report["findings"])


def test_audit_flags_missing_arbos_account():
    cfg = _good_config()
    cfg["channels"]["discord"]["accounts"].pop("arbos")
    report = guard.audit(cfg)
    assert report["status"] == "regression"
    assert any(f["code"] == "missing_arbos_account" for f in report["findings"])


def test_audit_flags_missing_guild_binding():
    cfg = _good_config()
    cfg["channels"]["discord"]["accounts"]["arbos"]["guilds"].pop("799672011265015819")
    report = guard.audit(cfg)
    assert report["status"] == "regression"
    assert any(f["code"] == "missing_guild_binding" for f in report["findings"])


def test_audit_flags_channel_disabled():
    cfg = _good_config()
    cfg["channels"]["discord"]["accounts"]["arbos"]["guilds"]["799672011265015819"][
        "channels"
    ]["1482026267392868583"]["enabled"] = False
    report = guard.audit(cfg)
    assert report["status"] == "regression"
    assert any(f["code"] == "channel_disabled" for f in report["findings"])


def test_audit_flags_auto_thread_drift_as_drift_not_regression():
    cfg = _good_config()
    cfg["channels"]["discord"]["accounts"]["arbos"]["guilds"]["799672011265015819"][
        "channels"
    ]["1482026267392868583"]["autoThread"] = False
    report = guard.audit(cfg)
    # Channel is still enabled; auto-thread off is a soft drift signal,
    # not a hard regression. We surface it but don't block.
    assert report["status"] == "drift"
    assert any(f["code"] == "auto_thread_disabled" for f in report["findings"])


def test_repair_appends_message_tool_idempotently():
    cfg = _good_config()
    cfg["agents"]["list"][0]["tools"]["allow"] = ["web_fetch", "web_search", "read"]
    new_cfg, actions = guard.repair(copy.deepcopy(cfg))
    assert actions, "expected at least one repair action"
    assert "message" in new_cfg["agents"]["list"][0]["tools"]["allow"]
    # Idempotent — second pass must be a no-op
    _, actions2 = guard.repair(copy.deepcopy(new_cfg))
    assert actions2 == []
    assert guard.audit(new_cfg)["status"] == "ok"


def test_repair_fixes_thread_bindings():
    cfg = _good_config()
    cfg["channels"]["discord"]["accounts"]["arbos"]["threadBindings"]["enabled"] = False
    new_cfg, actions = guard.repair(copy.deepcopy(cfg))
    assert any("threadBindings" in a for a in actions)
    assert (
        new_cfg["channels"]["discord"]["accounts"]["arbos"]["threadBindings"]["enabled"]
        is True
    )
    assert guard.audit(new_cfg)["status"] == "ok"


def test_repair_does_not_invent_missing_agent():
    cfg = _good_config()
    cfg["agents"]["list"] = [
        e for e in cfg["agents"]["list"] if e.get("id") != "sn97-bot"
    ]
    new_cfg, actions = guard.repair(copy.deepcopy(cfg))
    # Missing agent is a structural problem the guard refuses to invent
    # silently — the audit still flags it.
    assert "sn97-bot" not in [a.get("id") for a in new_cfg["agents"]["list"]]
    assert guard.audit(new_cfg)["status"] == "regression"


def test_repair_does_not_touch_token_field():
    cfg = _good_config()
    cfg["agents"]["list"][0]["tools"]["allow"] = ["web_fetch"]
    original_token = cfg["channels"]["discord"]["accounts"]["arbos"]["token"]
    new_cfg, _actions = guard.repair(copy.deepcopy(cfg))
    assert (
        new_cfg["channels"]["discord"]["accounts"]["arbos"]["token"] == original_token
    )


def test_main_check_mode_returns_zero_on_clean(tmp_path: Path, capsys):
    cfg_file = tmp_path / "openclaw.json"
    cfg_file.write_text(json.dumps(_good_config()))
    state_file = tmp_path / "state.json"
    rc = guard.main(["--config", str(cfg_file), "--state-file", str(state_file), "--quiet"])
    assert rc == 0
    saved = json.loads(state_file.read_text())
    assert saved["status"] == "ok"


def test_main_check_mode_returns_two_on_regression(tmp_path: Path):
    cfg = _good_config()
    cfg["agents"]["list"][0]["tools"]["allow"] = ["web_fetch", "read"]
    cfg_file = tmp_path / "openclaw.json"
    cfg_file.write_text(json.dumps(cfg))
    state_file = tmp_path / "state.json"
    rc = guard.main(["--config", str(cfg_file), "--state-file", str(state_file), "--quiet"])
    assert rc == 2
    saved = json.loads(state_file.read_text())
    assert saved["status"] == "regression"


def test_main_repair_writes_backup_and_fixes_file(tmp_path: Path):
    cfg = _good_config()
    cfg["agents"]["list"][0]["tools"]["allow"] = ["web_fetch", "read"]
    cfg_file = tmp_path / "openclaw.json"
    cfg_file.write_text(json.dumps(cfg))
    rc = guard.main(["--config", str(cfg_file), "--repair", "--quiet"])
    # Repair drove status back to ok, so exit 0
    new_cfg = json.loads(cfg_file.read_text())
    assert "message" in new_cfg["agents"]["list"][0]["tools"]["allow"]
    backups = list(tmp_path.glob("openclaw.json.bak.*-guard"))
    assert backups, "expected one *-guard backup to be written"
    # Bak must be the pre-repair content
    bak_cfg = json.loads(backups[0].read_text())
    assert bak_cfg["agents"]["list"][0]["tools"]["allow"] == ["web_fetch", "read"]
    # Re-audit returned ok, so the repair driver's overall exit is 0
    assert rc == 0


def test_main_handles_invalid_json(tmp_path: Path):
    cfg_file = tmp_path / "openclaw.json"
    cfg_file.write_text("{ not valid json")
    state_file = tmp_path / "state.json"
    rc = guard.main(["--config", str(cfg_file), "--state-file", str(state_file), "--quiet"])
    assert rc == 2
    saved = json.loads(state_file.read_text())
    assert saved["status"] == "invalid_json"


def test_main_handles_missing_file(tmp_path: Path):
    state_file = tmp_path / "state.json"
    rc = guard.main([
        "--config", str(tmp_path / "missing.json"),
        "--state-file", str(state_file),
        "--quiet",
    ])
    assert rc == 2
    saved = json.loads(state_file.read_text())
    assert saved["status"] == "missing"
