import os
import time
from types import SimpleNamespace

from scripts import sn97_healthcheck as health


def _make_hub(tmp_path, entries):
    """Build a fake HuggingFace hub layout.

    ``entries`` is a list of (name, size_bytes, age_hours) tuples. Uses
    sparse files so multi-GB sizes stay free and tests stay fast.
    """
    hub = tmp_path / "hub"
    hub.mkdir()
    now = time.time()
    for name, size, age_h in entries:
        d = hub / name
        d.mkdir()
        f = d / "blob"
        with open(f, "wb") as fh:
            if size > 0:
                fh.truncate(size)
        mtime = now - age_h * 3600
        os.utime(f, (mtime, mtime))
        os.utime(d, (mtime, mtime))
    return hub


def test_openclaw_discord_probe_flags_auth_loop(monkeypatch):
    def fake_run(cmd, timeout=20):
        assert cmd[:3] == ["journalctl", "-u", "openclaw"]
        return SimpleNamespace(
            returncode=0,
            stdout="\n".join(
                [
                    "May 09 [discord] [arbos] starting provider",
                    "May 09 [discord] channel resolve failed; using config entries. Discord API /users/@me/guilds failed (401): 401: Unauthorized",
                    "May 09 [discord] [arbos] channel exited: Failed to resolve Discord application id",
                ]
            ),
            stderr="",
        )

    monkeypatch.setattr(health, "run", fake_run)

    result = health.openclaw_discord_probe()

    assert result["ok"] is False
    assert result["auth_401"] == 1
    assert result["application_id_failures"] == 1
    assert result["provider_exits"] == 1
    assert result["last_error"] == "application_id_resolution_failed"


def test_openclaw_discord_probe_accepts_clean_journal(monkeypatch):
    monkeypatch.setattr(
        health,
        "run",
        lambda cmd, timeout=20: SimpleNamespace(
            returncode=0,
            stdout="May 09 [discord] [arbos] starting provider\n",
            stderr="",
        ),
    )

    result = health.openclaw_discord_probe()

    assert result == {
        "ok": True,
        "auth_401": 0,
        "application_id_failures": 0,
        "provider_exits": 0,
        "last_error": None,
        "since": "30 minutes ago",
    }


def test_repair_notifies_for_discord_auth_without_restart(monkeypatch):
    calls = []

    def fake_run(cmd, timeout=20):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(health, "run", fake_run)

    actions = health.repair(
        {
            "issues": ["openclaw:discord_auth_failed"],
            "services": {
                "validator": {"ok": True, "unit": "distil-validator"},
                "api": {"ok": True, "unit": "distil-api"},
                "dashboard": {"ok": True, "unit": "distil-dashboard"},
                "caddy": {"ok": True, "unit": "caddy"},
                "benchmark_timer": {"ok": True, "unit": "distil-benchmark-sync.timer"},
                "openclaw": {"ok": True, "unit": "openclaw"},
            },
            "http": {
                "api_local": {"ok": True},
                "api_public": {"ok": True},
                "dashboard_local": {"ok": True},
                "dashboard_public": {"ok": True},
                "chat_public": {"ok": True},
            },
            "open_webui": {"ok": True},
        }
    )

    assert actions == ["notify:openclaw:discord_provider_unhealthy"]
    assert not any(cmd[:2] == ["systemctl", "restart"] for cmd in calls)


def test_cleanup_hf_hub_keeps_current_student_and_recent(tmp_path):
    hub = _make_hub(
        tmp_path,
        [
            ("models--lapaliv--v16.0.0", 8 * 1024 * 1024 * 1024, 1),  # current, recent
            ("models--ghost--solution-5", 9 * 1024 * 1024 * 1024, 400),  # stale
            ("models--tom--distil-32", 9 * 1024 * 1024 * 1024, 400),  # stale
            ("models--small--noise", 1024, 400),  # too small
            ("models--fresh--build", 9 * 1024 * 1024 * 1024, 24),  # not stale enough
        ],
    )
    actions: list[str] = []

    result = health._cleanup_hf_hub(
        str(hub),
        current_student="lapaliv/v16.0.0",
        min_age_h=168,
        min_size_gb=1.0,
        actions=actions,
        tag="validator",
    )

    assert result["removed"] == 2
    remaining = sorted(p.name for p in hub.iterdir())
    assert remaining == [
        "models--fresh--build",
        "models--lapaliv--v16.0.0",
        "models--small--noise",
    ]
    assert actions == []


def test_cleanup_hf_hub_handles_missing_path():
    actions: list[str] = []
    result = health._cleanup_hf_hub(
        "/nonexistent/path",
        current_student=None,
        min_age_h=168,
        min_size_gb=1.0,
        actions=actions,
        tag="validator",
    )
    assert result == {"removed": 0, "gb": 0.0}


def test_render_markdown_handles_missing_chat_tunnel():
    report = {
        "timestamp": 1778304074,
        "revision": {"git": "abc1234", "file": None},
        "services": {
            "validator": {"active": "active"},
            "api": {"active": "active"},
            "dashboard": {"active": "active"},
            "openclaw": {"active": "active"},
        },
        "http": {
            "api_local": {"status": 200},
            "dashboard_local": {"status": 200},
            "chat_public": {"status": 200},
        },
        "openclaw_discord": {
            "ok": False,
            "last_error": "discord_unauthorized",
        },
        "validator": {
            "health": {
                "king_uid": 201,
                "eval_active": False,
            }
        },
        "actions": [],
        "issues": ["openclaw:discord_auth_failed"],
    }

    text = health.render_markdown(report)

    assert "| OpenClaw | active |" in text
    assert "| Discord bot | discord_unauthorized |" in text
    assert "| Chat | external / 200 |" in text
