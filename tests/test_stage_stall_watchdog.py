"""Regression tests for StageStallWatchdog.

Background: a 2026-05-07 incident on UID 213 ``const0312/wtbmts09`` left
the eval pinned in ``loading_weights`` for ~4h22m. The validator did not
notice until the outer 5-hour ``eval_timeout`` deadline. The watchdog
catches the same shape inside ``DISTIL_STAGE_STALL_LOAD_S`` (default
45 min). These tests pin the contract so future refactors can't silently
break detection or kill semantics.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest

from scripts.validator.pod_session import StageStallWatchdog  # noqa: E402


def _progress(*, student_idx=0, student="hanktensa/distil-985", stage="loading_weights",
              prompts_done=0, bench_axis_idx=None, teacher_done=300, phase="loading_student"):
    return {
        "phase": phase,
        "teacher_prompts_done": teacher_done,
        "current": {
            "student_idx": student_idx,
            "student_name": student,
            "stage": stage,
            "prompts_done": prompts_done,
            "bench_axis_idx": bench_axis_idx,
        },
    }


def _make_watchdog(*, t0=0.0, load=2700, default=1500, kill_enabled=True):
    """Build a watchdog with a controllable clock and capture warn/kill calls."""
    clock = [t0]

    def now():
        return clock[0]

    fired = {"warn": [], "kill": []}

    def warn(*, elapsed, **_):
        fired["warn"].append(elapsed)

    def kill(*, elapsed, **_):
        fired["kill"].append(elapsed)

    wd = StageStallWatchdog(
        load_timeout_s=load,
        default_timeout_s=default,
        kill_enabled=kill_enabled,
        warn_action=warn,
        kill_action=kill,
        time_fn=now,
    )

    def advance(seconds):
        clock[0] += seconds

    return wd, advance, fired


def test_watchdog_returns_ok_on_first_call_and_records_baseline():
    wd, _advance, fired = _make_watchdog()
    assert wd.check(_progress()) == "ok"
    assert wd.fingerprint is not None
    assert fired["warn"] == []
    assert fired["kill"] == []


def test_watchdog_resets_when_progress_changes():
    wd, advance, fired = _make_watchdog(load=600)
    wd.check(_progress(stage="loading_weights"))
    advance(400)
    # Stage changed → reset, no warn even though half-limit is 300s
    assert wd.check(_progress(stage="bench_battery:ifeval_bench")) == "ok"
    assert fired["warn"] == []
    advance(400)
    # Still no warn — moved to a different stage and prompts haven't ticked
    # but new fingerprint window started 400s ago, half-limit for default
    # (1500/2 = 750) hasn't elapsed yet.
    assert wd.check(_progress(stage="bench_battery:ifeval_bench")) == "ok"


def test_watchdog_warns_at_half_limit_then_kills_at_full_limit():
    wd, advance, fired = _make_watchdog(load=600)
    progress = _progress(stage="loading_weights")
    assert wd.check(progress) == "ok"  # baseline
    advance(299)
    assert wd.check(progress) == "ok"  # below half-limit
    assert fired["warn"] == []
    advance(2)
    # Past 300s = half of 600s load limit
    assert wd.check(progress) == "warn"
    assert len(fired["warn"]) == 1
    assert fired["warn"][0] == pytest.approx(301.0)
    assert fired["kill"] == []
    advance(200)
    # Still inside the limit: 501s elapsed
    assert wd.check(progress) == "ok"
    advance(100)
    # Now past the 600s limit
    assert wd.check(progress) == "killed"
    assert len(fired["kill"]) == 1
    assert fired["kill"][0] == pytest.approx(601.0)


def test_watchdog_kill_is_idempotent_and_armed_after_first_fire():
    wd, advance, fired = _make_watchdog(load=100)
    progress = _progress(stage="loading_weights")
    wd.check(progress)
    advance(200)
    assert wd.check(progress) == "killed"
    # subsequent calls (even with new fingerprints) should remain killed
    assert wd.check(_progress(stage="bench_battery:ifeval_bench", prompts_done=42)) == "killed"
    assert len(fired["kill"]) == 1


def test_watchdog_warn_fires_only_once_per_stall_window():
    wd, advance, fired = _make_watchdog(load=200)
    progress = _progress(stage="loading_weights")
    wd.check(progress)
    advance(120)
    assert wd.check(progress) == "warn"
    advance(10)
    # Already warned for this fingerprint; should NOT warn again
    assert wd.check(progress) == "ok"
    assert len(fired["warn"]) == 1


def test_watchdog_uses_default_timeout_for_non_loading_stages():
    wd, advance, fired = _make_watchdog(load=600, default=200)
    progress = _progress(stage="bench_battery:ifeval_bench")
    wd.check(progress)
    advance(150)
    # Half of default 200 = 100, so we should have warned by now
    assert wd.check(progress) == "warn"
    advance(100)
    assert wd.check(progress) == "killed"


def test_watchdog_kill_disabled_only_warns_and_does_not_arm():
    wd, advance, fired = _make_watchdog(load=100, kill_enabled=False)
    progress = _progress(stage="loading_weights")
    wd.check(progress)
    advance(60)
    assert wd.check(progress) == "warn"
    advance(60)
    # Past full limit but kill disabled → still reports killed status
    # (caller treats it the same — the watchdog has hit its terminal state)
    # but no kill action fires
    assert wd.check(progress) == "killed"
    assert fired["kill"] == []


def test_watchdog_handles_missing_or_malformed_progress_gracefully():
    wd, advance, fired = _make_watchdog()
    # No "current" → reset, ok
    assert wd.check({"phase": "teacher_generation"}) == "ok"
    # Random non-dict progress → ok, reset
    assert wd.check({"current": "not-a-dict"}) == "ok"
    advance(99999)
    assert fired["warn"] == []
    assert fired["kill"] == []


def test_watchdog_resets_when_student_advances():
    wd, advance, fired = _make_watchdog(load=600)
    wd.check(_progress(student_idx=0, stage="bench_battery:ifeval_bench"))
    advance(500)
    # Different student → fingerprint differs → reset
    assert wd.check(_progress(student_idx=1, student="other/model", stage="loading_weights")) == "ok"
    assert fired["warn"] == []
    advance(500)
    # Now into the second student's stall window
    assert wd.check(_progress(student_idx=1, student="other/model", stage="loading_weights")) == "warn"


def test_watchdog_fingerprint_includes_bench_axis_progress():
    """Bench-axis advancement should reset the timer too."""
    # bench_battery:* uses the default (non-loading) limit, so set both
    # tightly so a 150s advance crosses the half-limit.
    wd, advance, fired = _make_watchdog(load=200, default=200)
    p = _progress(stage="bench_battery:ifeval_bench", bench_axis_idx=5)
    wd.check(p)
    advance(150)
    # Same axis 5 → would warn at half-limit (100s)
    assert wd.check(p) == "warn"
    # New axis → resets
    p2 = _progress(stage="bench_battery:ifeval_bench", bench_axis_idx=6)
    assert wd.check(p2) == "ok"
    advance(50)
    # Only 50s into axis 6, well below half-limit
    assert wd.check(p2) == "ok"


def test_watchdog_load_stage_prefixes_match_pod_eval_strings():
    """Ensure the prefix list covers every actual stage label written by
    ``scripts/pod_eval_vllm.py`` for the loading window."""
    from scripts.validator.pod_session import _LOAD_STAGE_PREFIXES
    assert "loading_weights" in _LOAD_STAGE_PREFIXES
    assert "loading_student" in _LOAD_STAGE_PREFIXES
    assert "loading_teacher" in _LOAD_STAGE_PREFIXES


def test_watchdog_uses_load_timeout_for_loading_student_stage():
    """Belt-and-suspenders: ``loading_student`` (the student hand-off
    stage in pod_eval) must use the more generous load timeout."""
    wd, advance, fired = _make_watchdog(load=600, default=200)
    progress = _progress(stage="loading_student")
    wd.check(progress)
    advance(150)
    # Half of load (300) not yet reached, so no warn
    assert wd.check(progress) == "ok"
    advance(160)
    # 310s elapsed → past half of 600 → warn
    assert wd.check(progress) == "warn"
