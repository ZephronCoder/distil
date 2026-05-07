import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = REPO_ROOT / "api"
for path in (str(API_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from eval_queue import build_queue_slots, current_model_from_progress  # noqa: E402


def test_current_model_accepts_nested_current_student():
    assert current_model_from_progress({
        "current": {"student_name": "miner/model"},
    }) == "miner/model"


def test_build_queue_slots_marks_done_running_pending_and_deferred():
    slots = build_queue_slots(
        {
            "current_student": "b/model",
            "completed": [{"uid": "1", "student_name": "a/model"}],
            "eval_order": [
                {"uid": "1", "model": "a/model", "role": "king"},
                {"uid": 2, "model": "b/model", "role": "challenger"},
                {"uid": 3, "model": "c/model", "role": "challenger"},
            ],
        },
        {"models_to_eval": {"3": {"commit_block": 123, "revision": "abc"}}},
        {
            "pending": [
                {"uid": 3, "model": "c/model", "status": "deferred"},
                {"uid": 4, "model": "d/model", "status": "deferred"},
            ],
        },
    )

    assert [slot["status"] for slot in slots[:3]] == ["done", "running", "pending"]
    assert slots[2]["commit_block"] == 123
    assert slots[2]["revision"] == "abc"
    assert slots[3]["uid"] == 4
    assert slots[3]["status"] == "deferred"


def test_build_queue_slots_accepts_models_done_fallback():
    slots = build_queue_slots(
        {
            "models": {"a/model": {"status": "ok"}},
            "eval_order": [{"uid": 1, "model": "a/model", "role": "king"}],
        },
        {},
        {},
    )

    assert slots[0]["status"] == "done"
