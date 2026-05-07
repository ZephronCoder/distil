import os
import sys


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class _State:
    h2h_latest = {"king_uid": 166}
    scores = {"166": 1.204, "201": 1.164}
    dq_reasons = {}
    uid_hotkey_map = {"166": "hk166", "201": "hk201"}
    composite_scores = {
        "166": {"model": "winner/model", "final": 0.43, "worst": 0.33, "weighted": 0.72},
        "201": {"model": "fallback/model", "final": 0.49, "worst": 0.40, "weighted": 0.74},
    }


def test_resolve_king_uses_uid_keyed_valid_models_for_eligibility(monkeypatch):
    from scripts.validator import service

    monkeypatch.setattr(service, "is_single_eval_mode", lambda: True)
    monkeypatch.setattr(
        service,
        "select_king_by_composite",
        lambda *_args, **_kwargs: (201, _State.composite_scores["201"]),
    )

    valid_models = {
        166: {
            "model": "winner/model",
            "revision": "rev166",
            "commit_block": 100,
            "hotkey": "hk166",
            "is_reference": False,
        },
        201: {
            "model": "fallback/model",
            "revision": "rev201",
            "commit_block": 90,
            "hotkey": "hk201",
            "is_reference": False,
        },
    }

    king_uid, king_kl, source = service._resolve_king(valid_models, _State())

    assert king_uid == 166
    assert king_kl == 1.204
    assert source == "composite"
