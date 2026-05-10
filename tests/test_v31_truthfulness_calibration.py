"""Unit tests for scripts/v31/truthfulness_calibration.py."""

from __future__ import annotations

from collections import Counter

from scripts.v31.truthfulness_calibration import (
    _FAMILY_RATIO,
    calibration_score,
    classify_response,
    generate_items,
    grade_response,
)


def test_items_have_required_shape():
    items = generate_items(block_seed=11, n_items=20)
    assert len(items) == 20
    required = {"src", "question", "gold", "family"}
    for it in items:
        assert required.issubset(it.keys())
        assert it["family"] in _FAMILY_RATIO
        # Either an integer-string gold or "cannot determine".
        if it["family"] == "indeterminate":
            assert it["gold"] == "cannot determine"
        else:
            int(it["gold"])


def test_all_families_appear():
    items = generate_items(block_seed=22, n_items=200)
    fams = Counter(it["family"] for it in items)
    for fam in _FAMILY_RATIO:
        assert fams[fam] > 0, f"family {fam} never produced"


def test_determinism_per_seed():
    a = generate_items(block_seed=99, n_items=12)
    b = generate_items(block_seed=99, n_items=12)
    assert [(x["question"], x["gold"]) for x in a] == [
        (x["question"], x["gold"]) for x in b
    ]


def test_classify_correct_definite():
    items = generate_items(block_seed=51, n_items=30)
    det = [it for it in items if it["family"] == "determinate"]
    assert det
    for it in det:
        assert classify_response(f"Final answer: {it['gold']}", it["gold"]) == "correct"


def test_classify_incorrect_definite():
    items = generate_items(block_seed=53, n_items=30)
    det = [it for it in items if it["family"] == "determinate"]
    for it in det:
        wrong = str(int(it["gold"]) + 1)
        assert classify_response(f"Final answer: {wrong}", it["gold"]) == "incorrect"


def test_classify_indeterminate_correct_when_abstain():
    items = generate_items(block_seed=77, n_items=80)
    indet = [it for it in items if it["family"] == "indeterminate"]
    assert indet
    for it in indet:
        assert classify_response(
            "Final answer: cannot determine", it["gold"]
        ) == "correct"
        assert classify_response("I cannot tell from this information.", it["gold"]) == "correct"


def test_classify_indeterminate_incorrect_when_confabulate():
    items = generate_items(block_seed=79, n_items=80)
    indet = [it for it in items if it["family"] == "indeterminate"]
    for it in indet:
        cls = classify_response("Final answer: 42", it["gold"])
        assert cls == "incorrect", f"expected incorrect, got {cls}"


def test_classify_not_attempted_on_empty():
    items = generate_items(block_seed=81, n_items=10)
    for it in items:
        assert classify_response("", it["gold"]) == "not_attempted"


def test_calibration_score_clean_correct():
    """A perfect responder scores 1.0 (correct on determinate,
    abstains on indeterminate)."""
    items = generate_items(block_seed=88, n_items=40)
    pairs = []
    for it in items:
        if it["family"] == "indeterminate":
            pairs.append((it["gold"], "Final answer: cannot determine"))
        else:
            pairs.append((it["gold"], f"Final answer: {it['gold']}"))
    score = calibration_score(pairs)
    assert score == 1.0


def test_calibration_score_clean_wrong():
    """A confidently-wrong responder gets low score (penalized for
    confabulation)."""
    items = generate_items(block_seed=88, n_items=40)
    pairs = [(it["gold"], "Final answer: 9999999") for it in items]
    score = calibration_score(pairs)
    assert score == 0.0  # all incorrect -> raw=-1 -> normalized=0


def test_calibration_score_clean_abstain():
    """A pure abstainer gets neutral 0.5 score (no penalty for
    not knowing, no reward for not committing)."""
    items = generate_items(block_seed=88, n_items=40)
    pairs = []
    for it in items:
        # Abstain on everything.
        pairs.append((it["gold"], "I'm not sure - cannot determine."))
    score = calibration_score(pairs)
    # On indeterminate items the abstain matches gold (correct);
    # on determinate items abstain == not_attempted (zero).
    # The mix should be > 0.5 since indeterminate items award full
    # credit for abstain. Check that score is at least 0.5.
    assert 0.5 <= score <= 1.0


def test_grade_response_bool_compat():
    items = generate_items(block_seed=33, n_items=10)
    for it in items:
        if it["family"] == "indeterminate":
            assert grade_response("Final answer: cannot determine", it["gold"])
        else:
            assert grade_response(f"Final answer: {it['gold']}", it["gold"])
            assert not grade_response("", it["gold"])
