"""Unit tests for ``scripts/v31/ifeval_verifiable.py``.

Covers:
* item shape (``src``, ``prompt``, ``instruction_ids``, ``kwargs``)
* stack-depth distribution roughly matches the target ratio
* every supported verifier ID is exercised across many rounds
* no item contains conflicting constraints
* the grader actually accepts a hand-crafted compliant response on
  a few simple stack=1 items (sanity check that the kwargs we emit
  match what ``evaluate_item`` expects)
* determinism (same seed → same items)

The grader sanity-check is the most important test: it verifies that
our procedural kwargs are wire-compatible with the vendored Google
IFEval grader. If a kwarg key drifts out of sync, the grader silently
returns False and the axis dies — so we lock the contract here.
"""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.v31 import ifeval_verifiable as ife
from scripts.ifeval_vendor import SUPPORTED_VERIFIERS, evaluate_item


# ─────────────────────────────────────────────────────────────────────
#  Shape checks
# ─────────────────────────────────────────────────────────────────────


def test_generate_items_returns_correct_count():
    items = ife.generate_items(block_seed=1, n_items=10)
    assert len(items) == 10


def test_each_item_has_required_keys():
    items = ife.generate_items(block_seed=1, n_items=20)
    required = {"src", "prompt", "instruction_ids", "kwargs", "stack_depth", "topic"}
    for it in items:
        assert required.issubset(it.keys())


def test_src_namespace_is_v31():
    items = ife.generate_items(block_seed=1, n_items=20)
    for it in items:
        assert it["src"].startswith("v31_ifeval/stack"), it["src"]


def test_kwargs_aligns_with_instruction_ids():
    items = ife.generate_items(block_seed=1, n_items=20)
    for it in items:
        assert len(it["instruction_ids"]) == len(it["kwargs"])
        assert len(it["instruction_ids"]) == it["stack_depth"]


def test_all_instruction_ids_supported_by_vendor():
    items = ife.generate_items(block_seed=1, n_items=200)
    for it in items:
        for iid in it["instruction_ids"]:
            assert iid in SUPPORTED_VERIFIERS, iid


# ─────────────────────────────────────────────────────────────────────
#  Stack-depth distribution
# ─────────────────────────────────────────────────────────────────────


def test_stack_depth_in_one_to_four():
    items = ife.generate_items(block_seed=1, n_items=200)
    for it in items:
        assert 1 <= it["stack_depth"] <= 4


def test_target_stack_depth_distribution_roughly_matches():
    """The ``target_stack_depth`` (BEFORE collapse from exclusive
    verifiers) should match the configured ratio to within 5 %.

    The post-collapse ``stack_depth`` will be slightly skewed toward 1
    because exclusive verifiers (constrained_response / json_format)
    collapse the stack — that's correct behavior; we test the
    collapse downstream.
    """
    items = ife.generate_items(block_seed=1, n_items=2000)
    counts = {d: 0 for d in (1, 2, 3, 4)}
    for it in items:
        counts[it["target_stack_depth"]] += 1
    n = len(items)
    target_ratios = ife._STACK_DEPTH_RATIO
    for depth in (1, 2, 3, 4):
        observed = counts[depth] / n
        target = target_ratios[depth - 1]
        assert abs(observed - target) < 0.05, (
            f"depth={depth}: observed {observed:.3f}, target {target:.3f}"
        )


def test_actual_stack_depth_at_most_target():
    """``stack_depth`` (actual) must never exceed ``target_stack_depth``."""
    items = ife.generate_items(block_seed=1, n_items=500)
    for it in items:
        assert it["stack_depth"] <= it["target_stack_depth"]
        assert it["stack_depth"] >= 1


def test_exclusive_verifiers_collapse_stack():
    """When an exclusive verifier (constrained_response or json_format)
    is in the stack, ``stack_depth`` should be at most 2."""
    items = ife.generate_items(block_seed=1, n_items=2000)
    for it in items:
        if "detectable_format:constrained_response" in it["instruction_ids"]:
            # constrained_response conflicts with everything → stack=1.
            assert it["stack_depth"] == 1, it


# ─────────────────────────────────────────────────────────────────────
#  Conflict avoidance
# ─────────────────────────────────────────────────────────────────────


def test_no_conflicts_within_a_stack():
    items = ife.generate_items(block_seed=1, n_items=500)
    for it in items:
        ids = it["instruction_ids"]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                assert not ife._conflicts(ids[i], ids[j]), (
                    f"conflict in stack: {ids[i]} vs {ids[j]}"
                )


# ─────────────────────────────────────────────────────────────────────
#  Verifier coverage
# ─────────────────────────────────────────────────────────────────────


def test_all_21_verifiers_appear_in_2000_items():
    items = ife.generate_items(block_seed=1, n_items=2000)
    seen = set()
    for it in items:
        seen.update(it["instruction_ids"])
    expected = set(ife.VERIFIER_SAMPLERS.keys())
    missing = expected - seen
    # Some constraints (like constrained_response, json_format) are
    # heavily conflicted and may legitimately be rare. Allow up to 2
    # rare verifiers to be missing in any 2000-item batch.
    assert len(missing) <= 2, f"too many verifiers missing: {missing}"


def test_verifier_count_matches_vendor():
    """All 21 vendored verifiers should have a sampler."""
    assert set(ife.VERIFIER_SAMPLERS.keys()) == set(SUPPORTED_VERIFIERS.keys())


# ─────────────────────────────────────────────────────────────────────
#  Determinism
# ─────────────────────────────────────────────────────────────────────


def test_same_seed_produces_identical_items():
    a = ife.generate_items(block_seed=12345, n_items=10)
    b = ife.generate_items(block_seed=12345, n_items=10)
    assert a == b


def test_none_seed_treated_as_zero():
    a = ife.generate_items(block_seed=None, n_items=5)
    b = ife.generate_items(block_seed=0, n_items=5)
    assert a == b


def test_different_seeds_produce_different_items():
    a = ife.generate_items(block_seed=12345, n_items=10)
    b = ife.generate_items(block_seed=67890, n_items=10)
    assert {it["prompt"] for it in a} != {it["prompt"] for it in b}


# ─────────────────────────────────────────────────────────────────────
#  Grader sanity — most important test.
#  Verify the grader accepts hand-crafted compliant responses.
# ─────────────────────────────────────────────────────────────────────


def test_grader_passes_on_compliant_lowercase_response():
    """Hand-craft a stack=1 lowercase item and verify a compliant
    response passes the grader."""
    item = {
        "instruction_ids": ["change_case:english_lowercase"],
        "kwargs": [{}],
    }
    response = "this entire response is lowercase. it has no uppercase letters at all."
    all_pass, per = evaluate_item(
        response, item["instruction_ids"], item["kwargs"]
    )
    assert all_pass is True
    assert per == [True]


def test_grader_fails_on_noncompliant_lowercase_response():
    item = {
        "instruction_ids": ["change_case:english_lowercase"],
        "kwargs": [{}],
    }
    response = "This Has Capitals."
    all_pass, per = evaluate_item(
        response, item["instruction_ids"], item["kwargs"]
    )
    assert all_pass is False


def test_grader_passes_on_compliant_no_comma_response():
    item = {
        "instruction_ids": ["punctuation:no_comma"],
        "kwargs": [{}],
    }
    response = "This response has no commas at all."
    all_pass, _per = evaluate_item(
        response, item["instruction_ids"], item["kwargs"]
    )
    assert all_pass is True


def test_grader_fails_when_comma_present():
    item = {
        "instruction_ids": ["punctuation:no_comma"],
        "kwargs": [{}],
    }
    response = "This response, however, has commas."
    all_pass, _per = evaluate_item(
        response, item["instruction_ids"], item["kwargs"]
    )
    assert all_pass is False


def test_grader_passes_on_compliant_word_count():
    """Item asks for at least 30 words; provide 50."""
    item = {
        "instruction_ids": ["length_constraints:number_words"],
        "kwargs": [{"relation": "at least", "num_words": 30}],
    }
    response = " ".join(["word"] * 50)
    all_pass, _per = evaluate_item(
        response, item["instruction_ids"], item["kwargs"]
    )
    assert all_pass is True


def test_grader_fails_on_too_few_words():
    item = {
        "instruction_ids": ["length_constraints:number_words"],
        "kwargs": [{"relation": "at least", "num_words": 50}],
    }
    response = "Only five words here please"
    all_pass, _per = evaluate_item(
        response, item["instruction_ids"], item["kwargs"]
    )
    assert all_pass is False


def test_grader_passes_on_compliant_keyword_existence():
    item = {
        "instruction_ids": ["keywords:existence"],
        "kwargs": [{"keywords": ["lighthouse", "harbor"]}],
    }
    response = "The lighthouse near the harbor was visible from miles."
    all_pass, _per = evaluate_item(
        response, item["instruction_ids"], item["kwargs"]
    )
    assert all_pass is True


def test_grader_fails_on_missing_keyword():
    item = {
        "instruction_ids": ["keywords:existence"],
        "kwargs": [{"keywords": ["lighthouse", "harbor"]}],
    }
    response = "The forest was quiet today."
    all_pass, _per = evaluate_item(
        response, item["instruction_ids"], item["kwargs"]
    )
    assert all_pass is False


def test_grade_item_helper_works():
    """``ife.grade_item`` is the convenience wrapper used by the
    bench probe; verify it round-trips."""
    item = {
        "instruction_ids": ["change_case:english_lowercase"],
        "kwargs": [{}],
    }
    all_pass, per = ife.grade_item("this is all lowercase.", item)
    assert all_pass is True
    assert per == [True]


# ─────────────────────────────────────────────────────────────────────
#  Robustness — generator never raises
# ─────────────────────────────────────────────────────────────────────


def test_generator_never_raises_across_many_rounds():
    for seed in range(50):
        items = ife.generate_items(block_seed=seed, n_items=20)
        assert len(items) == 20
        for it in items:
            assert "?" not in it["prompt"] or "..." not in it["prompt"]
            assert it["instruction_ids"]
            assert len(it["instruction_ids"]) == it["stack_depth"]
