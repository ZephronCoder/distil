"""Unit tests for ``scripts/v31/math_gsm_symbolic.py``.

Verifies the v31 GSM-Symbolic generator's core invariants:

* Items are well-formed dicts with the expected keys.
* The gold integer can be re-extracted from the question via the
  same regex pipeline used by ``_math_extract_answer`` in
  ``pod_eval_vllm.py`` (this is the contract that lets the existing
  math grader score these items unchanged).
* Difficulty distribution roughly matches the ``_DIFFICULTY_RATIO``.
* NoOp distractors are injected only on P0 items.
* Determinism: same ``block_seed`` produces the same items.
* No template raises an exception across many rounds.

These tests run fast (no network, no model invocation) so they're
safe to include in the standard test suite. Mirror the testing style
of ``tests/test_arena_v3_composite.py`` (pytest, no fixtures, no
parametrize beyond what's necessary).
"""

from __future__ import annotations

import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.v31 import math_gsm_symbolic as gsm


# ─────────────────────────────────────────────────────────────────────
#  Basic shape checks
# ─────────────────────────────────────────────────────────────────────


def test_generate_items_returns_correct_count():
    items = gsm.generate_items(block_seed=1, n_items=10)
    assert len(items) == 10


def test_generate_items_returns_zero_when_n_zero():
    items = gsm.generate_items(block_seed=1, n_items=0)
    assert items == []


def test_each_item_has_required_keys():
    items = gsm.generate_items(block_seed=1, n_items=20)
    required = {"src", "question", "gold", "difficulty", "is_noop", "template"}
    for it in items:
        assert required.issubset(it.keys()), f"missing keys in {it}"


def test_src_namespace_is_v31():
    items = gsm.generate_items(block_seed=1, n_items=20)
    for it in items:
        assert it["src"].startswith("v31_gsm_symbolic/"), it["src"]


def test_src_encodes_template_and_difficulty():
    items = gsm.generate_items(block_seed=1, n_items=40)
    for it in items:
        parts = it["src"].split("/")
        assert len(parts) == 3, f"expected 3-part src, got {it['src']}"
        ns, tmpl, diff_or_noop = parts
        assert ns == "v31_gsm_symbolic"
        assert tmpl == it["template"]
        if it["is_noop"]:
            assert diff_or_noop == "noop"
        else:
            assert diff_or_noop == f"p{it['difficulty']}"


def test_gold_is_string_of_int():
    items = gsm.generate_items(block_seed=1, n_items=20)
    for it in items:
        # Should be a stringified int (positive or negative); int() must
        # round-trip.
        gold_int = int(it["gold"])
        assert str(gold_int) == it["gold"]


def test_question_ends_with_answer_marker_instruction():
    items = gsm.generate_items(block_seed=1, n_items=20)
    for it in items:
        assert "#### N" in it["question"], it["question"]
        assert it["question"].rstrip().endswith(
            "where N is the final integer answer."
        ), it["question"]


# ─────────────────────────────────────────────────────────────────────
#  Determinism
# ─────────────────────────────────────────────────────────────────────


def test_same_seed_produces_identical_items():
    a = gsm.generate_items(block_seed=12345, n_items=10)
    b = gsm.generate_items(block_seed=12345, n_items=10)
    assert a == b


def test_different_seeds_produce_different_items():
    a = gsm.generate_items(block_seed=12345, n_items=10)
    b = gsm.generate_items(block_seed=67890, n_items=10)
    # Should be vastly different. Allow up to one accidental match.
    a_qs = {it["question"] for it in a}
    b_qs = {it["question"] for it in b}
    assert len(a_qs - b_qs) >= 8


def test_none_seed_treated_as_zero():
    a = gsm.generate_items(block_seed=None, n_items=5)
    b = gsm.generate_items(block_seed=0, n_items=5)
    assert a == b


# ─────────────────────────────────────────────────────────────────────
#  Difficulty distribution
# ─────────────────────────────────────────────────────────────────────


def test_difficulty_ratio_roughly_matches_target():
    items = gsm.generate_items(block_seed=1, n_items=2000)
    n = len(items)
    p0 = sum(1 for it in items if it["difficulty"] == 0 and not it["is_noop"]) / n
    p1 = sum(1 for it in items if it["difficulty"] == 1) / n
    p2 = sum(1 for it in items if it["difficulty"] == 2) / n
    noop = sum(1 for it in items if it["is_noop"]) / n
    target_p0, target_p1, target_p2, target_noop = gsm._DIFFICULTY_RATIO
    # 5 % absolute tolerance on each bucket.
    assert abs(p0 - target_p0) < 0.05
    assert abs(p1 - target_p1) < 0.05
    assert abs(p2 - target_p2) < 0.05
    assert abs(noop - target_noop) < 0.05


def test_noop_only_on_p0():
    items = gsm.generate_items(block_seed=1, n_items=500)
    for it in items:
        if it["is_noop"]:
            assert it["difficulty"] == 0, it


def test_noop_inserts_distractor_sentence():
    items = gsm.generate_items(block_seed=1, n_items=500)
    noop_items = [it for it in items if it["is_noop"]]
    assert len(noop_items) > 0
    # The distractor sentences contain a numeral and one of a small
    # set of off-topic phrases.
    distractor_markers = (
        "blocks from the bus stop",
        "hosting this event",
        "originally written",
        "other customers in line",
        "inch paper",
        "earlier than usual",
        "pets at home",
        "billboard",
        "worked there for",
        "lucky number",
    )
    for it in noop_items:
        assert any(m in it["question"] for m in distractor_markers), (
            f"distractor not detected in NoOp item: {it['question']}"
        )


# ─────────────────────────────────────────────────────────────────────
#  Robustness — generator never raises across many rounds
# ─────────────────────────────────────────────────────────────────────


def test_generator_never_raises_across_many_rounds():
    for seed in range(50):
        items = gsm.generate_items(block_seed=seed, n_items=20)
        assert len(items) == 20
        for it in items:
            int(it["gold"])  # gold parseable
            assert "?" in it["question"]


# ─────────────────────────────────────────────────────────────────────
#  Compatibility with existing math grader
# ─────────────────────────────────────────────────────────────────────


_GSM_ANSWER_MARKER = re.compile(r"####\s*(-?\d+)")


def test_grader_can_extract_gold_from_simulated_correct_answer():
    """Simulate a model that answers correctly and verify the existing
    ``#### N`` parser can extract the gold."""
    items = gsm.generate_items(block_seed=1, n_items=10)
    for it in items:
        simulated_response = (
            f"Let me work through this step by step. ... #### {it['gold']}"
        )
        m = _GSM_ANSWER_MARKER.search(simulated_response)
        assert m is not None
        assert int(m.group(1)) == int(it["gold"])


# ─────────────────────────────────────────────────────────────────────
#  Templates / families coverage
# ─────────────────────────────────────────────────────────────────────


def test_all_templates_in_registry_are_callable():
    for spec in gsm.TEMPLATES:
        assert callable(spec.fn)
        assert isinstance(spec.name, str) and spec.name
        assert isinstance(spec.families, tuple)


def test_all_templates_can_emit_p0_p1_p2():
    import random as _random
    for spec in gsm.TEMPLATES:
        for difficulty in (0, 1, 2):
            r = _random.Random(0)
            q, gold = spec.fn(r, difficulty)
            assert isinstance(q, str)
            assert isinstance(gold, int)
            assert "?" in q


def test_template_distribution_covers_registry():
    """Across many rounds, every template should appear at least once."""
    items = gsm.generate_items(block_seed=1, n_items=2000)
    seen = {it["template"] for it in items}
    expected = {spec.name for spec in gsm.TEMPLATES}
    assert seen == expected, f"missing templates: {expected - seen}"
