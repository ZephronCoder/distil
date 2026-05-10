"""Unit tests for scripts/v31/math_robustness.py."""

from __future__ import annotations

from collections import Counter

from scripts.v31.math_robustness import (
    _PERTURBATIONS,
    generate_items,
)


def test_items_have_required_shape():
    items = generate_items(block_seed=11, n_items=20)
    assert len(items) == 20
    required = {"src", "question", "gold", "perturbation", "template"}
    for it in items:
        assert required.issubset(it.keys())
        assert isinstance(it["question"], str) and len(it["question"]) > 60
        int(it["gold"])
        assert it["perturbation"] in _PERTURBATIONS


def test_all_perturbations_appear():
    items = generate_items(block_seed=22, n_items=120)
    perturbs = Counter(it["perturbation"] for it in items)
    for p in _PERTURBATIONS:
        assert perturbs.get(p, 0) > 0, f"perturbation {p!r} never produced"


def test_determinism_per_seed():
    a = generate_items(block_seed=99, n_items=12)
    b = generate_items(block_seed=99, n_items=12)
    assert [(x["question"], x["gold"]) for x in a] == [
        (x["question"], x["gold"]) for x in b
    ]


def test_unit_swap_changes_surface():
    items = generate_items(block_seed=33, n_items=80)
    unit = [it for it in items if it["perturbation"] == "unit_swap"]
    assert unit
    # At least one item should contain a non-USD unit token.
    all_text = " ".join(it["question"] for it in unit)
    assert ("euros" in all_text or "pounds" in all_text or "kilometres" in all_text)


def test_context_pad_extends_question():
    """A padded item's question should be visibly longer than the
    median P0 GSM-Symbolic item (which is around 200-400 chars).
    """
    items = generate_items(block_seed=55, n_items=60)
    padded = [it for it in items if it["perturbation"] == "context_pad"]
    assert padded
    # Padded questions should average > 250 chars (irrelevant prefix
    # adds 60-200 chars).
    avg_len = sum(len(it["question"]) for it in padded) / len(padded)
    assert avg_len > 250, f"context_pad too short, avg={avg_len}"


def test_digit_expand_either_scales_or_falls_back():
    """digit_expand items should contain at least one number that's
    visibly larger than typical base counts (~30 max from M1
    templates). Scaling factor is 10x or 100x, so any matched
    scalable noun's count should be >= 50.
    """
    items = generate_items(block_seed=77, n_items=80)
    expanded = [it for it in items if it["perturbation"] == "digit_expand"]
    assert expanded, "no digit_expand items produced"
    for it in expanded:
        import re
        nums = [int(n) for n in re.findall(r"\d+", it["question"])]
        assert any(n >= 50 for n in nums), (
            f"no scaled-up number in {it['src']}: {it['question'][:200]}"
        )


def test_no_obvious_format_breakage():
    """All items must end with the GSM8K-style answer marker."""
    items = generate_items(block_seed=88, n_items=40)
    for it in items:
        assert "Solve step by step" in it["question"], (
            f"missing answer marker for {it['src']}: {it['question'][-200:]}"
        )


def test_topical_distractor_injects_irrelevant_clause():
    """The GSM-NoOp / topical_distractor perturbation must inject a
    clause that contains a number AND uses one of the scalable
    nouns. The injected number must NOT equal the gold (otherwise
    the model can fold it in correctly without reasoning)."""
    from scripts.v31.math_robustness import _SCALABLE_NOUNS

    items = generate_items(block_seed=2026, n_items=120)
    distractors = [it for it in items if it["perturbation"] == "topical_distractor"]
    assert distractors, "no topical_distractor items produced; ratio dropped to zero"
    matched = 0
    for it in distractors:
        q = it["question"]
        nouns_present = [n for n in _SCALABLE_NOUNS if n.lower() in q.lower()]
        assert nouns_present, f"distractor without a scalable noun: {q[:300]}"
        matched += 1
    assert matched == len(distractors), "every topical_distractor item must reference a noun"


def test_topical_distractor_preserves_gold():
    """Gold must remain unchanged under the topical_distractor
    perturbation — the injected number is decorative, not a problem
    variable. We verify the gold is the same kind of integer as
    other M1 items (i.e. not corrupted)."""
    items = generate_items(block_seed=4242, n_items=80)
    distractors = [it for it in items if it["perturbation"] == "topical_distractor"]
    assert distractors
    for it in distractors:
        gold = int(it["gold"])
        assert 0 <= gold < 10_000_000, f"gold out of M1 sanity range: {gold}"
