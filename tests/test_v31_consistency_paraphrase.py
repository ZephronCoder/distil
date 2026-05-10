"""Unit tests for scripts/v31/consistency_paraphrase.py."""

from __future__ import annotations

import random

from scripts.v31.consistency_paraphrase import (
    _rotate_names,
    consistency_score,
    generate_items,
)
from scripts.v31.math_gsm_symbolic import _NAMES_F as _M1_NAMES_F
from scripts.v31.math_gsm_symbolic import _NAMES_M as _M1_NAMES_M


def test_items_have_required_shape():
    items = generate_items(block_seed=11, n_items=20)
    assert len(items) == 20
    required = {"src", "question", "question_b", "gold", "template", "difficulty"}
    for it in items:
        assert required.issubset(it.keys())
        int(it["gold"])
        assert it["difficulty"] in (0, 1)
        assert it["question"] != it["question_b"]


def test_paraphrase_changes_surface():
    """The paraphrase should differ from the original by at least
    a few characters but preserve key structure."""
    items = generate_items(block_seed=22, n_items=40)
    diff_pairs = 0
    for it in items:
        if it["question"] != it["question_b"]:
            diff_pairs += 1
    # Paraphrase should produce a different surface > 80% of the time.
    assert diff_pairs / len(items) >= 0.8, (
        f"only {diff_pairs}/{len(items)} pairs differ"
    )


def test_determinism_per_seed():
    a = generate_items(block_seed=99, n_items=12)
    b = generate_items(block_seed=99, n_items=12)
    assert [(x["question"], x["question_b"], x["gold"]) for x in a] == [
        (x["question"], x["question_b"], x["gold"]) for x in b
    ]


def test_consistency_score_both_correct():
    items = generate_items(block_seed=33, n_items=10)
    for it in items:
        a = f"#### {it['gold']}"
        b = f"#### {it['gold']}"
        assert consistency_score(a, b, it["gold"]) == 1.0


def test_consistency_score_one_correct():
    items = generate_items(block_seed=44, n_items=10)
    for it in items:
        a = f"#### {it['gold']}"
        wrong = str(int(it["gold"]) + 1)
        b = f"#### {wrong}"
        assert consistency_score(a, b, it["gold"]) == 0.5


def test_consistency_score_neither_correct():
    items = generate_items(block_seed=55, n_items=10)
    for it in items:
        wrong = str(int(it["gold"]) + 1)
        a = f"#### {wrong}"
        b = f"#### {wrong}"
        assert consistency_score(a, b, it["gold"]) == 0.0


def test_consistency_score_empty():
    items = generate_items(block_seed=66, n_items=5)
    for it in items:
        assert consistency_score("", "", it["gold"]) == 0.0


def test_paraphrase_preserves_answer_marker():
    """The paraphrase should still end with the GSM8K-style answer
    instruction so the grader works."""
    items = generate_items(block_seed=77, n_items=20)
    for it in items:
        assert "Solve step by step" in it["question_b"]
        assert "#### N" in it["question_b"]


def test_rotate_names_swaps_within_gender():
    """Name rotation must replace a male name with another male name
    and a female name with another female name, never crossing gender
    or producing the identity. Aligns with the IPT defence in
    arXiv 2604.15149 — surface change with semantically null content."""
    rng = random.Random(2026)
    text = "Alice and Ben went to the bakery. Alice bought 5 muffins."
    out = _rotate_names(text, rng)
    assert out != text, "name rotation must change the surface"
    new_alice = None
    for n in _M1_NAMES_F:
        if n in out and n != "Alice":
            new_alice = n
            break
    assert new_alice is not None, f"Alice was not rotated to another female name: {out!r}"
    new_ben = None
    for n in _M1_NAMES_M:
        if n in out and n != "Ben":
            new_ben = n
            break
    assert new_ben is not None, f"Ben was not rotated to another male name: {out!r}"
    assert out.count(new_alice) == 2, (
        f"both occurrences of Alice must map to the same target name, got {out!r}"
    )


def test_rotate_names_handles_no_names():
    """Without any matching first names the rotation must return the
    original string unchanged."""
    rng = random.Random(7)
    text = "The bakery sold cookies and bagels yesterday."
    out = _rotate_names(text, rng)
    assert out == text


def test_rotate_names_no_self_collision():
    """Two distinct names in the source must rotate to two distinct
    targets — never collide onto the same actor (which would change
    semantics)."""
    rng = random.Random(31)
    text = "Alice met Bella at the café."
    out = _rotate_names(text, rng)
    found = [n for n in _M1_NAMES_F if n in out]
    assert len(found) >= 2, f"two distinct female names expected, got {out!r}"
    assert len(set(found)) == len(found), f"name collision in rotated output: {out!r}"


def test_paraphrase_uses_name_rotation_in_practice():
    """End-to-end: a meaningful fraction of generated paraphrase
    pairs should involve a different actor than the original (otherwise
    the IPT signal would be weak)."""
    items = generate_items(block_seed=2026, n_items=30)
    rotated = 0
    for it in items:
        a = it["question"]
        b = it["question_b"]
        a_names = {n for n in _M1_NAMES_M + _M1_NAMES_F if f" {n} " in f" {a} " or a.startswith(n + " ")}
        b_names = {n for n in _M1_NAMES_M + _M1_NAMES_F if f" {n} " in f" {b} " or b.startswith(n + " ")}
        if a_names and a_names != b_names:
            rotated += 1
    assert rotated >= 5, (
        f"at most {rotated}/{len(items)} items show a name rotation; the "
        "IPT signal is too weak. Tune the rotation probability."
    )
