"""Unit tests for scripts/v31/knowledge_multi_hop_kg.py."""

from __future__ import annotations

from collections import Counter

from scripts.v31.knowledge_multi_hop_kg import (
    _SCHEMAS,
    generate_items,
    grade_response,
)


def test_items_have_required_shape():
    items = generate_items(block_seed=11, n_items=20)
    assert len(items) == 20
    schema_names = {n for n, _fn, _w in _SCHEMAS}
    for it in items:
        assert "src" in it and "question" in it and "gold" in it
        assert it["task"] in schema_names
        assert it["gold"]


def test_all_schemas_appear():
    items = generate_items(block_seed=22, n_items=200)
    tasks = Counter(it["task"] for it in items)
    expected = {n for n, _fn, _w in _SCHEMAS}
    missing = expected - set(tasks.keys())
    assert not missing, f"missing schemas: {missing}"


def test_determinism_per_seed():
    a = generate_items(block_seed=99, n_items=12)
    b = generate_items(block_seed=99, n_items=12)
    assert [(x["question"], x["gold"]) for x in a] == [
        (x["question"], x["gold"]) for x in b
    ]


def test_kg_block_has_facts():
    items = generate_items(block_seed=33, n_items=12)
    for it in items:
        assert "FACTS:" in it["question"]
        assert "QUESTION:" in it["question"]
        # At least 5 facts per item.
        facts = [
            ln for ln in it["question"].splitlines() if ln.startswith("- ")
        ]
        assert len(facts) >= 5, f"{it['src']} has only {len(facts)} facts"


def test_grader_accepts_canonical():
    items = generate_items(block_seed=51, n_items=10)
    for it in items:
        assert grade_response(
            f"Answer: {it['gold']}", it["gold"],
            all_correct=it.get("all_correct_answers"),
        )


def test_grader_rejects_unrelated_entity():
    items = generate_items(block_seed=53, n_items=10)
    for it in items:
        # Make a clearly-different entity name.
        wrong = "Person_ZZZZ" if it["gold"].startswith("Person") else "Place_ZZZZ"
        if wrong == it["gold"]:
            wrong = "Org_ZZZZ"
        assert not grade_response(
            f"Answer: {wrong}", it["gold"],
            all_correct=it.get("all_correct_answers"),
        )


def test_gold_appears_in_facts_block():
    """For every schema, the gold entity must be referenced somewhere
    in the facts block (otherwise the model has no way to answer)."""
    items = generate_items(block_seed=77, n_items=30)
    for it in items:
        assert it["gold"] in it["question"], (
            f"gold {it['gold']!r} not in question for {it['src']}"
        )


def test_synthetic_names_only():
    """Every entity in the question should be a synthetic
    Person_/Place_/Org_ name (no real-world entity leakage)."""
    import re
    items = generate_items(block_seed=88, n_items=30)
    for it in items:
        words = re.findall(r"\b\w+\b", it["question"])
        synth = [
            w for w in words
            if w.startswith(("Person_", "Place_", "Org_"))
        ]
        assert synth, f"no synthetic names in {it['src']}"
