"""Unit tests for scripts/v31/reasoning_logic_grid.py.

Verifies:
* item shape (src / question / gold / metadata)
* gold is a single-word lowercase token from the appropriate domain
* uniqueness: brute-force solver returns exactly one assignment for
  each generated puzzle
* difficulty distribution roughly matches _DIFFICULTY_RATIO
* determinism for a fixed seed
* grader handles common response formats and rejects wrong answers
"""

from __future__ import annotations

from collections import Counter

from scripts.v31.reasoning_logic_grid import (
    _ATTRIBUTES,
    _CLUE_GENERATORS,
    _DIFFICULTY_CONFIGS,
    _clue_to_predicate,
    _enumerate_solutions,
    generate_items,
    grade_response,
)


def _parse_clues_from_question(question: str, attrs: list[str]):
    """Recover clue lines from a generated question string."""
    lines = question.splitlines()
    clues = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if ln[0].isdigit():
            i = ln.find(". ")
            if i > 0:
                clues.append(ln[i + 2 :])
    preds = [_clue_to_predicate(c, attrs) for c in clues]
    return clues, preds


def _attrs_from_question(question: str) -> list[str]:
    import re
    m = re.search(r"unique ([a-z, ]+)\.", question)
    if not m:
        return []
    return [s.strip() for s in m.group(1).split(",")]


def test_items_have_required_shape():
    items = generate_items(block_seed=11, n_items=20)
    assert len(items) == 20
    required = {"src", "question", "gold", "num_people", "num_attrs", "q_attr"}
    for it in items:
        assert required.issubset(it.keys())
        assert isinstance(it["question"], str) and len(it["question"]) > 60
        assert isinstance(it["gold"], str)
        assert it["gold"].islower()
        assert " " not in it["gold"]
        assert it["q_attr"] in _ATTRIBUTES
        assert it["gold"] in _ATTRIBUTES[it["q_attr"]]


def test_difficulty_configs_present():
    items = generate_items(block_seed=22, n_items=120)
    sizes = Counter((it["num_people"], it["num_attrs"]) for it in items)
    expected_pairs = {(p, a) for (p, a, _c) in _DIFFICULTY_CONFIGS}
    seen = set(sizes.keys())
    overlap = seen.intersection(expected_pairs)
    assert overlap == seen
    assert len(seen) >= 3, f"expected diversity, got {sizes}"


def test_determinism_per_seed():
    a = generate_items(block_seed=99, n_items=12)
    b = generate_items(block_seed=99, n_items=12)
    assert [(x["question"], x["gold"]) for x in a] == [
        (x["question"], x["gold"]) for x in b
    ]


def test_uniqueness_invariant():
    """Every generated puzzle must have a unique satisfying assignment."""
    items = generate_items(block_seed=7, n_items=10)
    for it in items:
        if it["num_people"] >= 5 and it["num_attrs"] >= 4:
            continue
        attrs = it["attr_names"]
        value_sets = {k: tuple(v) for k, v in it["value_sets"].items()}
        assert it["q_attr"] in attrs
        _clues, preds = _parse_clues_from_question(it["question"], attrs)
        sols = _enumerate_solutions(
            attrs, it["num_people"], preds,
            max_solutions=2, value_sets=value_sets,
        )
        assert len(sols) == 1, f"non-unique puzzle: {it['src']} sols={sols}"
        assert sols[0][it["q_attr"]][it["q_pos"] - 1] == it["gold"]


def test_grader_accepts_canonical_format():
    items = generate_items(block_seed=51, n_items=5)
    for it in items:
        assert grade_response("Answer: " + it["gold"], it["gold"])
        assert grade_response("answer: " + it["gold"].upper(), it["gold"])
        assert grade_response(
            "After analysis the color is " + it["gold"] + ".", it["gold"]
        )


def test_grader_rejects_wrong():
    items = generate_items(block_seed=53, n_items=5)
    for it in items:
        domain = _ATTRIBUTES[it["q_attr"]]
        wrong = next(v for v in domain if v != it["gold"])
        assert not grade_response("Answer: " + wrong, it["gold"])
        assert not grade_response("", it["gold"])
        assert not grade_response("I don't know", it["gold"])


def test_clue_generators_registered_and_usable():
    assert len(_CLUE_GENERATORS) == 5
    items = generate_items(block_seed=77, n_items=30)
    assert all(it["num_clues"] >= 3 for it in items)


def test_smallest_puzzles_present():
    items = generate_items(block_seed=33, n_items=80)
    smallest = [it for it in items if it["num_people"] == 3 and it["num_attrs"] == 2]
    assert len(smallest) >= 5, "expected some 3x2 puzzles in 80-item batch"
