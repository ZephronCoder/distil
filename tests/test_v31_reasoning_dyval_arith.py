"""Unit tests for scripts/v31/reasoning_dyval_arith.py."""

from __future__ import annotations

from collections import Counter

from scripts.v31.reasoning_dyval_arith import (
    _DIFFICULTY_DEPTHS,
    generate_items,
    grade_response,
)


def test_items_have_required_shape():
    items = generate_items(block_seed=11, n_items=20)
    assert len(items) == 20
    required = {"src", "question", "gold", "depth", "mode"}
    for it in items:
        assert required.issubset(it.keys())
        assert isinstance(it["question"], str) and len(it["question"]) > 30
        assert isinstance(it["gold"], str)
        # gold is a string-encoded int (possibly negative).
        int(it["gold"])


def test_depth_distribution():
    items = generate_items(block_seed=22, n_items=200)
    depths = Counter(it["depth"] for it in items)
    seen = set(depths.keys())
    expected = set(_DIFFICULTY_DEPTHS)
    assert seen.issubset(expected)
    assert len(seen) >= 4, f"low diversity: {depths}"


def test_two_modes_produced():
    items = generate_items(block_seed=33, n_items=80)
    modes = Counter(it["mode"] for it in items)
    assert modes.get("math", 0) > 5
    assert modes.get("nl_vars", 0) > 5


def test_determinism_per_seed():
    a = generate_items(block_seed=99, n_items=12)
    b = generate_items(block_seed=99, n_items=12)
    assert [(x["question"], x["gold"]) for x in a] == [
        (x["question"], x["gold"]) for x in b
    ]


def test_grader_accepts_canonical_format():
    items = generate_items(block_seed=51, n_items=10)
    for it in items:
        assert grade_response("Final answer: " + it["gold"], it["gold"])
        assert grade_response("\\boxed{" + it["gold"] + "}", it["gold"])
        assert grade_response("After computing, the result is " + it["gold"], it["gold"])


def test_grader_rejects_wrong():
    items = generate_items(block_seed=53, n_items=10)
    for it in items:
        target = int(it["gold"])
        wrong = str(target + 1)
        assert not grade_response("Final answer: " + wrong, it["gold"])
        assert not grade_response("", it["gold"])


def test_gold_matches_expression_value():
    """Sanity: the math-mode question contains an expression that
    evaluates (via Python) to exactly the gold value (using //
    semantics for the 'safe_div' op which is currently disabled - so
    we use plain math eval). This validates the rendered surface
    matches the computed gold.
    """
    items = generate_items(block_seed=77, n_items=20)
    for it in items:
        if it["mode"] != "math":
            continue
        # The expression appears between "= ?" pattern. Extract:
        import re
        m = re.search(r"\n\n(.+) = \?", it["question"], re.DOTALL)
        if not m:
            continue
        expr = m.group(1).strip()
        # Replace min(a, b) / max(a, b) with Python builtins; stay safe.
        local_ns = {"min": min, "max": max}
        try:
            v = eval(expr, {"__builtins__": {}}, local_ns)
        except Exception as e:
            raise AssertionError(f"expr {expr!r} failed: {e}")
        assert v == int(it["gold"]), f"expr={expr} -> {v}, gold={it['gold']}"


def test_no_division_by_zero():
    """Currently we don't include div in op set; enforce."""
    items = generate_items(block_seed=88, n_items=30)
    for it in items:
        assert "/" not in it["question"], "division removed for safety"
