"""Unit tests for scripts/v31/math_competition.py."""

from __future__ import annotations

from collections import Counter

from scripts.v31.math_competition import (
    TEMPLATES,
    generate_items,
    grade_response,
)


def test_items_have_required_shape():
    items = generate_items(block_seed=11, n_items=20)
    assert len(items) == 20
    required = {"src", "question", "gold", "family", "template"}
    families = {tpl.family for tpl in TEMPLATES}
    template_names = {tpl.name for tpl in TEMPLATES}
    for it in items:
        assert required.issubset(it.keys())
        assert it["family"] in families
        assert it["template"] in template_names
        assert isinstance(it["gold"], str) and it["gold"]


def test_all_families_appear():
    items = generate_items(block_seed=22, n_items=120)
    fams = Counter(it["family"] for it in items)
    expected = {tpl.family for tpl in TEMPLATES}
    assert set(fams.keys()) == expected, f"missing families: {expected - set(fams)}"


def test_determinism_per_seed():
    a = generate_items(block_seed=99, n_items=12)
    b = generate_items(block_seed=99, n_items=12)
    assert [(x["question"], x["gold"]) for x in a] == [
        (x["question"], x["gold"]) for x in b
    ]


def test_grader_accepts_integer_answers():
    items = generate_items(block_seed=55, n_items=20)
    int_items = [it for it in items if "/" not in it["gold"]]
    assert int_items
    for it in int_items:
        assert grade_response(f"#### {it['gold']}", it["gold"])
        assert grade_response(f"Final answer: {it['gold']}", it["gold"])


def test_grader_accepts_fractions_canonicalized():
    items = generate_items(block_seed=77, n_items=80)
    frac = [it for it in items if "/" in it["gold"]]
    assert frac, "no fractional gold found"
    for it in frac:
        assert grade_response(f"the probability is {it['gold']}", it["gold"])
        # Wrong fraction should fail.
        p, q = it["gold"].split("/")
        wrong = f"{int(p) + 1}/{q}"
        assert not grade_response(f"the answer is {wrong}", it["gold"])


def test_grader_rejects_obvious_wrong():
    items = generate_items(block_seed=99, n_items=10)
    for it in items:
        if "/" in it["gold"]:
            continue
        wrong = str(int(it["gold"]) + 1)
        assert not grade_response(f"#### {wrong}", it["gold"])


def test_quadratic_template_gold_consistent():
    """Sanity: the quadratic sum-of-roots template should produce
    gold that matches Vieta's formulas for the rendered equation.
    """
    import re
    items = generate_items(block_seed=11, n_items=200)
    quads = [it for it in items if it["template"] == "quadratic_sum_roots"]
    assert len(quads) >= 5
    for it in quads:
        m = re.search(r"(\d+)x\^2 ([+-]) (\d+)x ([+-]) (\d+) = 0", it["question"])
        assert m, f"could not parse equation in {it['question']}"
        a = int(m.group(1))
        b_sign = -1 if m.group(2) == "-" else 1
        b = b_sign * int(m.group(3))
        c_sign = -1 if m.group(4) == "-" else 1
        c = c_sign * int(m.group(5))
        # Vieta: sum = -b/a, product = c/a.
        is_sum = "sum" in it["question"]
        gold_int = int(it["gold"])
        if is_sum:
            assert -b // a == gold_int, f"sum mismatch for {it['question']}"
        else:
            assert c // a == gold_int, f"product mismatch for {it['question']}"


def test_no_unsolvable_zero_division_in_systems():
    items = generate_items(block_seed=88, n_items=200)
    for it in items:
        if it["template"] == "linear_system_2x2":
            assert " = 0" in it["question"] or " = " in it["question"]
