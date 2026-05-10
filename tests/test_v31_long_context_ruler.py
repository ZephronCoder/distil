"""Unit tests for scripts/v31/long_context_ruler.py."""

from __future__ import annotations

from collections import Counter

from scripts.v31.long_context_ruler import (
    _TASKS,
    generate_items,
    grade_response,
)


def test_items_have_required_shape():
    items = generate_items(block_seed=11, n_items=20)
    assert len(items) == 20
    required = {"src", "question", "gold", "task"}
    task_names = {n for n, _fn in _TASKS}
    for it in items:
        assert required.issubset(it.keys())
        assert it["task"] in task_names


def test_all_tasks_appear():
    items = generate_items(block_seed=22, n_items=200)
    tasks = Counter(it["task"] for it in items)
    expected = {n for n, _fn in _TASKS}
    assert set(tasks.keys()) == expected, (
        f"missing tasks: {expected - set(tasks.keys())}"
    )


def test_determinism_per_seed():
    a = generate_items(block_seed=99, n_items=10)
    b = generate_items(block_seed=99, n_items=10)
    assert [(x["question"], x["gold"]) for x in a] == [
        (x["question"], x["gold"]) for x in b
    ]


def test_haystack_substantial():
    """Each item's question should be at least ~500 chars (a small
    haystack fits in a few hundred chars)."""
    items = generate_items(block_seed=33, n_items=10)
    for it in items:
        assert len(it["question"]) > 500, (
            f"haystack too small for {it['src']}: {len(it['question'])}"
        )


def test_grader_accepts_canonical_format():
    items = generate_items(block_seed=51, n_items=8)
    for it in items:
        assert grade_response(f"Answer: {it['gold']}", it["gold"])


def test_grader_rejects_wrong_string():
    items = generate_items(block_seed=53, n_items=8)
    for it in items:
        if it["task"] == "aggregation_count":
            wrong = str(int(it["gold"]) + 1)
        elif it["task"] == "multihop_var":
            wrong = str(int(it["gold"]) + 1)
        else:
            wrong = "ZZZZZZ"
        assert not grade_response(f"Answer: {wrong}", it["gold"])


def test_niah_single_needle_is_in_passage():
    items = generate_items(block_seed=77, n_items=30)
    niah = [it for it in items if it["task"] == "niah_single"]
    assert niah
    for it in niah:
        # The gold value should literally appear in the question
        # (since we inserted "the secret code for KEY is VALUE").
        assert it["gold"] in it["question"]


def test_aggregation_count_target_word_appears_n_times():
    """The aggregation_count gold should equal the actual # of
    occurrences of the target word in the haystack body."""
    items = generate_items(block_seed=99, n_items=80)
    agg = [it for it in items if it["task"] == "aggregation_count"]
    assert agg
    for it in agg:
        target = it["target_word"]
        body_only = it["question"].split("Question:")[0]
        actual = body_only.count(target)
        assert actual == int(it["gold"]), (
            f"count mismatch for {target}: gold={it['gold']}, actual={actual}"
        )


def test_multihop_var_chain_is_consistent():
    """The first-variable assignment value should equal the gold."""
    import re
    items = generate_items(block_seed=88, n_items=60)
    mh = [it for it in items if it["task"] == "multihop_var"]
    assert mh
    for it in mh:
        m = re.search(r"Variable v_\w+ is set to (\d+)\.", it["question"])
        assert m, f"no initial assignment found: {it['question'][:300]}"
        assert m.group(1) == it["gold"]
