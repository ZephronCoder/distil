"""Unit tests for scripts/v31/code_humaneval_plus.py.

Verifies item shape, template diversity, that the procedurally
generated reference solution PASSES the procedurally generated test
suite (round-trip correctness), and that obviously broken submissions
FAIL the test suite.
"""

from __future__ import annotations

import re
from collections import Counter

from scripts.v31.code_humaneval_plus import (
    TEMPLATES,
    generate_items,
)


def test_items_have_required_shape():
    items = generate_items(block_seed=11, n_items=12)
    assert len(items) == 12
    required = {"task_id", "entry_point", "prompt", "test", "src", "template", "n_test_cases"}
    template_names = {t.name for t in TEMPLATES}
    for it in items:
        assert required.issubset(it.keys())
        assert it["template"] in template_names
        # entry_point ends with a 6-letter lowercase hash suffix.
        assert re.match(r"^[a-z_]+_[a-z]{6}$", it["entry_point"]), (
            f"unexpected entry_point format: {it['entry_point']!r}"
        )
        # Function name matches what's in the prompt.
        assert it["entry_point"] in it["prompt"]
        assert it["n_test_cases"] >= 30, (
            f"too few tests for {it['task_id']}: {it['n_test_cases']}"
        )


def test_all_templates_appear():
    items = generate_items(block_seed=22, n_items=120)
    tpls = Counter(it["template"] for it in items)
    expected = {t.name for t in TEMPLATES}
    assert set(tpls.keys()) == expected, f"missing: {expected - set(tpls)}"


def test_determinism_per_seed():
    a = generate_items(block_seed=99, n_items=8)
    b = generate_items(block_seed=99, n_items=8)
    assert [(x["task_id"], x["test"]) for x in a] == [
        (x["task_id"], x["test"]) for x in b
    ]


def _execute_in_sandbox(prompt: str, body: str, test_block: str, entry_point: str) -> tuple[bool, str | None]:
    """Run the test block against the function defined by prompt+body.

    The prompt contains the signature + docstring; we splice in the
    submitted ``body`` (indented), then exec the result in a fresh
    namespace. Returns (passed, error_message).
    """
    # Indent body to be inside the function.
    indented = "\n".join("    " + ln if ln else "" for ln in body.splitlines())
    full_source = prompt + indented + "\n\n" + test_block + f"\ncheck({entry_point})\n"
    ns: dict = {}
    try:
        exec(full_source, ns)
    except AssertionError as e:
        return False, f"AssertionError: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    return True, None


def _reference_body_for_template(template_name: str, prompt: str) -> str:
    """Return a Python body (raw, unindented) that implements the
    spec correctly. The sandbox helper will indent it 4 spaces."""
    if template_name == "count_in_list":
        return f"return sum(1 for x in {_arg_name(prompt, 0)} if x == {_arg_name(prompt, 1)})\n"
    if template_name == "reverse_words":
        return f"return ' '.join(w[::-1] for w in {_arg_name(prompt, 0)}.split(' '))\n"
    if template_name == "filter_above":
        return f"return [x for x in {_arg_name(prompt, 0)} if x > {_arg_name(prompt, 1)}]\n"
    if template_name == "dict_value_sum":
        return f"return sum({_arg_name(prompt, 0)}.values())\n"
    if template_name == "is_palindrome":
        a = _arg_name(prompt, 0)
        return f"return {a} == {a}[::-1]\n"
    if template_name == "max_consecutive_run":
        a = _arg_name(prompt, 0)
        return (
            f"if not {a}:\n"
            f"    return 0\n"
            f"best = cur = 1\n"
            f"for i in range(1, len({a})):\n"
            f"    if {a}[i] == {a}[i-1]:\n"
            f"        cur += 1\n"
            f"        best = max(best, cur)\n"
            f"    else:\n"
            f"        cur = 1\n"
            f"return best\n"
        )
    raise ValueError(template_name)


def _arg_name(prompt: str, idx: int) -> str:
    m = re.match(r"def \w+\(([^)]*)\):", prompt)
    assert m, f"no signature: {prompt[:80]}"
    args = [a.strip() for a in m.group(1).split(",")]
    return args[idx]


def test_reference_passes_all_test_cases():
    """The procedurally generated reference for each template must
    pass its own procedurally generated test suite (round-trip)."""
    items = generate_items(block_seed=33, n_items=30)
    by_tpl: dict = {}
    for it in items:
        by_tpl.setdefault(it["template"], []).append(it)
    for tpl_name, tpl_items in by_tpl.items():
        # Check the first item per template (sufficient for round-trip).
        it = tpl_items[0]
        body = _reference_body_for_template(tpl_name, it["prompt"])
        ok, err = _execute_in_sandbox(
            it["prompt"], body, it["test"], it["entry_point"]
        )
        assert ok, f"{tpl_name} reference failed: {err}"


def test_broken_submission_fails():
    """A trivially broken implementation (always returns None) should
    fail at least one test case for every template."""
    items = generate_items(block_seed=51, n_items=12)
    for it in items:
        body = "return None\n"
        ok, err = _execute_in_sandbox(
            it["prompt"], body, it["test"], it["entry_point"]
        )
        assert not ok, f"broken submission unexpectedly passed for {it['task_id']}"


def test_test_block_has_check_function():
    items = generate_items(block_seed=66, n_items=6)
    for it in items:
        assert "def check(candidate):" in it["test"]
        assert "assert candidate(" in it["test"]


def test_entry_point_in_prompt():
    items = generate_items(block_seed=77, n_items=8)
    for it in items:
        assert it["entry_point"] in it["prompt"]
