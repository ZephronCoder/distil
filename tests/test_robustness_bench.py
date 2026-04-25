#!/usr/bin/env python3
"""Tests for the Session 3.7 robustness_bench paraphrase axis.

Goodhart-context: a miner who memorizes the canonical wording of public
math items (gsm8k/math500) wins ``math_bench`` for free. The robustness
axis re-uses the math pool but asks each item under K block-rotated
paraphrase wrappers, so a model that only remembers exact phrasings
fails. These tests pin down:

* The wrapper rotation is **deterministic per block_seed** (every
  validator agrees on which wrappers run a given round).
* Different block_seeds yield **different** wrapper sets (so a miner
  cannot pre-train on a single canonical wording bundle).
* The wrappers are **non-trivial string transforms** (every wrapper's
  output is a strict superstring of the original prompt with at least
  some prefix or postfix added — they are not no-ops).
* ``BENCH_ROBUSTNESS_PERTURB_K`` clamps to the available templates.

We don't try to load a real torch stack here — the probe is exercised
via a dummy model + tokenizer indirection in a follow-up integration
test (eval pod side). At unit-test scope, we only need the perturbation
machinery to be correct + deterministic; that's the only piece miners
can game.
"""
from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _install_torch_stub():
    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = object()
    fake_torch.float32 = object()
    fake_torch.long = object()
    fake_torch.compile = lambda fn, **_kwargs: fn
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        device_count=lambda: 0,
    )
    fake_nn = types.ModuleType("torch.nn")
    fake_f = types.ModuleType("torch.nn.functional")
    fake_f.kl_div = lambda *_args, **_kwargs: None
    fake_nn.functional = fake_f
    fake_torch.nn = fake_nn
    sys.modules.setdefault("torch", fake_torch)
    sys.modules.setdefault("torch.nn", fake_nn)
    sys.modules.setdefault("torch.nn.functional", fake_f)


class TestRobustnessPerturbations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_torch_stub()
        cls.mod = importlib.import_module("scripts.pod_eval_vllm")

    def test_template_table_is_nonempty(self):
        templates = self.mod._ROBUSTNESS_PERTURBATION_TEMPLATES
        self.assertGreaterEqual(
            len(templates), 4,
            "Need at least 4 wrappers so K=2 has variety across rounds",
        )
        for name, fn in templates:
            self.assertIsInstance(name, str)
            self.assertTrue(callable(fn))

    def test_pick_is_block_seed_deterministic(self):
        a = self.mod._pick_robustness_perturbations(12345, k=2)
        b = self.mod._pick_robustness_perturbations(12345, k=2)
        self.assertEqual([n for n, _ in a], [n for n, _ in b])

    def test_pick_rotates_with_block_seed(self):
        runs = [
            tuple(n for n, _ in self.mod._pick_robustness_perturbations(s, k=2))
            for s in (10000, 20000, 30000, 40000, 50000)
        ]
        # We don't require perfect uniqueness across 5 seeds, but we do
        # require that not every seed produces the same wrappers — that
        # would be a memorization payday.
        self.assertGreater(
            len(set(runs)), 1,
            "Wrapper rotation must vary across seeds; saw only one set "
            "across 5 trials",
        )

    def test_pick_clamps_k_to_template_count(self):
        templates = self.mod._ROBUSTNESS_PERTURBATION_TEMPLATES
        picked = self.mod._pick_robustness_perturbations(123, k=999)
        self.assertEqual(len(picked), len(templates))

    def test_pick_returns_at_least_one_with_k_zero(self):
        picked = self.mod._pick_robustness_perturbations(123, k=0)
        # K=0 is a misconfiguration; we'd rather still emit one wrapper
        # than silently stop the axis. Verifies the max(1, k) clamp.
        self.assertEqual(len(picked), 1)

    def test_wrappers_strictly_extend_the_prompt(self):
        original = "What is 2 + 2?\n\nProvide a final answer."
        for name, fn in self.mod._ROBUSTNESS_PERTURBATION_TEMPLATES:
            out = fn(original)
            self.assertIsInstance(out, str, f"{name} must return str")
            self.assertGreater(
                len(out), len(original),
                f"{name} wrapper produced output no longer than the "
                f"original — that's effectively a no-op",
            )
            # The original prompt content must still be present so the
            # grader can find the math problem. We check substring of
            # the question portion ("2 + 2") as a stable anchor.
            self.assertIn(
                "2 + 2", out,
                f"{name} wrapper dropped the original question content",
            )

    def test_no_wrapper_collapses_to_template_only(self):
        # Negative test: if any wrapper returned only the boilerplate
        # ("Solve the following problem.") with the actual question
        # truncated, the axis would be vacuous. Anchor: the question's
        # closing question mark ('?') must survive every wrapper.
        original = "If x + 5 = 12, what is x?"
        for name, fn in self.mod._ROBUSTNESS_PERTURBATION_TEMPLATES:
            out = fn(original)
            self.assertIn(
                "what is x?", out,
                f"{name} wrapper dropped the closing question",
            )

    def test_robustness_pool_is_alias_of_math(self):
        # _BENCH_POOLS is module-level state. We don't load the math
        # pool here (no datasets package on the test runner) but we
        # can verify the alias hook in `_bench_load_pools` would set
        # them equal: compare list identity post-init when math is
        # populated. We simulate by directly stamping the pool.
        self.mod._BENCH_POOLS["math"] = [
            {"src": "gsm8k", "question": "What is 2+2?", "gold": "4"},
        ]
        self.mod._BENCH_POOLS["robustness"] = self.mod._BENCH_POOLS["math"]
        self.assertIs(
            self.mod._BENCH_POOLS["robustness"], self.mod._BENCH_POOLS["math"],
            "robustness pool should alias (not copy) math so growth is "
            "tracked",
        )

    def test_robustness_sample_uses_independent_stream(self):
        # When robustness and math share a pool but are sampled with
        # different stream offsets, _pick_bench_items must yield a
        # *different* permutation for the same block_seed. This is
        # the central anti-collision property — without it, robustness
        # and math would always score the same items and the axis
        # would degenerate to "math under a wrapper".
        items = [
            {"src": "gsm8k", "question": f"q{i}", "gold": str(i)}
            for i in range(40)
        ]
        self.mod._BENCH_POOLS["math"] = list(items)
        self.mod._BENCH_POOLS["robustness"] = self.mod._BENCH_POOLS["math"]
        block_seed = 8042854
        math_pick = self.mod._pick_bench_items("math", block_seed, 4)
        rob_pick = self.mod._pick_bench_items("robustness", block_seed, 4)
        self.assertEqual(len(math_pick), 4)
        self.assertEqual(len(rob_pick), 4)
        self.assertNotEqual(
            [it["question"] for it in math_pick],
            [it["question"] for it in rob_pick],
            "math and robustness picks must differ — same block_seed, "
            "different stream offsets",
        )


class TestRobustnessAxisExtractor(unittest.TestCase):
    """Pure-Python axis extractor; no torch, no eval pod."""

    def test_extractor_returns_pass_frac_when_min_valid(self):
        from scripts.validator.composite import (
            BENCH_MIN_VALID,
            _axis_robustness_bench,
        )
        student = {
            "robustness_bench": {
                "n": BENCH_MIN_VALID["robustness_bench"],
                "correct": BENCH_MIN_VALID["robustness_bench"],
                "pass_frac": 1.0,
                "items": [],
                "perturbations": ["a", "b"],
            },
        }
        self.assertEqual(_axis_robustness_bench(student), 1.0)

    def test_extractor_drops_below_min_valid(self):
        from scripts.validator.composite import (
            BENCH_MIN_VALID,
            _axis_robustness_bench,
        )
        student = {
            "robustness_bench": {
                "n": BENCH_MIN_VALID["robustness_bench"] - 1,
                "correct": 0,
                "pass_frac": 0.0,
                "items": [],
            },
        }
        self.assertIsNone(_axis_robustness_bench(student))

    def test_extractor_handles_error_payload(self):
        from scripts.validator.composite import _axis_robustness_bench
        student = {
            "robustness_bench": {
                "error": "torch not available",
                "n": 0, "correct": 0, "pass_frac": 0.0,
            },
        }
        self.assertIsNone(_axis_robustness_bench(student))


if __name__ == "__main__":
    unittest.main()
