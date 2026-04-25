"""Focused unit tests for SINGLE_EVAL_MODE (one-eval-per-commitment policy).

The validator switched to "one registration → one commitment → one eval" on
2026-04-25. These tests pin the contracts that policy depends on:

* ``select_challengers`` only returns commitments not yet in
  ``state.composite_scores`` (or whose stored commit signature changed).
* ``add_top5_contenders``/``add_dormant_rotation``/``cap_challengers``/
  ``assert_top_contenders_present`` are no-ops with the flag on.
* ``select_king_by_composite`` reads cross-round and prefers higher worst.
* ``commitment_changed`` flags re-commits regardless of which field moved.
* ``evict_stale_evaluated_uids`` clears the gate when a re-commit lands.
* ``merge_composite_scores`` writes one record per non-DQ scored UID.
"""

import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _stub_torch_modules():
    """Provide a minimal torch stub so importing pod_eval_vllm-adjacent
    modules doesn't pull a real torch wheel into the test environment.
    Mirrors the stub used in test_procedural_bench_generation."""
    if "torch" in sys.modules:
        return
    fake = types.ModuleType("torch")
    fake.bfloat16 = object()
    fake.float32 = object()
    fake.long = object()
    nn_mod = types.ModuleType("torch.nn")
    nn_func = types.ModuleType("torch.nn.functional")
    nn_func.kl_div = lambda *a, **k: None
    nn_mod.functional = nn_func
    fake.nn = nn_mod
    fake.compile = lambda fn=None, **k: (fn or (lambda x: x))
    fake.cuda = types.ModuleType("torch.cuda")
    fake.cuda.is_available = lambda: False
    sys.modules["torch"] = fake
    sys.modules["torch.nn"] = nn_mod
    sys.modules["torch.nn.functional"] = nn_func
    sys.modules["torch.cuda"] = fake.cuda


_stub_torch_modules()


from scripts.validator import single_eval  # noqa: E402
from scripts.validator import challengers as ch_mod  # noqa: E402
from eval import state as state_mod  # noqa: E402


class _FakeState:
    """Minimal stand-in for ValidatorState (just the attributes the
    challenger planner and single-eval helpers read/write)."""

    def __init__(self):
        self.scores = {}
        self.evaluated_uids = set()
        self.composite_scores = {}
        self.permanently_bad_models = set()
        self.model_score_history = {}
        self.top4_leaderboard = {"king": None, "contenders": [], "phase": "maintenance"}
        self.h2h_latest = {}
        self.h2h_tested_against_king = {}
        self.dq_reasons = {}

    def save_top4(self):
        pass


def _commit(uid, model, block, hotkey="", revision="main", is_reference=False):
    return {
        uid: {
            "model": model,
            "revision": revision,
            "commit_block": block,
            "hotkey": hotkey,
            "is_reference": is_reference,
        }
    }


class CommitmentChangedTests(unittest.TestCase):
    def test_no_record_means_changed(self):
        self.assertTrue(single_eval.commitment_changed(None, {"model": "m", "revision": "main", "commit_block": 1}))
        self.assertTrue(single_eval.commitment_changed({}, {"model": "m", "revision": "main", "commit_block": 1}))

    def test_same_signature_is_unchanged(self):
        rec = {"model": "miner/x", "revision": "main", "block": 100}
        info = {"model": "miner/x", "revision": "main", "commit_block": 100}
        self.assertFalse(single_eval.commitment_changed(rec, info))

    def test_block_change_is_detected(self):
        rec = {"model": "miner/x", "revision": "main", "block": 100}
        info = {"model": "miner/x", "revision": "main", "commit_block": 200}
        self.assertTrue(single_eval.commitment_changed(rec, info))

    def test_model_change_is_detected(self):
        rec = {"model": "miner/x", "revision": "main", "block": 100}
        info = {"model": "miner/y", "revision": "main", "commit_block": 100}
        self.assertTrue(single_eval.commitment_changed(rec, info))


class EvictStaleEvaluatedUidsTests(unittest.TestCase):
    def test_evicts_re_commits_only(self):
        state = _FakeState()
        state.composite_scores = {
            "1": {"model": "miner/a", "revision": "main", "block": 100, "worst": 0.5},
            "2": {"model": "miner/b", "revision": "main", "block": 100, "worst": 0.4},
        }
        state.evaluated_uids = {"1", "2"}
        state.scores = {"1": 0.1, "2": 0.2}
        valid_models = {}
        valid_models.update(_commit(1, "miner/a-v2", 200))  # re-commit
        valid_models.update(_commit(2, "miner/b", 100))  # unchanged
        evicted = single_eval.evict_stale_evaluated_uids(state, valid_models)
        self.assertEqual(evicted, ["1"])
        self.assertNotIn("1", state.evaluated_uids)
        self.assertNotIn("1", state.composite_scores)
        self.assertIn("2", state.evaluated_uids)
        self.assertIn("2", state.composite_scores)


class SelectChallengersSingleEvalTests(unittest.TestCase):
    def setUp(self):
        os.environ["SINGLE_EVAL_MODE"] = "1"

    def tearDown(self):
        os.environ.pop("SINGLE_EVAL_MODE", None)

    def test_returns_only_never_composite_scored(self):
        state = _FakeState()
        state.composite_scores = {
            "1": {"model": "miner/a", "revision": "main", "block": 100, "worst": 0.5},
        }
        state.evaluated_uids = {"1"}
        state.scores = {"1": 0.1}
        valid_models = {}
        valid_models.update(_commit(1, "miner/a", 100))
        valid_models.update(_commit(2, "miner/b", 200))
        valid_models.update(_commit(3, "miner/c", 300))
        challengers = ch_mod.select_challengers(
            valid_models, state, king_uid=1, king_kl=0.1, epoch_count=1,
        )
        self.assertEqual(set(challengers.keys()), {2, 3})

    def test_picks_up_re_commits(self):
        state = _FakeState()
        state.composite_scores = {
            "1": {"model": "miner/a", "revision": "main", "block": 100, "worst": 0.5},
        }
        state.evaluated_uids = {"1"}
        valid_models = _commit(1, "miner/a", 200)  # new block = re-commit
        challengers = ch_mod.select_challengers(
            valid_models, state, king_uid=None, king_kl=float("inf"),
            epoch_count=1,
        )
        self.assertEqual(set(challengers.keys()), {1})
        self.assertNotIn("1", state.composite_scores)

    def test_skips_reference_uid(self):
        state = _FakeState()
        valid_models = _commit(-1, "Qwen/Qwen3.5-4B", 0, is_reference=True)
        valid_models.update(_commit(2, "miner/b", 200))
        challengers = ch_mod.select_challengers(
            valid_models, state, king_uid=None, king_kl=float("inf"),
            epoch_count=1,
        )
        self.assertEqual(set(challengers.keys()), {2})

    def test_skips_permanently_bad(self):
        state = _FakeState()
        state.permanently_bad_models = {"miner/bad"}
        valid_models = _commit(2, "miner/bad", 200)
        valid_models.update(_commit(3, "miner/c", 300))
        challengers = ch_mod.select_challengers(
            valid_models, state, king_uid=None, king_kl=float("inf"),
            epoch_count=1,
        )
        self.assertEqual(set(challengers.keys()), {3})


class ReEvalHelpersAreNoopsTests(unittest.TestCase):
    def setUp(self):
        os.environ["SINGLE_EVAL_MODE"] = "1"

    def tearDown(self):
        os.environ.pop("SINGLE_EVAL_MODE", None)

    def test_top5_contenders_noop(self):
        state = _FakeState()
        state.top4_leaderboard["contenders"] = [
            {"uid": 7, "model": "miner/x", "h2h_kl": 0.05},
        ]
        valid_models = _commit(7, "miner/x", 100)
        challengers = {}
        ch_mod.add_top5_contenders(challengers, valid_models, state, king_uid=1)
        self.assertEqual(challengers, {})

    def test_dormant_rotation_noop(self):
        state = _FakeState()
        state.scores = {"5": 0.05, "6": 0.06}
        valid_models = _commit(5, "miner/x", 100)
        valid_models.update(_commit(6, "miner/y", 200))
        challengers = {}
        ch_mod.add_dormant_rotation(challengers, valid_models, state, king_uid=1, king_kl=0.10)
        self.assertEqual(challengers, {})

    def test_cap_challengers_noop(self):
        state = _FakeState()
        challengers = {i: {"model": f"m/{i}", "commit_block": i} for i in range(50)}
        ch_mod.cap_challengers(challengers, state, king_uid=1)
        self.assertEqual(len(challengers), 50)  # nothing truncated

    def test_assert_top_contenders_noop(self):
        state = _FakeState()
        state.top4_leaderboard["contenders"] = [
            {"uid": 9, "model": "miner/lb", "h2h_kl": 0.05},
        ]
        valid_models = _commit(9, "miner/lb", 100)
        challengers = {}
        # Should not warn, force, or evict — single-eval has no concept of
        # "must be present every round."
        ch_mod.assert_top_contenders_present(challengers, valid_models, state, king_uid=1)
        self.assertEqual(challengers, {})


class MergeCompositeScoresTests(unittest.TestCase):
    def test_merges_one_record_per_scored_row(self):
        state = _FakeState()
        h2h_results = [
            {
                "uid": 5, "model": "miner/a",
                "composite": {"worst": 0.42, "weighted": 0.55,
                              "axes": {"kl": 0.6, "capability": 0.4},
                              "present_count": 2},
            },
            {  # DQ row should be skipped
                "uid": 6, "model": "miner/dq", "disqualified": True,
                "composite": {"worst": 0.30},
            },
            {  # Reference row should be skipped
                "uid": -1, "model": "Qwen/Qwen3.5-4B", "is_reference": True,
                "composite": {"worst": 0.50},
            },
            {  # No composite payload — skipped
                "uid": 7, "model": "miner/missing", "composite": {"worst": None},
            },
        ]
        models_to_eval = {
            5: {"model": "miner/a", "revision": "main", "commit_block": 1234},
            6: {"model": "miner/dq", "revision": "main", "commit_block": 1235},
            -1: {"model": "Qwen/Qwen3.5-4B", "is_reference": True},
            7: {"model": "miner/missing", "revision": "main", "commit_block": 1236},
        }
        n = single_eval.merge_composite_scores(state, h2h_results, models_to_eval, current_block=2000)
        self.assertEqual(n, 1)
        self.assertIn("5", state.composite_scores)
        record = state.composite_scores["5"]
        self.assertEqual(record["model"], "miner/a")
        self.assertEqual(record["block"], 1234)
        self.assertAlmostEqual(record["worst"], 0.42)
        self.assertAlmostEqual(record["weighted"], 0.55)


class SelectKingByCompositeTests(unittest.TestCase):
    def test_picks_highest_worst(self):
        state = _FakeState()
        state.composite_scores = {
            "1": {"worst": 0.5, "weighted": 0.55},
            "2": {"worst": 0.7, "weighted": 0.65},
            "3": {"worst": 0.6, "weighted": 0.62},
        }
        valid_models = {1: {"model": "a"}, 2: {"model": "b"}, 3: {"model": "c"}}
        uid, rec = single_eval.select_king_by_composite(state, valid_models)
        self.assertEqual(uid, 2)
        self.assertAlmostEqual(rec["worst"], 0.7)

    def test_ignores_dq_uids(self):
        state = _FakeState()
        state.composite_scores = {
            "1": {"worst": 0.5},
            "2": {"worst": 0.9},
        }
        # UID 2 is DQ'd at its current commit_block.
        state.dq_reasons = {"hk2:100": "anti-finetune detected"}
        valid_models = {
            1: {"model": "a", "hotkey": "hk1", "commit_block": 100},
            2: {"model": "b", "hotkey": "hk2", "commit_block": 100},
        }
        uid, rec = single_eval.select_king_by_composite(
            state, valid_models, uid_to_hotkey={1: "hk1", 2: "hk2"},
        )
        self.assertEqual(uid, 1)

    def test_returns_none_when_no_candidates(self):
        state = _FakeState()
        valid_models = {1: {"model": "a"}}
        uid, rec = single_eval.select_king_by_composite(state, valid_models)
        self.assertIsNone(uid)
        self.assertIsNone(rec)


class ResolveDethroneTests(unittest.TestCase):
    def test_no_incumbent_accepts_any_positive(self):
        ch = {"worst": 0.4}
        self.assertTrue(single_eval.resolve_dethrone(None, None, 5, ch))
        self.assertFalse(single_eval.resolve_dethrone(None, None, 5, {"worst": 0}))

    def test_margin_required(self):
        inc = {"worst": 0.50}
        # 3% margin: 0.50 * 1.03 = 0.515. Anything <= 0.515 should not dethrone.
        self.assertFalse(single_eval.resolve_dethrone(7, inc, 8, {"worst": 0.51}, margin=0.03))
        self.assertFalse(single_eval.resolve_dethrone(7, inc, 8, {"worst": 0.515}, margin=0.03))
        self.assertTrue(single_eval.resolve_dethrone(7, inc, 8, {"worst": 0.52}, margin=0.03))

    def test_self_returns_true(self):
        # Same UID compared against itself is always "wins" — used in the
        # crown-retention branch where the king's own composite is the
        # benchmark and no challenger has surpassed it.
        inc = {"worst": 0.5}
        self.assertTrue(single_eval.resolve_dethrone(7, inc, 7, inc))


class IsSingleEvalModeTests(unittest.TestCase):
    def test_default_off(self):
        os.environ.pop("SINGLE_EVAL_MODE", None)
        self.assertFalse(single_eval.is_single_eval_mode())

    def test_explicit_on(self):
        with patch.dict(os.environ, {"SINGLE_EVAL_MODE": "1"}):
            self.assertTrue(single_eval.is_single_eval_mode())

    def test_other_values_off(self):
        for v in ("0", "true", "yes", "ok"):
            with patch.dict(os.environ, {"SINGLE_EVAL_MODE": v}):
                if v == "1":
                    continue
                self.assertFalse(single_eval.is_single_eval_mode(),
                                 f"value={v!r} should be off")


class BackwardCompatTests(unittest.TestCase):
    """With the flag OFF, the legacy planner path must remain unchanged."""

    def setUp(self):
        os.environ.pop("SINGLE_EVAL_MODE", None)

    def test_legacy_select_challengers_unchanged(self):
        state = _FakeState()
        state.scores = {"1": 0.1}
        state.evaluated_uids = {"1"}
        valid_models = _commit(1, "miner/a", 100)
        valid_models.update(_commit(2, "miner/b", 200))
        challengers = ch_mod.select_challengers(
            valid_models, state, king_uid=1, king_kl=0.1, epoch_count=1,
        )
        # In legacy mode, UID 1 is skipped (already in evaluated+scores) and
        # UID 2 is included as a P3 (never-evaluated). That's the existing
        # behavior; we want to confirm the SINGLE_EVAL gate didn't change it.
        self.assertEqual(set(challengers.keys()), {2})


if __name__ == "__main__":
    unittest.main()
