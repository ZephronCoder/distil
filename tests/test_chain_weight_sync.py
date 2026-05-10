"""Regression tests for the v30.4 multi-king payout split + the
``_sync_king_weights`` short-circuit on chain-equal-to-expected.

Pre-2026-05-09 the validator compared the chain's *first-tied-max* UID
to ``king_uid`` directly, so under multi-king splits every UID in the
payout set tied at the same raw weight and the chain looked "stale"
forever. Result: ~150 doomed ``set_weights`` calls per 8 hours, each
rate-limited by the chain ("too soon to commit weights"), plus a
cascade of misleading "stale weights" warnings. These tests pin the
new set-based comparison and ensure the helpers exposed by
``eval.chain`` keep their contract.
"""

from __future__ import annotations

import os
import sys
import types
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from eval.chain import (  # noqa: E402
    build_recent_kings_weights,
    build_winner_take_all_weights,
    get_validator_weight_pairs,
    get_validator_weight_target,
    get_validator_weight_targets,
)


class _FakeSubtensor:
    """Minimal stand-in for ``bittensor.subtensor`` exposing only
    ``weights(netuid)``. Returns the supplied rows verbatim so tests
    can pin both the multi-king split and edge cases (empty, missing,
    non-monotone)."""

    def __init__(self, rows):
        self._rows = rows

    def weights(self, _netuid):
        return self._rows


class GetWeightsTargetsTests(unittest.TestCase):
    def test_returns_full_set_for_multi_king_split(self):
        # On-chain payout: 5 recent kings tied at u16 max, exactly what
        # ``build_recent_kings_weights`` writes in production.
        rows = [(0, [(183, 65535), (190, 65535), (198, 65535), (200, 65535), (201, 65535)])]
        sub = _FakeSubtensor(rows)
        self.assertEqual(
            get_validator_weight_targets(sub, 97, 0),
            {183, 190, 198, 200, 201},
        )

    def test_singular_target_is_first_tied_max_for_back_compat(self):
        # The legacy single-target getter must keep returning a UID
        # in the tied set so callers that haven't migrated still get
        # *something* truthy (only the *meaning* changes — pre-v30.4
        # this UID was the unique winner; today it's just a member of
        # the multi-king split).
        rows = [(0, [(183, 65535), (190, 65535), (198, 65535), (200, 65535), (201, 65535)])]
        sub = _FakeSubtensor(rows)
        target = get_validator_weight_target(sub, 97, 0)
        self.assertIn(target, {183, 190, 198, 200, 201})

    def test_returns_none_when_validator_has_no_row(self):
        sub = _FakeSubtensor([(7, [(1, 100)])])
        self.assertIsNone(get_validator_weight_targets(sub, 97, 0))
        self.assertIsNone(get_validator_weight_target(sub, 97, 0))
        self.assertIsNone(get_validator_weight_pairs(sub, 97, 0))

    def test_returns_empty_set_when_row_present_but_zero_weights(self):
        sub = _FakeSubtensor([(0, [(1, 0), (2, 0)])])
        # An all-zero row is distinct from a missing row: chain knows
        # about us but we've actively cleared every weight. Surface
        # that as ``set()`` so callers can flag it as drift.
        self.assertEqual(get_validator_weight_targets(sub, 97, 0), set())

    def test_singular_returns_uid_with_max_weight_when_not_tied(self):
        # Mixed weights — ensures the getter still picks the unique max.
        rows = [(0, [(183, 100), (190, 65535), (198, 200), (200, 50)])]
        sub = _FakeSubtensor(rows)
        self.assertEqual(get_validator_weight_target(sub, 97, 0), 190)
        self.assertEqual(
            get_validator_weight_targets(sub, 97, 0), {183, 190, 198, 200}
        )


class ExpectedPayoutUidsTests(unittest.TestCase):
    """The helper that mirrors ``_build_emission_weights`` for the
    set-based comparison in ``_sync_king_weights``."""

    def _state(self, recent_kings):
        s = types.SimpleNamespace()
        s.recent_kings = list(recent_kings)
        return s

    def test_includes_live_king_at_front_even_if_missing_from_history(self):
        # Live king just dethroned the previous one and hasn't been
        # persisted into recent_kings yet (the gap before
        # ``_record_king_change``). The expected payout set must still
        # cover the live king.
        from scripts.validator.service import _expected_payout_uids

        state = self._state([201, 198, 183, 190])
        expected = _expected_payout_uids(state, king_uid=200)
        self.assertEqual(expected, {200, 201, 198, 183, 190})

    def test_dedupes_live_king_already_in_history(self):
        from scripts.validator.service import _expected_payout_uids

        state = self._state([200, 201, 198, 183, 190])
        expected = _expected_payout_uids(state, king_uid=200)
        self.assertEqual(expected, {200, 201, 198, 183, 190})

    def test_caps_at_recent_kings_max(self):
        from scripts.validator.service import _expected_payout_uids
        from eval.state import RECENT_KINGS_MAX

        state = self._state(list(range(100, 100 + RECENT_KINGS_MAX + 5)))
        expected = _expected_payout_uids(state, king_uid=200)
        self.assertEqual(len(expected), RECENT_KINGS_MAX)
        self.assertIn(200, expected)


class SyncKingWeightsTests(unittest.TestCase):
    def _make(self, chain_pairs, recent_kings, king_uid):
        from scripts.validator import service as svc

        calls: list[tuple[str, tuple]] = []

        def _fake_targets(_sub, _netuid, _vuid):
            return {uid for uid, w in chain_pairs if w > 0}

        def _fake_safe_set_weights(_sub, _wallet, _netuid, _n_uids, _w, _winner, _state_dir):
            calls.append(("set_weights", (_winner,)))
            return True

        state = types.SimpleNamespace(recent_kings=list(recent_kings))
        return svc, calls, state, _fake_targets, _fake_safe_set_weights

    def test_skips_when_chain_set_matches_expected(self):
        """Reproduces the 2026-05-09 false-positive: chain weights split
        across 5 recent kings; pre-fix this triggered set_weights every
        epoch (and every call hit "too soon to commit"). The fix must
        return early with no chain write."""
        chain_pairs = [(183, 65535), (190, 65535), (198, 65535), (200, 65535), (201, 65535)]
        svc, calls, state, fake_targets, fake_safe = self._make(
            chain_pairs, recent_kings=[200, 201, 198, 183, 190], king_uid=200,
        )
        import unittest.mock as mock

        with mock.patch.object(svc, "get_validator_weight_targets", fake_targets), \
             mock.patch.object(svc, "_safe_set_weights", fake_safe):
            svc._sync_king_weights(
                subtensor=object(), wallet=object(), netuid=97, n_uids=300,
                king_uid=200, validator_uid=0, state_dir="/tmp", state=state,
            )
        self.assertEqual(calls, [], "Should not call set_weights when chain matches expected")

    def test_syncs_when_chain_missing_live_king(self):
        # Chain still has the OLD top-5 (no UID 200); we just dethroned
        # so the live king isn't on chain yet → must sync.
        chain_pairs = [(183, 65535), (188, 65535), (190, 65535), (198, 65535), (201, 65535)]
        svc, calls, state, fake_targets, fake_safe = self._make(
            chain_pairs, recent_kings=[200, 201, 198, 183, 190], king_uid=200,
        )
        import unittest.mock as mock

        with mock.patch.object(svc, "get_validator_weight_targets", fake_targets), \
             mock.patch.object(svc, "_safe_set_weights", fake_safe):
            svc._sync_king_weights(
                subtensor=object(), wallet=object(), netuid=97, n_uids=300,
                king_uid=200, validator_uid=0, state_dir="/tmp", state=state,
            )
        self.assertEqual(calls, [("set_weights", (200,))])

    def test_no_sync_when_validator_uid_missing(self):
        svc, calls, state, fake_targets, fake_safe = self._make(
            [], recent_kings=[200], king_uid=200,
        )
        import unittest.mock as mock
        with mock.patch.object(svc, "_safe_set_weights", fake_safe):
            svc._sync_king_weights(
                subtensor=object(), wallet=object(), netuid=97, n_uids=300,
                king_uid=200, validator_uid=None, state_dir="/tmp", state=state,
            )
        self.assertEqual(calls, [])

    def test_no_sync_when_king_uid_missing(self):
        svc, calls, state, fake_targets, fake_safe = self._make(
            [], recent_kings=[], king_uid=None,
        )
        import unittest.mock as mock
        with mock.patch.object(svc, "_safe_set_weights", fake_safe):
            svc._sync_king_weights(
                subtensor=object(), wallet=object(), netuid=97, n_uids=300,
                king_uid=None, validator_uid=0, state_dir="/tmp", state=state,
            )
        self.assertEqual(calls, [])


class BuildWeightsRoundTripTests(unittest.TestCase):
    """Sanity-check that the set produced by
    ``build_recent_kings_weights`` is exactly what
    ``_expected_payout_uids`` predicts. If the two ever drift the
    sync helper would either spam or miss a needed update."""

    def test_round_trip_matches_expected_payout_set(self):
        from scripts.validator.service import _expected_payout_uids

        state = types.SimpleNamespace(recent_kings=[201, 198, 183, 190])
        expected = _expected_payout_uids(state, king_uid=200)
        # Build emission weights with the same input the validator uses.
        history = [200] + state.recent_kings  # mirrors _build_emission_weights
        weights = build_recent_kings_weights(n_uids=300, recent_kings=history, max_kings=5)
        non_zero = {i for i, w in enumerate(weights) if w > 0}
        self.assertEqual(non_zero, expected)

    def test_winner_take_all_fallback(self):
        weights = build_winner_take_all_weights(n_uids=300, winner_uid=42)
        self.assertEqual(weights[42], 1.0)
        self.assertEqual(sum(weights), 1.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
