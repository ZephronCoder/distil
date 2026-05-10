"""
Chain interaction for the SN97 validator.

Handles: metagraph fetching, commitment parsing, and weight setting.
All Bittensor RPC calls are wrapped with retry logic.
"""
import json
import logging
import time

logger = logging.getLogger("distillation.chain")


def _retry_chain(fn, max_attempts: int = 3, delay: float = 30, label: str = "chain RPC"):
    """Retry a chain RPC call with exponential backoff.

    Returns the result of fn() or raises on final failure.
    """
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            logger.warning(f"{label} failed (attempt {attempt + 1}/{max_attempts}): {e}")
            if attempt < max_attempts - 1:
                time.sleep(delay)
            else:
                raise


def fetch_metagraph(subtensor, netuid: int) -> tuple:
    """Fetch the metagraph and current block from the chain.

    Returns:
        (metagraph, current_block, block_hash) where block_hash may be None
        if the substrate call fails.
    """
    def _fetch():
        metagraph = subtensor.metagraph(netuid)
        current_block = subtensor.block
        block_hash = None
        try:
            block_hash = subtensor.substrate.get_block_hash(current_block)
        except Exception as bh_err:
            logger.warning(f"Block hash fetch failed: {bh_err}")
        return metagraph, current_block, block_hash

    return _retry_chain(_fetch, label="fetch_metagraph")


def parse_commitments(metagraph, revealed: dict, n_uids: int) -> tuple[dict, dict, dict]:
    """Parse revealed commitments into structured dicts.

    Args:
        metagraph: Bittensor metagraph object
        revealed: dict from subtensor.get_all_revealed_commitments()
        n_uids: number of UIDs in the metagraph

    Returns:
        (commitments, uid_to_hotkey, uid_to_coldkey) where:
        - commitments: {uid: {block, hotkey, model, revision, ...}}
        - uid_to_hotkey: {uid: hotkey_str}
        - uid_to_coldkey: {uid: coldkey_str}
    """
    commitments = {}
    uid_to_hotkey = {}
    uid_to_coldkey = {}

    for uid in range(n_uids):
        hotkey = str(metagraph.hotkeys[uid])
        uid_to_hotkey[uid] = hotkey
        try:
            uid_to_coldkey[uid] = str(metagraph.coldkeys[uid])
        except Exception:
            pass
        if hotkey in revealed and len(revealed[hotkey]) > 0:
            block, data = max(revealed[hotkey], key=lambda x: x[0])  # latest revealed commitment
            try:
                parsed = json.loads(data)
                if "model" in parsed:
                    commitments[uid] = {**parsed, "block": block, "hotkey": hotkey}
            except Exception:
                continue

    return commitments, uid_to_hotkey, uid_to_coldkey


def build_winner_take_all_weights(n_uids: int, winner_uid: int) -> list[float]:
    """Build a one-hot weight vector for the winning UID."""
    weights = [0.0] * max(n_uids, winner_uid + 1)
    weights[winner_uid] = 1.0
    return weights


def build_recent_kings_weights(
    n_uids: int,
    recent_kings: list[int],
    max_kings: int = 5,
) -> list[float]:
    """Build a weight vector that splits emission across recent kings.

    2026-05-01 (v30.4): replaces strict winner-takes-all on chain. The
    most recent ``max_kings`` distinct king UIDs each receive an equal
    share (1.0 / N for N ≤ ``max_kings``). When a UID has held the
    crown more than once we keep only the MOST RECENT entry (it stays
    in the payout queue once and is refreshed when re-crowned). When
    fewer than ``max_kings`` distinct kings have ever been crowned
    (boot phase, low-history validators), the available kings split
    1.0 equally so no emission is wasted.

    Rationale (Discord 2026-05-01): coffieex / svdeai07 / sebastian
    pointed out that winner-takes-all amplifies leaderboard noise —
    a 0.5 percent composite gap flips 100 percent of the emission.
    Splitting across the last 5 distinct kings smooths this without
    abandoning the king mechanic; a model that holds the crown for
    even one round earns ongoing incentive.

    Args:
        n_uids: metagraph size
        recent_kings: list of UIDs ordered MOST-RECENT FIRST (the
            current king is at index 0)
        max_kings: cap the split at this many UIDs (default 5)

    Returns:
        Length-``n_uids`` (or ``max(n_uids, last_king+1)``) list of
        floats summing to 1.0.
    """
    seen: list[int] = []
    for uid in recent_kings:
        try:
            uid_i = int(uid)
        except (TypeError, ValueError):
            continue
        if uid_i < 0:
            continue
        if uid_i in seen:
            continue
        seen.append(uid_i)
        if len(seen) >= max_kings:
            break
    if not seen:
        return [0.0] * n_uids
    weight_each = 1.0 / len(seen)
    out_len = max(n_uids, max(seen) + 1)
    weights = [0.0] * out_len
    for uid in seen:
        weights[uid] = weight_each
    return weights


def get_validator_weight_pairs(
    subtensor, netuid: int, validator_uid: int
) -> list[tuple[int, int]] | None:
    """Return the validator's full ``[(uid, raw_weight), ...]`` row from chain.

    Returns ``None`` if the validator hasn't submitted any weights yet
    (chain returns no row for the UID), or if the on-chain row is empty.
    Raw weights are u16 ints; callers that want a normalized view should
    divide by ``sum(weights)``.

    2026-05-09: introduced so callers can reason about multi-king splits.
    Pre-existing :func:`get_validator_weight_target` only surfaces a
    single UID, which silently lies under v30.4 multi-king payouts —
    every UID in the split shares the same raw weight, so ``max`` picks
    the lowest UID in the row regardless of which one is the live king.
    """

    def _fetch():
        rows = subtensor.weights(netuid)
        for row_uid, pairs in rows:
            if int(row_uid) != validator_uid:
                continue
            if not pairs:
                return None
            return [(int(uid), int(weight)) for uid, weight in pairs]
        return None

    return _retry_chain(_fetch, label="fetch_validator_weights")


def get_validator_weight_targets(
    subtensor, netuid: int, validator_uid: int
) -> set[int] | None:
    """Return the set of UIDs with non-zero on-chain weight for ``validator_uid``.

    ``None`` means the chain has no row for this validator (boot, or the
    UID never set weights). An empty set means the row is present but
    every weight is zero.

    Use this whenever you want to compare on-chain weights against a
    multi-king payout set. For a single dominant UID, prefer
    :func:`get_validator_weight_target`.
    """
    pairs = get_validator_weight_pairs(subtensor, netuid, validator_uid)
    if pairs is None:
        return None
    return {uid for uid, weight in pairs if weight > 0}


def get_validator_weight_target(subtensor, netuid: int, validator_uid: int) -> int | None:
    """Return one UID with the highest on-chain weight for this validator.

    .. deprecated:: v30.5
        Useful only for single-king-payout chains. Under v30.4 multi-king
        splits all targets share the same raw weight so the choice of
        ``max`` is arbitrary (today: lowest UID first because that's how
        the chain orders pairs). Prefer :func:`get_validator_weight_targets`
        (plural) when comparing against ``recent_kings``.
    """
    pairs = get_validator_weight_pairs(subtensor, netuid, validator_uid)
    if not pairs:
        return None
    best_uid, _ = max(pairs, key=lambda item: item[1])
    return best_uid


class SetWeightsError(RuntimeError):
    """Raised when set_weights exhausts all retries."""


def set_weights(subtensor, wallet, netuid: int, n_uids: int,
                weights: list[float], winner_uid: int, max_attempts: int = 3):
    """Set weights on-chain with retry.

    Raises SetWeightsError if every attempt fails so callers can decide
    whether to sleep + retry instead of silently leaving weights stale.
    """
    logger.info(f"Setting weights: UID {winner_uid} = 1.0")
    uids = list(range(n_uids))
    last_err: str | None = None

    for attempt in range(max_attempts):
        try:
            result = subtensor.set_weights(
                wallet=wallet, netuid=netuid,
                uids=uids, weights=weights,
                wait_for_inclusion=True,
                wait_for_finalization=True,
            )
            ok = result[0] if isinstance(result, (tuple, list)) else bool(result)
            if ok:
                logger.info("✓ Weights set on-chain!")
                return
            last_err = result[1] if isinstance(result, (tuple, list)) and len(result) > 1 else str(result)
            logger.warning(f"Attempt {attempt + 1}: rejected — {last_err}")
        except Exception as e:
            last_err = str(e)
            logger.error(f"Attempt {attempt + 1}: {e}")
        if attempt < max_attempts - 1:
            time.sleep(30)

    raise SetWeightsError(
        f"Failed to set weights (UID {winner_uid}) after {max_attempts} attempts: {last_err or 'unknown'}"
    )
