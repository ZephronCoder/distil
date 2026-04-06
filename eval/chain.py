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
                    commitments[uid] = {"block": block, "hotkey": hotkey, **parsed}
            except Exception:
                continue

    return commitments, uid_to_hotkey, uid_to_coldkey


def set_weights(subtensor, wallet, netuid: int, n_uids: int,
                weights: list[float], winner_uid: int, max_attempts: int = 3):
    """Set weights on-chain with retry.

    Args:
        subtensor: Bittensor subtensor instance
        wallet: Bittensor wallet for signing
        netuid: subnet ID
        n_uids: number of UIDs
        weights: weight array (1.0 for winner, 0.0 for all others)
        winner_uid: the UID receiving weight 1.0
        max_attempts: number of retry attempts
    """
    logger.info(f"Setting weights: UID {winner_uid} = 1.0")
    uids = list(range(n_uids))

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
            err_msg = result[1] if isinstance(result, (tuple, list)) and len(result) > 1 else str(result)
            logger.warning(f"Attempt {attempt + 1}: rejected — {err_msg}")
        except Exception as e:
            logger.error(f"Attempt {attempt + 1}: {e}")
        time.sleep(30)

    logger.error(f"Failed to set weights after {max_attempts} attempts")
