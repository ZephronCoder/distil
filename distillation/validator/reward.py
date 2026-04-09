"""
Reward / scoring logic for the GLM-5 distillation subnet.

Scoring:
  1. Verify the miner's declared model_size_params ≤ 74.4B (1/10 of GLM-5).
     If it exceeds this, score = 0.
  2. Compute KL-divergence between the teacher (GLM-5) logprobs and the miner's
     logprobs over the overlapping token sequence.
  3. Convert KL-divergence to a reward via  reward = exp(-kl_div).
  4. Winner-take-all: the miner with the highest reward (lowest KL) in each
     epoch gets ALL the weight (score = 1.0, others = 0.0).
"""

import math
import numpy as np
import bittensor as bt
from typing import List, Optional


# Maximum allowed model size in billions of parameters (1/10 of GLM-5's 744B)
MAX_MODEL_SIZE_B = 74.4


def compute_kl_divergence(
    teacher_logprobs: List[dict],
    miner_logprobs: List[dict],
) -> float:
    """
    Compute KL-divergence: KL(teacher || miner) = sum_i p_teacher(i) * [log p_teacher(i) - log p_miner(i)]

    Both inputs are lists of dicts with at minimum {"token": str, "logprob": float}.
    We align on the overlapping token sequence (by position).

    Returns:
        float: KL-divergence (non-negative). Returns float('inf') if sequences
               don't overlap or miner has missing logprobs.
    """
    if not teacher_logprobs or not miner_logprobs:
        return float("inf")

    # Use the minimum length of the two sequences
    n = min(len(teacher_logprobs), len(miner_logprobs))
    if n == 0:
        return float("inf")

    kl_sum = 0.0
    valid_tokens = 0

    for i in range(n):
        t_entry = teacher_logprobs[i]
        m_entry = miner_logprobs[i]

        t_logprob = t_entry.get("logprob")
        m_logprob = m_entry.get("logprob")

        if t_logprob is None or m_logprob is None:
            continue

        # p_teacher * (log p_teacher - log p_miner)
        # = exp(t_logprob) * (t_logprob - m_logprob)
        p_teacher = math.exp(t_logprob)

        # Clamp miner logprob to avoid -inf issues
        m_logprob_clamped = max(m_logprob, -100.0)

        kl_contribution = p_teacher * (t_logprob - m_logprob_clamped)
        kl_sum += kl_contribution
        valid_tokens += 1

    if valid_tokens == 0:
        return float("inf")

    # Average KL per token
    return kl_sum / valid_tokens


def reward_single(
    teacher_logprobs: Optional[List[dict]],
    miner_response: dict,
) -> float:
    """
    Compute reward for a single miner response.

    Args:
        teacher_logprobs: Reference logprobs from GLM-5 teacher.
        miner_response: Deserialized miner response dict with keys:
            completion, logprobs, model_size_params, model_name.

    Returns:
        float: Reward in [0, 1]. 0 means disqualified or terrible match.
    """
    if miner_response is None:
        return 0.0

    # Check model size constraint
    model_size = miner_response.get("model_size_params")
    if model_size is None or model_size > MAX_MODEL_SIZE_B:
        bt.logging.debug(
            f"Miner disqualified: model_size_params={model_size} > {MAX_MODEL_SIZE_B}B"
        )
        return 0.0

    miner_logprobs = miner_response.get("logprobs")
    if not miner_logprobs:
        return 0.0

    if not teacher_logprobs:
        bt.logging.warning("No teacher logprobs available for comparison.")
        return 0.0

    kl_div = compute_kl_divergence(teacher_logprobs, miner_logprobs)

    if math.isinf(kl_div) or math.isnan(kl_div):
        return 0.0

    # Reward = exp(-kl_div) — higher reward for lower KL
    reward_val = math.exp(-kl_div)

    bt.logging.debug(
        f"Miner {miner_response.get('model_name', '?')}: "
        f"KL={kl_div:.4f}, reward={reward_val:.4f}, "
        f"size={model_size}B"
    )

    return reward_val


def get_rewards(
    self,
    teacher_logprobs: Optional[List[dict]],
    responses: List[dict],
) -> np.ndarray:
    """
    Compute rewards for all miner responses and apply winner-take-all.

    The miner with the highest reward (lowest KL-divergence) receives a
    score of 1.0; all others receive 0.0.

    Args:
        self: Validator instance.
        teacher_logprobs: Reference logprobs from GLM-5 teacher.
        responses: List of deserialized miner response dicts.

    Returns:
        np.ndarray: Array of rewards, one per miner.
    """
    raw_rewards = np.array(
        [reward_single(teacher_logprobs, resp) for resp in responses],
        dtype=np.float32,
    )

    bt.logging.info(f"Raw rewards: {raw_rewards}")

    # Winner-take-all: only the best miner gets weight
    if raw_rewards.sum() > 0:
        winner_idx = np.argmax(raw_rewards)
        wta_rewards = np.zeros_like(raw_rewards)
        wta_rewards[winner_idx] = 1.0
        bt.logging.info(
            f"Winner-take-all: uid index {winner_idx} wins with "
            f"reward={raw_rewards[winner_idx]:.4f}"
        )
        return wta_rewards
    else:
        bt.logging.warning("No valid miner responses this epoch.")
        return raw_rewards
