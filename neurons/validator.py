"""
GLM-5 Distillation Subnet — Validator Neuron

The validator:
1. Selects a coding prompt from a curated pool
2. Queries the GLM-5 teacher model (via Z.AI API) for reference logprobs
3. Sends the prompt to miners via DistillationSynapse
4. Scores miners by KL-divergence between their logprobs and the teacher's
5. Applies winner-take-all: only the miner with the lowest KL-divergence
   receives weight for that epoch
6. Verifies each miner's declared model size is ≤ 74.4B params

Environment variables:
    ZAI_API_KEY: API key for the Z.AI service (GLM-5 teacher inference)
"""

import time
import bittensor as bt

from distillation.base.validator import BaseValidatorNeuron
from distillation.validator import forward


class Validator(BaseValidatorNeuron):
    """
    Validator neuron for the GLM-5 distillation subnet.

    Inherits from BaseValidatorNeuron which handles:
    - Network registration and metagraph syncing
    - Weight setting on chain
    - Score tracking with exponential moving average
    - State persistence

    The domain-specific logic lives in distillation/validator/forward.py
    and distillation/validator/reward.py.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()

    async def forward(self):
        """
        Validator forward pass. Delegates to the forward module which:
        1. Selects a coding prompt
        2. Queries GLM-5 teacher for reference logprobs
        3. Queries miners for their logprobs
        4. Scores by KL-divergence (winner-take-all)
        5. Updates scores
        """
        return await forward(self)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info(f"Validator running... {time.time()}")
            time.sleep(5)
