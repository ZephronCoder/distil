# The MIT License (MIT)
# Copyright © 2024 GLM-5 Distillation Subnet

"""
Synapse protocol for the GLM-5 distillation subnet.

The validator sends a coding prompt to miners. Each miner runs inference on its
distilled model and returns log-probabilities for the generated tokens along with
metadata about the model (total parameter count). The validator independently
queries the GLM-5 teacher via the Z.AI API, then scores each miner by
KL-divergence between its logprobs and the teacher's logprobs.
"""

import typing
import bittensor as bt
from pydantic import Field


class DistillationSynapse(bt.Synapse):
    """
    Protocol between validator and miner for distillation evaluation.

    Request (validator → miner):
        prompt: The coding prompt text.
        max_tokens: Maximum number of tokens the miner should generate.
        temperature: Sampling temperature (should be low / deterministic for
            logprob comparison; typically 0 or near-0).

    Response (miner → validator):
        completion: The generated text from the miner's distilled model.
        logprobs: Per-token log-probabilities for the generated completion.
            List of dicts, each with at minimum:
                {"token": str, "logprob": float}
            Optionally includes "top_logprobs": List[{"token": str, "logprob": float}]
        model_size_params: Total parameter count of the miner's model (in
            billions). Validator checks this is ≤ 74.4B.
        model_name: Human-readable model identifier (e.g. HuggingFace repo id).
    """

    # ── Request fields (filled by validator) ──────────────────────────────
    prompt: str = Field(
        ...,
        description="The coding prompt to evaluate.",
    )
    max_tokens: int = Field(
        default=256,
        description="Maximum tokens for the miner to generate.",
    )
    temperature: float = Field(
        default=0.0,
        description="Sampling temperature. 0 = greedy/deterministic.",
    )

    # ── Response fields (filled by miner) ─────────────────────────────────
    completion: typing.Optional[str] = Field(
        default=None,
        description="Generated text from the distilled model.",
    )
    logprobs: typing.Optional[typing.List[dict]] = Field(
        default=None,
        description=(
            "Per-token logprobs. Each entry: "
            '{"token": str, "logprob": float, "top_logprobs": [...]}'
        ),
    )
    model_size_params: typing.Optional[float] = Field(
        default=None,
        description="Total parameter count of miner model in billions.",
    )
    model_name: typing.Optional[str] = Field(
        default=None,
        description="Human-readable model name / HuggingFace repo.",
    )

    def deserialize(self) -> dict:
        """
        Deserialize the synapse response into a dictionary with all miner
        outputs.
        """
        return {
            "completion": self.completion,
            "logprobs": self.logprobs,
            "model_size_params": self.model_size_params,
            "model_name": self.model_name,
        }
