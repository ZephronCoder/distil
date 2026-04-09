"""
GLM-5 Distillation Subnet — Miner Neuron

The miner serves a distilled/quantized version of GLM-5 (≤74.4B params total)
and responds to validator queries with completions and per-token logprobs.

Miners must:
1. Load their distilled model (e.g. via vLLM, HuggingFace Transformers, etc.)
2. Generate completions with logprobs for incoming prompts
3. Report their model's parameter count honestly (validator may verify)
"""

import os
import time
import typing
import bittensor as bt

import distillation
from distillation.base.miner import BaseMinerNeuron
from distillation.protocol import DistillationSynapse


class Miner(BaseMinerNeuron):
    """
    Miner neuron that serves a distilled GLM-5 model.

    Override the forward() method with your distilled model inference.
    The default implementation uses vLLM or transformers to load and serve
    the model specified by --miner.model_name.
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)

        # Model loading — miners should customize this for their specific
        # distilled model. This is a reference implementation.
        self.model = None
        self.tokenizer = None
        self.model_name = os.environ.get(
            "MINER_MODEL_NAME", "Qwen/Qwen2.5-7B"
        )
        self.model_size_params = float(
            os.environ.get("MINER_MODEL_SIZE_B", "7.0")
        )

        bt.logging.info(
            f"Miner initialized with model: {self.model_name} "
            f"({self.model_size_params}B params)"
        )

        self._load_model()

    def _load_model(self):
        """
        Load the distilled model. This reference implementation uses
        HuggingFace transformers. For production, consider vLLM for
        better throughput.
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            bt.logging.info(f"Loading model {self.model_name}...")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

            bt.logging.info(f"Model {self.model_name} loaded successfully.")

        except Exception as e:
            bt.logging.error(
                f"Failed to load model {self.model_name}: {e}. "
                f"Miner will return empty responses until model is available."
            )

    async def forward(
        self, synapse: DistillationSynapse
    ) -> DistillationSynapse:
        """
        Process a distillation query from a validator.

        Generates a completion with per-token logprobs using the distilled model
        and populates the synapse response fields.
        """
        bt.logging.info(
            f"Received query: {synapse.prompt[:80]}... "
            f"(max_tokens={synapse.max_tokens}, temp={synapse.temperature})"
        )

        # Always report model metadata
        synapse.model_size_params = self.model_size_params
        synapse.model_name = self.model_name

        if self.model is None or self.tokenizer is None:
            bt.logging.warning("Model not loaded, returning empty response.")
            synapse.completion = ""
            synapse.logprobs = []
            return synapse

        try:
            import torch

            # Tokenize the prompt
            inputs = self.tokenizer(
                synapse.prompt,
                return_tensors="pt",
            ).to(self.model.device)

            # Generate with logprobs
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=synapse.max_tokens,
                    temperature=max(synapse.temperature, 0.01),  # avoid div/0
                    do_sample=synapse.temperature > 0,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            # Extract the generated token IDs (excluding the prompt)
            generated_ids = outputs.sequences[0, inputs.input_ids.shape[1]:]
            scores = outputs.scores  # tuple of (vocab_size,) tensors

            # Build logprobs list
            logprobs_list = []
            for i, (token_id, score) in enumerate(
                zip(generated_ids, scores)
            ):
                log_softmax = torch.nn.functional.log_softmax(
                    score[0], dim=-1
                )
                token_logprob = log_softmax[token_id].item()
                token_str = self.tokenizer.decode(
                    [token_id], skip_special_tokens=False
                )

                # Get top-5 logprobs
                top_values, top_indices = torch.topk(log_softmax, 5)
                top_logprobs = [
                    {
                        "token": self.tokenizer.decode(
                            [idx.item()], skip_special_tokens=False
                        ),
                        "logprob": val.item(),
                    }
                    for val, idx in zip(top_values, top_indices)
                ]

                logprobs_list.append(
                    {
                        "token": token_str,
                        "logprob": token_logprob,
                        "top_logprobs": top_logprobs,
                    }
                )

            # Decode the full completion
            synapse.completion = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )
            synapse.logprobs = logprobs_list

            bt.logging.info(
                f"Generated {len(logprobs_list)} tokens. "
                f"Completion: {synapse.completion[:80]}..."
            )

        except Exception as e:
            bt.logging.error(f"Inference error: {e}")
            synapse.completion = ""
            synapse.logprobs = []

        return synapse

    async def blacklist(
        self, synapse: DistillationSynapse
    ) -> typing.Tuple[bool, str]:
        """
        Blacklist logic: only allow registered validators with permits.
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning(
                "Received a request without a dendrite or hotkey."
            )
            return True, "Missing dendrite or hotkey"

        if (
            not self.config.blacklist.allow_non_registered
            and synapse.dendrite.hotkey not in self.metagraph.hotkeys
        ):
            bt.logging.trace(
                f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        if self.config.blacklist.force_validator_permit:
            uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(
                    f"Blacklisting non-validator hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Non-validator hotkey"

        bt.logging.trace(
            f"Not blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority(self, synapse: DistillationSynapse) -> float:
        """Priority based on validator stake."""
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            return 0.0
        caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        priority = float(self.metagraph.S[caller_uid])
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}"
        )
        return priority


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info(f"Miner running... {time.time()}")
            time.sleep(5)
