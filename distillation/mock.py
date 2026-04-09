"""Mock classes for local testing without a live Bittensor network."""

import time
import asyncio
import random
import bittensor as bt

from typing import List


class MockSubtensor(bt.MockSubtensor):
    def __init__(self, netuid, n=16, wallet=None, network="mock"):
        super().__init__()

        if not self.subnet_exists(netuid):
            self.create_subnet(netuid)

        # Register ourself (the validator) as a neuron at uid=0
        if wallet is not None:
            self.force_register_neuron(
                netuid=netuid,
                hotkey=wallet.hotkey.ss58_address,
                coldkey=wallet.coldkey.ss58_address,
                balance=100000,
                stake=100000,
            )

        # Register n mock neurons who will be miners
        for i in range(1, n + 1):
            self.force_register_neuron(
                netuid=netuid,
                hotkey=f"miner-hotkey-{i}",
                coldkey="mock-coldkey",
                balance=100000,
                stake=100000,
            )


class MockMetagraph(bt.Metagraph):
    def __init__(self, netuid=1, network="mock", subtensor=None):
        super().__init__(netuid=netuid, network=network, sync=False)

        if subtensor is not None:
            self.subtensor = subtensor
        self.sync(subtensor=subtensor)

        for axon in self.axons:
            axon.ip = "127.0.0.0"
            axon.port = 8091

        bt.logging.info(f"Metagraph: {self}")
        bt.logging.info(f"Axons: {self.axons}")


class MockDendrite(bt.Dendrite):
    """
    Mock dendrite that returns synthetic distillation responses for testing.
    """

    def __init__(self, wallet):
        super().__init__(wallet)

    async def forward(
        self,
        axons: List[bt.AxonInfo],
        synapse: bt.Synapse = bt.Synapse(),
        timeout: float = 12,
        deserialize: bool = True,
        run_async: bool = True,
        streaming: bool = False,
    ):
        if streaming:
            raise NotImplementedError("Streaming not implemented yet.")

        async def query_all_axons(streaming: bool):
            async def single_axon_response(i, axon):
                start_time = time.time()
                s = synapse.model_copy()
                process_time = random.random()
                if process_time < timeout:
                    # Mock: generate fake logprobs for the prompt
                    s.completion = "def hello():\n    print('hello world')"
                    s.logprobs = [
                        {"token": "def", "logprob": -0.1},
                        {"token": " hello", "logprob": -0.3},
                        {"token": "()", "logprob": -0.05},
                        {"token": ":", "logprob": -0.02},
                        {"token": "\n", "logprob": -0.01},
                    ]
                    s.model_size_params = random.uniform(1.0, 70.0)
                    s.model_name = f"mock-distilled-model-{i}"
                else:
                    s.completion = None
                    s.logprobs = None
                    s.model_size_params = None
                    s.model_name = None

                if deserialize:
                    return s.deserialize()
                else:
                    return s

            return await asyncio.gather(
                *(
                    single_axon_response(i, target_axon)
                    for i, target_axon in enumerate(axons)
                )
            )

        return await query_all_axons(streaming)

    def __str__(self) -> str:
        return "MockDendrite({})".format(self.keypair.ss58_address)
