"""
Validator forward pass for the GLM-5 distillation subnet.

Each step:
1. Select a random coding prompt from the prompt pool.
2. Query the GLM-5 teacher model via the Z.AI API for reference logprobs.
3. Send the prompt to sampled miners via the DistillationSynapse.
4. Score miners by KL-divergence against teacher logprobs.
5. Apply winner-take-all scoring and update weights.
"""

import os
import time
import random
import bittensor as bt
import httpx

from distillation.protocol import DistillationSynapse
from distillation.validator.reward import get_rewards
from distillation.utils.uids import get_random_uids


# ── Coding prompt pool ────────────────────────────────────────────────────────
# A representative set of coding prompts for distillation evaluation.
# In production, this would be loaded from a larger dataset or generated
# dynamically.
CODING_PROMPTS = [
    "Write a Python function that implements binary search on a sorted list and returns the index of the target element, or -1 if not found.",
    "Implement a Python class for a min-heap with insert, extract_min, and peek methods.",
    "Write a Python function that takes a string of parentheses and returns True if they are balanced.",
    "Implement the merge sort algorithm in Python. Include a function that sorts a list in-place.",
    "Write a Python function that finds all prime numbers up to n using the Sieve of Eratosthenes.",
    "Implement a trie (prefix tree) in Python with insert, search, and starts_with methods.",
    "Write a Python function that solves the N-Queens problem and returns all valid board configurations.",
    "Implement a Python LRU cache from scratch using a doubly-linked list and a dictionary.",
    "Write a Python function that performs topological sort on a directed acyclic graph represented as an adjacency list.",
    "Implement the A* pathfinding algorithm in Python for a 2D grid with obstacles.",
    "Write a Python function that converts a mathematical expression from infix to postfix notation using the shunting-yard algorithm.",
    "Implement a thread-safe producer-consumer queue in Python using threading primitives.",
    "Write a Python function to serialize and deserialize a binary tree to/from a string.",
    "Implement a Python function that finds the longest common subsequence of two strings using dynamic programming.",
    "Write a Python async function that fetches multiple URLs concurrently using aiohttp and returns results as they complete.",
    "Implement a Python decorator that adds retry logic with exponential backoff to any function.",
    "Write a Python function that implements the Knuth-Morris-Pratt string matching algorithm.",
    "Implement a basic Python REPL that evaluates arithmetic expressions with operator precedence.",
    "Write a Python generator that yields Fibonacci numbers lazily without storing the entire sequence.",
    "Implement a Python function that finds the shortest path in a weighted graph using Dijkstra's algorithm.",
]


# ── Z.AI Teacher API ──────────────────────────────────────────────────────────
ZAI_API_URL = "https://open.z.ai/api/paas/v4/chat/completions"
ZAI_MODEL = "glm-5"  # The GLM-5 teacher model identifier


async def query_teacher(
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> dict:
    """
    Query the GLM-5 teacher model via the Z.AI API with logprobs enabled.

    Returns:
        dict with keys:
            - "completion": str — the generated text
            - "logprobs": List[dict] — per-token logprobs
    """
    api_key = os.environ.get("ZAI_API_KEY", "")
    if not api_key:
        bt.logging.warning(
            "ZAI_API_KEY not set. Teacher query will return empty logprobs."
        )
        return {"completion": "", "logprobs": []}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": ZAI_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "logprobs": True,
        "top_logprobs": 5,
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                ZAI_API_URL, json=payload, headers=headers
            )
            response.raise_for_status()
            data = response.json()

        # Parse the OpenAI-compatible response format
        choice = data.get("choices", [{}])[0]
        completion = choice.get("message", {}).get("content", "")

        # Extract logprobs from the response
        logprobs_data = choice.get("logprobs", {})
        content_logprobs = logprobs_data.get("content", [])

        # Normalize to our standard format
        parsed_logprobs = []
        for entry in content_logprobs:
            parsed_entry = {
                "token": entry.get("token", ""),
                "logprob": entry.get("logprob", 0.0),
            }
            if "top_logprobs" in entry:
                parsed_entry["top_logprobs"] = [
                    {"token": tp.get("token", ""), "logprob": tp.get("logprob", 0.0)}
                    for tp in entry["top_logprobs"]
                ]
            parsed_logprobs.append(parsed_entry)

        bt.logging.info(
            f"Teacher returned {len(parsed_logprobs)} token logprobs for prompt."
        )
        return {"completion": completion, "logprobs": parsed_logprobs}

    except Exception as e:
        bt.logging.error(f"Teacher API query failed: {e}")
        return {"completion": "", "logprobs": []}


async def forward(self):
    """
    Validator forward pass:
    1. Select a coding prompt
    2. Query GLM-5 teacher for reference logprobs
    3. Query miners for their logprobs
    4. Score by KL-divergence (winner-take-all)
    5. Update scores
    """
    # 1. Select a random coding prompt
    prompt = random.choice(CODING_PROMPTS)
    max_tokens = 256
    temperature = 0.0

    bt.logging.info(f"Selected prompt: {prompt[:80]}...")

    # 2. Query the teacher model for reference logprobs
    teacher_result = await query_teacher(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    teacher_logprobs = teacher_result.get("logprobs", [])

    if not teacher_logprobs:
        bt.logging.warning(
            "No teacher logprobs available. Skipping this round."
        )
        time.sleep(5)
        return

    bt.logging.info(
        f"Got {len(teacher_logprobs)} teacher logprobs. "
        f"Teacher completion: {teacher_result.get('completion', '')[:80]}..."
    )

    # 3. Query miners
    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)

    if len(miner_uids) == 0:
        bt.logging.warning("No available miners to query.")
        time.sleep(5)
        return

    bt.logging.info(f"Querying {len(miner_uids)} miners...")

    responses = await self.dendrite(
        axons=[self.metagraph.axons[uid] for uid in miner_uids],
        synapse=DistillationSynapse(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        ),
        deserialize=True,
        timeout=self.config.neuron.timeout,
    )

    bt.logging.info(f"Received {len(responses)} responses from miners.")

    # 4. Score miners by KL-divergence (winner-take-all)
    rewards = get_rewards(
        self,
        teacher_logprobs=teacher_logprobs,
        responses=responses,
    )

    bt.logging.info(f"Scored responses: {rewards}")

    # 5. Update scores
    self.update_scores(rewards, miner_uids)

    # Brief pause to avoid hammering the network
    time.sleep(5)
