"""ifeval_verifiable — v31 procedural instruction-following axis.

Adopts the **full** Google IFEval kwarg surface (Zhou et al., 2023;
arXiv 2311.07911) via the vendored grader at
``scripts/ifeval_vendor.py`` (21 supported verifier types — covers all
non-deprecated IFEval categories). Each round, items are composed by
sampling 1-4 verifiers, generating procedural kwargs for each, and
asking the model to answer a topic prompt that satisfies all
constraints simultaneously.

Why this is fully procedural:

* IFEval was designed from the start as **verifiable instructions**:
  every constraint has a parametric kwargs spec (``num_words``,
  ``forbidden_words``, ``num_paragraphs``, …) and a programmatic
  checker. There's no "ground-truth answer" to memorise — the gold
  is whether the response satisfies the constraints, which is
  recomputed per-item from procedurally sampled kwargs.
* The 21 verifier types × random kwarg ranges × random topic prompts
  × random stack-depth (1-4 constraints) gives a per-round item space
  on the order of 10^8 items. Memorisation is impossible.

Why this is better than the existing ``_generate_ifeval_items``:

* Current ``ifeval_bench`` uses 13 of the 21 verifiers with stack
  depths {1, 2, 3}. This module uses **all 21 verifiers** with stack
  depths {1, 2, 3, 4} — wider surface and harder tail.
* Stack-depth weighted by frequency: 25 % single, 30 % 2-stack,
  25 % 3-stack, 20 % 4-stack. The 4-stack tier matches IFEval's
  toughest "compound" items.
* Constraint conflict checking via ``ifeval_vendor`` (e.g., "all
  lowercase" + "all uppercase" can't both be required).

Validation methodology:
* Run as SHADOW for ≥ 1 round.
* Compute Pearson r between this axis's pass_frac and the held-out
  ``canary_ifeval`` pass_frac across ≥ 4 paired UIDs.
* Promote (set composite weight > 0) only if r ≥ 0.5.

References:
* Zhou, J., et al. (2023). "Instruction-Following Evaluation for Large
  Language Models." arXiv:2311.07911 (Google).
* IFEval reference impl:
  github.com/google-research/google-research/tree/master/instruction_following_eval
* Verifier conflict matrix: same paper's INSTRUCTION_CONFLICTS table,
  ported in ``scripts/ifeval_vendor.py``.
"""

from __future__ import annotations

import random
import string
import sys
from pathlib import Path

# Make sure we can import the vendor grader from the repo root regardless
# of where we're called from.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Guarded import so unit tests can run without the validator's heavy
# bench infra; we only need the supported-verifier registry + the
# evaluate_item function from the vendored grader.
try:
    from scripts.ifeval_vendor import SUPPORTED_VERIFIERS, evaluate_item
except ImportError:  # pragma: no cover - dev environment
    SUPPORTED_VERIFIERS = {}

    def evaluate_item(*_args, **_kwargs):
        return False, []


# ─────────────────────────────────────────────────────────────────────
#  Topic and vocabulary inventory.
#
#  Topics deliberately span "everyday" subjects (commute, gardening,
#  routine domestic activities) so the constraint-following challenge
#  is the *constraints*, not the *content*. A deliberately mundane
#  topic also reduces the chance the model dumps memorised text.
# ─────────────────────────────────────────────────────────────────────


_TOPICS: tuple[str, ...] = (
    "a daily commute by bicycle through a small coastal town",
    "the joys of urban gardening on an apartment balcony",
    "long-distance lighthouse keepers and how their work has changed",
    "weather forecasting at sea before modern instruments",
    "early-morning bakery routines in a town with one main street",
    "alpine railway engineering and the hardest tunnels to dig",
    "the migration habits of monarch butterflies across continents",
    "a small library reopening after renovation",
    "the history of cartography and its slow march to satellites",
    "a public market that runs every Saturday year-round",
    "how to teach a child to ride a two-wheeled bicycle",
    "the daily life of a beekeeper in a temperate climate",
    "what makes a good lasagna versus a great lasagna",
    "the design of a public transit network for a coastal city",
    "the seasonal reorganisation of a local farmers' market",
)

_KEYWORDS: tuple[str, ...] = (
    "pelican", "lighthouse", "harbor", "compass", "blueprint",
    "magnolia", "obsidian", "carousel", "satellite", "sycamore",
    "lantern", "quartz", "marigold", "cobblestone", "anchor",
    "sapphire", "trellis", "cypress", "harbinger", "nautical",
)

_BULLET_TOPICS: tuple[str, ...] = (
    "tools you'd need for amateur astronomy",
    "kitchen items every home should have",
    "books that help with creative writing",
    "ways to reduce daily commute time",
    "ingredients in a basic vegetable soup",
    "habits of effective remote workers",
    "low-effort houseplants for beginners",
)

_END_PHRASES: tuple[str, ...] = (
    "Thank you for reading.",
    "I hope this is helpful.",
    "That is all I have to share.",
    "End of response.",
    "Best wishes.",
    "Take care!",
)

_SECTION_HEADERS: tuple[str, ...] = (
    "Introduction", "Background", "Discussion", "Examples",
    "Conclusion", "Summary", "Key Points", "Notes",
)


# ─────────────────────────────────────────────────────────────────────
#  Per-verifier kwarg samplers.
#
#  Each sampler returns ``(rendered_clause, kwargs_dict)`` where:
#   * ``rendered_clause`` is the human-readable instruction (gets
#     concatenated into the user prompt).
#   * ``kwargs_dict`` is the dict the IFEval grader expects, with the
#     exact keys named in the upstream Google reference.
#
#  Samplers are referenced by the canonical IFEval ID
#  (``length_constraints:number_words`` etc.) so the registry below
#  can dispatch by string.
# ─────────────────────────────────────────────────────────────────────


def _s_punctuation_no_comma(rng: random.Random) -> tuple[str, dict]:
    return (
        "Do not use any commas in your response.",
        {},
    )


def _s_length_number_words(rng: random.Random) -> tuple[str, dict]:
    relation = rng.choice(("at least", "at most"))
    n = rng.choice([30, 50, 80, 120, 200])
    return (
        f"Your response should contain {relation} {n} words.",
        {"relation": relation, "num_words": n},
    )


def _s_length_number_sentences(rng: random.Random) -> tuple[str, dict]:
    relation = rng.choice(("at least", "at most"))
    n = rng.choice([3, 5, 7, 10])
    return (
        f"Use {relation} {n} sentences in your response.",
        {"relation": relation, "num_sentences": n},
    )


def _s_length_number_paragraphs(rng: random.Random) -> tuple[str, dict]:
    n = rng.choice([2, 3, 4])
    return (
        f"Structure your response as exactly {n} paragraphs separated by two newlines.",
        {"num_paragraphs": n},
    )


def _s_keywords_existence(rng: random.Random) -> tuple[str, dict]:
    n = rng.choice([1, 2])
    keywords = rng.sample(_KEYWORDS, n)
    return (
        f"Include the following keyword{'s' if n > 1 else ''} in your "
        f"response: {', '.join(keywords)}.",
        {"keywords": keywords},
    )


def _s_keywords_forbidden_words(rng: random.Random) -> tuple[str, dict]:
    n = rng.choice([1, 2])
    forbidden = rng.sample(_KEYWORDS, n)
    return (
        f"Do not include the following word{'s' if n > 1 else ''} "
        f"anywhere in your response: {', '.join(forbidden)}.",
        {"forbidden_words": forbidden},
    )


def _s_keywords_frequency(rng: random.Random) -> tuple[str, dict]:
    keyword = rng.choice(_KEYWORDS)
    relation = rng.choice(("at least", "at most"))
    n = rng.choice([2, 3, 4])
    return (
        f"The word '{keyword}' should appear {relation} {n} times in your response.",
        {"keyword": keyword, "relation": relation, "frequency": n},
    )


def _s_keywords_letter_frequency(rng: random.Random) -> tuple[str, dict]:
    letter = rng.choice(string.ascii_lowercase)
    relation = rng.choice(("at least", "at most"))
    n = rng.choice([3, 5, 8, 12])
    return (
        f"The letter '{letter}' should appear {relation} {n} times in your response.",
        {"letter": letter, "let_relation": relation, "let_frequency": n},
    )


def _s_change_case_english_lowercase(rng: random.Random) -> tuple[str, dict]:
    return (
        "Write your entire response in lowercase letters; do not use any uppercase letters.",
        {},
    )


def _s_change_case_english_capital(rng: random.Random) -> tuple[str, dict]:
    return (
        "Write your entire response in capital (uppercase) letters.",
        {},
    )


def _s_change_case_capital_word_frequency(rng: random.Random) -> tuple[str, dict]:
    relation = rng.choice(("at least", "at most"))
    n = rng.choice([2, 4, 6, 10])
    return (
        f"Your response should contain {relation} {n} words in all "
        f"capital letters.",
        {"capital_relation": relation, "capital_frequency": n},
    )


def _s_startend_quotation(rng: random.Random) -> tuple[str, dict]:
    return (
        "Wrap your entire response in double quotation marks.",
        {},
    )


def _s_startend_end_checker(rng: random.Random) -> tuple[str, dict]:
    phrase = rng.choice(_END_PHRASES)
    return (
        f"Finish your response with this exact phrase: \"{phrase}\". "
        f"Nothing should come after that phrase.",
        {"end_phrase": phrase},
    )


def _s_detectable_format_number_bullet_lists(rng: random.Random) -> tuple[str, dict]:
    n = rng.choice([3, 4, 5, 6])
    return (
        f"Include exactly {n} bullet points in your response. Each "
        f"bullet point must start with the asterisk markdown character "
        f"and a space (\"* \").",
        {"num_bullets": n},
    )


def _s_detectable_format_number_highlighted_sections(rng: random.Random) -> tuple[str, dict]:
    n = rng.choice([2, 3, 4])
    return (
        f"Include exactly {n} sections highlighted with asterisks "
        f"(markdown style: *highlighted*).",
        {"num_highlights": n},
    )


def _s_detectable_format_title(rng: random.Random) -> tuple[str, dict]:
    return (
        "Your response must start with a title wrapped in double angle "
        "brackets, e.g. <<Your Title Here>>.",
        {},
    )


def _s_detectable_format_json_format(rng: random.Random) -> tuple[str, dict]:
    return (
        "Your entire response must be a single, well-formed JSON object "
        "(no markdown fences, no surrounding prose).",
        {},
    )


def _s_detectable_format_constrained_response(rng: random.Random) -> tuple[str, dict]:
    return (
        'Your response must be one of the following exactly: "My '
        'answer is yes.", "My answer is no.", or "My answer is maybe.". '
        "Use the response exactly as listed.",
        {},
    )


def _s_detectable_format_multiple_sections(rng: random.Random) -> tuple[str, dict]:
    n = rng.choice([2, 3, 4])
    section_marker = rng.choice(("Section", "Part", "Chapter"))
    return (
        f"Your response must contain {n} sections. Mark the beginning "
        f"of each section with \"{section_marker} X\", where X is the "
        f"section number.",
        {"section_spliter": section_marker, "num_sections": n},
    )


def _s_detectable_content_number_placeholders(rng: random.Random) -> tuple[str, dict]:
    n = rng.choice([2, 3, 4])
    return (
        f"Your response must contain at least {n} placeholders "
        f"represented by square brackets, such as [address] or [name].",
        {"num_placeholders": n},
    )


def _s_detectable_content_postscript(rng: random.Random) -> tuple[str, dict]:
    marker = rng.choice(("P.S.", "P.P.S."))
    return (
        f"At the end of your response, please add a postscript that "
        f"starts with \"{marker}\".",
        {"postscript_marker": marker},
    )


# ─────────────────────────────────────────────────────────────────────
#  Verifier registry.
#
#  Mirrors ``scripts/ifeval_vendor.SUPPORTED_VERIFIERS`` keys exactly
#  so we can hand kwargs straight to ``evaluate_item`` without
#  translation. The 21 entries below are all of the verifiers the
#  vendored grader supports today.
# ─────────────────────────────────────────────────────────────────────


VERIFIER_SAMPLERS: dict[str, callable] = {
    "punctuation:no_comma": _s_punctuation_no_comma,
    "length_constraints:number_words": _s_length_number_words,
    "length_constraints:number_sentences": _s_length_number_sentences,
    "length_constraints:number_paragraphs": _s_length_number_paragraphs,
    "keywords:existence": _s_keywords_existence,
    "keywords:forbidden_words": _s_keywords_forbidden_words,
    "keywords:frequency": _s_keywords_frequency,
    "keywords:letter_frequency": _s_keywords_letter_frequency,
    "change_case:english_lowercase": _s_change_case_english_lowercase,
    "change_case:english_capital": _s_change_case_english_capital,
    "change_case:capital_word_frequency": _s_change_case_capital_word_frequency,
    "startend:quotation": _s_startend_quotation,
    "startend:end_checker": _s_startend_end_checker,
    "detectable_format:number_bullet_lists": _s_detectable_format_number_bullet_lists,
    "detectable_format:number_highlighted_sections": _s_detectable_format_number_highlighted_sections,
    "detectable_format:title": _s_detectable_format_title,
    "detectable_format:json_format": _s_detectable_format_json_format,
    "detectable_format:constrained_response": _s_detectable_format_constrained_response,
    "detectable_format:multiple_sections": _s_detectable_format_multiple_sections,
    "detectable_content:number_placeholders": _s_detectable_content_number_placeholders,
    "detectable_content:postscript": _s_detectable_content_postscript,
}


# ─────────────────────────────────────────────────────────────────────
#  Conflict matrix (subset of Google's INSTRUCTION_CONFLICTS).
#
#  Stack-composition rejects samples where any pair of constraints
#  conflicts. We use a conservative subset of Google's matrix —
#  enough to catch the obvious clashes (case, length, format
#  exclusivity).
# ─────────────────────────────────────────────────────────────────────


_CONFLICTS: dict[str, set[str]] = {
    "change_case:english_lowercase": {
        "change_case:english_lowercase",
        "change_case:english_capital",
        "change_case:capital_word_frequency",
    },
    "change_case:english_capital": {
        "change_case:english_capital",
        "change_case:english_lowercase",
    },
    "change_case:capital_word_frequency": {
        "change_case:capital_word_frequency",
        "change_case:english_lowercase",
        "change_case:english_capital",
    },
    "detectable_format:json_format": (
        # JSON conflicts with most surface-form constraints.
        set(VERIFIER_SAMPLERS.keys()) - {
            "keywords:forbidden_words",
            "keywords:existence",
        }
    ),
    "detectable_format:constrained_response": set(VERIFIER_SAMPLERS.keys()),
    "detectable_format:title": {
        "detectable_format:title",
        "startend:quotation",
        "detectable_format:json_format",
        "detectable_format:constrained_response",
    },
    "startend:quotation": {
        "startend:quotation",
        "detectable_format:title",
        "detectable_format:json_format",
        "detectable_format:constrained_response",
    },
    "length_constraints:number_paragraphs": {
        "length_constraints:number_paragraphs",
        "length_constraints:number_sentences",
    },
    "length_constraints:number_sentences": {
        "length_constraints:number_sentences",
        "length_constraints:number_paragraphs",
    },
}


def _conflicts(a: str, b: str) -> bool:
    """Symmetric conflict check between two verifier IDs."""
    if a == b:
        return True
    return b in _CONFLICTS.get(a, set()) or a in _CONFLICTS.get(b, set())


# ─────────────────────────────────────────────────────────────────────
#  Item generator.
#
#  Stack-depth distribution (matches IFEval's compound tail):
#    25 % single-constraint
#    30 % 2-stack
#    25 % 3-stack
#    20 % 4-stack
#
#  ``_BENCH_STREAM_OFFSET`` picks an XOR offset distinct from
#  ``math_gsm_symbolic`` (0x5631) so the two pools are statistically
#  independent.
# ─────────────────────────────────────────────────────────────────────


_STACK_DEPTH_RATIO: tuple[float, float, float, float] = (0.25, 0.30, 0.25, 0.20)
_BENCH_STREAM_OFFSET = 0x5632  # "V2"


def _sample_stack_depth(rng: random.Random) -> int:
    r = rng.random()
    p1, p2, p3, _p4 = _STACK_DEPTH_RATIO
    if r < p1:
        return 1
    if r < p1 + p2:
        return 2
    if r < p1 + p2 + p3:
        return 3
    return 4


def _sample_constraint_stack(rng: random.Random, depth: int) -> list[tuple[str, str, dict]]:
    """Return ``depth`` non-conflicting (verifier_id, clause, kwargs)."""
    available = list(VERIFIER_SAMPLERS.keys())
    rng.shuffle(available)
    chosen: list[tuple[str, str, dict]] = []
    for vid in available:
        if any(_conflicts(vid, c[0]) for c in chosen):
            continue
        clause, kwargs = VERIFIER_SAMPLERS[vid](rng)
        chosen.append((vid, clause, kwargs))
        if len(chosen) == depth:
            break
    return chosen


def generate_items(block_seed: int | None, n_items: int) -> list[dict]:
    """Generate procedural IFEval-style items for one validator round.

    Each item carries:
        * ``src``             — telemetry tag (``v31_ifeval/stack<N>``).
        * ``prompt``          — the user-facing instruction.
        * ``instruction_ids`` — list of canonical IFEval IDs.
        * ``kwargs``          — list of per-instruction kwargs dicts
          (parallel to ``instruction_ids``).
        * ``stack_depth``     — number of constraints stacked.
        * ``topic``           — base topic (for telemetry).

    The grader at ``ifeval_vendor.evaluate_item`` reads
    ``instruction_ids`` + ``kwargs`` directly, so this format is
    interoperable with the existing IFEval scoring pipeline.

    Args:
        block_seed: per-round entropy from the validator's substrate.
            ``None`` is treated as 0 (deterministic, useful for tests).
        n_items: number of items to generate.

    Returns:
        list of item dicts.
    """
    seed = (int(block_seed or 0) ^ _BENCH_STREAM_OFFSET) & 0xFFFFFFFF
    rng = random.Random(seed)
    out: list[dict] = []
    for _ in range(n_items):
        per_item_seed = rng.randint(0, 2**31 - 1)
        item_rng = random.Random(per_item_seed)
        target_depth = _sample_stack_depth(item_rng)
        stack = _sample_constraint_stack(item_rng, target_depth)
        # Some verifier IDs (constrained_response, json_format)
        # conflict with most others; if one is picked first the stack
        # legitimately collapses below ``target_depth``. Report the
        # ACTUAL stack depth so the unit-test invariant
        # ``len(instruction_ids) == stack_depth`` holds.
        actual_depth = len(stack)
        topic = item_rng.choice(_TOPICS)
        clauses = [c[1] for c in stack]
        instruction_ids = [c[0] for c in stack]
        kwargs_list = [c[2] for c in stack]
        # Build the user-facing prompt. The base ask is "Write 1-2
        # paragraphs about <topic>" which is benign and doesn't conflict
        # with the constraint clauses; constraints are listed below as
        # bullets so the model has a clean checklist.
        constraint_block = "\n".join(f"- {c}" for c in clauses)
        prompt = (
            f"Write 1-2 paragraphs about {topic}. Your response must "
            f"satisfy ALL of the following constraints:\n"
            f"{constraint_block}\n\n"
            f"Now write the response."
        )
        out.append(
            {
                "src": f"v31_ifeval/stack{actual_depth}",
                "prompt": prompt,
                "question": prompt,  # alias for grader pipelines that read 'question'
                "instruction_ids": instruction_ids,
                "kwargs": kwargs_list,
                "stack_depth": actual_depth,
                "target_stack_depth": target_depth,
                "topic": topic,
            }
        )
    return out


# ─────────────────────────────────────────────────────────────────────
#  Grader entrypoint — re-exports ``evaluate_item`` from the vendored
#  IFEval grader. The bench probe in ``pod_eval_vllm.py`` calls this
#  to score a model response against the procedurally generated
#  constraints. Returns ``(all_pass, per_constraint_results)``.
# ─────────────────────────────────────────────────────────────────────


def grade_item(response: str, item: dict) -> tuple[bool, list[bool]]:
    return evaluate_item(response, item["instruction_ids"], item["kwargs"])


def _self_test_demo() -> None:  # pragma: no cover - dev helper
    items = generate_items(block_seed=42, n_items=8)
    for it in items:
        print("─" * 60)
        print(f"src     = {it['src']}")
        print(f"depth   = {it['stack_depth']}")
        print(f"ids     = {it['instruction_ids']}")
        print("prompt:")
        print(it["prompt"])
    print("─" * 60)
    print(f"generated {len(items)} items OK")


if __name__ == "__main__":  # pragma: no cover - dev helper
    _self_test_demo()
