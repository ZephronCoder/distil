# Dashboard Score Distinction: Global vs H2H

## The Problem

Users (e.g., s0wa48 on Discord) have been confused by the dashboard showing "global scores"
from different prompt sets alongside H2H (head-to-head) scores. This leads to incorrect
conclusions like "UID 64 is 36% better" when the scores are simply from different prompt sets.

## Key Distinction

### Global Score (Different Prompt Sets — NOT Directly Comparable)

- Each model was evaluated on a DIFFERENT set of prompts (seeded by the block hash at eval time)
- Prompt difficulty varies significantly between shards/blocks
- A model scoring 0.50 on one prompt set might score 0.35 on an easier set
- **You CANNOT compare global scores between models that were evaluated at different times**
- Cross-shard variance is ~10-20%, which dwarfs real model differences

### H2H Score (Same Prompts — Fair Comparison)

- Both models (king + challenger) are evaluated on the **exact same** prompt set
- Generated in the same epoch, on the same GPU, with the same teacher continuations
- This is the ONLY valid way to compare two models
- The "epsilon" threshold (currently 1%) accounts for remaining noise

## Dashboard Implementation Notes

When building/updating the dashboard UI:

1. **Label global scores clearly**: "Global Score (approximate, different prompt sets)"
2. **Label H2H scores clearly**: "H2H Score (fair comparison, same prompts)"
3. **Do NOT show a "% better" metric** between models evaluated on different prompt sets
4. **Show H2H results prominently** — these are the authoritative comparisons
5. **Consider hiding global scores** or showing them as secondary/collapsed info
6. **Add a tooltip/info icon** explaining why global scores aren't comparable

## Technical Context

- Prompts are sampled from `karpathy/climbmix-400b-shuffle` (6,542 shards)
- Block hash determines which shard is selected → different blocks = different prompts
- Teacher continuations are generated per-prompt → different prompts = different difficulty
- The `multi_shard_analysis.py` script can evaluate models on the SAME shards for fair comparison

## Future Work

- Add a "fair comparison" button in the dashboard that triggers a multi-shard analysis
- Show confidence intervals on all scores
- Color-code or visually separate global vs H2H sections
- Add a "last compared" timestamp showing when two models were last H2H'd
