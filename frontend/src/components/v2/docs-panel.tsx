"use client";

import { useState } from "react";

type DocKey =
  | "overview"
  | "protocol"
  | "miner"
  | "validator"
  | "scoring"
  | "antigaming"
  | "api"
  | "constants"
  | "links";

interface DocItem {
  key: DocKey;
  label: string;
}

const NAV: DocItem[] = [
  { key: "overview", label: "Overview" },
  { key: "protocol", label: "Protocol" },
  { key: "miner", label: "Miner quickstart" },
  { key: "validator", label: "Validator setup" },
  { key: "scoring", label: "Scoring & H2H" },
  { key: "antigaming", label: "Anti-spiral" },
  { key: "api", label: "API reference" },
  { key: "constants", label: "Constants" },
  { key: "links", label: "Links" },
];

/**
 * In-dashboard docs. Sidebar nav + body. Idiom from the v2 reference.
 *
 * Body content is composite-first. Last refresh: 2026-05-10 v31.2.
 * The eval is now 25+ axes, KL is one of them, the ranking key is
 * composite.final (0.85·worst_5_mean + 0.15·weighted), and the
 * anti-Goodhart defence is procedural item generation (the 11 v31
 * axes generate items per round from a block-seed; no static answer
 * key to memorise). Cites composite.py / pod_eval_vllm.py /
 * scripts/v31/ where useful so a reader can verify the claims.
 */
export function DocsPanel() {
  const [active, setActive] = useState<DocKey>("overview");
  return (
    <div className="grid grid-cols-1 md:grid-cols-[240px_1fr] min-h-[calc(100vh-3.5rem-3rem)]">
      <aside className="border-b md:border-b-0 md:border-r border-border bg-[var(--surface-soft)] px-6 py-8 overflow-y-auto">
        <h6 className="text-[10px] uppercase tracking-[0.18em] text-meta font-medium mb-3.5">
          Distil
        </h6>
        <ul className="space-y-1">
          {NAV.map((item) => (
            <li key={item.key}>
              <button
                onClick={() => setActive(item.key)}
                className={[
                  "block w-full text-left text-[13px] px-2 py-1.5 transition-colors",
                  active === item.key
                    ? "bg-foreground text-white"
                    : "text-foreground hover:bg-[var(--surface-elevated)]",
                ].join(" ")}
              >
                {item.label}
              </button>
            </li>
          ))}
        </ul>
      </aside>
      <div className="px-8 sm:px-12 py-12 max-w-3xl overflow-y-auto">
        <DocsBody active={active} />
      </div>
    </div>
  );
}

function DocsBody({ active }: { active: DocKey }) {
  switch (active) {
    case "overview":
      return (
        <Article>
          <h2>Distil · SN97</h2>
          <p className="lead">
            Bittensor subnet 97. Miners distil a frozen teacher into smaller
            students. Validators score them on a 25+ axis composite covering
            distribution match, capability against procedurally-generated ground
            truth, conversational quality, generation discipline, robustness,
            honesty under uncertainty, and isomorphic consistency. The highest
            <code> composite.final </code>wears the crown.
          </p>
          <h3>Teacher</h3>
          <p>
            <strong>moonshotai/Kimi-K2.6</strong> — 1T-parameter MoE,
            ~32B active. Served via OpenRouter (cloud-API, no local 1T-param
            load). Top-20 logprobs exposed. Vocab 163,840 (Kimi BPE).
          </p>
          <h3>Reference</h3>
          <p>
            <strong>Qwen/Qwen3.5-4B</strong> — the unfine-tuned baseline.
            The reference-broken-axes filter uses it to drop axes the base
            model itself can&apos;t pass under our eval setup.
          </p>
          <h3>Constraint</h3>
          <p>
            Students must stay under <strong>33B</strong> total parameters,
            <strong> deepseek_v3 / DeepseekV3ForCausalLM</strong> architecture,
            <strong> Kimi BPE vocab (163,840)</strong>. No
            teacher-activation copies. Identical safetensors hashes are
            blacklisted per commit.
          </p>
          <h3>Anti-Goodhart defence</h3>
          <p>
            All 11 v31 procedural axes (~50% of composite weight) generate
            their items per round from the block-seed.
            There is no static answer key to memorise; canonical wordings are
            blocked by isomorphic perturbation testing (IPT, name rotation)
            and GSM-NoOp topical-distractor injection. See
            <code> reports/2026-05-09-v31-axis-promotion.md</code>.
          </p>
        </Article>
      );
    case "protocol":
      return (
        <Article>
          <h2>Protocol</h2>
          <p className="lead">Three steps. Public, deterministic, on-chain.</p>
          <h3>1 · Commit</h3>
          <p>
            Each UID commits a Hugging Face repo + revision on-chain. Weights
            are pinned, sha-verified, and pulled by validators at the next eval
            boundary.
          </p>
          <h3>2 · Score</h3>
          <p>
            The validator generates <strong>300 prompts per round</strong> from
            the block-seed and queries the OpenRouter teacher API for top-20
            logprobs (~10 min). The student is then loaded on a Lium pod and
            scored sequentially on 25+ axes:
          </p>
          <ul>
            <li>
              <strong>Distillation tier</strong> — KL, on-policy RKL,
              top-K overlap, capability (~30% of composite).
            </li>
            <li>
              <strong>v31 procedural axes</strong> (11 axes, ~50% of composite) —
              math (gsm_symbolic, competition, robustness), code
              (humaneval_plus, ifeval_verifiable), reasoning (logic_grid,
              dyval_arith, long_context_ruler), knowledge (multi_hop_kg),
              honesty (truthfulness_calibration), consistency (paraphrase / IPT).
            </li>
            <li>
              <strong>Quality probes</strong> — judge_probe, long_form_judge,
              chat_turns_probe.
            </li>
            <li>
              <strong>Discipline</strong> — length, degeneracy,
              reasoning_density, calibration_bench.
            </li>
          </ul>
          <p>
            Each axis lands in [0, 1]. Legacy bench axes (math_bench, code_bench,
            tool_use_bench, etc.) still RUN as telemetry but their composite
            weight is 0 — they appear on the Axes tab in the &quot;Telemetry&quot;
            tier so you can monitor the legacy lens without it touching the
            ranking.
          </p>
          <h3>3 · Crown</h3>
          <p>
            The king is whoever has the highest <code>composite.final</code> =
            0.85·worst_5_mean + 0.15·weighted. A challenger dethrones the
            incumbent only when their <code>final</code> beats the king&apos;s
            by <strong>≥5%</strong> (raised from 3% on 2026-05-10 after the
            v31.1 variance-reduction sweep).
          </p>
        </Article>
      );
    case "miner":
      return (
        <Article>
          <h2>Miner quickstart</h2>
          <h3>Need</h3>
          <ul>
            <li>Hugging Face account + writable repo</li>
            <li>An H100 / H200 (or 8×A100) for training a 33B distillation</li>
            <li>OpenRouter API key (cheap) for teacher logprob queries</li>
            <li>Bittensor wallet on netuid 97</li>
          </ul>
          <h3>Steps</h3>
          <ol>
            <li>
              Clone <code>github.com/unarbos/distil</code>
            </li>
            <li>
              Distil from{" "}
              <code>moonshotai/Kimi-K2.6</code> via OpenRouter
              top-20 logprobs · stay under{" "}
              <strong>33B params</strong> ·{" "}
              <code>deepseek_v3 / DeepseekV3ForCausalLM</code> arch ·{" "}
              <strong>vocab 163,840</strong> (Kimi BPE)
            </li>
            <li>Push weights to a HF repo · note the revision sha</li>
            <li>
              Commit on-chain:{" "}
              <code>btcli s commit --netuid 97 --hf_repo &lt;repo&gt; --rev &lt;sha&gt;</code>
            </li>
            <li>
              Wait for the next eval boundary · watch the <em>Live</em> tab
            </li>
          </ol>
          <h3>Constraints</h3>
          <dl className="kv">
            <dt>Max params</dt>
            <dd>33B</dd>
            <dt>Architecture</dt>
            <dd>deepseek_v3 / DeepseekV3ForCausalLM</dd>
            <dt>Vocab size</dt>
            <dd>163,840 (Kimi BPE)</dd>
            <dt>Max new tokens</dt>
            <dd>8192</dd>
            <dt>Max prompt tokens</dt>
            <dd>1024</dd>
            <dt>Activation copy threshold</dt>
            <dd>0.99999</dd>
            <dt>One-eval-per-registration</dt>
            <dd>Each hotkey is scored exactly once. Fresh registrations to retry.</dd>
          </dl>
          <h3>What to optimise</h3>
          <p>
            <strong>composite.final</strong> (= 0.85·worst_5_mean + 0.15·weighted),
            not KL. KL is one of 25+ axes. A pure-KL model that loops on
            <code> &quot;Hi&quot;</code> or fails grade-school math cannot take
            the crown. Especially focus on the <strong>v31 procedural axes</strong> —
            they generate items every round from the block-seed, so memorising
            a public dataset doesn&apos;t help. Mix in IPT-defended training
            data (canonical paraphrases, name rotations) and avoid GSM-NoOp
            topical-distractor pitfalls (irrelevant clauses that look
            mathy but contribute nothing).
          </p>
          <p>
            Read the axis playbook in <code>docs/MINER_FAQ.md</code> for what
            each axis rewards and what training data to mix in for it.
          </p>
        </Article>
      );
    case "validator":
      return (
        <Article>
          <h2>Validator setup</h2>
          <h3>Hardware</h3>
          <ul>
            <li>Multi-GPU node, 80GB+ each</li>
            <li>vLLM concurrency: 32</li>
            <li>NVMe scratch for teacher logits</li>
          </ul>
          <h3>Run</h3>
          <pre>{`git clone https://github.com/unarbos/distil
cd distil
uv sync
python -m distil.validator --netuid 97`}</pre>
          <p>
            Logs surface in the <em>Live</em> tab via{" "}
            <code>/api/eval-progress</code> and <code>/api/gpu-logs</code>.
          </p>
        </Article>
      );
    case "scoring":
      return (
        <Article>
          <h2>Scoring &amp; H2H</h2>
          <h3>Per-round (single-eval mode)</h3>
          <p>
            Each commitment is scored exactly once on its own block-seeded
            300-prompt set. The teacher (Kimi-K2.6 via OpenRouter, top-20
            logprobs) runs first and is cached. Each student is then loaded
            sequentially on the Lium pod (student forward pass → 11 v31
            procedural axes + legacy telemetry battery → judge probes). The
            reference baseline (Qwen3.5-4B) runs every round so the
            reference-broken-axes filter can drop axes the base model itself
            can&apos;t pass under our eval setup.
          </p>
          <h3>The composite (v31.2 weights)</h3>
          <p>
            All axes in [0, 1]. Higher-is-better. Live weights from
            <code> composite.py</code> + <code> configs/eval_policy.json</code>:
          </p>
          <ul>
            <li>
              <strong>Distillation tier (relative axes, ~30%):</strong>{" "}
              <code>on_policy_rkl 0.39</code> · <code>top_k_overlap 0.09</code>{" "}
              · <code>kl 0.05</code> · <code>capability 0.05</code> ·{" "}
              <code>length 0.05</code> · <code>degeneracy 0.05</code>.
            </li>
            <li>
              <strong>v31 procedural axes (~50%):</strong>{" "}
              <code>v31_math_gsm_symbolic 0.06</code> ·{" "}
              <code>v31_math_competition 0.05</code> ·{" "}
              <code>v31_math_robustness 0.03</code> ·{" "}
              <code>v31_code_humaneval_plus 0.08</code> ·{" "}
              <code>v31_reasoning_logic_grid 0.05</code> ·{" "}
              <code>v31_reasoning_dyval_arith 0.04</code> ·{" "}
              <code>v31_long_context_ruler 0.05</code> ·{" "}
              <code>v31_knowledge_multi_hop_kg 0.04</code> ·{" "}
              <code>v31_ifeval_verifiable 0.04</code> ·{" "}
              <code>v31_truthfulness_calibration 0.03</code> ·{" "}
              <code>v31_consistency_paraphrase 0.03</code>.
            </li>
            <li>
              <strong>Quality:</strong>{" "}
              <code>judge_probe 0.20</code> ·{" "}
              <code>long_form_judge 0.20</code> ·{" "}
              <code>long_gen_coherence 0.25</code> ·{" "}
              <code>chat_turns 0.10</code>.
            </li>
            <li>
              <strong>Discipline + standalone:</strong>{" "}
              <code>reasoning_density 0.05</code> · <code>calibration_bench 0.05</code>.
            </li>
            <li>
              <strong>Telemetry-only (composite weight 0):</strong>{" "}
              all legacy <code>*_bench</code> axes + skill groups.
              They still run for the dashboard but no longer touch ranking.
            </li>
          </ul>
          <h3>Dethrone gates (all must pass)</h3>
          <ol>
            <li>
              <strong>Composite.final margin.</strong>{" "}
              <code>challenger.final &gt; king.final × 1.05</code> (raised
              from 1.03 in v31.1, 2026-05-10).
            </li>
            <li>
              <strong>Worst-axis floor.</strong>{" "}
              <code>composite.worst &lt; 0.20</code> vetoes the dethrone
              even if the margin passes.
            </li>
            <li>
              <strong>Pareto-dominance.</strong> Soft majority:{" "}
              <code>n_wins ≥ n_losses</code> with a 2% noise margin.
              Insufficient comparable axes fails open.
            </li>
            <li>
              <strong>King-canary streak.</strong> The king must hold the
              top spot on the canary axes (held-out GSM8K / HumanEval /
              BBH / MMLU-Pro / IFEval) for ≥1 consecutive round to
              maintain the crown — telemetry-only.
            </li>
          </ol>
          <h3>Variance reduction (v31.1)</h3>
          <p>
            Three gate-strengthening changes shipped 2026-05-10:
          </p>
          <ul>
            <li>
              Per-axis n bumped on every v31 axis (n=8 → 14-18, gsm_symbolic
              16 → 24): per-axis SE drops ~30%.
            </li>
            <li>
              <code>WORST_3_MEAN_K</code> 3 → 5: a single chance-driven low
              axis can no longer dominate the composite (-23% SE on the
              bottom-K mean).
            </li>
            <li>
              <code>SINGLE_EVAL_DETHRONE_MARGIN</code> 0.03 → 0.05: combined
              with the SE reductions above, false-positive dethrone rate from
              pure noise drops from ~27% to &lt;6% per round.
            </li>
          </ul>
        </Article>
      );
    case "antigaming":
      return (
        <Article>
          <h2>Anti-spiral &amp; anti-gaming</h2>
          <p className="lead">
            Why the eval isn&apos;t just KL — and why memorising public
            benchmarks doesn&apos;t help.
          </p>
          <h3>The Goodhart trap</h3>
          <p>
            Earlier versions of distil used static benches (math_bench,
            code_bench, etc.) — same items every round. Miners overfit to
            the items and posted high composite scores while their
            held-out (GSM8K, HumanEval, MMLU-Pro) numbers stayed flat or
            regressed. That&apos;s Goodhart&apos;s law: when a measure
            becomes a target, it stops being a measure.
          </p>
          <p>
            v31 (2026-05-09) replaces the static surface with{" "}
            <strong>11 procedurally-generated axes</strong> seeded from the
            on-chain block. The block changes every round, so no two
            evaluations share items. There is no static answer key to
            memorise.
          </p>
          <h3>Countermeasures shipped (v31.2)</h3>
          <ul>
            <li>
              <strong>Procedural item generation</strong> on all 11 v31
              axes: gsm_symbolic (numeric/name/operand variants), competition
              (AMC/AIME-style with answer integrity), robustness (GSM-Plus +
              GSM-NoOp topical-distractor injection), humaneval_plus
              (EvalPlus-augmented test cases, sandbox-graded), logic_grid
              (zebra-puzzle constraint sat), dyval_arith (DAG arithmetic on
              dynamic graphs), long_context_ruler (NIAH at variable
              context), multi_hop_kg (procedural KG, 2-3 hop), ifeval_verifiable
              (constraint-driven IF), truthfulness_calibration (Brier-scored),
              consistency_paraphrase (IPT name-rotation).
            </li>
            <li>
              <strong>Isomorphic Perturbation Testing (IPT)</strong> on{" "}
              <code>v31_consistency_paraphrase</code>: presents two
              isomorphic versions of the same problem (different names,
              same structure) and grades for consistent answers. A model
              that memorised the canonical wording fails the rotated one.
            </li>
            <li>
              <strong>GSM-NoOp topical distractor</strong> on{" "}
              <code>v31_math_robustness</code>: injects mathematically
              irrelevant clauses into otherwise-valid problems. A
              memorised solution path that ignores the distractor still
              works; a memorised distribution that triggers on surface
              keywords fails.
            </li>
            <li>
              <code>thinking_collapse_probe</code>: greedy 1024-token
              budget on three trivial prompts. Flags any 6-word phrase
              repeated ≥15× or &lt;2/3 prompts hitting EOS. Sets the
              composite <code>worst</code> to a DQ floor so the model
              never wins H2H.
            </li>
            <li>
              <code>reasoning_density</code> axis: mean generation tokens
              normalised by per-task target × pass_frac. Penalises both
              over-thinking trivia AND verbose-but-wrong answers. Cannot
              be gamed by short-wrong: pass_frac=0 → axis=0.
            </li>
            <li>
              <code>on_policy_rkl</code> axis: reverse-KL under the
              student&apos;s own sampling. Catches &ldquo;matches teacher
              logits but collapses under free generation&rdquo;.
              (Currently disabled under OpenRouter teacher mode — restored
              when self-hosted teacher returns.)
            </li>
          </ul>
          <h3>Anti-copy</h3>
          <ul>
            <li>SHA256 hash duplicate detection — first committer owns the hash.</li>
            <li>
              Logit fingerprinting — cosine similarity &gt; 0.99999 on
              activation vectors flags functional copies even when hashes
              differ.
            </li>
            <li>
              <strong>One-eval-per-registration:</strong> each hotkey is
              scored exactly once. Re-commits on the same hotkey are
              rejected. To retry you need a fresh registration.
            </li>
          </ul>
          <h3>Variance reduction (v31.1)</h3>
          <p>
            The dethrone gate was tightened on 2026-05-10 to make it
            substantively impossible to win by chance:{" "}
            <code>SINGLE_EVAL_DETHRONE_MARGIN</code> 3% → 5%, per-axis n
            bumped on every v31 axis (~30% lower SE), and the worst-K
            mean widened from K=3 to K=5. The false-positive dethrone
            rate from pure RNG variance dropped from ~27% per round to{" "}
            <strong>&lt;6%</strong>. See{" "}
            <code>reports/2026-05-10-variance-reduction.md</code>.
          </p>
        </Article>
      );
    case "api":
      return (
        <Article>
          <h2>API reference</h2>
          <p className="lead">
            Public read-only API. Cached 60s. Base:{" "}
            <code>https://api.arbos.life</code>.
          </p>
          <p>
            <strong>Live OpenAPI / Swagger UI:</strong>{" "}
            <a
              href="https://api.arbos.life/docs"
              target="_blank"
              rel="noreferrer"
              className="underline"
            >
              api.arbos.life/docs
            </a>{" "}
            — every endpoint, every parameter, runnable in the browser.
            Bookmark it; this page is just the cheatsheet.
          </p>
          <h3>Endpoint cheatsheet</h3>
          <dl className="kv">
            <dt>GET /api/health</dt>
            <dd>service health, current king, code revision</dd>
            <dt>GET /api/metagraph</dt>
            <dd>chain block + per-UID stake / hotkey / commitment</dd>
            <dt>GET /api/scores</dt>
            <dd>latest KL per UID (one of 25+ axes)</dd>
            <dt>GET /api/leaderboard</dt>
            <dd>top-N with composite worst + weighted breakdown</dd>
            <dt>GET /api/miner/{`{uid}`}</dt>
            <dd>per-UID card: model, KL, full composite axes, H2H tail</dd>
            <dt>GET /api/h2h-latest</dt>
            <dd>last bout, full composite per UID, king flag</dd>
            <dt>GET /api/h2h-history?limit=N</dt>
            <dd>past N rounds with composite per result</dd>
            <dt>GET /api/king-history</dt>
            <dd>king flips with reign_blocks</dd>
            <dt>GET /api/history</dt>
            <dd>KL axis time series</dd>
            <dt>GET /api/eval-progress</dt>
            <dd>validator phase + progress (active eval)</dd>
            <dt>GET /api/eval-stream</dt>
            <dd>SSE stream of live eval events</dd>
            <dt>GET /api/benchmarks</dt>
            <dd>
              held-out evalscope reports for the king, teacher, reference (NOT
              the validator&apos;s composite — see Bench tab footer)
            </dd>
            <dt>GET /api/queue</dt>
            <dd>pending challengers in the next eval round</dd>
            <dt>GET /api/incidents</dt>
            <dd>recent ops events (king flips, restarts, DQ events)</dd>
            <dt>GET /api/price</dt>
            <dd>α / τ / USD</dd>
            <dt>GET /api/gpu-logs?limit=N</dt>
            <dd>recent GPU pod log lines (sanitised)</dd>
          </dl>
          <h3>Where to ask questions</h3>
          <p>
            Discord <code>#ა・distil・97</code> in the Bittensor server. Open
            issues / PRs at{" "}
            <a
              href="https://github.com/unarbos/distil"
              target="_blank"
              rel="noreferrer"
              className="underline"
            >
              github.com/unarbos/distil
            </a>
            .
          </p>
        </Article>
      );
    case "constants":
      return (
        <Article>
          <h2>Constants</h2>
          <dl className="kv">
            <dt>netuid</dt>
            <dd>97</dd>
            <dt>teacher</dt>
            <dd>moonshotai/Kimi-K2.6 (cloud-API via OpenRouter)</dd>
            <dt>reference</dt>
            <dd>Qwen/Qwen3.5-4B</dd>
            <dt>max student params</dt>
            <dd>33B</dd>
            <dt>architecture</dt>
            <dd>deepseek_v3 / DeepseekV3ForCausalLM</dd>
            <dt>vocab size</dt>
            <dd>163,840 (Kimi BPE)</dd>
            <dt>eval prompts (single-eval)</dt>
            <dd>300</dd>
            <dt>policy version</dt>
            <dd>2026-05-12-v32.3-broader-cap</dd>
            <dt>weighted axes (live)</dt>
            <dd>25+ (11 v31 procedural + distillation + judge + discipline)</dd>
            <dt>v31 axes share of composite</dt>
            <dd>~50%</dd>
            <dt>SINGLE_EVAL_DETHRONE_MARGIN</dt>
            <dd>0.05 (raised from 0.03 on 2026-05-10)</dd>
            <dt>WORST_3_MEAN_K</dt>
            <dd>5 (was 3 in v30.7)</dd>
            <dt>COMPOSITE_FINAL_BOTTOM_WEIGHT (α)</dt>
            <dd>0.85 (was 0.7 in v30.5)</dd>
            <dt>COMPOSITE_DETHRONE_FLOOR</dt>
            <dd>0.20</dd>
            <dt>POD_PER_MODEL_TIMEOUT</dt>
            <dd>2400s</dd>
            <dt>activation copy threshold</dt>
            <dd>0.99999</dd>
          </dl>
        </Article>
      );
    case "links":
      return (
        <Article>
          <h2>Links</h2>
          <ul>
            <li>
              <a href="https://github.com/unarbos/distil">
                GitHub · unarbos/distil
              </a>
            </li>
            <li>
              <a href="https://chat.arbos.life">Chat with the king</a>
            </li>
            <li>
              <a href="https://api.arbos.life/docs">API docs · OpenAPI / Swagger</a>
            </li>
            <li>
              <a href="https://openrouter.ai/moonshotai/kimi-k2.6">
                Teacher on OpenRouter (Kimi-K2.6)
              </a>
            </li>
            <li>
              <a href="https://huggingface.co/Qwen/Qwen3.5-4B">
                Reference on Hugging Face
              </a>
            </li>
            <li>
              <a href="https://github.com/unarbos/distil/blob/main/reports/2026-05-09-v31-axis-promotion.md">
                Report · v31 procedural axis promotion
              </a>
            </li>
            <li>
              <a href="https://github.com/unarbos/distil/blob/main/reports/2026-05-10-variance-reduction.md">
                Report · v31.1 variance reduction
              </a>
            </li>
            <li>
              <a href="https://github.com/unarbos/distil/blob/main/docs/MINER_FAQ.md">
                Miner FAQ — axis playbook
              </a>
            </li>
            <li>
              <a href="https://github.com/unarbos/distil/tree/main/scripts/v31">
                scripts/v31 — procedural axis source
              </a>
            </li>
            <li>
              <a href="https://github.com/unarbos/distil/blob/main/scripts/validator/composite.py">
                composite.py — axis weights live here
              </a>
            </li>
            <li>
              <a href="https://taomarketcap.com/subnets/97">
                SN97 on TaoMarketCap
              </a>
            </li>
          </ul>
        </Article>
      );
  }
}

function Article({ children }: { children: React.ReactNode }) {
  return <article className="docs-body space-y-4">{children}</article>;
}
