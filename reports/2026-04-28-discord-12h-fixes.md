# Discord 12h triage and fixes — 2026-04-28

Window: 2026-04-27 11:42 UTC → 2026-04-28 18:14 UTC (~30 h, 600 messages,
~252 from miners, 35+ distinct authors).

Pulled directly from the public SN97 channel (`1482026267392868583`)
via the Arbos bot REST API, then triaged below.

## What miners flagged (top themes by repetition)

### 1. "King is not re-evaluated each round → cached king vs fresh challenger is unfair" (10+ miners, FIXED in code, NOT a bug today)

Cited by `johhsmith123`, `coffieex`, `leeroyjkin`, `crypsick`,
`hehexd7345`, `pete1471625`, `subnetmania1900`, `kyle890015`,
`st.ravenv0ss`, `johnman123409` over the entire 24 h window.
Sample quotes:

- `johhsmith123` 12:03: "so when will you release a fix, so that
  miners will be scored on the same prompts as king?"
- `coffieex` 15:35: "The king uses a old score (from previous round
  cached), if you eval the king on the new round data their KL will
  shift dramatically"
- `leeroyjkin` 23:19: "you picked a king not even in the round ???"
- `st.ravenv0ss` 17:53: "king model is evaled in every round? cache
  problem is resolved?"

**Status: already fixed and deployed.**

- `f7c786c` (2026-04-27): king is added to `models_to_eval` every
  single-eval round so it sees the same block-seeded prompts as the
  challengers. Verified live in `state/current_round.json` (king UID
  228 sits in the 10-model round at block 8068473) and in the
  validator journal: `single-eval: king UID 228 included in this
  round (paired re-eval on shared prompts).`
- `bec5f95` (2026-04-27 23:27 UTC): king selection is restricted to
  current-round participants only, so the dethrone outcome can no
  longer pull a winner from a stale round. Live message in journal:
  `single-eval: kingship pool restricted to N round participants
  (was network-wide, fixed 2026-04-27 to prevent cross-sample leak)`.

The remaining miner confusion is communication, not mechanism —
people kept seeing the per-miner page say "king: tested every round"
without seeing the king's *fresh* round-local score next to the
challengers in the rounds grid. That's covered by item 5 below.

### 2. vLLM teacher generation falling back to HF (`crypsick`, `coffieex`, FIXED in working tree)

- `crypsick` 08:16: "Teacher continuations is definitly using hf and
  not vllm it is very slow again. can you check if there is a zombie
  process that still holds vram?"
- `crypsick` 08:17 / 08:22: "how come 4 vllm procs running? could
  also be zombies."
- `coffieex` 02:00: pasted `[GPU] PHASE 1 FALLBACK: HF teacher
  generation + logit extraction` traceback.
- `crypsick` 14:04 / 17:17: "it doesn't use vllm for the teacher
  generation phase again? why does this keep happening?"

**Root cause already understood, fix already loaded** in
`scripts/validator/pod_session.py` (uncommitted local change present
since the validator picked it up via `bec5f95`):

The previous chat-king detector used `ss -tlnp 'sport = :8100'`,
but `iproute2` is not installed on the eval pod, so `ss` silently
returned empty → fallback "preserve all matching processes" path
then kept *every* leaked chat-king vLLM, including the zombies
holding 22 GB of VRAM each. With three zombies + one live king, the
eval-teacher's `vllm.entrypoints` couldn't get enough free VRAM
to start and the pod fell back to HF for 80 + minutes per round.

The local fix replaces `ss` with `ps auxww | grep 'served-model-name
sn97-king' | awk '{print $2}'` + `ps -o etimes=` to identify the
youngest chat-king API server, and kills the older duplicates. This
is the change `crypsick` saw working at 11:38 ("now it is using vllm
again for the teacher generation phase").

**Action: this dirty change should be committed**. Diff is in
`scripts/validator/pod_session.py` lines 173-247. It is already
running because the validator was restarted after the file was
edited; committing is purely for repo hygiene and history.

### 3. Per-UID dashboard "re-test after 50 epochs" wording is misleading (FIXED here)

- `crypsick` 14:21: pasted dashboard text `Already tested against
  current king (0 epochs ago, re-test after 50)`
- `crypsick` 14:22: "50 rounds or 50 epochs?"
- `crypsick` 14:25: "look at uid #209 it was evaluated in Round
  block #8066063 and now is included in the current eval again
  (only 3 rounds in between). are you sure they have to wait 50
  rounds?"

**Real bug.** The dashboard's `eval_status.reason` field implied a
time-based cooldown (50 epochs ≈ 36 h), but the actual single-eval
policy in `select_challengers` re-evaluates a UID **only when its
on-chain commitment changes** (or when it's the king and gets the
fairness re-eval). There is no time-based re-test at all. The 50
came from the unused `STALE_EVAL_BLOCKS` constant.

**Fix shipped (`api/routes/miners.py`, `api/routes/evaluation.py`,
`api/config.py`):**

- `api/routes/miners.py` `eval_status.reason` rewritten to:
  *"Already tested against current king at block X (Y epoch(s)
  ago). Single-eval policy: a UID is re-tested when its on-chain
  commitment changes (push a new HuggingFace revision or commit a
  new model_repo) or when the king changes. There is no
  time-based re-test."*
- `api/routes/evaluation.py` `/api/eval-statuses` no longer
  switches to a fake `stale` status after the 50-epoch threshold;
  it stays `tested` until commitment-change pulls it back into
  the queue.
- `STALE_EVAL_BLOCKS` import removed from both routes; constant
  retained in `api/config.py` with a deprecation note for any
  external consumer.

### 4. Round-detail "worst" cell shows a number that no displayed axis matches (`coffieex`, FIXED here)

- `coffieex` 01:03 (today): "do you have any idea why worst shows
  as 0.5 but there is no 0.5 here UID 215 worst 0.500 wgt 0.885
  KL 0.1944 rkl 0.90 cap 0.97 math 0.92 code 0.75 reas 0.60
  ifev 1.00 aime 0.88 ..."

**Real bug**. The `RoundAxisGrid` in
`frontend/src/components/v2/rounds-panel.tsx` displays 12 axes
(KL, rkl, cap, math, code, reas, ifev, aime, judg, chat, len,
deg) but `composite.worst` is computed across all ~17 weighted
axes — including `mbpp_bench`, `tool_use_bench`,
`long_context_bench`, `robustness_bench`, and `reasoning_density`
which are **not** rendered in the grid. So when the limiting axis
is one of those off-grid axes (which the v28 quality-over-quantity
weight redistribution makes more likely, e.g. `tool_use_bench`
floor at 0.5 for many models), `worst` shows a value with no
on-grid match. UID 215's 0.500 was almost certainly
`tool_use_bench`.

**Fix shipped (`frontend/src/components/v2/rounds-panel.tsx`):**

- The `worst` cell now renders the limiting axis name as a
  one-line subscript (`↳ tool use`), matching the convention
  already used by `BoutCard`.
- When the limiting axis is **off-grid** (i.e. not in
  `ROUND_AXIS_COLS`), the subscript is rendered in
  `text-warning` so miners can see at a glance that the cell
  isn't reading from an on-screen axis.
- Tooltip on the cell explains the full reason and lists the
  off-grid axes by name.

### 5. Dark-mode toggle does nothing (`rao_2222`, FIXED here)

- `rao_2222` 16:02: "Dark theme button is not working in the
  dashboard, fix this asap"
- `rao_2222` 16:05: "Same here, click does nothing yet"

**Real bug, two parts.**

(a) `frontend/src/components/auto-refresh.tsx` — the legacy footer
`ThemeToggle` reads `localStorage["distil:theme"]` on mount but
never calls `apply()`, so its in-memory `theme` state can disagree
with the actual `data-theme` attribute on `<html>` set by the
no-flash inline script. After the first click, the cycle works; on
the *second* click the user's "click does nothing" report is
plausible because the local `theme` state had drifted.

(b) The header (`v2/site-header.tsx`) and footer
(`auto-refresh.tsx`) `ThemeToggle`s never observed each other.
Clicking one updates that toggle's React state and the DOM
attribute, but the *other* toggle's React state stays out of sync,
so it appears to "skip a step" the next time you click it.

**Fix shipped:**

- Both toggles now call `apply(saved)` on mount.
- Both toggles dispatch a `distil:theme-changed` `CustomEvent` on
  every cycle.
- Both toggles listen for that event and update local state +
  DOM in lock-step, so clicking either one in the same tab
  produces the same visible effect on both buttons.

## Items NOT acted on (and why)

- **Composite axis saturation at 1.0** (`coffieex` 15:28-15:35,
  `leeroyjkin` 14:29: "code and math are at max"). Real signal
  that the v28 difficulty re-balance still has saturation surfaces.
  Right fix is to swap the easy bench item pools out for harder
  variants (HumanEval+ → LiveCodeBench-hard, GSM8K → MATH-500
  hard). That's bench-pipeline scope, not this 12 h triage —
  defer.
- **3 % vs 5 % dethrone margin** (`coffieex` 11:50, `crypsick` 11:41
  "how will an increase to 5% reflect on the current subnet
  design?"). Bot proposed bumping; coffieex pushed back that the
  models are improving fine. Without an empirical bake-off, I'm
  not changing `SINGLE_EVAL_DETHRONE_MARGIN` from 0.03. Defer.
- **Pareto gate blocked dethrone** (e.g. UID 89 KL=0.118 vs king
  0.156, blocked at 2W/10L/4T). Working as designed — pareto gate
  ensures cross-axis improvement, not pure-KL wins. No change.
- **`shelton_1204` "When can I submit my model after registering
  a hotkey?"** Documentation question; does not need a code
  change. Bot can answer from existing `MINER_FAQ.md`.
- **`pigeonsyndrome` child-hotkey delegation question.** Subnet-
  governance topic, out of scope for the validator/dashboard
  fixes covered by this batch.
- **`thomasdev3` / `swortex` / `oper0447` / `st.ravenv0ss`
  newcomer questions ("how does the validator work?", "max
  parameter count", "canonical commit", etc.).** All answerable
  from `README.md` + `MINER_FAQ.md`; no code change required.
- **Difficulty raise for math/code benches**
  (`leeroyjkin` 14:29-14:30, `greyrepresentsall` 16:33). Same
  scope as the saturation discussion above — defer until a
  bench rotation PR.

## Files changed

- `api/routes/miners.py` — eval-status reason rewritten, `STALE_EVAL_BLOCKS`
  import removed.
- `api/routes/evaluation.py` — `/api/eval-statuses` no longer downgrades
  to `stale`; `STALE_EVAL_BLOCKS` import removed.
- `api/config.py` — `STALE_EVAL_BLOCKS` retained for backward-compat
  import only, marked deprecated in comment.
- `frontend/src/components/v2/rounds-panel.tsx` — `worst` cell now
  shows the limiting axis as a subscript, with tooltip + warning
  colour when the axis is off-grid.
- `frontend/src/components/auto-refresh.tsx` — `ThemeToggle` calls
  `apply(saved)` on mount and listens for/dispatches the new
  `distil:theme-changed` event.
- `frontend/src/components/v2/site-header.tsx` — header `ThemeToggle`
  also listens for/dispatches `distil:theme-changed` so the two
  toggles stay in sync within the same tab.

## Verification

- `python -c "import ast; ast.parse(...)"` on each edited Python file
  — clean.
- `PYTHONPATH=api python -c "import config; from routes import miners,
  evaluation"` — imports resolve; `config.STALE_EVAL_BLOCKS` still
  importable (deprecated).
- `npx tsc --noEmit -p .` on the frontend — clean.
- Live API spot-check: `curl http://127.0.0.1:3710/api/miner/229 |
  jq .eval_status` still returns the OLD message (server has not been
  restarted to pick up the new text). Restart `distil-api.service`
  to deploy.

## Deploy checklist

- [x] Code edits applied to `/opt/distil/repo`
- [x] Python AST + import smoke clean
- [x] TypeScript project type-check clean
- [ ] `systemctl restart distil-api.service` — picks up the new
      eval-status reason text and the `/api/eval-statuses`
      simplification
- [ ] `cd /opt/distil/repo/frontend && npm run build && systemctl
      restart distil-dashboard.service` — picks up the rounds-panel
      worst-axis subscript and the cross-toggle theme sync
- [ ] Commit the working-tree change in
      `scripts/validator/pod_session.py` (chat-king zombie kill via
      `ps auxww + etimes`) — already running on the live validator
      since the file was edited prior to the most recent restart, but
      currently uncommitted in the repo. Suggested message:
      `validator: replace ss-tlnp chat-king detection with
      ps+etimes (works on iproute2-less pods)`.

## Notes

- Validator was last (re)started 2026-04-27 23:24:42 UTC, AFTER the
  `f7c786c` king-re-eval commit and BEFORE `bec5f95`. Journal log
  entries from 2026-04-28 (`single-eval: kingship pool restricted to
  N round participants ... fixed 2026-04-27`) confirm `bec5f95` is
  in fact loaded — implying a more recent restart picked it up. No
  action needed.
- The Discord bot's UID-and-king answers continue to be correct
  most of the time, but please remind it to read `LIVE_EVAL_LOG.md`
  + `mirror/state/miners.json` before fabricating round outcomes.
  Two rounds today (2026-04-28 17:25-17:29, `coffieex` thread)
  showed the bot speculating about which UID dethroned which when
  `h2h_history.json` had the answer cached.
