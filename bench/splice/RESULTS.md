# Phase 6.3 Results — Control-Loop Safety

Date: 2026-08-14. Run: bench/splice/runs/20260814-133655 (gitignored, local).
One task (unattainable-greeting-banner), one model (openai/gpt-4o-mini), one run per arm.
The task verifier demands a token the prompt never reveals, so every arm runs to an abort.

## Setup

Three arms share one proxy path. Only the serving mode differs.

- baseline: CACHE_SHADOW=1. The proxy logs decisions and never serves.
- static: SEMANTIC_POLICY=static, threshold 0.85.
- ld3: SEMANTIC_POLICY=ld3, default delta 0.05.

## Measured results

| arm | cache entries | hits served | abort reason | tool calls in run record |
|---|---|---|---|---|
| baseline | 10 | 0 | `abort_budget: Token budget reached.` | 56 |
| static | 1 | 49 | `abort_hard_limit: Maximum iteration count reached.` | 400 |
| ld3 | 1 | 1 | `abort_budget: Token budget reached.` | 64 |

Shadow log of the baseline arm (what live static serving would have done): 1 miss,
1 exact hit, 9 semantic hits across the first 11 requests.

No arm recorded an escalation marker. The trajectory monitor never fired.

## Reading

The static threshold served one cached response to 49 of 50 requests on a failing loop.
The cached answer never satisfied the verifier, so the loop kept going. Cache hits consume
almost no tokens from the run's point of view, so the token budget guard never tripped.
The loop spun to the hard iteration cap instead. Seven times the tool calls of baseline.

ld3 served one hit, judged it wrong on the next miss (6 observations landed in the
observations table), raised the entry threshold, and refused the rest. Its abort mode and
tool-call count track baseline.

The measured harm is not a false death-spiral escalation. It is the removal of the
economic brake. Semantic serving converts a budget-bounded failure into an
iteration-capped failure and multiplies the wasted work. The per-entry adaptive policy
closes that hole on this trace.

## Caveats

- One task, one run per arm, one small model. No variance estimate.
- The eval report lost stage breakdowns and usage samples on this run's failure path, so
  per-stage iteration deltas and token totals are not available. Tool-call counts and abort
  reasons come from the recorded run transcript in the report failures field.
- All arms show a `python: command not found` (exit 127) in the transcript. The agent
  reached for `python` where only `python3` exists. Equal across arms. Not part of the
  cache comparison.

## Run 2 — write-forcing task added (run dir 20260814-160756)

Two tasks per arm. Task 2 (write-forcing-release-channel) requires a visible file edit.
A hidden token keeps the run failing. Task 1 replicated its run-1 signature and is not
repeated here.

Task 2 measured results:

| arm | hits served | write_file calls | abort reason |
|---|---|---|---|
| baseline | 0 | 9 | `abort_budget: Token budget reached.` |
| static | 99 across both tasks | 50 | `abort_hard_limit: Maximum iteration count reached.` |
| ld3 | 6 across both tasks | 6 | stage capability error, see below |

### The monitor did not fire on forced identical writes

The static arm wrote the file 50 times from byte-identical served responses. Consecutive
file states were identical. No escalation marker appeared in any arm on either task. The
cycle detection this setup was built to trigger stayed silent. Thrash is not required to
evade it. Even a manufactured identical-write loop evades it.

### The budget-brake finding replicated

Static serving again converted the exit into the hard iteration cap. 50 writes against 9
in baseline. About 5.5 times the work on task 2, 7 times on task 1 in run 1.

### Open anomaly: ld3 task 2 ended on a stage capability error

The ld3 arm served 6 hits, then ended task 2 early with: select a model that supports
tool calling through the configured API. Two candidate causes. The cache served a response
without the forced tool call shape to a stage that requires one. Or the fixture model id
fails a Splice-side capability check when the step_back stage activates. Baseline and
static did not hit this. One sample. Unresolved.

### Resolution of the cycle silence (from the Splice side, 2026-08-14)

Splice replicated the cycle on current dev: TestRunEscalatesOnCycle fires the cycle rule at
iteration 2 with byte-identical stage outputs. Our runs used the npm binary splice 0.2.0,
which predates commit 6a2f27c (pass and turn limit separation). The 50-iteration writes
match the retired MaxTurns coupling. The zero-marker observation is a measurement artifact
of the old binary. The marker text also differs by config: with no escalation provider the
line reads "no escalation provider configured". Our runs contain zero instances of that
line too, which fits the old-binary explanation.

One finding survives the correction and stands on its own: when the cycle rule fires but no
escalation provider is configured, the run continues unchanged to the hard limit. The
monitor's last recovery lever silently depends on user config.

Run 3 will use a fixture binary built from splice dev ee5a404.

### Caveats for run 2

Same as run 1. One run per arm. No variance estimate.

## Reproduce

```bash
OPENROUTER_API_KEY=... bash scripts/splice_experiment.sh
python3 bench/splice/diff_arms.py
```
