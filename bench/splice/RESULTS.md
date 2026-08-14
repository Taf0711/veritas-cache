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

## Reproduce

```bash
OPENROUTER_API_KEY=... bash scripts/splice_experiment.sh
python3 bench/splice/diff_arms.py
```
