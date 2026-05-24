# Conformal calibration of Verity's verdict thresholds

Verity's `pass` / `warn` / `fail` thresholds were v1 by reading
audit corpora and watching critic behaviour. That is fine for a first
pass, but it gives no statistical guarantee that the false-fail rate sits
where you want it. Conformal calibration replaces the guess with a
finite-sample bound, given a held-out labelled set.

This page covers the maths, the moving parts, and the operational steps.

## The maths in one paragraph

Pick a target residual error rate, `alpha` (say 0.1). Build a calibration
set of clean and hallucinated examples with gold labels. For each clean
row, compute Verity's continuous nonconformity score (described below).
The (1 - alpha)-quantile of those clean scores is the conformal cut-off
for the `fail` gate. Under the exchangeability assumption, a fresh clean
answer scores at or above the cut-off with probability at most alpha.
That bounds the false-fail rate by alpha. The same construction at a
softer alpha (default 2 * alpha) gives the `warn` cut-off.

References:

- Yadkori et al., "To Believe or Not to Believe Your LLM" /
  "Conformal Abstention", 2024, arXiv:2405.01563.
- Quach et al., "Conformal Language Modeling", 2023, arXiv:2306.10193.

## The nonconformity score

The score is a deliberately simple sum of signal counts:

```
score = critic_disagree_count
      + nli_contradictions
      + nli_unsupported
      + recompute_mismatches
```

It is computed in two equivalent places:

- `aggregator.computeNonconformityScore(...)` is the live runtime
  function. It sees the same data the consensus rules see.
- `bench-common.trialToCalibrationRow(...)` writes the same scalar to the
  JSONL the calibration script consumes.

Both must use the same definition. The unit tests pin this.

`hallucinationScore` in the bench harness also sums critic severity and
consistency divergence; the conformal score does not, deliberately, so
the calibrated threshold lives on counts that are stable across runs.

## How to run it

You need:

1. A labelled benchmark set — RAGTruth or LLM-AggreFact, normalised to
   the `(question, answer, context, gold)` JSONL shape the existing bench
   loaders consume.
2. A working Verity install: critics loaded in Ollama, worker model in
   LM Studio.

### Step 1: produce a calibration JSONL via the bench harness

```
node project/scripts/bench-ragtruth.ts \
  --data path/to/ragtruth.jsonl \
  --mode standard \
  --out bench/ragtruth-standard.tsv \
  --emit-calibration bench/calibration.jsonl
```

`--emit-calibration` writes one JSON object per row alongside the TSV.
Each object carries the gold label, the continuous score, and the
component counts. The TSV is unchanged.

### Step 2: compute conformal thresholds

```
node project/scripts/calibrate-thresholds.ts \
  --in bench/calibration.jsonl \
  --alpha 0.1 \
  --warn-alpha 0.2 \
  --dataset ragtruth \
  --out project/src/calibrated-thresholds.json
```

`--alpha` is the target false-fail rate for the `fail` gate.
`--warn-alpha` is the softer threshold for `warn` (default 2 * alpha).

The output is a JSON file with the warn and fail score cut-offs and
metadata about the calibration set.

### Step 3: ship the file

The aggregator loads `calibrated-thresholds.json` from the directory next
to `aggregator.js` at runtime (`dist/calibrated-thresholds.json` after a
build). The build step copies `project/src/calibrated-thresholds.json`
over if you keep it under source control; otherwise place the file by
hand. The startup log confirms which source is in use:

```
[aggregator] using CALIBRATED thresholds from .../calibrated-thresholds.json
  (alpha=0.1, n=250, warn=0.42, fail=0.78)
```

or, when the file is absent or malformed:

```
[aggregator] using HAND-TUNED thresholds from config.ts
  (calibrated-thresholds.json not found at ...)
```

The v1 defaults in `config.ts` are the documented fallback;
Verity never refuses to start on a missing calibration file.

## Interpreting alpha

Smaller alpha = stricter `fail` gate, fewer false fails, more false
passes. Larger alpha = looser `fail`, more false fails. A reasonable
starting point is `alpha = 0.1`: at most 10% of clean answers will be
marked `fail` by the conformal escalation. Re-calibrate when the critic
panel changes (different model, different prompt, different worker), the
NLI model changes, or the dataset distribution shifts.

## Bidirectional and additive-only modes

From 2026-05-24 the aggregator runs calibrated thresholds in
**bidirectional mode by default**. The calibrated cut-offs ARE the
decision boundary: score above `fail_score_threshold` → fail; above
`warn_score_threshold` → warn; below → pass. The v1 multi-axis
rules in `config.ts` still contribute their counts to the nonconformity
score, but the threshold decision is the conformal one. This is the
textbook conformal-prediction behaviour.

Why the change: the 2026-05-23 RAGTruth empirical sweep (50 rows,
standard mode) ran pre- and post-calibration with identical results
(BA 0.46, FPR 1.00 both times). The earlier escalation-only logic could
only push verdicts up, so it could not relax the v1 rules in
cases where they over-flagged. Bidirectional makes the upgrade actually
useful on the dominant failure mode.

Set `VERITY_CALIBRATED_THRESHOLDS_ADDITIVE_ONLY=1` to restore the legacy
escalation-only behaviour. In that mode calibrated cut-offs can only push
`pass` → `warn` → `fail`, never the other way. Useful if the v1
ladder is the conservative safety floor and the calibration is intended
purely to catch under-flagging.

An `error` verdict (too many critics unavailable) is never overridden in
either mode — that is a system-state signal, not a quality signal.

## Swapping thresholds

Three ways to switch:

- Rebuild after editing `project/src/calibrated-thresholds.json`. The
  startup log confirms the new values.
- Point `VERITY_CALIBRATED_THRESHOLDS_PATH` at a different file (absolute
  path). Useful for A/B comparing two calibration sets without rebuilding.
- Delete the JSON file. The aggregator falls back to the v1
  defaults in `config.ts`.

## What the machinery does not do

The calibration script does not measure detection rate. It picks a
cut-off given a target error rate; whether that cut-off catches
hallucinations is a separate question answered by running the bench
harness with the calibrated thresholds in place. RAGTruth and
LLM-AggreFact have not yet been benchmarked end-to-end on Verity; see
`design.md` § 11 for the deferred-work entry. The conformal machinery
ships, the run waits for a labelled set.
