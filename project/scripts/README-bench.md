# Bench harness — runbook for all five datasets

Verity ships harnesses for five public hallucination benchmarks:
RAGTruth, LLM-AggreFact, FaithBench, HaluEval, and ANAH-v2. Each
harness drives the same `verify_answer` pipeline over a JSONL of rows,
writes a per-row TSV, and tallies summary metrics into a footer.

This file is the end-to-end recipe: prerequisites, dataset downloads,
canonical run commands, how to read the TSV, reference numbers,
conformal calibration, and the usual symptoms.

## Contents

1. Prerequisites
2. Dataset downloads
3. Running the harnesses
4. Reading the TSV
5. Reference numbers
6. Conformal calibration workflow
7. Troubleshooting
8. Bilingual ANAH-v2

## 1. Prerequisites

The harness imports `dist/pipeline.js` and calls the real critic stack.
Nothing is mocked.

- **Worker (logprob endpoint).** LM Studio running the worker model on
  port 1234 with the logprob-bearing OpenAI-compatible endpoint live.
  See design doc § 2 for the reference hardware and § 18 for the
  worker-handshake details.
- **Critics.** Ollama running the two critic models. The defaults are
  Granite 4.1 8B and Ministral 3B. Both must respond on
  `http://localhost:11434/api/chat`.
- **NLI judge.** Bundled `@huggingface/transformers` NLI checker; no
  extra service required.

Build once before any run:

```bash
npm run build
```

The scripts type-check on every invocation (the `npm run bench:*`
entries chain `tsc` before `node`), so a stale `dist/` is not a
problem, but the first compile is the slow one.

### VRAM headroom

If a run drags out beyond the per-row latency the harness logs to
stderr, open Task Manager and watch *GPU → Shared GPU Memory*. Any
non-zero shared usage means the worker has spilled into system RAM and
every token now costs an order of magnitude more wall time. Fixes:

- Eject any extra models loaded in LM Studio.
- Reload the worker with a smaller context (32K is usually enough for
  these datasets; 128K is wasteful).
- Switch Thinking off on the critic models — the reasoning traces
  inflate KV cache for no benefit on a binary verdict task.

## 2. Dataset downloads

The harness does not download anything. Fetch the data yourself.

The reference data root is `C:/AI/verify-data/` (outside the repo).
RAGTruth, HaluEval, and ANAH have already been fetched there under
their respective MIT / MIT / Apache-2.0 licences. See `NOTICE.md` at
the repo root for the full citation list. LLM-AggreFact and FaithBench
are not auto-fetched; see the notes below.

### 2.1 RAGTruth

```bash
git clone https://github.com/ParticleMedia/RAGTruth
```

Test split lives at `RAGTruth/dataset/response.jsonl` (~18k rows). The
upstream layout splits the data across `source_info.jsonl` (question +
retrieved context, keyed by `source_id`) and `response.jsonl` (response
+ per-span hallucination labels). Join them with `jq`:

```bash
jq -s '
  (.[0] | map({(.source_id|tostring): .}) | add) as $src
  | .[1]
  | map(
      . as $r
      | $src[$r.source_id|tostring] as $s
      | {
          id:       ($r.id // "\($r.source_id)_\($r.model)"),
          question: ($s.question // $s.prompt // ""),
          answer:   $r.response,
          context:  ($s.source_info // $s.passages // $s.context // ""),
          label:    (if ($r.labels|length) > 0 then 1 else 0 end)
        }
    )
  | .[]
' RAGTruth/dataset/source_info.jsonl RAGTruth/dataset/response.jsonl > ragtruth.flat.jsonl
```

Confirm:

```bash
head -n1 ragtruth.flat.jsonl | jq 'keys'
```

Expect `["answer","context","id","label","question"]`.

### 2.2 LLM-AggreFact

Licence is CC-BY-ND-4.0 and the dataset card explicitly permits use as
an evaluation benchmark. The HuggingFace download is gated behind a
contact form, so this one is user-side: visit
`https://huggingface.co/datasets/lytang/LLM-AggreFact`, accept the
terms, then:

```bash
pip install datasets
huggingface-cli login    # one-off after accepting on the dataset page
python -c "
from datasets import load_dataset
ds = load_dataset('lytang/LLM-AggreFact', split='test')
ds.to_json('aggrefact.jsonl', orient='records', lines=True)
"
```

Each row carries `dataset`, `doc`, `claim`, `label`. The harness reads
those names directly. `label = 1` means supported (gold-clean); the
harness inverts when recording `gold` in the TSV.

### 2.3 FaithBench

Licence is CC-BY-NC-SA-4.0. The NonCommercial clause exceeds
attribution-only, so this dataset is user-side and only for
non-commercial research. To use it:

```bash
git clone https://github.com/vectara/FaithBench
```

Per-batch JSON files under `FaithBench/data_for_release/batch_*.json`
(~700 rows total). The harness accepts a single batch file, a directory
of `batch_*.json`, or a concatenated JSONL. The exact field names vary
by batch; the loader normalises them.

### 2.4 HaluEval

```bash
git clone https://github.com/RUCAIBox/HaluEval
```

Four files under `HaluEval/data/`:

- `qa_data.json` (10K rows)
- `dialogue_data.json` (10K rows)
- `summarization_data.json` (10K rows)
- `general_data.json` (5K rows)

The first three are JSONL despite the `.json` suffix. `general_data.json`
is a single JSON array; convert with `jq -c '.[]' general_data.json >
general_data.jsonl` before passing to the harness.

### 2.5 ANAH

The public ANAH dataset on HuggingFace (Apache-2.0) is topic-grouped:
each row carries a topic, its reference documents, three questions,
two model responses per question (GPT-3.5 and InternLM), and per-
sentence human annotations with hallucination labels in a tagged
inline format (`<要点>`, `<幻觉>`, `<参考>`, `<改正>`).

```bash
git clone https://huggingface.co/datasets/opencompass/anah  C:/AI/verify-data/anah
```

The harness expects a flat per-sentence JSONL with
`{question, response, sentence, annotation, language}`. Run the
included converter once to produce it:

```bash
node project/scripts/convert-anah-source.ts \
  --input  C:/AI/verify-data/anah \
  --output C:/AI/verify-data/anah/anah-flat.jsonl
```

Expected output on the current public slice: 783 topics → 1,846
per-sentence rows. Label distribution roughly 55% `ok`, 13%
`contradictory`, 32% `unverifiable`. The `nofact` label is rare to
absent in this slice.

**The public slice is Chinese-only.** All 1,846 rows carry
`language: "zh"`. Run with `--language zh` or `--language both`;
`--language en` matches nothing here. The bilingual claim in the ANAH
paper refers to the full unreleased v2 set; only the Chinese subset is
public under Apache-2.0.

## 3. Running the harnesses

Build once, then run any of:

```bash
npm run bench:ragtruth   -- --data C:/AI/verify-data/RAGTruth/dataset/response.jsonl --mode deep --limit 200
npm run bench:aggrefact  -- --data aggrefact.jsonl --mode deep --limit 200
npm run bench:faithbench -- --data FaithBench/data_for_release/ --mode deep --limit 200
npm run bench:halueval   -- --data C:/AI/verify-data/HaluEval/data/qa_data.json --task qa --mode deep --limit 200
npm run bench:anah       -- --data C:/AI/verify-data/anah/anah-flat.jsonl --mode deep --language zh --limit 200
```

All five npm scripts are wired in `package.json`. `npm run calibrate`
is also wired (see section 6).

Common flags:

- `--data <path>` required, JSONL.
- `--mode standard|deep|deeper` defaults to `standard`. `standard` is
  the four-critic + NLI ensemble; `deep` adds consistency and
  perplexity; `deeper` adds wider re-sampling.
- `--all` runs all three modes sequentially. Writes three TSVs.
- `--limit N` caps rows. Use a small cap for smoke runs.
- `--out <path>` overrides the default TSV location.
- `--emit-calibration <path>` also writes a calibration JSONL; see
  section 6.

ANAH-only flag:

- `--language en|zh|both` defaults to `en`. The public dataset slice is
  Chinese-only, so use `--language zh` or `--language both`. With
  `both`, the harness writes one TSV per language (e.g.
  `bench/anah-en-deep.tsv`, `bench/anah-zh-deep.tsv`).

Default output path: `bench/<dataset>-<mode>.tsv` at the repo root.
The `bench/` folder is intended to be gitignored; check `.gitignore`
before committing TSVs.

## 4. Reading the TSV

One header row, one row per item, then a comment-block footer.

| column | meaning |
|---|---|
| id | per-row id from the dataset, or synthesised |
| gold | 1 = hallucinated, 0 = clean |
| predicted | 1 = Verity flagged, 0 = Verity passed |
| verdict | raw consensus (`pass` / `warn` / `fail` / `error`) |
| latency_ms | wall-clock of the verification call |
| claims_total | total claims extracted from the answer |
| claims_flagged | NLI contradictions + unsupported |
| critic_agree_count | majority-class verdict count among live critics |
| critic_disagree_count | minority-class count |

Verdict-to-prediction mapping: `pass` is clean; `warn`, `fail` and
`error` are all flagged. `error` only fires when too many critics were
unavailable, so an `error`-heavy run means Ollama or LM Studio is not
serving the critic models.

Footer fields (one line each, all start with `#`):

- **balanced_accuracy.** Mean of true-positive rate and true-negative
  rate. The headline number. Robust to class imbalance, which matters
  on FaithBench (skewed clean) and ANAH-v2 (skewed hallucinated after
  `nofact` rows are dropped).
- **recall.** True positive rate. Fraction of hallucinated rows the
  pipeline flagged. Read alongside FPR — high recall with high FPR is
  a "flag everything" detector and not useful.
- **false_positive_rate.** Fraction of clean rows Verity flagged. The
  cost of using the pipeline as a guardrail in production.
- **precision.** Of the rows Verity flagged, how many were
  actually hallucinated.
- **auroc.** Rank-based AUROC over a continuous proxy score (NLI flag
  count + critic severity + recompute mismatches + consistency
  divergence). The verdict itself is categorical, so a verdict-only
  AUROC would degenerate. Treat AUROC as supplementary, not headline.
- **mean_latency_ms.** Mean wall-clock per row. Standard ≈ 11 s,
  deep ≈ 30 s, deeper ≈ 50 s on the reference hardware.

ANAH-v2 TSVs also carry `# anah_aggregation:` and `# anah_language:`
lines so a future reader can see how per-sentence labels were folded
into per-row gold without re-deriving it.

## 5. Reference numbers

HalluGuard, a fine-tuned 4B SRM, is the comparison anchor:

- **RAGTruth.** 84.0 balanced accuracy.
- **LLM-AggreFact.** 75.7 balanced accuracy.

Aim for these as the floor on the matching datasets.

**FaithBench is a stress test, not a leaderboard.** The FaithBench
paper frames it as "best detectors near 50%": the dataset is
deliberately built from cases the existing detectors disagree on.
Treat anything above 55 balanced accuracy as a real signal.

**HaluEval** has four task splits. The community headline number is
accuracy on the binary-classification version of `general_data.json`;
HaluEval's own GPT-4 baseline lands around 70.

**ANAH-v2** is sentence-level upstream and per-row downstream. The
ANAH-v2 annotator (7B) reports F1 around 0.81 versus human labels on
the English split. A different framing from RAGTruth, but useful as a
qualitative anchor.

Latency budget when picking `--limit`: standard mode is about
11 s/row, deep about 30 s/row, deeper about 50 s/row. 200 deep-mode
rows is roughly a 100-minute run; a full RAGTruth sweep at deep is the
better part of a day.

## 6. Conformal calibration workflow

Verity's aggregator gates verdict at `warn` and `fail` thresholds.
Hand-tuned defaults live in `project/src/config.ts`; calibrated
overrides live alongside the compiled `aggregator.js` at
`dist/calibrated-thresholds.json` (and are picked up at process boot).

To calibrate from any benchmark:

```bash
# 1. Run the harness with --emit-calibration.
npm run bench:ragtruth -- --data C:/AI/verify-data/RAGTruth/dataset/response.jsonl \
  --mode deep --limit 500 --emit-calibration bench/ragtruth-calib.jsonl

# 2. Compute the conformal cut-offs.
npm run calibrate -- \
  --in bench/ragtruth-calib.jsonl \
  --alpha 0.1 \
  --out dist/calibrated-thresholds.json \
  --dataset ragtruth
```

`--alpha` sets the target false-fail rate (default 0.1). `--warn-alpha`
sets the softer warn gate (defaults to `2 * alpha`). The script writes
JSON in the shape the aggregator expects, and the aggregator logs which
source and mode it loaded on next startup, e.g.
`[aggregator] using CALIBRATED thresholds from ... (mode=BIDIRECTIONAL,
alpha=0.1, n=200, warn=11, fail=12.8)`.

From 2026-05-24 calibrated thresholds run in **bidirectional mode** by
default: the conformal cut-off IS the decision boundary, in both
directions. A low nonconformity score can pull a v1 `fail` down
to `pass`. This is what makes the calibration upgrade useful for the
dominant failure mode (over-flagging on RAGTruth-style data). Set
`VERITY_CALIBRATED_THRESHOLDS_ADDITIVE_ONLY=1` to restore the legacy
escalation-only behaviour. `error` verdicts are never overridden in
either mode.

For the maths and the exchangeability caveat, see
[`docs/calibration.md`](../../docs/calibration.md).

## 7. Troubleshooting

- **Most rows verdict=`error`.** A critic is unreachable. Restart
  Ollama (`ollama serve` in a separate terminal) and confirm the two
  critic models are pulled. The MCP plugin handshake between LM Studio
  and the worker also has a known race; see design doc Appendix A.14.
- **Run drags out per-row.** VRAM spill. Open Task Manager → GPU →
  Shared GPU Memory. If non-zero, eject extra LM Studio models, switch
  Thinking off on critics, or reload the worker with a smaller
  context.
- **Mock smoke run shows `auroc: 0.5`.** Expected on a tiny mock; AUROC
  needs both classes and enough rows to rank.
- **`labels` arrays of length 0 after loading.** ANAH-v2 only: every
  sentence in that row mapped to `nofact`, so the row was dropped.
  Verbose mode (`VERITY_BENCH_VERBOSE=1`) logs each skip.
- **`Cannot find module dist/pipeline.js`.** Run `npm run build`
  first.

## 8. Bilingual note for ANAH

The current public ANAH slice (`opencompass/anah`) is entirely
Chinese. The harness does no translation; it relies on the critic
models to handle Chinese natively. Granite 4.1 and Ministral 3B both
advertise multilingual coverage, but neither has been formally
evaluated against Verity on Chinese rows. Flag the result accordingly
when reporting.

The source annotations use the Chinese tag vocabulary `<要点>`,
`<幻觉>`, `<参考>`, `<改正>` with values `无` (none / ok), `矛盾`
(contradictory), `无法验证` (unverifiable), `无事实` (nofact). The
converter (`convert-anah-source.ts`) maps those to the harness's
internal four-token vocabulary (`ok`, `contradictory`, `unverifiable`,
`nofact`) before writing the flat JSONL the harness eats.
