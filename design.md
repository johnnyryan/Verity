# Verity — Design

LLMs claim untrue things with confidence. Verity catches them. It runs locally on cheap, old hardware. This document explains how it is built and why.

The README (`project/readmev.md`) covers install and daily use. This document covers the design and the lessons that shaped it.

## Current line-up

Four roles. The names below are the current pick; treat them as placeholders. The rest of this document refers to the worker, Critic A, Critic B, and the NLI check.

| Role     | Current model                     | Where it runs              |
|----------|-----------------------------------|----------------------------|
| Worker   | Qwen 3.5 9B (Q4_K_M)              | Strong GPU, via LM Studio  |
| Critic A | IBM Granite 4.1 8B (Q4_K_M)       | Weak GPU, via Ollama       |
| Critic B | Ministral 3B, Mistral AI (Q4_K_M) | Weak GPU, via Ollama       |
| NLI      | DeBERTa-v3-large (ONNX)           | CPU                        |

---

## 1. Goals

1. Catch confident wrong claims from a local LLM.
2. Use cheap, old hardware. Run everything locally.
3. Pick critics from different training families so their blind spots do not overlap.
4. Keep all four models resident. No swapping during use.
5. Fail soft. One critic timing out should not kill the verdict.
6. Stay simple. One JSON tool. One MCP server. One config file.

---

## 2. Reference hardware

The reference build is a 2021 PC.

| Part                  | Role                                  |
|-----------------------|---------------------------------------|
| NVIDIA RTX 5070 Ti    | 16 GB, CUDA. Hosts the worker.        |
| AMD Radeon RX 5700 XT | 8 GB, Vulkan. Hosts both critics.     |
| Intel i5 12th gen     | CPU plus 32 GB system RAM.            |
| Windows 11            | OS.                                   |

Notes:

- The 5700 XT is RDNA1. ROCm no longer supports it. Vulkan is the only viable backend; it runs at about 60-70% of the CUDA throughput per watt.
- The NLI check runs on CPU via ONNX Runtime from Node.js. No Python.
- Other hardware works fine. See "Adapt for your hardware" in the README.

---

## 3. The four roles

The verification pass uses two cross-family critic LLMs plus a non-generative classifier. Different training pipelines make different mistakes; the panel catches more than any single LLM could.

### Worker

The LLM you chat with. Lives on the strong GPU. Verity asks the worker for nothing in standard mode; in deep modes it re-samples the worker to test for consistency.

### Critic A (strong)

Largest critic. Catches subtle code bugs, off-by-ones, missing null checks, citation errors. Weight = 2 in the aggregator so a lone Critic B "fail" cannot outvote a confident Critic A "pass".

### Critic B (fast)

Smaller, and from a different vendor than Critic A: Mistral, where A is IBM. The two critics no longer share a training family. Quick second voice. Weight = 1.

### NLI check (not an LLM)

A 0.4 B encoder transformer. Takes a sentence pair and outputs three numbers: entailment, contradiction, neutral. Cannot generate text; cannot hallucinate. Trained on entailment-labelled data, not on helpfulness preferences. Its mistakes look nothing like a chat model's mistakes; that is the point.

### Family diversity, in plain terms

If both critics share the worker's training data, they all share the same blind spots. Two cross-family models trained on different corpora catch errors a single larger model would miss. The current panel spans three vendors: Alibaba for the worker, IBM for Critic A, Mistral for Critic B. That restores the cross-family axis an all-IBM critic pair had thinned (see Appendix A.3 and the 2026-05-12 critic swap in the change log).

---

## 4. Architecture

```
    Strong GPU                  Weak GPU                CPU
    ┌──────────────┐            ┌──────────────┐        ┌──────────────┐
    │ LM Studio    │            │ Ollama       │        │ ONNX Runtime │
    │  :1234       │            │  :11434      │        │              │
    │              │            │              │        │              │
    │  Worker      │            │  Critic A    │        │  NLI check   │
    │              │            │  Critic B    │        │              │
    └──────────────┘            └──────────────┘        └──────────────┘
            ▲                          ▲                       ▲
            │                          │                       │
            └──────────────────────────┴───────────────────────┘
                                    │
                       ┌────────────────────────┐
                       │ Verity MCP server      │
                       │  :8090/mcp  (Node.js)  │
                       │   verify_answer        │
                       │   consult_second_opinion│
                       └────────────────────────┘
```

The strong GPU only hosts the worker. Both critics share the weak GPU. The NLI runs on CPU. A small Node.js process orchestrates them and exposes two tools over MCP. LM Studio's chat client speaks to that process.

### What happens when you type `/verify`

1. The worker calls `verify_answer` with the question, its answer, and (optionally) prior context.
2. The Verity server fans out in parallel:
   - HTTP POST to Ollama: Critic A.
   - HTTP POST to Ollama: Critic B. (Both critics share one GPU; Ollama serialises them at the hardware level.)
   - Claim extraction plus NLI classification (CPU).
   - Recompute pass (CPU, no LLM).
3. Each critic returns JSON: verdict, severity 0 to 5, concerns, suggested fixes.
4. The aggregator combines all signals into one verdict: pass, warn, fail, or error.
5. A disputes table is computed after the verdict. It surfaces critic-vs-critic disagreement. It never changes the verdict.
6. The tool returns a Markdown block. The worker pastes that block into chat.

### Latency

Wall-clock for standard mode is dominated by Critic A. The two critics serialise on the weak GPU; NLI and recompute run in parallel.

| Stage                                  | Time    |
|----------------------------------------|---------|
| Critic A (8 B class, weak GPU)         | 1.5-3 s |
| Critic B (2 B class, weak GPU)         | 0.5-1 s |
| NLI (parallel with critics, CPU)       | 0.5-2 s |
| Recompute pass (CPU, no LLM)           | <0.1 s  |
| Total, standard mode                   | 3-5 s   |

Deep mode adds 2-sample consistency and a perplexity rescore. Deeper mode raises to 5 samples and adds a regeneration fallback. Wall-clock is dominated by worker re-samples: about 20 s for deep, 30-40 s for deeper.

---

## 5. Context handling

The worker runs at high context. Critics do not need that much. More context adds distractors, not signal (Chen et al., 2024). Three modes manage how much goes through.

- **Minimal** (default). Question and answer only. 2-8 k tokens. Best for code review, maths, and self-contained prose.
- **With context**. Worker passes the earlier messages the answer depends on (documents, specifications, data, constraints). Aim for under 24 k tokens.
- **Full**. Whole conversation. Critic input may exceed a critic's context limit; the pipeline truncates from the head if so and reports the truncation in the verdict.

The NLI check needs a premise. Without prior context, it has nothing to compare claims against and is effectively skipped. An older fallback (claim-against-claim within one answer) was tested as zero signal and is off by default.

---

## 6. The checks

Five checks. Each catches a different kind of wrong.

### Critics (two LLMs)

Read the answer, return structured JSON. System prompt picked by detected task type (code, prose, reasoning, research). The auto-detector in `prompts.ts` picks unless the user overrides with `/verify as code` and so on.

### NLI claim check

Each factual claim is paired with the prior context. The cross-encoder labels each pair as entailment, contradiction, or neutral. Contradictions are a hard fail signal. Two or more "neutral" labels (unsupported by the premise) escalate to warn; one alone is noise.

The cross-encoder cannot generate. The worst case is a wrong label, not a fabricated fact.

### Recompute pass

The cheapest check, and the only one with 100% precision when it fires. A regex pulls arithmetic, range enumerations, and unit conversions out of the answer; each expression is evaluated. A mismatch is a hard fail.

For linear equations and range comprehensions the verifier follows step (3) of Chain-of-Verification (Dhuliawala et al., 2023): the expression to recompute is sourced from the question, and the draft answer is read only for the claimed value or solution. With the draft visible during expression discovery the verifier risks anchoring on the draft and reproducing its mistakes, which is the load-bearing failure CoVe identified. Gated by `RECOMPUTE_INDEPENDENT` (default on); set to 0 for A/B comparison only.

A bonus: when recompute confirms a numeric expression, NLI contradictions whose claim contains that expression are suppressed. This kills the "math-subtle" false positive where the LLM-based claim checker mis-flags correct arithmetic.

### Consistency (deep modes only)

Re-ask the worker N times at temperature 0.7. Compare each re-sample against the original via NLI. The fraction of original claims contradicted or unsupported across re-samples is the divergence score.

K defaults to 5 in deep mode and 8 in deeper mode, following the diminishing-returns curves in Wang et al. 2022 (Self-Consistency, arXiv:2203.11171) and Manakul et al. 2023 (SelfCheckGPT): most of the gain lands by K = 5, with a long tail to ~20. Override via `CONSISTENCY_K_DEEP` / `CONSISTENCY_K_DEEPER`.

Published version: SelfCheckGPT (Manakul et al., 2023). Hallucinations tend to flicker across re-samples; real knowledge stays put.

Catches low-confidence guessing. Does not catch consistent overconfidence.

### Semantic entropy (advisory)

Re-sample the worker a small number of extra times, cluster the samples by meaning (two samples share a cluster when they bidirectionally entail one another under NLI), and report the Shannon entropy of the cluster-size distribution. Low entropy means the worker converges on one meaning across re-samples; high entropy means the surface forms differed but the underlying uncertainty was high — the confabulation signature from Farquhar et al., "Detecting Hallucinations in Large Language Models Using Semantic Entropy", Nature 2024.

Why a meaning-cluster entropy and not a text-edit-distance entropy: token differences flag rare wording even when the answers agree, whereas mutual entailment treats paraphrases as the same cluster. The cross-encoder Verity already loads is enough to do the clustering, so the only added cost is the extra sample batch.

Advisory only. The signal is surfaced in the rendered Markdown block between consistency and perplexity; it never moves the verdict. The consistency check remains the deep-mode hallucination spine.

### Perplexity (deep modes only, advisory)

Read the worker's own token probabilities. Low-probability spans mark where the worker was hesitant. Reported as model uncertainty, and advisory only: it never moves the verdict (see § 7), because token confidence is blind to a confident, fluent hallucination and runs high on rare-but-correct wording. The consistency check is the real guard against confident error; this is a nudge to look closer.

Where the numbers come from. LM Studio exposes logprobs on one endpoint only, its responses API; the chat-completions and completions endpoints return null. So:

1. Forward-pass rescore of the exact answer. Fast (1-2 s), but needs an echo-capable endpoint, such as a llama.cpp side-car, vLLM, or TGI. LM Studio cannot do it.
2. Regenerate through the responses endpoint (deeper mode). The logprobs are exact for the regenerated text, which matches the original on a deterministic answer and is a near-twin otherwise.

If neither is available, the signal is skipped with a note. The pipeline degrades gracefully.

---

## 7. Aggregator

Fixed rules. No machine learning. The aggregator is the only place verdicts are decided.

```
recompute mismatch:                                     fail
any critic.severity >= 3, or NLI contradicts:            fail
consistency divergence >= 0.5 (deep modes only):         fail
any critic.severity >= 2, or NLI unsupported (>= 2):     warn
consistency divergence >= 0.15 (deep modes only):        warn
otherwise:                                                pass
```

Perplexity is deliberately absent from these rules. Since 2026-05-22 it is an advisory note only and never changes the verdict. Semantic entropy (2026-05-23) sits next to perplexity in the same advisory band and likewise does not move the verdict.

Recompute suppression: a verified arithmetic expression cancels any NLI contradiction whose claim contains that expression.

Disputes are computed after the verdict. They list concerns one critic raised but not the other, plus verdict mismatches. Disputes never alter the verdict.

**Conformal thresholds (2026-05-23).** On boot the aggregator looks for `calibrated-thresholds.json` next to its own compiled module. When present, the calibrated cut-offs add a final escalation pass: a continuous nonconformity score (the sum of critic disagreements, NLI flags, and recompute mismatches) is checked against warn and fail cut-offs derived from the (1-alpha)-quantile of a held-out clean set. The escalation is additive — it can take pass to warn or fail, and warn to fail, but never the other way. When the file is absent, the v1 thresholds in `config.ts` are used and the startup log records the source. See `docs/calibration.md` for the run procedure.

---

## 8. MCP tools

Two tools, exposed over standard MCP (JSON-RPC over HTTP, port 8090). Any MCP-aware host can use them.

### `verify_answer`

Post-hoc audit. The worker calls this after composing its answer.

Inputs:

| Field           | Type   | Required | What it is                                     |
|-----------------|--------|----------|------------------------------------------------|
| `question`      | string | yes      | The user's latest question.                    |
| `answer`        | string | yes      | The worker's composed answer (full prose).     |
| `mode`          | string | no       | `standard` (default), `deep`, or `deeper`.     |
| `task_type`     | string | no       | `auto` (default), `code`, `prose`, `reasoning`, `research`. Picks the critic lens; auto-detected from the answer when unset. |
| `prior_context` | string | no       | Earlier chat content the answer depends on.    |

The schema is kept deliberately small; smaller models drop fields when it is wide. `task_type` is optional and defaults to auto-detection, so a typical call still passes only the question and answer plus `mode`.

Output: a JSON blob with critic verdicts, NLI result, recompute / consistency / perplexity blocks, disputes, consensus, and a pre-rendered Markdown summary. The worker pastes the Markdown block verbatim.

### `consult_second_opinion`

Pre-final-answer consultation. The worker may call this before writing its answer.

Two cross-family models answer the same question in parallel, one on each GPU. A third call compares them and returns `{agreements, disputes, table_html, table_md}`.

| Field             | Type   | Required | What it is                                            |
|-------------------|--------|----------|-------------------------------------------------------|
| `question`        | string | yes      | The user's question.                                  |
| `worker_draft`    | string | no       | Optional in-progress draft. Enables an agreement score. |
| `prior_context`   | string | no       | Same semantics as `verify_answer`.                    |
| `model`           | string | no       | Optional Ollama tag override (forces the legacy single-Ollama path). |
| `resolution_mode` | string | no       | `manual` (default) or `auto`.                         |

`resolution_mode` decides who reconciles disagreements:

- `manual`: return both raw answers plus the diff table. The user picks.
- `auto`: the analysis pass also synthesises a `final_answer` that resolves the disputes. The user sees the table below.

`manual` is safer. `auto` commits to one answer at the cost of one extra analysis call.

`/second` and `/verify` are complementary. If the user types `/verify`, the worker must still call `verify_answer` after writing its answer, even if it called `/second` first.

Set `CONSULT_DUAL=0` on smaller hardware to disable the dual-GPU path.

---

## 9. System prompt for the worker

Verity ships with a recommended system prompt. The full text is in `docs/system-prompt.md`. Key clauses:

- Treat `/verify` and `/second` as tool triggers, not English words.
- Source non-trivial claims with a working URL fetched first via the fetch tool. The required citation format is `[N], [author], [publisher], [year], [page], [url]`.
- After `verify_answer` returns, paste the response block verbatim. Do not redraft on your own.
- If the user replies `redraft`, rewrite the answer to address the findings. Every URL in the redraft must be fetched first.
- If the user replies `/verifydeeper` (or `yes`, `OK`, after a `/verifydeeper` offer), call `verify_answer` with `mode='deeper'`.

The prompt also covers the empty case (blank system prompt). The tool descriptions carry the same rules so Verity works without a system prompt at all.

---

## 10. Known limits

- **Vulkan on Windows for the RDNA1 card is second-class.** Driver hiccups happen. The pipeline degrades gracefully: a critic that times out is marked unavailable; the surviving critic still votes. Consensus is "error" only when more than `MAX_UNAVAILABLE_CRITICS` critics fall over.
- **Convergent failure.** Even cross-family critics can share a mistake when the same wrong fact sits in all their training data. The three-vendor panel reduces this; it does not eliminate it.
- **Claim extraction is heuristic in standard mode.** Sentence splitting plus filters for numbers, dates, and named entities. The deep-mode LLM extractor is much better.
- **NLI needs a premise.** Without prior context the check is effectively skipped. The pairwise fallback (claim against claim within one answer) tested as zero signal and is off by default.
- **Consistency catches uncertainty, not confident error.** Re-sampling the same model just gives N samples from the same distribution.
- **LM Studio MCP plugin handshake race.** The first chat sent within a few seconds of an LM Studio restart can miss Verity entirely. See Appendix A.14.
- **Offline.** Verity never calls out. Claims past the worker's training cutoff cannot be checked unless the worker uses a separate fetch tool.

---

## 11. Deferred

- **Debate rounds.** Critics see one another's verdicts and respond. Catches more, at about twice the latency.
- **Hybrid cloud option.** A Groq, Gemini, or Claude call as an extra critic would expand family diversity without local hardware cost. Trade-off: data leaves the device.
- **Bi-encoder NLI pre-filter.** Cheap shortlister for long answers; cross-encoder runs only on the survivors. Cuts NLI cost on long answers in half. Not yet measured.
- **RAGTruth and LLM-AggreFact benchmark harness.** Verity has no measured detection rate. A small driver would feed each benchmark's (question, retrieved context, answer, gold label) tuples through `verify_answer` in standard, deep, and deeper modes, then compute balanced accuracy and AUROC against the labels. Output: a one-line headline number for the readme that replaces "two LLMs plus an NLI check" with a real detection rate. Reference targets: 84% balanced accuracy on RAGTruth and 76% across LLM-AggreFact, reported by HalluGuard's fine-tuned 4B SRM. **Partially landed in v0.2.0**: harness shipped, datasets downloaded, first numbers in (RAGTruth BA 0.61, HaluEval QA BA 0.74 at standard mode with per-dataset calibrated thresholds). The aspirational 84% / 76% targets remain open and depend on a richer signal stack (see also: per-task threshold files below, and deep-mode sweeps).
- **Critic-cited disputed spans.** Critics currently return agree or disagree per claim. Asking each critic to additionally return the answer-text span that triggered the dispute would let the agree-and-disagree table point at the offending clause rather than just naming it. Near-zero added cost; tightens the table. Borrowed from the evidence-grounded justification output used in HalluGuard's SRM.
- **Per-task calibration files (post-v0.2.0).** v0.2.0 confirmed the score distributions differ across task families (HaluEval QA `warn=7.8` vs RAGTruth `warn=11` at the same alpha=0.1). A single `calibrated-thresholds.json` is tuned to a single dominant task family. The cleanest fix is per-task-type files (`calibrated-thresholds-research.json`, `-prose.json`, `-reasoning.json`, `-code.json`) selected by the existing `task_type` parameter on `verify_answer`. The aggregator's load path picks the right file when the verdict ladder runs. Deferred until there is enough cross-task usage to justify the extra moving parts.

---

## 12. Confidence proxy and cloud workers (optional)

Two optional ways to run Verity, both added 2026-05-22. Neither is needed for `/verify`.

**Cloud worker.** A standard `/verify` reads only the question and the answer text; it never calls the worker. So the worker can be a cloud model with no change. Deep and deeper modes do call the worker, for re-sampling, claim extraction, and regeneration; `WORKER_ENDPOINT` and `WORKER_API_KEY` point those at any OpenAI-compatible endpoint. The cost is per-token billing and the answer leaving the machine. Keep the worker local if that matters.

**Confidence proxy.** A built-in chat window cannot be intercepted, and once an answer is generated through it the per-token probabilities are gone, so Verity cannot force a confidence check on every answer there. An external client can: point Open Web UI, Jan, LibreChat, or AnythingLLM at `project/src/proxy/` instead of the backend, and every answer it produces carries a confidence note. The proxy forwards everything byte-for-byte except a fully serviceable plain-text request, which it routes once through the responses endpoint for exact, free logprobs. Anything with tools, structured output, images, or multiple completions passes through untouched and ungraded. This is forced by the API. On LM Studio you cannot have both zero loss of functionality and logprobs on every answer, because logprobs live only on the responses endpoint, which has a smaller feature surface than chat-completions. So the proxy scores the plain-text case and leaves the rest alone.

---

# Appendix A — Implementation Log

What we learned running the system against real audit corpora. Entries are roughly chronological. File and line references point at source comments that record the same decision in code.

## A.1  Critic A migration

The v1 design ran a 14 B critic on the strong GPU. In practice the path was much messier:

| Period      | Critic A model     | Backend          | Result                                                              |
|-------------|--------------------|------------------|---------------------------------------------------------------------|
| early April | 14 B reasoning     | Strong GPU / CUDA | 45 s per verify; KV-cache contention with the worker.              |
| mid April   | 3.8 B mini         | Weak GPU / Vulkan | iGPU misrouting dropped speed to 6 tok/s.                          |
| mid April   | 3.8 B mini (pinned)| Weak GPU / Vulkan | After AMD pinning: 96 tok/s, 4 of 4 correct.                       |
| 2026-04-17  | 1 B (different family) | Weak GPU / Vulkan | 150 tok/s, fits alongside Critic B.                              |
| current     | 8 B Granite        | Weak GPU / Vulkan | Strong critic, weight = 2.                                         |

The 14 B Critic A on the strong card was abandoned because:

- KV-cache pressure. The worker plus Critic A used about 13.9 GB of weights on a 16 GB card. Long-context use ate the rest; Critic A was getting evicted under load.
- A small fast critic on the weak card hits the same quality on `code-subtle-bug` once weighted correctly. See A.3.

## A.2  Small-model sweep

8 candidate critics tested on AMD Vulkan at 4 KB context, Q8 KV cache. Test corpus: 4 cases (one each of `code-clean`, `code-subtle-bug`, `math-subtle`, `hedge-valid`).

| Model                | tok/s warm | per-call ms | correct |
|----------------------|------------|-------------|---------|
| 2 B Granite          | 144        | 334         | 4 of 4  |
| 3.8 B Phi-class      | 96         | 482         | 4 of 4  |
| 1 B Gemma-class      | 150        | 280         | 3 of 4  |
| 3 B Llama-class      | 88         | 510         | 3 of 4  |
| (4 others omitted)   | -          | -           | <= 3 of 4 |

Granite 2 B won fastest and matched Phi at 4 of 4 correct. The 8 B Granite was added afterward as the strong critic.

## A.3  Three critics to two

The v1 design fitted three critics on the 8 GB card at 4 KB context each. In practice three Q4 critics plus Q8 KV cache spilled past the ceiling under load. Ollama started evicting one mid-call, costing a cold-load (about 3 s) on every other verification.

Fix: two critics at 4 KB context with comfortable KV headroom. The aggregator was updated for an N=2 fleet with `MAX_UNAVAILABLE_CRITICS = 1`. A single transient failure still permits the surviving critic to vote.

## A.4  WARN threshold raise

V1 rule: any critic at severity >= 1 -> warn.

Problem: severity-1 nitpicks ("could phrase this better", "missing a docstring") were flipping otherwise-clean answers to warn. Genuine concerns sat at severity >= 2 in the critic prompts.

Fix: `WARN_SEVERITY_THRESHOLD = 2`. The `hedge-valid` corpus case now passes. A `code-null-bug` case still misses, but both critics under-called that one at severity 1, so it was already failing. Net win.

## A.5  NLI model swap

V1 used `Xenova/deberta-v3-large-mnli` (generic MNLI training).

Problem: 0 of 6 contradictions caught on the NLI audit corpus. The model returned "neutral" on textbook contradictions ("Paris is in France" vs "Paris is in Germany").

Fix: switched to `Xenova/nli-deberta-v3-large`, a sentence-transformers cross-encoder purpose-built for entailment / contradiction / neutral. Decisive on real contradictions. About 1 GB download; warmed at boot.

## A.6  Unsupported-claim threshold

V1 rule: any single NLI "unsupported" claim escalated to warn.

Problem: 8 of 8 false positives on `ctx-entailed`. Answers genuinely entailed by the premise were getting "neutral" labels, which the aggregator read as unsupported.

Fix: one unsupported is noise. Two escalate. One alone escalates only when critics also raised something.

## A.7  Pairwise intra-answer NLI: off by default

V1 fallback: when no prior context, NLI ran claim-against-claim within the answer.

Problem: zero signal across the audit corpus. The cross-encoder is too eager to call mutually-supportive declarative statements "neutral". Cost: up to about 190 NLI calls per request for no benefit.

Fix: `NLI_REQUIRE_CONTEXT=true` by default. Pairwise mode is opt-in for A/B work.

## A.8  Aggregator off-by-one

V1 rule: `if (unavailable.length >= MAX_UNAVAILABLE_CRITICS) return "error"`.

With a 2-critic panel and `MAX_UNAVAILABLE_CRITICS = 1`, the gate fired when *any* critic was unavailable, even though the survivor had a verdict. The config comment said "1 critic can still vote"; the operator contradicted the intent.

Fix: gate flipped to strict `>`. One unavailable critic in a 2-panel still produces a verdict from the survivor.

## A.9  Recompute pass plus NLI suppression

Added to address `math-subtle` false positives, where the LLM claim-checker mis-flagged correct arithmetic as contradicted.

Two rules:

1. Recompute mismatch -> hard fail. No model uncertainty.
2. When recompute confirms an arithmetic expression, NLI contradictions whose claim contains that expression are dropped. Suppression applies to contradictions only, not to unsupported labels.

## A.10  Disputes table

The aggregator emits one consensus verdict. Users were missing the disagreement when one critic flagged a real issue and the other passed.

Fix: `computeDisputes` (token-Jaccard match, threshold 0.4) surfaces concern-only-in-A, concern-only-in-B, and verdict-mismatch entries as a diagnostic table. Diagnostics never change the verdict.

## A.11  Second-opinion tool

Added `consult_second_opinion`. Pre-final-answer, not post-hoc. Two cross-family models on different GPUs answer in parallel; an analysis pass compares them.

Default: weak-GPU leg via Ollama, strong-GPU leg = the worker via LM Studio. Set `CONSULT_DUAL=0` for the single-Ollama legacy path.

## A.12  Boot-time warmup

Two niceties:

- The NLI classifier and tokeniser are pre-loaded after `app.listen`. The first `/verify` no longer pays the ~1 GB ONNX cold-load (about 5 s).
- A single `LlmClient` factory at `src/llm/client.ts`. Five files used to instantiate per-module OpenAI clients independently; all now route through one cached factory keyed on `endpoint|apiKey`.

## A.13  Wire-id rename

Through 2026-05-11 the JSON output used `phi4_reasoning` and `nemotron_mini` as critic keys, left over from v1 models that no longer ran. The keys kept showing up in worker hallucinations: the worker would call `verify_answer`, get the rendered Markdown back, then in its reasoning trace invent fake critic verdicts using the legacy wire ids.

Fix on 2026-05-11: renamed to `granite_3_2_8b` and `granite_3_2_2b`. The tool description also has an explicit "do not invent these names" guardrail to catch any remaining training-data leakage.

**2026-05-20 follow-up:** the wire ids were generalised again to `critic_a` / `critic_b`. Pinning the slot name to a specific model meant every model swap (e.g. the 2026-05-20 04:50 swap to `granite4.1:8b` + `ministral-3:3b`) left the wire id lagging. The model-agnostic names follow the convention already established by `CRITIC_A_MODEL` / `CRITIC_B_MODEL` env vars and `displayName: "Critic A" / "Critic B"`, so a future swap is now a one-file change in `critic-configs.ts`.

## A.14  LM Studio MCP plugin handshake race

**Symptom.** The first chat sent after restarting LM Studio gets a prose-only answer; `verify_answer` is not called. The second chat onwards calls the tool. Reproducible: 100% of restarts, identical input.

**Cause.** A race between LM Studio's UI readiness and its MCP plugin registration. On startup:

1. The UI accepts text within 1 to 2 seconds.
2. In parallel, LM Studio connects to each MCP plugin. Each connection has three phases: WebSocket client created, plugin replies with its tool list, LM Studio commits the registration.
3. Only after the third phase is each plugin's tool list visible to the worker.

Plugin registration can take 2 to 10 seconds. Sample timing from a clean Verity restart:

```
08:00:15  verity            Client created          (Phase A)
08:00:17  verity            setToolsProvider        (Phase C, +2 s)
08:00:20  fetch             setToolsProvider        (+5 s)
08:00:21  brave-search      setToolsProvider        (+6 s)
```

A chat message sent in the 2-to-6-second gap is compiled with whatever tools are registered at that moment. Tools not yet registered are simply absent from the request. The worker has no way to call a tool it does not know exists, so it falls back to prose. By the time the user reads the reply and types a second message, all plugins are registered.

**Workarounds (no code change required).**

- Wait about 5 seconds after LM Studio finishes loading.
- Or send a one-word "hi" first. Plugin registration completes while you read the reply.
- Or script the launch: start LM Studio, wait for `setToolsProvider` events on every plugin in the log, then send the first user message.

**Documentation status.** Not documented by LM Studio. Their MCP plugin docs at `lmstudio.ai/docs/app/plugins/mcp` cover installation and configuration but do not call out the race. The MCP specification does not constrain host UX timing. Worth filing with LM Studio: the chat input should be gated on plugin readiness, or a "plugins loading" indicator should be shown.

**Why this matters for Verity.** A chat-only MCP plugin would degrade quietly: the user just does not get tool-augmented behaviour on the first turn. For Verity, the user has explicitly typed `/verify`. Getting prose instead is a visible failure rather than a quiet degradation.

## A.15  Verbose tool description (2026-05-12 reversion)

Over a single day in May 2026, the `verify_answer` tool description grew from about 2 k characters to over 18 k as failure modes were patched with new prohibitions ("do not paraphrase", "do not redraft", "paste verbatim"). Each addition was reasonable on its own; the combined description fragmented the worker's attention. The model spent up to 7 seconds reconciling overlapping rules before acting, and sometimes ran out of output budget before pasting the block.

Fix: cut back to about 6 k characters. Move the "paste verbatim, do not redraft" rule into the rendered block itself, where the worker sees it next to the data. Trust the description to convey the trigger and the flow; trust the block to enforce paste behaviour at the point of use.

## A.16  Tool name: `verify_previous_answer` -> `verify_answer`

The original tool name was `verify_previous_answer`. Smaller workers read "previous" as "the previous turn's answer". In a fresh chat with no prior turn, the worker had nothing to verify and fell back to prose. In a chat with a greeting before the user's real question ("hi" -> "Hello!" -> "Do X? /verify"), the worker passed the greeting as the "previous answer" to verify.

Fix on 2026-05-12: renamed to `verify_answer`. The tool description now explicitly says "the answer to audit is the one you just composed in the current turn, not something from prior chat history". The `answer` parameter description says the same.

## A.17  Verity returns the answer in its response

The worker sometimes passes a composed answer to `verify_answer` but never emits that prose in a visible assistant message. The user sees a stack of tool-call accordions (search, fetch, verify) followed by the verdict block, with no answer in between.

Fix on 2026-05-12: the rendered Markdown block now echoes the answer at the top, under an `## Answer` heading, followed by the verdict table. The block is what the worker pastes; the answer is visible in chat regardless of where the worker put it during composition.

---

## Change log

Append-only record of substantive changes to Verity (code, scripts, and this design doc). Most recent on top. The Implementation-Log entries above (A.1 through A.17) remain the canonical narrative for the early-development period; this section gives a dated index of what shipped, in reverse chronological order, with cross-references back to the Appendix A entries and to git when applicable.

### Versioning

Dated entries carry a `vX.Y.Z` marker on the headings that correspond to a tagged release, and `package.json` moves with the latest release. Semver: behavioural-default changes or new public-API surface bump the minor; bug fixes with no API change bump the patch; the major is reserved for breaking changes to the MCP tool interface. Released versions to date:

- **v0.1.0** (12 May 2026) — pre-publication baseline; the tool-name rename and critic line-up at the time.
- **v0.2.0** (22 May 2026) — first published release. Cloud worker, logprob confidence via the `/v1/responses` endpoint, perplexity demoted to advisory, confidence proxy, docs caught up to the critic swap.
- **v0.3.0** (24 May 2026) — calibration layer starts working: HalluGuard adoptions, upgrades #2-6, benchmark harnesses + dataset downloads, bidirectional conformal calibration, score unification across the verdict gate and bench harness, first measured detection numbers against published hallucination test sets.

Earlier work is recorded by date only.

### 2026-05-24 — v0.3.0: the calibration layer starts working (bidirectional default + score unification + first measured numbers)

Glossary for this entry, in plain language. *Balanced accuracy* is the mean of "fraction of real hallucinations Verity flagged" and "fraction of clean answers Verity passed"; 0.5 is chance, 1.0 is perfect. *False-positive rate* is the fraction of clean answers Verity wrongly flagged; lower is better. *Precision* is the fraction of Verity's flags that turned out to be real hallucinations. *AUROC* (area under the receiver-operating curve) is a number from 0 to 1 that measures how well the underlying score separates clean from hallucinated, independent of where the threshold sits; 0.5 is chance, 1.0 is perfect. *Alpha* is the conformal "target error rate" knob: lower alpha means stricter (fewer false alarms, more misses), higher alpha means looser (more catches, more false alarms). *Nonconformity score* is the single number Verity computes per answer from its critic, NLI and recompute signals; the threshold gate turns that number into pass / warn / fail. *V1 ladder* is the original heuristic threshold rules in `config.ts` that shipped before conformal calibration arrived on 2026-05-23.

Three changes ship together. The first is a behavioural-default change; the second a bug fix without which the first was inert; the third the empirical work that exposed both.

**Bidirectional default.** The 2026-05-23 conformal-calibration upgrade shipped in escalation-only mode: calibrated cut-offs could only push a verdict up (`pass` → `warn` → `fail`), never relax one down. A 50-row RAGTruth sweep at standard mode produced identical numbers before and after calibration was switched on: balanced accuracy 0.46, false-positive rate 1.00 (every clean answer flagged), recall 0.93, precision 0.27. The startup log confirmed the calibrated thresholds (`warn=11`, `fail=12.8`, computed from a 200-row sweep at alpha 0.1) were loaded; the verdict gate just had nothing to escalate because the v1 rules already flagged every clean row, and the escalation-only logic could not pull them back down. The fix: `aggregator.ts` now runs calibrated thresholds in **bidirectional** mode by default. The conformal score is the decision boundary in both directions: score at or above `fail_score_threshold` → fail; at or above `warn_score_threshold` → warn; below both → pass. The v1 rules still contribute their counts to the nonconformity score (they make up the score that gets compared to the threshold), but they no longer gate the verdict. Set `VERITY_CALIBRATED_THRESHOLDS_ADDITIVE_ONLY=1` to restore the legacy escalation-only behaviour, which is the right choice when the v1 rules are a deliberate safety floor and calibration is intended only to catch under-flagging. An `error` verdict (too many critics unavailable) is never overridden in either mode; it is a system-state signal, not a quality signal.

**Score unification — the bug that hid the first finding.** With bidirectional turned on, a HaluEval QA 200-row sweep still produced `verdict=pass` on every row. AUROC was 0.88 in the same run, meaning the underlying score did rank hallucinated above clean correctly; but the score the aggregator computed at verdict time was always below the threshold. Root cause: two functions had drifted out of sync. The bench harness emitted one score into the calibration file (sum of: contradictions, unsupported claims, recompute mismatches, each critic's severity, each critic's concerns count, and consistency divergence × 5 — observed range roughly 0 to 15). The aggregator computed a thinner score for the verdict (sum of: count of critics that disagreed with the majority, contradictions, unsupported claims, recompute mismatches — observed range roughly 0 to 5). Calibration set thresholds in the first range; the gate decided verdicts in the second range; the two never overlapped. The fix is one source of truth: `computeNonconformityScore` in the aggregator now uses the richer formula, and `bench-common.ts` imports the aggregator's function rather than carrying its own copy. The `aggregate()` call site passes consistency through so divergence enters the score in deep modes. The one test that assumed the thinner formula was updated.

**First measured numbers.** With both fixes landed, the calibration layer actually does something. Headline results from 50-row standard-mode sweeps against the real RAGTruth and HaluEval QA datasets, with each dataset's thresholds calibrated separately at alpha 0.1:

| Dataset      | v1 BA | calibrated BA | v1 FPR | calibrated FPR | v1 precision | calibrated precision |
|--------------|------:|--------------:|-------:|---------------:|-------------:|---------------------:|
| HaluEval QA  |  0.73 |      **0.74** |   0.53 |       **0.24** |         0.65 |             **0.75** |
| RAGTruth     |  0.46 |      **0.61** |   1.00 |       **0.28** |         0.27 |             **0.41** |

On both datasets, calibrated bidirectional cuts false-positive rate by more than half and raises precision by 10-15 percentage points. Recall trades down (HaluEval 1.00 → 0.72; RAGTruth 0.93 → 0.50) as the conformal cut-off lets the hardest cases through. That is the right trade for the chat use-case: when Verity flags an answer it is now meaningfully more likely to be right than under the v1 ladder, and the false-alarm rate falls into a range that does not train the worker to ignore the signal.

AUROC is a property of the underlying score and the dataset, not of the threshold choice, and so is unchanged by calibration: HaluEval QA AUROC 0.88 (the score separates clean from hallucinated well), RAGTruth AUROC 0.57 (the score barely separates — on the summarisation and data-to-text rows that make up most of RAGTruth, clean and hallucinated answers have nearly identical score distributions). RAGTruth still benefits from calibration despite the weak signal because the calibrated threshold puts the cut-off at the right place even when the distributions overlap; calibration cannot manufacture separation the signal does not produce, but it stops the over-flagging the v1 ladder produced on those distributions.

Calibration is **per-task**. HaluEval-calibrated thresholds (`warn=7.8`, `fail=8.0`) and RAGTruth-calibrated thresholds (`warn=11`, `fail=12.8`) are not interchangeable. The score distributions differ across task families. A deployment that runs Verity across mixed task types should hold per-task-type threshold files or accept that one calibration is tuned to a single dominant task. Captured as deferred work in § 11.

Tests: the loader skips `calibrated-thresholds.json` under `node --test` (an inherited calibration file from a developer's earlier run would otherwise silently shift verdict assertions across the suite; `NODE_TEST_CONTEXT` detection broadened from `=== "1"` to `!= undefined` since Node sets it to `"child"`). Bidirectional behaviour exercised by mutating `ACTIVE_THRESHOLDS` within a single test scope and restoring in `finally`. New tests: "low score pulls fail to pass", "high score pulls pass to fail", "error verdict never overridden", "calibrated thresholds skipped under node --test". The obsolete "calibrated escalation: cannot override fail to pass" test is replaced. 281 → 284 tests passing. `docs/calibration.md` and `project/scripts/README-bench.md` updated. Datasets downloaded under attribution-only licences (RAGTruth MIT, HaluEval MIT, ANAH Apache-2.0) into `C:/AI/verify-data/`; LLM-AggreFact (CC-BY-ND-4.0, HuggingFace contact-form gated) and FaithBench (CC-BY-NC-SA-4.0) deferred to user-side downloads. Attribution roster at `NOTICE.md`.

Polish landed in the same release: a single module-level `IS_UNDER_TEST` constant replaces the two near-duplicate test-mode checks that had drifted out of sync in `aggregator.ts` (loader gated on `NODE_TEST_CONTEXT !== undefined`, startup log on `=== "1"`; Node sets the variable to `"child"`, so the loader skipped under test but the startup log still printed). The brittle `process.argv.some((a) => a.includes("--test"))` half of the test detection is removed. A duplicate JSDoc block above `computeNonconformityScore` (the obsolete pre-unification description) is deleted so IDE tooling shows the right formula. The `hallucinationScore` adapter in `project/scripts/bench-common.ts` gained a clearer doc string flagging the historical drift it exists to prevent. The Chinese-tag regex in `project/scripts/convert-anah-source.ts` is annotated with its truncate-at-first-`<` assumption (fine for the four hallucination-label values we read, but warns any future caller widening its use). Style: `if (typeof div === "number")` tightened to `if (div != null)`.

### 2026-05-23 — HalluGuard review: two deferred items added

Reviewed the HalluGuard framework. It bundles three separate detectors: spectral analysis of attention matrices, NTK-based risk decomposition over gradients, and a fine-tuned Qwen3-4B reasoning model with LoRA + ORPO that emits JSON verdicts with evidence-grounded justification spans. The two internal-state methods need attention matrices and gradients and so are unreachable to a tool that speaks OpenAI-compatible HTTP; logprobs remain Verity's ceiling for in-band model signal. The fine-tuned SRM does the same job as Verity's critic panel but with a trained artefact to maintain, which conflicts with the zero-maintenance goal. Two adoptable ideas captured in § 11: a RAGTruth and LLM-AggreFact benchmark harness (to replace mechanism description with a measured detection rate in the readme), and critic-cited disputed spans (borrowed from the SRM's evidence-grounded justification output). The "loud liar versus quiet failure" finding (some models' hallucinations are obvious in one signal, others need several) is consistent with the 2026-05-22 decision to keep perplexity advisory and to rely on cross-family critics rather than any single number.

**Upgrade #2 — CoVe step-3 independence applied to the recompute pass.** Reread Chain-of-Verification (Dhuliawala et al., 2023, arXiv:2309.11495) and noticed Verity's deterministic recompute had the same draft-anchoring failure mode their step (3) was designed to remove: the linear-equation and enumeration detectors both took `question + "\n" + answer` as a single source string for expression discovery, so an answer that re-stated the problem incorrectly seeded the solver with the wrong equation and the verifier then "confirmed" the draft's mistake. `signals/recompute.ts` now splits the source into an `expressionScope` (the question alone, by default) and a `claimScope` (the answer); the solver only registers equations the question supplies and reads only the answer's claimed solution. New `RECOMPUTE_INDEPENDENT` flag in `config.ts` (default on) restores legacy behaviour for A/B work. Six new tests added; 270/270 pass.

- **Upgrade #3: consistency K bumped to the literature sweet spot.** `CONSISTENCY_SAMPLES_DEEP` 2 → 5 and `CONSISTENCY_SAMPLES_DEEPER` 5 → 8. Both Wang et al. 2022 (Self-Consistency, arXiv:2203.11171) and Manakul et al. 2023 (SelfCheckGPT, arXiv:2303.08896) place K ≥ 5 at the diminishing-returns knee; deeper mode pushes further into the tail. New env overrides `CONSISTENCY_K_DEEP` / `CONSISTENCY_K_DEEPER`. Source comment added near the K-read site in `signals/consistency.ts`; § 6 "Consistency" prose updated to match. Test added: `consistency.test.ts` asserts the defaults and that the worker is called exactly K times.
- **Upgrade #4 — Semantic entropy (Farquhar et al., Nature 2024).** New `project/src/signals/semantic-entropy.ts` clusters worker re-samples by bidirectional NLI entailment and reports Shannon entropy over the cluster-size distribution. High entropy means surface-different answers with the same underlying uncertainty — a better signal than token-level perplexity for confident-but-wrong cases. The pipeline draws a parallel sample batch (consistency.ts is owned by another agent and does not expose its sample stream) and attaches `semantic_entropy` and `semantic_cluster_count` to the `ConsistencyResult` post-hoc. Surfaced advisory-only in the rendered Markdown block, between consistency and perplexity; never flips the verdict. Env gate `VERITY_SEMANTIC_ENTROPY=0` to disable; `VERITY_SEMANTIC_ENTROPY_SAMPLES` to tune N (default 3).
- **Upgrade #5 — Conformal calibration (Yadkori 2024 / Quach 2023).** New `project/scripts/calibrate-thresholds.ts` reads a calibration JSONL emitted by the bench harness (`--emit-calibration <path>`) and writes warn / fail score cut-offs at the (1-alpha)-quantile of clean-row nonconformity scores. `aggregator.ts` loads `calibrated-thresholds.json` at boot from the directory next to itself, with the path overridable via `VERITY_CALIBRATED_THRESHOLDS_PATH`. When loaded, the calibrated cut-offs can escalate `pass` → `warn` / `fail` and `warn` → `fail`, but never the other way; v1 multi-axis rules remain the floor. Missing file → silent fallback to `config.ts` defaults with a startup log noting the source. `docs/calibration.md` covers the maths, the run procedure, and how to swap thresholds. The conformal machinery lands; the measured run waits for a labelled RAGTruth / LLM-AggreFact pass — same "machinery now, measurement later" framing as the existing bench harness deferred entry.
- **Upgrade #6 — Critic shuffle + verbosity-bias check (Zheng 2023, MT-Bench / Chatbot Arena, arXiv:2306.05685).** Critic dispatch order is now shuffled per `/verify` call via `shuffleCritics()` in `pipeline.ts`; the timeout-fallback path now indexes the shuffled list so a timed-out critic surfaces under the correct id. Gate `CRITIC_SHUFFLE=0` restores deterministic dispatch for replay. Verbosity-bias investigation: reviewed `aggregator.ts` — the verdict is rule-based over critic severity (max), NLI counts, recompute mismatches, and consistency divergence; `totalConcerns` is computed for the summary text only and never enters the verdict ladder. A critic listing 1 versus 8 concerns at the same severity therefore lands the same verdict. Length-normalisation is moot here; documented in a comment near the aggregation point rather than forcing a fake change.
- **Benchmark datasets downloaded under attribution-only licences.** RAGTruth (MIT), HaluEval (MIT) and ANAH (Apache 2.0) cloned into `C:/AI/verify-data/`. LLM-AggreFact (CC-BY-ND-4.0, eval-permitted but gated by a HuggingFace contact form) and FaithBench (CC-BY-NC-SA-4.0, non-commercial) are user-side downloads only; the runbook documents how. Attribution roster captured in `NOTICE.md` at the repo root. ANAH's source layout is topic-grouped with Chinese-tagged annotations (`<要点>`, `<幻觉>`, etc.); new `project/scripts/convert-anah-source.ts` flattens 783 topic rows into 1,846 per-sentence rows in the shape the existing harness eats. The public ANAH slice is Chinese-only; the bilingual claim refers to the unreleased ANAH-v2 set. Real-data smoke tests against a closed-port worker confirmed all three downloaded harnesses parse the upstream format end-to-end (3 rows each, formatting + label-mapping + TSV emission all correct).

### 2026-05-22 — v0.2.0: Logprob confidence works (responses endpoint); perplexity demoted to advisory; confidence proxy; cloud worker; docs caught up to the critic swap

Committed as `916caed`; 238/238 tests pass.

- **Perplexity / logprobs now functional, via the responses endpoint only.** LM Studio exposes token logprobs on `/v1/responses` alone; `/v1/chat/completions` and `/v1/completions` return `logprobs: null` by design (upstream lmstudio-ai/lms#60). Verified live on the PC (GGUF + MLX gemma) and on the Mac's MLX engine. `signals/perplexity.ts` now uses the responses path; the echo-rescore and chat-completions paths remain as fallbacks for hosts that expose logprobs differently (vLLM `prompt_logprobs`, TGI `decoder_input_details`, llama.cpp `n_probs`).
- **New `signals/confidence.ts`.** Maps a token-logprob stream to a band (ok / mild / low / very_low) and a recommended depth, on three axes: weakest token, whole-answer perplexity, and low-confidence density. Framed as model uncertainty, never correctness.
- **Perplexity demoted to advisory (§ 6, § 7).** A three-member Opus panel found the signal blind to confident, fluent hallucinations and noisy on rare-but-correct wording. It no longer contributes to the consensus; `aggregator.ts` drops the `perplexity flagged -> warn` rule. The consistency check is the deep-mode spine.
- **Cloud worker (§ 12).** `WORKER_ENDPOINT` / `WORKER_API_KEY` thread through every worker call, so a cloud model can be the worker for deep and deeper modes. Standard `/verify` never calls the worker.
- **Confidence proxy (§ 12, `project/src/proxy/`).** Optional transparent proxy for external chat apps. Pass-through for everything except a fully serviceable plain-text request, which routes once through `/v1/responses` for exact, free logprobs plus a note. Capability matrix in `docs/confidence-proxy.md`.
- **Docs caught up to the 2026-05-12 critic swap.** The § 3 line-up, Critic B description, family-diversity note, and the Known-limits / Deferred entries still named Granite 3.2 and "thinner than v1"; they now reflect Granite 4.1 8B (IBM) + Ministral 3B (Mistral), three vendors. `config.ts` had been correct since the swap; only the prose lagged.

### 2026-05-20 — Phase A through F commit wave + initial git import + critic-model swap + `web/` landing page

The entire codebase was committed to a fresh git repository on this date in 9 commits between 05:01:34 and 05:46:20 +0100, with two further parallel workstreams (critic-model swap and `web/` page build) following immediately after. Two Claude Code sessions drive this date end-to-end, both launched from the Lore project's working directory (which is why no transcript exists under `C:\Users\johnn\.claude\projects\C--AI-verify\`):

- **Session `60f4f2d1-da52-4efd-8375-2549c6a87dc3`** (Lore project, 14 MB, 5,323 lines) — opened 2026-05-09T20:14 with *"Examine `C:\AI\verify` and consider whether the code and the system can be made more efficient, faster, easier to maintain"*. Ran intermittently over 11 days, culminating in a sustained 04:00-04:50 block on 2026-05-20 that produced the 9 commits below plus the critic-model swap.
- **Session `5ab66aa5-fc14-445f-abc3-cc8810dd5e99`** (Lore project, 3.1 MB, 417 lines) — opened 2026-05-20T04:21 with *"Delegate a panel of agents to create a web animation that shows what Verity does. It should look like something Apple would do to explain important features"*. Produced the `web/` directory in ~31 minutes via six parallel section agents.

The Phase A-F push was driven by **six parallel code-review subagents** dispatched ~03:55 on 2026-05-20 (the user's preceding prompt at 2026-05-12: *"Delegate agents to do a `/code-review` of all verity"*). Each review covered a disjoint surface — pipeline + aggregator + config; critics + NLI + claim extraction; `/second`; MCP server + tool surface; operational scripts + tests; audit-plan synthesis — and returned CRITICAL/IMPORTANT/NICE-TO-HAVE/POSITIVE structured reports. A subsequent steelman/skeptic agent pair argued the findings before the Phase A-F TodoWrite landed at 04:05 with 35 numbered items mapped to specific files and lines. Each phase shipped under a build → 175/175-test → `git commit` → `git push backup master` gate. The commit messages reproduce the audit findings; the underlying audit reports are in the subagent transcripts under `60f4f2d1.../subagents/`. File mtimes show much wider scope than the 9 commits suggest: 8 TypeScript source files (`aggregator.ts`, `pipeline.ts`, `consult.ts`, `recompute.ts`, `classifier.ts`, `index.ts`, `prompts.ts`, `client.ts`), the test suite, all of `dist/`, plus a new `web/` directory containing six HTML section files totalling ~166 KB. All commits report "Tests: 175/175 pass."

**Parallel critic-model swap (04:50, after Phase F polish).** A background research subagent (`abaf97e100afaf330`) ran 04:07-04:08 surveying 2026 Granite alternatives — Granite 4.1 (Apr 29 2026), Llama 3.1/3.2, Ministral-3, OLMo 3, Gemma 4, Phi-4-mini, ZAYA1-8B — against the 8 GB weak-card constraint and the cross-family-diversity goal. Recommendation: `granite4.1:8b` (post-trained with LLM-as-Judge filtering, IFEval 87.1, BFCL V3 68.3) + `llama3.2:3b`. The user accepted Granite 4.1 but rejected Llama 3.2 as too old; the agent re-shortlisted and landed on `ministral-3:3b` (Mistral AI, December 2025 "instruct-2512" build, Apache 2.0, MCP-validated tool calling). Both models pulled successfully in background. `config.ts` defaults updated at 04:50 with explanatory comments. **Net effect: cross-family axis restored** — IBM (Granite) + Mistral + Qwen worker = three pretraining corpora. The `design.md` § 3 "thinner than v1" caveat and Appendix A.13's `granite_3_2_8b` / `granite_3_2_2b` wire ids both pre-dated this swap and lagged the code; rather than chase the model with another model-specific rename, the follow-up landed later the same day as a generic `critic_a` / `critic_b` pair (see A.13 "2026-05-20 follow-up"), so future swaps don't drag the wire id with them.

**`web/` landing page (04:21-04:52).** The 5ab66aa5 session dispatched six parallel section agents (hero, problem, panel, flow, checks, architecture) plus the orchestrator wrote `index.html` shell + nav + Quickstart CTA in the same window. Engineering constraints honoured: scoped CSS prefixes per section, transform/opacity-only animations gated behind `@media (prefers-reduced-motion: no-preference)`, no emojis/external fonts/images/network calls/framework deps, reveals via existing `[data-reveal]` + `data-reveal-delay` + IntersectionObserver-driven `verity:section-enter` events in `scroll.js`. Verified mobile rendering at 375 px (the preview tool times out on desktop viewports under the heavy SVG/CSS animation load). The user immediately followed up by asking the assistant to review `https://www.iccl.ie/digital-data/verity-mcp/` — the first transcript-anchored evidence of a public publication target for Verity. The `_assemble.py` script that concatenates fragments into `index.html` was already present pre-session.

**Other 2026-05-20 transcript-evidenced facts not in the commit bodies:**
- Two backup-mirror remotes are configured: `C:\AI\verify\.git` (working) + `C:\AI\backups\verity.git` (bare). Each commit was `git push backup master`-ed immediately.
- No `gh` CLI is installed. A GitHub mirror would need `winget install GitHub.cli` first.
- Phase A-F's "build + test + commit" loop took ~45 minutes wall-clock end-to-end across 9 commits, gated by the 175-test suite at every step.

- **`190c8b6` 05:46:20 — Phase F polish (final).** F1: trim `verify_answer` tool description from ~3.5 k chars to ~1.6 k (Qwen 3.5 9B was spending up to ~7 s reconciling overlapping FLOW / DO NOT / ALSO CALL sections; the agent-preface inside the rendered block already enforces paste-verbatim co-located with the data). F3: promote seven inline truncation budgets (120 / 140 / 400 / 600 / 800 / 1000) from `aggregator.ts` to named constants in `config.ts`, each one documenting which renderer cell or finding bullet it sizes. See also A.15.
- **`db9c398` 05:42:58 — Phase E (part 3) + Phase F polish.** E14: `VerifyOutput.critics` is now `Record<string, CriticResult>` keyed on critic id rather than a literal object listing each id statically — adding or renaming a critic is now a one-file change in `critic-configs.ts`. E15: `Warm-CriticModels` reads warmup targets from `ollama /api/tags` rather than hardcoding model names; survives critic swaps without code change. E16: defensive ICD-array handling in `Get-AmdVulkanIcd` (some AMD drivers register multiple Vulkan ICD paths on one registry value as a string array; the `[string]` cast had been joining them with whitespace). E17: remove dead `Apply-VerityQwenConfig` / `Revert-VerityQwenConfig` pair from `start-verity.ps1` (disabled 2026-05-11 after the empty `operation.fields` broke LM Studio chat-schema validation; AMD-pinning launcher now solves the original problem). E18: new `-SkipProbe` switch on the AMD launcher (the synchronous 60 s `/api/generate` Probe-LoadModel can wedge the launcher for a full minute on cold start). F2: hash the `(endpoint, apiKey)` cache key in `llm/client.ts` with SHA-256 so the literal API key never appears in the Map's key space. F4: replaced stale critic-config header comments still referring to Phi-4-mini and Nemotron Mini (last used 2026-04-17).
- **`b1ff283` 05:35:10 — Phase E (part 2): renderer & verdict-logic polish.** E7: strip backticks from recompute-mismatch `expr_text` before wrapping in inline-code backticks (an expression text containing a backtick would close the inline-code span prematurely). E8: use `CONSISTENCY_FAIL_THRESHOLD` / `CONSISTENCY_WARN_THRESHOLD` from `config.ts` for the consistency chip colour rather than hardcoded 0.5 / 0.15. E9: move `AGGREGATOR_WEIGHTED_VOTE` from direct `process.env` to an exported config.ts constant with full `[ADAPT]` documentation. E10: generalise `computeDisputes` to all C(n,2) critic pairs (no-op on the current 2-critic panel; matters if `CRITIC_C` is ever re-enabled). E12: pass `amdModelName` explicitly through the analysis pass (`renderDisputesTable` / `renderDisputesMarkdown` / `analysisUserPrompt` had hardcoded "Granite 3.2 8B" even though `SECOND_OPINION_MODEL` is configurable). E13: `buildDatePreamble` uses local TZ rather than UTC (users near midnight in their local zone would otherwise see "today" off by a day).
- **`860955f` 05:25:49 — Phase E (part 1): timer hygiene + concurrency robustness.** E2: move `clearTimeout` to `finally` in `call-critic.ts` and `extract-claims-llm.ts` so the `AbortController` timer is always cleared (the prior `clearTimeout` inside `try` was unreachable on throw; orphan setTimeouts fired on dead AbortControllers and kept the Node event loop alive past process intent). E3: cache the lazy NLI pipeline promise only on success in `classifier.ts` (a rejected promise from a cold-load OOM was poisoning every subsequent `classifyEntailment` call). E4: surface partial-sample failure in `ConsistencyResult.notes` ("Generated N/M alternate samples; K sample(s) failed"). E5: parallelise per-claim entailment classification via `Promise.all` (sequential JS-side awaits had been the dominant cost on premise-bearing `/verify` calls with `NLI_MAX_CLAIMS=14`). E6: wrap each pipeline signal with its own timeout-with-fallback so a global ceiling preserves partial work (the outer `Promise.all` on pipeline timeout was discarding every critic / NLI / recompute result that had finished, replacing the whole tuple with synthesised "timed out" stubs).
- **`7304d17` 05:20:47 — Phase D: /second tool robustness.** D1: bound `withEndpointLock` acquisition (the per-endpoint promise chain had no timeout; a leaked prior promise — e.g. an `AbortController` failed to cancel an underlying HTTP socket — would wedge subsequent callers forever). Lock acquisition now races against `ENDPOINT_LOCK_MAX_WAIT_MS` (5 minutes). D2: split `extractAnalysisJson` into a detailed variant distinguishing three failure modes — `empty`, `no_json_found` (the canonical runaway-reasoning-trace failure where the model consumes `max_tokens` without ever emitting JSON), and `parse_error` (a candidate JSON span found but `JSON.parse` rejected). The backwards-compatible wrapper is preserved.
- **`ae18161` 05:18:36 — Phase C: ReDoS hardening + regex concurrency safety.** C1: cap `ARITHMETIC_RE`'s expr quantifier to `{2,200}` (the unbounded `{2,}` let the engine do quadratic work on adversarial input; the post-filter length check at line 288 had been happening AFTER the engine had already spent the time). C2: eliminate shared-regex `lastIndex` races — six module-level regexes with the `g` flag (`ARITHMETIC_RE`, `LEAP_DAYS_RE`, `LINEAR_EQ_RE`, `VAR_CLAIM_RE`, the `UNIT_CONSTANTS` specs, and all `INJECTION_PATTERNS` entries) had been reset via `re.lastIndex = 0` before each use; with concurrent `/verify` calls on the same Node process, one's reset would corrupt the other's iteration position. Each call now constructs a fresh `RegExp` instance via `new RegExp(RE.source, RE.flags)`. C3: cap the approval-mode injection pattern's `[^.]*?` to `[^.]{0,200}?` (unbounded lazy quantifier on a permissive character class was a textbook ReDoS surface).
- **`7ebf147` 05:14:43 — Phase B: security hardening.** B1: replace the 50 MB `express.json` limit with `MAX_REQUEST_BYTES` (4 MB default; override via `VERITY_MAX_REQUEST_BYTES`). Add per-field char caps (`MAX_QUESTION_CHARS=32k`, `MAX_ANSWER_CHARS=200k`, `MAX_PRIOR_CONTEXT_CHARS=800k`) enforced in `handleToolCall` before pipeline dispatch. B2: bind the HTTP server to `127.0.0.1` by default rather than `0.0.0.0` — Verity has no authentication and trusts every caller; localhost binding eliminates the session-hijack surface for the default deployment. Override via `VERITY_HOST`. E1 (rolled in): validate enum membership for `mode` / `task_type` / `context_mode` before casting (the previous `as TaskType` bypassed validation; a caller passing `task_type: "rm -rf"` reached the pipeline unchecked). B3: defensive installer hardening — `Set-StrictMode -Version Latest` and `ValidatePattern` on `-RepoUrl` accepting only `https github.com` URLs in `install-verity.ps1`; `set -euo pipefail` and matching REPO_URL regex in `install-verity-mac.command` (was just `set -e`); dropped a no-op `cd`.
- **`e2375e9` 05:10:00 — Phase A: critical correctness fixes.** A1: drop the 500 ms inter-critic await + 100 ms pre-loop sleep in `pipeline.ts`; critics now fire genuinely in parallel via `ALL_CRITICS.map`. The original guard against an Ollama cold-load race is no longer relevant because `OLLAMA_MAX_LOADED_MODELS=2` keeps both critics resident. Wall-clock savings: ~500 ms per `/verify`. A2: wrap `extractClaimsLLM` in try/catch (the function documents null-on-failure but a hard throw bypassed the regex fallback). A3: replace raw `fetch` on `${OLLAMA_URL}/chat/completions` with the shared `getLlmClient` (the raw call 404s when `OLLAMA_URL` is the documented root URL without `/v1/`, and every LLM-NLI claim silently returned null). A3+E11: replace naive `text.match(/\{[\s\S]*?\}/)` JSON extraction with `stripReasoningTraces` + `findBalancedJsonObject`. A4: fix the unsupported-escalation comment that claimed `maxSeverity >= 1` while the code used `WARN_SEVERITY_THRESHOLD` (currently 2). Updated `displayName-reflects-model` test to the post-2026-05-12 role-only design.
- **`619540b` 05:01:34 — Initial commit: Verity multi-agent verification MCP server.** First git initialisation of the project. 87 files, ~27,798 insertions. Snapshot of everything pre-existing at that moment: MCP server source under `project/src/`, operational scripts (`start-verity.ps1`, `install-verity.ps1`, `install-verity-mac.command`, `CLI/ollama-amd.ps1`), documentation (`readme.md` + `design.md` at root), the parallel `GITHUB/` copy prepared for public release, and the 175-test suite under `project/src/__tests__/`. Excluded via `.gitignore`: `node_modules/`, `dist/`, `*.tsbuildinfo`, `project/verifier-backup-*`, `project/verifier-archive/`, `.env`, `*.log`. Every prior change in this design doc's Appendix A predates this commit and is not represented in git history.

### 2026-05-13 — design.md and readme.md mtime baseline

The pre-git design.md and readme.md files reached their current shape on 2026-05-13. The git initial commit a week later carried both forward unchanged. The 60f4f2d1 Lore-launched session was open across this date but the visible turns are short clarification questions; the substantive documentation work is upstream of this.

### 2026-05-12 — Tool-name and verbose-description reversion (A.15, A.16, A.17) + handshake-race documentation + fake-URL-on-redraft fix

Three Appendix-A fixes plus two significant new findings landed on this date, all visible in Lore session `60f4f2d1-da52-4efd-8375-2549c6a87dc3` (lines ~4290-4540).

**The three documented fixes:** the `verify_answer` tool description was cut back from ~18 k characters to ~6 k after the verbose description fragmented the worker's attention (A.15). The tool was renamed from `verify_previous_answer` to `verify_answer` after smaller workers had been reading "previous" as "the previous turn's answer" (A.16). The rendered Markdown block now echoes the answer at the top under an `## Answer` heading so it is visible in chat regardless of where the worker placed it during composition (A.17).

**Newly anchored: A.14 first-authored on this date.** The full A.14 text (LM Studio MCP plugin handshake race, the 3-phase A/B/C registration model, the 2-6 second window, the sample log timestamps) was written into the design doc at the time then under `C:\AI\Lore\multi-agent-verifier-design.md`. The sample timestamps in A.14 are empirically observed — a scheduled monitor task (`bnz34lk0t`) was set up to watch the LM Studio auth log filtered for `setToolsProvider` and `Client created` events, captured the relevant timing pattern, then handed it to the design-doc Edit. The text in `C:\AI\verify\design.md` today is verbatim from that turn.

**Newly surfaced: fake-URLs-on-redraft.** Same turn (line 4360), the user reported: *"in testing, verity (correctly) said the LLM had not used a plurality of sources. When I asked for a redraft, many sources were given but they were fake…"*. Fix landed in the same session: the "Awaiting your reply" block in `aggregator.ts` lines 885-906 now explicitly says *"Every URL in the redraft MUST be fetched first via the fetch tool to confirm it resolves. Do not invent URLs to address a 'needs more sources' finding — fabricated URLs are worse than no URLs. If you can't find a working source for a claim, drop the claim."*. The same wording was duplicated into the `verify_answer` tool description's "redraft" handler section. This is a textbook instance of the "fix-by-strengthening-the-contract-text" pattern: explicit clauses co-located with the data the worker is about to act on, rather than far-away tool-description clauses the small worker won't reliably follow.

### 2026-05-11 — Wire-id rename: `phi4_reasoning` / `nemotron_mini` → `granite_3_2_8b` / `granite_3_2_2b` (A.13)

The legacy v1 critic keys had been showing up in the worker's reasoning traces as hallucinated critic verdicts — the worker would call `verify_answer`, get the rendered Markdown back, then in its own chain-of-thought invent fake verdicts using the legacy wire ids it remembered from training data. Renamed alongside an explicit "do not invent these names" guardrail in the tool description.

### 2026-05-10 — "Is Verify novel?" + audience-pivot to competent generalist

Lore session `60f4f2d1` line 788 (13:30): *"is Verify novel?"*. Drove the survey subagents on 2026-05-20 (`a5403dc3a442a0e00` commercial+MCP verifiers, `aa9ac0c6c256b2c12` open-source verifiers), and reshaped the design doc / readme audience. Same date, lines 822 + 828: *"What is 'the worker'"* and *"what is an 'off by one'"*. Both questions established that the design doc's audience is competent generalists, not LLM specialists. The plain-English glossary at the top of `design.md` and the framing of `readme.md` both descend from these two questions.

### 2026-05-09 — Long Lore-session begins; A.7 + A.8 + A.12 landed

Lore session `60f4f2d1-da52-4efd-8375-2549c6a87dc3` opens at 20:14 with *"Examine `C:\AI\verify` and consider whether the code and the system can be made more efficient, faster, easier to maintain"*. Same evening, Appendix A.7 (`NLI_REQUIRE_CONTEXT=true` by default — pairwise intra-answer NLI tested as zero signal), A.8 (aggregator off-by-one on the 2-critic panel — gate flipped from `>=` to strict `>`), and A.12 (boot-time NLI warmup + `LlmClient` factory consolidation across 5 call-sites) all landed in this session per their dating tags. The git initial commit of 2026-05-20 carries all three forward; this date is the actual landing, not the commit timestamp.

### 2026-04-21 evening — Folder rename: `C:\AI\verifier\` → `C:\AI\verify\` + project sub-tree split

At 2026-04-21 21:39:37 the user directed: *"Move all files required for verify to `C:\AI\verify` and move all other files from the verify project to `C:\AI\verify\project`."* The Node.js source moved under `project/`; the operational layer (`install-verity.ps1`, `start-verity.ps1`, `Verity.lnk`, etc.) settled at root. This is the rename that created the present odd situation where the directory is named `verify` but every user-visible artefact says "Verity". Captured in session transcript `d8abaaf3-5a2c-4150-9187-05df2cd6b5d2.jsonl` line 2471.

### 2026-04-21 — Phase 3 verify-critic disputes shipped + AMD-Ollama pinning hardened

The `computeDisputes` token-Jaccard 0.4 dispute-detection logic landed between the `verifier-backup-2026-04-20-212456/` and `verifier-backup-2026-04-21-101412/` snapshots (`diff -rq` shows aggregator.ts, pipeline.ts, types.ts and the aggregator test changed). Captured as A.10 above. Also on 2026-04-21: the user directed creation of a desktop shortcut to load AMD Ollama (*"How do we avoid the AMD Ollama issue occuring again? Create a desktop shortcut to load AMD Ollama"* at 22:21) — origin of `CLI/ollama-amd.ps1` and the `Verity.lnk` / `Verity Stop.lnk` launchers. Session: `d8abaaf3`.

### 2026-04-20 evening — Dual-endpoint `/second` dispatch shipped, then reverted

Made `consult_second_opinion` fire two parallel LLM calls — one to Ollama on AMD, one to LM Studio on NVIDIA — putting the worker's idle slice to work alongside the AMD critic. Code passed `npm test` (90/93, 3 pre-existing aggregator failures unchanged), built cleanly, and was then reverted on user direction: *"no keep critics on AMD."* Full restoration recipe with the three file diffs survives at `C:\AI\verify\project\verifier-archive\dual-dispatch-2026-04-20.md`; kill-switch was `PARALLEL_CRITICS=0`. The energy reallocated to Phase 3 verify-critic disputes (above). Session: `d8abaaf3`.

### 2026-04-20 — Multi-reboot stability sequence + verifier-autostart fixes

A run of system crashes (RAM-near-OOM, LM Studio DOWN, Verifier DOWN, AMD card disappearing from LM Studio) traced largely to GPU-affinity environment variables being process-global, so a misconfigured Who-research MCP could push Ollama onto the NVIDIA card and break Verity. Resolution: per-tool launcher scripts that set `VK_DRIVER_FILES` explicitly rather than relying on inherited environment. The HW-threshold monitor task iterated v2 → v3 → v5 across reboots (one-shot alerts, ASCII-only event strings after Unicode broke the task-notification renderer, tighter RAM-near-OOM gating). User directive at 08:44: *"Amend the automatic loader to check for this problem and fix it in future."* Session: `724ef3f8` → `d8abaaf3`.

### 2026-04-19 — Recompute pass design + Nemotron Mini critic test (negative result)

User prompted (`724ef3f8` L1394): *"would a small nemotron model be better than granite"* — tested, did not improve, removed. Same day saw the deterministic-arithmetic recompute pass go from plan to implementation: *"propose a plan for Deterministic arithmetic / enumeration recompute pass"* (L1408). Captured as A.9 above (the recompute / NLI-suppression rule pair).

### 2026-04-18 — Project genesis: NLI A/B/C test + autonomous test plan + literature review + critic sweep

Day-1 session `724ef3f8` opens with the user's autonomous-test brief: *"read the two md files at `C:\AI` and create a plan to do the following: 1. extensively test and further optimise for accuracy and speed using available system resources... 2. examine scholarly literature in relevant domains and evaluate whether this project contributes anything new. 3. If so, plan a scholarly paper. Do this unattended."* Outputs spanning the day:

- A.5 — NLI model swap from `Xenova/deberta-v3-large-mnli` to `Xenova/nli-deberta-v3-large` (0/6 contradictions caught → decisive on real contradictions)
- A.6 — Unsupported-claim threshold raised from 1 to 2
- A.7 — Pairwise intra-answer NLI disabled by default (`NLI_REQUIRE_CONTEXT=true`); tested as zero signal
- A.4 — WARN severity threshold raised from 1 to 2 to stop nitpick escalation
- A.2 — 8-candidate small-model sweep on AMD Vulkan at 4 KB context, Q8 KV; Granite 2 B won fastest at 144 tok/s with 4/4 correct
- A.3 — Three-critic → two-critic refit; `MAX_UNAVAILABLE_CRITICS = 1` so a single transient failure still permits the survivor to vote
- A.11 — `consult_second_opinion` tool added (pre-final-answer, not post-hoc)
- A.12 — Boot-time NLI warmup + `LlmClient` factory consolidation (five files used to instantiate per-module OpenAI clients independently; all now route through one cached factory keyed on `endpoint|apiKey`)
- A.14 — LM Studio MCP plugin handshake race documented (the 2–6 s window where the first chat after restart misses tool registration)

Hardware-equivalence side-research session (`fffb2352`) explored AMD GPUs newer than the RX 5700 XT in European pricing; no upgrade landed.

### Pre-2026-04-18 — Pre-transcript history (Appendix A summary)

The Implementation Log entries A.1 through A.3 capture the pre-transcript period:

- A.1 — Critic A migration from 14 B reasoning on the strong GPU (45 s per verify, KV-cache contention with the worker) through 3.8 B Phi-class on the weak GPU (initial iGPU misrouting → 6 tok/s, then 96 tok/s after AMD pinning) to the final 8 B Granite + 2 B Granite pair both on the weak GPU
- A.2 — The 8-candidate small-model sweep that selected Granite (recorded 2026-04-18 above)
- A.3 — Three-critic to two-critic refit (KV-cache spill above the 8 GB ceiling at three Q4 critics, Ollama evicting mid-call costing ~3 s cold-load on every other verification)

No git history exists for any pre-2026-05-20 work. The Appendix A narrative above is the canonical record of these early decisions.
