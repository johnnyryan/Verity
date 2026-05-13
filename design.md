# Verity — Design

LLMs claim untrue things with confidence. Verity catches them. It runs locally on cheap, old hardware. This document explains how it is built and why.

The README (`project/readmev.md`) covers install and daily use. This document covers the design and the lessons that shaped it.

## Current line-up

Four roles. The names below are the current pick; treat them as placeholders. The rest of this document refers to the worker, Critic A, Critic B, and the NLI check.

| Role     | Current model               | Where it runs              |
|----------|-----------------------------|----------------------------|
| Worker   | Qwen 3.5 9B (Q4_K_M)        | Strong GPU, via LM Studio  |
| Critic A | IBM Granite 3.2 8B (Q4_K_M) | Weak GPU, via Ollama       |
| Critic B | IBM Granite 3.2 2B (Q4_K_M) | Weak GPU, via Ollama       |
| NLI      | DeBERTa-v3-large (ONNX)     | CPU                        |

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

Smaller, same vendor as Critic A, distinct training corpus. Quick second voice. Weight = 1.

### NLI check (not an LLM)

A 0.4 B encoder transformer. Takes a sentence pair and outputs three numbers: entailment, contradiction, neutral. Cannot generate text; cannot hallucinate. Trained on entailment-labelled data, not on helpfulness preferences. Its mistakes look nothing like a chat model's mistakes; that is the point.

### Family diversity, in plain terms

If both critics share the worker's training data, they all share the same blind spots. Two cross-family models trained on different corpora catch errors a single larger model would miss. The current panel uses one vendor for both critics; the cross-family axis is thinner than v1 (see Appendix A.3). Re-introducing a second vendor when VRAM allows is on the deferred list.

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

A bonus: when recompute confirms a numeric expression, NLI contradictions whose claim contains that expression are suppressed. This kills the "math-subtle" false positive where the LLM-based claim checker mis-flags correct arithmetic.

### Consistency (deep modes only)

Re-ask the worker N times at temperature 0.7. Compare each re-sample against the original via NLI. The fraction of original claims contradicted or unsupported across re-samples is the divergence score.

Published version: SelfCheckGPT (Manakul et al., 2023). Hallucinations tend to flicker across re-samples; real knowledge stays put.

Catches low-confidence guessing. Does not catch consistent overconfidence.

### Perplexity (deep modes only)

Score the answer's tokens. Tokens with low log-probability flag spans the worker was hesitant about; those are often where hallucinations hide.

Two strategies, tried in order:

1. Forward-pass rescore. Fast (1-2 s). Works only if the worker model supports completion-style scoring.
2. Regenerate with logprobs enabled (deeper mode only). About 8 s. The new answer may differ from the user's original, but the uncertainty signal is still meaningful.

If both fail, the signal is skipped with a note. The pipeline degrades gracefully.

---

## 7. Aggregator

Fixed rules. No machine learning. The aggregator is the only place verdicts are decided.

```
recompute mismatch:                                     fail
any critic.severity >= 3, or NLI contradicts:            fail
consistency divergence >= 0.5 (deep modes only):         fail
any critic.severity >= 2, or NLI unsupported (>= 2):     warn
consistency divergence >= 0.15, or perplexity flagged:   warn
otherwise:                                                pass
```

Recompute suppression: a verified arithmetic expression cancels any NLI contradiction whose claim contains that expression.

Disputes are computed after the verdict. They list concerns one critic raised but not the other, plus verdict mismatches. Disputes never alter the verdict.

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
| `prior_context` | string | no       | Earlier chat content the answer depends on.    |

The schema is small on purpose. Smaller models drop fields when the schema is wide.

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

Verity ships with a recommended system prompt. The full text is in `project/readmev2.md`. Key clauses:

- Treat `/verify` and `/second` as tool triggers, not English words.
- Source non-trivial claims with a working URL fetched first via the fetch tool. The required citation format is `[N], [author], [publisher], [year], [page], [url]`.
- After `verify_answer` returns, paste the response block verbatim. Do not redraft on your own.
- If the user replies `redraft`, rewrite the answer to address the findings. Every URL in the redraft must be fetched first.
- If the user replies `/verifydeeper` (or `yes`, `OK`, after a `/verifydeeper` offer), call `verify_answer` with `mode='deeper'`.

The prompt also covers the empty case (blank system prompt). The tool descriptions carry the same rules so Verity works without a system prompt at all.

---

## 10. Known limits

- **Vulkan on Windows for the RDNA1 card is second-class.** Driver hiccups happen. The pipeline degrades gracefully: a critic that times out is marked unavailable; the surviving critic still votes. Consensus is "error" only when more than `MAX_UNAVAILABLE_CRITICS` critics fall over.
- **Family diversity is thinner than v1.** The current panel uses one vendor for both critics. A second vendor when hardware allows would close the gap.
- **Claim extraction is heuristic in standard mode.** Sentence splitting plus filters for numbers, dates, and named entities. The deep-mode LLM extractor is much better.
- **NLI needs a premise.** Without prior context the check is effectively skipped. The pairwise fallback (claim against claim within one answer) tested as zero signal and is off by default.
- **Consistency catches uncertainty, not confident error.** Re-sampling the same model just gives N samples from the same distribution.
- **LM Studio MCP plugin handshake race.** The first chat sent within a few seconds of an LM Studio restart can miss Verity entirely. See Appendix A.14.
- **Offline.** Verity never calls out. Claims past the worker's training cutoff cannot be checked unless the worker uses a separate fetch tool.

---

## 11. Deferred

- **Debate rounds.** Critics see one another's verdicts and respond. Catches more, at about twice the latency.
- **Hybrid cloud option.** A Groq, Gemini, or Claude call as an extra critic would expand family diversity without local hardware cost. Trade-off: data leaves the device.
- **A non-IBM critic.** Replace one of the current critics with a Phi-4 or similar at the same memory budget. Restores the cross-family axis.
- **Bi-encoder NLI pre-filter.** Cheap shortlister for long answers; cross-encoder runs only on the survivors. Cuts NLI cost on long answers in half. Not yet measured.

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