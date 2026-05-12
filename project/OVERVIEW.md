# The Verity — Plain-Language Overview

## In one sentence

It's a **second-opinion service for LLM answers**: when the user types `/verify` after a reply from the worker model, an MCP server runs the answer through a panel of independent critics plus deterministic checks and returns a structured pass/warn/fail verdict.

## What it does

There are two tools exposed via MCP:

- `verify_answer` — *post-hoc* auditing. The worker has already replied; this checks the reply.
- `consult_second_opinion` (`/second`) — *pre-final-answer* consultation. Two cross-family models on different GPUs answer the same question in parallel; a third pass compares them.

Verify has three depth modes:

- **standard** (~11 s) — 2 critics + NLI
- **deep** (~30 s) — adds 2-sample consistency check + perplexity rescore
- **deeper** (~50 s) — 5 samples + regeneration fallback for perplexity

## Where it's effective — and not

The codebase is unusually empirical about what works. Concrete categories from its own audit corpus:

| Test bucket | What it catches | Verdict |
|---|---|---|
| `math-subtle` | Arithmetic errors masked as plausible prose | Recompute pass nails this — deterministic, 100% precision |
| `code-subtle-bug` | Off-by-one, missing null checks | Granite 8B catches; smaller models miss |
| `ctx-entailed` | Claims that the prior context supports | Was 8/8 false positives with the original DeBERTa MNLI model; mitigated 2026-04-18 by switching to the cross-encoder NLI variant; further mitigated by raising the unsupported-claim escalation threshold |
| `code-clean` | Correct code that critics shouldn't flag | A lone 2B critic occasionally false-flags; CRITIC_A weight bumped to 2 to dampen this |
| `hedge-valid` | "I think X" / "approximately Y" — legitimate hedging | Severity-1 nitpicks no longer escalate (WARN raised from 1 to 2 on 2026-04-18) |

**Where it's not effective:**

- Bare claims with **no prior context**. Pairwise intra-answer NLI ("does sentence 3 contradict sentence 7?") tested as **zero signal** on the audit corpus and is now off by default.
- Aesthetic / stylistic complaints — filtered.
- Convergent failure modes — if the worker AND both critics share a training mistake, the panel agrees confidently and is wrong. Mitigated by family diversity (Qwen worker + IBM critics) but not eliminated.
- Anything beyond the critics' 4 KB context window without truncation.
- Recent facts past training cutoffs (everything is local + offline).
- The consistency check catches *low-confidence guessing*, not *consistent overconfidence*. Re-sampling the same model just gives N samples from the same distribution.

## How it works

**Standard mode** (every signal runs in parallel, bounded by a 180 s wall-clock):

1. **Two critics** fire in parallel via `callCritic()`. Each gets a system prompt picked by detected task type (code/prose/reasoning/research) and returns strict JSON: `{verdict, severity 0–5, concerns[], suggested_fixes[]}`.
2. **Claim extraction** — regex heuristic (~0 ms) for standard mode, worker LLM (~1–2 s) for deep modes with regex fallback if the LLM call fails.
3. **NLI check** — DeBERTa-v3-large cross-encoder (~150 ms/pair on CPU) classifies each claim against the premise as entailed / contradicted / neutral.
4. **Recompute pass** — regexes out arithmetic / unit conversions, evaluates them, flags mismatches. Hard-fails on any mismatch; also *suppresses* NLI contradictions on expressions it verified (handles the case where the LLM claim-checker false-flags correct math).
5. **Aggregator** — fixed rules: any critic at fail severity → fail; any NLI contradiction → fail; recompute mismatch → fail; warn-tier signals → warn; otherwise pass.

**Deep / deeper mode** adds:

6. **Consistency** — re-sample the worker 2 or 5 times at temp 0.7; classify each claim against each sample. Divergence ≥ 0.5 → fail, ≥ 0.15 → warn.
7. **Perplexity rescore** — replay through the worker with `echo=true, max_tokens=0` for token-level logprobs. Falls back to regenerate-with-logprobs if the rescore fails.

**Disputes** are computed *after* consensus is decided — a diagnostic table that surfaces critic-vs-critic disagreements (verdict mismatch, concern raised by one but not the other) using token-overlap Jaccard. Never changes the verdict, but the user always sees real disagreement.

## Technology choices

- **Node + TypeScript** — matches the rest of the user's MCP-LMstudio stack; first-class MCP SDK; clean OpenAI SDK types.
- **MCP (Model Context Protocol)** — instead of a custom CLI, this exposes `verify_answer` as a tool the worker model decides to call when it sees `/verify`. Trigger phrase works regardless of which model the user is chatting with.
- **OpenAI-compatible SDK against local servers** — both LM Studio (port 1234) and Ollama (port 11434) speak `/v1/chat/completions`. One client lib covers everything; no vendor-specific code.
- **DeBERTa-v3-large NLI cross-encoder** (1.4 GB ONNX, CPU via `@huggingface/transformers`). The author originally tried generic MNLI-trained DeBERTa; on the audit corpus it returned "neutral" on textbook contradictions. Switched 2026-04-18 to the purpose-built cross-encoder variant. Cross-encoder over bi-encoder because it judges *pairs* directly — much higher contradiction precision.
- **Granite 3.2 8B + 2B as critics** — picked from a 2026-04-17 sweep of 8 small models on AMD Vulkan. Granite 2B hit **144 tok/s warm, 334 ms/call, 4/4 correct** on the test suite. IBM's training stack is distinct from Microsoft (Phi), NVIDIA (Nemotron), Meta (Llama), Google (Gemma), so a Phi+Granite or Granite+Granite panel still has *family diversity* on a single hardware budget. (Two Granites are paired here; the family name is shared but the 8B and 2B were trained from different scratch corpora.)
- **Two backends, not one** — LM Studio holds the worker on the strong GPU, Ollama holds the critics on the weak GPU. Splits VRAM so the worker stays hot for chat and critics stay hot for verification. `OLLAMA_MAX_LOADED_MODELS=2` keeps both critics resident.
- **Worker-as-claim-extractor** — already loaded; reuse beats standing up a second classifier.
- **SelfCheckGPT-style sampling** — published technique (Manakul et al. 2023): hallucinations tend to be *inconsistent* across re-samples while real knowledge is stable. Cheap to implement.

## Adapting to different hardware

The architectural premise — **worker on strong GPU, critics on weak GPU, NLI on CPU** — generalises cleanly. Knobs all live in `config.ts` (everything user-tunable is marked `[ADAPT]`).

**New + old NVIDIA pair (e.g. RTX 4090 + GTX 1080):**
- LM Studio on the strong card → set `WORKER_MODEL_NAME` to your 9–13 B model.
- Ollama on the old card → set `CRITIC_A_MODEL` and `CRITIC_B_MODEL` to whatever 2–4 GB Q4 quants fit. Keep family diversity (e.g. one Granite, one Phi).

**AMD (old) + NVIDIA (new), like the reference machine:**
- Ollama needs the Vulkan build for AMD.
- On Linux you may need `GGML_VK_VISIBLE_DEVICES=1` to pin Ollama to the AMD card.
- Set `OLLAMA_MAX_LOADED_MODELS=2` so both critics stay resident.

**Apple Silicon (single unified-memory GPU):**
- Drop the dual-backend split — point both `LM_STUDIO_URL` and `OLLAMA_URL` at LM Studio (or both at Ollama). Size critics small enough that the worker stays resident.
- DeBERTa on CPU still works fine.

**Asymmetric (one big, one tiny):**
- Drop to a 1-critic panel — edit `ALL_CRITICS` in `critic-configs.ts` to one entry, set `MAX_UNAVAILABLE_CRITICS = 0`, accept the loss of cross-critic disputes.
- Or skip Ollama entirely and load a 2 B critic alongside the worker on the big card.
- For `/second`: set `CONSULT_DUAL=0` to revert to a single-Ollama path (no dual dispatch, no analysis).

**Knobs that almost always need tuning:**
- `CRITIC_TIMEOUT_MS` (45 s default — covers cold JIT load on AMD; faster hardware can lower).
- `PIPELINE_TIMEOUT_MS` — keep at ~3× the slowest critic.
- `WARN_SEVERITY_THRESHOLD` and `FAIL_SEVERITY_THRESHOLD` — tighten if critics are quiet, loosen if noisy.
- Critic system prompts in `prompts.ts` — second-biggest tuning surface after model choice.

**What stays the same on any hardware:** MCP wiring, aggregator rules (consensus logic + recompute suppression of NLI false-flags), the disputes-surfacing logic, recompute / NLI / consistency / perplexity flow. Pure logic; doesn't care which silicon is underneath.
