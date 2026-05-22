/**
 * ═══════════════════════════════════════════════════════════════════════════
 *  MACHINE-SPECIFIC CONFIGURATION
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *  This file is the primary place to adapt the verity to your specific
 *  setup. Almost every value you might need to change on a different machine
 *  is here.
 *
 *  When moving this verity to another PC, read through this file first.
 *  Everything you'd realistically need to tweak is flagged with an
 *  [ADAPT] comment.
 */

// ───────────────────────────────────────────────────────────────────────────
// Server port
// ───────────────────────────────────────────────────────────────────────────

/**
 * [ADAPT] Port this MCP verity server listens on.
 * Must match the URL in LM Studio's MCP client configuration.
 * Defaults to 8090 to sit alongside the other MCP-LMstudio servers
 * (filesystem 8080, fetch 8081, memory 8082, git 8083).
 */
export const SERVER_PORT = Number(process.env.VERIFIER_PORT ?? 8090);

/**
 * [ADAPT] Network interface to bind the MCP server to. Defaults to
 * localhost (loopback only) because Verity has no authentication and
 * trusts every caller. Binding to 0.0.0.0 exposes the server to the
 * local network; only set VERITY_HOST=0.0.0.0 if you understand the
 * threat model and want to.
 *
 * 2026-05-12: was implicitly 0.0.0.0 via `app.listen(port, cb)` with
 * no host arg. The code-review flagged session-hijack risk if the
 * server is reachable off-host; localhost binding eliminates that
 * surface entirely for the default deployment.
 */
export const SERVER_HOST = process.env.VERITY_HOST ?? "127.0.0.1";

// ───────────────────────────────────────────────────────────────────────────
// Request size limits  [security: DoS guards]
// ───────────────────────────────────────────────────────────────────────────

/**
 * [ADAPT] Maximum total request body size accepted by the HTTP server,
 * in bytes. The worker model is untrusted (it can pass arbitrary
 * strings via the MCP tool call); a generous limit invites trivial DoS
 * via one huge `answer` or `prior_context`. 4 MB is comfortably more
 * than any legitimate /verify payload (typical: 10-100 KB) but well
 * below the previous 50 MB default.
 */
export const MAX_REQUEST_BYTES = Number(
  process.env.VERITY_MAX_REQUEST_BYTES ?? 4 * 1024 * 1024
);

/**
 * [ADAPT] Per-field length caps (characters). Enforced in
 * handleToolCall before the pipeline runs. Beyond these lengths the
 * critics' context windows would truncate anyway and the latency cost
 * is borne by the verity process; cleaner to reject up front.
 */
export const MAX_QUESTION_CHARS = Number(
  process.env.VERITY_MAX_QUESTION_CHARS ?? 32_000
);
export const MAX_ANSWER_CHARS = Number(
  process.env.VERITY_MAX_ANSWER_CHARS ?? 200_000
);
export const MAX_PRIOR_CONTEXT_CHARS = Number(
  process.env.VERITY_MAX_PRIOR_CONTEXT_CHARS ?? 800_000
);

// ───────────────────────────────────────────────────────────────────────────
// Upstream model servers
// ───────────────────────────────────────────────────────────────────────────

/**
 * [ADAPT] LM Studio's OpenAI-compatible endpoint.
 * Default 1234 is LM Studio's standard local-server port.
 * If LM Studio runs on a different host, change "localhost" here.
 */
export const LM_STUDIO_URL =
  process.env.LM_STUDIO_URL ?? "http://localhost:1234/v1";

/**
 * [ADAPT] Ollama's OpenAI-compatible endpoint.
 * Default 11434 is Ollama's standard port.
 * If Ollama runs on a different host, change "localhost" here.
 */
export const OLLAMA_URL =
  process.env.OLLAMA_URL ?? "http://localhost:11434/v1";

// ───────────────────────────────────────────────────────────────────────────
// Model identifiers  [★ UPDATE THESE FOR YOUR MACHINE ★]
// ───────────────────────────────────────────────────────────────────────────
//
// Each constant below must EXACTLY match the id/tag the backend reports.
// Use these commands to discover the right string on your host:
//
//   LM Studio (for the worker + Critic A):
//     curl http://localhost:1234/v1/models
//     → use the "id" field verbatim. Examples seen in the wild:
//         "qwen/qwen3.5-9b"                ← this machine (2026-04)
//         "qwen3.5-9b-instruct"            ← older LM Studio naming
//         "Qwen/Qwen3.5-9B-Instruct"       ← HF-style full path
//
//   Ollama (for Critic B + Critic C):
//     ollama list
//     → use the NAME column verbatim. Examples:
//         "gemma4:e4b", "gemma4:e4b-q4_k_m"
//         "llama3.2:3b", "llama3.2:3b-instruct-q4_K_M"
//
// To SWAP a critic to a different model (different family, size, or quant):
// change only the corresponding constant here. The calling code lives in
// critics/critic-configs.ts and has no model-specific assumptions; if the
// new model has a smaller native window, set the critic's `contextLimit`
// (and `contextHeadroom`) there so callCritic() trims prior_context to fit.
//
// To MOVE a critic between backends (e.g. Gemma from Ollama to LM Studio):
// don't touch this file — edit `src/critics/critic-configs.ts` and change
// the critic's endpoint + apiKey. The model tag then needs to match what
// the new backend reports.

/**
 * [ADAPT ★] Worker model id. The worker is the model a user is chatting
 * with in LM Studio — it's the one that CALLS this verity via MCP and
 * also provides re-samples / LLM claim extraction for deep modes.
 */
export const WORKER_MODEL_NAME =
  process.env.WORKER_MODEL ?? "qwen/qwen3.5-9b";

/**
 * [ADAPT] Worker endpoint + API key. Default to the local LM Studio server
 * (its key is ignored). These are the ONLY knobs needed to point the
 * worker-dependent deep/deeper signals (consistency re-sampling, LLM claim
 * extraction, perplexity regeneration) at a different backend — including a
 * CLOUD model:
 *
 *   OpenAI:   WORKER_ENDPOINT=https://api.openai.com/v1
 *             WORKER_API_KEY=sk-...   WORKER_MODEL=gpt-4o
 *   Anthropic / Gemini (not OpenAI-compatible): run an OpenAI-compatible
 *             gateway (e.g. LiteLLM) and point WORKER_ENDPOINT at it.
 *
 * Standard /verify never calls the worker — it only reads the question and
 * answer text — so a cloud worker needs no change there. See the readme
 * section "Cloud model as the worker".
 */
export const WORKER_ENDPOINT =
  process.env.WORKER_ENDPOINT ?? LM_STUDIO_URL;
export const WORKER_API_KEY =
  process.env.WORKER_API_KEY ?? "lm-studio";

/**
 * [ADAPT ★] Critic A model tag (Google-family critic on Ollama/AMD).
 *
 * History on this machine:
 *   Phi-4 14B on LM Studio/CUDA  (2026-04 start)
 *     -> 45s per verify, VRAM fight with Qwen
 *   Phi-3.5-mini 3.8B on Ollama/Vulkan
 *     -> iGPU misrouting dragged speed to 6 tok/s
 *   Phi-4-mini 3.8B on Ollama/Vulkan  (after AMD pinning)
 *     -> 96 tok/s, 4/4 correct, but 2.9 GB doesn't coexist with Granite
 *        under Ollama's conservative 4 GiB-available-on-AMD budget
 *   Gemma 3 1B on Ollama/Vulkan  (current, 2026-04-17)
 *     -> 150 tok/s, 1.2 GB. Fits cleanly alongside Granite 3.2 (1.8 GB).
 *        Both critics stay resident, no eviction, no cold-load per call.
 */
export const CRITIC_A_MODEL =
  process.env.CRITIC_A_MODEL ?? "granite4.1:8b";
// 2026-05-12: bumped from granite3.2:8b (Mar 2025) to granite4.1:8b
// (Apr 29 2026). The newer model is post-trained with LLM-as-Judge
// filtering — literally the critic role — and reports IFEval 87.1,
// BFCL V3 68.3, both up from Granite 3.2.
//
// VRAM note: granite4.1:8b is ~5.3 GB at Q4_K_M; combined with the
// 3 GB Ministral-3 3B Critic B, the pair uses ~8.3 GB on an 8 GB
// card — tight. If KV pressure shows (Ollama evicting one critic
// mid-call, cold-load latency on every other /verify), switch to
// `granite4.1:8b-q3_K_M` (~4.3 GB) for a combined ~7.3 GB footprint
// with KV-cache headroom intact. Set via:
//   export CRITIC_A_MODEL=granite4.1:8b-q3_K_M

/**
 * [ADAPT ★] Critic B model tag in Ollama.
 * Must match `ollama list`.
 *
 * Winner from the Phase-3 sweep (2026-04-17) across 8 small models tested
 * on AMD Vulkan at 4k context, Q8 KV cache: IBM Granite 3.2 2B came in
 * fastest (144 tok/s, 334 ms warm) AND matched Phi-4-mini at 4/4 correct
 * on the test suite. Different training pipeline from Microsoft/NVIDIA,
 * giving real family diversity. Only 2.1 GB so plenty of VRAM headroom.
 */
export const CRITIC_B_MODEL =
  process.env.CRITIC_B_MODEL ?? "ministral-3:3b";
// 2026-05-12: bumped from granite3.2:2b (Mar 2025) to ministral-3:3b
// (Dec 2025 / "instruct-2512" build, Mistral AI). Restores the
// cross-family axis that the old "both Granite" pair gave up: now
// Critic A is IBM, Critic B is Mistral, worker is Qwen — three
// distinct pretraining corpora and post-training recipes. Apache
// 2.0 license, 256k native context (irrelevant for a critic but
// good), tool-calling validated with MCP demos. ~2-3 GB at Q4_K_M.

/**
 * [ADAPT ★] Critic C model tag in Ollama.
 *
 * Currently UNUSED — see src/critics/critic-configs.ts where ALL_CRITICS
 * was reduced to 2 entries (Phi-4-mini + Nemotron Mini) after observing
 * that 3 critics at 4k context each spilled past the 5700 XT's 8 GB. The
 * 2-critic design holds 8k context per model comfortably.
 *
 * Keep this value so it's easy to re-enable a third critic later by
 * adding it back to ALL_CRITICS. `llama3.2:3b` is still on disk.
 */
export const CRITIC_C_MODEL =
  process.env.CRITIC_C_MODEL ?? "llama3.2:3b";

// ───────────────────────────────────────────────────────────────────────────
// Context limits
// ───────────────────────────────────────────────────────────────────────────
//
// Per-critic context windows live on each CriticConfig in
// critics/critic-configs.ts (`contextLimit` + `contextHeadroom`). callCritic
// uses those values directly to trim prior_context. The previously-exported
// PHI4_MAX_CONTEXT / PHI4_CONTEXT_HEADROOM / OLLAMA_MAX_CONTEXT constants
// were never read anywhere (audit 2026-05-09) and were removed.

// ───────────────────────────────────────────────────────────────────────────
// Generation parameters
// ───────────────────────────────────────────────────────────────────────────

/**
 * Critic generation parameters. Low temperature because we want focused
 * critique, not creative writing.
 *
 * [ADAPT] If critics start returning verbose rambles, lower max_tokens.
 * If they return truncated verdicts, raise it.
 */
// CRITIC_TEMPERATURE = 0 makes critic output deterministic at the
// per-token level (tied-breaking aside). Regression tests become
// reproducible. 2026-04-18 tuning: was 0.2.
export const CRITIC_TEMPERATURE = 0.0;
// CRITIC_MAX_TOKENS reduced from 1200 -> 600 after observing that
// actual critic JSON responses are 50-200 tokens. The cap only matters
// when a critic goes runaway (reasoning loops, JSON-inside-JSON).
// 600 still leaves plenty of room for legitimate verbose concerns.
export const CRITIC_MAX_TOKENS = 600;
export const CRITIC_TOP_P = 0.9;

// ───────────────────────────────────────────────────────────────────────────
// Timeouts
// ───────────────────────────────────────────────────────────────────────────

/**
 * [ADAPT] Per-critic timeout in milliseconds.
 *
 * Observed on this machine (2026-04) — warm steady state:
 *   Phi-4 14B on CUDA        : ~2–5 s (plain phi-4, no reasoning)
 *   Nemotron Mini 4B (Vulkan) : ~0.8 s
 *   Llama 3.2 3B (Vulkan)     : ~0.7 s
 *
 * 45s leaves headroom for the first-ever call per model, which pays a
 * JIT-load cost (~3–8 GB read from disk). If a reasoning model is
 * re-introduced as Critic A, expect 60–120 s and raise this accordingly.
 */
export const CRITIC_TIMEOUT_MS = 45_000;

/**
 * [ADAPT] NLI model inference timeout per claim. DeBERTa-v3-large-mnli
 * runs at ~150 ms per pair on a modern CPU, so 2 seconds is very loose.
 */
export const NLI_TIMEOUT_MS = 2_000;

// ───────────────────────────────────────────────────────────────────────────
// Aggregation thresholds
// ───────────────────────────────────────────────────────────────────────────

/**
 * Any single critic reporting severity >= FAIL_THRESHOLD sets the
 * consensus to 'fail'. Default 3 on a 0–5 scale means "significant issue".
 * [ADAPT] Raise to 4 if you only want hard failures to fail the consensus.
 */
export const FAIL_SEVERITY_THRESHOLD = 3;

/**
 * Any single critic reporting severity >= WARN_THRESHOLD sets the
 * consensus to 'warn' (unless anything reaches the fail threshold).
 *
 * 2026-04-18 bumped 1 → 2. At 1, severity-1 "minor nitpick" annotations
 * were flipping consensus to warn on otherwise-pass answers, producing
 * the `hedge-valid` ~warn miss on the easy sweep. At 2, only genuine
 * concerns (stylistic-plus or real-issue-minor) fire warn. Side effect:
 * `code-null-bug` also stops flipping to warn — but both critics were
 * already calling that one severity-1 (under-calling the real bug), so
 * the case was already a ~warn miss against expected fail.
 */
export const WARN_SEVERITY_THRESHOLD = 2;

/**
 * If this many critics are unavailable (timeout, connection refused, model
 * not loaded), the consensus becomes 'error' and a note is added to the
 * output.
 *
 * Sized to match ALL_CRITICS.length in critic-configs.ts:
 *   3 critics -> MAX_UNAVAILABLE_CRITICS = 2 (1 critic can still vote)
 *   2 critics -> MAX_UNAVAILABLE_CRITICS = 1 (requires at least 1 critic)
 * Currently 2 critics (Phi-4-mini + Nemotron Mini), so 1.
 */
export const MAX_UNAVAILABLE_CRITICS = 1;

/**
 * [ADAPT] Opt-in weighted-vote override (default OFF).
 *
 * When ON, a higher-weight critic's `pass` can outvote a lower-weight
 * critic's `fail`. The 2026-04-18 sweep found this was a wash on the
 * audit corpus: the lone-2B-fail downgrade flipped `code-clean`
 * (false positive) from MISS to warn (better) but also flipped
 * `code-subtle-bug` (real bug only the 2B caught) from OK to warn
 * (worse). 50/50 on which critic is right when they disagree at the
 * extremes, so kept default OFF.
 *
 * Set AGGREGATOR_WEIGHTED_VOTE=1 in env to opt in for A/B work.
 *
 * 2026-05-12: moved from a direct process.env read inside aggregator.ts
 * to honour the "all knobs in config.ts" contract.
 */
export const AGGREGATOR_WEIGHTED_VOTE =
  process.env.AGGREGATOR_WEIGHTED_VOTE === "1";

/**
 * Truncation budgets used by the human-readable Markdown renderer.
 *
 * Table cells get tight budgets so the table stays scannable; bullets
 * in the Findings list get larger budgets so the text isn't clipped
 * mid-thought.
 *
 * 2026-05-12 (F3): these were inline magic numbers scattered through
 * aggregator.ts. Promoted here so a renderer-wide widening is one
 * edit instead of seven.
 */
export const RENDER_CELL_TOPIC_CHARS = 120;     // critic table top-concern cell
export const RENDER_CELL_CONCERN_CHARS = 140;   // dispute table concern cell
export const RENDER_FINDING_CONCERN_CHARS = 800; // Findings bullet, critic concern
export const RENDER_FINDING_EXTRA_CHARS = 600;  // Findings bullet, "also:" / fix
export const RENDER_FINDING_NLI_CLAIM_CHARS = 1000; // NLI claim text in Findings
export const RENDER_FINDING_RECOMPUTE_CHARS = 400;  // recompute mismatch expr

/**
 * [ADAPT] Echo the verified answer at the top of the /verify block. Default
 * OFF: the block shows only the critics table and the bold conclusion (the
 * worker shows its own answer per the FLOW / system prompt). Set
 * VERITY_SUMMARY_ECHO_ANSWER=1 to restate the answer inside the block, useful
 * for a worker that calls verify_answer without first emitting a visible answer.
 */
export const SUMMARY_ECHO_ANSWER =
  process.env.VERITY_SUMMARY_ECHO_ANSWER === "1";

/**
 * Divergence threshold (0–1) at or above which the consistency check alone
 * flips the consensus to 'fail'. 0.5 means "more than half the answer's
 * claims were contradicted or unsupported across alternate samples".
 * [ADAPT] Lower to be stricter; raise to require strong disagreement.
 */
export const CONSISTENCY_FAIL_THRESHOLD = 0.5;

/**
 * Divergence threshold (0–1) at or above which the consistency check alone
 * contributes a 'warn'. Replaces the old "any non-zero divergence warns"
 * behaviour, which fired on trivial single-claim noise.
 * [ADAPT] Lower to 0.05 for stricter surfacing of minor inconsistencies.
 */
export const CONSISTENCY_WARN_THRESHOLD = 0.15;

// ───────────────────────────────────────────────────────────────────────────
// NLI settings
// ───────────────────────────────────────────────────────────────────────────

/**
 * [ADAPT] Which NLI implementation to use.
 *   "deberta" — cross-encoder DeBERTa-v3 via @huggingface/transformers on CPU.
 *     Default; fast (~150ms/pair) but only flags explicit contradictions.
 *   "llm"     — LLM-based claim checker using one of the Ollama critic models
 *     (Granite 3.2 2B by default). Slower (~300-600ms/claim on AMD Vulkan)
 *     but can reason about subtle inconsistencies the cross-encoder misses.
 *   "off"     — skip NLI entirely. Use if latency matters more than signal.
 *
 * 2026-04-18 added to support A/B/C option comparison described in
 * `experiments/PAPER-PLAN-2026-04-18.md`.
 */
export const NLI_IMPL: "deberta" | "llm" | "off" =
  (process.env.NLI_IMPL as "deberta" | "llm" | "off") ?? "deberta";

/**
 * [ADAPT] If true, only run NLI when prior_context is non-empty (entailment
 * mode). Skip pairwise intra-answer mode, which produced zero signal across
 * the designed-to-fail audit corpus.
 *
 * 2026-04-18 Option A in the NLI comparison.
 * 2026-05-09: default flipped from false → true. The pairwise intra-answer
 * mode is documented above as zero-signal; defaulting it on cost up to ~190
 * NLI calls per request for no gain. Set NLI_REQUIRE_CONTEXT=0 to opt back
 * into the old behaviour for A/B experiments.
 */
export const NLI_REQUIRE_CONTEXT =
  process.env.NLI_REQUIRE_CONTEXT === undefined
    ? true
    : !(
        process.env.NLI_REQUIRE_CONTEXT === "0" ||
        process.env.NLI_REQUIRE_CONTEXT.toLowerCase() === "false"
      );

/**
 * [ADAPT] Ollama model tag used by the LLM-based claim-checker when
 * NLI_IMPL=llm. Defaults to the small-and-fast critic already loaded on AMD
 * (Critic B, Ministral 3B).
 */
export const NLI_LLM_MODEL =
  process.env.NLI_LLM_MODEL ?? "ministral-3:3b";

/**
 * [ADAPT] HuggingFace model id for the NLI classifier.
 *
 * History on this machine (2026-04-18):
 *   "Xenova/deberta-v3-large-mnli" (original)
 *     -> Generic MNLI-trained; on the NLI audit corpus it flagged 0
 *        contradictions / 0 unsupported claims across 6 cases designed
 *        to exercise it. Classifier returned 'neutral' on subtle logical
 *        oppositions ("Paris is in France. Paris is in Germany.").
 *   "Xenova/nli-deberta-v3-large" (TRYING 2026-04-18)
 *     -> Cross-encoder NLI model from the sentence-transformers project,
 *        purpose-built for pairwise entailment / contradiction /
 *        neutral classification. Expected to be more decisive than the
 *        MNLI variant on genuine contradictions.
 *
 * Both are available as ONNX in the Xenova namespace for
 * @huggingface/transformers. ~1 GB download on first use.
 */
export const NLI_MODEL_ID =
  process.env.NLI_MODEL ?? "Xenova/nli-deberta-v3-large";

/**
 * [ADAPT] Device for NLI inference.
 *   "cpu" — always works, ~150 ms per claim. Recommended default on this
 *     machine. Verified 2026-04-17: device="webgpu" in Node.js transformers
 *     picks the system default high-perf adapter (= NVIDIA 5070 Ti), NOT
 *     the Intel UHD 770 iGPU as the design doc hoped. Since the 5070 Ti
 *     is already saturated by the worker + Critic A, running NLI there
 *     steals LLM cycles. CPU is cleaner.
 *   "webgpu" — lands on whichever GPU the WebGPU runtime chooses.
 *     No way to pin to the iGPU from transformers.js; would need OpenVINO
 *     or DirectML, neither of which is a transformers.js backend.
 *   "wasm" — CPU via WebAssembly; slower than "cpu", keep as fallback.
 */
export const NLI_DEVICE: "cpu" | "webgpu" | "wasm" = "cpu";

/**
 * Maximum claims to extract and check per answer. Beyond this, later
 * claims are ignored to bound latency. [ADAPT] Raise for research-heavy
 * answers with many factual assertions.
 */
export const NLI_MAX_CLAIMS = 20;

/**
 * Minimum NLI contradiction probability to count as "contradicted".
 * [ADAPT] Lower this (e.g. 0.5) to be more paranoid about hallucinations.
 */
export const NLI_CONTRADICTION_THRESHOLD = 0.65;

// ───────────────────────────────────────────────────────────────────────────
// Deep / deeper modes (Signal 1: consistency check)
// ───────────────────────────────────────────────────────────────────────────

/**
 * [ADAPT] Number of alternate worker samples to draw in deep mode.
 *   2 samples → ~16 s extra, weak but cheap signal
 *   3 samples → ~24 s extra, the classic SelfCheckGPT setting
 * 2 keeps /verifydeep under ~30 s total.
 */
export const CONSISTENCY_SAMPLES_DEEP = 2;

/**
 * [ADAPT] Number of alternate worker samples in deeper mode.
 *   5–10 samples is the range used in semantic-entropy literature.
 * 5 keeps /verifydeeper under ~60 s total.
 */
export const CONSISTENCY_SAMPLES_DEEPER = 5;

/**
 * [ADAPT] Sampling temperature for the alternate samples. Higher = more
 * diverse alternates (better for catching hallucinations) but slower
 * convergence. 0.7 matches SelfCheckGPT's default.
 */
export const CONSISTENCY_TEMPERATURE = 0.7;

// ───────────────────────────────────────────────────────────────────────────
// Deep / deeper modes (Signal 2: perplexity / logprobs)
// ───────────────────────────────────────────────────────────────────────────

/**
 * [ADAPT] Token logprob threshold below which we flag the token as
 * "low confidence". -3.0 corresponds to roughly 5% probability.
 *   -1.5 (≈22%) → noisy, lots of flags
 *   -3.0 (≈5%)  → balanced, default
 *   -4.5 (≈1%)  → only egregiously uncertain tokens
 */
export const PERPLEXITY_LOW_CONFIDENCE_LOGPROB = -3.0;

/**
 * [ADAPT] Maximum number of low-confidence spans surfaced in the result.
 * Prevents the output from blowing up on a paragraph of guesses.
 */
export const PERPLEXITY_MAX_FLAGGED_SPANS = 8;

/**
 * [ADAPT] Per-token alternatives requested from /v1/responses. We only need
 * the chosen token's logprob, so 1 keeps the payload small. Raise if you
 * want to surface the runner-up tokens in the confidence note.
 */
export const PERPLEXITY_RESPONSES_TOP_LOGPROBS = Number(
  process.env.VERITY_RESPONSES_TOP_LOGPROBS ?? 1
);

/**
 * [ADAPT] Max tokens for the /v1/responses regeneration used to obtain
 * logprobs. Mirrors the worker's typical answer length.
 */
export const PERPLEXITY_REGEN_MAX_TOKENS = Number(
  process.env.VERITY_REGEN_MAX_TOKENS ?? 800
);

// ───────────────────────────────────────────────────────────────────────────
// Generation-confidence gate (logprob-driven escalation)
// ───────────────────────────────────────────────────────────────────────────
//
// Maps a generation's per-token logprobs to a confidence band, and each band
// to a recommended verify depth:
//   ok       → (nothing)
//   mild     → /verify       (standard)
//   low      → /verifydeep
//   very_low → /verifydeeper
//
// A band fires if ANY of its three axes trips; the worst matched band wins.
//   - local axis : the single weakest token's logprob (catches one bad
//     name/number/date in an otherwise fluent answer)
//   - global axis: whole-answer perplexity
//   - density axis: fraction of tokens at/below PERPLEXITY_LOW_CONFIDENCE_LOGPROB
//
// [ADAPT] These defaults are first-pass and uncalibrated against a labelled
// corpus — tune once you have ground-truth confident/unconfident answers.

/** Master switch for the confidence gate. Set VERITY_CONFIDENCE_GATE=0 off. */
export const CONFIDENCE_GATE_ENABLED =
  (process.env.VERITY_CONFIDENCE_GATE ?? "1") !== "0";

// Local axis: weakest single-token logprob thresholds (more negative = worse).
// -3.0 ≈ 5% prob, -4.5 ≈ 1.1%, -6.0 ≈ 0.25%.
export const CONFIDENCE_MILD_MIN_LOGPROB = Number(
  process.env.VERITY_CONF_MILD_MIN_LOGPROB ?? -3.0
);
export const CONFIDENCE_LOW_MIN_LOGPROB = Number(
  process.env.VERITY_CONF_LOW_MIN_LOGPROB ?? -4.5
);
export const CONFIDENCE_VERYLOW_MIN_LOGPROB = Number(
  process.env.VERITY_CONF_VERYLOW_MIN_LOGPROB ?? -6.0
);

// Global axis: whole-answer perplexity thresholds (higher = worse).
export const CONFIDENCE_MILD_PERPLEXITY = Number(
  process.env.VERITY_CONF_MILD_PERPLEXITY ?? 2.2
);
export const CONFIDENCE_LOW_PERPLEXITY = Number(
  process.env.VERITY_CONF_LOW_PERPLEXITY ?? 4.0
);
export const CONFIDENCE_VERYLOW_PERPLEXITY = Number(
  process.env.VERITY_CONF_VERYLOW_PERPLEXITY ?? 8.0
);

// Density axis: fraction of low-confidence tokens (higher = worse).
export const CONFIDENCE_MILD_RATIO = Number(
  process.env.VERITY_CONF_MILD_RATIO ?? 0.05
);
export const CONFIDENCE_LOW_RATIO = Number(
  process.env.VERITY_CONF_LOW_RATIO ?? 0.12
);
export const CONFIDENCE_VERYLOW_RATIO = Number(
  process.env.VERITY_CONF_VERYLOW_RATIO ?? 0.25
);

// ───────────────────────────────────────────────────────────────────────────
// Logging
// ───────────────────────────────────────────────────────────────────────────

/**
 * [ADAPT] Enable verbose per-request logging. Useful while tuning prompts
 * and tracking down why a specific critic is misbehaving. Disable for
 * production to keep terminal output clean.
 */
export const VERBOSE_LOGGING = process.env.VERIFIER_VERBOSE === "1";

// ───────────────────────────────────────────────────────────────────────────
// HTTP transport lifecycle
// ───────────────────────────────────────────────────────────────────────────

/**
 * [ADAPT] Idle timeout (ms) for MCP session transports. Clients sometimes
 * disconnect without triggering onclose (network blip, LM Studio restart);
 * without a sweep, the session map leaks. Default 10 minutes.
 */
export const SESSION_IDLE_TIMEOUT_MS = 10 * 60 * 1000;

/**
 * [ADAPT] How often (ms) to sweep stale sessions. Short enough to reclaim
 * memory promptly, long enough to be background-ish.
 */
export const SESSION_SWEEP_INTERVAL_MS = 60 * 1000;

/**
 * [ADAPT] Hard wall-clock ceiling (ms) per verification request. If all
 * critics + signals together exceed this, the pipeline aborts and returns
 * whatever partial results are ready.
 *
 * Sized generously for 'deeper' mode on this hardware (standard critics
 * ~1 s, but Qwen re-sampling adds ~8 s per sample × up to 5 samples plus
 * regenerate-with-logprobs fallback for perplexity).
 */
export const PIPELINE_TIMEOUT_MS = 180_000;

// ───────────────────────────────────────────────────────────────────────────
// /second — parallel cross-family second-opinion consult
// ───────────────────────────────────────────────────────────────────────────

/**
 * [ADAPT ★] Model used for the /second parallel consult. Must be available
 * on the Ollama server (`ollama list`).
 *
 * Default `granite4.1:8b` is the strongest local critic (it is also Critic A),
 * already resident on AMD, so there is no cold-load hit on first use, and it is
 * a different family from the Qwen worker. The /second model generates a full
 * answer, not just a critique, so the larger of the two new critics is the
 * better pick (Ministral 3B is the lightweight one). For more contrast, override
 * to another provider, e.g. `gemma4:e4b` (Google) or `phi4-mini:3.8b` (Microsoft).
 */
export const SECOND_OPINION_MODEL =
  process.env.SECOND_OPINION_MODEL ?? "granite4.1:8b";

/**
 * [ADAPT] Upper bound on second-opinion length. 400 is enough for a focused
 * paragraph and keeps Ollama responsive.
 */
export const SECOND_OPINION_MAX_TOKENS = Number(
  process.env.SECOND_OPINION_MAX_TOKENS ?? 400
);

/**
 * [ADAPT] Sampling temperature for the second-opinion model. Non-zero so the
 * second opinion exposes genuinely different reasoning, not just a replica
 * of the top-1 completion.
 */
export const SECOND_OPINION_TEMPERATURE = Number(
  process.env.SECOND_OPINION_TEMPERATURE ?? 0.3
);

/**
 * [ADAPT] Timeout (ms) on a single /second call. Generous because the
 * selected second model may be cold on first invocation (2–5 s for
 * granite4.1:8b, 1–2 s for smaller candidates).
 */
export const SECOND_OPINION_TIMEOUT_MS = Number(
  process.env.SECOND_OPINION_TIMEOUT_MS ?? 30_000
);

// ───────────────────────────────────────────────────────────────────────────
// /second — dual-GPU consult + analysis pass (2026-04-21)
// ───────────────────────────────────────────────────────────────────────────
//
// Phase A+B resurrected from the 2026-04-20 archived dual-dispatch:
// both GPUs run the question independently in parallel (Ollama/AMD +
// LM Studio/NVIDIA). Phase C (new) adds an analysis pass: once both
// answers come back, NVIDIA is asked to compare them and emit a
// structured {agreements, disputes, table_html} object. In auto mode the
// analysis call also synthesises a final_answer from both.
//
// Kill-switch:
//   CONSULT_DUAL=0 reverts to the pre-2026-04-20 single-Ollama path.
//   The legacy `second_opinion` / `model` / `latency_ms` / `diff_summary`
//   wire fields are preserved either way for back-compat with
//   test-second-opinion.ps1.

/**
 * [ADAPT] Enable dual-endpoint /second + analysis pass. When on and
 * no explicit `input.model` is supplied, fires Ollama (AMD) AND LM
 * Studio (NVIDIA) in parallel, then runs a third analysis call on
 * LM Studio to produce structured {agreements, disputes, table_html}.
 * Set `CONSULT_DUAL=0` to revert to the single-Ollama pre-2026-04-20
 * path (preserves the legacy wire shape; no `dual_opinion`/`analysis`).
 */
export const SECOND_OPINION_DUAL_ENABLED =
  (process.env.CONSULT_DUAL ?? "1") !== "0";

/**
 * Primary-GPU endpoint used by the dual /second companion call and the
 * analysis pass. Defaults to LM_STUDIO_URL so the idle NVIDIA slice is
 * put to work while Ollama serves the AMD side.
 */
export const SECOND_OPINION_PRIMARY_ENDPOINT =
  process.env.SECOND_OPINION_PRIMARY_ENDPOINT ?? LM_STUDIO_URL;

/**
 * Model tag served on SECOND_OPINION_PRIMARY_ENDPOINT for the NVIDIA leg.
 * Default is the worker model — LM Studio normally has that loaded, so
 * no cold-load hit.
 */
export const SECOND_OPINION_PRIMARY_MODEL =
  process.env.SECOND_OPINION_PRIMARY_MODEL ?? WORKER_MODEL_NAME;

/**
 * API key for the primary endpoint. LM Studio ignores the value but the
 * OpenAI SDK requires a non-empty string.
 */
export const SECOND_OPINION_PRIMARY_API_KEY =
  process.env.SECOND_OPINION_PRIMARY_API_KEY ?? "lm-studio";

/**
 * [ADAPT] Concurrent-path timeout (ms). Tighter than SECOND_OPINION_TIMEOUT_MS
 * (30 s solo) because dual firing both GPUs simultaneously is thermally
 * more aggressive — 20 s keeps each leg from stretching into a sustained
 * load window on the AMD card.
 */
export const SECOND_OPINION_CONCURRENT_TIMEOUT_MS = Number(
  process.env.SECOND_OPINION_CONCURRENT_TIMEOUT_MS ?? 20_000
);

/**
 * [ADAPT] Model used for the Phase C analysis pass (comparing the two
 * answers, emitting agreements/disputes/table_html). Default is the
 * worker model — same loaded weights as the NVIDIA answerer leg, so
 * no extra VRAM pressure.
 */
export const SECOND_OPINION_ANALYSIS_MODEL =
  process.env.SECOND_OPINION_ANALYSIS_MODEL ?? WORKER_MODEL_NAME;

/**
 * [ADAPT] Timeout (ms) for the analysis call. Shorter than the answerer
 * timeout — analysis is a constrained JSON-output task with both inputs
 * already in hand, so it should not need more than 15 s.
 */
export const SECOND_OPINION_ANALYSIS_TIMEOUT_MS = Number(
  process.env.SECOND_OPINION_ANALYSIS_TIMEOUT_MS ?? 15_000
);

/**
 * [ADAPT] Max tokens for the analysis JSON. Enough for ~5 disputes plus a
 * short synthesised final_answer in auto mode.
 *
 * 2026-04-21 bumped 800 -> 2000. Qwen3.5-9B emits a large
 * <think>...</think> reasoning trace (typically 600-1200 tokens) BEFORE
 * the final JSON. At 800, completion routinely truncated mid-<think>
 * with `finish_reason: length` — leaving an unclosed `<think>` tag that
 * the reasoning-strip regex cannot remove, and letting draft JSON inside
 * the thinking block leak into extractAnalysisJson → parse failure →
 * analysis.unavailable = true → user sees the fallback message, no table.
 * 2000 gives the model room to close </think> and emit real JSON.
 */
export const SECOND_OPINION_ANALYSIS_MAX_TOKENS = Number(
  process.env.SECOND_OPINION_ANALYSIS_MAX_TOKENS ?? 2000
);

// ───────────────────────────────────────────────────────────────────────────
// Confidence proxy  (transparent OpenAI-compatible front-door)
// ───────────────────────────────────────────────────────────────────────────
//
// The confidence proxy (src/proxy/server.ts) is a small HTTP server that an
// external OpenAI-compatible chat client (Open WebUI, Jan, LibreChat,
// AnythingLLM, ...) points at INSTEAD of LM Studio's http://localhost:1234/v1.
// It intercepts POST /v1/chat/completions, routes the GENERATION through the
// upstream's /v1/responses endpoint (the only logprobs-bearing path on LM
// Studio), scores the returned tokens with the shared confidence classifier,
// and appends the low-confidence note to the answer when the band is not "ok".
// Every other path is forwarded transparently.
//
// Why a separate process and port: the built-in chat UIs (the LM Studio app,
// `ollama run`) are sealed and cannot be intercepted. The proxy is the ONLY
// way to enforce the confidence gate on EVERY answer, and it only works for
// clients that talk to the API port. See docs/confidence-proxy.md.
//
// [ADAPT] The upstream MUST expose /v1/responses with logprobs. LM Studio
// (0.3.x+) does; Ollama currently does NOT, so this proxy targets LM
// Studio-backed setups for now.

/**
 * [ADAPT] Port the confidence proxy listens on. Defaults to 1235 to sit one
 * above LM Studio's standard 1234, so the two run side by side: clients point
 * at the proxy (1235), the proxy forwards to LM Studio (1234). Point your
 * external client's "OpenAI-compatible API base URL" at
 * http://localhost:1235/v1.
 */
export const PROXY_PORT = Number(process.env.VERITY_PROXY_PORT ?? 1235);

/**
 * [ADAPT] Network interface the proxy binds to. Defaults to localhost
 * (loopback only), matching SERVER_HOST's reasoning: the proxy has no
 * authentication and forwards an upstream key, so it should not be reachable
 * off-host by default. Set VERITY_PROXY_HOST=0.0.0.0 only if you deliberately
 * want a LAN client (e.g. a phone running a chat app) to reach it and you
 * understand the threat model.
 */
export const PROXY_HOST = process.env.VERITY_PROXY_HOST ?? "127.0.0.1";

/**
 * [ADAPT] Upstream OpenAI-compatible base URL the proxy forwards to. Defaults
 * to LM_STUDIO_URL (http://localhost:1234/v1). This is the backend that must
 * expose /v1/responses with logprobs. To put the proxy in front of a
 * different LM Studio host, or a llama-server / vLLM build that implements the
 * Responses API, change this.
 */
export const PROXY_UPSTREAM_URL =
  process.env.VERITY_PROXY_UPSTREAM_URL ?? LM_STUDIO_URL;

/**
 * [ADAPT] Per-token alternatives the proxy requests from /v1/responses. The
 * confidence classifier only reads the chosen token's logprob, so 1 keeps the
 * payload small. Mirrors PERPLEXITY_RESPONSES_TOP_LOGPROBS; kept separate so
 * the proxy can be tuned without touching the /verify perplexity signal.
 */
export const PROXY_RESPONSES_TOP_LOGPROBS = Number(
  process.env.VERITY_PROXY_RESPONSES_TOP_LOGPROBS ?? 1
);

/**
 * [ADAPT] Default cap on generated tokens when an incoming chat-completions
 * request does not specify max_tokens. A finite default keeps a client that
 * omits the field from letting the model run away. Generous enough for a long
 * prose answer.
 */
export const PROXY_DEFAULT_MAX_OUTPUT_TOKENS = Number(
  process.env.VERITY_PROXY_DEFAULT_MAX_OUTPUT_TOKENS ?? 2048
);

/**
 * [ADAPT] Wall-clock timeout (ms) for a single upstream /v1/responses
 * generation initiated by the proxy. Sized for a long answer on a local
 * model; raise for very large max_tokens or a slow/cold backend.
 */
export const PROXY_UPSTREAM_TIMEOUT_MS = Number(
  process.env.VERITY_PROXY_UPSTREAM_TIMEOUT_MS ?? 120_000
);

/**
 * [ADAPT] Maximum request body (bytes) the proxy accepts. A chat-completions
 * body carries the whole conversation, so this is larger than the MCP
 * server's MAX_REQUEST_BYTES. Still a DoS guard against an unbounded body.
 */
export const PROXY_MAX_REQUEST_BYTES = Number(
  process.env.VERITY_PROXY_MAX_REQUEST_BYTES ?? 16 * 1024 * 1024
);
