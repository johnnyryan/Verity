/**
 * Shared TypeScript interfaces for the verity.
 */

export type TaskType = "code" | "prose" | "reasoning" | "research" | "auto";

export type ContextMode = "minimal" | "with_context" | "full";

export type Verdict = "pass" | "warn" | "fail" | "error";

/**
 * Verification depth.
 *   "standard" → 3 critics + NLI                          (~11 s)
 *   "deep"     → standard + consistency (2 samples) + perplexity rescore (~30 s)
 *   "deeper"   → standard + consistency (5 samples) + perplexity (with regeneration fallback)  (~50 s)
 */
export type VerifyMode = "standard" | "deep" | "deeper";

/**
 * A single critic's structured output.
 * Critics are instructed to return JSON matching this shape.
 */
export interface CriticResult {
  /** Stable identifier for the critic (e.g. "critic_a"). */
  id: string;
  /** Human-readable display name. */
  display_name: string;
  /** Model family (for diversity reporting). */
  family: string;
  /** Overall verdict from this critic. */
  verdict: Verdict;
  /** Severity on a 0–5 scale. 0 = pass, 5 = critical failure. */
  severity: number;
  /** Specific concerns found by this critic. */
  concerns: string[];
  /** Suggested fixes for the concerns, in matching order where possible. */
  suggested_fixes: string[];
  /**
   * Optional verbatim quote from the answer that triggered the critic's
   * disagreement. Borrowed from HalluGuard's SRM evidence span. Parsed
   * from the critic's `disputed_span` field (snake_case on the wire).
   * Display only — does not influence verdict or consensus. Absent when
   * the critic agreed, omitted the field, or emitted a non-substring
   * paraphrase that the parser could not validate.
   */
  disputedSpan?: string;
  /** Any notes from the pipeline (e.g. "input truncated"). */
  notes: string[];
  /** True if the critic was unreachable or errored. */
  unavailable?: boolean;
  /** Error message if unavailable. */
  error?: string;
  /** Per-critic wall-clock latency in ms. */
  latency_ms?: number;
  /** Aggregator vote weight (from CriticConfig.weight, default 1). */
  weight?: number;
}

/**
 * Result of the NLI entailment check.
 */
export interface NliResult {
  /** Whether NLI ran at all. */
  ran: boolean;
  /** Number of claims extracted and classified. */
  claims_checked: number;
  /** Claims that NLI classified as contradicting the premise. */
  contradictions: Array<{
    claim: string;
    premise_snippet?: string;
    confidence: number;
  }>;
  /** Claims that had no supporting premise (only meaningful with prior_context). */
  unsupported: Array<{
    claim: string;
  }>;
  /** Notes about how the check was run or why it was skipped. */
  notes: string;
}

/**
 * Result of the SelfCheckGPT-style consistency check.
 * Only present when mode is "deep" or "deeper".
 */
export interface ConsistencyResult {
  ran: boolean;
  /** Number of usable alternate samples generated. */
  samples_generated: number;
  /** Number of factual claims extracted from the original answer. */
  claims_checked: number;
  /** Original-answer claims contradicted by at least one alternate sample. */
  contradicted: Array<{
    claim: string;
    contradicting_sample_index: number;
    confidence: number;
  }>;
  /** Original-answer claims neither entailed nor contradicted by any alternate. */
  unsupported: Array<{ claim: string }>;
  /** (contradicted + unsupported) / total claims, in [0,1]. Higher = less consistent. */
  divergence_score: number;
  latency_ms: number;
  notes: string;
  /**
   * Semantic entropy in nats (Farquhar et al., Nature 2024). High = the
   * model produced surface-different answers with the same underlying
   * uncertainty (confabulation). Computed over the same sample stream
   * the consistency check uses; clustering is by bidirectional NLI
   * entailment. Optional because it is only available in deep modes
   * and only when NLI is loaded; the pipeline attaches it post-hoc
   * (consistency.ts itself does not compute it).
   *
   * ADVISORY: surfaced in the rendered Markdown block alongside the
   * perplexity nudge; never flips the verdict.
   */
  semantic_entropy?: number;
  /** Distinct-meaning cluster count over the sample stream. */
  semantic_cluster_count?: number;
}

/**
 * Single deterministic recompute verification — one matched
 * arithmetic / enumeration / leap-year / unit claim.
 */
export interface RecomputeVerification {
  kind: "arithmetic" | "enumeration" | "unit" | "leap-year" | "linear-equation";
  expr_text: string;
  claimed: string;
  computed: string;
  matches: boolean;
  /** Always 1.0 — the pass is deterministic, no model uncertainty. */
  confidence: number;
}

/**
 * Result of the deterministic recompute pass (src/signals/recompute.ts).
 * Runs in parallel with NLI in standard mode. Mismatches trigger a hard
 * fail in the aggregator; matches can suppress NLI contradiction flags
 * on the same expression.
 */
export interface RecomputeResult {
  ran: boolean;
  /** Number of expressions that parsed + computed successfully. */
  expressions_found: number;
  /** Every successfully-computed claim, matching or not. */
  verifications: RecomputeVerification[];
  /** Subset of verifications where computed != claimed. */
  mismatches: Array<{
    kind: string;
    expr_text: string;
    claimed: string;
    computed: string;
  }>;
  notes: string;
  latency_ms: number;
}

/**
 * Confidence band derived from a generation's per-token logprobs.
 *   ok       → nothing to flag
 *   mild     → recommend a standard /verify
 *   low      → recommend /verifydeep
 *   very_low → recommend /verifydeeper
 */
export type ConfidenceBand = "ok" | "mild" | "low" | "very_low";

/**
 * Raw confidence statistics computed from a token-logprob stream. Both a
 * global axis (mean/perplexity, low-confidence token ratio) and a local
 * axis (the single weakest token) so a lone hallucinated name/number is
 * caught even when the answer is otherwise fluent.
 */
export interface ConfidenceMetrics {
  tokens_scored: number;
  /** Mean per-token logprob. */
  mean_logprob: number;
  /** exp(-mean_logprob). Higher = more uncertain. */
  perplexity: number;
  /** Weakest single-token logprob in the stream (global minimum). */
  min_logprob: number;
  /** Text of the weakest token. */
  min_logprob_token: string;
  /** Count of tokens at or below PERPLEXITY_LOW_CONFIDENCE_LOGPROB. */
  low_confidence_tokens: number;
  /** low_confidence_tokens / tokens_scored, in [0,1]. */
  low_confidence_ratio: number;
}

/**
 * The output of the confidence classifier (src/signals/confidence.ts).
 * Used both by the perplexity signal and by the generation-confidence gate.
 */
export interface ConfidenceAssessment {
  band: ConfidenceBand;
  /** Verify depth to escalate to, or null when the answer is confident. */
  recommended_mode: VerifyMode | null;
  /** One-line, human-readable reason the band fired. */
  reason: string;
  metrics: ConfidenceMetrics;
}

/**
 * Result of the logprob-based perplexity / token-entropy check.
 * Only present when mode is "deep" or "deeper".
 */
export interface PerplexityResult {
  ran: boolean;
  /**
   * Which method actually produced the result:
   *   forward_pass_rescore — fast, ~1–2 s, scores the ACTUAL answer; needs an
   *     echo-capable /v1/completions (a llama-server side-car — LM Studio
   *     returns null here, see design doc).
   *   responses_logprobs — regenerates via /v1/responses and scores the fresh
   *     answer. The working logprobs path on LM Studio (0.3.x+, both GGUF and
   *     MLX). Used in "deeper" mode. Cost: ~8 s.
   *   regenerate_with_logprobs — legacy /v1/chat/completions fallback; LM
   *     Studio returns null logprobs here, kept only for non-LM-Studio hosts.
   */
  method:
    | "forward_pass_rescore"
    | "responses_logprobs"
    | "regenerate_with_logprobs";
  tokens_scored: number;
  /** Mean per-token logprob across the answer. */
  mean_logprob: number;
  /** exp(-mean_logprob). Higher = more uncertain. */
  perplexity: number;
  /** Spans of consecutive tokens whose logprob fell below the threshold. */
  low_confidence_spans: Array<{
    text: string;
    min_logprob: number;
  }>;
  /**
   * Confidence band + escalation recommendation derived from the per-token
   * logprobs. Present only when the check ran with usable logprobs.
   */
  confidence?: ConfidenceAssessment;
  latency_ms: number;
  notes: string;
}

/**
 * A single diagnostic disagreement between the two /verify critics.
 *
 * Disputes are surfaced in VerifyOutput.disputes for transparency only —
 * they do NOT influence the consensus verdict. See aggregator.computeDisputes.
 *   - "verdict-mismatch": the two critics returned different verdicts
 *     (e.g. one pass, one warn, or one pass / one fail). severity="hard"
 *     iff the pair straddles the ship/halt line (pass vs fail).
 *   - "concern-only-in-a" / "concern-only-in-b": one critic raised a
 *     concern that the other did not raise a fuzzy-matched equivalent of.
 */
export interface Disagreement {
  kind: "verdict-mismatch" | "concern-only-in-a" | "concern-only-in-b";
  critic_a_id: string;
  critic_b_id: string;
  critic_a: { verdict: Verdict; severity: number; concern?: string };
  critic_b: { verdict: Verdict; severity: number; concern?: string };
  severity: "hard" | "soft";
}

/**
 * Inputs to the pipeline from the MCP tool call.
 */
export interface VerifyInput {
  question: string;
  answer: string;
  task_type?: TaskType;
  context_mode?: ContextMode;
  prior_context?: string;
  use_nli?: boolean;
  mode?: VerifyMode;
}

/**
 * Final output returned by the pipeline.
 */
export interface VerifyOutput {
  /**
   * Keyed by critic id (the `id` field of each CriticConfig in
   * critic-configs.ts). The set of keys at runtime matches ALL_CRITICS;
   * swap or add critics there and the output shape adapts automatically.
   *
   * 2026-05-12: was a literal object type listing each critic id
   * statically. That forced three files to be touched in lockstep
   * (critic-configs.ts, types.ts, pipeline.ts) every time the panel
   * changed. Now a generic Record so adding a critic is a one-file
   * change.
   */
  critics: Record<string, CriticResult>;
  /**
   * Diagnostic-only record of disagreements between the two critics.
   * Always present; empty array when the critics agree (or when too few
   * critics are available to compare). Never flips the consensus verdict.
   */
  disputes: Disagreement[];
  nli_check: NliResult;
  /** Deterministic arithmetic / enumeration / leap-year / unit recompute. */
  recompute?: RecomputeResult;
  /** Present only when mode is "deep" or "deeper". */
  consistency_check?: ConsistencyResult;
  /** Present only when mode is "deep" or "deeper". */
  perplexity?: PerplexityResult;
  consensus: Verdict;
  summary: string;
  /**
   * 2026-04-21 additive: a pre-rendered human-readable Markdown block the
   * worker can paste verbatim into chat. Contains the consensus verdict,
   * a critic-by-critic table keyed on display_name (so the user sees
   * the running model's display label rather than the stable wire id
   * "critic_a"), recompute / NLI / disputes tallies, and a collapsed
   * <details> wrapper around the raw JSON payload for developers who
   * still want it.
   *
   * The worker tool description (src/index.ts) instructs the LLM to render
   * this field AS-IS. The rest of VerifyOutput is unchanged — summary_md
   * is strictly additive and is rendered from the same payload the worker
   * already has in hand.
   */
  summary_md: string;
  latency_ms: number;
  meta: {
    mode: VerifyMode;
    task_type: TaskType;
    context_mode: ContextMode;
    granite_8b_input_truncated: boolean;
    critics_unavailable: string[];
  };
}
