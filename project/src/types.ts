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
  /** Stable identifier for the critic (e.g. "granite_3_2_8b"). */
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
 * Result of the logprob-based perplexity / token-entropy check.
 * Only present when mode is "deep" or "deeper".
 */
export interface PerplexityResult {
  ran: boolean;
  /**
   * Which method actually produced the result:
   *   forward_pass_rescore — fast, ~1–2 s, requires LM Studio /v1/completions support
   *   regenerate_with_logprobs — fallback, ~8 s, only used in "deeper" mode
   */
  method: "forward_pass_rescore" | "regenerate_with_logprobs";
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
  critics: {
    // [ADAPT] These keys track the ids in critic-configs.ts. Swap a critic
    // by renaming the key here + updating pipeline.ts findCritic() calls.
    granite_3_2_8b: CriticResult;
    granite_3_2_2b: CriticResult;
    // llama32_3b: CriticResult;  // re-add if Critic C is re-enabled
  };
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
   * "IBM Granite 3.2 8B" rather than the stable wire id "granite_3_2_8b"),
   * recompute / NLI / disputes tallies, and a collapsed <details> wrapper
   * around the raw JSON payload for developers who still want it.
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
