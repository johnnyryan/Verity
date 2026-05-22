/**
 * ═══════════════════════════════════════════════════════════════════════════
 *  CRITIC DEFINITIONS  [★ THIS IS WHERE YOU ADD/REMOVE/SWAP CRITICS ★]
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Each critic is a single config object that pairs a stable identity
 * (id, display name, family) with an endpoint + model tag. The pipeline
 * iterates `ALL_CRITICS` and calls the unified `callCritic()` for each;
 * there are no model-specific code paths outside of context fitting.
 *
 *   Swap a critic's backend (e.g. Gemma from Ollama → LM Studio):
 *     change `endpoint` and `apiKey`. Nothing else.
 *
 *   Swap a critic's model (same backend):
 *     change `model`. If the new model has a smaller context window, set
 *     `contextLimit` so the caller trims prior_context automatically.
 *
 *   Add a fourth critic:
 *     append a new CriticConfig to ALL_CRITICS, extend
 *     VerifyOutput.critics in types.ts with the new id as a key, and
 *     populate it in pipeline.ts's final result assembly.
 *
 *   Remove a critic:
 *     delete it from ALL_CRITICS, drop its key from VerifyOutput.critics,
 *     and lower MAX_UNAVAILABLE_CRITICS in config.ts to match the new
 *     fleet size.
 */

import {
  OLLAMA_URL,
  CRITIC_A_MODEL,
  CRITIC_B_MODEL,
  CRITIC_C_MODEL,
} from "../config.js";

export interface CriticConfig {
  /** Stable identifier used as the key in VerifyOutput.critics. */
  id: string;
  /** Human-readable name surfaced in summaries. */
  displayName: string;
  /** Training family, used in diversity reporting. */
  family: string;
  /** Full endpoint URL, e.g. "http://localhost:1234/v1". */
  endpoint: string;
  /**
   * Dummy auth string. Local servers (LM Studio, Ollama) ignore the value
   * but the OpenAI SDK requires it to be a non-empty string.
   */
  apiKey: string;
  /** Model tag/id as the backend reports it (see config.ts for how to find it). */
  model: string;
  /**
   * Native context window in tokens. When set, callCritic() truncates
   * prior_context so the input fits within (contextLimit - contextHeadroom
   * - CRITIC_MAX_TOKENS). Leave undefined for 128k+ models where we never
   * realistically hit the ceiling.
   */
  contextLimit?: number;
  /**
   * Reserved headroom (tokens) for generation + formatting overhead.
   * Subtracted from contextLimit before deciding how much prior_context to
   * pass through.
   */
  contextHeadroom?: number;
  /**
   * Relative vote weight in the aggregator. Higher-capability critics get
   * larger weights so they can break ties. Default 1.
   *
   * 2026-04-18: Granite 3.2 8B → 2, Granite 3.2 2B → 1. Evidence: the 8B
   * is measurably stronger on citation detection and catches errors the
   * 2B misses (RESULTS.md). Bumping its weight to 2 means a lone 2B fail
   * cannot outvote a confident 8B pass (fixes `code-clean` false positive).
   */
  weight?: number;
}

// ─────────────────────────────────────────────────────────────────────────
// Critic A — the larger / "strong" critic on the weak GPU via Ollama.
//
// History (see CRITIC_A_MODEL comment in config.ts for the full
// timeline): Phi-4 14B on LM Studio/CUDA → Phi-3.5-mini on Ollama →
// Phi-4-mini → Granite 3.2 8B → Granite 4.1 8B (current default;
// pinned via CRITIC_A_MODEL env). The slot is model-agnostic — the
// `family` field below reflects whatever model is wired up.
//
// Pick the largest critic that still fits alongside Critic B in the
// weak GPU's VRAM budget. Critic A's job is to catch subtle code
// bugs, off-by-ones, missing null checks, and citation errors that
// the smaller Critic B misses. weight = 2 so a lone Critic B "fail"
// can't outvote a confident Critic A "pass".
//
// [ADAPT] contextLimit is sized so the critic can ingest a multi-
// paragraph answer plus moderate prior_context without truncation.
// Beyond contextLimit, callCritic() truncates prior_context from
// the head. Use the model card's documented native context window;
// 4 KB / 8 KB are typical for small critics.
// ─────────────────────────────────────────────────────────────────────────
export const CRITIC_A: CriticConfig = {
  // Wire id is model-agnostic (renamed 2026-05-20 from the prior
  // model-specific "granite_3_2_8b" — itself renamed 2026-05-11 from the
  // legacy "phi4_reasoning"). Decoupling the slot name from the model
  // means future critic swaps don't require touching three files.
  // displayName is the human-facing label.
  id: "critic_a",
  displayName: "Critic A",
  family: "IBM",
  endpoint: OLLAMA_URL,
  apiKey: "ollama",
  model: CRITIC_A_MODEL,
  contextLimit: 4_096,
  contextHeadroom: 512,
  weight: 2,
};

// ─────────────────────────────────────────────────────────────────────────
// Critic B — the smaller / "fast" critic on the weak GPU via Ollama.
//
// Critic B's job is a quick second voice: catch simple errors, confirm
// or dissent from Critic A, and provide cross-family diversity. Pick
// a 1-3 B model from a different training corpus than Critic A (or at
// minimum a distinct scratch corpus from the same vendor — IBM's 2B
// and 8B Granites qualified per IBM's release notes). The 2026-04-17
// Phase-3 sweep selected Granite 3.2 2B (144 tok/s warm, 334 ms per
// critic call, 4/4 correct on the test corpus). 2026-05-12 the slot
// was swapped to Ministral 3 3B (Mistral) to restore the cross-family
// axis (Critic A = IBM, Critic B = Mistral, worker = Qwen). The slot
// itself is model-agnostic; whatever CRITIC_B_MODEL points at gets
// loaded. weight = 1.
// ─────────────────────────────────────────────────────────────────────────
export const CRITIC_B: CriticConfig = {
  // 2026-05-12: critic B is now Ministral-3 3B (Mistral) instead of
  // Granite 3.2 2B (IBM). 2026-05-20: wire id renamed from the prior
  // model-specific "granite_3_2_2b" to the model-agnostic "critic_b"
  // so future model swaps don't drag the wire id with them.
  id: "critic_b",
  displayName: "Critic B",
  family: "Mistral",
  endpoint: OLLAMA_URL,
  apiKey: "ollama",
  model: CRITIC_B_MODEL,
  contextLimit: 4_096,
  contextHeadroom: 512,
  weight: 1,
};

// ─────────────────────────────────────────────────────────────────────────
// Critic C — CURRENTLY DISABLED (Llama 3.2 3B / Meta)
//
// Dropped from ALL_CRITICS in the 2-critic / 8k-context redesign. Kept
// as an exported config so a third critic can be re-enabled by adding
// CRITIC_C back to ALL_CRITICS (and updating MAX_UNAVAILABLE_CRITICS).
// VerifyOutput.critics is now a Record<string, CriticResult> keyed by
// CriticConfig.id, so re-enabling here is a one-file change and the
// pipeline picks up the new key automatically.
// Model file is still on disk (`ollama list` shows llama3.2:3b).
// ─────────────────────────────────────────────────────────────────────────
export const CRITIC_C: CriticConfig = {
  id: "llama32_3b",
  displayName: "Llama 3.2 3B",
  family: "Meta",
  endpoint: OLLAMA_URL,
  apiKey: "ollama",
  model: CRITIC_C_MODEL,
  contextLimit: 8_192,
  contextHeadroom: 1_024,
};

export const ALL_CRITICS: readonly CriticConfig[] = [
  CRITIC_A,
  CRITIC_B,
  // CRITIC_C,  // re-enable with matching changes listed on CRITIC_C above
];
