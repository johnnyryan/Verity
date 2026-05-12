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
// Critic A — Microsoft-family critic on Ollama/Vulkan (AMD 5700 XT)
//
// History (see CRITIC_A_MODEL comment in config.ts for full timeline):
//   Phi-4 14B on LM Studio/CUDA -> Phi-3.5-mini on Ollama -> Phi-4-mini.
//
// Phi-4-mini 3.8B (Microsoft, Jan 2025) is the current pick - newer and
// stronger than Phi-3.5-mini on reasoning benchmarks, same memory footprint.
// Paired with Nemotron Mini in a 2-critic design that fits the 8 GB AMD
// card with 8k context per model and room to breathe.
//
// [ADAPT] contextLimit = 8192 is sized so the critic can ingest a
// multi-paragraph answer plus moderate prior_context without truncation.
// Beyond 8k, callCritic() truncates prior_context from the head.
// ─────────────────────────────────────────────────────────────────────────
export const CRITIC_A: CriticConfig = {
  // Wire id matches the actual running model (renamed 2026-05-11 from the
  // legacy "phi4_reasoning"). displayName is the human-facing label.
  id: "granite_3_2_8b",
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
// Critic B — IBM-family critic on Ollama/Vulkan (AMD 5700 XT)
//
// Granite 3.2 2B won the Phase-3 sweep of 8 small models on this hardware:
// 144 tok/s warm, 334 ms per critic call, 4/4 correct on the test suite.
// That's faster than Phi-4-mini and catches the same errors.
//
// IBM's training corpus + instruction tuning are distinct from Microsoft
// (Phi), NVIDIA (Nemotron), Meta (Llama), and Google (Gemma), so pairing
// Phi-4-mini + Granite 3.2 gives a real training-family diversity axis.
//
// Wire id matches the actual running model (renamed 2026-05-11 from the
// legacy "nemotron_mini"). VerifyOutput.critics.granite_3_2_2b in types.ts
// is the matching JSON key.
// ─────────────────────────────────────────────────────────────────────────
export const CRITIC_B: CriticConfig = {
  id: "granite_3_2_2b",
  displayName: "Critic B",
  family: "IBM",
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
// CRITIC_C back to ALL_CRITICS (and updating MAX_UNAVAILABLE_CRITICS +
// VerifyOutput.critics.llama32_3b + the findCritic() call in pipeline.ts).
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
