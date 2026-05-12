/**
 * Centralised token counting.
 *
 * Uses js-tiktoken with the cl100k_base encoding (GPT-3.5/4 family). That
 * isn't the exact tokenizer any of our non-OpenAI models use, but it
 * produces counts that are within ~10–20% of the real value for English,
 * code, and most non-English — far better than the previous
 * `Math.ceil(chars / 3.5)` estimate, which can be off by 40% on code.
 *
 * We use this only to decide when to truncate prior_context so inputs fit
 * within a critic's context window. Exact counts aren't needed — safe
 * over-estimation is what we want. cl100k tends to be slightly conservative
 * for code and non-English, which is the right failure direction here.
 *
 * [ADAPT] If you adopt a model whose tokenizer diverges a lot from
 * cl100k_base (e.g. some multilingual models), either increase the
 * contextHeadroom on its CriticConfig, or swap in the real tokenizer via
 * @huggingface/transformers.
 */

import { getEncoding } from "js-tiktoken";

// Lazy-initialised so the BPE data is only loaded when token counting is
// actually called. Avoids startup cost for non-Phi-4 critics that don't
// need context fitting.
let _enc: ReturnType<typeof getEncoding> | null = null;

function enc(): ReturnType<typeof getEncoding> {
  if (!_enc) _enc = getEncoding("cl100k_base");
  return _enc;
}

export function countTokens(text: string): number {
  if (!text) return 0;
  try {
    return enc().encode(text).length;
  } catch {
    // Defensive fallback: should never fire, but if tiktoken blows up on
    // exotic input, we'd rather over-estimate than crash the pipeline.
    return Math.ceil(text.length / 3);
  }
}

/**
 * Boot-time warmup. Forces tiktoken's BPE-table init and one encode pass
 * so the first real countTokens() / truncateToTokenBudget() call doesn't
 * pay the lazy-init cost on the request hot path.
 *
 * Fault-tolerant: any error is logged and swallowed.
 */
export function warmupTokenizer(): void {
  try {
    enc().encode("warmup");
  } catch (err) {
    console.error("[tokenizer] warmup failed (non-fatal):", err);
  }
}

/**
 * Truncate `text` so its token count fits within `maxTokens`, preserving
 * the tail (most-recent content). Returns the original string unchanged
 * when it already fits.
 *
 * Used for prior_context trimming: the latest turns are the most relevant
 * to the question being critiqued, so we drop from the front.
 */
export function truncateToTokenBudget(
  text: string,
  maxTokens: number
): { text: string; truncated: boolean } {
  if (maxTokens <= 0) return { text: "", truncated: true };
  const encoding = enc();
  const tokens = encoding.encode(text);
  if (tokens.length <= maxTokens) return { text, truncated: false };
  const kept = tokens.slice(tokens.length - maxTokens);
  return { text: encoding.decode(kept), truncated: true };
}
