/**
 * LLM-driven factual-claim extraction.
 *
 * The regex heuristic in ./extract-claims.ts is cheap (~0ms) but coarse —
 * it misses implicit claims and surfaces false positives on sentences
 * containing numbers or proper nouns that aren't actually factual
 * assertions. In `deep` and `deeper` modes we can afford a ~1–2s call to
 * the worker model to get a much higher-signal claim list, which then
 * feeds both the NLI check and the consistency re-sampling.
 *
 * Implementation notes:
 *   - We reuse the worker model (Qwen 3.5 9B) rather than standing up a
 *     dedicated classifier. It's already loaded and is competent at
 *     this task with a tight prompt.
 *   - We request JSON object mode. Both LM Studio and Ollama's
 *     OpenAI-compatible endpoints honour response_format.
 *   - On any failure (timeout, parse error, empty result) we return `null`
 *     so the caller can transparently fall back to the regex extractor.
 */

import OpenAI from "openai";

import {
  CRITIC_TIMEOUT_MS,
  WORKER_ENDPOINT,
  WORKER_API_KEY,
  NLI_MAX_CLAIMS,
  VERBOSE_LOGGING,
  WORKER_MODEL_NAME,
} from "../config.js";
import { findBalancedJsonObject } from "../critics/parse.js";
import { getLlmClient } from "../llm/client.js";
import { stripReasoningTraces } from "../sanitize.js";

const SYSTEM_PROMPT = `You extract factual claims from an assistant's answer so a verity can fact-check them.

A "factual claim" is a declarative sentence asserting something verifiable:
  - a number, date, statistic, or measurement
  - a named entity's identity, role, or property
  - a causal or historical statement
  - a definition presented as authoritative

EXCLUDE:
  - questions, commands, and suggestions
  - pure opinions ("I think", "in my view")
  - code snippets, syntax, or shell commands
  - meta-statements about the response itself ("as an AI", "here is my answer")
  - hedged statements offered as uncertain ("maybe", "possibly")

Return ONLY a JSON object matching this schema:
  { "claims": ["claim 1", "claim 2", ...] }

Each claim must be a self-contained sentence that can be verified on its
own. Rewrite for clarity if needed, but do not introduce facts that aren't
in the source. Return an empty array if the text has no factual claims.
Return at most ${NLI_MAX_CLAIMS} of the most important claims.`;

/**
 * Extract factual claims via the worker model. Returns null on any
 * failure so the caller can fall back to the regex extractor.
 */
export async function extractClaimsLLM(answer: string): Promise<string[] | null> {
  const start = Date.now();
  // 2026-05-12 (E2): timer + abort moved outside the try so the
  // finally can always clear it. Previously the clearTimeout was
  // inside the try, unreachable on any throw, leaving an orphan
  // setTimeout that fired on a dead AbortController and kept the
  // Node event loop alive.
  const abort = new AbortController();
  const timer = setTimeout(() => abort.abort(), CRITIC_TIMEOUT_MS);
  try {
    const response = await getLlmClient({
      endpoint: WORKER_ENDPOINT,
      apiKey: WORKER_API_KEY,
    }).chat.completions.create(
      {
        model: WORKER_MODEL_NAME,
        messages: [
          { role: "system", content: SYSTEM_PROMPT },
          { role: "user", content: answer },
        ],
        temperature: 0,
        // Claims output is a JSON array of <= NLI_MAX_CLAIMS short
        // sentences — typical realised size is ~150-300 tokens. 600 leaves
        // ample headroom and frees KV-cache the worker can use for its
        // actual generation. Was 1500 (legacy default).
        max_tokens: 600,
        // [ADAPT] If your worker model refuses response_format, drop this
        // line and rely on the robust JSON parsing below; the prompt still
        // asks for JSON.
        response_format: { type: "json_object" },
      } as OpenAI.Chat.ChatCompletionCreateParamsNonStreaming,
      { signal: abort.signal }
    );

    const raw = response.choices[0]?.message?.content ?? "";
    if (VERBOSE_LOGGING) {
      console.error(
        `[extract-claims-llm] ${raw.length} chars in ${Date.now() - start}ms`
      );
    }

    return parseClaimsJson(raw);
  } catch (err) {
    if (VERBOSE_LOGGING) console.error("[extract-claims-llm] error:", err);
    return null;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Parse the worker's JSON response, tolerating markdown fences or minor
 * prose before/after the JSON object. Returns null if no usable claim
 * array can be recovered.
 */
export function parseClaimsJson(raw: string): string[] | null {
  if (!raw) return null;

  let cleaned = stripReasoningTraces(raw);
  cleaned = cleaned.replace(/^```(?:json)?\s*/i, "");
  cleaned = cleaned.replace(/\s*```\s*$/i, "");

  // Use the shared balanced-brace finder so prose containing `{` characters
  // before the real JSON object doesn't poison the span (the previous
  // indexOf/lastIndexOf scanner could pick up a `{` inside the preamble).
  const candidate = findBalancedJsonObject(cleaned, { requireKey: "claims" });
  if (!candidate) return null;

  let parsed: unknown;
  try {
    parsed = JSON.parse(candidate);
  } catch {
    return null;
  }
  if (!parsed || typeof parsed !== "object") return null;
  const p = parsed as { claims?: unknown };
  if (!Array.isArray(p.claims)) return null;

  const claims = p.claims
    .filter((c): c is string => typeof c === "string")
    .map((c) => c.trim())
    .filter((c) => c.length > 0)
    .slice(0, NLI_MAX_CLAIMS);

  return claims;
}
