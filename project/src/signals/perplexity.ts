/**
 * Signal 2: Logprob-based perplexity / token-entropy check.
 *
 * Two strategies, tried in order:
 *
 *   B) Forward-pass rescore via /v1/completions
 *      Sends the question+answer to the legacy completions endpoint with
 *      `echo: true, logprobs: 1, max_tokens: 0`. The model does a single
 *      forward pass and returns logprobs for every token. Cost: ~1–2s.
 *      Works only if LM Studio handles completion-style scoring for the
 *      worker model (chat-tuned models sometimes refuse).
 *
 *   C) Re-generate with logprobs (deeper mode only)
 *      If B fails and the caller allowed regeneration, re-issue the
 *      original question to the worker with `logprobs: true` and capture
 *      the logprobs of the freshly generated answer. The new answer may
 *      differ from what the user originally saw, but the uncertainty
 *      signal is still meaningful. Cost: ~8s.
 *
 * If both fail, return ran: false with a descriptive note. The pipeline
 * gracefully degrades.
 */

import {
  LM_STUDIO_URL,
  WORKER_MODEL_NAME,
  PERPLEXITY_LOW_CONFIDENCE_LOGPROB,
  PERPLEXITY_MAX_FLAGGED_SPANS,
  CRITIC_TIMEOUT_MS,
  VERBOSE_LOGGING,
} from "../config.js";
import { getLlmClient } from "../llm/client.js";
import type { PerplexityResult } from "../types.js";

/**
 * Try to score the existing answer via /v1/completions with echo=true.
 * Returns null if the endpoint is unavailable or returns no logprobs.
 *
 * [ADAPT] LM Studio's behavior with chat-tuned models on the legacy
 * completions endpoint is inconsistent. If this never works on your
 * setup, deeper mode will fall back to method C automatically.
 */
async function tryForwardPassRescore(
  question: string,
  answer: string
): Promise<PerplexityResult | null> {
  const start = Date.now();

  // Concatenate as a flat completion. Some chat models will emit chat
  // tokens here; we accept that for v1 — the logprobs are still mostly
  // representative of the answer's tokens.
  //
  // [ADAPT] If your worker model has a specific chat template, you can
  // construct the prompt to match (e.g. <|im_start|>user\n... patterns
  // for ChatML models like Qwen).
  const prompt = `${question.trim()}\n\n${answer.trim()}`;

  try {
    const abort = new AbortController();
    const timer = setTimeout(() => abort.abort(), CRITIC_TIMEOUT_MS);

    // The OpenAI SDK exposes legacy completions on `client.completions.create`.
    const response: any = await getLlmClient({
      endpoint: LM_STUDIO_URL,
      apiKey: "lm-studio",
    }).completions.create(
      {
        model: WORKER_MODEL_NAME,
        prompt,
        max_tokens: 0,
        echo: true,
        logprobs: 1,
        temperature: 0,
      } as any,
      { signal: abort.signal }
    );

    clearTimeout(timer);

    const logprobsBlock = response?.choices?.[0]?.logprobs;
    const allTokens: string[] = logprobsBlock?.tokens ?? [];
    const allLogprobs: (number | null)[] = logprobsBlock?.token_logprobs ?? [];
    const textOffsets: number[] = logprobsBlock?.text_offset ?? [];

    if (!allTokens.length || !allLogprobs.length) return null;

    // Slice to just the answer's tokens. Without this, mean logprob and
    // perplexity are diluted by the question's tokens (which are usually
    // more predictable than the answer). `text_offset` is per-token char
    // offset into the prompt we sent; answer begins at `question.length`
    // (accounting for the "\n\n" separator added below).
    const answerCharStart = question.trim().length + 2; // "\n\n"
    let sliceFrom = 0;
    if (textOffsets.length === allTokens.length) {
      const idx = textOffsets.findIndex((off) => off >= answerCharStart);
      if (idx >= 0) sliceFrom = idx;
    }

    const tokens = allTokens.slice(sliceFrom);
    const tokenLogprobs = allLogprobs.slice(sliceFrom);

    if (!tokens.length) return null;

    return computeStats({
      tokens,
      tokenLogprobs,
      method: "forward_pass_rescore",
      start,
    });
  } catch (err) {
    if (VERBOSE_LOGGING) {
      console.error("[perplexity] forward-pass rescore failed:", err);
    }
    return null;
  }
}

/**
 * Method C: re-generate the answer with logprobs and capture them.
 * Costs a fresh generation (~8s on Qwen 3.5 9B).
 */
async function regenerateWithLogprobs(
  question: string
): Promise<PerplexityResult | null> {
  const start = Date.now();

  try {
    const abort = new AbortController();
    const timer = setTimeout(() => abort.abort(), CRITIC_TIMEOUT_MS);

    // Chat completions support logprobs in newer OpenAI SDK / LM Studio versions.
    const response: any = await getLlmClient({
      endpoint: LM_STUDIO_URL,
      apiKey: "lm-studio",
    }).chat.completions.create(
      {
        model: WORKER_MODEL_NAME,
        messages: [{ role: "user", content: question }],
        temperature: 0,
        max_tokens: 800,
        logprobs: true,
        // [ADAPT] top_logprobs limits the alternatives returned per token.
        // We don't use them; 1 keeps the response small.
        top_logprobs: 1,
      } as any,
      { signal: abort.signal }
    );

    clearTimeout(timer);

    const choice = response?.choices?.[0];
    const content = choice?.logprobs?.content as
      | Array<{ token: string; logprob: number }>
      | undefined;

    if (!content || content.length === 0) return null;

    const tokens = content.map((c) => c.token);
    const tokenLogprobs = content.map((c) => c.logprob);

    return computeStats({
      tokens,
      tokenLogprobs,
      method: "regenerate_with_logprobs",
      start,
    });
  } catch (err) {
    if (VERBOSE_LOGGING) {
      console.error("[perplexity] regenerate failed:", err);
    }
    return null;
  }
}

/**
 * Aggregate per-token logprobs into a perplexity result.
 */
function computeStats(params: {
  tokens: string[];
  tokenLogprobs: (number | null)[];
  method: PerplexityResult["method"];
  start: number;
}): PerplexityResult {
  const { tokens, tokenLogprobs, method, start } = params;

  // Drop nulls (some endpoints return null for the first BOS token).
  const valid: Array<{ tok: string; lp: number; idx: number }> = [];
  for (let i = 0; i < tokens.length; i++) {
    const lp = tokenLogprobs[i];
    if (typeof lp === "number" && Number.isFinite(lp)) {
      valid.push({ tok: tokens[i] ?? "", lp, idx: i });
    }
  }

  if (valid.length === 0) {
    return {
      ran: false,
      method,
      tokens_scored: 0,
      mean_logprob: 0,
      perplexity: 0,
      low_confidence_spans: [],
      latency_ms: Date.now() - start,
      notes: "No usable logprobs returned.",
    };
  }

  const meanLogprob =
    valid.reduce((s, v) => s + v.lp, 0) / valid.length;
  // Perplexity = exp(-mean_logprob). High = uncertain.
  const perplexity = Math.exp(-meanLogprob);

  // Find spans of consecutive low-confidence tokens.
  const flagged: PerplexityResult["low_confidence_spans"] = [];
  let runStart: number | null = null;
  let runTokens: string[] = [];
  let runMin = 0;

  for (let i = 0; i < valid.length; i++) {
    const { tok, lp } = valid[i]!;
    if (lp <= PERPLEXITY_LOW_CONFIDENCE_LOGPROB) {
      if (runStart === null) {
        runStart = i;
        runTokens = [tok];
        runMin = lp;
      } else {
        runTokens.push(tok);
        runMin = Math.min(runMin, lp);
      }
    } else if (runStart !== null) {
      flagged.push({
        text: runTokens.join(""),
        min_logprob: runMin,
      });
      if (flagged.length >= PERPLEXITY_MAX_FLAGGED_SPANS) break;
      runStart = null;
      runTokens = [];
    }
  }
  if (runStart !== null && flagged.length < PERPLEXITY_MAX_FLAGGED_SPANS) {
    flagged.push({ text: runTokens.join(""), min_logprob: runMin });
  }

  return {
    ran: true,
    method,
    tokens_scored: valid.length,
    mean_logprob: Number(meanLogprob.toFixed(3)),
    perplexity: Number(perplexity.toFixed(3)),
    low_confidence_spans: flagged,
    latency_ms: Date.now() - start,
    notes:
      `Mean token logprob ${meanLogprob.toFixed(2)} → perplexity ` +
      `${perplexity.toFixed(2)}. ${flagged.length} low-confidence span(s) ` +
      `(threshold: logprob <= ${PERPLEXITY_LOW_CONFIDENCE_LOGPROB}).`,
  };
}

/**
 * Public entry point.
 *
 * @param allowRegeneration  If true (deeper mode), fall back to method C
 *                           when method B fails. If false (deep mode), give
 *                           up after method B and report unavailable.
 */
export async function runPerplexityCheck(params: {
  question: string;
  answer: string;
  allowRegeneration: boolean;
}): Promise<PerplexityResult> {
  const start = Date.now();

  // Method B: forward-pass rescore (fast, ~1–2s)
  const methodB = await tryForwardPassRescore(params.question, params.answer);
  if (methodB && methodB.ran) return methodB;

  if (!params.allowRegeneration) {
    return {
      ran: false,
      method: "forward_pass_rescore",
      tokens_scored: 0,
      mean_logprob: 0,
      perplexity: 0,
      low_confidence_spans: [],
      latency_ms: Date.now() - start,
      notes:
        "Forward-pass rescore unavailable on this LM Studio setup. " +
        "Use /verifydeeper to enable regeneration fallback (~8s).",
    };
  }

  // Method C: regenerate with logprobs (deeper mode only)
  const methodC = await regenerateWithLogprobs(params.question);
  if (methodC && methodC.ran) return methodC;

  return {
    ran: false,
    method: "regenerate_with_logprobs",
    tokens_scored: 0,
    mean_logprob: 0,
    perplexity: 0,
    low_confidence_spans: [],
    latency_ms: Date.now() - start,
    notes:
      "Both forward-pass rescore and regeneration with logprobs failed. " +
      "LM Studio may not expose logprobs for this worker model.",
  };
}
