/**
 * Unified critic caller.
 *
 * Replaces the separate `lm-studio.ts` / `ollama.ts` modules. Every critic
 * now goes through this single function, parameterised by a CriticConfig
 * (see ./critic-configs.ts). Adding, removing, or swapping a critic is a
 * pure-config change — no code path is specific to any one model.
 *
 * Responsibilities:
 *   1. Fit the user message to the critic's context window (if the config
 *      declares one), trimming the PRIOR CONTEXT block from the oldest
 *      end until the total input fits.
 *   2. Issue the OpenAI-compatible chat.completions call with a per-call
 *      abort timer.
 *   3. Parse the response with the robust parser; promote parse failures
 *      to `unavailable: true` so aggregation treats them correctly.
 *   4. Return a uniformly shaped CriticResult for success, parse failure,
 *      and transport failure alike.
 */

import {
  CRITIC_MAX_TOKENS,
  CRITIC_TEMPERATURE,
  CRITIC_TIMEOUT_MS,
  CRITIC_TOP_P,
  VERBOSE_LOGGING,
} from "../config.js";
import { getLlmClient } from "../llm/client.js";
import { countTokens, truncateToTokenBudget } from "../tokenizer.js";
import type { CriticResult } from "../types.js";
import type { CriticConfig } from "./critic-configs.js";
import { parseCriticJson } from "./parse.js";

/**
 * Trim prior_context inside an already-built user message so the whole
 * input (system + user + reserved generation) fits within `contextLimit`.
 * Trims from the head of prior_context first (preserves most-recent turns).
 * As a last resort, trims the ASSISTANT ANSWER block too - this is what
 * lets the verity handle 10k+ char answers against 4k-context critics.
 */
function fitToContext(
  systemPrompt: string,
  userMessage: string,
  contextLimit: number,
  contextHeadroom: number
): { userMessage: string; truncated: boolean } {
  const budget = contextLimit - contextHeadroom - CRITIC_MAX_TOKENS;
  const baseTokens = countTokens(systemPrompt) + countTokens(userMessage);
  if (baseTokens <= budget) return { userMessage, truncated: false };

  // Try to locate the PRIOR CONTEXT block. If none, fall through to answer trim.
  const contextRegex =
    /### PRIOR CONTEXT[^\n]*\n([\s\S]*?)(?=\n### USER QUESTION)/;
  const match = userMessage.match(contextRegex);
  if (!match) {
    // No prior_context to trim. Try trimming the ANSWER block instead.
    return trimAnswerBlock(systemPrompt, userMessage, budget);
  }

  const fullBlock = match[0];
  const originalContext = match[1] ?? "";
  const systemTokens = countTokens(systemPrompt);
  const userWithoutContextTokens = countTokens(
    userMessage.replace(fullBlock, "")
  );
  const availableForContext =
    budget - systemTokens - userWithoutContextTokens - 32; // 32-token slack for the header

  if (availableForContext <= 0) {
    // Even without any prior_context we'd be over. Strip it entirely and
    // hope question+answer fit.
    return {
      userMessage: userMessage.replace(fullBlock, ""),
      truncated: true,
    };
  }

  const { text: trimmed } = truncateToTokenBudget(
    originalContext,
    availableForContext
  );
  const replacement =
    "### PRIOR CONTEXT (TRUNCATED to fit context window)\n" +
    trimmed.trim() +
    "\n";
  const replaced = userMessage.replace(fullBlock, replacement);
  // After trimming prior_context, we might STILL be over budget if the
  // answer alone is huge. Fall through to answer-trim if so.
  if (countTokens(systemPrompt) + countTokens(replaced) > budget) {
    return trimAnswerBlock(systemPrompt, replaced, budget);
  }
  return { userMessage: replaced, truncated: true };
}

/**
 * Trim the ASSISTANT ANSWER block as a last resort when the answer itself
 * is larger than the critic's context window. Preserves the head of the
 * answer (usually the more important part) and appends an elision marker.
 */
function trimAnswerBlock(
  systemPrompt: string,
  userMessage: string,
  budget: number
): { userMessage: string; truncated: boolean } {
  const answerRegex =
    /### ASSISTANT ANSWER TO CRITIQUE\n([\s\S]*?)(?=\n### YOUR TASK)/;
  const match = userMessage.match(answerRegex);
  if (!match) return { userMessage, truncated: false };

  const originalAnswer = match[1] ?? "";
  const overhead =
    countTokens(userMessage.replace(match[0], "")) + countTokens(systemPrompt);
  const available = budget - overhead - 64; // slack for header + elision marker
  if (available <= 0) {
    // Not even the prompt scaffolding fits. Give up; backend will error.
    return { userMessage, truncated: false };
  }

  // Keep the HEAD of the answer (first portion) - usually where the thesis
  // statement / main claim is. Append an elision marker.
  const { text: head } = truncateToTokenBudget(originalAnswer, available - 16);
  const replacement =
    "### ASSISTANT ANSWER TO CRITIQUE (TRUNCATED - tail removed to fit context)\n" +
    head.trim() +
    "\n...[truncated]\n";
  return {
    userMessage: userMessage.replace(match[0], replacement),
    truncated: true,
  };
}

export async function callCritic(
  cfg: CriticConfig,
  args: {
    systemPrompt: string;
    userMessage: string;
  }
): Promise<CriticResult> {
  const start = Date.now();
  const notes: string[] = [];
  let userMessage = args.userMessage;

  if (cfg.contextLimit !== undefined) {
    const fitted = fitToContext(
      args.systemPrompt,
      userMessage,
      cfg.contextLimit,
      cfg.contextHeadroom ?? 2_000
    );
    userMessage = fitted.userMessage;
    if (fitted.truncated) {
      notes.push(
        `${cfg.displayName} input truncated to fit ${cfg.contextLimit}-token context window.`
      );
    }
  }

  // 2026-05-12 (E2): timer + abort moved outside the try so the
  // finally can always clear it. Previously the clearTimeout was at
  // line 179 inside the try, unreachable on any throw. The orphan
  // setTimeout then fired on a dead AbortController and kept the
  // Node event loop alive past process intent.
  const abort = new AbortController();
  const timer = setTimeout(() => abort.abort(), CRITIC_TIMEOUT_MS);
  try {
    const response = await getLlmClient({
      endpoint: cfg.endpoint,
      apiKey: cfg.apiKey,
    }).chat.completions.create(
      {
        model: cfg.model,
        messages: [
          { role: "system", content: args.systemPrompt },
          { role: "user", content: userMessage },
        ],
        temperature: CRITIC_TEMPERATURE,
        max_tokens: CRITIC_MAX_TOKENS,
        top_p: CRITIC_TOP_P,
      },
      { signal: abort.signal }
    );

    const raw = response.choices[0]?.message?.content ?? "";
    if (VERBOSE_LOGGING) {
      console.error(`[${cfg.id}] raw response:`, raw.slice(0, 500));
    }

    const parsed = parseCriticJson(raw);
    if (!parsed) {
      return {
        id: cfg.id,
        display_name: cfg.displayName,
        family: cfg.family,
        verdict: "error",
        severity: 0,
        concerns: [],
        suggested_fixes: [],
        notes,
        unavailable: true,
        error: `Could not parse JSON from critic response (${raw.length} chars).`,
        latency_ms: Date.now() - start,
      };
    }

    return {
      id: cfg.id,
      display_name: cfg.displayName,
      family: cfg.family,
      verdict: parsed.verdict,
      severity: parsed.severity,
      concerns: parsed.concerns,
      suggested_fixes: parsed.suggested_fixes,
      notes,
      latency_ms: Date.now() - start,
      weight: cfg.weight ?? 1,
    };
  } catch (err) {
    if (VERBOSE_LOGGING) console.error(`[${cfg.id}] error:`, err);
    return {
      id: cfg.id,
      display_name: cfg.displayName,
      family: cfg.family,
      verdict: "error",
      severity: 0,
      concerns: [],
      suggested_fixes: [],
      notes,
      unavailable: true,
      error: err instanceof Error ? err.message : String(err),
      latency_ms: Date.now() - start,
    };
  } finally {
    clearTimeout(timer);
  }
}
