/**
 * Main verification pipeline.
 *
 * Always runs: 3 critics + NLI (the standard ensemble).
 *
 * In "deep" or "deeper" modes, also runs:
 *   - Consistency check (SelfCheckGPT-style worker re-sampling)
 *   - Perplexity check (logprob-based token entropy)
 *
 * The two extra signals are kicked off in parallel with the critics, so
 * total wall-clock for "deep" is dominated by whichever takes longest
 * (typically the worker re-samples).
 */

import { aggregate, renderSummaryMarkdown } from "./aggregator.js";
import {
  CONSISTENCY_SAMPLES_DEEP,
  CONSISTENCY_SAMPLES_DEEPER,
  PIPELINE_TIMEOUT_MS,
  VERBOSE_LOGGING,
} from "./config.js";
import { callCritic } from "./critics/call-critic.js";
import { ALL_CRITICS } from "./critics/critic-configs.js";
import { runNliCheck } from "./nli/classifier.js";
import { extractClaims } from "./nli/extract-claims.js";
import { extractClaimsLLM } from "./nli/extract-claims-llm.js";
import {
  buildCriticUserMessage,
  getCriticPrompt,
} from "./prompts.js";
import { runConsistencyCheck } from "./signals/consistency.js";
import { runPerplexityCheck } from "./signals/perplexity.js";
import { runRecomputePass } from "./signals/recompute.js";
import type {
  ConsistencyResult,
  CriticResult,
  NliResult,
  PerplexityResult,
  RecomputeResult,
  VerifyInput,
  VerifyMode,
  VerifyOutput,
} from "./types.js";

/**
 * Pick a claim extractor based on verification depth.
 *   - standard mode: cheap regex heuristic (~0 ms, noisy).
 *   - deep / deeper: worker-model extraction with JSON mode (~1–2 s,
 *     much higher signal). Falls back to the regex extractor if the LLM
 *     call fails or returns an empty list, so a stuck worker never
 *     starves the downstream NLI / consistency checks.
 */
async function extractClaimsForMode(
  answer: string,
  mode: VerifyMode
): Promise<string[]> {
  if (mode === "standard") return extractClaims(answer);
  const llmClaims = await extractClaimsLLM(answer);
  if (llmClaims && llmClaims.length > 0) return llmClaims;
  // LLM extractor returned null (error) or empty → fall back to regex so
  // we still get *something* for NLI / consistency to work with.
  return extractClaims(answer);
}

/**
 * Synthesise a CriticResult for use when the pipeline ceiling fires before
 * the real call completes. Marked unavailable so aggregation counts it
 * toward MAX_UNAVAILABLE_CRITICS rather than treating it as a silent pass.
 */
function makeTimedOutCritic(
  id: string,
  displayName: string,
  family: string
): CriticResult {
  return {
    id,
    display_name: displayName,
    family,
    verdict: "error",
    severity: 0,
    concerns: [],
    suggested_fixes: [],
    notes: [],
    unavailable: true,
    error: "Pipeline wall-clock timeout reached before critic responded.",
    latency_ms: 0,
  };
}

export async function runVerificationPipeline(
   input: VerifyInput
 ): Promise<VerifyOutput> {
   const start = Date.now();

   const question = input.question ?? "";
   const answer = input.answer ?? "";
   const taskType = input.task_type ?? "auto";
   // Default: with_context - the tool description asks Qwen to always pass
   // prior_context. See AUTO-POPULATE note in src/index.ts.
   const contextMode = input.context_mode ?? "with_context";
   // Raw prior_context is kept for NLI regardless of context_mode (NLI benefits
   // from a premise without suffering 'too much context hurts' the way critics
   // do - Chen et al. 2024).
   const priorContextForNli =
     contextMode === "minimal" ? undefined : input.prior_context?.trim() || undefined;
   // Critics get prior_context only on explicit 'with_context' or 'full'.
   // We intentionally narrow it more aggressively for critics: pass only if
   // the user really asked for it via the trigger modifier. This keeps the
   // critic prompts focused while NLI still benefits from the premise.
   const priorContextForCritics =
     contextMode === "with_context" || contextMode === "full"
       ? priorContextForNli
       : undefined;
   const useNli = input.use_nli !== false;
   const mode: VerifyMode = input.mode ?? "standard";

   if (VERBOSE_LOGGING) {
     console.error(
       `[pipeline] mode=${mode} task_type=${taskType} context_mode=${contextMode} ` +
         `use_nli=${useNli} q_len=${question.length} a_len=${answer.length} ` +
         `nli_premise_len=${priorContextForNli?.length ?? 0}`
     );
   }

   const { prompt: systemPrompt, resolved_task_type } = getCriticPrompt(
     taskType,
     answer
   );

   const userMessage = buildCriticUserMessage(
     question,
     answer,
     priorContextForCritics
   );

   // Add small delay before starting critics to allow for clean transitions
   // This helps prevent "already exists" conflicts when models are being swapped
   await new Promise(resolve => setTimeout(resolve, 100));

   // ── Critics (run sequentially with delays to prevent model conflicts) ─────
   //
   // Running critics sequentially with small delays helps prevent "already exists"
   // conflicts and ensures clean model transitions in Ollama.
   const criticPromises: Promise<CriticResult>[] = [];
   for (const cfg of ALL_CRITICS) {
     // Add delay before each critic call (except the first) to allow for clean transitions
     if (criticPromises.length > 0) {
       await new Promise(resolve => setTimeout(resolve, 500));
     }
     criticPromises.push(callCritic(cfg, { systemPrompt, userMessage }));
   }

  // ── Claim extraction (fires in parallel with critics) ────────────────
  //
  // Both the NLI check and the consistency check need a list of claims.
  // We extract them ONCE here and share, so:
  //   - NLI and consistency are evaluated against the same evidence set,
  //     making their results directly comparable.
  //   - We pay the extraction cost at most once per request.
  //
  // In standard mode this resolves synchronously (regex); in deep modes
  // it's a ~1–2 s call to the worker model with a regex fallback.
  // Defensive .catch: extractClaimsForMode is documented to swallow LLM
  // failures and fall back to regex, but an upstream throw (e.g. abort,
  // network reset) would otherwise propagate as an unhandled rejection
  // through the chained .then below.
  const claimsPromise: Promise<string[]> = extractClaimsForMode(
    answer,
    mode
  ).catch((err) => {
    if (VERBOSE_LOGGING) {
      console.error("[pipeline] claim extraction errored, using empty list:", err);
    }
    return [];
  });

  const nliPromise: Promise<NliResult> = useNli
    ? claimsPromise.then((claims) =>
        runNliCheck(answer, priorContextForNli, claims).catch((err) => ({
          ran: false,
          claims_checked: 0,
          contradictions: [],
          unsupported: [],
          notes: `NLI check errored: ${err instanceof Error ? err.message : String(err)}`,
        }))
      )
    : Promise.resolve<NliResult>({
        ran: false,
        claims_checked: 0,
        contradictions: [],
        unsupported: [],
        notes: "NLI disabled via use_nli=false.",
      });

  // ── Deterministic recompute pass (always, fast, no LLM calls) ────────
  //
  // Runs in parallel with NLI. Catches arithmetic, enumeration, leap-year
  // and unit-constant errors with 100% precision (no model uncertainty).
  // See src/signals/recompute.ts.
  const recomputePromise: Promise<RecomputeResult> = runRecomputePass(
    answer,
    question
  ).catch((err) => ({
    ran: false,
    expressions_found: 0,
    verifications: [],
    mismatches: [],
    notes: `Recompute errored: ${err instanceof Error ? err.message : String(err)}`,
    latency_ms: 0,
  }));

  // ── Deep-mode signals (only run when requested) ──────────────────────
  const wantsDeepSignals = mode === "deep" || mode === "deeper";
  const allowRegeneration = mode === "deeper";
  const consistencySamples =
    mode === "deeper" ? CONSISTENCY_SAMPLES_DEEPER : CONSISTENCY_SAMPLES_DEEP;

  const consistencyPromise: Promise<ConsistencyResult | undefined> =
    wantsDeepSignals
      ? claimsPromise.then((claims) =>
          runConsistencyCheck({
            question,
            originalAnswer: answer,
            numSamples: consistencySamples,
            preExtractedClaims: claims,
          })
        ).catch((err) => ({
          ran: false,
          samples_generated: 0,
          claims_checked: 0,
          contradicted: [],
          unsupported: [],
          divergence_score: 0,
          latency_ms: 0,
          notes: `Consistency check errored: ${
            err instanceof Error ? err.message : String(err)
          }`,
        }))
      : Promise.resolve(undefined);

  const perplexityPromise: Promise<PerplexityResult | undefined> =
    wantsDeepSignals
      ? runPerplexityCheck({
          question,
          answer,
          allowRegeneration,
        }).catch((err) => ({
          ran: false,
          method: "forward_pass_rescore" as const,
          tokens_scored: 0,
          mean_logprob: 0,
          perplexity: 0,
          low_confidence_spans: [],
          latency_ms: 0,
          notes: `Perplexity check errored: ${
            err instanceof Error ? err.message : String(err)
          }`,
        }))
      : Promise.resolve(undefined);

  // ── Wait for everything, bounded by a hard wall-clock ceiling ────────
  //
  // Each critic has its own timeout (CRITIC_TIMEOUT_MS), but the pipeline
  // as a whole also needs an upper bound so a single stuck GPU call can't
  // hold an HTTP request open past the client's patience. If we hit the
  // ceiling, any still-running promise is left to settle in the background
  // (its timer will abort it eventually) and we surface a synthetic
  // "unavailable" result.
  const resultsPromise = Promise.all([
    Promise.all(criticPromises),
    nliPromise,
    consistencyPromise,
    perplexityPromise,
    recomputePromise,
  ]);

  let timedOut = false;
  let timeoutHandle: ReturnType<typeof setTimeout> | undefined;
  const timeoutPromise = new Promise<"__pipeline_timeout__">((resolve) => {
    timeoutHandle = setTimeout(() => {
      timedOut = true;
      resolve("__pipeline_timeout__");
    }, PIPELINE_TIMEOUT_MS);
    // Don't keep the event loop alive solely on this timer — if the
    // server is shutting down, the pending request times out cleanly.
    timeoutHandle.unref?.();
  });

  let raced: Awaited<typeof resultsPromise> | "__pipeline_timeout__";
  try {
    raced = await Promise.race([resultsPromise, timeoutPromise]);
  } finally {
    // Clear the timer on the happy path so it doesn't fire 180 s later
    // and resolve a promise nobody is awaiting. Memory cost is small but
    // accumulates in long-lived servers handling many requests.
    if (timeoutHandle !== undefined) clearTimeout(timeoutHandle);
  }

  const [criticResults, nliResult, consistencyResult, perplexityResult, recomputeResult] =
    timedOut || raced === "__pipeline_timeout__"
      ? [
          // Synthesise one "timed out" critic result per configured critic
          // so aggregation reports them as unavailable rather than silently
          // passing. Derived from ALL_CRITICS so renames propagate.
          ALL_CRITICS.map((c) =>
            makeTimedOutCritic(c.id, c.displayName, c.family)
          ),
          {
            ran: false,
            claims_checked: 0,
            contradictions: [],
            unsupported: [],
            notes: `NLI skipped: pipeline timeout after ${PIPELINE_TIMEOUT_MS}ms.`,
          } satisfies NliResult,
          undefined,
          undefined,
          {
            ran: false,
            expressions_found: 0,
            verifications: [],
            mismatches: [],
            notes: `Recompute skipped: pipeline timeout.`,
            latency_ms: 0,
          } satisfies RecomputeResult,
        ] as [CriticResult[], NliResult, ConsistencyResult | undefined, PerplexityResult | undefined, RecomputeResult]
      : (raced as [CriticResult[], NliResult, ConsistencyResult | undefined, PerplexityResult | undefined, RecomputeResult]);

  // Look up critics by id so a reordering of ALL_CRITICS in
  // critic-configs.ts doesn't silently mis-populate the output keys.
  const findCritic = (id: string): CriticResult => {
    const hit = criticResults.find((c) => c.id === id);
    if (hit) return hit;
    // Should be unreachable — criticResults is sized from ALL_CRITICS —
    // but returning a well-formed unavailable result is safer than throwing
    // inside the final assembly.
    return makeTimedOutCritic(id, id, "Unknown");
  };

  const critics = {
    granite_3_2_8b: findCritic("granite_3_2_8b"),
    granite_3_2_2b: findCritic("granite_3_2_2b"),
    // llama32_3b removed with 2-critic redesign; re-add here if CRITIC_C
    // is re-enabled in critic-configs.ts
  };
  const granite8b = critics.granite_3_2_8b;

  const aggregated = aggregate(criticResults, nliResult, {
    consistency: consistencyResult,
    perplexity: perplexityResult,
    recompute: recomputeResult,
  });

  const latency = Date.now() - start;
  if (VERBOSE_LOGGING) {
    console.error(
      `[pipeline] done in ${latency}ms consensus=${aggregated.consensus} mode=${mode}`
    );
  }

  // Assemble the final output *without* summary_md first so the renderer
  // sees everything except the field it's about to produce. Then render
  // and stamp summary_md on at the end.
  const output: VerifyOutput = {
    critics,
    disputes: aggregated.disputes,
    nli_check: nliResult,
    recompute: recomputeResult,
    consistency_check: consistencyResult,
    perplexity: perplexityResult,
    consensus: aggregated.consensus,
    summary: aggregated.summary,
    // Placeholder; filled in after the rest of the payload is known so the
    // renderer can read from a fully-formed VerifyOutput.
    summary_md: "",
    latency_ms: latency,
    meta: {
      mode,
      task_type: resolved_task_type,
      context_mode: contextMode,
      granite_8b_input_truncated: granite8b.notes.some((n) => n.includes("truncated")),
      critics_unavailable: aggregated.critics_unavailable,
    },
  };
  output.summary_md = renderSummaryMarkdown(output, { answer });
  return output;
}
