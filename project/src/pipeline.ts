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
  // 2026-05-12: extractClaimsLLM documents that it returns null on any
  // failure, but a hard throw (network reset, aborted client, JSON
  // structural panic) would bubble up and reject this function. The
  // outer pipeline catches it, but the regex fallback never runs.
  // Wrap defensively so the documented contract is genuinely honoured.
  let llmClaims: string[] | null = null;
  try {
    llmClaims = await extractClaimsLLM(answer);
  } catch (err) {
    if (VERBOSE_LOGGING) {
      console.error("[pipeline] extractClaimsLLM threw; falling back to regex:", err);
    }
  }
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

   // ── Critics (fire in parallel) ──────────────────────────────────────────
  //
  // 2026-05-12: removed a 500ms inter-iteration await + a 100ms pre-loop
  // sleep that were originally added to dodge an Ollama "model already
  // exists" race during cold-load. Both critics share a single Ollama
  // process and a single GPU; Ollama serialises actual GPU work at the
  // hardware level. Adding a 500ms gap in the orchestrator just delayed
  // dispatch without helping concurrency. Net wall-clock saving per
  // /verify call: ~500ms with two critics, scaling linearly if the panel
  // grows. The cold-load conflict it was guarding against was specific
  // to JIT-loading; OLLAMA_MAX_LOADED_MODELS=2 keeps both critics warm
  // (see start-verity.ps1).
  const criticPromises: Promise<CriticResult>[] = ALL_CRITICS.map((cfg) =>
    callCritic(cfg, { systemPrompt, userMessage })
  );

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
  // 2026-05-12 (E6): each signal is now wrapped with its own
  // timeout-with-fallback so the pipeline-wide ceiling preserves any
  // signal that finished in time. Previously a single missed critic
  // would trip the outer Promise.all and the synthesized "timed out"
  // stubs would replace WHOLE the result set — discarding every
  // critic / NLI / recompute that did finish.
  //
  // Each wrapped promise resolves to its fulfilled value if the
  // underlying call beat the timeout; otherwise it resolves to a
  // domain-appropriate "skipped" stub. Promise.all on the wrapped
  // set therefore never rejects and never throws away partial work.
  function withTimeoutFallback<T>(
    p: Promise<T>,
    ms: number,
    fallback: () => T
  ): Promise<T> {
    return new Promise<T>((resolve) => {
      const timer = setTimeout(() => resolve(fallback()), ms);
      timer.unref?.();
      p.then(
        (v) => {
          clearTimeout(timer);
          resolve(v);
        },
        () => {
          // Underlying call rejected (shouldn't, because every leg has
          // its own .catch returning a structured stub, but defensive).
          clearTimeout(timer);
          resolve(fallback());
        }
      );
    });
  }

  const [criticResults, nliResult, consistencyResult, perplexityResult, recomputeResult] =
    await Promise.all([
      Promise.all(
        criticPromises.map((p, i) =>
          withTimeoutFallback<CriticResult>(p, PIPELINE_TIMEOUT_MS, () =>
            makeTimedOutCritic(
              ALL_CRITICS[i].id,
              ALL_CRITICS[i].displayName,
              ALL_CRITICS[i].family
            )
          )
        )
      ),
      withTimeoutFallback<NliResult>(nliPromise, PIPELINE_TIMEOUT_MS, () => ({
        ran: false,
        claims_checked: 0,
        contradictions: [],
        unsupported: [],
        notes: `NLI skipped: pipeline timeout after ${PIPELINE_TIMEOUT_MS}ms.`,
      })),
      withTimeoutFallback<ConsistencyResult | undefined>(
        consistencyPromise,
        PIPELINE_TIMEOUT_MS,
        () => undefined
      ),
      withTimeoutFallback<PerplexityResult | undefined>(
        perplexityPromise,
        PIPELINE_TIMEOUT_MS,
        () => undefined
      ),
      withTimeoutFallback<RecomputeResult>(
        recomputePromise,
        PIPELINE_TIMEOUT_MS,
        () => ({
          ran: false,
          expressions_found: 0,
          verifications: [],
          mismatches: [],
          notes: `Recompute skipped: pipeline timeout.`,
          latency_ms: 0,
        })
      ),
    ]);

  // 2026-05-12 (E14): keyed by id, derived from ALL_CRITICS. Adding
  // or renaming a critic is now a one-file change in critic-configs.ts;
  // the output schema follows automatically. Previously this block
  // had each critic id literally listed and types.ts mirrored them.
  const critics: Record<string, CriticResult> = {};
  for (const cfg of ALL_CRITICS) {
    const hit = criticResults.find((c) => c.id === cfg.id);
    critics[cfg.id] = hit ?? makeTimedOutCritic(cfg.id, cfg.displayName, cfg.family);
  }

  // Strong critic for the meta.*_input_truncated flag (defaults to
  // the first critic in ALL_CRITICS — currently CRITIC_A / Granite 8B).
  const strongCriticId = ALL_CRITICS[0]?.id;
  const strongCritic = strongCriticId ? critics[strongCriticId] : undefined;

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
      granite_8b_input_truncated: !!strongCritic?.notes.some((n) =>
        n.includes("truncated")
      ),
      critics_unavailable: aggregated.critics_unavailable,
    },
  };
  output.summary_md = renderSummaryMarkdown(output, { answer });
  return output;
}
