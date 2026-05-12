/**
 * Signal 1: SelfCheckGPT-style consistency check.
 *
 * Re-sample the worker N times at high temperature, then check whether
 * the original answer's claims hold up across the alternate samples.
 *
 * Logic:
 *   - Extract claims from the original answer.
 *   - For each claim, classify it against each alternate sample as
 *     premise (entailment / contradiction / neutral).
 *   - A claim is "supported" if at least one alternate entails it.
 *   - A claim is "contradicted" if any alternate contradicts it with
 *     confidence above NLI_CONTRADICTION_THRESHOLD.
 *   - Divergence = (contradicted + unsupported) / total claims.
 *
 * Reference: Manakul et al., 2023, "SelfCheckGPT: Zero-Resource Black-Box
 * Hallucination Detection for Generative Large Language Models."
 */

import {
  CONSISTENCY_TEMPERATURE,
  NLI_CONTRADICTION_THRESHOLD,
  VERBOSE_LOGGING,
} from "../config.js";
import { sampleWorkerN } from "../critics/worker.js";
import { extractClaims } from "../nli/extract-claims.js";
import { classifyEntailment } from "../nli/classifier.js";
import { stripReasoningTraces } from "../sanitize.js";
import type { ConsistencyResult } from "../types.js";

export async function runConsistencyCheck(params: {
  question: string;
  originalAnswer: string;
  numSamples: number;
  /**
   * If supplied, skip the built-in regex extractor and use these instead.
   * The pipeline passes LLM-extracted claims here in deep / deeper modes
   * so NLI and the consistency check see the same claim set.
   */
  preExtractedClaims?: string[];
}): Promise<ConsistencyResult> {
  const start = Date.now();
  const { question, originalAnswer, numSamples, preExtractedClaims } = params;

  if (numSamples < 1) {
    return {
      ran: false,
      samples_generated: 0,
      claims_checked: 0,
      contradicted: [],
      unsupported: [],
      divergence_score: 0,
      latency_ms: Date.now() - start,
      notes: "numSamples < 1; check skipped.",
    };
  }

  // Step 1: Generate alternate samples.
  //
  // Reasoning-model workers (Qwen-QwQ, Phi-4-reasoning, DeepSeek-R1) emit
  // <think>…</think> traces before the answer. Those traces are irrelevant
  // to the claim being checked and — worse — drown out the actual answer
  // text when fed to the NLI classifier as a premise. Strip them now so
  // classifyEntailment() below sees only the model's final answer.
  const samples = await sampleWorkerN({
    question,
    n: numSamples,
    temperature: CONSISTENCY_TEMPERATURE,
  });
  const usableSamples = samples
    .map((s) => ({ ...s, text: stripReasoningTraces(s.text) }))
    .filter((s) => s.text.trim().length > 0);

  if (VERBOSE_LOGGING) {
    console.error(
      `[consistency] ${usableSamples.length}/${numSamples} samples generated`
    );
  }

  if (usableSamples.length === 0) {
    return {
      ran: false,
      samples_generated: 0,
      claims_checked: 0,
      contradicted: [],
      unsupported: [],
      divergence_score: 0,
      latency_ms: Date.now() - start,
      notes:
        "All worker re-samples failed. Check that the worker model is " +
        "still loaded in LM Studio.",
    };
  }

  // Step 2: Extract claims from the original answer (unless the pipeline
  // already did a higher-quality LLM-based extraction).
  const claims = preExtractedClaims ?? extractClaims(originalAnswer);
  if (claims.length === 0) {
    return {
      ran: true,
      samples_generated: usableSamples.length,
      claims_checked: 0,
      contradicted: [],
      unsupported: [],
      divergence_score: 0,
      latency_ms: Date.now() - start,
      notes: "No factual claims detected in the original answer.",
    };
  }

  // Step 3: For each claim, check it against each alternate sample.
  //
  // Parallelisation: the original implementation ran a doubly-nested serial
  // loop with `await classifyEntailment(...)` inside, so up to
  // claims × samples (e.g. 20 × 5 = 100) NLI calls executed strictly
  // sequentially. Each call is a transformers.js / ONNX inference of
  // ~150 ms, and awaits linearise the JS event loop even though ONNX
  // itself can interleave work — so wall time was ~claims × samples × 150 ms.
  //
  // We now fan out every (claim, sample) pair via Promise.all. Per-claim
  // aggregation (highest-confidence contradiction, "supported by at least
  // one") happens deterministically in a second pass over `results`, so the
  // output ordering of `contradicted` and `unsupported` exactly matches the
  // original (claims-array order). No short-circuit / early-exit semantics
  // existed in the old loop — every sample was visited per claim — so
  // nothing is lost by issuing all calls in parallel.
  const contradicted: ConsistencyResult["contradicted"] = [];
  const unsupported: ConsistencyResult["unsupported"] = [];

  type PairResult =
    | { ok: true; label: string; score: number }
    | { ok: false };

  const results: PairResult[][] = await Promise.all(
    claims.map((claim) =>
      Promise.all(
        usableSamples.map(async (sample): Promise<PairResult> => {
          try {
            const { label, score } = await classifyEntailment(
              sample.text,
              claim
            );
            return { ok: true, label, score };
          } catch (err) {
            if (VERBOSE_LOGGING)
              console.error("[consistency] NLI error for claim:", err);
            return { ok: false };
          }
        })
      )
    )
  );

  for (let c = 0; c < claims.length; c++) {
    const claim = claims[c]!;
    const perSample = results[c]!;
    let supportedByAtLeastOne = false;
    let contradictedBy: { sample_index: number; confidence: number } | null =
      null;

    for (let i = 0; i < perSample.length; i++) {
      const r = perSample[i]!;
      if (!r.ok) continue;
      const { label, score } = r;
      if (label.includes("contradict") && score >= NLI_CONTRADICTION_THRESHOLD) {
        if (!contradictedBy || score > contradictedBy.confidence) {
          contradictedBy = { sample_index: i, confidence: score };
        }
      } else if (label.includes("entail")) {
        supportedByAtLeastOne = true;
      }
    }

    if (contradictedBy) {
      contradicted.push({
        claim,
        contradicting_sample_index: contradictedBy.sample_index,
        confidence: contradictedBy.confidence,
      });
    } else if (!supportedByAtLeastOne) {
      unsupported.push({ claim });
    }
  }

  const flagged = contradicted.length + unsupported.length;
  const divergence = claims.length > 0 ? flagged / claims.length : 0;

  return {
    ran: true,
    samples_generated: usableSamples.length,
    claims_checked: claims.length,
    contradicted,
    unsupported,
    divergence_score: Number(divergence.toFixed(3)),
    latency_ms: Date.now() - start,
    notes:
      `Generated ${usableSamples.length} alternate samples at ` +
      `T=${CONSISTENCY_TEMPERATURE}. ` +
      `${contradicted.length} claim(s) contradicted by an alternate; ` +
      `${unsupported.length} claim(s) not supported by any alternate.`,
  };
}
