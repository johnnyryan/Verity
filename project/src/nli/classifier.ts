/**
 * NLI (Natural Language Inference) classifier wrapper.
 *
 * Uses @huggingface/transformers (the JS port of HF Transformers) to run
 * DeBERTa-v3-large-mnli via ONNX Runtime on CPU. No Python dependency.
 *
 * Two modes:
 *   1. With prior_context: classify whether each extracted claim is
 *      ENTAILED by the prior_context (premise). Flag CONTRADICTIONs
 *      and UNSUPPORTED (non-entailment) claims.
 *
 *   2. Without prior_context: check for intra-answer contradictions —
 *      look for pairs of claims in the same answer that disagree with
 *      each other.
 *
 * The model is loaded lazily on first use and kept in memory for the
 * process lifetime.
 */

import { pipeline, env, type TextClassificationPipeline } from "@huggingface/transformers";

import {
  NLI_MODEL_ID,
  NLI_DEVICE,
  NLI_CONTRADICTION_THRESHOLD,
  NLI_IMPL,
  NLI_REQUIRE_CONTEXT,
  VERBOSE_LOGGING,
} from "../config.js";
import type { NliResult } from "../types.js";
import { extractClaims } from "./extract-claims.js";
import { runLlmClaimCheck } from "./claim-check-llm.js";

// [ADAPT] If you want to cache ONNX weights somewhere specific on the PC
// (e.g. to avoid re-downloading), set env.cacheDir here before the first
// call. By default it goes to the HuggingFace cache directory.
//
// env.cacheDir = "C:/ai-models/hf-cache";

// Disable telemetry from the JS port.
env.allowRemoteModels = true;
env.allowLocalModels = true;

// Lazily-initialized classifier. First call downloads the model (~1 GB),
// subsequent calls use the in-memory instance.
//
// 2026-05-12 (E3): we cache the promise only on success. The previous
// implementation cached the promise unconditionally, including a
// rejected one. A single transient failure (cold-load OOM, download
// interrupted, model file corrupt) would poison every subsequent
// call with the cached rejection. Now: on failure we clear the cache
// so the next caller gets a clean retry.
let classifierPromise: Promise<TextClassificationPipeline> | null = null;

function getClassifier(): Promise<TextClassificationPipeline> {
  if (classifierPromise) return classifierPromise;
  if (VERBOSE_LOGGING) {
    console.error(`[NLI] Loading ${NLI_MODEL_ID} on device=${NLI_DEVICE}`);
  }
  // "text-classification" with an NLI model returns entailment /
  // contradiction / neutral labels when given a "premise [SEP] hypothesis"
  // pair. Some models prefer a different format — see the model card if
  // you swap NLI_MODEL_ID.
  const pending = pipeline("text-classification", NLI_MODEL_ID, {
    device: NLI_DEVICE,
  });
  classifierPromise = pending;
  pending.catch(() => {
    // Cache invalidation: only clear if THIS pending promise is the
    // one in the cache slot. A successful concurrent load would have
    // replaced classifierPromise; we must not stamp it back to null.
    if (classifierPromise === pending) classifierPromise = null;
  });
  return pending;
}

/**
 * Public: run an NLI check on a premise/hypothesis pair.
 * Returns label + score from the classifier.
 * Used by both this module and signals/consistency.ts.
 */
export async function classifyEntailment(
  premise: string,
  hypothesis: string
): Promise<{ label: string; score: number }> {
  return classifyPair(premise, hypothesis);
}

/**
 * Boot-time warmup. Triggers the lazy ONNX model load and runs ONE
 * no-op classification so the runtime is fully JIT-compiled before
 * the first /verify request arrives.
 *
 * Fault-tolerant: any failure (download error, model corrupt, OOM) is
 * logged and swallowed — boot must not crash because warmup didn't go.
 * The next real request will retry the load via getClassifier().
 */
export async function warmupClassifier(): Promise<void> {
  try {
    await classifyPair("The sky is blue.", "The sky is colored.");
  } catch (err) {
    console.error("[NLI] warmup failed (non-fatal):", err);
  }
}

/**
 * Internal pair-classification helper. Returns the top-1 label and score.
 *
 * 2026-04-18: previously used implicit top_k=1 from transformers.js. That
 * meant a claim where contradiction scored (e.g.) 0.49 but neutral scored
 * 0.51 would be returned as "neutral" with 0.51, and checkEntailment would
 * never see that contradiction was close. classifyPairAll below returns
 * every label's score so callers can examine contradiction specifically.
 */
async function classifyPair(
  premise: string,
  hypothesis: string
): Promise<{ label: string; score: number }> {
  const { top } = await classifyPairAll(premise, hypothesis);
  return top;
}

/**
 * Return the top label AND the full label→score map for a pair. Used by
 * the entailment/consistency checks so contradiction can be detected even
 * when "neutral" wins the top-1 race by a small margin.
 */
async function classifyPairAll(
  premise: string,
  hypothesis: string
): Promise<{
  top: { label: string; score: number };
  all: Record<string, number>;
}> {
  const classifier = await getClassifier();

  // top_k: null asks transformers.js to return every label's score.
  // The library accepts null at runtime but the d.ts only types it as
  // `number | undefined`; cast to satisfy the compiler without altering
  // behaviour.
  const result = await classifier(
    [premise, hypothesis],
    { top_k: null as unknown as number | undefined }
  );

  // Result shape for pairwise text-classification with top_k:null is
  // [ [ {label, score}, {label, score}, {label, score} ] ] — batch of 1.
  // Normalize to a flat array of {label, score}.
  let items: Array<{ label: string; score: number }> = [];
  if (Array.isArray(result)) {
    if (Array.isArray(result[0])) {
      items = result[0] as Array<{ label: string; score: number }>;
    } else if (result.length > 0 && typeof result[0] === "object") {
      items = result as Array<{ label: string; score: number }>;
    }
  }

  const all: Record<string, number> = {};
  let topLabel = "neutral";
  let topScore = 0;
  for (const item of items) {
    const l = String(item?.label ?? "").toLowerCase();
    const s = Number(item?.score ?? 0);
    if (l) {
      all[l] = s;
      if (s > topScore) {
        topScore = s;
        topLabel = l;
      }
    }
  }
  return { top: { label: topLabel, score: topScore }, all };
}

/**
 * Map a raw NLI label string to a coarse bucket.
 *
 * Label names differ by model: "contradiction", "CONTRADICTION",
 * "contradict", "CONTRADICT", "ENTAILMENT", "entailment", "entail",
 * "neutral", and even "LABEL_0/1/2" in some cases. We match loosely
 * on the substrings "contradict" and "entail"; anything else falls
 * through to "neutral".
 *
 * Exported so consistency.ts can use the same dispatch rules as the
 * classifier — duplicating `label.includes("contradict")` etc. was
 * making two files own NLI label semantics.
 */
export function classifyNliLabel(
  label: string
): "contradict" | "entail" | "neutral" {
  const l = label.toLowerCase();
  if (l.includes("contradict")) return "contradict";
  if (l.includes("entail")) return "entail";
  return "neutral";
}

/**
 * Extract the contradiction probability from the all-label map.
 */
function contradictionScore(all: Record<string, number>): number {
  for (const [label, score] of Object.entries(all)) {
    if (classifyNliLabel(label) === "contradict") return score;
  }
  return 0;
}

/**
 * Extract the entailment probability from the all-label map.
 */
function entailmentScore(all: Record<string, number>): number {
  for (const [label, score] of Object.entries(all)) {
    if (classifyNliLabel(label) === "entail") return score;
  }
  return 0;
}

/**
 * Entailment check: is each claim supported by the premise?
 *
 * 2026-05-12 (E5): claim classification now fans out in parallel via
 * Promise.all. The consistency signal already parallelised
 * claim × sample pairs; doing this serially here was the dominant
 * cost on premise-bearing /verify calls (especially `with context`
 * where NLI_MAX_CLAIMS can be 14+). The ONNX runtime serialises at
 * the model level anyway, but JS-side awaits between turns add up.
 * On a 14-claim load this typically halves NLI wall-clock.
 */
async function checkEntailment(
  premise: string,
  claims: string[]
): Promise<NliResult> {
  const contradictions: NliResult["contradictions"] = [];
  const unsupported: NliResult["unsupported"] = [];

  const perClaim = await Promise.all(
    claims.map(async (claim) => {
      try {
        const { all } = await classifyPairAll(premise, claim);
        return {
          claim,
          cScore: contradictionScore(all),
          eScore: entailmentScore(all),
        };
      } catch (err) {
        if (VERBOSE_LOGGING) console.error("[NLI] classifyPair error:", err);
        return null;
      }
    })
  );

  for (const r of perClaim) {
    if (!r) continue;
    if (r.cScore >= NLI_CONTRADICTION_THRESHOLD) {
      contradictions.push({
        claim: r.claim,
        premise_snippet: premise.slice(0, 200),
        confidence: r.cScore,
      });
    } else if (r.eScore < NLI_CONTRADICTION_THRESHOLD) {
      // Not confidently entailed → unsupported.
      unsupported.push({ claim: r.claim });
    }
  }

  return {
    ran: true,
    claims_checked: claims.length,
    contradictions,
    unsupported,
    notes:
      `Entailment checked against prior_context (${premise.length} chars). ` +
      `${contradictions.length} contradiction(s), ${unsupported.length} unsupported.`,
  };
}

/**
 * Intra-answer consistency check: look for pairs of claims that
 * contradict each other within the same answer.
 *
 * Quadratic in the number of claims, so we cap at NLI_MAX_CLAIMS.
 */
async function checkIntraAnswerConsistency(
  claims: string[]
): Promise<NliResult> {
  const contradictions: NliResult["contradictions"] = [];

  for (let i = 0; i < claims.length; i++) {
    for (let j = i + 1; j < claims.length; j++) {
      try {
        const { all } = await classifyPairAll(claims[i]!, claims[j]!);
        const cScore = contradictionScore(all);
        if (cScore >= NLI_CONTRADICTION_THRESHOLD) {
          contradictions.push({
            claim: claims[j]!,
            premise_snippet: claims[i]!.slice(0, 200),
            confidence: cScore,
          });
        }
      } catch {
        // Skip and keep going.
      }
    }
  }

  return {
    ran: true,
    claims_checked: claims.length,
    contradictions,
    unsupported: [],
    notes:
      `No prior_context supplied; checked ${claims.length} claims pairwise ` +
      `for intra-answer contradictions.`,
  };
}

/**
 * Main entry point. Decides which check to run based on whether
 * prior_context is available.
 *
 * @param preExtractedClaims  If supplied, skip the built-in regex
 *                            extractor and use these instead. The pipeline
 *                            passes LLM-extracted claims here in deep /
 *                            deeper modes so NLI and the consistency check
 *                            see the same claim set.
 */
export async function runNliCheck(
  answer: string,
  priorContext: string | undefined,
  preExtractedClaims?: string[]
): Promise<NliResult> {
  // Option "off": NLI disabled entirely.
  if (NLI_IMPL === "off") {
    return {
      ran: false,
      claims_checked: 0,
      contradictions: [],
      unsupported: [],
      notes: "NLI disabled (NLI_IMPL=off).",
    };
  }

  // Option "llm": route to the Granite-based claim checker.
  if (NLI_IMPL === "llm") {
    return runLlmClaimCheck(answer, priorContext, preExtractedClaims);
  }

  const claims = preExtractedClaims ?? extractClaims(answer);

  if (claims.length === 0) {
    return {
      ran: true,
      claims_checked: 0,
      contradictions: [],
      unsupported: [],
      notes: "No factual claims detected in the answer.",
    };
  }

  const hasContext = !!(priorContext && priorContext.trim().length > 0);

  // Option A: NLI_REQUIRE_CONTEXT=1 skips pairwise intra-answer mode, which
  // has empirically produced zero signal and only adds latency.
  if (!hasContext && NLI_REQUIRE_CONTEXT) {
    return {
      ran: true,
      claims_checked: 0,
      contradictions: [],
      unsupported: [],
      notes:
        "NLI skipped: no prior_context provided and NLI_REQUIRE_CONTEXT=1 " +
        "(pairwise intra-answer mode disabled).",
    };
  }

  if (hasContext) {
    return checkEntailment(priorContext!, claims);
  }
  return checkIntraAnswerConsistency(claims);
}
