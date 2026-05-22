/**
 * Generation-confidence classifier.
 *
 * Turns a stream of per-token logprobs (from /v1/responses, a forward-pass
 * rescore, or any logprob-bearing generation) into a confidence band and a
 * recommended escalation depth. This is the shared core of the
 * "examine logprobs → report low confidence → recommend /verify" feature:
 * it is deliberately free of any I/O so it can be driven from the perplexity
 * signal, a generation front-door tool, or a proxy without change.
 *
 * Design notes:
 *   - An LLM cannot read its own logprobs. They are a property of a specific
 *     generation, available only from the endpoint that produced it. So the
 *     answer must be generated through a logprobs-capable path
 *     (LM Studio /v1/responses) and the numbers fed here.
 *   - We score on three axes and take the WORST band any axis reports:
 *       local  : the single weakest token (one bad name/number/date)
 *       global : whole-answer perplexity (diffuse uncertainty)
 *       density: fraction of low-confidence tokens (sustained guessing)
 *     A single axis tripping is enough — the failure modes are independent.
 */

import {
  PERPLEXITY_LOW_CONFIDENCE_LOGPROB,
  CONFIDENCE_MILD_MIN_LOGPROB,
  CONFIDENCE_LOW_MIN_LOGPROB,
  CONFIDENCE_VERYLOW_MIN_LOGPROB,
  CONFIDENCE_MILD_PERPLEXITY,
  CONFIDENCE_LOW_PERPLEXITY,
  CONFIDENCE_VERYLOW_PERPLEXITY,
  CONFIDENCE_MILD_RATIO,
  CONFIDENCE_LOW_RATIO,
  CONFIDENCE_VERYLOW_RATIO,
} from "../config.js";
import type {
  ConfidenceAssessment,
  ConfidenceBand,
  ConfidenceMetrics,
  VerifyMode,
} from "../types.js";

export interface TokenLogprob {
  token: string;
  logprob: number;
}

/** Band → user-facing slash command. */
export function modeToCommand(mode: VerifyMode): string {
  switch (mode) {
    case "standard":
      return "/verify";
    case "deep":
      return "/verifydeep";
    case "deeper":
      return "/verifydeeper";
  }
}

const BAND_TO_MODE: Record<ConfidenceBand, VerifyMode | null> = {
  ok: null,
  mild: "standard",
  low: "deep",
  very_low: "deeper",
};

// User-facing word for each band. We deliberately speak of "uncertainty",
// NOT "confidence" or "correctness": high token uncertainty often just means
// rare-but-correct wording (proper nouns, code, jargon), and fluent
// hallucinations carry LOW uncertainty. So this signal is a nudge to look,
// never a verdict on whether the answer is right. (Panel guidance 2026-05-22.)
const BAND_TO_UNCERTAINTY_LABEL: Record<ConfidenceBand, string> = {
  ok: "Low",
  mild: "Some",
  low: "Elevated",
  very_low: "High",
};

/** logprob → probability percentage string, e.g. -3.0 → "5.0%". */
function pct(logprob: number): string {
  return `${(Math.exp(logprob) * 100).toFixed(1)}%`;
}

/**
 * Reduce a token-logprob stream to confidence statistics. Tokens with a
 * non-finite logprob (some endpoints emit null/-Infinity for the first
 * token) are dropped.
 */
export function computeConfidence(tokens: TokenLogprob[]): ConfidenceMetrics {
  const valid = tokens.filter(
    (t) => typeof t.logprob === "number" && Number.isFinite(t.logprob)
  );

  if (valid.length === 0) {
    return {
      tokens_scored: 0,
      mean_logprob: 0,
      perplexity: 0,
      min_logprob: 0,
      min_logprob_token: "",
      low_confidence_tokens: 0,
      low_confidence_ratio: 0,
    };
  }

  let sum = 0;
  let min = Infinity;
  let minToken = "";
  let lowCount = 0;
  for (const t of valid) {
    sum += t.logprob;
    if (t.logprob < min) {
      min = t.logprob;
      minToken = t.token;
    }
    if (t.logprob <= PERPLEXITY_LOW_CONFIDENCE_LOGPROB) lowCount++;
  }

  const mean = sum / valid.length;
  return {
    tokens_scored: valid.length,
    mean_logprob: Number(mean.toFixed(3)),
    perplexity: Number(Math.exp(-mean).toFixed(3)),
    min_logprob: Number(min.toFixed(3)),
    min_logprob_token: minToken,
    low_confidence_tokens: lowCount,
    low_confidence_ratio: Number((lowCount / valid.length).toFixed(3)),
  };
}

/**
 * Classify confidence metrics into a band + escalation recommendation.
 * The worst band reported by any of the three axes wins.
 */
export function classifyConfidence(
  metrics: ConfidenceMetrics
): ConfidenceAssessment {
  // No usable tokens → cannot assess; treat as ok (the caller decides what
  // "couldn't measure" means; we don't manufacture a warning from nothing).
  if (metrics.tokens_scored === 0) {
    return {
      band: "ok",
      recommended_mode: null,
      reason: "No usable logprobs to assess confidence.",
      metrics,
    };
  }

  const { min_logprob, perplexity, low_confidence_ratio } = metrics;

  // Evaluate each axis to its own band, then take the worst.
  const localBand: ConfidenceBand =
    min_logprob <= CONFIDENCE_VERYLOW_MIN_LOGPROB
      ? "very_low"
      : min_logprob <= CONFIDENCE_LOW_MIN_LOGPROB
        ? "low"
        : min_logprob <= CONFIDENCE_MILD_MIN_LOGPROB
          ? "mild"
          : "ok";

  const globalBand: ConfidenceBand =
    perplexity >= CONFIDENCE_VERYLOW_PERPLEXITY
      ? "very_low"
      : perplexity >= CONFIDENCE_LOW_PERPLEXITY
        ? "low"
        : perplexity >= CONFIDENCE_MILD_PERPLEXITY
          ? "mild"
          : "ok";

  const densityBand: ConfidenceBand =
    low_confidence_ratio >= CONFIDENCE_VERYLOW_RATIO
      ? "very_low"
      : low_confidence_ratio >= CONFIDENCE_LOW_RATIO
        ? "low"
        : low_confidence_ratio >= CONFIDENCE_MILD_RATIO
          ? "mild"
          : "ok";

  const rank: Record<ConfidenceBand, number> = {
    ok: 0,
    mild: 1,
    low: 2,
    very_low: 3,
  };
  const axes: Array<{ name: string; band: ConfidenceBand }> = [
    { name: "weakest-token", band: localBand },
    { name: "perplexity", band: globalBand },
    { name: "low-conf-density", band: densityBand },
  ];
  const worst = axes.reduce((a, b) => (rank[b.band] > rank[a.band] ? b : a));
  const band = worst.band;

  if (band === "ok") {
    return {
      band,
      recommended_mode: null,
      reason: `Low model uncertainty (perplexity ${perplexity}, weakest token ${pct(
        min_logprob
      )}).`,
      metrics,
    };
  }

  // Build a concise reason naming the driving axis.
  let driver: string;
  switch (worst.name) {
    case "weakest-token":
      driver = `weakest token "${metrics.min_logprob_token}" at ${pct(
        min_logprob
      )}`;
      break;
    case "perplexity":
      driver = `answer perplexity ${perplexity}`;
      break;
    default:
      driver = `${metrics.low_confidence_tokens}/${metrics.tokens_scored} tokens low-confidence`;
  }

  return {
    band,
    recommended_mode: BAND_TO_MODE[band],
    reason: `${BAND_TO_UNCERTAINTY_LABEL[band]} model uncertainty: ${driver}.`,
    metrics,
  };
}

/**
 * Render a one-line, paste-ready note for the user. Empty string when the
 * answer is confident (nothing to say).
 */
export function renderConfidenceNote(a: ConfidenceAssessment): string {
  if (a.band === "ok" || !a.recommended_mode) return "";
  const cmd = modeToCommand(a.recommended_mode);
  const icon = a.band === "very_low" ? "⛔" : a.band === "low" ? "⚠️" : "ℹ️";
  // A nudge, not a verdict: this reports the model's own uncertainty, which
  // runs high on rare-but-correct wording and LOW on fluent hallucinations.
  return `${icon} ${a.reason} Possible guessing rather than a known error. Consider \`${cmd}\`.`;
}

/** Convenience: assess straight from a token-logprob stream. */
export function assessConfidence(tokens: TokenLogprob[]): ConfidenceAssessment {
  return classifyConfidence(computeConfidence(tokens));
}
