/**
 * Heuristic factual-claim extraction from assistant answers.
 *
 * A "claim" is a declarative sentence that asserts something verifiable —
 * typically containing a number, a date, a named entity, or a specific
 * factual statement. Questions, imperatives, code, and hedged opinions are
 * excluded.
 *
 * v1 is deliberately simple: regex-based sentence splitting with a few
 * filters. This will miss some claims and catch some non-claims, but it's
 * good enough to feed an NLI classifier without drowning it in noise.
 *
 * [ADAPT] If you're doing a lot of research-heavy verification, replace
 * this with a small trained classifier. spaCy's sentencizer + a custom
 * filter would be a strong v2.
 */

import { NLI_MAX_CLAIMS } from "../config.js";

// 2026-05-20: extended the "next sentence starts with" class to include
// Unicode uppercase (\p{Lu}) so accented capitals (É, Ñ, Ü, ...) that
// legitimately begin Romance / Germanic sentences are caught. We
// deliberately do NOT include \p{Ll} here: doing so would split inside
// abbreviations like "etc. and ..." or "i.e. some ...", which the
// uppercase-only class avoids by design.
const SENTENCE_BOUNDARY = /(?<=[.!?])\s+(?=[A-Z0-9"'\(\[\p{Lu}])/gu;

/**
 * Strip code fences and inline code before claim extraction so we don't
 * try to NLI-check lines of Python.
 */
function stripCode(text: string): string {
  return text
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/`[^`]*`/g, " ");
}

/**
 * Does a sentence look like a factual claim worth checking?
 *
 * 2026-04-18 rewrite: the previous heuristic was far too strict. It required
 * a sentence to contain one of: a 2+digit number, a 1900-2099 year, a
 * citation pattern, or both a proper noun AND one of [is/are/was/were/has/
 * have/had]. On the NLI audit corpus this rejected obvious claims like
 * "The paper found that participants showed improved performance" (no
 * matching features) and "Earth is completely flat" (proper noun Earth at
 * sentence start fails the test).
 *
 * New rule: treat any reasonable declarative sentence as a claim. We
 * only exclude questions, imperatives, and heavy hedges. Downstream NLI
 * classifier is what decides if the claim holds up against evidence; we
 * should feed it more candidates, not gatekeep upstream.
 */
function looksLikeFactualClaim(sentence: string): boolean {
  const s = sentence.trim();

  // Too short to be a meaningful sentence
  if (s.length < 12) return false;
  if (s.split(/\s+/).length < 3) return false;

  // Questions are not assertions
  if (s.endsWith("?")) return false;

  // Imperatives are instructions, not claims
  if (/^(please\s+)?(do|don't|use|try|avoid|consider|remember|note|let's|let\s+us)\b/i.test(s))
    return false;

  // Heavy hedging: the speaker isn't asserting, they're speculating
  if (/^(i think|i believe|in my opinion|perhaps|maybe|it seems|it might|it could)/i.test(s))
    return false;

  // Meta-commentary about the response itself
  if (/^(here is|here are|below is|the following|as an ai|as a language model|i cannot|i'm unable)/i.test(s))
    return false;

  return true;
}

/**
 * Extract up to NLI_MAX_CLAIMS factual-looking sentences from the answer.
 * Returns them in order of appearance.
 */
export function extractClaims(answer: string): string[] {
  const withoutCode = stripCode(answer);
  const sentences = withoutCode
    .split(SENTENCE_BOUNDARY)
    .map((s) => s.trim())
    .filter(Boolean);

  const claims: string[] = [];
  for (const sentence of sentences) {
    if (claims.length >= NLI_MAX_CLAIMS) break;
    if (looksLikeFactualClaim(sentence)) {
      claims.push(sentence);
    }
  }

  return claims;
}
