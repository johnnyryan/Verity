/**
 * Unit tests for the generation-confidence classifier.
 *
 * Pure logic — no LM Studio / Ollama calls. Validates band selection across
 * the three axes (weakest token, perplexity, low-confidence density) and the
 * band → verify-depth mapping.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  computeConfidence,
  classifyConfidence,
  assessConfidence,
  renderConfidenceNote,
  modeToCommand,
  type TokenLogprob,
} from "../signals/confidence.js";

// Helper: build a stream of n tokens at a fixed logprob.
function flat(n: number, logprob: number): TokenLogprob[] {
  return Array.from({ length: n }, (_, i) => ({ token: `t${i}`, logprob }));
}

// ────────────────────────────────────────────────────────────────────────
// computeConfidence
// ────────────────────────────────────────────────────────────────────────

test("computeConfidence: empty stream → all-zero metrics", () => {
  const m = computeConfidence([]);
  assert.equal(m.tokens_scored, 0);
  assert.equal(m.perplexity, 0);
});

test("computeConfidence: drops non-finite logprobs", () => {
  const m = computeConfidence([
    { token: "a", logprob: -0.1 },
    { token: "b", logprob: NaN },
    { token: "c", logprob: -Infinity },
    { token: "d", logprob: -0.2 },
  ]);
  assert.equal(m.tokens_scored, 2);
});

test("computeConfidence: finds weakest token + low-confidence count", () => {
  const m = computeConfidence([
    { token: "x", logprob: -0.01 },
    { token: "BAD", logprob: -7.0 },
    { token: "y", logprob: -0.01 },
  ]);
  assert.equal(m.min_logprob_token, "BAD");
  assert.equal(m.min_logprob, -7.0);
  assert.equal(m.low_confidence_tokens, 1); // only -7.0 <= -3.0
});

// ────────────────────────────────────────────────────────────────────────
// classifyConfidence — band selection
// ────────────────────────────────────────────────────────────────────────

test("classify: confident stream → ok, no recommendation", () => {
  const a = classifyConfidence(computeConfidence(flat(12, -0.01)));
  assert.equal(a.band, "ok");
  assert.equal(a.recommended_mode, null);
  assert.equal(renderConfidenceNote(a), "");
});

test("classify: one very-weak token → very_low → deeper (local axis)", () => {
  const tokens = [...flat(9, -0.01), { token: "ZÜRICH", logprob: -7.0 }];
  const a = assessConfidence(tokens);
  assert.equal(a.band, "very_low");
  assert.equal(a.recommended_mode, "deeper");
  assert.match(renderConfidenceNote(a), /\/verifydeeper/);
});

test("classify: diffuse uncertainty → low → deep (perplexity axis)", () => {
  // mean logprob -1.5 → perplexity ≈ 4.48 (>= 4.0 low, < 8.0 very_low);
  // no single token below -4.5, none below -3.0 so density stays ok.
  const a = assessConfidence(flat(10, -1.5));
  assert.equal(a.band, "low");
  assert.equal(a.recommended_mode, "deep");
  assert.match(renderConfidenceNote(a), /\/verifydeep/);
});

test("classify: borderline token → mild → standard /verify", () => {
  const tokens = [...flat(9, -0.01), { token: "maybe", logprob: -3.2 }];
  const a = assessConfidence(tokens);
  assert.equal(a.band, "mild");
  assert.equal(a.recommended_mode, "standard");
  assert.match(renderConfidenceNote(a), /\/verify\b/);
});

test("classify: no tokens → ok with explanatory reason", () => {
  const a = classifyConfidence(computeConfidence([]));
  assert.equal(a.band, "ok");
  assert.match(a.reason, /No usable logprobs/);
});

// ────────────────────────────────────────────────────────────────────────
// modeToCommand
// ────────────────────────────────────────────────────────────────────────

test("modeToCommand maps every depth", () => {
  assert.equal(modeToCommand("standard"), "/verify");
  assert.equal(modeToCommand("deep"), "/verifydeep");
  assert.equal(modeToCommand("deeper"), "/verifydeeper");
});
