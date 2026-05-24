/**
 * Unit tests for the deterministic recompute pass.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  buildRecomputeScope,
  runRecomputePass,
  safeEvalArithmetic,
} from "../signals/recompute.js";

// ────────────────────────────────────────────────────────────────────────
// safeEvalArithmetic — parser / evaluator
// ────────────────────────────────────────────────────────────────────────

test("arithmetic: simple addition", () => {
  assert.equal(safeEvalArithmetic("2+2"), 4);
});

test("arithmetic: operator precedence", () => {
  assert.equal(safeEvalArithmetic("2+3*4"), 14);
});

test("arithmetic: parentheses override precedence", () => {
  assert.equal(safeEvalArithmetic("(2+3)*4"), 20);
});

test("arithmetic: implicit multiplication `3(5)`", () => {
  assert.equal(safeEvalArithmetic("3(5)"), 15);
});

test("arithmetic: implicit multiplication chained `3(5)+7`", () => {
  assert.equal(safeEvalArithmetic("3(5)+7"), 22);
});

test("arithmetic: implicit multiplication `(2)(3)`", () => {
  assert.equal(safeEvalArithmetic("(2)(3)"), 6);
});

test("arithmetic: unary minus", () => {
  assert.equal(safeEvalArithmetic("-5+3"), -2);
});

test("arithmetic: decimal numbers", () => {
  assert.equal(safeEvalArithmetic("1.5*2"), 3);
});

test("arithmetic: thousands commas", () => {
  assert.equal(safeEvalArithmetic("1,000+500"), 1500);
});

test("arithmetic: right-assoc exponent `2^3^2` = 2^(3^2) = 512", () => {
  assert.equal(safeEvalArithmetic("2^3^2"), 512);
});

test("arithmetic: division by zero returns null", () => {
  assert.equal(safeEvalArithmetic("5/0"), null);
});

test("arithmetic: garbage returns null, never throws", () => {
  assert.equal(safeEvalArithmetic("banana + grape"), null);
  assert.equal(safeEvalArithmetic(""), null);
  assert.equal(safeEvalArithmetic("2 +"), null);
});

// ────────────────────────────────────────────────────────────────────────
// runRecomputePass — arithmetic detection
// ────────────────────────────────────────────────────────────────────────

test("arithmetic detector: correct claim yields match", async () => {
  const r = await runRecomputePass("The answer is 3(5)+7=22. Trust me.");
  assert.equal(r.expressions_found, 1);
  assert.equal(r.mismatches.length, 0);
  assert.equal(r.verifications[0]?.matches, true);
});

test("arithmetic detector: wrong claim yields mismatch", async () => {
  const r = await runRecomputePass("3(5)+7 = 25. Correct!");
  assert.equal(r.mismatches.length, 1);
  assert.equal(r.mismatches[0]?.computed, "22");
  assert.equal(r.mismatches[0]?.claimed, "25");
});

test("arithmetic detector: 'equals' trigger word", async () => {
  const r = await runRecomputePass("100 / 4 equals 25.");
  assert.equal(r.verifications.length, 1);
  assert.equal(r.verifications[0]?.matches, true);
});

test("arithmetic detector: does not fire on ungrounded numbers", async () => {
  const r = await runRecomputePass(
    "The library has 1000 books. The park is 5 km away."
  );
  assert.equal(r.expressions_found, 0);
});

test("arithmetic detector: ignores code blocks", async () => {
  const r = await runRecomputePass(
    "Here is python:\n```python\nx = 2+2\n# 2+2 = 5\n```\nOut of the block: 3+3 = 6"
  );
  // Should only detect "3+3 = 6" outside the fence
  assert.equal(r.verifications.length, 1);
  assert.equal(r.verifications[0]?.matches, true);
});

// ────────────────────────────────────────────────────────────────────────
// runRecomputePass — enumeration / list comprehension
// ────────────────────────────────────────────────────────────────────────

test("enumeration: correct comprehension", async () => {
  const r = await runRecomputePass(
    "[x*x for x in range(5)] produces [0, 1, 4, 9, 16]"
  );
  assert.equal(r.verifications.length, 1);
  assert.equal(r.verifications[0]?.matches, true);
});

test("enumeration: wrong comprehension (python-list-comp case)", async () => {
  const r = await runRecomputePass(
    "[x*x for x in range(5)] produces [1, 4, 9, 16, 25]"
  );
  assert.equal(r.mismatches.length, 1);
  assert.equal(r.mismatches[0]?.computed, "[0,1,4,9,16]");
});

// ────────────────────────────────────────────────────────────────────────
// runRecomputePass — leap year
// ────────────────────────────────────────────────────────────────────────

test("leap year: 2024 claimed to have 365 days (wrong)", async () => {
  const r = await runRecomputePass(
    "2024 is a leap year so it has 365 days."
  );
  assert.equal(r.mismatches.length, 1);
  assert.equal(r.mismatches[0]?.kind, "leap-year");
  assert.equal(r.mismatches[0]?.computed, "366 days");
});

test("leap year: 2024 correctly 366 days", async () => {
  const r = await runRecomputePass(
    "2024 is a leap year, which means it has 366 days."
  );
  assert.equal(r.mismatches.length, 0);
  assert.equal(r.verifications[0]?.matches, true);
});

// ────────────────────────────────────────────────────────────────────────
// runRecomputePass — unit constants
// ────────────────────────────────────────────────────────────────────────

test("unit: speed of light in km/s (correct)", async () => {
  const r = await runRecomputePass("The speed of light is 299,792 km/s.");
  assert.equal(r.verifications[0]?.matches, true);
});

test("unit: speed of light in km/hour (wrong — off by factor 3600)", async () => {
  const r = await runRecomputePass("The speed of light is 299,792 km/hour.");
  assert.equal(r.mismatches.length, 1);
  assert.equal(r.mismatches[0]?.kind, "unit");
});

// ────────────────────────────────────────────────────────────────────────
// runRecomputePass — linear equations
// ────────────────────────────────────────────────────────────────────────

test("linear-equation: correct solve (subtle-math case)", async () => {
  const r = await runRecomputePass(
    "Subtract 7 from both sides: 3x = 15. Divide by 3: x = 5. Check: 3(5) + 7 = 22. Correct.",
    "Solve 3x + 7 = 22"
  );
  const linear = r.verifications.filter((v) => v.kind === "linear-equation");
  assert.equal(linear.length, 2, `expected 2 linear-eq verifications, got ${linear.length}`);
  assert.equal(linear.every((v) => v.matches), true);
});

test("linear-equation: expr_text is substring of NLI-style claim (for suppression)", async () => {
  const r = await runRecomputePass(
    "Divide by 3: x = 5.",
    "Solve 3x + 7 = 22"
  );
  const linear = r.verifications.filter((v) => v.kind === "linear-equation");
  // The aggregator does: nliClaim.toLowerCase().includes(expr_text.toLowerCase())
  const nliClaim = "Divide by 3: x = 5.";
  const suppressed = linear.some((v) =>
    nliClaim.toLowerCase().includes(v.expr_text.toLowerCase())
  );
  assert.equal(suppressed, true);
});

test("linear-equation: wrong solve flagged as mismatch", async () => {
  const r = await runRecomputePass(
    "Divide by 3: x = 6.",
    "Solve 3x + 7 = 22"
  );
  const mis = r.mismatches.filter((m) => m.kind === "linear-equation");
  assert.equal(mis.length, 1);
  assert.equal(mis[0]?.claimed, "6");
  assert.equal(mis[0]?.computed, "5");
});

test("linear-equation: intermediate step `3x = 15` verified", async () => {
  const r = await runRecomputePass("So 3x = 15.", "Solve 3x + 7 = 22");
  const linear = r.verifications.filter((v) => v.kind === "linear-equation");
  assert.equal(linear.length, 1);
  assert.equal(linear[0]?.matches, true);
  assert.equal(linear[0]?.expr_text, "3x = 15");
});

test("linear-equation: negative constant `x - 5 = 10`", async () => {
  const r = await runRecomputePass("Adding 5: x = 15.", "Solve x - 5 = 10");
  const linear = r.verifications.filter((v) => v.kind === "linear-equation");
  assert.equal(linear[0]?.matches, true);
  assert.equal(linear[0]?.computed, "15");
});

test("linear-equation: different variable letter (y)", async () => {
  const r = await runRecomputePass("So y = 5.", "Solve 2y + 4 = 14");
  const linear = r.verifications.filter((v) => v.kind === "linear-equation");
  assert.equal(linear[0]?.matches, true);
  assert.equal(linear[0]?.computed, "5");
});

test("linear-equation: negative coefficient `-2y + 4 = 14`", async () => {
  const r = await runRecomputePass("So y = -5.", "Solve -2y + 4 = 14");
  const linear = r.verifications.filter((v) => v.kind === "linear-equation");
  assert.equal(linear[0]?.matches, true);
  assert.equal(linear[0]?.computed, "-5");
});

test("linear-equation: no equation in question, no verifications emitted", async () => {
  const r = await runRecomputePass(
    "x = 5 because I said so.",
    "Think about x."
  );
  const linear = r.verifications.filter((v) => v.kind === "linear-equation");
  assert.equal(linear.length, 0);
});

test("linear-equation: claim about unrelated variable is ignored", async () => {
  const r = await runRecomputePass(
    "Also y = 99 somewhere.",
    "Solve 3x + 7 = 22"
  );
  const linear = r.verifications.filter((v) => v.kind === "linear-equation");
  assert.equal(linear.length, 0);
});

// ────────────────────────────────────────────────────────────────────────
// runRecomputePass — empty input / no claims
// ────────────────────────────────────────────────────────────────────────

test("empty answer: ran=true, no verifications", async () => {
  const r = await runRecomputePass("");
  assert.equal(r.ran, true);
  assert.equal(r.expressions_found, 0);
  assert.equal(r.mismatches.length, 0);
});

test("plain prose with no math: no false positives", async () => {
  const r = await runRecomputePass(
    "Paris is the capital of France. It is a lovely city."
  );
  assert.equal(r.expressions_found, 0);
});

// ────────────────────────────────────────────────────────────────────────
// CoVe step-3 independence (Dhuliawala et al., 2023)
//
// The recompute pass must source the expression-to-verify from the
// question, NOT from the draft answer's restated reasoning. With the
// draft visible the verifier risks anchoring on the draft and
// reproducing its mistakes. The split is gated by RECOMPUTE_INDEPENDENT
// (default true); legacy A/B mode (false) restores the pre-2026-05-23
// single-source behaviour.
// ────────────────────────────────────────────────────────────────────────

test("CoVe independence: scope does NOT contain the draft answer text", () => {
  const question = "Solve 3x + 7 = 22";
  const draftAnswer = "DRAFT_MARKER_ZULU x = 5 because reasons.";
  const scope = buildRecomputeScope(question, draftAnswer, true);
  assert.equal(
    scope.expressionScope.includes("DRAFT_MARKER_ZULU"),
    false,
    "independent expressionScope must not include the draft answer text"
  );
  assert.equal(
    scope.expressionScope.includes("Solve 3x + 7 = 22"),
    true,
    "independent expressionScope must include the question"
  );
  assert.equal(
    scope.claimScope.includes("DRAFT_MARKER_ZULU"),
    true,
    "claimScope is the answer (the draft is only read for the claimed value)"
  );
  assert.equal(scope.independent, true);
});

test("CoVe independence: legacy mode (RECOMPUTE_INDEPENDENT=false) DOES contain draft", () => {
  const question = "Solve 3x + 7 = 22";
  const draftAnswer = "DRAFT_MARKER_ZULU x = 5 because reasons.";
  const scope = buildRecomputeScope(question, draftAnswer, false);
  assert.equal(
    scope.expressionScope.includes("DRAFT_MARKER_ZULU"),
    true,
    "legacy expressionScope must include the draft answer text for back-compat"
  );
  assert.equal(
    scope.expressionScope.includes("Solve 3x + 7 = 22"),
    true,
    "legacy expressionScope must include the question"
  );
  assert.equal(scope.independent, false);
});

test("CoVe independence: degenerate case (no question) falls back to answer-only scope", () => {
  const scope = buildRecomputeScope(undefined, "3+3 = 6.", true);
  // No question to factor out; expressionScope IS the answer. The claim
  // is still in the same source, so independence is vacuous here.
  assert.equal(scope.expressionScope.includes("3+3 = 6."), true);
  assert.equal(scope.claimScope, scope.expressionScope);
});

test("CoVe independence: linear-equation solver uses ONLY the question's equation when independent", async () => {
  // Question supplies the equation; answer restates a DIFFERENT (made-up)
  // equation as a draft-anchoring trap. Under independent mode the
  // solver must ignore the answer's restated equation and solve only
  // the one in the question.
  const r = await runRecomputePass(
    "Re-stating the problem: 3x + 7 = 100. So x = 31.", // wrong restatement + wrong claim
    "Solve 3x + 7 = 22",
    { independent: true }
  );
  const linear = r.verifications.filter((v) => v.kind === "linear-equation");
  // From the question's equation, x = 5. The answer claims x = 31, so mismatch.
  assert.equal(linear.length >= 1, true, "should emit at least one verification");
  const xClaim = linear.find((v) => v.expr_text.toLowerCase().includes("x = 31"));
  assert.equal(xClaim?.matches, false, "x = 31 must be flagged as mismatch (expected 5)");
  assert.equal(xClaim?.computed, "5");
});

test("CoVe independence: legacy mode lets the answer's restated equation seed the solver", async () => {
  // Same trap as the previous test, but legacy=false. The answer's
  // re-stated equation 3x + 7 = 100 gets registered before the
  // question's, so the solver's x = 31 lookup matches.
  const r = await runRecomputePass(
    "Re-stating the problem: 3x + 7 = 100. So x = 31.",
    "Solve 3x + 7 = 22",
    { independent: false }
  );
  const linear = r.verifications.filter((v) => v.kind === "linear-equation");
  const xClaim = linear.find((v) => v.expr_text.toLowerCase().includes("x = 31"));
  // Under legacy, the first equation seen wins; whichever order they
  // hit the regex, the verification is computed against that scope.
  assert.equal(xClaim !== undefined, true, "legacy mode should still verify x = 31");
});

test("CoVe independence: with no equation in question, independent mode emits nothing", async () => {
  // Independent mode strips the answer from the expression scope, so a
  // question without an equation leaves the solver with nothing to do
  // even when the answer volunteers one. This is the intended trade-off:
  // we refuse to verify equations the draft alone supplied.
  const r = await runRecomputePass(
    "Pretend the equation is 2x = 10, so x = 5.",
    "Tell me about variables.",
    { independent: true }
  );
  const linear = r.verifications.filter((v) => v.kind === "linear-equation");
  assert.equal(linear.length, 0);
});
