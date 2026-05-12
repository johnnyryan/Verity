/**
 * Unit tests for the deterministic recompute pass.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
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
