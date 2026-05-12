/**
 * Unit tests for the critic JSON parser.
 *
 * Run with: npm test
 * (compiled from src/__tests__/*.test.ts → dist/__tests__/*.test.js, then
 * executed via node --test)
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { parseCriticJson } from "../critics/parse.js";

test("returns null for empty input", () => {
  assert.equal(parseCriticJson(""), null);
  assert.equal(parseCriticJson("   \n  "), null);
});

test("returns null for non-string input", () => {
  // @ts-expect-error — deliberately malformed for this test
  assert.equal(parseCriticJson(null), null);
  // @ts-expect-error — deliberately malformed for this test
  assert.equal(parseCriticJson(undefined), null);
  // @ts-expect-error — deliberately malformed for this test
  assert.equal(parseCriticJson(42), null);
});

test("returns null when there is no JSON object", () => {
  assert.equal(parseCriticJson("This answer looks fine to me."), null);
  assert.equal(parseCriticJson("pass"), null);
});

test("returns null when the JSON lacks a verdict field", () => {
  assert.equal(parseCriticJson('{"severity": 2, "concerns": []}'), null);
});

test("parses a plain JSON object", () => {
  const out = parseCriticJson(
    '{"verdict": "fail", "severity": 4, "concerns": ["math error"], "suggested_fixes": ["recompute"]}'
  );
  assert.ok(out);
  assert.equal(out.verdict, "fail");
  assert.equal(out.severity, 4);
  assert.deepEqual(out.concerns, ["math error"]);
  assert.deepEqual(out.suggested_fixes, ["recompute"]);
});

test("strips ```json fences", () => {
  const raw = '```json\n{"verdict":"pass","severity":0}\n```';
  const out = parseCriticJson(raw);
  assert.ok(out);
  assert.equal(out.verdict, "pass");
  assert.equal(out.severity, 0);
});

test("strips <think>…</think> blocks from reasoning models", () => {
  const raw =
    "<think>Let me consider each claim...</think>\n" +
    '{"verdict":"warn","severity":2,"concerns":["hedging"]}';
  const out = parseCriticJson(raw);
  assert.ok(out);
  assert.equal(out.verdict, "warn");
  assert.equal(out.severity, 2);
});

test("tolerates preamble and trailing commentary", () => {
  const raw =
    "Here is my critique:\n" +
    '{"verdict":"fail","severity":5}\n' +
    "Let me know if you need more detail.";
  const out = parseCriticJson(raw);
  assert.ok(out);
  assert.equal(out.verdict, "fail");
  assert.equal(out.severity, 5);
});

test("clamps severity to 0..5", () => {
  const high = parseCriticJson('{"verdict":"fail","severity":99}');
  assert.ok(high);
  assert.equal(high.severity, 5);

  const low = parseCriticJson('{"verdict":"pass","severity":-3}');
  assert.ok(low);
  assert.equal(low.severity, 0);
});

// 2026-05-09 — behaviour changed (audit finding 2). Unknown verdicts used
// to coerce to "pass", which silently masked critics that emitted off-spec
// labels like "reject" or "unsure". Now: known synonyms map directly,
// genuinely unknown labels coerce to "warn" (the conservative middle).
test("recognises common verdict synonyms (reject/caution/ok/accept)", () => {
  assert.equal(parseCriticJson('{"verdict":"reject","severity":3}')?.verdict, "fail");
  assert.equal(parseCriticJson('{"verdict":"caution","severity":2}')?.verdict, "warn");
  assert.equal(parseCriticJson('{"verdict":"ok","severity":0}')?.verdict, "pass");
  assert.equal(parseCriticJson('{"verdict":"accept","severity":0}')?.verdict, "pass");
});

test("coerces genuinely unknown verdict labels to warn (not silent pass)", () => {
  const out = parseCriticJson('{"verdict":"maybe","severity":1}');
  assert.ok(out);
  assert.equal(out.verdict, "warn");
});

test("caps concerns and suggested_fixes arrays for misbehaving critics", () => {
  // Generate 50 short concerns and 50 short fixes; expect cap at 20 each.
  const concerns = Array.from({ length: 50 }, (_, i) => `concern ${i}`);
  const suggested_fixes = Array.from({ length: 50 }, (_, i) => `fix ${i}`);
  const raw = JSON.stringify({ verdict: "warn", severity: 1, concerns, suggested_fixes });
  const out = parseCriticJson(raw);
  assert.ok(out);
  assert.equal(out.concerns.length, 20);
  assert.equal(out.suggested_fixes.length, 20);
});

test("truncates oversize concern strings with ellipsis", () => {
  const huge = "x".repeat(5_000);
  const raw = JSON.stringify({ verdict: "warn", severity: 1, concerns: [huge] });
  const out = parseCriticJson(raw);
  assert.ok(out);
  assert.equal(out.concerns.length, 1);
  // 2_000 + the trailing ellipsis character.
  assert.equal(out.concerns[0].length, 2_001);
  assert.ok(out.concerns[0].endsWith("…"));
});

test("ignores non-string array entries", () => {
  const out = parseCriticJson(
    '{"verdict":"warn","severity":1,"concerns":["ok",42,null,"also ok"]}'
  );
  assert.ok(out);
  assert.deepEqual(out.concerns, ["ok", "also ok"]);
});

test("defaults missing arrays to empty", () => {
  const out = parseCriticJson('{"verdict":"pass","severity":0}');
  assert.ok(out);
  assert.deepEqual(out.concerns, []);
  assert.deepEqual(out.suggested_fixes, []);
});

test("returns null for truncated JSON", () => {
  // Model output cut off mid-object.
  assert.equal(
    parseCriticJson('{"verdict":"fail","concerns":["bad math'),
    null
  );
});
