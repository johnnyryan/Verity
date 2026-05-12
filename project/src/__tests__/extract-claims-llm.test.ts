/**
 * Unit tests for the LLM claim extractor's response parser.
 *
 * The HTTP path is not exercised here — it needs a live LM Studio. We
 * only test the deterministic response-parsing logic, which is the part
 * with real failure modes.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { parseClaimsJson } from "../nli/extract-claims-llm.js";

test("parses a plain JSON object", () => {
  const raw = '{"claims":["Water boils at 100°C at sea level.","The Eiffel Tower is in Paris."]}';
  const out = parseClaimsJson(raw);
  assert.deepEqual(out, [
    "Water boils at 100°C at sea level.",
    "The Eiffel Tower is in Paris.",
  ]);
});

test("strips ```json fences", () => {
  const raw = '```json\n{"claims":["A."]}\n```';
  assert.deepEqual(parseClaimsJson(raw), ["A."]);
});

test("strips <think>…</think> prefixes", () => {
  const raw =
    "<think>Let me look at the answer...</think>\n" +
    '{"claims":["Claim one.","Claim two."]}';
  assert.deepEqual(parseClaimsJson(raw), ["Claim one.", "Claim two."]);
});

test("returns [] for valid JSON with empty claims array", () => {
  assert.deepEqual(parseClaimsJson('{"claims":[]}'), []);
});

test("returns null when there is no JSON", () => {
  assert.equal(parseClaimsJson("Here you go"), null);
});

test("returns null when JSON lacks a claims array", () => {
  assert.equal(parseClaimsJson('{"other":"value"}'), null);
});

test("returns null for truncated JSON", () => {
  assert.equal(parseClaimsJson('{"claims":["half a '), null);
});

test("drops non-string entries in the array", () => {
  const raw = '{"claims":["ok",42,null,"also ok",true]}';
  assert.deepEqual(parseClaimsJson(raw), ["ok", "also ok"]);
});

test("trims whitespace and drops empty entries", () => {
  const raw = '{"claims":["  padded  ","","\\n"]}';
  assert.deepEqual(parseClaimsJson(raw), ["padded"]);
});

test("tolerates prose before and after the JSON object", () => {
  const raw =
    "Sure, here are the claims I identified:\n" +
    '{"claims":["Claim A.","Claim B."]}\n' +
    "Let me know if you want more detail.";
  assert.deepEqual(parseClaimsJson(raw), ["Claim A.", "Claim B."]);
});
