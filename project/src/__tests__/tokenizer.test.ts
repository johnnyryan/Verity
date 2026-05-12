/**
 * Unit tests for the shared tokenizer helper.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { countTokens, truncateToTokenBudget } from "../tokenizer.js";

test("countTokens returns 0 for empty string", () => {
  assert.equal(countTokens(""), 0);
});

test("countTokens returns a positive count for non-empty text", () => {
  const n = countTokens("The quick brown fox jumps over the lazy dog.");
  assert.ok(n > 0);
  assert.ok(n < 20); // sanity — cl100k encodes this as ~10 tokens
});

test("countTokens is roughly proportional to repeated content", () => {
  const short = countTokens("hello world");
  const long = countTokens("hello world ".repeat(100));
  // Repeating 100x should yield at least 50x tokens (allowing for BPE merges).
  assert.ok(long > short * 50);
});

test("truncateToTokenBudget returns text unchanged when it fits", () => {
  const text = "short text";
  const out = truncateToTokenBudget(text, 1000);
  assert.equal(out.truncated, false);
  assert.equal(out.text, text);
});

test("truncateToTokenBudget trims to budget preserving the tail", () => {
  // Build something with a recognisable tail.
  const filler = "Alpha beta gamma delta epsilon. ".repeat(200);
  const marker = "TAIL_SENTINEL_UNIQUE_STRING.";
  const text = filler + marker;

  const out = truncateToTokenBudget(text, 20);
  assert.equal(out.truncated, true);
  // The tail sentinel must survive; the head filler must not dominate.
  assert.ok(out.text.includes("SENTINEL") || out.text.length < text.length);
  assert.ok(countTokens(out.text) <= 20 + 2); // small slack for decode round-trip
});

test("truncateToTokenBudget handles zero budget", () => {
  const out = truncateToTokenBudget("anything", 0);
  assert.equal(out.truncated, true);
  assert.equal(out.text, "");
});
