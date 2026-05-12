/**
 * Unit tests for the reasoning-trace sanitiser.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { stripReasoningTraces } from "../sanitize.js";

test("returns empty string unchanged", () => {
  assert.equal(stripReasoningTraces(""), "");
});

test("returns text without reasoning markers unchanged", () => {
  const s = "Paris is the capital of France.";
  assert.equal(stripReasoningTraces(s), s);
});

test("strips a single <think> block", () => {
  const raw =
    "<think>Let me consider the options...</think>\n" +
    "The capital of France is Paris.";
  assert.equal(
    stripReasoningTraces(raw),
    "The capital of France is Paris."
  );
});

test("strips multiple think blocks", () => {
  const raw =
    "<think>First thought.</think>Answer one. " +
    "<think>Second thought.</think>Answer two.";
  assert.equal(stripReasoningTraces(raw), "Answer one. Answer two.");
});

test("strips <thinking> blocks case-insensitively", () => {
  const raw = "<Thinking>reasoning</Thinking>The result.";
  assert.equal(stripReasoningTraces(raw), "The result.");
});

test("strips <reasoning> blocks (DeepSeek-R1 style)", () => {
  const raw =
    "<reasoning>Step 1: consider x.\nStep 2: conclude y.</reasoning>\n" +
    "y holds.";
  assert.equal(stripReasoningTraces(raw), "y holds.");
});

test("strips OpenAI Harmony-style analysis channels", () => {
  const raw =
    "<|channel|>analysis thinking out loud here<|end|>" +
    "Final answer.";
  assert.equal(stripReasoningTraces(raw), "Final answer.");
});

test("handles unclosed markers by leaving them alone", () => {
  // If the model emits a partial marker, we deliberately don't try to
  // guess where it ends — better to keep the raw text than drop too much.
  const raw = "<think>never closed. The answer is 42.";
  assert.equal(stripReasoningTraces(raw), raw);
});

test("handles multiline blocks", () => {
  const raw =
    "<think>\nline 1\nline 2\nline 3\n</think>\n\nThe final answer.";
  assert.equal(stripReasoningTraces(raw), "The final answer.");
});
