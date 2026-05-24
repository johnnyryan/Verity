/**
 * Tests for the critic-dispatch shuffle (Upgrade #6, 2026-05-23).
 *
 * The shuffle is per-call position-bias mitigation borrowed from Zheng
 * 2023 (MT-Bench / Chatbot Arena, arXiv:2306.05685). Verity's critics
 * don't see each other's verdicts so the direct LLM-as-judge bias does
 * not apply inside one call, but the shuffle guards anything downstream
 * that picks "first critic" as authoritative.
 *
 * Tests cover the shuffleCritics helper directly. Plumbing into the full
 * /verify pipeline depends on env CRITIC_SHUFFLE; the helper is the
 * deterministic core to exercise.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { shuffleCritics } from "../pipeline.js";
import { ALL_CRITICS } from "../critics/critic-configs.js";

test("shuffleCritics returns a new array with the same elements", () => {
  const original = ALL_CRITICS;
  const shuffled = shuffleCritics(original);
  // Should not be the literal same reference.
  assert.notEqual(shuffled, original);
  // Same length, same set of ids — only the order can differ.
  assert.equal(shuffled.length, original.length);
  const ids = new Set(shuffled.map((c) => c.id));
  for (const c of original) assert.ok(ids.has(c.id));
});

test("shuffleCritics: when only two critics, order varies across calls", () => {
  // With N=2 the shuffle is biased 50/50: original or swapped. Across
  // many calls we should see BOTH orderings, otherwise the shuffle is
  // a no-op. Probability of a false negative (all 200 calls same
  // order) under fair coin = 2^-199 ≈ 0. Good enough.
  if (ALL_CRITICS.length < 2) return;
  const firstA = ALL_CRITICS[0]!.id;
  let sawA = false;
  let sawNotA = false;
  for (let i = 0; i < 200; i++) {
    const shuffled = shuffleCritics(ALL_CRITICS);
    if (shuffled[0]!.id === firstA) sawA = true;
    else sawNotA = true;
    if (sawA && sawNotA) break;
  }
  assert.ok(sawA, "expected at least one shuffle to leave first critic in place");
  assert.ok(sawNotA, "expected at least one shuffle to swap first critic out");
});

test("shuffleCritics: synthetic N=5 panel is not always identity-ordered", () => {
  // The shuffle should also work for a larger panel, in case a third
  // critic ever comes back online. Use synthetic configs so the test
  // isn't tied to live config.
  const ids = ["a", "b", "c", "d", "e"];
  const synthetic = ids.map((id) => ({
    id,
    displayName: id,
    family: "Test",
    endpoint: "http://x",
    apiKey: "x",
    model: id,
  }));
  let identityCount = 0;
  let nonIdentityCount = 0;
  const target = ids.join("");
  for (let i = 0; i < 200; i++) {
    const order = shuffleCritics(synthetic).map((c) => c.id).join("");
    if (order === target) identityCount++;
    else nonIdentityCount++;
  }
  // 5! = 120, identity probability = 1/120 ≈ 0.83%. Across 200 trials
  // we expect roughly 1-2 identity orderings and ~198 non-identity.
  assert.ok(
    nonIdentityCount > 100,
    `expected >100 non-identity orderings, got ${nonIdentityCount}`
  );
});

test("shuffleCritics: does not mutate the input array", () => {
  const before = ALL_CRITICS.map((c) => c.id);
  for (let i = 0; i < 20; i++) shuffleCritics(ALL_CRITICS);
  const after = ALL_CRITICS.map((c) => c.id);
  assert.deepEqual(after, before);
});

test("shuffleCritics: empty input → empty output", () => {
  const r = shuffleCritics([]);
  assert.deepEqual(r, []);
});

test("shuffleCritics: single-element input → identical single-element output", () => {
  const single = [ALL_CRITICS[0]!];
  const r = shuffleCritics(single);
  assert.equal(r.length, 1);
  assert.equal(r[0]!.id, single[0]!.id);
});
