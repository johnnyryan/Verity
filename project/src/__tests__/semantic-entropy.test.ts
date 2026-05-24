/**
 * Unit tests for the semantic-entropy signal (Farquhar et al., Nature 2024).
 *
 * Tests run against a stub NLI entailer so the ~1 GB DeBERTa model is
 * never loaded. The stub returns "entail" when the two strings are
 * identical (case-insensitive after whitespace trim) and "neutral"
 * otherwise. That's enough to exercise the clustering + entropy maths
 * without touching the real cross-encoder.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  computeSemanticEntropy,
  type NliEntailer,
} from "../signals/semantic-entropy.js";

/**
 * Identity-only NLI stub: returns "entail" with score 0.99 when the
 * normalised premise equals the normalised hypothesis, "neutral" 0.5
 * otherwise. Bidirectional clustering with this stub groups together
 * exact-equal samples only.
 */
const identityNli: NliEntailer = async (premise, hypothesis) => {
  const a = premise.trim().toLowerCase();
  const b = hypothesis.trim().toLowerCase();
  if (a === b) return { label: "entailment", score: 0.99 };
  return { label: "neutral", score: 0.5 };
};

/**
 * Permissive NLI stub: always returns "entail". Forces every sample into
 * a single cluster regardless of surface form. Used to verify the all-
 * agree degenerate case yields entropy 0 even when samples are distinct
 * strings.
 */
const allEntailNli: NliEntailer = async () => ({
  label: "entailment",
  score: 0.95,
});

/**
 * Restrictive NLI stub: always returns "neutral". Forces every sample
 * into its own cluster. Used to verify the all-disagree case yields
 * entropy log(N).
 */
const allNeutralNli: NliEntailer = async () => ({
  label: "neutral",
  score: 0.5,
});

test("three identical samples → entropy 0, one cluster", async () => {
  const samples = ["The sky is blue.", "The sky is blue.", "The sky is blue."];
  const r = await computeSemanticEntropy(samples, identityNli);
  assert.equal(r.entropy, 0);
  assert.equal(r.clusterCount, 1);
  assert.equal(r.clusters.length, 1);
  assert.equal(r.clusters[0].length, 3);
});

test("three completely different samples → entropy ≈ log(3), three clusters", async () => {
  const samples = ["alpha answer one", "beta answer two", "gamma answer three"];
  const r = await computeSemanticEntropy(samples, identityNli);
  // log(3) = 1.0986; allow tiny rounding slack.
  assert.equal(r.clusterCount, 3);
  assert.ok(
    Math.abs(r.entropy - Math.log(3)) < 1e-3,
    `expected entropy ≈ log(3) = ${Math.log(3).toFixed(4)}, got ${r.entropy}`
  );
});

test("empty array → entropy 0, zero clusters", async () => {
  const r = await computeSemanticEntropy([], identityNli);
  assert.equal(r.entropy, 0);
  assert.equal(r.clusterCount, 0);
  assert.deepEqual(r.clusters, []);
});

test("single sample → entropy 0, one cluster", async () => {
  const r = await computeSemanticEntropy(["only one"], identityNli);
  assert.equal(r.entropy, 0);
  assert.equal(r.clusterCount, 1);
  assert.equal(r.clusters[0][0], "only one");
});

test("two identical and one different → entropy = H(2/3, 1/3)", async () => {
  const samples = ["same answer", "same answer", "different answer"];
  const r = await computeSemanticEntropy(samples, identityNli);
  assert.equal(r.clusterCount, 2);
  // H = -(2/3) ln(2/3) - (1/3) ln(1/3) ≈ 0.6365 nats.
  const expected = -(2 / 3) * Math.log(2 / 3) - (1 / 3) * Math.log(1 / 3);
  assert.ok(
    Math.abs(r.entropy - expected) < 1e-3,
    `expected ${expected.toFixed(4)}, got ${r.entropy}`
  );
});

test("all-entail NLI: every sample collapses into one cluster, entropy 0", async () => {
  // Verifies the "model collapses cluster" branch independent of identity:
  // even with surface-different samples, if NLI calls each pair entailed
  // both ways, they form one meaning-cluster.
  const samples = ["a", "b", "c", "d", "e"];
  const r = await computeSemanticEntropy(samples, allEntailNli);
  assert.equal(r.clusterCount, 1);
  assert.equal(r.entropy, 0);
});

test("all-neutral NLI: every sample is its own cluster, entropy = log(N)", async () => {
  // Verifies the "model never agrees" branch: with neutral on every pair
  // the greedy clustering puts each sample in its own cluster.
  const samples = ["a", "b", "c", "d"];
  const r = await computeSemanticEntropy(samples, allNeutralNli);
  assert.equal(r.clusterCount, 4);
  assert.ok(
    Math.abs(r.entropy - Math.log(4)) < 1e-3,
    `expected log(4) = ${Math.log(4).toFixed(4)}, got ${r.entropy}`
  );
});

test("empty-string samples are filtered out", async () => {
  // Whitespace-only / empty inputs should not corrupt the cluster count.
  const samples = ["", "  ", "valid sample", "valid sample"];
  const r = await computeSemanticEntropy(samples, identityNli);
  assert.equal(r.clusterCount, 1);
  assert.equal(r.clusters[0].length, 2);
});

test("NLI failure is treated as non-entailment (conservative)", async () => {
  // An NLI primitive that throws should not crash the entropy computer;
  // pairs that error are treated as not-mutually-entailed, biasing
  // entropy upward.
  const throwingNli: NliEntailer = async () => {
    throw new Error("nli backend down");
  };
  const r = await computeSemanticEntropy(
    ["one", "two", "three"],
    throwingNli
  );
  // Every pair "fails" → every sample is its own cluster → entropy log(3).
  assert.equal(r.clusterCount, 3);
  assert.ok(Math.abs(r.entropy - Math.log(3)) < 1e-3);
});
