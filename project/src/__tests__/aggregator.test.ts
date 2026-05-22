/**
 * Unit tests for the consensus aggregator.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { aggregate, computeDisputes } from "../aggregator.js";
import { CONSISTENCY_FAIL_THRESHOLD } from "../config.js";
import type {
  ConsistencyResult,
  CriticResult,
  NliResult,
  PerplexityResult,
} from "../types.js";

function critic(overrides: Partial<CriticResult> = {}): CriticResult {
  return {
    id: overrides.id ?? "test_critic",
    display_name: overrides.display_name ?? "Test Critic",
    family: overrides.family ?? "Test",
    verdict: overrides.verdict ?? "pass",
    severity: overrides.severity ?? 0,
    concerns: overrides.concerns ?? [],
    suggested_fixes: overrides.suggested_fixes ?? [],
    notes: overrides.notes ?? [],
    unavailable: overrides.unavailable,
    error: overrides.error,
    latency_ms: overrides.latency_ms ?? 10,
  };
}

const cleanNli: NliResult = {
  ran: true,
  claims_checked: 3,
  contradictions: [],
  unsupported: [],
  notes: "clean",
};

test("all-pass critics + clean NLI → pass", () => {
  const out = aggregate(
    [
      critic({ id: "a", verdict: "pass", severity: 0 }),
      critic({ id: "b", verdict: "pass", severity: 0 }),
      critic({ id: "c", verdict: "pass", severity: 0 }),
    ],
    cleanNli
  );
  assert.equal(out.consensus, "pass");
  assert.deepEqual(out.critics_unavailable, []);
});

test("one critic at fail severity → fail", () => {
  const out = aggregate(
    [
      critic({ id: "a", severity: 0 }),
      critic({ id: "b", severity: 4, verdict: "fail" }),
      critic({ id: "c", severity: 0 }),
    ],
    cleanNli
  );
  assert.equal(out.consensus, "fail");
});

test("NLI contradictions → fail regardless of critics", () => {
  const nli: NliResult = {
    ran: true,
    claims_checked: 1,
    contradictions: [{ claim: "Paris is in Spain", confidence: 0.95 }],
    unsupported: [],
    notes: "found 1 contradiction",
  };
  const out = aggregate(
    [
      critic({ id: "a", severity: 0 }),
      critic({ id: "b", severity: 0 }),
      critic({ id: "c", severity: 0 }),
    ],
    nli
  );
  assert.equal(out.consensus, "fail");
});

test("warn-severity critic alone → warn", () => {
  // Uses severity=2 (the current WARN_SEVERITY_THRESHOLD). The 2026-04-18
  // afternoon session raised WARN from 1→2 deliberately so severity-1
  // stylistic nitpicks no longer fire warn by themselves.
  const out = aggregate(
    [
      critic({ id: "a", severity: 2, verdict: "warn" }),
      critic({ id: "b", severity: 0 }),
      critic({ id: "c", severity: 0 }),
    ],
    cleanNli
  );
  assert.equal(out.consensus, "warn");
});

test("two unavailable critics → error (MAX_UNAVAILABLE_CRITICS=2)", () => {
  const out = aggregate(
    [
      critic({ id: "a", unavailable: true }),
      critic({ id: "b", unavailable: true }),
      critic({ id: "c", severity: 0 }),
    ],
    cleanNli
  );
  assert.equal(out.consensus, "error");
  assert.deepEqual(out.critics_unavailable, ["a", "b"]);
});

test("one unavailable critic does not block consensus", () => {
  const out = aggregate(
    [
      critic({ id: "a", unavailable: true }),
      critic({ id: "b", severity: 0 }),
      critic({ id: "c", severity: 0 }),
    ],
    cleanNli
  );
  assert.equal(out.consensus, "pass");
  assert.deepEqual(out.critics_unavailable, ["a"]);
});

// 2026-05-09 — regression test for finding 1 in the audit. The current
// fleet is 2 critics (Granite 3.2 8B + 2B) with MAX_UNAVAILABLE_CRITICS=1.
// The previous `>=` gate flipped consensus to "error" the moment one
// critic was unavailable, even though the surviving critic had a clear
// verdict — defeating the whole point of allowing 1 critic to vote.
test("2-critic panel with 1 unavailable still reaches consensus from the other", () => {
  const out = aggregate(
    [
      critic({ id: "a", unavailable: true, error: "timeout" }),
      critic({ id: "b", severity: 0, verdict: "pass" }),
    ],
    cleanNli
  );
  assert.equal(out.consensus, "pass");
  assert.deepEqual(out.critics_unavailable, ["a"]);
});

test("2-critic panel with both unavailable → error", () => {
  const out = aggregate(
    [
      critic({ id: "a", unavailable: true }),
      critic({ id: "b", unavailable: true }),
    ],
    cleanNli
  );
  assert.equal(out.consensus, "error");
  assert.deepEqual(out.critics_unavailable, ["a", "b"]);
});

test("consistency divergence >= fail threshold → fail", () => {
  const consistency: ConsistencyResult = {
    ran: true,
    samples_generated: 2,
    claims_checked: 4,
    contradicted: [
      { claim: "x", contradicting_sample_index: 0, confidence: 0.9 },
      { claim: "y", contradicting_sample_index: 1, confidence: 0.9 },
    ],
    unsupported: [],
    divergence_score: CONSISTENCY_FAIL_THRESHOLD,
    latency_ms: 500,
    notes: "",
  };
  const out = aggregate(
    [
      critic({ id: "a", severity: 0 }),
      critic({ id: "b", severity: 0 }),
      critic({ id: "c", severity: 0 }),
    ],
    cleanNli,
    { consistency }
  );
  assert.equal(out.consensus, "fail");
});

test("tiny non-zero divergence below warn threshold no longer fires warn", () => {
  // Pre-fix behaviour: any divergence > 0 warned. Post-fix: must hit
  // CONSISTENCY_WARN_THRESHOLD (default 0.15).
  const consistency: ConsistencyResult = {
    ran: true,
    samples_generated: 2,
    claims_checked: 20,
    contradicted: [],
    unsupported: [{ claim: "one minor unsupported" }],
    divergence_score: 0.05,
    latency_ms: 500,
    notes: "",
  };
  const out = aggregate(
    [
      critic({ id: "a", severity: 0 }),
      critic({ id: "b", severity: 0 }),
      critic({ id: "c", severity: 0 }),
    ],
    cleanNli,
    { consistency }
  );
  assert.equal(out.consensus, "pass");
});

test("divergence at warn threshold → warn", () => {
  const consistency: ConsistencyResult = {
    ran: true,
    samples_generated: 2,
    claims_checked: 5,
    contradicted: [],
    unsupported: [{ claim: "a" }],
    divergence_score: 0.2,
    latency_ms: 500,
    notes: "",
  };
  const out = aggregate(
    [
      critic({ id: "a", severity: 0 }),
      critic({ id: "b", severity: 0 }),
      critic({ id: "c", severity: 0 }),
    ],
    cleanNli,
    { consistency }
  );
  assert.equal(out.consensus, "warn");
});

test("perplexity low-confidence spans alone → pass (advisory, never flips the verdict)", () => {
  // 2026-05-22: perplexity / model-uncertainty was demoted to an advisory
  // nudge. Low-confidence spans no longer escalate the consensus; only the
  // critics, NLI, recompute, and consistency do. Clean critics + a flagged
  // perplexity must therefore still read "pass".
  const perplexity: PerplexityResult = {
    ran: true,
    method: "forward_pass_rescore",
    tokens_scored: 100,
    mean_logprob: -2.1,
    perplexity: 8.2,
    low_confidence_spans: [{ text: "uncertain phrase", min_logprob: -3.5 }],
    latency_ms: 1500,
    notes: "",
  };
  const out = aggregate(
    [
      critic({ id: "a", severity: 0 }),
      critic({ id: "b", severity: 0 }),
      critic({ id: "c", severity: 0 }),
    ],
    cleanNli,
    { perplexity }
  );
  assert.equal(out.consensus, "pass");
});

test("fail beats warn when both present", () => {
  const nli: NliResult = {
    ran: true,
    claims_checked: 1,
    contradictions: [{ claim: "x", confidence: 0.9 }],
    unsupported: [{ claim: "y" }],
    notes: "",
  };
  const out = aggregate(
    [
      critic({ id: "a", severity: 1, verdict: "warn" }),
      critic({ id: "b", severity: 0 }),
      critic({ id: "c", severity: 0 }),
    ],
    nli
  );
  assert.equal(out.consensus, "fail");
});

// ─── Phase 3: computeDisputes (diagnostic-only) ───────────────────────────

test("computeDisputes: both critics agree on pass → no disputes", () => {
  const disputes = computeDisputes([
    critic({ id: "a", verdict: "pass", severity: 0, concerns: [] }),
    critic({ id: "b", verdict: "pass", severity: 0, concerns: [] }),
  ]);
  assert.deepEqual(disputes, []);
});

test("computeDisputes: A pass, B fail → exactly one verdict-mismatch, severity hard", () => {
  const disputes = computeDisputes([
    critic({ id: "a", verdict: "pass", severity: 0, concerns: [] }),
    critic({
      id: "b",
      verdict: "fail",
      severity: 4,
      concerns: ["serious issue detected"],
    }),
  ]);
  const mismatches = disputes.filter((d) => d.kind === "verdict-mismatch");
  assert.equal(mismatches.length, 1);
  assert.equal(mismatches[0].severity, "hard");
  assert.equal(mismatches[0].critic_a.verdict, "pass");
  assert.equal(mismatches[0].critic_b.verdict, "fail");
  assert.equal(mismatches[0].critic_a_id, "a");
  assert.equal(mismatches[0].critic_b_id, "b");
  assert.equal(mismatches[0].critic_b.concern, "serious issue detected");
});

test("computeDisputes: A warn, B pass → one verdict-mismatch, severity soft", () => {
  const disputes = computeDisputes([
    critic({
      id: "a",
      verdict: "warn",
      severity: 2,
      concerns: ["something minor"],
    }),
    critic({ id: "b", verdict: "pass", severity: 0, concerns: [] }),
  ]);
  const mismatches = disputes.filter((d) => d.kind === "verdict-mismatch");
  assert.equal(mismatches.length, 1);
  assert.equal(mismatches[0].severity, "soft");
  assert.equal(mismatches[0].critic_a.concern, "something minor");
});

test("computeDisputes: A concern with no B match → one concern-only-in-a", () => {
  const disputes = computeDisputes([
    critic({
      id: "a",
      verdict: "warn",
      severity: 2,
      concerns: ["bad arithmetic in step three"],
    }),
    critic({ id: "b", verdict: "warn", severity: 2, concerns: [] }),
  ]);
  const onlyA = disputes.filter((d) => d.kind === "concern-only-in-a");
  assert.equal(onlyA.length, 1);
  assert.equal(onlyA[0].critic_a.concern, "bad arithmetic in step three");
  assert.equal(onlyA[0].severity, "soft");
  // No verdict-mismatch because both are warn.
  assert.equal(
    disputes.filter((d) => d.kind === "verdict-mismatch").length,
    0
  );
});

test("computeDisputes: each side has a unique concern → one each", () => {
  const disputes = computeDisputes([
    critic({
      id: "a",
      verdict: "warn",
      severity: 2,
      concerns: ["missing citation for historical claim"],
    }),
    critic({
      id: "b",
      verdict: "warn",
      severity: 2,
      concerns: ["tone inconsistent between paragraphs"],
    }),
  ]);
  const onlyA = disputes.filter((d) => d.kind === "concern-only-in-a");
  const onlyB = disputes.filter((d) => d.kind === "concern-only-in-b");
  assert.equal(onlyA.length, 1);
  assert.equal(onlyB.length, 1);
  assert.equal(onlyA[0].critic_a.concern, "missing citation for historical claim");
  assert.equal(onlyB[0].critic_b.concern, "tone inconsistent between paragraphs");
});

test("computeDisputes: fuzzy-matched shared concern → no concern-only-* for that pair", () => {
  // Token sets after filtering words length > 3:
  //   "wrong calculation"       → {wrong, calculation}
  //   "calculation is wrong"    → {calculation, wrong}
  // Jaccard = 2/2 = 1.0 → above 0.40 → match.
  const disputes = computeDisputes([
    critic({
      id: "a",
      verdict: "warn",
      severity: 2,
      concerns: ["wrong calculation"],
    }),
    critic({
      id: "b",
      verdict: "warn",
      severity: 2,
      concerns: ["calculation is wrong"],
    }),
  ]);
  assert.equal(
    disputes.filter(
      (d) => d.kind === "concern-only-in-a" || d.kind === "concern-only-in-b"
    ).length,
    0
  );
});

test("computeDisputes: one critic errored → no disputes", () => {
  const disputes = computeDisputes([
    critic({
      id: "a",
      verdict: "error",
      severity: 0,
      concerns: [],
      unavailable: true,
      error: "timeout",
    }),
    critic({
      id: "b",
      verdict: "fail",
      severity: 4,
      concerns: ["bug"],
    }),
  ]);
  assert.deepEqual(disputes, []);
});
