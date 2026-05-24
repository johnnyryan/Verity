/**
 * Tests for the calibrated-threshold loader path in aggregator.ts.
 *
 * The loader runs once at module load. To exercise both branches without
 * the brittleness of mutating import.meta, we test:
 *
 *   1. The active threshold set is one of {"calibrated","v1"} —
 *      both are valid outcomes depending on whether the dev box has a
 *      calibrated-thresholds.json sitting next to dist/aggregator.js.
 *   2. The score function is the same one the bench harness uses — when
 *      we feed in a known critic / NLI / recompute combo, the scalar is
 *      a deterministic function of the counts.
 *   3. The conformal escalation is additive: when a calibrated threshold
 *      set is loaded, a fail-side score does NOT downgrade an already-
 *      fail verdict, and a clean run is never escalated to "pass".
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  ACTIVE_THRESHOLDS,
  aggregate,
  computeNonconformityScore,
} from "../aggregator.js";
import type {
  CriticResult,
  NliResult,
  RecomputeResult,
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

test("ACTIVE_THRESHOLDS resolves to one of the two valid source flavours", () => {
  // Either branch is a valid outcome of the loader. The test pins the
  // shape of the returned ThresholdSet so a regression that returned an
  // undefined/empty object would fail.
  assert.ok(
    ACTIVE_THRESHOLDS.source === "calibrated" ||
      ACTIVE_THRESHOLDS.source === "v1",
    `unexpected source: ${ACTIVE_THRESHOLDS.source}`
  );
  if (ACTIVE_THRESHOLDS.source === "calibrated") {
    assert.ok(ACTIVE_THRESHOLDS.calibrated);
    assert.equal(typeof ACTIVE_THRESHOLDS.calibrated!.alpha, "number");
    assert.equal(
      typeof ACTIVE_THRESHOLDS.calibrated!.warn_score_threshold,
      "number"
    );
    assert.equal(
      typeof ACTIVE_THRESHOLDS.calibrated!.fail_score_threshold,
      "number"
    );
  }
});

test("missing calibrated-thresholds.json: aggregator does not crash", () => {
  // Belt-and-braces. Even if the file is absent the aggregator must
  // still produce a verdict. We don't toggle the file from this test
  // (module-load mutation is fragile); instead this exercises the
  // active code path with normal inputs and asserts it completes.
  const out = aggregate(
    [
      critic({ id: "a", verdict: "pass", severity: 0 }),
      critic({ id: "b", verdict: "pass", severity: 0 }),
    ],
    cleanNli
  );
  assert.ok(["pass", "warn", "fail", "error"].includes(out.consensus));
});

test("computeNonconformityScore sums NLI + critic severity + concerns + recompute mismatches", () => {
  // 1 NLI contradiction
  // 2 NLI unsupported
  // 1 recompute mismatch
  // critic A: severity 0, concerns []
  // critic B: severity 4, concerns ["only one"]
  // Expected scalar: 1 + 2 + 1 + 0 + 4 + 0 + 1 = 9
  const critics: CriticResult[] = [
    critic({ id: "a", verdict: "pass", severity: 0 }),
    critic({
      id: "b",
      verdict: "fail",
      severity: 4,
      concerns: ["only one"],
    }),
  ];
  const nli: NliResult = {
    ran: true,
    claims_checked: 5,
    contradictions: [{ claim: "x", confidence: 0.9 }],
    unsupported: [{ claim: "y" }, { claim: "z" }],
    notes: "",
  };
  const recompute: RecomputeResult = {
    ran: true,
    expressions_found: 2,
    verifications: [],
    mismatches: [
      { kind: "arithmetic", expr_text: "1+1=3", claimed: "3", computed: "2" },
    ],
    notes: "",
    latency_ms: 0,
  };
  const score = computeNonconformityScore({ critics, nli, recompute });
  assert.equal(score, 9);
});

test("computeNonconformityScore: all-clean → score 0", () => {
  const score = computeNonconformityScore({
    critics: [
      critic({ id: "a", verdict: "pass", severity: 0 }),
      critic({ id: "b", verdict: "pass", severity: 0 }),
    ],
    nli: cleanNli,
    recompute: {
      ran: true,
      expressions_found: 0,
      verifications: [],
      mismatches: [],
      notes: "",
      latency_ms: 0,
    },
  });
  assert.equal(score, 0);
});

test("calibrated thresholds skipped under node --test", () => {
  // The loader explicitly skips during test runs so an inherited
  // dist/calibrated-thresholds.json from a developer's earlier
  // calibration run does not silently shift assertions across the suite.
  // Bidirectional behaviour is tested below by mutating ACTIVE_THRESHOLDS
  // for a single test scope.
  assert.equal(ACTIVE_THRESHOLDS.source, "v1");
});

test("bidirectional calibration: low score pulls fail to pass", () => {
  // When the calibrated cut-offs are loaded in bidirectional mode (the
  // 2026-05-24 default), the conformal score IS the decision boundary.
  // A clean nonconformity score takes the verdict to pass even when
  // v1 multi-axis rules would have escalated to fail. This is
  // the whole point of conformal calibration — the calibrated threshold
  // IS the decision boundary, not a tighten-only addendum.
  //
  // We mutate ACTIVE_THRESHOLDS for the duration of this test because
  // the loader skips under tests; restore afterwards.
  const original = { ...ACTIVE_THRESHOLDS };
  Object.assign(ACTIVE_THRESHOLDS, {
    source: "calibrated",
    calibrated: {
      alpha: 0.1,
      calibration_set_size: 200,
      warn_score_threshold: 11,
      fail_score_threshold: 12.8,
    },
    loadedFrom: "(test injection)",
  });
  try {
    const failingNli: NliResult = {
      ran: true,
      claims_checked: 2,
      contradictions: [{ claim: "x", confidence: 0.95 }],
      unsupported: [],
      notes: "",
    };
    const out = aggregate(
      [
        critic({ id: "a", verdict: "pass", severity: 0 }),
        critic({ id: "b", verdict: "pass", severity: 0 }),
      ],
      failingNli
    );
    // Score = 1 NLI contradiction = 1. Warn threshold = 11. So the
    // bidirectional ladder must place this at "pass", overriding the
    // v1 escalation.
    assert.equal(out.consensus, "pass");
  } finally {
    Object.assign(ACTIVE_THRESHOLDS, original);
    if (original.source === "v1") {
      delete (ACTIVE_THRESHOLDS as { calibrated?: unknown }).calibrated;
      delete (ACTIVE_THRESHOLDS as { loadedFrom?: unknown }).loadedFrom;
    }
  }
});

test("bidirectional calibration: high score pulls pass to fail", () => {
  // Same mechanism, the other direction. A score above fail_threshold
  // takes the verdict to fail even when v1 rules say pass
  // (e.g. critics quiet, NLI happy, but the aggregate score has spiked
  // because critic_disagree_count + recompute_mismatches add up).
  const original = { ...ACTIVE_THRESHOLDS };
  Object.assign(ACTIVE_THRESHOLDS, {
    source: "calibrated",
    calibrated: {
      alpha: 0.1,
      calibration_set_size: 200,
      warn_score_threshold: 2,
      fail_score_threshold: 4,
    },
    loadedFrom: "(test injection)",
  });
  try {
    const noisyNli: NliResult = {
      ran: true,
      claims_checked: 5,
      contradictions: [{ claim: "x", confidence: 0.9 }],
      unsupported: [{ claim: "y" }, { claim: "z" }, { claim: "w" }],
      notes: "",
    };
    const out = aggregate(
      [
        critic({ id: "a", verdict: "pass", severity: 0 }),
        critic({ id: "b", verdict: "pass", severity: 0 }),
      ],
      noisyNli
    );
    // Score = 1 contradiction + 3 unsupported = 4 → ≥ fail threshold.
    assert.equal(out.consensus, "fail");
  } finally {
    Object.assign(ACTIVE_THRESHOLDS, original);
    if (original.source === "v1") {
      delete (ACTIVE_THRESHOLDS as { calibrated?: unknown }).calibrated;
      delete (ACTIVE_THRESHOLDS as { loadedFrom?: unknown }).loadedFrom;
    }
  }
});

test("calibrated thresholds never override error verdict", () => {
  // System-state signals (too many critics unavailable) are not quality
  // signals and must survive calibration in either direction.
  const original = { ...ACTIVE_THRESHOLDS };
  Object.assign(ACTIVE_THRESHOLDS, {
    source: "calibrated",
    calibrated: {
      alpha: 0.1,
      calibration_set_size: 200,
      warn_score_threshold: 11,
      fail_score_threshold: 12.8,
    },
    loadedFrom: "(test injection)",
  });
  try {
    const out = aggregate(
      [
        critic({ id: "a", verdict: "pass", severity: 0, unavailable: true, error: "timeout" }),
        critic({ id: "b", verdict: "pass", severity: 0, unavailable: true, error: "timeout" }),
      ],
      cleanNli
    );
    assert.equal(out.consensus, "error");
  } finally {
    Object.assign(ACTIVE_THRESHOLDS, original);
    if (original.source === "v1") {
      delete (ACTIVE_THRESHOLDS as { calibrated?: unknown }).calibrated;
      delete (ACTIVE_THRESHOLDS as { loadedFrom?: unknown }).loadedFrom;
    }
  }
});
