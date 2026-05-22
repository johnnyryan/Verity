/**
 * Smoke tests for the critic config wiring.
 *
 * These tests enforce invariants that, if broken, would silently
 * mis-populate VerifyOutput.critics: the output shape has fixed keys
 * and pipeline.ts looks each one up by id.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { ALL_CRITICS } from "../critics/critic-configs.js";

// 2026-05-09: refreshed for the live 2-critic panel. The original tests
// asserted the 3-critic layout (Phi-4-mini + Nemotron Mini + Llama 3.2 3B)
// from the early design. The fleet was reduced to 2 critics in the 8k-
// context redesign (see CRITIC_C comment in critic-configs.ts) and these
// tests were stale until the npm-test script wiring was repaired.
test("ALL_CRITICS contains the configured fleet (currently 2 critics)", () => {
  assert.equal(ALL_CRITICS.length, 2);
});

test("ALL_CRITICS ids match the keys VerifyOutput.critics declares", () => {
  // Wire ids are model-agnostic (renamed 2026-05-20 from the prior
  // model-specific "granite_3_2_8b"/"granite_3_2_2b" labels; themselves
  // renamed 2026-05-11 from the legacy "phi4_reasoning"/"nemotron_mini"
  // that leaked into Qwen hallucinations). Decoupling slot name from
  // model means future swaps don't require touching three files.
  const ids = ALL_CRITICS.map((c) => c.id).sort();
  assert.deepEqual(ids, ["critic_a", "critic_b"]);
});

test("MAX_UNAVAILABLE_CRITICS is sized to leave at least one critic to vote", () => {
  // Belt-and-braces: with the gate fixed to `>` (audit finding 1, 2026-05-09),
  // having MAX_UNAVAILABLE_CRITICS = ALL_CRITICS.length would let a panel of
  // all-unavailable critics pass through. Pin the relationship.
  // Imported dynamically so a future re-enable of CRITIC_C just needs both
  // numbers updated.
  return import("../config.js").then(({ MAX_UNAVAILABLE_CRITICS }) => {
    assert.ok(
      MAX_UNAVAILABLE_CRITICS < ALL_CRITICS.length,
      `MAX_UNAVAILABLE_CRITICS (${MAX_UNAVAILABLE_CRITICS}) must be < ` +
        `ALL_CRITICS.length (${ALL_CRITICS.length}) so at least one critic ` +
        `can survive a degraded run`
    );
  });
});

test("each critic has the required fields", () => {
  for (const cfg of ALL_CRITICS) {
    assert.ok(cfg.id, `missing id: ${JSON.stringify(cfg)}`);
    assert.ok(cfg.displayName, `missing displayName for ${cfg.id}`);
    assert.ok(cfg.family, `missing family for ${cfg.id}`);
    assert.ok(cfg.endpoint.startsWith("http"), `bad endpoint for ${cfg.id}`);
    assert.ok(cfg.apiKey, `missing apiKey for ${cfg.id}`);
    assert.ok(cfg.model, `missing model for ${cfg.id}`);
  }
});

test("displayName is a non-empty human-readable label", () => {
  // 2026-05-12: the displayName used to be tied to a specific model
  // ("IBM Granite 3.2 8B") and a regex matched on "granite". That tied
  // the test to the live roster; Johnny's "treat models as
  // placeholders" directive moved displayNames to role-only labels
  // ("Critic A", "Critic B"). Test now checks the contract that
  // matters: displayName exists and isn't just the wire id leaking
  // through.
  for (const cfg of ALL_CRITICS) {
    assert.ok(
      cfg.displayName && cfg.displayName.trim().length > 0,
      `displayName must be a non-empty string for ${cfg.id}`
    );
    assert.notEqual(
      cfg.displayName,
      cfg.id,
      `displayName must differ from the wire id for ${cfg.id}` +
        ` (got "${cfg.displayName}")`
    );
  }
});

test("contextLimit implies contextHeadroom is also set", () => {
  for (const cfg of ALL_CRITICS) {
    if (cfg.contextLimit !== undefined) {
      assert.ok(
        cfg.contextHeadroom !== undefined,
        `${cfg.id} has contextLimit but no contextHeadroom`
      );
      assert.ok(
        cfg.contextHeadroom < cfg.contextLimit,
        `${cfg.id} headroom must be smaller than limit`
      );
    }
  }
});
