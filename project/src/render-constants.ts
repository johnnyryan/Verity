/**
 * Verdict and check-chip emoji constants shared by the markdown
 * renderer (aggregator.ts) and the test suite.
 *
 * Extracted 2026-05-20 from inlined literals in aggregator.ts +
 * summary-md.test.ts so a future emoji swap is a one-file edit
 * instead of having to chase grep-able strings across both.
 *
 * If you add a new chip, add it here AND update the consumers.
 */

/** Verdict-line chips, used in the bold conclusion under the testing table. */
export const VERDICT_CHIPS = {
  pass: "✅ pass",
  warn: "⚠️ warn",
  fail: "❌ fail",
  error: "⛔ error",
} as const;

/**
 * Per-row "outcome" chips in the testing table. Shared between critics,
 * recompute, NLI, consistency, and perplexity rows.
 */
export const CHECK_CHIPS = {
  pass: "✅ pass",
  warn: "⚠️ warn",
  fail: "❌ fail",
  skipped: "— skipped",
  error: "⛔ unavailable",
  n_a: "— N/A",
  unable: "❓ unable to assess",
} as const;
