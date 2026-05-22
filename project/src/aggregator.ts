/**
 * Aggregator: combine critic verdicts, NLI results, and (in deep modes)
 * consistency + perplexity signals into a single consensus.
 *
 * Rules (see config.ts for thresholds):
 *   - If any critic severity >= FAIL_SEVERITY_THRESHOLD → fail
 *   - If NLI finds any contradictions → fail
 *   - If consistency divergence >= CONSISTENCY_FAIL_THRESHOLD → fail
 *   - Elif any critic severity >= WARN_SEVERITY_THRESHOLD → warn
 *   - Elif NLI finds unsupported claims → warn
 *   - Elif consistency divergence >= CONSISTENCY_WARN_THRESHOLD → warn
 *   - Elif too many critics unavailable → error
 *   - Else → pass
 *
 * The logprob perplexity / model-uncertainty signal is ADVISORY only: it is
 * surfaced as a nudge but never flips the consensus (2026-05-22). It is blind
 * to fluent hallucinations and noisy on rare-but-correct wording, so the
 * consistency check is the deep-mode hallucination spine.
 *
 * [ADAPT] If you want weighted voting (e.g. Phi-4 counts 2x because it's
 * the strongest critic), or if you want consistency to be a stronger
 * signal than critics, change the logic here.
 */

import {
  AGGREGATOR_WEIGHTED_VOTE,
  FAIL_SEVERITY_THRESHOLD,
  WARN_SEVERITY_THRESHOLD,
  MAX_UNAVAILABLE_CRITICS,
  CONSISTENCY_FAIL_THRESHOLD,
  CONSISTENCY_WARN_THRESHOLD,
  NLI_IMPL,
  RENDER_CELL_TOPIC_CHARS,
  RENDER_CELL_CONCERN_CHARS,
  RENDER_FINDING_CONCERN_CHARS,
  RENDER_FINDING_EXTRA_CHARS,
  RENDER_FINDING_NLI_CLAIM_CHARS,
  RENDER_FINDING_RECOMPUTE_CHARS,
  SUMMARY_ECHO_ANSWER,
} from "./config.js";
import { CHECK_CHIPS, VERDICT_CHIPS } from "./render-constants.js";
import type {
  ConsistencyResult,
  CriticResult,
  Disagreement,
  NliResult,
  PerplexityResult,
  RecomputeResult,
  Verdict,
  VerifyOutput,
} from "./types.js";

export interface AggregatedResult {
  consensus: Verdict;
  summary: string;
  critics_unavailable: string[];
  /** Diagnostic-only disagreements between the two critics (empty when none). */
  disputes: Disagreement[];
}

/**
 * Tokenise a free-form concern string into a lowercase bag of
 * length-greater-than-3 tokens, for Jaccard-overlap fuzzy matching.
 * Mirrors the helper pattern used in second-opinion/consult.ts so both
 * code paths agree on what counts as "the same idea".
 */
function tokeniseConcern(s: string): Set<string> {
  return new Set(
    s
      .toLowerCase()
      .replace(/[^a-z0-9 ]+/g, " ")
      .split(/\s+/)
      .filter((t) => t.length > 3)
  );
}

function jaccard(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 || b.size === 0) return 0;
  let intersect = 0;
  for (const t of a) if (b.has(t)) intersect += 1;
  return intersect / (a.size + b.size - intersect);
}

const CONCERN_MATCH_THRESHOLD = 0.4;

/**
 * Compute a diagnostic list of disagreements between the two /verify critics.
 *
 * - If fewer than two usable (non-errored / non-unavailable) critics are
 *   present, returns []. Dispute surfacing requires two live critics.
 * - Compares the first two CriticResult entries in `critics` as a, b.
 *   (The /verify ensemble runs exactly two critics — see ALL_CRITICS.)
 * - "verdict-mismatch": emitted when a.verdict !== b.verdict. severity
 *   is "hard" iff the pair straddles the ship/halt line (pass vs fail);
 *   otherwise "soft". Each critic's top concern (concerns[0]) is attached.
 * - "concern-only-in-X": for each concern on side X that does not have
 *   a fuzzy token-Jaccard match (>= CONCERN_MATCH_THRESHOLD) on the other
 *   side, emit one entry. Catches "critic A saw X, critic B didn't".
 *
 * Disputes are diagnostic-only; they never change consensus/verdict.
 */
export function computeDisputes(critics: CriticResult[]): Disagreement[] {
  const usable = critics.filter(
    (c) => !c.unavailable && c.verdict !== "error"
  );
  if (usable.length < 2) return [];

  // 2026-05-12 (E10): previously this function only inspected
  // usable[0] and usable[1] — disagreement from a third critic was
  // silently dropped. With the 2-critic panel that was a no-op, but
  // any future return to a 3-critic fleet would re-introduce the
  // blind spot. Now: fan out across every pair (a, b) where index
  // a < b, so disputes from any pairing are surfaced.
  const disputes: Disagreement[] = [];
  for (let i = 0; i < usable.length; i++) {
    for (let j = i + 1; j < usable.length; j++) {
      collectPairDisputes(usable[i], usable[j], disputes);
    }
  }
  return disputes;
}

function collectPairDisputes(
  a: CriticResult,
  b: CriticResult,
  disputes: Disagreement[]
): void {
  // ── Verdict mismatch ─────────────────────────────────────────────────
  if (a.verdict !== b.verdict) {
    const straddlesShipHalt =
      (a.verdict === "pass" && b.verdict === "fail") ||
      (a.verdict === "fail" && b.verdict === "pass");
    disputes.push({
      kind: "verdict-mismatch",
      critic_a_id: a.id,
      critic_b_id: b.id,
      critic_a: {
        verdict: a.verdict,
        severity: a.severity,
        concern: a.concerns[0],
      },
      critic_b: {
        verdict: b.verdict,
        severity: b.severity,
        concern: b.concerns[0],
      },
      severity: straddlesShipHalt ? "hard" : "soft",
    });
  }

  // ── Concern-only-in-X ─────────────────────────────────────────────────
  const aTokens = a.concerns.map(tokeniseConcern);
  const bTokens = b.concerns.map(tokeniseConcern);

  const hasFuzzyMatch = (needle: Set<string>, haystack: Set<string>[]): boolean =>
    haystack.some((h) => jaccard(needle, h) >= CONCERN_MATCH_THRESHOLD);

  for (let i = 0; i < a.concerns.length; i++) {
    if (!hasFuzzyMatch(aTokens[i], bTokens)) {
      disputes.push({
        kind: "concern-only-in-a",
        critic_a_id: a.id,
        critic_b_id: b.id,
        critic_a: {
          verdict: a.verdict,
          severity: a.severity,
          concern: a.concerns[i],
        },
        critic_b: {
          verdict: b.verdict,
          severity: b.severity,
        },
        severity: "soft",
      });
    }
  }

  for (let i = 0; i < b.concerns.length; i++) {
    if (!hasFuzzyMatch(bTokens[i], aTokens)) {
      disputes.push({
        kind: "concern-only-in-b",
        critic_a_id: a.id,
        critic_b_id: b.id,
        critic_a: {
          verdict: a.verdict,
          severity: a.severity,
        },
        critic_b: {
          verdict: b.verdict,
          severity: b.severity,
          concern: b.concerns[i],
        },
        severity: "soft",
      });
    }
  }
}

export function aggregate(
  critics: CriticResult[],
  nli: NliResult,
  deep?: {
    consistency?: ConsistencyResult;
    perplexity?: PerplexityResult;
    recompute?: RecomputeResult;
  }
): AggregatedResult {
  const unavailable = critics.filter((c) => c.unavailable);
  const available = critics.filter((c) => !c.unavailable);
  const criticsUnavailable = unavailable.map((c) => c.id);

  // Gate is `>` not `>=`: with N critics and MAX_UNAVAILABLE_CRITICS = N-1
  // we want at-least-one critic to vote. The previous `>=` meant a single
  // unavailable critic in the 2-critic panel produced consensus="error"
  // even though one critic was still alive.
  if (unavailable.length > MAX_UNAVAILABLE_CRITICS) {
    return {
      consensus: "error",
      summary:
        `${unavailable.length} of ${critics.length} critics unavailable ` +
        `(${criticsUnavailable.join(", ")}). Cannot form consensus.`,
      critics_unavailable: criticsUnavailable,
      disputes: [],
    };
  }

  const maxSeverity = available.reduce((m, c) => Math.max(m, c.severity), 0);
  const totalConcerns = available.reduce((n, c) => n + c.concerns.length, 0);

  // 2026-04-20 — deterministic recompute pass integration.
  //
  // Rule 1 (hard fail): any recompute mismatch is a confident failure.
  //   The pass is deterministic — if 3*5+7 != 25, there is no model
  //   uncertainty to weigh.
  //
  // Rule 2 (NLI suppression on match): when the recompute pass VERIFIED
  //   an expression as correct, we suppress any NLI contradiction flag
  //   whose `claim` text contains that expression. This directly targets
  //   the `math-subtle` / `subtle-math` failure mode where the LLM claim-
  //   checker false-flags correct arithmetic.
  //
  // Rule 2 does NOT suppress unsupported flags — those are "couldn't
  // verify against premise", not "contradicts premise", and are orthogonal
  // to whether arithmetic checks out.
  const recompute = deep?.recompute;
  const verifiedExprs =
    recompute?.verifications
      ?.filter((v) => v.matches)
      .map((v) => v.expr_text) ?? [];
  const suppressNliClaim = (claim: string): boolean => {
    if (verifiedExprs.length === 0) return false;
    const lc = claim.toLowerCase();
    return verifiedExprs.some((e) => lc.includes(e.toLowerCase()));
  };
  const filteredContradictions = nli.contradictions.filter(
    (c) => !suppressNliClaim(c.claim)
  );
  const filteredUnsupported = nli.unsupported;

  const contradictionCount = filteredContradictions.length;
  const unsupportedCount = filteredUnsupported.length;

  // 2026-04-18 weighted voting (OPT-IN, default OFF).
  //
  // The infrastructure (CriticConfig.weight, CriticResult.weight) is kept
  // for future experiments. But the override rule is gated by env var
  // AGGREGATOR_WEIGHTED_VOTE=1 because the 2026-04-18 afternoon sweep
  // showed the trade was a wash on our corpus: the lone-2B-fail downgrade
  // flipped code-clean (false positive) from MISS to ~warn (better) but
  // also flipped code-subtle-bug (true bug caught by 2B only) from OK to
  // ~warn (worse). 50/50 on when the 2B vs 8B is right when they
  // disagree. Default behaviour preserves the conservative contract that
  // any critic at fail severity flips consensus to fail.
  // 2026-05-12 (E9): was `process.env.AGGREGATOR_WEIGHTED_VOTE === "1"`
  // read directly here, bypassing the "all knobs in config.ts" contract.
  // Now imported from config.ts where the [ADAPT] documentation lives
  // alongside the rest of the verdict-shaping constants.
  const weightedVoteOn = AGGREGATOR_WEIGHTED_VOTE;
  const criticWeight = (c: CriticResult): number => c.weight ?? 1;
  const maxFailWeight = available
    .filter((c) => c.severity >= FAIL_SEVERITY_THRESHOLD)
    .reduce((m, c) => Math.max(m, criticWeight(c)), 0);
  const maxPassWeight = available
    .filter((c) => c.severity === 0)
    .reduce((m, c) => Math.max(m, criticWeight(c)), 0);
  const failOverriddenByHigherWeightPass =
    weightedVoteOn &&
    maxFailWeight > 0 &&
    maxPassWeight > maxFailWeight;

  const consistency = deep?.consistency;
  const perplexity = deep?.perplexity;

  const divergence = consistency?.ran ? consistency.divergence_score : 0;
  const hasConsistencyContradiction =
    consistency?.ran && consistency.contradicted.length > 0;
  // NOTE: perplexity / model-uncertainty deliberately does NOT contribute to
  // the consensus (2026-05-22). It is rendered as an advisory nudge only. See
  // the header comment for why; the consistency check is the spine instead.

  // 2026-04-18: tightened the NLI-unsupported rule. Previously, any single
  // NLI "unsupported" claim would escalate consensus to warn. On the 48-case
  // NLI corpus this produced 8/8 false positives on `ctx-entailed` cases
  // (answers entailed by prior_context but flagged "neutral" by DeBERTa,
  // which the aggregator treats as unsupported). New rule: a single
  // unsupported is noise; 2+ escalates; 1+ only escalates when critics
  // also raised something at WARN severity or higher (i.e.
  // maxSeverity >= WARN_SEVERITY_THRESHOLD, currently 2). The comment
  // here used to say "maxSeverity >= 1" which contradicted the code;
  // corrected 2026-05-12.
  const unsupportedEscalates =
    unsupportedCount >= 2 ||
    (unsupportedCount >= 1 && maxSeverity >= WARN_SEVERITY_THRESHOLD);

  // Apply the standard fail rule, then the weighted-vote override.
  const anyCriticFailed = maxSeverity >= FAIL_SEVERITY_THRESHOLD;
  const criticSeverityForcesFailure =
    anyCriticFailed && !failOverriddenByHigherWeightPass;

  let consensus: Verdict;
  if (
    criticSeverityForcesFailure ||
    contradictionCount > 0 ||
    divergence >= CONSISTENCY_FAIL_THRESHOLD ||
    (recompute?.mismatches?.length ?? 0) > 0
  ) {
    consensus = "fail";
  } else if (
    anyCriticFailed ||
    maxSeverity >= WARN_SEVERITY_THRESHOLD ||
    unsupportedEscalates ||
    hasConsistencyContradiction ||
    divergence >= CONSISTENCY_WARN_THRESHOLD
  ) {
    consensus = "warn";
  } else {
    consensus = "pass";
  }

  // Disputes are computed *after* consensus is assembled so aggregator
  // logic stays untouched: disputes are diagnostic-only and never flip
  // the verdict. Placed before buildSummary per the Phase 3 design.
  const disputes = computeDisputes(critics);

  const summary = buildSummary({
    consensus,
    totalConcerns,
    contradictionCount,
    unsupportedCount,
    unavailableCount: unavailable.length,
    available,
    consistency,
    perplexity,
  });

  return {
    consensus,
    summary,
    critics_unavailable: criticsUnavailable,
    disputes,
  };
}

function buildSummary(params: {
  consensus: Verdict;
  totalConcerns: number;
  contradictionCount: number;
  unsupportedCount: number;
  unavailableCount: number;
  available: CriticResult[];
  consistency?: ConsistencyResult;
  perplexity?: PerplexityResult;
}): string {
  const parts: string[] = [];

  if (params.consensus === "pass") {
    // 2026-05-11 v3: "pass with concerns" — a critic returned verdict
    // 'pass' but raised a concern (typically: insufficient sources to
    // verify). Renderer shows "❓ unable to assess" instead of "✅ pass"
    // for that critic; the summary should reflect that not all critics
    // were able to fully endorse.
    if (params.totalConcerns > 0) {
      parts.push(
        `Critics did not flag the answer as wrong, but raised ` +
          `${params.totalConcerns} concern(s) about verifiability.`
      );
    } else {
      parts.push("All critics passed.");
    }
  } else if (params.consensus === "warn") {
    if (params.totalConcerns > 0) {
      parts.push(`${params.totalConcerns} concern(s) raised by critics.`);
    }
    if (params.unsupportedCount > 0) {
      parts.push(
        `${params.unsupportedCount} factual claim(s) lacked supporting evidence (NLI).`
      );
    }
  } else if (params.consensus === "fail") {
    const failing = params.available
      .filter((c) => c.severity >= FAIL_SEVERITY_THRESHOLD)
      .map((c) => c.display_name);
    if (failing.length > 0) {
      parts.push(`Flagged as serious by: ${failing.join(", ")}.`);
    }
    if (params.contradictionCount > 0) {
      parts.push(
        `${params.contradictionCount} factual contradiction(s) detected by NLI.`
      );
    }
  }

  // Consistency-specific summary additions.
  if (params.consistency?.ran) {
    if (params.consistency.contradicted.length > 0) {
      parts.push(
        `${params.consistency.contradicted.length} claim(s) contradicted by ` +
          `worker re-samples (divergence ${params.consistency.divergence_score}).`
      );
    } else if (params.consistency.unsupported.length > 0) {
      parts.push(
        `${params.consistency.unsupported.length} claim(s) unsupported by ` +
          `worker re-samples (divergence ${params.consistency.divergence_score}).`
      );
    }
  }

  // Perplexity-specific summary additions.
  if (params.perplexity?.ran && params.perplexity.low_confidence_spans.length > 0) {
    parts.push(
      `${params.perplexity.low_confidence_spans.length} low-confidence ` +
        `token span(s) detected (perplexity ${params.perplexity.perplexity}).`
    );
  }

  if (params.unavailableCount > 0) {
    parts.push(`${params.unavailableCount} critic(s) unavailable.`);
  }

  return parts.join(" ") || "No significant signal.";
}

// ─────────────────────────────────────────────────────────────────────────
// Human-readable summary renderer (2026-04-21)
// ─────────────────────────────────────────────────────────────────────────
//
// Motivation: /verify previously returned raw JSON and the tool description
// told the worker to paste that JSON as a fenced code block. Users found
// the wall-of-JSON hard to scan at a glance, and were confused by legacy
// wire IDs ("phi4_reasoning" / "nemotron_mini") that didn't match the
// actual running models. 2026-05-11: those legacy IDs were renamed to
// "granite_3_2_8b" / "granite_3_2_2b" so the wire id matched the model
// then in play. 2026-05-20: the wire ids were generalised again to
// "critic_a" / "critic_b" so future model swaps don't drag the wire id
// with them. We still render a human-friendly markdown block that uses
// `display_name` (the model's human-readable label) for the critic
// column. The raw JSON payload is preserved inside a collapsed <details>
// block so developers can still see everything.

/** Short emoji verdict chip for the header line. */
function verdictEmoji(v: Verdict): string {
  return VERDICT_CHIPS[v];
}

/**
 * Escape a free-form string so it's safe to drop into a Markdown pipe
 * table cell. Mirrors the conventions used in
 * second-opinion/consult.ts → renderDisputesMarkdown:
 *   - backslashes doubled
 *   - `|` replaced with `\|` (otherwise it terminates the cell)
 *   - newlines replaced with `<br>` (pipe tables can't have literal newlines;
 *     LM Studio honours inline HTML)
 */
function escapeMarkdownCell(value: string | undefined): string {
  if (value === undefined || value === null) return "";
  return String(value)
    .replace(/\\/g, "\\\\")
    .replace(/\|/g, "\\|")
    .replace(/\r?\n/g, "<br>")
    .trim();
}

/** Truncate a long cell value with ellipsis so the markdown table stays scannable. */
function truncateCell(s: string, max = 160): string {
  if (s.length <= max) return s;
  return s.slice(0, max - 1).trimEnd() + "…";
}

/** Build the A/B label row. Uses display_name when available, falls back to id. */
function criticLabel(c: CriticResult): string {
  return c.display_name || c.id;
}

/** Short "top concern" column: first concern truncated, or em-dash. */
function topConcernCell(c: CriticResult): string {
  const first = c.concerns[0];
  if (!first) return "—";
  return `"${truncateCell(escapeMarkdownCell(first), RENDER_CELL_TOPIC_CHARS)}"`;
}

/**
 * Map the wire enum `Disagreement.kind` to a plain-English label using the
 * critics' actual display names. Replaces the prior raw enum values
 * (concern-only-in-a / concern-only-in-b / verdict-mismatch) that confused
 * end users who couldn't tell which critic was "a" vs "b".
 */
function humanKindLabel(
  kind: Disagreement["kind"],
  aLabel: string,
  bLabel: string
): string {
  switch (kind) {
    case "verdict-mismatch":
      return "Verdicts differ";
    case "concern-only-in-a":
      return `Only ${aLabel} raised`;
    case "concern-only-in-b":
      return `Only ${bLabel} raised`;
  }
}

/**
 * Render the disputes sub-table, mirroring /second's table_md style.
 * Returns a ready-to-paste markdown fragment. No leading/trailing blank
 * lines -- caller controls spacing.
 *
 * Columns: Kind (human-readable) | Critic A cell | Critic B cell | Severity.
 * A cell shows "verdict (N/5) -- concern text" when the critic raised one,
 * or "(no concern)" when the disagreement is concern-only-in-other.
 */
function renderDisputesSubTable(
  disputes: Disagreement[],
  aLabel: string,
  bLabel: string
): string {
  const rows: string[] = [];
  rows.push(`| Kind | ${escapeMarkdownCell(aLabel)} | ${escapeMarkdownCell(bLabel)} | Severity |`);
  rows.push(`| --- | --- | --- | --- |`);
  const cellFor = (side: { verdict: string; severity: number; concern?: string }) => {
    if (side.concern !== undefined) {
      return `${side.verdict} (${side.severity}/5) -- "${truncateCell(escapeMarkdownCell(side.concern), RENDER_CELL_TOPIC_CHARS)}"`;
    }
    // The critic didn't raise this specific concern. Show their headline
    // verdict + severity so users see what they DID say, not just a bare
    // word that floats without context.
    return `${side.verdict} (${side.severity}/5) -- _(no matching concern)_`;
  };
  for (const d of disputes) {
    rows.push(
      `| ${humanKindLabel(d.kind, aLabel, bLabel)} | ${cellFor(d.critic_a)} | ${cellFor(d.critic_b)} | ${d.severity} |`
    );
  }
  return rows.join("\n");
}

/**
 * Render a VerifyOutput as a human-readable markdown block. The worker
 * pastes the return value verbatim into chat; see src/index.ts tool
 * description.
 *
 * Shape (roughly):
 *   **Verdict: ✅ pass** (consensus)
 *
 *   | Critic | Model | Verdict | Severity | Top concern |
 *   |---|---|---|---|---|
 *   | A | IBM Granite 3.2 8B | pass | 0 | — |
 *   | B | IBM Granite 3.2 2B | warn | 2 | "…" |
 *
 *   - **Recompute:** N verifications, M mismatches
 *   - **NLI:** N contradictions, M unsupported          (omitted if off)
 *   - **Disputes:** N  [+ sub-table when N > 0]
 *
 *   <details><summary>Raw JSON output</summary>
 *
 *   ```json
 *   {…full payload…}
 *   ```
 *   </details>
 */
// Pass / warn / fail / skipped chip for inline use in the Verity testing
// table. Skipped is used when a check didn't run by design (e.g. deep-mode
// signals in standard mode, NLI disabled, NLI without prior context).
function checkChip(
  state: "pass" | "warn" | "fail" | "skipped" | "error" | "n_a" | "unable"
): string {
  return CHECK_CHIPS[state];
}

export function renderSummaryMarkdown(
  output: VerifyOutput,
  context: { answer?: string } = {}
): string {
  const criticEntries = Object.entries(output.critics) as Array<
    [string, CriticResult]
  >;

  const parts: string[] = [];

  // 0. Brief paste-marker — one short line so Qwen doesn't have to
  //    reconcile a long meta-directive with the tool description.
  //    Earlier versions had a paragraph-length 🚫 preface here; that
  //    triggered ~7 seconds of meta-cognition in Qwen 3.5 9B before
  //    each paste, and the resulting "what should I do" thinking
  //    sometimes consumed enough of the output budget that the block
  //    never got pasted. Shortened 2026-05-12 to a single declarative
  //    line. The "paste verbatim / no redraft" rules live in the tool
  //    description; saying them twice was making the model deliberate
  //    rather than act.
  parts.push("_Paste this block into your reply (LM Studio collapses tool results)._");
  parts.push("");

  // 0b. Optionally echo the verified answer. Default OFF
  // (SUMMARY_ECHO_ANSWER). The concise /verify style restates only the
  // critics table and the bold conclusion; the worker shows its own answer
  // per the FLOW / system prompt. Set VERITY_SUMMARY_ECHO_ANSWER=1 to restate
  // the answer inside the block (the 2026-05-11 v4 behaviour), for a worker
  // that calls verify_answer without first emitting a visible answer.
  if (
    SUMMARY_ECHO_ANSWER &&
    context.answer &&
    context.answer.trim().length > 0
  ) {
    parts.push("## Answer");
    parts.push("");
    parts.push(context.answer.trim());
    parts.push("");
  }

  // 1. Verity testing table — ALWAYS the first thing the user sees.
  //    Johnny's system-prompt rule (2026-05-11 v2): "the first thing to
  //    show the user ALWAYS is a table". The verdict chip used to appear
  //    as a separate header line above the table, but that pushed the
  //    table to position 2; now the verdict chip is incorporated into
  //    the bold conclusion at the bottom of the block instead.
  parts.push("### Verity testing");
  parts.push("");
  const testingRows: string[] = [];
  testingRows.push(`| Check | Outcome | Detail |`);
  testingRows.push(`| --- | --- | --- |`);

  // 2a. Critics -- one row per critic.
  //
  // 2026-05-11 v3: "pass with concerns" no longer renders as ✅ pass.
  // A critic that returns verdict="pass" but raises a concern (typical
  // example: "the sources are not detailed enough for verification") is
  // NOT endorsing the answer — it's flagging that it could not fully
  // verify. That maps to "❓ unable to assess", not "✅ pass". A true
  // pass requires verdict="pass" AND no concerns raised.
  criticEntries.forEach(([, c]) => {
    const model = escapeMarkdownCell(criticLabel(c));
    if (c.unavailable) {
      testingRows.push(
        `| **${model}** (critic) | ${checkChip("error")} | _${escapeMarkdownCell(c.error ?? "no response")}_ |`
      );
      return;
    }
    let state: "pass" | "warn" | "fail" | "unable";
    if (c.verdict === "fail") state = "fail";
    else if (c.verdict === "warn") state = "warn";
    else if (c.concerns.length > 0) state = "unable";
    else state = "pass";
    const sevLabel =
      state === "unable"
        ? checkChip(state)
        : `${checkChip(state)} (${c.severity}/5)`;
    const detail = c.concerns[0]
      ? `"${truncateCell(escapeMarkdownCell(c.concerns[0]), RENDER_CELL_CONCERN_CHARS)}"`
      : "no concern raised";
    testingRows.push(`| **${model}** (critic) | ${sevLabel} | ${detail} |`);
  });

  // 2b. Recompute pass.
  //
  // 2026-05-11 v3: when the answer contains no arithmetic, recompute has
  // nothing to verify, so showing "✅ pass" is misleading — recompute
  // didn't actually do anything. Renders "— N/A" in that case. A real
  // pass requires expressions were found AND all verified clean.
  const rc = output.recompute;
  if (rc && rc.ran) {
    if (rc.verifications.length === 0) {
      testingRows.push(
        `| **Recompute pass** (deterministic) | ${checkChip("n_a")} | no arithmetic in the answer to verify |`
      );
    } else {
      const state = rc.mismatches.length > 0 ? "fail" : "pass";
      const detail =
        rc.mismatches.length > 0
          ? `${rc.mismatches.length} mismatch(es) out of ${rc.verifications.length} expression(s)`
          : `${rc.verifications.length} expression(s) verified, 0 mismatches`;
      testingRows.push(`| **Recompute pass** (deterministic) | ${checkChip(state)} | ${detail} |`);
    }
  } else {
    testingRows.push(`| **Recompute pass** (deterministic) | ${checkChip("skipped")} | did not run |`);
  }

  // 2c. NLI claim-checker.
  const nli = output.nli_check;
  if (NLI_IMPL === "off") {
    testingRows.push(`| **NLI claim-checker** | ${checkChip("skipped")} | disabled at server (NLI_IMPL=off) |`);
  } else if (!nli || nli.ran === false) {
    const note = nli && nli.notes ? escapeMarkdownCell(nli.notes) : "did not run";
    testingRows.push(`| **NLI claim-checker** | ${checkChip("skipped")} | ${note} |`);
  } else {
    const cs = nli.contradictions.length;
    const us = nli.unsupported.length;
    let state: "pass" | "warn" | "fail";
    if (cs > 0) state = "fail";
    else if (us >= 2) state = "warn";
    else state = "pass";
    const detail = `${cs} contradiction(s), ${us} unsupported claim(s) across ${nli.claims_checked} claim(s)`;
    testingRows.push(`| **NLI claim-checker** | ${checkChip(state)} | ${detail} |`);
  }

  // Mode the pipeline ran in -- used to compose accurate "why skipped"
  // messages for the deep-mode signals. "standard" means deep signals
  // weren't enabled by the request; "deep"/"deeper" means they were
  // enabled but failed to run (worker doesn't support logprobs etc).
  const pipelineMode = output.meta?.mode ?? "standard";

  // Helper: build the "why a deep-mode signal didn't run" string.
  // Same three-branch decision for consistency and perplexity, with
  // signal-specific final fallback wording.
  const deepSkippedReason = (
    signal: { notes?: string } | undefined,
    defaultReason: string
  ): string => {
    if (pipelineMode === "standard") return "not enabled in standard mode";
    if (signal && signal.notes) return escapeMarkdownCell(signal.notes);
    return defaultReason;
  };

  // 2d. Consistency (deep modes only).
  // 2026-05-12 (E8): use the imported CONSISTENCY_*_THRESHOLD constants
  // instead of the hardcoded 0.5 / 0.15 literals so a user who tunes
  // the config knob sees the chip colour match the actual verdict.
  const consistency = output.consistency_check;
  if (consistency?.ran) {
    const div = consistency.divergence_score;
    let state: "pass" | "warn" | "fail";
    if (div >= CONSISTENCY_FAIL_THRESHOLD) state = "fail";
    else if (div >= CONSISTENCY_WARN_THRESHOLD) state = "warn";
    else state = "pass";
    const detail = `divergence ${div}, ${consistency.contradicted.length} contradicted, ${consistency.unsupported.length} unsupported across ${consistency.samples_generated} re-sample(s)`;
    testingRows.push(`| **Consistency** (deep mode) | ${checkChip(state)} | ${detail} |`);
  } else {
    const reason = deepSkippedReason(consistency, "did not run");
    testingRows.push(`| **Consistency** (deep mode) | ${checkChip("skipped")} | ${reason} |`);
  }

  // 2e. Perplexity / model uncertainty (deep modes only). ADVISORY: this row
  //     is a nudge and does NOT change the consensus (2026-05-22) -- logprob
  //     uncertainty misses fluent hallucinations and fires on rare-but-correct
  //     wording, so it informs rather than votes.
  const perplexity = output.perplexity;
  if (perplexity?.ran) {
    const n = perplexity.low_confidence_spans.length;
    const state = n > 0 ? "warn" : "pass";
    const detail = `perplexity ${perplexity.perplexity}, ${n} low-confidence span(s); advisory only, does not change the verdict`;
    testingRows.push(`| **Perplexity** (deep mode, advisory) | ${checkChip(state)} | ${detail} |`);
  } else {
    const reason = deepSkippedReason(
      perplexity,
      "did not run (worker may not support logprobs)"
    );
    testingRows.push(`| **Perplexity** (deep mode, advisory) | ${checkChip("skipped")} | ${reason} |`);
  }

  parts.push(testingRows.join("\n"));

  // 3. Findings -- full text of every concern, mismatch, contradiction.
  //    Always rendered so users see what triggered the verdict (or, on a
  //    pass, the explicit "nothing raised" statement). This is the "why".
  const findings: string[] = [];

  // Helper: format a piece of free-form text for a Findings BULLET (not a
  // table cell). Unlike escapeMarkdownCell, we don't replace newlines with
  // <br> -- bullets can carry multiple lines via continuation indentation,
  // and converting newlines to <br> mangles claims that legitimately
  // contain table rows or paragraph breaks. We collapse internal whitespace
  // runs so the result is one scannable paragraph but preserve the original
  // characters. Truncation budget is much larger here (800) than the table
  // cells' 140 -- bullets aren't space-constrained.
  const findingText = (raw: string | undefined, max = 800): string => {
    if (!raw) return "";
    // Collapse all whitespace runs (incl. newlines) to single spaces. This
    // unrolls a multi-row table into one line, but at least no mid-row
    // truncation produces broken markdown.
    const flat = String(raw).replace(/\s+/g, " ").trim();
    return truncateCell(flat, max);
  };

  // Critic concerns in full.
  //
  // 2026-05-11 v3: "verdict=pass but concerns raised" is rendered in the
  // table as "❓ unable to assess" (the critic didn't endorse, it just
  // couldn't fully verify). Mirror that wording here in Findings — the
  // prior "raised — pass (severity 0/5):" read as the critic having
  // approved, which is the opposite of what it actually said.
  criticEntries.forEach(([, c]) => {
    if (c.unavailable || c.concerns.length === 0) return;
    const label =
      c.verdict === "pass"
        ? "unable to assess"
        : `${c.verdict} (severity ${c.severity}/5)`;
    findings.push(
      `- **${criticLabel(c)}** — ${label}: "${findingText(c.concerns[0], RENDER_FINDING_CONCERN_CHARS)}"`
    );
    for (const extra of c.concerns.slice(1, 4)) {
      findings.push(
        `    - also: "${findingText(extra, RENDER_FINDING_EXTRA_CHARS)}"`
      );
    }
    if (c.suggested_fixes && c.suggested_fixes[0]) {
      findings.push(
        `    - _suggested fix:_ "${findingText(c.suggested_fixes[0], RENDER_FINDING_EXTRA_CHARS)}"`
      );
    }
  });

  // Recompute mismatches.
  // 2026-05-12 (E7): strip backticks from expr_text before wrapping
  // in inline-code backticks. A `expr_text` containing a literal
  // backtick (rare but possible if the critic's regex picks up a
  // markdown-formatted expression) would close the inline-code span
  // prematurely and leak escape characters into the rest of the
  // bullet. Same defence the consult.ts table renderers use.
  if (rc?.ran && rc.mismatches.length > 0) {
    for (const m of rc.mismatches.slice(0, 5)) {
      const exprText = ((m as { expr_text?: string }).expr_text ?? "").replace(
        /`/g,
        ""
      );
      findings.push(
        `- **Recompute** found arithmetic mismatch: \`${findingText(exprText, RENDER_FINDING_RECOMPUTE_CHARS)}\``
      );
    }
  }

  // NLI contradictions + unsupported. Raised the truncation budget from
  // 240 to 1000 chars in the bullet so long claims (e.g. when claim
  // extraction grabbed a multi-row table as one unit) don't get clipped
  // mid-row. Newlines inside the claim are collapsed to spaces.
  if (nli?.ran) {
    for (const c of nli.contradictions.slice(0, 5)) {
      const claim = (c as { claim?: string }).claim ?? "";
      findings.push(
        `- **NLI** found a claim CONTRADICTED by prior context: "${findingText(claim, RENDER_FINDING_NLI_CLAIM_CHARS)}"`
      );
    }
    for (const u of nli.unsupported.slice(0, 3)) {
      const claim = (u as { claim?: string }).claim ?? "";
      findings.push(
        `- **NLI** found a claim unsupported by prior context: "${findingText(claim, RENDER_FINDING_NLI_CLAIM_CHARS)}"`
      );
    }
  }

  // Disputes count (one line, always shown so users know whether to look
  // across critics for disagreement).
  const disputeCount = output.disputes.length;
  if (disputeCount > 0) {
    findings.push(
      `- Critics disagreed on ${disputeCount} point(s) — compare rows in the table above.`
    );
  }

  parts.push("");
  if (findings.length === 0) {
    parts.push("**Findings:** none raised by any check.");
  } else {
    parts.push("**Findings:**");
    parts.push("");
    parts.push(findings.join("\n"));
  }

  // 4. Bold conclusion — a one-line bolded sentence placed UNDER the
  //    table+findings as the considered-after-evidence TL;DR. Now also
  //    carries the verdict emoji chip (since the top-of-block verdict
  //    header was removed so the table can be the first thing). Format:
  //      "**<emoji> <verdict> — <summary>**"
  //    Always emitted so users see a clear "what does this all add up to"
  //    line even on a clean pass.
  const verdictChip = verdictEmoji(output.consensus);
  const conclusionBody =
    output.summary && output.summary.trim().length > 0
      ? `${verdictChip} — ${escapeMarkdownCell(output.summary)}`
      : verdictChip;
  parts.push("");
  parts.push(`**${conclusionBody}**`);

  // 5. Follow-up prompt — Verity does NOT redraft answers or run deeper
  //    checks automatically. When findings warrant, offer the user a
  //    three-way choice between (a) a redraft that addresses the
  //    findings, (b) /verifydeeper for additional checks the standard
  //    pipeline didn't run (consistency re-sampling + perplexity), or
  //    (c) leaving the answer as-is.
  //
  //    2026-05-11 v3: this prompt is the self-sufficient mechanism for
  //    /verifydeeper follow-up — moved out of the README-only system
  //    prompt so Verity works without any user-side prompt configuration.
  //    Trigger: at least one item in the Findings list.
  if (findings.length > 0) {
    // In deeper mode, /verifydeeper has already run — don't offer it again.
    const deeperAlreadyRan = pipelineMode === "deeper";
    parts.push("");
    if (deeperAlreadyRan) {
      parts.push(
        "> ⏸ **Awaiting your reply.** Type `redraft` to have the " +
          "answer rewritten to address the findings above (every URL " +
          "in the redraft MUST be fetched first via the fetch tool to " +
          "confirm it resolves — do not invent plausible-looking URLs), " +
          "or `no` to leave the answer as-is."
      );
    } else {
      parts.push(
        "> ⏸ **Awaiting your reply.** Type one of:"
      );
      parts.push(
        "> - `redraft` — rewrite the answer to address the findings " +
          "above. **Every URL in the redraft MUST be fetched first** " +
          "via the fetch tool to confirm it resolves. Do not invent " +
          "URLs to address a 'needs more sources' finding — fabricated " +
          "URLs are worse than no URLs. If you can't find a working " +
          "source for a claim, drop the claim."
      );
      parts.push(
        "> - `/verifydeeper` — run deeper checks (consistency re-sampling " +
          "+ perplexity) that this run skipped"
      );
      parts.push("> - `no` — leave the answer as-is");
    }
  }

  // 6. Latency footer.
  parts.push("");
  parts.push(
    `_${output.latency_ms} ms · mode=${output.meta.mode} · task_type=${output.meta.task_type}_`
  );

  // Note: the "End of /verify turn" marker that used to sit here was
  // removed 2026-05-12 as part of the meta-instruction reduction. The
  // "Awaiting your reply" prompt above already signals turn boundary;
  // saying it twice was contributing to Qwen's meta-cognition spiral.

  return parts.join("\n");
}
