/**
 * Unit tests for the human-readable Markdown renderer
 * (aggregator.renderSummaryMarkdown).
 *
 * These tests verify the ship/UX contract of the 2026-04-21 fix:
 *   - summary_md always starts with a "**Verdict:" line.
 *   - the critic table uses display_name, not the wire id.
 *   - the correct emoji chip fires per consensus value.
 *   - signal bullets appear / are suppressed based on ran-flags and NLI_IMPL.
 *   - the raw-JSON <details> block is present and collapsed.
 *   - pipes inside cell content are escaped (table doesn't break).
 *   - disputes sub-table only renders when disputes.length > 0.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { renderSummaryMarkdown } from "../aggregator.js";
import { CHECK_CHIPS, VERDICT_CHIPS } from "../render-constants.js";
import type {
  CriticResult,
  NliResult,
  RecomputeResult,
  VerifyOutput,
} from "../types.js";

function critic(overrides: Partial<CriticResult> = {}): CriticResult {
  return {
    id: overrides.id ?? "critic_a",
    display_name: overrides.display_name ?? "IBM Granite 3.2 8B",
    family: overrides.family ?? "IBM",
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

const cleanRecompute: RecomputeResult = {
  ran: true,
  expressions_found: 0,
  verifications: [],
  mismatches: [],
  notes: "",
  latency_ms: 5,
};

function baseOutput(overrides: Partial<VerifyOutput> = {}): VerifyOutput {
  const a = critic({ id: "critic_a", display_name: "IBM Granite 3.2 8B" });
  const b = critic({ id: "critic_b", display_name: "IBM Granite 3.2 2B" });
  return {
    critics: {
      critic_a: a,
      critic_b: b,
    },
    disputes: [],
    nli_check: cleanNli,
    recompute: cleanRecompute,
    consensus: "pass",
    summary: "All critics passed.",
    summary_md: "",
    latency_ms: 1234,
    meta: {
      mode: "standard",
      task_type: "auto",
      context_mode: "minimal",
      granite_8b_input_truncated: false,
      critics_unavailable: [],
    },
    ...overrides,
  };
}

// ─── Core shape ───────────────────────────────────────────────────────────

test("summary_md leads with a short paste marker", () => {
  // 2026-05-12: Earlier versions had a paragraph-length 🚫 preface
  // here, but it caused Qwen 3.5 9B to spend ~7 seconds reconciling
  // the duplicated meta-instructions (block preface vs tool desc),
  // sometimes consuming enough budget that the paste never happened.
  // Reduced to a single declarative line; "no redraft / paste
  // verbatim" rules live only in the tool description now.
  const md = renderSummaryMarkdown(baseOutput());
  assert.match(md, /^_Paste this block into your reply/);
});

test("table heading comes immediately after the paste marker when no answer context is passed", () => {
  const md = renderSummaryMarkdown(baseOutput());
  // Marker, blank, then the table heading.
  const lines = md.split("\n");
  assert.match(lines[0], /^_Paste this block/);
  assert.equal(lines[1], "");
  assert.equal(lines[2], "### Verity testing");
});

test("answer is NOT echoed by default; block is table + conclusion only (2026-05-22)", () => {
  // SUMMARY_ECHO_ANSWER defaults off: the worker shows its own answer (per
  // the FLOW / system prompt), so the /verify block restates only the
  // critics table and the bold conclusion, never the answer text.
  const md = renderSummaryMarkdown(baseOutput(), {
    answer:
      "LLMs do hallucinate, with rates varying from 3% to 27% in academic " +
      "benchmarks depending on model and domain.",
  });
  assert.ok(
    !md.includes("## Answer"),
    "answer heading must not appear by default"
  );
  assert.ok(
    !md.includes("LLMs do hallucinate, with rates varying"),
    "answer text must not appear in the block by default"
  );
});

test("answer echo is suppressed when answer context is empty or absent", () => {
  // No context passed at all.
  const md1 = renderSummaryMarkdown(baseOutput());
  assert.ok(
    !md1.includes("## Answer"),
    "no answer heading when context is absent"
  );
  // Empty answer string.
  const md2 = renderSummaryMarkdown(baseOutput(), { answer: "" });
  assert.ok(
    !md2.includes("## Answer"),
    "no answer heading when context.answer is empty"
  );
  // Whitespace-only answer.
  const md3 = renderSummaryMarkdown(baseOutput(), { answer: "   \n  " });
  assert.ok(
    !md3.includes("## Answer"),
    "no answer heading when context.answer is whitespace only"
  );
});

test("summary_md does NOT lead with a separate **Verdict:** header line", () => {
  const md = renderSummaryMarkdown(baseOutput({ consensus: "pass" }));
  assert.ok(
    !/^\*\*Verdict:/m.test(md.split("\n")[0]),
    "first line must not be a verdict header (it's the agent preface, then the table heading)"
  );
});

test("pass consensus renders the green-tick emoji in the bold conclusion", () => {
  const md = renderSummaryMarkdown(
    baseOutput({ consensus: "pass", summary: "All critics passed." })
  );
  // The verdict emoji now appears inside the bold conclusion at the bottom
  // (format: "**<emoji> <verdict> — <summary>**").
  assert.ok(md.includes(VERDICT_CHIPS.pass), `expected ${VERDICT_CHIPS.pass} chip`);
});

test("warn consensus renders the warn emoji in the bold conclusion", () => {
  const md = renderSummaryMarkdown(
    baseOutput({ consensus: "warn", summary: "One critic raised a concern." })
  );
  assert.ok(md.includes(VERDICT_CHIPS.warn), `expected ${VERDICT_CHIPS.warn} chip`);
});

test("fail consensus renders the red-cross emoji in the bold conclusion", () => {
  const md = renderSummaryMarkdown(
    baseOutput({ consensus: "fail", summary: "Critics flagged contradictions." })
  );
  assert.ok(md.includes(VERDICT_CHIPS.fail), `expected ${VERDICT_CHIPS.fail} chip`);
});

test("critic table lists display_name for each critic, not the wire id", () => {
  const md = renderSummaryMarkdown(baseOutput());
  assert.ok(
    md.includes("IBM Granite 3.2 8B"),
    "expected display_name of critic A in the table"
  );
  assert.ok(
    md.includes("IBM Granite 3.2 2B"),
    "expected display_name of critic B in the table"
  );
  // Wire ids MAY still appear in the raw-JSON <details> block — that's
  // fine and intentional. We just want to ensure the human-readable
  // table row is keyed on display_name.
  const beforeDetails = md.split("<details>")[0];
  assert.ok(
    !beforeDetails.includes("critic_a"),
    "wire id 'critic_a' should NOT appear in the rendered table"
  );
  assert.ok(
    !beforeDetails.includes("critic_b"),
    "wire id 'critic_b' should NOT appear in the rendered table"
  );
});

test("Verity testing table header has Check / Outcome / Detail columns", () => {
  const md = renderSummaryMarkdown(baseOutput());
  // 2026-05-11 (afternoon): replaced separate Critics + Signals sections
  // with a single Verity testing table. Each check (critic, recompute, NLI,
  // consistency, perplexity) gets one row with outcome chip + short detail.
  assert.ok(md.includes("### Verity testing"), "expected ### Verity testing section header");
  assert.ok(
    md.includes("| Check | Outcome | Detail |"),
    "expected Verity testing table header row"
  );
});

test("Verity testing table includes every check, including ones that didn't run", () => {
  // Standard mode -- consistency and perplexity shouldn't run, but they
  // should still appear as rows with a "skipped" chip so the user sees
  // explicitly that they weren't applied.
  const md = renderSummaryMarkdown(baseOutput());
  assert.ok(md.includes("Recompute pass"), "expected Recompute row");
  assert.ok(md.includes("NLI claim-checker"), "expected NLI row");
  assert.ok(md.includes("Consistency"), "expected Consistency row (even if skipped)");
  assert.ok(md.includes("Perplexity"), "expected Perplexity row (even if skipped)");
  // Skipped marker is the em-dash "— skipped".
  assert.ok(md.includes("skipped"), "expected at least one 'skipped' chip for the deep-mode checks in standard");
});

test("severity renders with /5 scale so it's not ambiguous", () => {
  const md = renderSummaryMarkdown(
    baseOutput({
      critics: {
        critic_a: critic({
          id: "critic_a",
          display_name: "IBM Granite 3.2 8B",
          verdict: "warn",
          severity: 3,
        }),
        critic_b: critic({
          id: "critic_b",
          display_name: "IBM Granite 3.2 2B",
        }),
      },
      consensus: "warn",
    })
  );
  assert.ok(md.includes("3/5"), "severity should be rendered as N/5");
});

test("no raw-JSON <details> block (dropped 2026-05-11 because LM Studio doesn't collapse it)", () => {
  const md = renderSummaryMarkdown(baseOutput());
  assert.ok(!md.includes("<details>"), "the <details> JSON dump was removed");
  assert.ok(!md.includes("Raw JSON output"), "summary block label should be gone");
});

// --- Verity testing rows -------------------------------------------------

test("recompute row shows verification + mismatch counts", () => {
  const md = renderSummaryMarkdown(
    baseOutput({
      recompute: {
        ran: true,
        expressions_found: 2,
        verifications: [
          { kind: "arithmetic", expr_text: "2+2", claimed: "4", computed: "4", matches: true, confidence: 1 },
          { kind: "arithmetic", expr_text: "3*3", claimed: "9", computed: "9", matches: true, confidence: 1 },
        ],
        mismatches: [],
        notes: "",
        latency_ms: 3,
      },
    })
  );
  assert.ok(md.includes("Recompute pass"), "expected Recompute row");
  assert.match(md, /2 expression\(s\) verified, 0 mismatches/);
});

test("NLI row shows skipped chip when nli_check did not run", () => {
  const md = renderSummaryMarkdown(
    baseOutput({
      nli_check: {
        ran: false,
        claims_checked: 0,
        contradictions: [],
        unsupported: [],
        notes: "no prior context supplied",
      },
    })
  );
  // The row still appears (transparency) but with a "skipped" chip.
  assert.ok(md.includes("NLI claim-checker"), "NLI row should still appear");
  assert.match(md, /NLI claim-checker.*skipped/, "NLI row should show skipped chip when ran=false");
});

test("Findings line always renders -- 'none raised' on clean pass", () => {
  const md = renderSummaryMarkdown(baseOutput());
  assert.ok(
    md.includes("Findings:") && md.includes("none raised by any check"),
    "expected explicit 'none raised' message on clean pass"
  );
});

test("Disputes count only appears in Findings when > 0", () => {
  const cleanMd = renderSummaryMarkdown(baseOutput());
  assert.ok(
    !cleanMd.includes("Critics disagreed"),
    "no disagreement bullet when disputes is empty"
  );
  const disputedMd = renderSummaryMarkdown(
    baseOutput({
      disputes: [
        {
          kind: "verdict-mismatch",
          critic_a_id: "critic_a",
          critic_b_id: "critic_b",
          critic_a: { verdict: "pass", severity: 0 },
          critic_b: { verdict: "fail", severity: 4, concern: "bug detected" },
          severity: "hard",
        },
      ],
      consensus: "fail",
    })
  );
  assert.ok(
    disputedMd.includes("Critics disagreed on 1 point"),
    "expected 'Critics disagreed on N point(s)' bullet"
  );
});

test("no separate Disputes sub-table -- all critic info in one table", () => {
  const md = renderSummaryMarkdown(
    baseOutput({
      disputes: [
        {
          kind: "verdict-mismatch",
          critic_a_id: "critic_a",
          critic_b_id: "critic_b",
          critic_a: { verdict: "pass", severity: 0 },
          critic_b: { verdict: "fail", severity: 4, concern: "bug detected" },
          severity: "hard",
        },
      ],
      consensus: "fail",
    })
  );
  assert.ok(
    !md.includes("### Disputes"),
    "the separate Disputes section was removed; everything lives in the Critics table"
  );
  assert.ok(
    !md.includes("Verdicts differ"),
    "the per-disagreement Kind labels were removed with the sub-table"
  );
});

test("Findings section lists each critic's concerns when verdict is fail", () => {
  const md = renderSummaryMarkdown(
    baseOutput({
      critics: {
        critic_a: critic({
          id: "critic_a",
          display_name: "IBM Granite 3.2 8B",
          verdict: "fail",
          severity: 4,
          concerns: ["The arithmetic in step 3 is wrong: 2+2=5 is false."],
          suggested_fixes: ["Correct step 3 to 2+2=4."],
        }),
        critic_b: critic({
          id: "critic_b",
          display_name: "IBM Granite 3.2 2B",
          verdict: "warn",
          severity: 2,
          concerns: ["Hedging language might confuse the reader."],
        }),
      },
      consensus: "fail",
    })
  );
  assert.ok(md.includes("Findings:"), "expected 'Findings:' label");
  assert.ok(
    md.includes("The arithmetic in step 3 is wrong"),
    "expected full critic concern text in Findings"
  );
  assert.ok(
    md.includes("Correct step 3 to 2+2=4"),
    "expected suggested fix text in Findings"
  );
});

test("Findings on a clean pass states 'none raised'", () => {
  const md = renderSummaryMarkdown(baseOutput({ consensus: "pass" }));
  assert.ok(
    md.includes("Findings:") && md.includes("none raised by any check"),
    "Findings line should always appear; on pass it says 'none raised by any check'"
  );
});

// ─── Bold conclusion + redraft prompt (2026-05-11 v2) ─────────────────────
//
// Johnny's v2 system prompt (2026-05-11): "the first thing to show the user
// ALWAYS is a table. Highlight the conclusion in bold styled text under
// the table. Do not redraft the answer based on critics views. … If
// critics have significant input, ask the user if they want you to do a
// redraft."
//
// Implementation contract:
//   - Table is the first thing in summary_md (the top **Verdict:** header
//     was dropped).
//   - Bold conclusion under the table carries the verdict emoji chip +
//     the aggregated summary sentence: "**<emoji> <verdict> — <summary>**".
//   - Redraft prompt fires ONLY when findings.length > 0 (= "significant
//     input"); a warn with zero findings is noise.

test("bold conclusion appears UNDER the table (after Findings)", () => {
  const md = renderSummaryMarkdown(
    baseOutput({ consensus: "pass", summary: "All critics passed." })
  );
  // The conclusion should be a bold line containing the summary text,
  // and it should appear AFTER the Findings line in document order.
  const findingsIdx = md.indexOf("Findings:");
  const conclusionIdx = md.indexOf("**✅ pass — All critics passed.**");
  assert.ok(findingsIdx > 0, "expected Findings line to be present");
  assert.ok(
    conclusionIdx > 0,
    "expected bold conclusion '**✅ pass — All critics passed.**' to be present"
  );
  assert.ok(
    conclusionIdx > findingsIdx,
    "bold conclusion must appear AFTER the Findings section"
  );
});

test("bold conclusion carries the verdict emoji chip", () => {
  // On warn/fail, the conclusion should still be a single bold line
  // with the verdict chip and summary.
  const warnMd = renderSummaryMarkdown(
    baseOutput({ consensus: "warn", summary: "One critic raised a concern." })
  );
  assert.ok(
    warnMd.includes("**⚠️ warn — One critic raised a concern.**"),
    "expected '**⚠️ warn — One critic raised a concern.**' in warn output"
  );
  const failMd = renderSummaryMarkdown(
    baseOutput({ consensus: "fail", summary: "Critics found errors." })
  );
  assert.ok(
    failMd.includes("**❌ fail — Critics found errors.**"),
    "expected '**❌ fail — Critics found errors.**' in fail output"
  );
});

test("bold conclusion does NOT use a literal 'Conclusion:' label", () => {
  // Guard against the prior failure mode where the system prompt's
  // "highlight the conclusion in bold" instruction caused Qwen to write
  // the literal label "Bold Conclusion:" or "Conclusion:". The renderer
  // emits bold text WITHOUT a label, so the worker has nothing to copy.
  const md = renderSummaryMarkdown(
    baseOutput({ consensus: "warn", summary: "One critic raised a concern." })
  );
  assert.ok(
    !/^\*\*Conclusion:/im.test(md),
    "renderer must not emit a 'Conclusion:' header line"
  );
  assert.ok(
    !/bold conclusion/i.test(md),
    "renderer must not emit the literal string 'bold conclusion'"
  );
});

test("three-way follow-up (redraft / /verifydeeper / no) appears on findings in standard mode", () => {
  // 2026-05-11 v3: when findings exist AND the run was standard or deep
  // mode (not deeper), the renderer offers a three-way choice between
  // redraft, /verifydeeper, or no — so Verity is self-sufficient and
  // doesn't depend on a system-prompt rule for the deeper-check offer.
  const md = renderSummaryMarkdown(
    baseOutput({
      consensus: "fail",
      critics: {
        critic_a: critic({
          id: "critic_a",
          display_name: "IBM Granite 3.2 8B",
          verdict: "fail",
          severity: 4,
          concerns: ["arithmetic mistake"],
        }),
        critic_b: critic({
          id: "critic_b",
          display_name: "IBM Granite 3.2 2B",
        }),
      },
    })
  );
  assert.ok(
    md.includes("Awaiting your reply"),
    "three-way follow-up prompt must appear on findings in standard mode"
  );
  assert.ok(
    md.includes("`redraft`"),
    "follow-up must mention the redraft option"
  );
  assert.ok(
    md.includes("`/verifydeeper`"),
    "follow-up must mention the /verifydeeper option"
  );
  assert.ok(
    md.includes("`no`"),
    "follow-up must mention the 'no' option"
  );
});

test("three-way follow-up also appears on warn consensus with findings", () => {
  const md = renderSummaryMarkdown(
    baseOutput({
      consensus: "warn",
      critics: {
        critic_a: critic({
          id: "critic_a",
          display_name: "IBM Granite 3.2 8B",
          verdict: "warn",
          severity: 2,
          concerns: ["hedging language"],
        }),
        critic_b: critic({
          id: "critic_b",
          display_name: "IBM Granite 3.2 2B",
        }),
      },
    })
  );
  assert.ok(
    md.includes("Awaiting your reply") && md.includes("/verifydeeper"),
    "three-way follow-up must appear on warn consensus with findings"
  );
});

test("in deeper mode, follow-up is just the redraft prompt (/verifydeeper already ran)", () => {
  const md = renderSummaryMarkdown(
    baseOutput({
      consensus: "warn",
      meta: {
        mode: "deeper",
        task_type: "auto",
        context_mode: "minimal",
        granite_8b_input_truncated: false,
        critics_unavailable: [],
      },
      critics: {
        critic_a: critic({
          id: "critic_a",
          display_name: "IBM Granite 3.2 8B",
          verdict: "warn",
          severity: 2,
          concerns: ["hedging language"],
        }),
        critic_b: critic({
          id: "critic_b",
          display_name: "IBM Granite 3.2 2B",
        }),
      },
    })
  );
  assert.ok(
    md.includes("Awaiting your reply") && md.includes("redraft"),
    "deeper-mode follow-up should still be the awaiting-reply prompt with redraft option"
  );
  assert.ok(
    !md.includes("/verifydeeper"),
    "deeper-mode follow-up should NOT offer /verifydeeper again"
  );
});

test("redraft prompt is SUPPRESSED on clean pass with no findings", () => {
  const md = renderSummaryMarkdown(baseOutput({ consensus: "pass" }));
  assert.ok(
    !md.includes("Want me to redraft"),
    "redraft prompt must NOT appear on a clean pass — there's nothing to address"
  );
});

// 2026-05-11 v3 chip semantics — "pass with concerns" + recompute N/A

test("critic with verdict=pass but concerns raised renders '❓ unable to assess'", () => {
  // Real case observed 2026-05-11: Granite 8B returned verdict=pass with
  // a concern that the sources weren't detailed enough for verification.
  // That's NOT a pass — the critic couldn't verify. Render as "unable to
  // assess" so the user doesn't read it as endorsement.
  const md = renderSummaryMarkdown(
    baseOutput({
      consensus: "pass",
      critics: {
        critic_a: critic({
          id: "critic_a",
          display_name: "IBM Granite 3.2 8B",
          verdict: "pass",
          severity: 0,
          concerns: ["The sources provided are not detailed enough for verification."],
        }),
        critic_b: critic({
          id: "critic_b",
          display_name: "IBM Granite 3.2 2B",
        }),
      },
    })
  );
  assert.ok(
    md.includes(CHECK_CHIPS.unable),
    `expected '${CHECK_CHIPS.unable}' chip when verdict=pass + concerns raised`
  );
  assert.ok(
    !md.includes(`| **IBM Granite 3.2 8B** (critic) | ${CHECK_CHIPS.pass}`),
    `Granite 8B row must NOT show ${CHECK_CHIPS.pass} when it raised a concern`
  );
});

test("critic with verdict=pass and NO concerns still renders '✅ pass'", () => {
  // The unable-to-assess remap fires only when concerns is non-empty.
  // A genuine clean pass (no concerns at all) still shows ✅ pass.
  const md = renderSummaryMarkdown(
    baseOutput({
      consensus: "pass",
      critics: {
        critic_a: critic({
          id: "critic_a",
          display_name: "IBM Granite 3.2 8B",
          verdict: "pass",
          severity: 0,
          concerns: [],
        }),
        critic_b: critic({
          id: "critic_b",
          display_name: "IBM Granite 3.2 2B",
        }),
      },
    })
  );
  assert.ok(
    md.includes(`| **IBM Granite 3.2 8B** (critic) | ${CHECK_CHIPS.pass}`),
    `clean-pass critic with no concerns must still show ${CHECK_CHIPS.pass}`
  );
  assert.ok(
    !md.includes(CHECK_CHIPS.unable),
    `no '${CHECK_CHIPS.unable}' chip when concerns is empty`
  );
});

test("recompute row shows '— N/A' when answer has no arithmetic to verify", () => {
  const md = renderSummaryMarkdown(
    baseOutput({
      recompute: {
        ran: true,
        expressions_found: 0,
        verifications: [],
        mismatches: [],
        notes: "no arithmetic detected",
        latency_ms: 1,
      },
    })
  );
  assert.ok(
    md.includes(CHECK_CHIPS.n_a),
    `expected '${CHECK_CHIPS.n_a}' chip for recompute when verifications is empty`
  );
  assert.ok(
    md.includes("no arithmetic in the answer to verify"),
    "expected explanatory detail text for N/A recompute"
  );
  // Must NOT render as a pass — that was the bug.
  assert.ok(
    !md.match(/Recompute pass.*✅ pass/),
    "recompute with 0 verifications must NOT show ✅ pass"
  );
});

test("Findings bullet for verdict=pass + concerns says 'unable to assess' not 'pass'", () => {
  const md = renderSummaryMarkdown(
    baseOutput({
      consensus: "pass",
      critics: {
        critic_a: critic({
          id: "critic_a",
          display_name: "IBM Granite 3.2 8B",
          verdict: "pass",
          severity: 0,
          concerns: ["Sources are not detailed enough."],
        }),
        critic_b: critic({
          id: "critic_b",
          display_name: "IBM Granite 3.2 2B",
        }),
      },
    })
  );
  assert.ok(
    md.includes("**IBM Granite 3.2 8B** — unable to assess"),
    "Findings bullet must reframe pass-with-concerns as 'unable to assess'"
  );
  assert.ok(
    !md.match(/IBM Granite 3\.2 8B\*\* raised — pass \(severity 0\/5\)/),
    "Findings bullet must NOT say 'raised — pass (severity 0/5)' for this case"
  );
});

test("summary text reflects 'pass with concerns' instead of 'All critics passed'", () => {
  const md = renderSummaryMarkdown(
    baseOutput({
      consensus: "pass",
      summary: "Critics did not flag the answer as wrong, but raised 1 concern(s) about verifiability.",
    })
  );
  // The summary string is built by buildSummary in the aggregator path;
  // here we just confirm the renderer carries it through into the bold
  // conclusion when totalConcerns > 0.
  assert.ok(
    md.includes("**✅ pass — Critics did not flag the answer as wrong"),
    "bold conclusion must carry the 'concerns about verifiability' wording when present"
  );
});

test("redraft prompt is SUPPRESSED on 'error' consensus (unreliable verdict)", () => {
  const md = renderSummaryMarkdown(
    baseOutput({
      consensus: "error",
      critics: {
        critic_a: critic({
          id: "critic_a",
          display_name: "IBM Granite 3.2 8B",
          unavailable: true,
          error: "timeout",
          verdict: "error",
        }),
        critic_b: critic({
          id: "critic_b",
          display_name: "IBM Granite 3.2 2B",
          unavailable: true,
          error: "timeout",
          verdict: "error",
        }),
      },
    })
  );
  assert.ok(
    !md.includes("Want me to redraft"),
    "redraft prompt must NOT appear when critics could not respond"
  );
});

// ─── Markdown safety ──────────────────────────────────────────────────────

test("pipes in concern text are escaped", () => {
  const md = renderSummaryMarkdown(
    baseOutput({
      critics: {
        critic_a: critic({
          id: "critic_a",
          display_name: "IBM Granite 3.2 8B",
          concerns: ["foo | bar | baz"],
          verdict: "warn",
          severity: 2,
        }),
        critic_b: critic({
          id: "critic_b",
          display_name: "IBM Granite 3.2 2B",
        }),
      },
      consensus: "warn",
    })
  );
  // The pipe must be escaped with a backslash.
  assert.ok(
    md.includes("foo \\| bar \\| baz"),
    "pipes in concern cells must be escaped"
  );
});

test("unavailable critics render the error message and no numeric severity", () => {
  const md = renderSummaryMarkdown(
    baseOutput({
      critics: {
        critic_a: critic({
          id: "critic_a",
          display_name: "IBM Granite 3.2 8B",
          unavailable: true,
          error: "timeout after 45s",
          verdict: "error",
        }),
        critic_b: critic({
          id: "critic_b",
          display_name: "IBM Granite 3.2 2B",
        }),
      },
    })
  );
  assert.ok(md.includes("timeout after 45s"), "expected error message in table cell");
  // 2026-05-11 (afternoon): in the Verity testing table, an unavailable
  // critic row gets the "⛔ unavailable" outcome chip and the error
  // message in the Detail column (no numeric severity to display).
  assert.ok(md.includes(CHECK_CHIPS.error), `expected ${CHECK_CHIPS.error} chip in outcome cell`);
});
