/**
 * Unit tests for the dual-GPU consult + Phase C analysis pass.
 *
 * Strategy:
 *   - Pure helpers (computeDisputes, extractAnalysisJson, renderDisputesTable)
 *     are tested directly.
 *   - The analysis pass is tested by pointing `runAnalysisPass` at a local
 *     http.Server that replies with a canned OpenAI-compatible chat
 *     completion. This exercises the real JSON-parse path + fence/preamble
 *     stripping + table rendering.
 *   - The full `runSecondOpinion` flow is tested by pointing BOTH endpoints
 *     (Ollama + LM Studio) at local mock servers. The Ollama leg's endpoint
 *     is config-fixed (OLLAMA_URL = http://localhost:11434/v1) so tests
 *     launch their Ollama mock there; LM Studio's port 1234 is used for
 *     the NVIDIA leg. When Ollama isn't running locally this is safe;
 *     when it is, port 11434 is already bound by Ollama so the mock binds
 *     fails — those tests use `callOneBackend` against dead ports instead.
 *
 * Availability-path tests (AMD unavailable, NVIDIA unavailable, both
 * unavailable) use `callOneBackend` against dead ports since that is the
 * cleanest way to produce an unavailable result without network.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import type { AddressInfo } from "node:net";

import {
  callOneBackend,
  computeDisputes,
  extractAnalysisJson,
  MAX_ANALYSIS_DISPUTES,
  renderDisputesTable,
  renderDisputesMarkdown,
  runAnalysisPass,
  type AnalysisDispute,
} from "../second-opinion/consult.js";

// ─────────────────────────────────────────────────────────────────────────
// Mock-server helpers
// ─────────────────────────────────────────────────────────────────────────

/**
 * Minimal OpenAI-compatible chat-completions mock. The `handler` decides
 * what content string to return for each request. Returns the base URL
 * plus a teardown function. Base URL includes `/v1` so it looks like
 * LM Studio / Ollama.
 */
async function startMockChatServer(
  handler: (reqBody: unknown) => { content: string }
): Promise<{ baseURL: string; close: () => Promise<void> }> {
  const server = http.createServer((req, res) => {
    if (req.method !== "POST") {
      res.writeHead(405);
      res.end();
      return;
    }
    const chunks: Buffer[] = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => {
      let reqBody: unknown = undefined;
      try {
        reqBody = JSON.parse(Buffer.concat(chunks).toString("utf8"));
      } catch {
        /* ignore */
      }
      const { content } = handler(reqBody);
      const body = JSON.stringify({
        id: "chatcmpl-mock",
        object: "chat.completion",
        created: 0,
        model: "mock",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
      });
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(body);
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const addr = server.address() as AddressInfo;
  return {
    baseURL: `http://127.0.0.1:${addr.port}/v1`,
    close: () =>
      new Promise<void>((resolve) => server.close(() => resolve())),
  };
}

// ─────────────────────────────────────────────────────────────────────────
// 1. extractAnalysisJson — pure parser
// ─────────────────────────────────────────────────────────────────────────

test("extractAnalysisJson: strips markdown fences", () => {
  const raw =
    "```json\n" +
    `{"agreements":["both mention Paris"],"disputes":[]}` +
    "\n```";
  const parsed = extractAnalysisJson(raw);
  assert.ok(parsed && typeof parsed === "object");
  assert.deepEqual(
    (parsed as Record<string, unknown>).agreements,
    ["both mention Paris"]
  );
});

test("extractAnalysisJson: skips prose preamble", () => {
  const raw =
    'Thinking Process: okay, comparing the two answers ...\n\n' +
    `{"agreements":[],"disputes":[{"topic":"t","amd_position":"a","nvidia_position":"b"}]}`;
  const parsed = extractAnalysisJson(raw);
  assert.ok(parsed && typeof parsed === "object");
  const disputes = (parsed as Record<string, unknown>).disputes as unknown[];
  assert.equal(disputes.length, 1);
});

test("extractAnalysisJson: garbage returns null", () => {
  assert.equal(extractAnalysisJson("this is not json at all"), null);
  assert.equal(extractAnalysisJson(""), null);
});

// 2026-04-21 — hardened against qwen3.5-9b truncating mid-<think>.
// Without the fix, the draft JSON inside the unclosed <think> block leaks
// into the parser and fails; with the fix, the opener-through-end is
// stripped and the real JSON after it (or null) is returned.
test("extractAnalysisJson: unclosed <think> is discarded (real JSON after it wins)", () => {
  const raw =
    "<think>Let me plan...\nDraft JSON: { \"agreements\": [\"DRAFT\"]"
    + " // the model was still drafting when it ran out of tokens...\n"
    + "</think>\n"
    + JSON.stringify({ agreements: ["final"], disputes: [] });
  const parsed = extractAnalysisJson(raw);
  assert.ok(parsed && typeof parsed === "object");
  assert.deepEqual((parsed as Record<string, unknown>).agreements, ["final"]);
});

test("extractAnalysisJson: unclosed <think> with NO real JSON after -> null (unavailable path)", () => {
  // Simulates mid-<think> truncation. Before the fix, the draft JSON inside
  // the unclosed tag leaked and produced spurious parse successes/failures.
  // After the fix, everything from <think> to EOS is stripped, no `{` found,
  // returns null, caller marks analysis unavailable.
  const raw =
    "<think>Thinking Process:\n"
    + "1. Plan...\n"
    + "2. Draft: { \"agreements\": [\"draft\"], \"disputes\": [ ]\n"
    + "3. Refine — but I ran out of tokens before closing </think>";
  const parsed = extractAnalysisJson(raw);
  assert.equal(parsed, null,
    "unclosed <think> with no trailing JSON must return null (→ analysis.unavailable)");
});

test("extractAnalysisJson: 'Thinking Process:' preamble without tags skipped to real JSON", () => {
  // Some variants skip the <think> wrapper entirely; a plain-text
  // "Thinking Process:" paragraph is followed by the real JSON.
  const raw =
    "Thinking Process:\n\n"
    + "I consider the two answers and conclude they agree.\n\n"
    + JSON.stringify({ agreements: ["both agree"], disputes: [] });
  const parsed = extractAnalysisJson(raw);
  assert.ok(parsed && typeof parsed === "object");
  assert.deepEqual((parsed as Record<string, unknown>).agreements, ["both agree"]);
});

// ─────────────────────────────────────────────────────────────────────────
// 2. renderDisputesTable
// ─────────────────────────────────────────────────────────────────────────

test("renderDisputesTable: empty -> HTML fallback", () => {
  const out = renderDisputesTable([], "qwen/qwen3.5-9b", "granite3.2:8b");
  assert.equal(out, "<p><em>No disputes — models agreed.</em></p>");
});

test("renderDisputesTable: HTML-escapes cell content, newlines -> <br>, emits valid table", () => {
  const disputes: AnalysisDispute[] = [
    {
      topic: "time & scope",
      amd_position: "line one\nline two",
      nvidia_position: "uses <angle> \"quotes\"",
    },
  ];
  const out = renderDisputesTable(disputes, "qwen/qwen3.5-9b", "granite3.2:8b");
  // Structural bones
  assert.ok(out.startsWith("<table"));
  assert.ok(out.endsWith("</table>"));
  assert.ok(out.includes("<thead>"));
  assert.ok(out.includes("<tbody>"));
  assert.ok(out.includes(">Topic</th>"));
  assert.ok(out.includes(">AMD (granite3.2:8b)</th>"));
  assert.ok(out.includes(">NVIDIA (qwen/qwen3.5-9b)</th>"));
  // Cell escaping
  assert.ok(out.includes("time &amp; scope"));          // &
  assert.ok(out.includes("line one<br>line two"));      // \n -> <br>
  assert.ok(out.includes("&lt;angle&gt;"));             // < >
  assert.ok(out.includes("&quot;quotes&quot;"));        // "
  // Topic rendered in <strong>
  assert.ok(out.includes("<strong>time &amp; scope</strong>"));
});

// ─────────────────────────────────────────────────────────────────────────
// 2a. renderDisputesMarkdown (2026-04-21 additive — LM Studio fallback)
// ─────────────────────────────────────────────────────────────────────────

test("renderDisputesMarkdown: empty -> markdown italic fallback", () => {
  const out = renderDisputesMarkdown([], "qwen/qwen3.5-9b", "granite3.2:8b");
  assert.equal(out, "_No disputes — models agreed._");
});

test("renderDisputesMarkdown: escapes pipes, newlines -> <br>, bolds topic", () => {
  const disputes: AnalysisDispute[] = [
    {
      topic: "pipe|containing topic",
      amd_position: "line one\nline two",
      nvidia_position: "has | pipe",
    },
  ];
  const out = renderDisputesMarkdown(disputes, "qwen/qwen3.5-9b", "granite3.2:8b");
  const lines = out.split("\n");
  // header + separator + 1 row
  assert.equal(lines.length, 3);
  assert.ok(lines[0].startsWith("| Topic | AMD (granite3.2:8b) | NVIDIA (qwen/qwen3.5-9b) |"));
  assert.equal(lines[1], "| --- | --- | --- |");
  // Pipes inside cells escaped, newline converted to <br>, topic bolded.
  assert.ok(lines[2].includes("**pipe\\|containing topic**"));
  assert.ok(lines[2].includes("line one<br>line two"));
  assert.ok(lines[2].includes("has \\| pipe"));
});

// ─────────────────────────────────────────────────────────────────────────
// 3. computeDisputes — cheap heuristic still works (archive tests preserved)
// ─────────────────────────────────────────────────────────────────────────

test("computeDisputes: identical answers -> no disputes", () => {
  const a = "The capital of France is Paris. It is on the Seine.";
  assert.deepEqual(computeDisputes(a, a), []);
});

test("computeDisputes: polarity mismatch -> 1 dispute", () => {
  const a = "Yes, that statement is correct and accurate.";
  const b = "No, that statement is incorrect and wrong.";
  const out = computeDisputes(a, b);
  assert.equal(out.length, 1);
  assert.equal(out[0].kind, "polarity-mismatch");
});

test("computeDisputes: low-overlap -> no-overlap dispute", () => {
  const a =
    "The capital is Paris, famous for the Eiffel Tower and the Louvre museum.";
  const b =
    "Economics studies resource allocation, scarcity, and production decisions today.";
  const out = computeDisputes(a, b);
  assert.equal(out.length, 1);
  assert.equal(out[0].kind, "no-overlap");
});

// ─────────────────────────────────────────────────────────────────────────
// 4. callOneBackend — unavailable path (from archive; still valid)
// ─────────────────────────────────────────────────────────────────────────

test("callOneBackend: unreachable endpoint -> unavailable:true, no throw", async () => {
  const out = await callOneBackend(
    "http://127.0.0.1:1/v1",
    "irrelevant",
    "fake-model",
    { question: "ping" },
    800
  );
  assert.equal(out.unavailable, true);
  assert.ok(out.second_opinion.startsWith("(unavailable:"));
  assert.equal(out.model, "fake-model");
  assert.ok(typeof out.latency_ms === "number");
});

// ─────────────────────────────────────────────────────────────────────────
// 5. runAnalysisPass — manual mode, both-agree, 2 disputes
// ─────────────────────────────────────────────────────────────────────────

test("runAnalysisPass: manual mode, 2 disputes -> table_html is a valid HTML table, no final_answer", async () => {
  const cannedJson = JSON.stringify({
    agreements: ["Both mention Paris."],
    disputes: [
      {
        topic: "population",
        amd_position: "The city proper has around 2.1 million residents.",
        nvidia_position: "The Paris metro area has around 10 million residents.",
      },
      {
        topic: "language emphasis",
        amd_position: "Answer centres on administrative status.",
        nvidia_position: "Answer centres on cultural role.",
      },
    ],
    final_answer: "SHOULD NOT APPEAR IN MANUAL MODE",
  });

  const mock = await startMockChatServer(() => ({ content: cannedJson }));
  try {
    const out = await runAnalysisPass({
      question: "Tell me about Paris.",
      amdAnswer: "Paris is the capital of France with ~2.1M people.",
      nvidiaAnswer: "Paris, France's capital, metro is ~10M.",
      primaryModelName: "qwen/qwen3.5-9b",
      mode: "manual",
      endpoint: mock.baseURL,
      apiKey: "test",
      model: "mock-analysis-model",
      timeoutMs: 3000,
    });

    assert.equal(out.agreements.length, 1);
    assert.equal(out.disputes.length, 2);
    assert.equal(out.final_answer, undefined); // manual mode
    assert.ok(out.table_html.startsWith("<table"));
    assert.ok(out.table_html.includes(">Topic</th>"));
    assert.ok(out.table_html.includes("<strong>population</strong>"));
    assert.ok(out.table_html.includes("<strong>language emphasis</strong>"));
    assert.equal(out.model, "mock-analysis-model");
    assert.ok(!out.unavailable);
    // 2026-04-21 shape pin: table_md always populated alongside table_html.
    assert.equal(typeof out.table_md, "string");
    assert.ok(out.table_md.includes("| Topic |"));
    assert.ok(out.table_md.includes("**population**"));
    assert.ok(out.table_md.includes("**language emphasis**"));
  } finally {
    await mock.close();
  }
});

// ─────────────────────────────────────────────────────────────────────────
// 6. runAnalysisPass — auto mode, final_answer populated
// ─────────────────────────────────────────────────────────────────────────

test("runAnalysisPass: auto mode -> final_answer populated", async () => {
  const cannedJson = JSON.stringify({
    agreements: ["Apollo 11 landed in 1969."],
    disputes: [],
    final_answer: "Apollo 11 landed on the Moon in 1969.",
  });
  const mock = await startMockChatServer(() => ({ content: cannedJson }));
  try {
    const out = await runAnalysisPass({
      question: "When did Apollo 11 land?",
      amdAnswer: "Apollo 11 landed on the Moon in July 1969.",
      nvidiaAnswer: "The Apollo 11 landing took place on July 20, 1969.",
      primaryModelName: "qwen/qwen3.5-9b",
      mode: "auto",
      endpoint: mock.baseURL,
      apiKey: "test",
      model: "mock-analysis-model",
      timeoutMs: 3000,
    });

    assert.equal(out.disputes.length, 0);
    assert.equal(out.final_answer, "Apollo 11 landed on the Moon in 1969.");
    assert.equal(out.agreements.length, 1);
    assert.equal(out.table_html, "<p><em>No disputes — models agreed.</em></p>");
    assert.ok(!out.unavailable);
  } finally {
    await mock.close();
  }
});

// ─────────────────────────────────────────────────────────────────────────
// 7. runAnalysisPass — malformed JSON / Thinking Process preamble fallback
// ─────────────────────────────────────────────────────────────────────────

test("runAnalysisPass: malformed JSON (preamble, no JSON object) -> unavailable + fallback table_html", async () => {
  const cannedJunk = "Thinking Process: I don't know how to produce JSON here.";
  const mock = await startMockChatServer(() => ({ content: cannedJunk }));
  try {
    const out = await runAnalysisPass({
      question: "q",
      amdAnswer: "a",
      nvidiaAnswer: "b",
      primaryModelName: "qwen/qwen3.5-9b",
      mode: "manual",
      endpoint: mock.baseURL,
      apiKey: "test",
      model: "mock-analysis-model",
      timeoutMs: 3000,
    });

    assert.equal(out.unavailable, true);
    assert.equal(out.table_html, "<p><em>analysis unavailable (parse failure)</em></p>");
    assert.deepEqual(out.agreements, []);
    assert.deepEqual(out.disputes, []);
    assert.equal(out.final_answer, undefined);
    assert.ok(out.error);
  } finally {
    await mock.close();
  }
});

// ─────────────────────────────────────────────────────────────────────────
// 8. runAnalysisPass — 0 disputes + agreements -> "No disputes" table
// ─────────────────────────────────────────────────────────────────────────

test("runAnalysisPass: 0 disputes -> table_html is 'No disputes' fallback, agreements populated", async () => {
  const cannedJson = JSON.stringify({
    agreements: [
      "Both cite 1969.",
      "Both identify the mission as Apollo 11.",
    ],
    disputes: [],
  });
  const mock = await startMockChatServer(() => ({ content: cannedJson }));
  try {
    const out = await runAnalysisPass({
      question: "Apollo 11 year?",
      amdAnswer: "1969",
      nvidiaAnswer: "1969",
      primaryModelName: "qwen/qwen3.5-9b",
      mode: "manual",
      endpoint: mock.baseURL,
      apiKey: "test",
      model: "mock",
      timeoutMs: 3000,
    });
    assert.deepEqual(out.disputes, []);
    assert.equal(out.agreements.length, 2);
    assert.equal(out.table_html, "<p><em>No disputes — models agreed.</em></p>");
    assert.ok(!out.unavailable);
  } finally {
    await mock.close();
  }
});

// ─────────────────────────────────────────────────────────────────────────
// 9. runAnalysisPass — JSON fenced in ```json ... ``` block still parses
// ─────────────────────────────────────────────────────────────────────────

test("runAnalysisPass: markdown-fenced JSON still parses", async () => {
  const cannedJson =
    "```json\n" +
    JSON.stringify({
      agreements: ["agree on year"],
      disputes: [
        {
          topic: "exact date",
          amd_position: "July 1969.",
          nvidia_position: "July 20, 1969.",
        },
      ],
    }) +
    "\n```";
  const mock = await startMockChatServer(() => ({ content: cannedJson }));
  try {
    const out = await runAnalysisPass({
      question: "q",
      amdAnswer: "a",
      nvidiaAnswer: "b",
      primaryModelName: "qwen/qwen3.5-9b",
      mode: "manual",
      endpoint: mock.baseURL,
      apiKey: "test",
      model: "mock",
      timeoutMs: 3000,
    });
    assert.equal(out.disputes.length, 1);
    assert.equal(out.disputes[0].topic, "exact date");
    assert.ok(!out.unavailable);
  } finally {
    await mock.close();
  }
});

// ─────────────────────────────────────────────────────────────────────────
// 10. runAnalysisPass — disputes array capped at 5
// ─────────────────────────────────────────────────────────────────────────

test("runAnalysisPass: disputes array cap enforced at MAX_ANALYSIS_DISPUTES", async () => {
  const mkDispute = (i: number) => ({
    topic: `t${i}`,
    amd_position: `a${i}`,
    nvidia_position: `n${i}`,
  });
  // Feed two more than the cap so we can assert the cap actually kicked in.
  const oversizedDisputes = Array.from(
    { length: MAX_ANALYSIS_DISPUTES + 2 },
    (_, i) => mkDispute(i + 1)
  );
  const cannedJson = JSON.stringify({
    agreements: [],
    disputes: oversizedDisputes,
  });
  const mock = await startMockChatServer(() => ({ content: cannedJson }));
  try {
    const out = await runAnalysisPass({
      question: "q",
      amdAnswer: "a",
      nvidiaAnswer: "b",
      primaryModelName: "qwen/qwen3.5-9b",
      mode: "manual",
      endpoint: mock.baseURL,
      apiKey: "test",
      model: "mock",
      timeoutMs: 3000,
    });
    assert.equal(out.disputes.length, MAX_ANALYSIS_DISPUTES);
  } finally {
    await mock.close();
  }
});

// ─────────────────────────────────────────────────────────────────────────
// 11. runAnalysisPass — <think> tags stripped before parse
// ─────────────────────────────────────────────────────────────────────────

test("runAnalysisPass: <think>...</think> prefix is stripped", async () => {
  const cannedJson =
    "<think>Let me consider...</think>\n" +
    JSON.stringify({
      agreements: ["agree on year"],
      disputes: [],
    });
  const mock = await startMockChatServer(() => ({ content: cannedJson }));
  try {
    const out = await runAnalysisPass({
      question: "q",
      amdAnswer: "a",
      nvidiaAnswer: "b",
      primaryModelName: "qwen/qwen3.5-9b",
      mode: "manual",
      endpoint: mock.baseURL,
      apiKey: "test",
      model: "mock",
      timeoutMs: 3000,
    });
    assert.equal(out.agreements.length, 1);
    assert.equal(out.disputes.length, 0);
    assert.ok(!out.unavailable);
  } finally {
    await mock.close();
  }
});

// ─────────────────────────────────────────────────────────────────────────
// 12. runAnalysisPass — auto mode but model forgot final_answer -> undefined
// ─────────────────────────────────────────────────────────────────────────

test("runAnalysisPass: auto mode, model omits final_answer -> field absent (not crash)", async () => {
  const cannedJson = JSON.stringify({
    agreements: ["both agree"],
    disputes: [],
    // final_answer intentionally omitted
  });
  const mock = await startMockChatServer(() => ({ content: cannedJson }));
  try {
    const out = await runAnalysisPass({
      question: "q",
      amdAnswer: "a",
      nvidiaAnswer: "b",
      primaryModelName: "qwen/qwen3.5-9b",
      mode: "auto",
      endpoint: mock.baseURL,
      apiKey: "test",
      model: "mock",
      timeoutMs: 3000,
    });
    assert.equal(out.final_answer, undefined);
    assert.equal(out.agreements.length, 1);
    assert.ok(!out.unavailable);
  } finally {
    await mock.close();
  }
});

// ─────────────────────────────────────────────────────────────────────────
// 13. runAnalysisPass — endpoint unreachable -> unavailable + fallback
// ─────────────────────────────────────────────────────────────────────────

test("runAnalysisPass: unreachable endpoint -> unavailable + fallback table", async () => {
  const out = await runAnalysisPass({
    question: "q",
    amdAnswer: "a",
    nvidiaAnswer: "b",
    primaryModelName: "qwen/qwen3.5-9b",
    mode: "manual",
    endpoint: "http://127.0.0.1:1/v1",
    apiKey: "irrelevant",
    model: "mock",
    timeoutMs: 800,
  });
  assert.equal(out.unavailable, true);
  assert.equal(out.table_html, "<p><em>analysis unavailable (parse failure)</em></p>");
  assert.ok(out.error);
});
