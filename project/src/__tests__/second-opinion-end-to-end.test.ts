/**
 * End-to-end tests for runSecondOpinion — covers the orchestration
 * logic across legacy path vs dual path vs partial-availability paths.
 *
 * Strategy: set env vars to point at mock endpoints BEFORE importing
 * the consult module. We bind two local HTTP servers on random ports
 * that impersonate Ollama (AMD) and LM Studio (NVIDIA + analysis).
 * The analysis pass uses the same primary endpoint as the NVIDIA leg
 * (same port), and the mock handler varies its response by inspecting
 * the request body (system prompt differs for answerer vs analysis).
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import type { AddressInfo } from "node:net";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

// ─────────────────────────────────────────────────────────────────────────
// Start two mock servers BEFORE importing consult.ts, then wire env vars
// so consult.ts picks them up at import time.
// ─────────────────────────────────────────────────────────────────────────

type Handler = (body: unknown) => { content: string };
type MockServer = { baseURL: string; server: http.Server; port: number };

function startMock(handler: Handler): Promise<MockServer> {
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      if (req.method !== "POST") {
        res.writeHead(405);
        res.end();
        return;
      }
      const chunks: Buffer[] = [];
      req.on("data", (c) => chunks.push(c));
      req.on("end", () => {
        let body: unknown;
        try {
          body = JSON.parse(Buffer.concat(chunks).toString("utf8"));
        } catch {
          /* ignore */
        }
        const { content } = handler(body);
        const payload = JSON.stringify({
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
        res.end(payload);
      });
    });
    server.listen(0, "127.0.0.1", () => {
      const addr = server.address() as AddressInfo;
      resolve({
        baseURL: `http://127.0.0.1:${addr.port}/v1`,
        server,
        port: addr.port,
      });
    });
  });
}

// Router state — mutable per-test so we can simulate unavailable legs.
interface RouterState {
  ollamaAvailable: boolean;
  primaryAvailable: boolean;
  ollamaAnswer: string;
  primaryAnswer: string;
  analysisContent: string;
}

const state: RouterState = {
  ollamaAvailable: true,
  primaryAvailable: true,
  ollamaAnswer: "Paris is the capital of France.",
  primaryAnswer:
    "The capital of France is Paris; it sits on the Seine river.",
  analysisContent: JSON.stringify({
    agreements: ["Both identify Paris as the capital."],
    disputes: [],
  }),
};

function isAnalysisRequest(body: unknown): boolean {
  if (!body || typeof body !== "object") return false;
  const messages = (body as Record<string, unknown>).messages;
  if (!Array.isArray(messages)) return false;
  const first = messages[0] as Record<string, unknown> | undefined;
  return (
    typeof first?.content === "string" &&
    (first.content as string).includes("analyzing two independent answers")
  );
}

// Spin up Ollama mock
const ollamaMock = await startMock((body) => {
  // When ollamaAvailable=false we still respond — the "unavailable" case is
  // exercised via a dead port instead (see below for that test scenario).
  // This mock is the "AMD succeeds" backend.
  void body;
  return { content: state.ollamaAnswer };
});

// Spin up Primary mock (serves both the NVIDIA answerer leg AND the
// analysis pass — distinguished by the system prompt).
const primaryMock = await startMock((body) => {
  if (isAnalysisRequest(body)) {
    return { content: state.analysisContent };
  }
  return { content: state.primaryAnswer };
});

// Point config at mocks. MUST be done before importing consult.ts.
process.env.OLLAMA_URL = ollamaMock.baseURL;
process.env.SECOND_OPINION_PRIMARY_ENDPOINT = primaryMock.baseURL;
process.env.SECOND_OPINION_PRIMARY_MODEL = "mock-nvidia-model";
process.env.SECOND_OPINION_ANALYSIS_MODEL = "mock-analysis-model";
process.env.SECOND_OPINION_MODEL = "mock-amd-model";
process.env.SECOND_OPINION_CONCURRENT_TIMEOUT_MS = "3000";
process.env.SECOND_OPINION_ANALYSIS_TIMEOUT_MS = "3000";
process.env.SECOND_OPINION_TIMEOUT_MS = "3000";
process.env.CONSULT_DUAL = "1";

// Dynamic import now that env is set.
const { runSecondOpinion } = await import(
  "../second-opinion/consult.js"
);

// ─────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────

test("runSecondOpinion: dual manual -> both answers + analysis populated + disputes alongside", async () => {
  // Configure a two-dispute analysis response + deliberately low-overlap
  // dual answers so computeDisputes (Phase B) also fires. That verifies
  // both surfaces coexist.
  state.ollamaAnswer = "Paris serves as France's administrative centre.";
  state.primaryAnswer =
    "Completely different text about economics and resource scarcity zones.";
  state.analysisContent = JSON.stringify({
    agreements: ["Both refer to a capital."],
    disputes: [
      {
        topic: "scope",
        amd_position: "AMD focuses on administrative status.",
        nvidia_position: "NVIDIA answer is off-topic about economics.",
      },
      {
        topic: "accuracy",
        amd_position: "AMD correctly identifies Paris.",
        nvidia_position: "NVIDIA answer does not identify a city.",
      },
    ],
    final_answer: "SHOULD NOT APPEAR",
  });

  const out = await runSecondOpinion({
    question: "What is the capital of France?",
    worker_draft: "Paris.",
    resolution_mode: "manual",
  });

  // Legacy fields populated from AMD leg
  assert.equal(out.model, "mock-amd-model");
  assert.equal(
    out.second_opinion,
    "Paris serves as France's administrative centre."
  );
  assert.ok(!out.unavailable);
  assert.ok(typeof out.diff_summary === "string"); // cheap heuristic still runs

  // Dual leg populated
  assert.ok(out.dual_opinion);
  assert.equal(out.dual_opinion!.model, "mock-nvidia-model");
  assert.equal(
    out.dual_opinion!.second_opinion,
    "Completely different text about economics and resource scarcity zones."
  );
  assert.ok(!out.dual_opinion!.unavailable);

  // Phase B cheap heuristic: disputes[] coexists with analysis
  assert.ok(Array.isArray(out.disputes));
  assert.ok(out.disputes.length >= 1, "expected no-overlap dispute from heuristic");

  // Phase C analysis
  assert.ok(out.analysis);
  assert.equal(out.analysis!.agreements.length, 1);
  assert.equal(out.analysis!.disputes.length, 2);
  assert.equal(out.analysis!.final_answer, undefined); // manual mode
  assert.ok(out.analysis!.table_html.startsWith("<table"));
  assert.ok(out.analysis!.table_html.includes(">Topic</th>"));
  assert.ok(out.analysis!.table_html.includes("<strong>scope</strong>"));

  // Mode echoed
  assert.equal(out.resolution_mode, "manual");
});

test("runSecondOpinion: dual auto -> analysis.final_answer populated", async () => {
  state.ollamaAnswer = "Apollo 11 landed in July 1969.";
  state.primaryAnswer = "The Apollo 11 landing occurred on July 20, 1969.";
  state.analysisContent = JSON.stringify({
    agreements: ["Both cite 1969."],
    disputes: [],
    final_answer: "Apollo 11 landed on the Moon on July 20, 1969.",
  });

  const out = await runSecondOpinion({
    question: "When did Apollo 11 land?",
    resolution_mode: "auto",
  });

  assert.equal(out.resolution_mode, "auto");
  assert.ok(out.analysis);
  assert.equal(
    out.analysis!.final_answer,
    "Apollo 11 landed on the Moon on July 20, 1969."
  );
  assert.equal(out.analysis!.disputes.length, 0);
  assert.equal(out.analysis!.table_html, "<p><em>No disputes — models agreed.</em></p>");
});

test("runSecondOpinion: legacy single path when input.model set -> no dual_opinion, no analysis, disputes=[]", async () => {
  // The legacy path hits Ollama only (via OLLAMA_URL mock).
  state.ollamaAnswer = "2+2 equals 4.";
  const out = await runSecondOpinion({
    question: "What is 2+2?",
    model: "granite3.2:2b",
  });

  assert.equal(out.model, "granite3.2:2b");
  assert.equal(out.second_opinion, "2+2 equals 4.");
  assert.equal(out.dual_opinion, undefined);
  assert.equal(out.analysis, undefined);
  assert.deepEqual(out.disputes, []);
  assert.ok(!out.unavailable);
});

test("runSecondOpinion: CONSULT_DUAL=0 simulated -> legacy single path (dual surfaces absent)", async () => {
  // We can't restart the module with a new CONSULT_DUAL at this point
  // (consult.ts reads it at import). Instead, simulate the same
  // observable behaviour by passing an explicit model — the legacy
  // branch is taken in BOTH cases (model override OR dual disabled).
  state.ollamaAnswer = "Paris is the capital.";
  const out = await runSecondOpinion({
    question: "Capital of France?",
    model: "granite3.2:8b",
  });

  // This is the shape CONSULT_DUAL=0 would also yield.
  assert.equal(out.dual_opinion, undefined);
  assert.equal(out.analysis, undefined);
  assert.deepEqual(out.disputes, []);
});

test("runSecondOpinion: AMD unavailable -> second_opinion starts with '(unavailable'; dual_opinion present; analysis absent", async () => {
  // Point only the AMD URL at a dead port for this test.
  // Cleanest way: use OpenAI client's own error path. Since the runtime
  // config is frozen, we emulate by closing our Ollama mock briefly.
  await new Promise<void>((r) => ollamaMock.server.close(() => r()));

  try {
    const out = await runSecondOpinion({
      question: "What is water?",
      resolution_mode: "manual",
    });

    assert.equal(out.unavailable, true);
    assert.ok(out.second_opinion.startsWith("(unavailable:"));
    // dual_opinion still present (NVIDIA leg succeeded)
    assert.ok(out.dual_opinion);
    assert.ok(!out.dual_opinion!.unavailable);
    // analysis absent — one leg unavailable
    assert.equal(out.analysis, undefined);
    // disputes empty — one leg unavailable
    assert.deepEqual(out.disputes, []);
  } finally {
    // Re-bind Ollama mock on the same port for later tests.
    await new Promise<void>((resolve) => {
      const srv = http.createServer((req, res) => {
        if (req.method !== "POST") {
          res.writeHead(405);
          res.end();
          return;
        }
        const chunks: Buffer[] = [];
        req.on("data", (c) => chunks.push(c));
        req.on("end", () => {
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(
            JSON.stringify({
              id: "chatcmpl-mock",
              object: "chat.completion",
              created: 0,
              model: "mock",
              choices: [
                {
                  index: 0,
                  message: { role: "assistant", content: state.ollamaAnswer },
                  finish_reason: "stop",
                },
              ],
              usage: {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
              },
            })
          );
        });
      });
      srv.listen(ollamaMock.port, "127.0.0.1", () => {
        ollamaMock.server = srv;
        resolve();
      });
    });
  }
});

test("runSecondOpinion: NVIDIA unavailable -> second_opinion ok, dual_opinion.unavailable=true, analysis absent", async () => {
  state.ollamaAnswer = "AMD leg's answer here.";

  // Stop the primary mock for this test.
  await new Promise<void>((r) => primaryMock.server.close(() => r()));

  try {
    const out = await runSecondOpinion({
      question: "What is water?",
      resolution_mode: "manual",
    });

    assert.ok(!out.unavailable, "AMD leg should be available");
    assert.equal(out.second_opinion, "AMD leg's answer here.");
    assert.ok(out.dual_opinion);
    assert.equal(out.dual_opinion!.unavailable, true);
    assert.equal(out.analysis, undefined);
    assert.deepEqual(out.disputes, []);
  } finally {
    // Re-bind primary mock on the same port.
    await new Promise<void>((resolve) => {
      const srv = http.createServer((req, res) => {
        if (req.method !== "POST") {
          res.writeHead(405);
          res.end();
          return;
        }
        const chunks: Buffer[] = [];
        req.on("data", (c) => chunks.push(c));
        req.on("end", () => {
          let body: unknown;
          try {
            body = JSON.parse(Buffer.concat(chunks).toString("utf8"));
          } catch {
            /* ignore */
          }
          const content = isAnalysisRequest(body)
            ? state.analysisContent
            : state.primaryAnswer;
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(
            JSON.stringify({
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
              usage: {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
              },
            })
          );
        });
      });
      srv.listen(primaryMock.port, "127.0.0.1", () => {
        primaryMock.server = srv;
        resolve();
      });
    });
  }
});

test("runSecondOpinion: analysis returns malformed JSON -> analysis.unavailable=true, fallback table", async () => {
  state.ollamaAnswer = "Paris is the capital of France.";
  state.primaryAnswer = "Paris is France's capital city.";
  state.analysisContent = "Thinking Process: I can't produce JSON here.";

  const out = await runSecondOpinion({
    question: "Capital?",
    resolution_mode: "manual",
  });

  assert.ok(out.analysis);
  assert.equal(out.analysis!.unavailable, true);
  assert.equal(
    out.analysis!.table_html,
    "<p><em>analysis unavailable (parse failure)</em></p>"
  );
});

test("runSecondOpinion: analysis 0-disputes -> 'No disputes' fallback + agreements present", async () => {
  state.ollamaAnswer = "Water boils at 100°C.";
  state.primaryAnswer = "Water's boiling point is 100°C at 1 atm.";
  state.analysisContent = JSON.stringify({
    agreements: ["Both cite 100°C.", "Both discuss standard pressure."],
    disputes: [],
  });

  const out = await runSecondOpinion({
    question: "Boiling point of water?",
    resolution_mode: "manual",
  });

  assert.ok(out.analysis);
  assert.deepEqual(out.analysis!.disputes, []);
  assert.equal(out.analysis!.agreements.length, 2);
  assert.equal(out.analysis!.table_html, "<p><em>No disputes — models agreed.</em></p>");
  assert.equal(out.analysis!.final_answer, undefined);
});

test("runSecondOpinion: default resolution_mode is 'manual' when omitted", async () => {
  state.ollamaAnswer = "short answer a";
  state.primaryAnswer = "short answer b";
  state.analysisContent = JSON.stringify({
    agreements: ["both are short"],
    disputes: [],
    final_answer: "SHOULD NOT APPEAR IN MANUAL MODE",
  });

  const out = await runSecondOpinion({ question: "Q?" });
  assert.equal(out.resolution_mode, "manual");
  assert.ok(out.analysis);
  assert.equal(out.analysis!.final_answer, undefined);
});

test("runSecondOpinion: BOTH unavailable -> legacy shape preserved, analysis absent, disputes=[]", async () => {
  // Close both mocks simultaneously, then exercise runSecondOpinion.
  await new Promise<void>((r) => ollamaMock.server.close(() => r()));
  await new Promise<void>((r) => primaryMock.server.close(() => r()));

  try {
    const out = await runSecondOpinion({
      question: "Q",
      resolution_mode: "manual",
    });

    // Legacy fields still present
    assert.equal(out.unavailable, true);
    assert.ok(out.second_opinion.startsWith("(unavailable:"));
    assert.ok(typeof out.diff_summary === "string");

    // dual_opinion present but marked unavailable
    assert.ok(out.dual_opinion);
    assert.equal(out.dual_opinion!.unavailable, true);
    // analysis absent — can't compare when nothing is ok
    assert.equal(out.analysis, undefined);
    // disputes empty
    assert.deepEqual(out.disputes, []);
    // resolution_mode echoed
    assert.equal(out.resolution_mode, "manual");
  } finally {
    // Leave mocks closed — the next test block is teardown.
  }
});

// ─────────────────────────────────────────────────────────────────────────
// 2026-04-21 shape pins — added after user-facing "no table" bug.
// Guards against regressions in the wire shape the worker depends on.
// ─────────────────────────────────────────────────────────────────────────

test("runSecondOpinion: analysis exposes BOTH table_html and table_md strings (shape pin)", async () => {
  // Re-bind both mocks (prior tests closed them).
  const ollamaSrv = http.createServer((req, res) => {
    if (req.method !== "POST") { res.writeHead(405); res.end(); return; }
    const chunks: Buffer[] = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({
        id: "x", object: "chat.completion", created: 0, model: "m",
        choices: [{ index: 0, message: { role: "assistant", content: state.ollamaAnswer }, finish_reason: "stop" }],
        usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
      }));
    });
  });
  await new Promise<void>((r) => ollamaSrv.listen(ollamaMock.port, "127.0.0.1", () => r()));
  ollamaMock.server = ollamaSrv;

  const primarySrv = http.createServer((req, res) => {
    if (req.method !== "POST") { res.writeHead(405); res.end(); return; }
    const chunks: Buffer[] = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => {
      let body: unknown;
      try { body = JSON.parse(Buffer.concat(chunks).toString("utf8")); } catch { /* ignore */ }
      const content = isAnalysisRequest(body) ? state.analysisContent : state.primaryAnswer;
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({
        id: "x", object: "chat.completion", created: 0, model: "m",
        choices: [{ index: 0, message: { role: "assistant", content }, finish_reason: "stop" }],
        usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
      }));
    });
  });
  await new Promise<void>((r) => primarySrv.listen(primaryMock.port, "127.0.0.1", () => r()));
  primaryMock.server = primarySrv;

  state.ollamaAnswer = "AMD sez alpha.";
  state.primaryAnswer = "NVIDIA sez beta.";
  state.analysisContent = JSON.stringify({
    agreements: ["both are short"],
    disputes: [
      { topic: "flavor", amd_position: "alpha flavor", nvidia_position: "beta flavor" },
      { topic: "tone",   amd_position: "terse",       nvidia_position: "terse too" },
    ],
  });

  const out = await runSecondOpinion({ question: "Q?", resolution_mode: "manual" });

  assert.ok(out.analysis, "analysis must be present");
  const a = out.analysis!;

  // Shape: both renderings always populated when analysis ran (even on 0 disputes).
  assert.equal(typeof a.table_html, "string", "table_html must be a string");
  assert.equal(typeof a.table_md, "string", "table_md must be a string");

  // HTML: real table with row per dispute.
  assert.ok(a.table_html.startsWith("<table"), "table_html starts with <table");
  const trMatches = a.table_html.match(/<tr[\s>]/g) ?? [];
  // 1 header row + 2 data rows = 3 total
  assert.equal(trMatches.length, a.disputes.length + 1,
    "HTML <tr> count must be disputes.length + 1 (header)");

  // Markdown: pipe table with header + separator + disputes rows.
  assert.ok(a.table_md.includes("| Topic |"), "MD header row present");
  assert.ok(a.table_md.includes("| --- |"), "MD separator row present");
  assert.ok(a.table_md.includes("**flavor**"), "MD bolds the topic");
  const mdLines = a.table_md.split("\n");
  assert.equal(mdLines.length, a.disputes.length + 2, // header + separator + rows
    "MD line count must be disputes.length + 2 (header + separator)");
});

test("runSecondOpinion: 0-disputes still populates table_md fallback string (shape pin)", async () => {
  state.ollamaAnswer = "Water boils at 100C.";
  state.primaryAnswer = "Water's boiling point is 100C.";
  state.analysisContent = JSON.stringify({
    agreements: ["both cite 100C"],
    disputes: [],
  });
  const out = await runSecondOpinion({ question: "Boiling point?" });
  assert.ok(out.analysis);
  assert.equal(out.analysis!.disputes.length, 0);
  assert.equal(out.analysis!.table_html, "<p><em>No disputes — models agreed.</em></p>");
  assert.equal(out.analysis!.table_md, "_No disputes — models agreed._");
});

test("runSecondOpinion: analysis unavailable populates both table_html AND table_md fallback strings (shape pin)", async () => {
  // Force malformed content so analysis is unavailable.
  state.ollamaAnswer = "answer a";
  state.primaryAnswer = "answer b";
  state.analysisContent = "Thinking Process: I lost my JSON brain.";
  const out = await runSecondOpinion({ question: "Q?" });
  assert.ok(out.analysis);
  assert.equal(out.analysis!.unavailable, true);
  assert.equal(out.analysis!.table_html, "<p><em>analysis unavailable (parse failure)</em></p>");
  assert.equal(out.analysis!.table_md, "_analysis unavailable (parse failure)_");
});

// Tool description pin — the worker LLM reads this description to decide
// how to render the tool output. If future edits drop the explicit
// instruction to show the table, we want to catch it in CI.
// We read the built index.js (not import it — index.ts starts a listening
// HTTP server at module-load time, which collides with the running
// verity on :8090 in dev).
test("tool description: consult_second_opinion mentions table_html and table_md (worker hint pin)", () => {
  const here = fileURLToPath(import.meta.url);
  // `dist/__tests__/second-opinion-end-to-end.test.js` -> `dist/index.js`
  const indexPath = path.resolve(path.dirname(here), "..", "index.js");
  const src = readFileSync(indexPath, "utf8");
  // Find the compiled object literal for consult_second_opinion.
  assert.ok(src.includes('"consult_second_opinion"') || src.includes("'consult_second_opinion'"),
    "consult_second_opinion must be registered");
  assert.ok(src.includes("table_html"),
    "tool description must mention table_html so the worker knows to render it");
  assert.ok(src.includes("table_md"),
    "tool description must mention table_md so the worker has a Markdown fallback");
});

test("teardown-2: final close of mocks", async () => {
  await new Promise<void>((r) => ollamaMock.server.close(() => r()));
  await new Promise<void>((r) => primaryMock.server.close(() => r()));
  assert.ok(true);
});

// Teardown: mocks already closed by the prior test; nothing to do.
test("teardown: (mocks already closed)", () => {
  assert.ok(true);
});
