/**
 * Unit tests for the consistency-check K plumbing.
 *
 * Verifies that:
 *   1. config.ts reads CONSISTENCY_K_DEEP / CONSISTENCY_K_DEEPER env vars
 *      with the documented defaults (5 / 8). Defaults track Wang et al.
 *      2022 (Self-Consistency) + Manakul et al. 2023 (SelfCheckGPT).
 *   2. runConsistencyCheck calls the worker exactly K times for the
 *      requested numSamples.
 *
 * Strategy: stand up a mock HTTP server that impersonates LM Studio's
 * /v1/chat/completions, point WORKER_ENDPOINT at it BEFORE importing
 * the module, and count requests. Pass preExtractedClaims=[] so the
 * NLI classifier is never invoked — we only need to verify the K
 * pass-through, not the downstream claim check.
 */

import { test, after } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import type { AddressInfo } from "node:net";

// ─────────────────────────────────────────────────────────────────────────
// Mock LM Studio worker endpoint
// ─────────────────────────────────────────────────────────────────────────

interface CallCounter {
  count: number;
  reset(): void;
}

function startMockWorker(counter: CallCounter): Promise<{
  baseURL: string;
  server: http.Server;
}> {
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      if (req.method !== "POST") {
        res.writeHead(405);
        res.end();
        return;
      }
      counter.count += 1;
      const chunks: Buffer[] = [];
      req.on("data", (c) => chunks.push(c));
      req.on("end", () => {
        const payload = JSON.stringify({
          id: "chatcmpl-mock-worker",
          object: "chat.completion",
          created: 0,
          model: "mock-worker",
          choices: [
            {
              index: 0,
              message: {
                role: "assistant",
                content: "Mock re-sample text.",
              },
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
      });
    });
  });
}

const counter: CallCounter = {
  count: 0,
  reset() {
    this.count = 0;
  },
};

const mock = await startMockWorker(counter);

// Wire env BEFORE importing config + consistency.
process.env.WORKER_ENDPOINT = mock.baseURL;
process.env.WORKER_API_KEY = "mock-key";
process.env.WORKER_MODEL = "mock-worker";
// Deliberately do NOT set CONSISTENCY_K_DEEP / CONSISTENCY_K_DEEPER here —
// the first assertion below relies on the SHIPPED defaults (5 / 8).

const config = await import("../config.js");
const { runConsistencyCheck } = await import("../signals/consistency.js");

// ─────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────

test("config: CONSISTENCY_SAMPLES_DEEP defaults to 5 (Wang 2022 / SelfCheckGPT 2023 sweet spot)", () => {
  assert.equal(config.CONSISTENCY_SAMPLES_DEEP, 5);
});

test("config: CONSISTENCY_SAMPLES_DEEPER defaults to 8 (further into the diminishing-returns tail)", () => {
  assert.equal(config.CONSISTENCY_SAMPLES_DEEPER, 8);
});

test("runConsistencyCheck calls the worker exactly numSamples times (K=5 deep default)", async () => {
  counter.reset();
  const result = await runConsistencyCheck({
    question: "calibration test query alpha",
    originalAnswer: "calibration test answer alpha",
    numSamples: config.CONSISTENCY_SAMPLES_DEEP,
    preExtractedClaims: [],
  });
  assert.equal(
    counter.count,
    config.CONSISTENCY_SAMPLES_DEEP,
    `expected ${config.CONSISTENCY_SAMPLES_DEEP} worker calls, got ${counter.count}`
  );
  // No claims passed → ran=true but claims_checked=0 path.
  assert.equal(result.ran, true);
  assert.equal(result.claims_checked, 0);
  assert.equal(result.samples_generated, config.CONSISTENCY_SAMPLES_DEEP);
});

test("runConsistencyCheck calls the worker exactly numSamples times (K=8 deeper default)", async () => {
  counter.reset();
  const result = await runConsistencyCheck({
    question: "calibration test query beta",
    originalAnswer: "calibration test answer beta",
    numSamples: config.CONSISTENCY_SAMPLES_DEEPER,
    preExtractedClaims: [],
  });
  assert.equal(
    counter.count,
    config.CONSISTENCY_SAMPLES_DEEPER,
    `expected ${config.CONSISTENCY_SAMPLES_DEEPER} worker calls, got ${counter.count}`
  );
  assert.equal(result.samples_generated, config.CONSISTENCY_SAMPLES_DEEPER);
});

test("runConsistencyCheck with numSamples < 1 skips the worker entirely", async () => {
  counter.reset();
  const result = await runConsistencyCheck({
    question: "calibration test query gamma",
    originalAnswer: "calibration test answer gamma",
    numSamples: 0,
    preExtractedClaims: [],
  });
  assert.equal(counter.count, 0);
  assert.equal(result.ran, false);
  assert.equal(result.samples_generated, 0);
});

// Tear down the mock server when the suite finishes so the test process
// exits cleanly under `node --test`.
after(async () => {
  await new Promise<void>((r) => mock.server.close(() => r()));
});
