/**
 * Unit tests for the confidence proxy's request-shape guard and its
 * request/response translation.
 *
 * Pure logic — no LM Studio / Ollama calls. Validates:
 *   - isScorable: a plain-text chat request is scorable; every non-scorable
 *     shape (vision, tools, response_format, n>1, stop/seed/penalties/
 *     logit_bias/top_k, tool_calls in a message, empty/absent messages) is
 *     rejected with a reason. This is the predicate that decides divert-vs-
 *     pass-through, so it is the most important thing under test;
 *   - chatRequestToResponses: field mapping for a scorable request (content
 *     flattening, max_tokens -> max_output_tokens, logprobs include);
 *   - extractFromResponses: result -> (text + tokens) (structured walk +
 *     output_text fallback);
 *   - mapFinishReason: real finish reason from the upstream status (NOT a
 *     hard-coded "stop");
 *   - applyConfidenceNote: note appended when band != ok, absent when ok;
 *   - buildChatResponse: non-streaming shape + usage mapping + mapped finish;
 *   - buildScorableSseChunks: role priming, answer deltas, the note as a final
 *     delta, terminal finish_reason.
 *
 * Mirrors the style of confidence.test.ts / aggregator.test.ts: node:test,
 * node:assert/strict, no live backend. Test prompts are content-free markers.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  isScorable,
  flattenContent,
  chatRequestToResponses,
  extractFromResponses,
  mapFinishReason,
  applyConfidenceNote,
  buildChatResponse,
  buildScorableSseChunks,
  type ChatCompletionsRequest,
  type ResponsesResult,
} from "../proxy/translate.js";
import {
  PROXY_DEFAULT_MAX_OUTPUT_TOKENS,
  PROXY_RESPONSES_TOP_LOGPROBS,
} from "../config.js";
import type { TokenLogprob } from "../signals/confidence.js";

// Helper: build a stream of n tokens at a fixed logprob (as in confidence.test.ts).
function flat(n: number, logprob: number): TokenLogprob[] {
  return Array.from({ length: n }, (_, i) => ({ token: `t${i}`, logprob }));
}

// A minimal, definitely-scorable request used as the base for "one bad field"
// negative tests. Plain text, one user message, no exotic params.
function baseScorable(): ChatCompletionsRequest {
  return {
    model: "m",
    messages: [{ role: "user", content: "calibration test query alpha" }],
  };
}

// ────────────────────────────────────────────────────────────────────────
// isScorable — the request-shape guard (divert vs pass-through)
// ────────────────────────────────────────────────────────────────────────

test("isScorable: plain-text chat request is scorable", () => {
  const v = isScorable({
    model: "m",
    messages: [
      { role: "system", content: "be terse" },
      { role: "user", content: "calibration test query alpha" },
    ],
    temperature: 0.4,
    top_p: 0.8,
    max_tokens: 256,
    stream: true,
  });
  assert.equal(v.scorable, true);
});

test("isScorable: content-parts that are all text are scorable", () => {
  const v = isScorable({
    model: "m",
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: "part one " },
          { type: "text", text: "part two" },
        ],
      },
    ],
  } as ChatCompletionsRequest);
  assert.equal(v.scorable, true);
});

test("isScorable: n=1 is scorable; n>1 is not", () => {
  assert.equal(isScorable({ ...baseScorable(), n: 1 }).scorable, true);
  const v = isScorable({ ...baseScorable(), n: 2 });
  assert.equal(v.scorable, false);
  if (!v.scorable) assert.match(v.reason, /n=2/);
});

test("isScorable: empty container params do not disqualify (tools:[], stop:[])", () => {
  // Some SDKs always include the key with an empty value; that is "not set".
  assert.equal(isScorable({ ...baseScorable(), tools: [] }).scorable, true);
  assert.equal(isScorable({ ...baseScorable(), stop: [] }).scorable, true);
  assert.equal(
    isScorable({ ...baseScorable(), logit_bias: {} } as ChatCompletionsRequest)
      .scorable,
    true
  );
});

// --- Each non-scorable shape (vision / tools / structured output / params) ---

test("isScorable: NOT scorable — image_url content part (vision)", () => {
  const v = isScorable({
    model: "m",
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: "describe" },
          { type: "image_url", image_url: { url: "data:image/png;base64,AAAA" } },
        ],
      },
    ],
  } as ChatCompletionsRequest);
  assert.equal(v.scorable, false);
  if (!v.scorable) assert.match(v.reason, /image_url|non-text/);
});

test("isScorable: NOT scorable — input_image content part (vision)", () => {
  const v = isScorable({
    model: "m",
    messages: [
      { role: "user", content: [{ type: "input_image", image_url: "x" }] },
    ],
  } as ChatCompletionsRequest);
  assert.equal(v.scorable, false);
});

test("isScorable: NOT scorable — input_audio content part", () => {
  const v = isScorable({
    model: "m",
    messages: [
      { role: "user", content: [{ type: "input_audio", input_audio: {} }] },
    ],
  } as ChatCompletionsRequest);
  assert.equal(v.scorable, false);
});

test("isScorable: NOT scorable — tools present", () => {
  const v = isScorable({
    ...baseScorable(),
    tools: [
      {
        type: "function",
        function: { name: "get_weather", parameters: {} },
      },
    ],
  } as ChatCompletionsRequest);
  assert.equal(v.scorable, false);
  if (!v.scorable) assert.match(v.reason, /tools/);
});

test("isScorable: NOT scorable — tool_choice present", () => {
  const v = isScorable({
    ...baseScorable(),
    tool_choice: "auto",
  } as ChatCompletionsRequest);
  assert.equal(v.scorable, false);
});

test("isScorable: NOT scorable — legacy functions present", () => {
  const v = isScorable({
    ...baseScorable(),
    functions: [{ name: "f", parameters: {} }],
  } as ChatCompletionsRequest);
  assert.equal(v.scorable, false);
});

test("isScorable: NOT scorable — response_format (structured output)", () => {
  const v = isScorable({
    ...baseScorable(),
    response_format: { type: "json_schema", json_schema: { name: "s", schema: {} } },
  } as ChatCompletionsRequest);
  assert.equal(v.scorable, false);
  if (!v.scorable) assert.match(v.reason, /response_format/);
});

test("isScorable: NOT scorable — stop sequences set", () => {
  const v = isScorable({ ...baseScorable(), stop: ["\n\n"] });
  assert.equal(v.scorable, false);
  if (!v.scorable) assert.match(v.reason, /stop/);
});

test("isScorable: NOT scorable — seed set", () => {
  const v = isScorable({ ...baseScorable(), seed: 42 } as ChatCompletionsRequest);
  assert.equal(v.scorable, false);
  if (!v.scorable) assert.match(v.reason, /seed/);
});

test("isScorable: NOT scorable — presence_penalty set", () => {
  const v = isScorable({
    ...baseScorable(),
    presence_penalty: 0.5,
  } as ChatCompletionsRequest);
  assert.equal(v.scorable, false);
  if (!v.scorable) assert.match(v.reason, /presence_penalty/);
});

test("isScorable: NOT scorable — frequency_penalty set", () => {
  const v = isScorable({
    ...baseScorable(),
    frequency_penalty: 0.5,
  } as ChatCompletionsRequest);
  assert.equal(v.scorable, false);
});

test("isScorable: NOT scorable — logit_bias set", () => {
  const v = isScorable({
    ...baseScorable(),
    logit_bias: { "123": 5 },
  } as ChatCompletionsRequest);
  assert.equal(v.scorable, false);
  if (!v.scorable) assert.match(v.reason, /logit_bias/);
});

test("isScorable: NOT scorable — top_k set", () => {
  const v = isScorable({ ...baseScorable(), top_k: 40 } as ChatCompletionsRequest);
  assert.equal(v.scorable, false);
});

test("isScorable: NOT scorable — client asks for its own logprobs", () => {
  // A client-driven logprobs request would collide with the include/top_logprobs
  // WE set, so such a request passes through untouched.
  assert.equal(
    isScorable({ ...baseScorable(), logprobs: true } as ChatCompletionsRequest)
      .scorable,
    false
  );
  assert.equal(
    isScorable({ ...baseScorable(), top_logprobs: 5 } as ChatCompletionsRequest)
      .scorable,
    false
  );
});

test("isScorable: NOT scorable — a message carries tool_calls", () => {
  const v = isScorable({
    model: "m",
    messages: [
      {
        role: "assistant",
        content: null,
        tool_calls: [
          { id: "c1", type: "function", function: { name: "f", arguments: "{}" } },
        ],
      },
      { role: "tool", content: "result", tool_call_id: "c1" },
    ],
  } as ChatCompletionsRequest);
  assert.equal(v.scorable, false);
  if (!v.scorable) assert.match(v.reason, /tool_calls/);
});

test("isScorable: NOT scorable — no messages / empty messages", () => {
  assert.equal(isScorable({ model: "m" }).scorable, false);
  assert.equal(isScorable({ model: "m", messages: [] }).scorable, false);
});

test("isScorable: NOT scorable — non-object body", () => {
  assert.equal(isScorable(null as unknown as ChatCompletionsRequest).scorable, false);
  assert.equal(
    isScorable("nope" as unknown as ChatCompletionsRequest).scorable,
    false
  );
});

// ────────────────────────────────────────────────────────────────────────
// flattenContent
// ────────────────────────────────────────────────────────────────────────

test("flattenContent: string passes through", () => {
  assert.equal(flattenContent("hello"), "hello");
});

test("flattenContent: null -> empty string", () => {
  assert.equal(flattenContent(null), "");
});

test("flattenContent: content-parts array concatenates text parts", () => {
  const parts = [
    { type: "text", text: "alpha " },
    { type: "text", text: "beta" },
  ];
  assert.equal(flattenContent(parts as never), "alpha beta");
});

// ────────────────────────────────────────────────────────────────────────
// chatRequestToResponses  (scorable requests only)
// ────────────────────────────────────────────────────────────────────────

test("chatRequestToResponses: maps messages, model, and always asks for logprobs", () => {
  const req: ChatCompletionsRequest = {
    model: "some-model",
    messages: [
      { role: "system", content: "be terse" },
      { role: "user", content: "calibration test query alpha" },
    ],
    temperature: 0.4,
    top_p: 0.8,
    max_tokens: 256,
  };
  const out = chatRequestToResponses(req);

  assert.equal(out.model, "some-model");
  assert.deepEqual(out.include, ["message.output_text.logprobs"]);
  assert.equal(out.top_logprobs, PROXY_RESPONSES_TOP_LOGPROBS);
  assert.equal(out.max_output_tokens, 256);
  assert.equal(out.temperature, 0.4);
  assert.equal(out.top_p, 0.8);
  assert.deepEqual(out.input, [
    { role: "system", content: "be terse" },
    { role: "user", content: "calibration test query alpha" },
  ]);
});

test("chatRequestToResponses: max_tokens -> max_output_tokens; default when omitted", () => {
  const out = chatRequestToResponses({
    model: "m",
    messages: [{ role: "user", content: "test marker beta" }],
  });
  assert.equal(out.max_output_tokens, PROXY_DEFAULT_MAX_OUTPUT_TOKENS);
});

test("chatRequestToResponses: honours max_completion_tokens alias", () => {
  const out = chatRequestToResponses({
    model: "m",
    messages: [{ role: "user", content: "test marker gamma" }],
    max_completion_tokens: 99,
  } as ChatCompletionsRequest);
  assert.equal(out.max_output_tokens, 99);
});

test("chatRequestToResponses: omits sampling params the client did not set", () => {
  const out = chatRequestToResponses({
    model: "m",
    messages: [{ role: "user", content: "test marker delta" }],
  });
  assert.equal("temperature" in out, false);
  assert.equal("top_p" in out, false);
});

test("chatRequestToResponses: flattens content-parts messages", () => {
  const out = chatRequestToResponses({
    model: "m",
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: "part one " },
          { type: "text", text: "part two" },
        ],
      },
    ],
  } as ChatCompletionsRequest);
  assert.equal(out.input[0]?.content, "part one part two");
});

test("chatRequestToResponses: empty/missing messages -> empty input", () => {
  const out = chatRequestToResponses({ model: "m" });
  assert.deepEqual(out.input, []);
});

// ────────────────────────────────────────────────────────────────────────
// extractFromResponses
// ────────────────────────────────────────────────────────────────────────

test("extractFromResponses: walks output[].content[].logprobs[]", () => {
  const result: ResponsesResult = {
    model: "m",
    output: [
      {
        type: "message",
        role: "assistant",
        content: [
          {
            type: "output_text",
            text: "Hi there",
            logprobs: [
              { token: "Hi", logprob: -0.1 },
              { token: " there", logprob: -0.2 },
            ],
          },
        ],
      },
    ],
  };
  const { text, tokens } = extractFromResponses(result);
  assert.equal(text, "Hi there");
  assert.equal(tokens.length, 2);
  assert.equal(tokens[0]?.token, "Hi");
  assert.equal(tokens[1]?.logprob, -0.2);
});

test("extractFromResponses: concatenates text across multiple output_text items", () => {
  const result: ResponsesResult = {
    output: [
      {
        content: [
          { type: "output_text", text: "foo ", logprobs: [{ token: "foo ", logprob: -0.1 }] },
          { type: "output_text", text: "bar", logprobs: [{ token: "bar", logprob: -0.1 }] },
        ],
      },
    ],
  };
  const { text, tokens } = extractFromResponses(result);
  assert.equal(text, "foo bar");
  assert.equal(tokens.length, 2);
});

test("extractFromResponses: missing logprob becomes NaN (classifier drops it)", () => {
  const result: ResponsesResult = {
    output: [
      {
        content: [
          {
            type: "output_text",
            text: "x",
            logprobs: [{ token: "x" }], // no logprob field
          },
        ],
      },
    ],
  };
  const { tokens } = extractFromResponses(result);
  assert.equal(tokens.length, 1);
  assert.equal(Number.isNaN(tokens[0]?.logprob), true);
});

test("extractFromResponses: falls back to output_text when no structured content", () => {
  const result: ResponsesResult = { output_text: "flat fallback" };
  const { text, tokens } = extractFromResponses(result);
  assert.equal(text, "flat fallback");
  assert.equal(tokens.length, 0); // no logprobs -> classifier sees "ok"
});

// ────────────────────────────────────────────────────────────────────────
// mapFinishReason — real finish reason, never hard-coded
// ────────────────────────────────────────────────────────────────────────

test("mapFinishReason: completed -> stop", () => {
  assert.equal(mapFinishReason({ status: "completed" }), "stop");
});

test("mapFinishReason: incomplete + max_output_tokens -> length", () => {
  assert.equal(
    mapFinishReason({
      status: "incomplete",
      incomplete_details: { reason: "max_output_tokens" },
    }),
    "length"
  );
});

test("mapFinishReason: incomplete + content_filter -> content_filter", () => {
  assert.equal(
    mapFinishReason({
      status: "incomplete",
      incomplete_details: { reason: "content_filter" },
    }),
    "content_filter"
  );
});

test("mapFinishReason: incomplete with unknown reason -> length", () => {
  assert.equal(mapFinishReason({ status: "incomplete" }), "length");
});

test("mapFinishReason: missing status -> stop (safe default)", () => {
  assert.equal(mapFinishReason({}), "stop");
});

// ────────────────────────────────────────────────────────────────────────
// applyConfidenceNote — the note-injection logic
// ────────────────────────────────────────────────────────────────────────

test("applyConfidenceNote: confident answer is returned unchanged (no note)", () => {
  const text = "A confident answer.";
  const { content, assessment, note } = applyConfidenceNote(text, flat(12, -0.01));
  assert.equal(assessment.band, "ok");
  assert.equal(note, "");
  assert.equal(content, text); // byte-for-byte unchanged
});

test("applyConfidenceNote: very-low band appends a /verifydeeper note", () => {
  const text = "Answer with one shaky token.";
  // One very-weak token trips the local axis -> very_low -> /verifydeeper.
  const tokens: TokenLogprob[] = [...flat(9, -0.01), { token: "GUESS", logprob: -7.0 }];
  const { content, assessment, note } = applyConfidenceNote(text, tokens);
  assert.equal(assessment.band, "very_low");
  assert.notEqual(note, "");
  assert.match(content, /\/verifydeeper/);
  // Note is appended AFTER the original text, separated by a blank line.
  assert.equal(content.startsWith(text), true);
  assert.match(content, /Answer with one shaky token\.\n\n/);
});

test("applyConfidenceNote: empty token stream -> ok -> unchanged", () => {
  const text = "No logprobs were available.";
  const { content, assessment } = applyConfidenceNote(text, []);
  assert.equal(assessment.band, "ok");
  assert.equal(content, text);
});

// ────────────────────────────────────────────────────────────────────────
// buildChatResponse
// ────────────────────────────────────────────────────────────────────────

test("buildChatResponse: standard chat.completion shape with mapped usage", () => {
  const upstream: ResponsesResult = {
    model: "upstream-model",
    status: "completed",
    usage: { input_tokens: 11, output_tokens: 7, total_tokens: 18 },
  };
  const resp = buildChatResponse({
    content: "final content",
    model: "client-model",
    upstream,
  });
  assert.equal(resp.object, "chat.completion");
  assert.equal(resp.choices.length, 1);
  assert.equal(resp.choices[0]?.message.role, "assistant");
  assert.equal(resp.choices[0]?.message.content, "final content");
  // finish_reason is the mapped real reason from upstream status.
  assert.equal(resp.choices[0]?.finish_reason, "stop");
  // chat-completions usage names, sourced from responses usage.
  assert.equal(resp.usage.prompt_tokens, 11);
  assert.equal(resp.usage.completion_tokens, 7);
  assert.equal(resp.usage.total_tokens, 18);
});

test("buildChatResponse: maps a real length finish_reason from upstream status", () => {
  const resp = buildChatResponse({
    content: "truncated",
    model: "m",
    upstream: {
      status: "incomplete",
      incomplete_details: { reason: "max_output_tokens" },
    },
  });
  assert.equal(resp.choices[0]?.finish_reason, "length");
});

test("buildChatResponse: explicit finishReason override wins", () => {
  const resp = buildChatResponse({
    content: "x",
    model: "m",
    upstream: { status: "completed" },
    finishReason: "length",
  });
  assert.equal(resp.choices[0]?.finish_reason, "length");
});

test("buildChatResponse: total_tokens derived when upstream omits it", () => {
  const resp = buildChatResponse({
    content: "x",
    model: "m",
    upstream: { usage: { input_tokens: 3, output_tokens: 4 } },
  });
  assert.equal(resp.usage.total_tokens, 7);
});

test("buildChatResponse: zero usage when upstream omits usage entirely", () => {
  const resp = buildChatResponse({ content: "x", model: "m", upstream: {} });
  assert.equal(resp.usage.prompt_tokens, 0);
  assert.equal(resp.usage.completion_tokens, 0);
  assert.equal(resp.usage.total_tokens, 0);
});

// ────────────────────────────────────────────────────────────────────────
// buildScorableSseChunks — streaming emission for the scorable path
// ────────────────────────────────────────────────────────────────────────

test("buildScorableSseChunks: first chunk primes the assistant role, last carries finish_reason", () => {
  const chunks = buildScorableSseChunks({
    answer: "hello world",
    note: "",
    model: "m",
    finishReason: "stop",
  });

  const first = chunks[0] as {
    choices: Array<{ delta: { role?: string }; finish_reason: string | null }>;
  };
  assert.equal(first.choices[0]?.delta.role, "assistant");
  assert.equal(first.choices[0]?.finish_reason, null);

  const last = chunks[chunks.length - 1] as {
    choices: Array<{ delta: Record<string, unknown>; finish_reason: string | null }>;
  };
  assert.deepEqual(last.choices[0]?.delta, {});
  assert.equal(last.choices[0]?.finish_reason, "stop");
});

test("buildScorableSseChunks: terminal chunk carries the REAL finish_reason", () => {
  const chunks = buildScorableSseChunks({
    answer: "abc",
    note: "",
    model: "m",
    finishReason: "length",
  });
  const last = chunks[chunks.length - 1] as {
    choices: Array<{ finish_reason: string | null }>;
  };
  assert.equal(last.choices[0]?.finish_reason, "length");
});

test("buildScorableSseChunks: every chunk is a chat.completion.chunk object", () => {
  const chunks = buildScorableSseChunks({ answer: "abc", note: "", model: "m" });
  for (const c of chunks) {
    assert.equal((c as { object: string }).object, "chat.completion.chunk");
  }
});

test("buildScorableSseChunks: answer deltas reconstruct the original answer exactly (no note)", () => {
  const answer =
    "The quick brown fox jumps over the lazy dog, then keeps on running well past one chunk boundary.";
  const chunks = buildScorableSseChunks({ answer, note: "", model: "m" });
  const reconstructed = chunks
    .map(
      (c) =>
        (c as { choices: Array<{ delta: { content?: string } }> }).choices[0]
          ?.delta.content ?? ""
    )
    .join("");
  assert.equal(reconstructed, answer);
});

test("buildScorableSseChunks: a note is emitted as a single FINAL content delta before the terminal chunk", () => {
  const answer = "Likely Zurich.";
  const note = "info: consider /verifydeeper.";
  const chunks = buildScorableSseChunks({ answer, note, model: "m" });

  // The terminal chunk is last; the note delta is the one immediately before.
  const noteChunk = chunks[chunks.length - 2] as {
    choices: Array<{ delta: { content?: string }; finish_reason: string | null }>;
  };
  assert.equal(noteChunk.choices[0]?.finish_reason, null);
  assert.equal(noteChunk.choices[0]?.delta.content, `\n\n${note}`);

  // Full reconstruction is answer + blank line + note.
  const reconstructed = chunks
    .map(
      (c) =>
        (c as { choices: Array<{ delta: { content?: string } }> }).choices[0]
          ?.delta.content ?? ""
    )
    .join("");
  assert.equal(reconstructed, `${answer}\n\n${note}`);
});

test("buildScorableSseChunks: empty answer + no note -> role chunk + terminal chunk only", () => {
  const chunks = buildScorableSseChunks({ answer: "", note: "", model: "m" });
  assert.equal(chunks.length, 2); // role priming + terminal, no content/note deltas
});

// ────────────────────────────────────────────────────────────────────────
// End-to-end translation (no I/O): responses result -> annotated stream
// ────────────────────────────────────────────────────────────────────────

test("end-to-end: low-confidence responses result yields a noted SSE stream", () => {
  // Simulate what the upstream /v1/responses would return for a shaky answer.
  const upstream: ResponsesResult = {
    model: "m",
    status: "completed",
    output: [
      {
        type: "message",
        content: [
          {
            type: "output_text",
            text: "Likely Zurich.",
            logprobs: [
              { token: "Likely", logprob: -0.05 },
              { token: " Z", logprob: -7.5 }, // very weak -> very_low band
              { token: "urich", logprob: -0.05 },
              { token: ".", logprob: -0.05 },
            ],
          },
        ],
      },
    ],
  };

  const { text, tokens } = extractFromResponses(upstream);
  const { note } = applyConfidenceNote(text, tokens);
  const chunks = buildScorableSseChunks({
    answer: text,
    note,
    model: "m",
    finishReason: mapFinishReason(upstream),
  });

  const reconstructed = chunks
    .map(
      (c) =>
        (c as { choices: Array<{ delta: { content?: string } }> }).choices[0]
          ?.delta.content ?? ""
    )
    .join("");

  // The streamed text is the answer plus the appended confidence note.
  assert.equal(reconstructed.startsWith("Likely Zurich."), true);
  assert.match(reconstructed, /\/verifydeeper/);
});
