/**
 * Confidence proxy — pure request/response translation + the scorability guard.
 *
 * This module is the I/O-free core of the confidence proxy. It owns three
 * concerns, all free of fetch()/express so the unit tests can drive every
 * branch with no live LM Studio (mirroring how signals/confidence.ts is tested
 * in isolation from signals/perplexity.ts):
 *
 *   1. The REQUEST-SHAPE GUARD (isScorable). The proxy's default behaviour is
 *      byte-for-byte pass-through (see proxy/server.ts). A chat-completions
 *      request is only diverted through /v1/responses when it is "scorable" —
 *      i.e. when /v1/responses can serve it with FULL FIDELITY and zero feature
 *      loss. The guard is the single source of truth for that decision.
 *
 *   2. chat-completions  ->  responses  translation (chatRequestToResponses),
 *      used ONLY for a request the guard has already certified scorable.
 *
 *   3. responses  ->  chat-completions  translation back to the wire shape the
 *      client expects (extractFromResponses + applyConfidenceNote +
 *      buildChatResponse + the streaming SSE helpers), mapping the REAL finish
 *      reason and the REAL usage rather than hard-coding anything.
 *
 * Why route a scorable request through /v1/responses at all: it is the only
 * logprobs-bearing path on LM Studio, and because the proxy itself drives that
 * single generation the logprobs correspond EXACTLY to the text returned. The
 * confidence note is therefore exact and free — one generation, the logprobs of
 * the very answer the user sees, no second probe.
 *
 * ───────────────────────────────────────────────────────────────────────────
 * WHY THE GUARD IS STRICT (the /v1/responses parameter surface)
 * ───────────────────────────────────────────────────────────────────────────
 * The OpenAI Responses API is a DIFFERENT primitive from Chat Completions, not
 * a superset. Confirmed against the OpenAI Responses reference and LM Studio's
 * OpenResponses docs (see docs/confidence-proxy.md for the URLs), and against
 * the installed openai SDK's ResponseCreateParamsBase type
 * (node_modules/openai/resources/responses/responses.d.ts): the Responses
 * request body's ONLY sampling/length knobs are `temperature`, `top_p` and
 * `max_output_tokens`. The Chat Completions knobs `stop`, `seed`,
 * `presence_penalty`, `frequency_penalty`, `logit_bias`, `top_k` and `n` have
 * NO equivalent on /v1/responses, structured output uses a different shape
 * (`text.format`, not `response_format`), tool calling uses a different tool
 * schema, and vision uses `input_image` content parts.
 *
 * So a request that sets any of those Chat-Completions-only knobs, or asks for
 * tools / structured output / vision / n>1, CANNOT be served by /v1/responses
 * without silently dropping the feature. The whole point of the refactor is to
 * never drop a feature: such a request is NOT scorable and is passed through
 * verbatim by the server with no note. WHEN IN DOUBT, NOT scorable.
 *
 * UK English, no em-dashes, heavy comments to match the house style.
 */

import {
  PROXY_RESPONSES_TOP_LOGPROBS,
  PROXY_DEFAULT_MAX_OUTPUT_TOKENS,
} from "../config.js";
import {
  assessConfidence,
  renderConfidenceNote,
  type TokenLogprob,
} from "../signals/confidence.js";
import type { ConfidenceAssessment } from "../types.js";

// ───────────────────────────────────────────────────────────────────────────
// Wire shapes (the subset we read / produce)
// ───────────────────────────────────────────────────────────────────────────
//
// We deliberately type only the fields we touch and keep an index signature so
// unknown fields are visible to the guard (which inspects them by name) without
// being modelled individually.

/** A single chat message as sent by the client. `content` may be a string or
 * the OpenAI "content parts" array. The guard rejects any non-text content
 * part, so by the time we flatten content for a scorable request it is plain
 * text only. */
export interface ChatMessage {
  role: "system" | "user" | "assistant" | "tool" | "developer" | string;
  content:
    | string
    | Array<{ type?: string; text?: string; [k: string]: unknown }>
    | null;
  [k: string]: unknown;
}

/** The incoming POST /v1/chat/completions body (subset we read). The index
 * signature lets the guard probe arbitrary fields (logit_bias, seed, ...) that
 * we deliberately do not model individually. */
export interface ChatCompletionsRequest {
  model?: string;
  messages?: ChatMessage[];
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  /** Newer OpenAI alias for max_tokens; honoured if present. */
  max_completion_tokens?: number;
  stream?: boolean;
  /** Present here only so the guard can reject it: /v1/responses has no `stop`. */
  stop?: string | string[];
  [k: string]: unknown;
}

/** The POST /v1/responses body we send upstream (only the fields we set). We
 * set ONLY parameters /v1/responses faithfully honours, so a scorable request
 * loses nothing in translation. */
export interface ResponsesRequest {
  model?: string;
  /** Responses API accepts a plain string OR a structured input array. We use
   * the structured form so multi-turn role information survives. */
  input: ResponsesInputItem[];
  include: string[];
  top_logprobs: number;
  max_output_tokens: number;
  temperature?: number;
  top_p?: number;
}

export interface ResponsesInputItem {
  role: string;
  content: string;
}

/** One per-token logprob entry inside a /v1/responses output_text item. */
interface ResponsesLogprob {
  token?: string;
  logprob?: number;
  bytes?: number[];
  top_logprobs?: Array<{ token?: string; logprob?: number }>;
}

/** The /v1/responses result (only the fields we read). */
export interface ResponsesResult {
  id?: string;
  model?: string;
  /** Overall generation status. We map this to a chat finish_reason rather than
   * hard-coding "stop". OpenAI Responses uses
   * "completed" | "incomplete" | "failed" | "in_progress" | "cancelled" |
   * "queued"; LM Studio mirrors the subset that applies to a local generation. */
  status?: string;
  /** When status is "incomplete", why. reason "max_output_tokens" maps to the
   * chat finish_reason "length"; "content_filter" maps to "content_filter". */
  incomplete_details?: { reason?: string; [k: string]: unknown } | null;
  /** output[] -> items; a message item has content[]; an output_text content
   * item carries text + logprobs[]. */
  output?: Array<{
    type?: string;
    role?: string;
    content?: Array<{
      type?: string;
      text?: string;
      logprobs?: ResponsesLogprob[];
      [k: string]: unknown;
    }>;
    [k: string]: unknown;
  }>;
  /** Some builds surface a flattened convenience field. We fall back to it. */
  output_text?: string;
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
    total_tokens?: number;
    [k: string]: unknown;
  };
  [k: string]: unknown;
}

/** The chat-completions response object we hand back to the client. */
export interface ChatCompletionsResponse {
  id: string;
  object: "chat.completion";
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: { role: "assistant"; content: string };
    finish_reason: string;
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

// ───────────────────────────────────────────────────────────────────────────
// REQUEST-SHAPE GUARD — is this chat request scorable?
// ───────────────────────────────────────────────────────────────────────────
//
// A request is scorable ONLY if /v1/responses can serve it with full fidelity.
// If ANY check below fails, the request is NOT scorable and the server forwards
// it byte-for-byte with no note (never silently dropping a feature).
//
// The list of rejected parameters is exactly the set of Chat-Completions knobs
// that /v1/responses cannot honour (see the module header for the evidence).
// Keeping the list as a named constant makes the guard auditable and lets the
// "unsupported parameter" branch report precisely which field tripped it.

/**
 * Chat-completions request fields that /v1/responses cannot faithfully honour.
 * A request that sets any of these (to a meaningful value) is NOT scorable.
 *
 *   stop / stop_sequences  — no stop-sequence parameter on /v1/responses
 *   seed                   — no deterministic-seed parameter
 *   presence_penalty       — no presence penalty
 *   frequency_penalty      — no frequency penalty
 *   logit_bias             — no per-token bias map
 *   top_k                  — not part of the OpenAI sampling surface
 *   logprobs / top_logprobs— client-driven logprob requests would collide with
 *                            the include/top_logprobs WE set; safer to pass such
 *                            a request through untouched
 *   response_format        — structured output uses text.format, a different
 *                            shape; honouring it would mean reshaping, so reject
 *   tools / functions      — different tool schema on /v1/responses
 *   tool_choice / function_call
 *   n                      — handled separately (only n>1 is rejected; n===1 ok)
 */
const UNSUPPORTED_PARAM_KEYS = [
  "stop",
  "stop_sequences",
  "seed",
  "presence_penalty",
  "frequency_penalty",
  "logit_bias",
  "top_k",
  "logprobs",
  "top_logprobs",
  "response_format",
  "tools",
  "tool_choice",
  "functions",
  "function_call",
] as const;

/** The verdict from the guard. `scorable:false` always carries a human-readable
 * reason naming the disqualifying feature (used only for verbose logging; the
 * client never sees it, the request just passes through). */
export type ScorabilityVerdict =
  | { scorable: true }
  | { scorable: false; reason: string };

/**
 * Decide whether an incoming chat-completions request can be routed through
 * /v1/responses with zero feature loss.
 *
 * Scorable IFF ALL hold:
 *   - there is at least one message (nothing to score otherwise),
 *   - every message's content is plain text only (string, or content-parts
 *     that are all type "text"); any image_url / input_image / audio / file
 *     part disqualifies it (vision/audio cannot round-trip through our
 *     text-only translation),
 *   - no tools / tool_choice / functions / function_call,
 *   - no response_format / structured output,
 *   - n is unset or exactly 1,
 *   - no parameter from UNSUPPORTED_PARAM_KEYS is set to a meaningful value.
 *
 * "Meaningful value" means present and not null/undefined; for the collection
 * params (stop, tools, logit_bias, ...) an empty array / empty object is
 * treated as "not set" so a client that sends `tools: []` or `stop: []` (some
 * SDKs always include the key) is not needlessly disqualified.
 *
 * WHEN IN DOUBT, NOT scorable: any unexpected shape returns scorable:false.
 */
export function isScorable(req: ChatCompletionsRequest): ScorabilityVerdict {
  // A non-object body cannot be reasoned about; pass it through.
  if (req == null || typeof req !== "object") {
    return { scorable: false, reason: "request body is not an object" };
  }

  const messages = Array.isArray(req.messages) ? req.messages : null;
  if (!messages || messages.length === 0) {
    return { scorable: false, reason: "no messages to score" };
  }

  // Every message must be plain text. A string is fine. A content-parts array
  // is fine ONLY if every part is a text part; any non-text part (image/audio/
  // file) means the model is being asked to do something /v1/responses-as-text
  // cannot reproduce, so we must not divert it.
  for (const m of messages) {
    const verdict = messageIsPlainText(m);
    if (!verdict.ok) {
      return { scorable: false, reason: verdict.reason };
    }
  }

  // n: only the default single completion is scorable. /v1/responses returns a
  // single response; n>1 would silently collapse to one answer.
  if (req.n != null) {
    const n = req.n;
    if (typeof n !== "number" || n !== 1) {
      return { scorable: false, reason: `n=${String(n)} (only n=1 is scorable)` };
    }
  }

  // Any unsupported parameter set to a meaningful value disqualifies the
  // request. We check by name against the curated list so a future client knob
  // we have not modelled still trips the guard if it lands in that list.
  for (const key of UNSUPPORTED_PARAM_KEYS) {
    if (isMeaningfullySet((req as Record<string, unknown>)[key])) {
      return {
        scorable: false,
        reason: `unsupported parameter "${key}" set (/v1/responses cannot honour it)`,
      };
    }
  }

  return { scorable: true };
}

/**
 * Is a value "meaningfully set" for guard purposes? Present and non-null, and
 * not an empty array / empty object (so a client that always includes
 * `tools: []` or `stop: []` is not disqualified by an empty container).
 */
function isMeaningfullySet(v: unknown): boolean {
  if (v == null) return false;
  if (Array.isArray(v)) return v.length > 0;
  if (typeof v === "object") return Object.keys(v as object).length > 0;
  // A primitive (number/string/boolean) that is present counts as set. An empty
  // string for `stop` is unusual but we treat it as set and pass through, which
  // is the conservative choice.
  return true;
}

/** Result of inspecting a single message for plain-text-only content. */
type PlainTextCheck = { ok: true } | { ok: false; reason: string };

/**
 * A message is plain text iff its content is a string, or null/empty (an
 * assistant turn with only tool_calls has null content — but tool_calls at the
 * top level are rejected separately; a null content message on its own is
 * harmless and flattens to ""), or a content-parts array whose every part is a
 * text part.
 *
 * We also reject a message that carries tool-call payloads
 * (`tool_calls` / `function_call`), because replaying a tool-call turn through
 * a text-only /v1/responses translation would drop the call.
 */
function messageIsPlainText(m: ChatMessage): PlainTextCheck {
  if (m == null || typeof m !== "object") {
    return { ok: false, reason: "message is not an object" };
  }

  // A message that itself carries a tool/function call cannot round-trip.
  if (isMeaningfullySet((m as Record<string, unknown>)["tool_calls"])) {
    return { ok: false, reason: "message carries tool_calls" };
  }
  if (isMeaningfullySet((m as Record<string, unknown>)["function_call"])) {
    return { ok: false, reason: "message carries a function_call" };
  }

  const content = m.content;
  if (content == null || typeof content === "string") return { ok: true };

  if (Array.isArray(content)) {
    for (const part of content) {
      // A bare string part is text. An object part is text ONLY if its `type`
      // is exactly "text" (OpenAI) or unset with a `text` string present. Any
      // other type (image_url, input_image, input_audio, audio, file, ...) is
      // a non-text modality and disqualifies the request.
      if (typeof part === "string") continue;
      if (part == null || typeof part !== "object") {
        return { ok: false, reason: "message has a non-text content part" };
      }
      const type = (part as { type?: unknown }).type;
      if (type === undefined || type === null) {
        // No explicit type: accept only if it looks like a text part.
        if (typeof (part as { text?: unknown }).text === "string") continue;
        return { ok: false, reason: "message has an untyped non-text content part" };
      }
      if (type !== "text") {
        return {
          ok: false,
          reason: `message has a non-text content part (type "${String(type)}")`,
        };
      }
    }
    return { ok: true };
  }

  // Some other content shape we do not understand: do not divert it.
  return { ok: false, reason: "message content has an unrecognised shape" };
}

// ───────────────────────────────────────────────────────────────────────────
// Message normalisation
// ───────────────────────────────────────────────────────────────────────────

/**
 * Flatten a SCORABLE chat message's `content` into a plain string.
 *
 * Only ever called on content the guard already certified plain-text-only, so
 * the array branch concatenates text parts and the (now-unreachable for
 * scorable requests) non-text parts contribute "". Kept defensive so a direct
 * unit-test call with mixed parts still behaves sensibly.
 */
export function flattenContent(content: ChatMessage["content"]): string {
  if (content == null) return "";
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => (typeof part?.text === "string" ? part.text : ""))
      .join("");
  }
  return "";
}

// ───────────────────────────────────────────────────────────────────────────
// chat-completions  ->  responses   (scorable requests only)
// ───────────────────────────────────────────────────────────────────────────

/**
 * Translate a SCORABLE chat-completions request into a /v1/responses request
 * that asks for per-token logprobs.
 *
 * Precondition: isScorable(req).scorable === true. Because the guard has
 * already rejected every parameter /v1/responses cannot honour, this mapping is
 * lossless: there is nothing left to drop.
 *
 * Field mapping:
 *   model                 -> model                 (verbatim)
 *   messages[]            -> input[] (role+content) (content flattened to text)
 *   temperature           -> temperature            (forwarded if set)
 *   top_p                 -> top_p                  (forwarded if set)
 *   max_tokens /          -> max_output_tokens      (chat caps the COMPLETION;
 *     max_completion_tokens                          responses caps OUTPUT)
 *   (always)              -> include:["message.output_text.logprobs"]
 *   (always)              -> top_logprobs: N         (so logprobs[] is populated)
 *
 * `stream` is intentionally NOT forwarded here. Whether the upstream is driven
 * streaming or not is decided in server.ts; this builder only produces the
 * request body (the same body works for both, with `stream` added by the
 * caller when needed).
 */
export function chatRequestToResponses(
  req: ChatCompletionsRequest
): ResponsesRequest {
  const messages = Array.isArray(req.messages) ? req.messages : [];
  const input: ResponsesInputItem[] = messages.map((m) => ({
    // The Responses API recognises system/user/assistant/developer roles.
    // Anything exotic is forwarded as-is; the upstream decides what to do with
    // it rather than us guessing a remapping.
    role: typeof m.role === "string" ? m.role : "user",
    content: flattenContent(m.content),
  }));

  // chat-completions allows either max_tokens (classic) or
  // max_completion_tokens (newer alias). Prefer the classic field if both are
  // somehow present; fall back to the configured default so an omitted cap does
  // not let the model run away.
  const maxOut =
    typeof req.max_tokens === "number"
      ? req.max_tokens
      : typeof req.max_completion_tokens === "number"
        ? req.max_completion_tokens
        : PROXY_DEFAULT_MAX_OUTPUT_TOKENS;

  const out: ResponsesRequest = {
    model: req.model,
    input,
    include: ["message.output_text.logprobs"],
    top_logprobs: PROXY_RESPONSES_TOP_LOGPROBS,
    max_output_tokens: maxOut,
  };

  // Only forward sampling params the client actually set, so we never override
  // an upstream default with an accidental undefined->0 coercion.
  if (typeof req.temperature === "number") out.temperature = req.temperature;
  if (typeof req.top_p === "number") out.top_p = req.top_p;

  return out;
}

// ───────────────────────────────────────────────────────────────────────────
// responses  ->  finish_reason
// ───────────────────────────────────────────────────────────────────────────

/**
 * Map a /v1/responses result's status to a chat-completions finish_reason.
 *
 * We do NOT hard-code "stop". The Responses object carries an overall `status`
 * and, when incomplete, an `incomplete_details.reason`:
 *   status "completed"                              -> "stop"
 *   status "incomplete" + reason "max_output_tokens"-> "length"
 *   status "incomplete" + reason "content_filter"   -> "content_filter"
 *   status "incomplete" (other/unknown reason)      -> "length"  (the usual
 *                                                      cause of an incomplete
 *                                                      local generation)
 *   status "failed"                                 -> "stop"  (the body still
 *                                                      carries whatever text
 *                                                      was produced; the HTTP
 *                                                      layer surfaces hard
 *                                                      failures separately)
 *   status missing/unknown                          -> "stop"
 *
 * A missing status defaults to "stop": older/edge LM Studio builds may omit it,
 * and "stop" is the only safe non-alarming default for a response that did
 * arrive with text.
 */
export function mapFinishReason(result: ResponsesResult): string {
  const status = typeof result.status === "string" ? result.status : "";
  if (status === "incomplete") {
    const reason = result.incomplete_details?.reason;
    if (reason === "content_filter") return "content_filter";
    // max_output_tokens, or any other incomplete reason, is reported as the
    // length cutoff that chat clients understand.
    return "length";
  }
  // completed, failed, missing, or anything else: a normal stop.
  return "stop";
}

// ───────────────────────────────────────────────────────────────────────────
// responses  ->  (text + tokens)
// ───────────────────────────────────────────────────────────────────────────

/**
 * Extract the assistant text and the per-token logprob stream from a
 * /v1/responses result.
 *
 * Walk output[] -> content[]; an output_text content item carries `text` and
 * `logprobs[]`. We concatenate text across all output_text items and collect
 * every token-logprob pair. Tokens whose logprob is missing/non-finite are
 * still pushed (with NaN) so the classifier's own filtering decides what is
 * usable — computeConfidence already drops non-finite logprobs.
 *
 * If the structured walk yields no text we fall back to the flattened
 * `output_text` convenience field some builds provide. In that fallback case
 * there are no logprobs to score (tokens stays empty), which the classifier
 * treats as "cannot assess" -> band ok -> no note. That is the correct,
 * non-fabricating behaviour: we never invent a warning from missing data.
 */
export function extractFromResponses(result: ResponsesResult): {
  text: string;
  tokens: TokenLogprob[];
} {
  let text = "";
  const tokens: TokenLogprob[] = [];

  for (const item of result.output ?? []) {
    for (const content of item.content ?? []) {
      // Accept the canonical "output_text" type, but do not require it: some
      // builds omit/rename `type` while still carrying text + logprobs.
      if (typeof content.text === "string") text += content.text;
      const lps = content.logprobs;
      if (Array.isArray(lps)) {
        for (const lp of lps) {
          tokens.push({
            token: typeof lp?.token === "string" ? lp.token : "",
            logprob: typeof lp?.logprob === "number" ? lp.logprob : Number.NaN,
          });
        }
      }
    }
  }

  if (text.length === 0 && typeof result.output_text === "string") {
    text = result.output_text;
  }

  return { text, tokens };
}

// ───────────────────────────────────────────────────────────────────────────
// Confidence note injection
// ───────────────────────────────────────────────────────────────────────────

/**
 * Score the returned tokens and, when the confidence band is not "ok", append
 * the paste-ready low-confidence note to the answer text.
 *
 * The note is separated from the answer by a blank line so it reads as a
 * distinct trailing remark in the client UI rather than running into the last
 * sentence. When the band is "ok" the note is "" and the text is returned
 * unchanged — confident answers carry no extra text, exactly as in the /verify
 * path (renderConfidenceNote returns "" for ok).
 *
 * Returns the (possibly augmented) text plus the assessment AND the bare note,
 * so the caller can log the band, or stream the note as a separate final SSE
 * delta, without recomputing.
 */
export function applyConfidenceNote(
  text: string,
  tokens: TokenLogprob[]
): { content: string; assessment: ConfidenceAssessment; note: string } {
  const assessment = assessConfidence(tokens);
  const note = renderConfidenceNote(assessment);
  const content = note ? `${text}\n\n${note}` : text;
  return { content, assessment, note };
}

// ───────────────────────────────────────────────────────────────────────────
// (text)  ->  chat-completions response  (scorable, non-streaming)
// ───────────────────────────────────────────────────────────────────────────

let monotonicCounter = 0;
/** Stable-ish id for a synthesised completion. The exact value is opaque to
 * clients; we only need uniqueness within a process. */
function makeCompletionId(): string {
  monotonicCounter = (monotonicCounter + 1) % Number.MAX_SAFE_INTEGER;
  return `chatcmpl-proxy-${Date.now().toString(36)}-${monotonicCounter}`;
}

/**
 * Build a standard non-streaming chat-completions response object around the
 * final assistant `content`.
 *
 * finish_reason is the REAL mapped reason from the upstream Responses status
 * (mapFinishReason), unless the caller overrides it. We never hard-code "stop".
 *
 * Usage numbers are taken from the upstream responses result when present
 * (input_tokens/output_tokens) and mapped to the chat-completions names
 * (prompt_tokens/completion_tokens). They are best-effort: a client that does
 * not display usage is unaffected, and one that does sees the upstream's real
 * counts. We do NOT recount the appended note into completion_tokens because
 * the note was not model-generated; the small discrepancy is acceptable and
 * documented here.
 */
export function buildChatResponse(params: {
  content: string;
  model: string;
  upstream: ResponsesResult;
  finishReason?: string;
}): ChatCompletionsResponse {
  const { content, model, upstream } = params;
  const usage = upstream.usage ?? {};
  const promptTokens =
    typeof usage.input_tokens === "number" ? usage.input_tokens : 0;
  const completionTokens =
    typeof usage.output_tokens === "number" ? usage.output_tokens : 0;
  const totalTokens =
    typeof usage.total_tokens === "number"
      ? usage.total_tokens
      : promptTokens + completionTokens;

  return {
    id: makeCompletionId(),
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content },
        // Real finish reason from the upstream status, not a hard-coded "stop".
        finish_reason: params.finishReason ?? mapFinishReason(upstream),
      },
    ],
    usage: {
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
      total_tokens: totalTokens,
    },
  };
}

// ───────────────────────────────────────────────────────────────────────────
// Streaming (SSE) emission  (scorable path only)
// ───────────────────────────────────────────────────────────────────────────
//
// STREAMING APPROACH FOR THE SCORABLE PATH — buffer-then-replay, by necessity.
//
// The confidence band depends on the WHOLE token-logprob set (the single
// weakest token might be the last one generated), so the note can only be
// emitted AFTER the full answer is known. Ideally we would still stream the
// answer's text deltas to the client as they arrive (preserving
// time-to-first-token) and then append the note as a final delta. That is only
// possible if /v1/responses streaming delivers logprobs INCREMENTALLY, attached
// to each `response.output_text.delta` event.
//
// As of LM Studio's documented Responses streaming (verified 2026-05-22 against
// the OpenResponses docs and the documented delta payload shape), logprobs are
// documented as a property of the FINALISED `output_text` content object, NOT
// of each streaming delta. The streaming-events docs do not specify a per-delta
// logprobs field. Per the design decision, when streaming does not deliver
// logprobs incrementally we fall back to buffering for THIS scorable path: we
// drive the upstream generation to completion (non-streaming upstream), obtain
// the full text + logprobs, compute the band, append the note, then replay the
// finished content to the client as SSE chunks followed by `[DONE]`.
//
// IMPORTANT SCOPE: this buffering applies ONLY to the small "scorable plain
// text + client wants a stream" slice. EVERY non-scorable chat request and
// every non-chat request streams NATIVELY via byte-for-byte pass-through in
// server.ts (the upstream's own SSE bytes are piped straight through, never
// buffered). So tools, vision, structured output, n>1, and any request with an
// unsupported parameter keep true first-token latency; only the gated,
// plain-text, scorable answers trade TTFT for an exact confidence note. This is
// documented for operators in docs/confidence-proxy.md (capability matrix).
//
// We chunk the answer text on a fixed character budget rather than
// re-tokenising: the SSE `delta.content` field is plain text and clients
// reassemble by concatenation, so any split is valid.

/** Characters per streamed SSE content delta. Small enough to look incremental,
 * large enough to avoid an event per character. Cosmetic only (clients
 * concatenate deltas regardless), so not a config knob. */
const SSE_CHUNK_CHARS = 48;

/** The shared envelope every chunk in one streamed response carries. */
function sseBase(id: string, created: number, model: string) {
  return { id, object: "chat.completion.chunk" as const, created, model };
}

/**
 * Convert a finished assistant ANSWER plus its (already-computed) confidence
 * NOTE into the ordered list of SSE data payloads that reproduce a streaming
 * chat-completions response.
 *
 * Sequence:
 *   - one role-priming chunk            (delta:{role:"assistant"})
 *   - N answer content chunks           (delta:{content:"..."})
 *   - one note delta, iff note != ""    (delta:{content:"\n\n<note>"}) emitted
 *     as a SINGLE FINAL delta before the terminal chunk, exactly as the design
 *     requires ("emit the confidence note as a FINAL delta before [DONE]")
 *   - one terminal chunk                (delta:{}, finish_reason:<real reason>)
 * The caller writes each as `data: <json>\n\n` then `data: [DONE]\n\n`.
 *
 * Passing the note separately (rather than pre-concatenated into `content`)
 * keeps the note as a distinct trailing delta on the wire, which is what the
 * design asks for and is easy to assert on in tests. When `note` is "" the
 * stream is just the answer (role + content deltas + terminal), byte-identical
 * in meaning to a confident non-streamed answer.
 *
 * Returning plain objects (not pre-serialised strings or written bytes) keeps
 * this pure and unit-testable.
 */
export function buildScorableSseChunks(params: {
  answer: string;
  note: string;
  model: string;
  finishReason?: string;
}): Array<Record<string, unknown>> {
  const { answer, note, model } = params;
  const id = makeCompletionId();
  const created = Math.floor(Date.now() / 1000);
  const base = sseBase(id, created, model);

  const chunks: Array<Record<string, unknown>> = [];

  // 1) Role-priming delta. OpenAI streams emit the assistant role once, up
  //    front, before any content. Clients rely on this to open the message.
  chunks.push({
    ...base,
    choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }],
  });

  // 2) Answer content deltas. Split on a fixed character budget. An empty
  //    answer yields no content chunks (valid empty completion).
  for (let i = 0; i < answer.length; i += SSE_CHUNK_CHARS) {
    chunks.push({
      ...base,
      choices: [
        {
          index: 0,
          delta: { content: answer.slice(i, i + SSE_CHUNK_CHARS) },
          finish_reason: null,
        },
      ],
    });
  }

  // 3) The confidence note, as a single final content delta, separated from the
  //    answer by a blank line. Skipped entirely for a confident (ok) answer.
  if (note) {
    chunks.push({
      ...base,
      choices: [
        { index: 0, delta: { content: `\n\n${note}` }, finish_reason: null },
      ],
    });
  }

  // 4) Terminal delta: empty delta carrying the REAL finish_reason. The caller
  //    follows this with `data: [DONE]`.
  chunks.push({
    ...base,
    choices: [
      { index: 0, delta: {}, finish_reason: params.finishReason ?? "stop" },
    ],
  });

  return chunks;
}
