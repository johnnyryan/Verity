#!/usr/bin/env node
/**
 * Confidence proxy — HTTP server (the I/O half).
 *
 * A transparent OpenAI-compatible front-door. External chat clients (Open
 * WebUI, Jan, LibreChat, AnythingLLM, ...) point their "OpenAI-compatible API
 * base URL" at this proxy (default http://localhost:1235/v1) INSTEAD of LM
 * Studio's http://localhost:1234/v1.
 *
 * ───────────────────────────────────────────────────────────────────────────
 * DESIGN: default is byte-for-byte pass-through, EVERYWHERE.
 * ───────────────────────────────────────────────────────────────────────────
 * The proxy's default for EVERY path, INCLUDING POST /v1/chat/completions, is
 * to forward the request to the upstream verbatim and stream the upstream's
 * response straight back. Native streaming, the upstream's real finish_reason /
 * usage / errors / headers, no schema coupling. From the client's point of view
 * it is talking to LM Studio directly.
 *
 * The ONE exception is a chat request the request-shape guard certifies
 * "scorable" (isScorable in ./translate.ts): plain text only, no tools /
 * structured output / vision / n>1, and no parameter /v1/responses cannot
 * honour. ONLY such a request is diverted: it is routed through the upstream's
 * /v1/responses endpoint EXACTLY ONCE so the answer AND its per-token logprobs
 * come from the SAME single generation, scored with the shared confidence
 * classifier, and the low-confidence note (if any) appended. Because the proxy
 * drove that one generation, the logprobs correspond EXACTLY to the returned
 * text, so the note is exact and free (no second probe).
 *
 * Anything NOT scorable (tools, response_format, vision, stop/seed/penalties/
 * logit_bias/top_k, n>1, ...) is forwarded verbatim with NO note. We never
 * silently drop a feature: when in doubt, pass through.
 *
 * All translation + the guard live in ./translate.ts (pure, unit-tested). This
 * file owns only the HTTP wiring and the upstream fetch() calls, mirroring how
 * signals/perplexity.ts owns the I/O while signals/confidence.ts owns scoring.
 *
 * Run:
 *   npm run build && npm run proxy
 *
 * UK English, no em-dashes, heavy comments to match the house style.
 */

import express from "express";
import type { Request, Response } from "express";
import { Readable } from "node:stream";

import {
  PROXY_PORT,
  PROXY_HOST,
  PROXY_UPSTREAM_URL,
  PROXY_UPSTREAM_TIMEOUT_MS,
  PROXY_MAX_REQUEST_BYTES,
  WORKER_API_KEY,
  VERBOSE_LOGGING,
} from "../config.js";
import {
  isScorable,
  chatRequestToResponses,
  extractFromResponses,
  applyConfidenceNote,
  buildChatResponse,
  buildScorableSseChunks,
  mapFinishReason,
  type ChatCompletionsRequest,
  type ResponsesResult,
} from "./translate.js";

// ───────────────────────────────────────────────────────────────────────────
// Upstream helpers
// ───────────────────────────────────────────────────────────────────────────

/**
 * Normalise the configured upstream base URL into a path-join-safe origin.
 * PROXY_UPSTREAM_URL is like "http://localhost:1234/v1"; we strip a trailing
 * slash so `${base}/responses` never doubles up.
 */
function upstreamBase(): string {
  return PROXY_UPSTREAM_URL.replace(/\/+$/, "");
}

/**
 * The Authorization header forwarded on every upstream call. LM Studio ignores
 * the value; a cloud upstream (OpenAI) would require it. WORKER_API_KEY is the
 * single shared credential knob (config.ts), reused here so the proxy and the
 * /verify worker calls authenticate identically.
 */
function authHeader(): Record<string, string> {
  return { Authorization: `Bearer ${WORKER_API_KEY}` };
}

// ───────────────────────────────────────────────────────────────────────────
// Express app
// ───────────────────────────────────────────────────────────────────────────

const app = express();

// We capture the RAW body for EVERY route, including /v1/chat/completions. This
// is the key to true byte-for-byte pass-through: a non-scorable chat request
// must be forwarded with the exact bytes the client sent, so we must NOT let a
// JSON body-parser consume and re-serialise the stream (which would reorder
// keys, drop unknown fields, and change whitespace). Instead we keep the raw
// Buffer and JSON.parse a COPY only for the scorability decision on the chat
// route. Non-chat routes never parse at all.
//
// The raw parser uses a single permissive type matcher and the configured size
// cap as the only guard.
const rawParser = express.raw({
  type: () => true,
  limit: PROXY_MAX_REQUEST_BYTES,
});

// Health check — `curl http://localhost:1235/healthz`. Named /healthz rather
// than /health so it cannot collide with an upstream that happens to expose
// /health (we forward unknown paths, but our own routes win first).
app.get("/healthz", (_req, res) => {
  res.json({
    status: "ok",
    name: "verity-confidence-proxy",
    upstream: upstreamBase(),
    port: PROXY_PORT,
  });
});

// ───────────────────────────────────────────────────────────────────────────
// POST /v1/chat/completions  —  divert ONLY if scorable, else pass through
// ───────────────────────────────────────────────────────────────────────────

app.post(
  "/v1/chat/completions",
  rawParser,
  async (req: Request, res: Response) => {
    // Parse a COPY of the raw body purely to run the scorability guard. If it
    // is not valid JSON, it cannot be a scorable chat request, so we pass it
    // through verbatim and let the upstream produce the real parse error.
    const raw = Buffer.isBuffer(req.body) ? (req.body as Buffer) : null;
    let parsed: ChatCompletionsRequest | null = null;
    if (raw && raw.length > 0) {
      try {
        parsed = JSON.parse(raw.toString("utf8")) as ChatCompletionsRequest;
      } catch {
        parsed = null;
      }
    }

    const verdict = parsed
      ? isScorable(parsed)
      : ({ scorable: false, reason: "unparseable body" } as const);

    if (!verdict.scorable) {
      // NOT scorable -> byte-for-byte pass-through, exactly as if the client
      // talked to LM Studio directly. Native streaming, real finish_reason,
      // real errors, no note. This is the default path.
      if (VERBOSE_LOGGING) {
        // Reason only (a field/feature name); never the message content.
        console.error(`[proxy] chat pass-through (${verdict.reason})`);
      }
      return forwardVerbatim(req, res, raw);
    }

    // Scorable -> drive a single /v1/responses generation, score, annotate.
    // parsed is non-null here (scorable implies it parsed).
    await handleScorableChat(parsed as ChatCompletionsRequest, res);
  }
);

/**
 * Handle a request the guard certified scorable: route it through
 * /v1/responses EXACTLY ONCE, score the returned logprobs, and reply with a
 * faithful chat-completions response (streamed or not, per the client).
 */
async function handleScorableChat(
  body: ChatCompletionsRequest,
  res: Response
): Promise<void> {
  const wantsStream = body.stream === true;
  const fallbackModel = typeof body.model === "string" ? body.model : "unknown";

  try {
    // Translate to a /v1/responses request that asks for logprobs. The guard
    // has already removed any feature this could not carry, so this is lossless.
    const responsesReq = chatRequestToResponses(body);

    // Drive the upstream generation to COMPLETION (non-streaming upstream).
    // Even when the client wants a stream we need the FULL token-logprob set
    // before we can score the band (the weakest token may be the last one), and
    // LM Studio documents logprobs on the finalised output_text rather than per
    // streaming delta. We therefore buffer THIS scorable path only and replay
    // as SSE; every other path streams natively (forwardVerbatim). See the
    // streaming note in translate.ts and docs/confidence-proxy.md.
    const abort = new AbortController();
    const timer = setTimeout(() => abort.abort(), PROXY_UPSTREAM_TIMEOUT_MS);
    let upstreamRes: globalThis.Response;
    try {
      upstreamRes = await fetch(`${upstreamBase()}/responses`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeader() },
        body: JSON.stringify(responsesReq),
        signal: abort.signal,
      });
    } finally {
      clearTimeout(timer);
    }

    if (!upstreamRes.ok) {
      // Surface the upstream's status + body so the client sees a real error
      // rather than a generic 500. We do not parse it; pass the text through.
      const detail = await safeText(upstreamRes);
      if (VERBOSE_LOGGING) {
        console.error(
          `[proxy] upstream /responses ${upstreamRes.status}: ${detail.slice(0, 500)}`
        );
      }
      return sendError(
        res,
        upstreamRes.status,
        `Upstream /v1/responses returned ${upstreamRes.status}. ` +
          `The confidence proxy routes scorable chat requests through ` +
          `/v1/responses (LM Studio exposes it with logprobs; Ollama does ` +
          `not). Detail: ${detail.slice(0, 500)}`
      );
    }

    const upstream = (await upstreamRes.json()) as ResponsesResult;

    // Pull the answer text + per-token logprobs, then score + annotate.
    const { text, tokens } = extractFromResponses(upstream);
    const { content, assessment, note } = applyConfidenceNote(text, tokens);
    const finishReason = mapFinishReason(upstream);
    const model =
      typeof upstream.model === "string" ? upstream.model : fallbackModel;

    if (VERBOSE_LOGGING) {
      // Counts + band only — never the generated text (no user-content logs).
      console.error(
        `[proxy] scored ${tokens.length} tokens -> band=${assessment.band}` +
          (wantsStream ? " (streamed)" : " (non-streamed)")
      );
    }

    if (wantsStream) {
      // Replay as SSE: role chunk, answer content deltas, the note as a single
      // FINAL delta (when not ok), then a terminal chunk carrying the REAL
      // finish_reason, then [DONE].
      res.writeHead(200, {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache, no-transform",
        Connection: "keep-alive",
        // Defeat proxy buffering so the client renders deltas promptly.
        "X-Accel-Buffering": "no",
      });
      const chunks = buildScorableSseChunks({
        answer: text,
        note,
        model,
        finishReason,
      });
      for (const chunk of chunks) {
        res.write(`data: ${JSON.stringify(chunk)}\n\n`);
      }
      res.write("data: [DONE]\n\n");
      res.end();
      return;
    }

    // Non-streaming client: a standard chat.completion object with the REAL
    // finish_reason and the REAL upstream usage.
    const completion = buildChatResponse({
      content,
      model,
      upstream,
      finishReason,
    });
    res.json(completion);
  } catch (err) {
    // Distinguish a timeout/abort from other failures for a clearer message.
    const aborted =
      err instanceof Error &&
      (err.name === "AbortError" || /abort/i.test(err.message));
    if (VERBOSE_LOGGING) console.error("[proxy] scorable chat error:", err);
    sendError(
      res,
      aborted ? 504 : 502,
      aborted
        ? `Upstream generation timed out after ${PROXY_UPSTREAM_TIMEOUT_MS} ms.`
        : `Confidence proxy failed to reach the upstream at ${upstreamBase()}. ` +
            `Is LM Studio running there with the server enabled?`
    );
  }
}

// ───────────────────────────────────────────────────────────────────────────
// Transparent pass-through: non-scorable chat AND every other path
// ───────────────────────────────────────────────────────────────────────────
//
// Forwards to the upstream verbatim: same method, same path + query, the body
// bytes as received, and a forwarded Authorization header. The upstream
// response is STREAMED straight back (status, content-type, and the body bytes
// piped as they arrive) so streaming clients keep true first-token latency and
// binary/text both survive. This keeps /v1/models, /v1/embeddings, a direct
// /v1/responses call, a non-scorable /v1/chat/completions, and any future
// endpoint working exactly as if the client had talked to LM Studio.

app.use(rawParser, async (req: Request, res: Response) => {
  const raw = Buffer.isBuffer(req.body) ? (req.body as Buffer) : null;
  await forwardVerbatim(req, res, raw);
});

/**
 * Forward the current request to the upstream byte-for-byte and stream the
 * upstream response back unmodified.
 *
 * `rawBody` is the already-captured request body (from express.raw); passing it
 * in lets both the chat route (after the scorability check) and the catch-all
 * share one implementation without re-reading the stream.
 */
async function forwardVerbatim(
  req: Request,
  res: Response,
  rawBody: Buffer | null
): Promise<void> {
  // Reconstruct the upstream URL: the upstream base ends in "/v1" and
  // req.originalUrl includes the leading "/v1/..." path, so we join the ORIGIN
  // of the base (scheme+host) with the original path to avoid a doubled
  // "/v1/v1". Example: base "http://localhost:1234/v1" -> origin
  // "http://localhost:1234"; path "/v1/models" -> "http://localhost:1234/v1/models".
  let origin: string;
  try {
    origin = new URL(upstreamBase()).origin;
  } catch {
    return sendError(res, 500, `Invalid PROXY_UPSTREAM_URL: ${PROXY_UPSTREAM_URL}`);
  }
  const target = `${origin}${req.originalUrl}`;

  try {
    // Forward a curated header set. We copy content-type and accept (so SSE /
    // JSON negotiation survives) and force our Authorization. We deliberately
    // drop hop-by-hop headers (host, connection, content-length) which fetch
    // recomputes.
    const headers: Record<string, string> = { ...authHeader() };
    const ct = req.headers["content-type"];
    if (typeof ct === "string") headers["Content-Type"] = ct;
    const accept = req.headers["accept"];
    if (typeof accept === "string") headers["Accept"] = accept;

    // Only methods that carry a body get one. We hand fetch a standalone
    // ArrayBuffer (an unambiguous BodyInit across lib versions; a
    // Buffer/Uint8Array sometimes fails to match the BodyInit type when the
    // installed lib uses the parameterised Uint8Array<ArrayBufferLike>). The
    // copy is the body's bytes exactly as received -> truly byte-for-byte.
    const hasBody =
      req.method !== "GET" &&
      req.method !== "HEAD" &&
      rawBody != null &&
      rawBody.length > 0;
    let bodyBytes: ArrayBuffer | undefined;
    if (hasBody) {
      const src = rawBody as Buffer;
      const copy = new ArrayBuffer(src.byteLength);
      new Uint8Array(copy).set(src);
      bodyBytes = copy;
    }

    const abort = new AbortController();
    const timer = setTimeout(() => abort.abort(), PROXY_UPSTREAM_TIMEOUT_MS);
    let upstreamRes: globalThis.Response;
    try {
      upstreamRes = await fetch(target, {
        method: req.method,
        headers,
        body: bodyBytes,
        signal: abort.signal,
      });
    } finally {
      clearTimeout(timer);
    }

    // Mirror the upstream status + content-type, then STREAM the body straight
    // through as it arrives. Streaming (rather than buffering the whole body)
    // is what preserves time-to-first-token for a streaming pass-through, e.g.
    // a non-scorable chat request with stream:true, or a direct /v1/responses
    // stream. If the upstream gave us no body stream (rare: e.g. 204), end the
    // response immediately.
    res.status(upstreamRes.status);
    const upstreamCt = upstreamRes.headers.get("content-type");
    if (upstreamCt) res.setHeader("Content-Type", upstreamCt);
    // Some upstreams set this to defeat intermediary buffering on SSE; honour
    // it so a streamed body is flushed promptly to the client.
    const accelBuffering = upstreamRes.headers.get("x-accel-buffering");
    if (accelBuffering) res.setHeader("X-Accel-Buffering", accelBuffering);

    if (!upstreamRes.body) {
      res.end();
      return;
    }

    // Pipe the Web ReadableStream straight to the Express response. Node 18+
    // gives us Readable.fromWeb; the pipe forwards backpressure and flushes
    // each chunk as it arrives, so SSE deltas reach the client incrementally.
    const nodeStream = Readable.fromWeb(
      upstreamRes.body as Parameters<typeof Readable.fromWeb>[0]
    );
    // If the client disconnects mid-stream, stop pulling from the upstream.
    res.on("close", () => {
      nodeStream.destroy();
    });
    nodeStream.on("error", (streamErr: unknown) => {
      if (VERBOSE_LOGGING) {
        console.error(`[proxy] passthrough ${target} stream error:`, streamErr);
      }
      // Headers are already sent (status written above), so we can only end.
      try {
        res.end();
      } catch {
        /* ignore */
      }
    });
    nodeStream.pipe(res);
  } catch (err) {
    if (VERBOSE_LOGGING) console.error(`[proxy] passthrough ${target} error:`, err);
    const aborted =
      err instanceof Error &&
      (err.name === "AbortError" || /abort/i.test(err.message));
    sendError(
      res,
      aborted ? 504 : 502,
      aborted
        ? `Upstream request to ${req.originalUrl} timed out.`
        : `Confidence proxy failed to forward ${req.originalUrl} to ${origin}.`
    );
  }
}

// ───────────────────────────────────────────────────────────────────────────
// Small helpers
// ───────────────────────────────────────────────────────────────────────────

/** Read a fetch Response body as text without throwing (best-effort error
 * detail). */
async function safeText(r: globalThis.Response): Promise<string> {
  try {
    return await r.text();
  } catch {
    return "";
  }
}

/**
 * Send an error in the OpenAI error envelope shape so clients that parse it
 * (Open WebUI surfaces `error.message`) show something useful. Guards against a
 * double-send if headers were already flushed (e.g. mid-stream failure).
 */
function sendError(res: Response, status: number, message: string): void {
  if (res.headersSent) {
    // Mid-stream: we can only end the connection; the client will treat the
    // truncated stream as a transport error.
    try {
      res.end();
    } catch {
      /* ignore */
    }
    return;
  }
  res.status(status).json({
    error: {
      message,
      type: "proxy_error",
      code: status,
    },
  });
}

// ───────────────────────────────────────────────────────────────────────────
// Listen
// ───────────────────────────────────────────────────────────────────────────

app.listen(PROXY_PORT, PROXY_HOST, () => {
  // console.error for diagnostics, matching index.ts's logging convention.
  console.error(
    `[proxy] confidence proxy listening on http://${PROXY_HOST}:${PROXY_PORT}/v1`
  );
  console.error(`[proxy] forwarding to upstream ${upstreamBase()}`);
  console.error(
    `[proxy] point your client's OpenAI-compatible base URL at ` +
      `http://${PROXY_HOST}:${PROXY_PORT}/v1`
  );
});
