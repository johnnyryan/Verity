#!/usr/bin/env node
/**
 * MCP Verity Server — entry point.
 *
 * Exposes a single MCP tool `verify_answer` over HTTP. Registered
 * with LM Studio's MCP client via the LM Studio settings UI.
 *
 * Transport: Streamable HTTP on SERVER_PORT, matching the conventions of
 * the other servers in the MCP-LMstudio repo.
 *
 * Run:
 *   npm run build && npm start
 * Or from the MCP-LMstudio repo root:
 *   npm run start:verity
 */

import express from "express";
import { randomUUID } from "node:crypto";
import { createRequire } from "node:module";

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

import {
  MAX_ANSWER_CHARS,
  MAX_PRIOR_CONTEXT_CHARS,
  MAX_QUESTION_CHARS,
  MAX_REQUEST_BYTES,
  NLI_IMPL,
  SERVER_HOST,
  SERVER_PORT,
  SESSION_IDLE_TIMEOUT_MS,
  SESSION_SWEEP_INTERVAL_MS,
  VERBOSE_LOGGING,
} from "./config.js";
import { runVerificationPipeline } from "./pipeline.js";
import type { VerifyInput, TaskType, ContextMode } from "./types.js";
import { runSecondOpinion } from "./second-opinion/consult.js";
import type { ConsultInput } from "./second-opinion/consult.js";
import { warmupClassifier } from "./nli/classifier.js";
import { warmupTokenizer } from "./tokenizer.js";

// Resolve the shipped version once at startup so the MCP handshake and
// /health agree with package.json without manual duplication.
const require = createRequire(import.meta.url);
const PKG: { name?: string; version?: string } = require("../package.json");
const SERVER_VERSION = PKG.version ?? "0.0.0";

// ═══════════════════════════════════════════════════════════════════════════
// MCP server setup
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Build a fresh MCP Server instance.
 *
 * The SDK's Server class owns exactly one transport at a time. If we
 * reused a singleton across sessions, the second client's handshake
 * would error with "Already connected to a transport" (seen 2026-04-16
 * when a second LM Studio session opened). Creating one per session is
 * cheap — handler registration is a few function assignments — and lets
 * each session's transport own its own Server.
 */
function buildServer(): Server {
  const server = new Server(
    {
      name: "verity",
      version: SERVER_VERSION,
    },
    {
      capabilities: {
        tools: {},
      },
    }
  );

  server.setRequestHandler(ListToolsRequestSchema, async () => TOOL_LISTING);
  server.setRequestHandler(CallToolRequestSchema, handleToolCall);

  return server;
}

const TOOL_LISTING = {
  tools: [
    {
      name: "verify_answer",
      description:
        "Audits an answer with critic models + NLI claim-check. Returns " +
        "a ready-to-paste Markdown block (answer echoed, verdict table, " +
        "findings, follow-up prompt).\n" +
        "\n" +
        "WHEN TO CALL: user typed `/verify`, `/verifydeep`, or " +
        "`/verifydeeper` in their latest message. Map to mode=standard / " +
        "deep / deeper.\n" +
        "\n" +
        "ALSO CALL when your last assistant turn was a Verity block " +
        "ending with the 'Awaiting your reply' prompt and the user " +
        "replied affirmatively (yes / OK / `/verifydeeper` / sure) — " +
        "call with mode='deeper'. If the user replied 'redraft', " +
        "rewrite the answer (fetch-verifying every URL — do not " +
        "fabricate sources) then call this tool again on the rewrite. " +
        "If they replied 'no', do nothing.\n" +
        "\n" +
        "FLOW for `/verify` on a fresh question:\n" +
        "  1. Write a substantive prose answer. Back non-trivial claims " +
        "with fetched URLs cited as [N], [author], [publisher], [year], " +
        "[page], [url]. Drop claims you can't source.\n" +
        "  2. Call verify_answer with question + your prose answer.\n" +
        "  3. Paste the returned Markdown block verbatim into chat. LM " +
        "Studio collapses tool results by default; the user only sees " +
        "what's in your assistant message body.\n" +
        "  4. Stop. The block has its own follow-up prompt; wait for " +
        "the user's reply, do not redraft unprompted.\n" +
        "\n" +
        "DO NOT: skip the tool call when /verify was typed; paraphrase " +
        "the block (paste verbatim); redraft without explicit user " +
        "consent; invent the verdict table from your own reasoning.",
      // Tool description history (length tunings, schema simplifications)
      // is recorded in design.md → "Change log". Substantive guidance for
      // operators lives there, not inline.
      inputSchema: {
        // Hidden but still server-controlled:
        //   task_type   — always "auto" (server auto-detects code vs
        //                 prose vs reasoning vs research).
        //   context_mode — derived: "with_context" if prior_context is
        //                  provided, "minimal" otherwise.
        //   use_nli     — always true (NLI is ~300ms, no reason to skip).
        type: "object",
        properties: {
          question: {
            type: "string",
            description: "The user's most recent question.",
          },
          answer: {
            type: "string",
            description:
              "REQUIRED. The prose answer YOU JUST COMPOSED in the " +
              "current turn — what the user is going to read. This is " +
              "the text the critics will examine. Do NOT pass an " +
              "earlier assistant message from chat history (e.g. a " +
              "greeting from a prior turn); pass the answer you wrote " +
              "in response to the user's current question. Do NOT pass " +
              "a placeholder, summary, or 'see above'; pass the full " +
              "composed prose. If you have not yet composed an answer, " +
              "compose it first, then call this tool.",
          },
          mode: {
            type: "string",
            enum: ["standard", "deep", "deeper"],
            default: "standard",
            description:
              "Verification depth. 'standard' (~11s) runs critics + NLI. " +
              "'deep' (~30s) adds consistency check (2 worker re-samples) " +
              "and perplexity rescore. 'deeper' (~50s) adds 5 worker " +
              "re-samples and a regeneration fallback for perplexity. " +
              "Map /verify→standard, /verifydeep→deep, /verifydeeper→deeper.",
          },
          prior_context: {
            type: "string",
            description:
              "Optional. The earlier conversation turns the answer depends " +
              "on — documents / code / data / specs the user pasted, plus " +
              "any prior question-answer pairs relevant to the current " +
              "answer. Include all earlier turns that informed the " +
              "answer; omit small talk. Keep under 24k tokens. OMIT this " +
              "field entirely if there is no relevant prior context (e.g. " +
              "the user's question is the first message in the session) — " +
              "do NOT pass empty string, 'none', 'n/a', or a placeholder.",
          },
          task_type: {
            type: "string",
            enum: ["auto", "code", "prose", "reasoning", "research"],
            default: "auto",
            description:
              "Optional. The kind of answer, which picks the critic lens. " +
              "Leave unset (auto) and the server detects it from the answer " +
              "(code fences -> code; citations / years -> research, etc.). " +
              "Set it only when the user forces a lens with 'as code', " +
              "'as prose', 'as reasoning', or 'as research'.",
          },
        },
        required: ["question", "answer"],
      },
    },
    {
      name: "consult_second_opinion",
      description:
        "Consult two independent cross-family models (one per GPU) in " +
        "parallel for a second opinion on the user's question, then run an " +
        "analysis pass on NVIDIA comparing the two answers. Returns both " +
        "answers plus a structured {agreements, disputes, table_html, table_md} " +
        "analysis object. In auto mode the analysis also synthesises a " +
        "final_answer. " +
        "\n\n" +
        "SOURCING NOTE: If the user also typed /verify in this turn, the " +
        "answer you compose AFTER this tool returns must follow the " +
        "verify_answer sourcing contract — every non-trivial " +
        "fact backed by a fetched/working URL, structured " +
        "[N],[author],[publisher],[year],[page],[url] citations, no " +
        "fabricated fields. See verify_answer's description for " +
        "the full contract. /second's parallel opinion does not exempt " +
        "you from sourcing.\n\n" +
        "INVOKE THIS TOOL when:\n" +
        "  - the user's latest message matches '/second' or 'second opinion' " +
        "or 'ask the other model',\n" +
        "  - or you are about to commit to a non-trivial answer and want a " +
        "parallel sanity check — call this tool once, near the start of " +
        "your reasoning, BEFORE you write your final answer.\n" +
        "\n" +
        "COMPLEMENTARY WITH /verify: consult_second_opinion and " +
        "verify_answer are independent tools, not alternatives. " +
        "/second runs BEFORE you write your answer (parallel opinion); " +
        "/verify runs AFTER (audit of what you wrote). If the user " +
        "appended '/verify' to their question, you MUST still call " +
        "verify_answer after composing your answer — even if you " +
        "already called this /second tool proactively at the start. Calling " +
        "/second does NOT replace or satisfy a /verify request. Skipping " +
        "/verify when the user explicitly asked for it is a hard error.\n" +
        "\n" +
        "Inputs:\n" +
        "  question         — the user's current question (required).\n" +
        "  worker_draft     — your in-progress draft answer (optional but " +
        "recommended; enables a rough agreement score in the return).\n" +
        "  prior_context    — earlier chat content the question depends on, " +
        "same semantics as verify_answer's prior_context.\n" +
        "  model            — optional override of the Ollama model tag to " +
        "consult. If set, the tool runs the legacy single-Ollama path (no " +
        "dual dispatch, no analysis).\n" +
        "  resolution_mode  — 'manual' (default) returns both answers plus a " +
        "structured diff analysis so the user can decide. 'auto' also " +
        "generates a synthesized final_answer from NVIDIA.\n" +
        "\n" +
        "AFTER THE TOOL RETURNS: In manual mode, show the user both answers " +
        "AND render the disputes table verbatim. The tool provides BOTH " +
        "`analysis.table_html` (pre-styled HTML with red header row and " +
        "bordered cells) AND `analysis.table_md` (plain Markdown pipe table) " +
        "— paste one of them into chat EXACTLY AS-IS, do not reformat or " +
        "paraphrase it. The HTML variant renders richer in LM Studio if " +
        "HTML passthrough is enabled; the Markdown variant is the safe " +
        "fallback. Do NOT strip, summarise, or replace the table with a " +
        "textual recap — the user explicitly wants to see the table. " +
        "In auto mode, present the analysis.final_answer directly, still " +
        "followed by the table so the user sees the reasoning. " +
        "Treat the second opinions as advice, not as verdicts.",
      inputSchema: {
        type: "object",
        properties: {
          question: {
            type: "string",
            description: "The user's current question.",
          },
          worker_draft: {
            type: "string",
            description:
              "Your (primary model's) in-progress or draft answer. Optional " +
              "but enables a rough agreement score in the return. A single " +
              "paragraph summarising your intended answer is enough.",
          },
          prior_context: {
            type: "string",
            description:
              "Relevant earlier conversation / pasted docs. Same semantics " +
              "as verify_answer's prior_context.",
          },
          model: {
            type: "string",
            description:
              "Optional Ollama model tag to consult (e.g. 'phi4-mini:3.8b', " +
              "'gemma3:4b'). If set, forces the legacy single-Ollama path " +
              "(no dual dispatch, no analysis). Omit to use the default " +
              "dual-GPU + analysis flow.",
          },
          resolution_mode: {
            type: "string",
            enum: ["manual", "auto"],
            default: "manual",
            description:
              "Manual: return both answers plus a structured diff analysis " +
              "so the user can decide. Auto: also generate a synthesized " +
              "final_answer from NVIDIA. Default is 'manual'.",
          },
        },
        required: ["question"],
      },
    },
  ],
} as const;

// Tool invocation: route to the appropriate handler.
async function handleToolCall(
  request: { params: { name: string; arguments?: unknown } }
): Promise<{ content: Array<{ type: "text"; text: string }> }> {
  const toolName = request.params.name;

  if (toolName === "verify_answer") {
    const args = (request.params.arguments ?? {}) as Partial<VerifyInput>;

    if (typeof args.question !== "string" || typeof args.answer !== "string") {
      throw new Error(
        "verify_answer requires 'question' and 'answer' strings"
      );
    }

    // 2026-05-12 (B1): per-field length caps. The worker is untrusted
    // (passes arbitrary strings via MCP). Anything past these limits
    // would get truncated by callCritic's fitToContext anyway and the
    // latency cost is borne by the verity process — cleaner to reject
    // up front with a clear error.
    if (args.question.length > MAX_QUESTION_CHARS) {
      throw new Error(
        `verify_answer: 'question' exceeds ${MAX_QUESTION_CHARS} chars ` +
          `(got ${args.question.length})`
      );
    }
    if (args.answer.length > MAX_ANSWER_CHARS) {
      throw new Error(
        `verify_answer: 'answer' exceeds ${MAX_ANSWER_CHARS} chars ` +
          `(got ${args.answer.length})`
      );
    }
    if (
      typeof args.prior_context === "string" &&
      args.prior_context.length > MAX_PRIOR_CONTEXT_CHARS
    ) {
      throw new Error(
        `verify_answer: 'prior_context' exceeds ${MAX_PRIOR_CONTEXT_CHARS} ` +
          `chars (got ${args.prior_context.length})`
      );
    }

    // 2026-05-12 (E1): validate enum membership for mode / task_type /
    // context_mode before casting. The previous `as TaskType` etc.
    // bypassed validation entirely — a caller passing
    // `task_type: "rm -rf"` reached the pipeline unchecked. Bad values
    // now fall back to the default rather than poisoning downstream
    // prompt selection.
    const VALID_MODES = new Set(["standard", "deep", "deeper"] as const);
    const VALID_TASK_TYPES = new Set(
      ["code", "prose", "reasoning", "research", "auto"] as const
    );
    const VALID_CONTEXT_MODES = new Set(
      ["minimal", "with_context", "full"] as const
    );
    const argsRecord = args as Record<string, unknown>;
    const mode = VALID_MODES.has(args.mode as never)
      ? (args.mode as "standard" | "deep" | "deeper")
      : "standard";
    const rawTaskType = argsRecord.task_type;
    const taskType: TaskType = VALID_TASK_TYPES.has(rawTaskType as never)
      ? (rawTaskType as TaskType)
      : "auto";
    const hasPriorContext =
      typeof args.prior_context === "string" && args.prior_context.trim().length > 0;
    const rawContextMode = argsRecord.context_mode;
    const contextMode: ContextMode = VALID_CONTEXT_MODES.has(
      rawContextMode as never
    )
      ? (rawContextMode as ContextMode)
      : hasPriorContext
        ? "with_context"
        : "minimal";

    const result = await runVerificationPipeline({
      question: args.question,
      answer: args.answer,
      mode,
      task_type: taskType,
      context_mode: contextMode,
      prior_context: hasPriorContext ? args.prior_context : undefined,
      use_nli: argsRecord.use_nli === false ? false : true,
    });

    // Send the pre-rendered Markdown summary as the primary tool-response
    // text. Smaller worker models tend to paste tool output verbatim into
    // chat; returning the JSON-stringified VerifyOutput made them dump
    // raw JSON. Returning summary_md means whatever the worker pastes is
    // already a clean Markdown table -- no extraction step required.
    // 2026-05-20: previous fallback dumped the entire raw VerifyOutput as
    // JSON when summary_md happened to be empty. That dump leaked the
    // structured payload into chat (including any LLM-side internals the
    // critics returned) and was almost never what the user wanted. Use
    // a short canned message instead; the structured fields are still
    // available via the MCP resource path for callers that need them.
    const text =
      result.summary_md ||
      "Verification result was empty; see structured fields for details.";
    return {
      content: [{ type: "text" as const, text }],
    };
  }

  if (toolName === "consult_second_opinion") {
    const args = (request.params.arguments ?? {}) as Partial<ConsultInput>;
    if (typeof args.question !== "string") {
      throw new Error("consult_second_opinion requires a 'question' string");
    }
    // 2026-05-12 (B1): same per-field caps as verify_answer.
    if (args.question.length > MAX_QUESTION_CHARS) {
      throw new Error(
        `consult_second_opinion: 'question' exceeds ${MAX_QUESTION_CHARS} ` +
          `chars (got ${args.question.length})`
      );
    }
    if (
      typeof args.worker_draft === "string" &&
      args.worker_draft.length > MAX_ANSWER_CHARS
    ) {
      throw new Error(
        `consult_second_opinion: 'worker_draft' exceeds ${MAX_ANSWER_CHARS} ` +
          `chars (got ${args.worker_draft.length})`
      );
    }
    if (
      typeof args.prior_context === "string" &&
      args.prior_context.length > MAX_PRIOR_CONTEXT_CHARS
    ) {
      throw new Error(
        `consult_second_opinion: 'prior_context' exceeds ` +
          `${MAX_PRIOR_CONTEXT_CHARS} chars (got ${args.prior_context.length})`
      );
    }
    const resolution_mode =
      args.resolution_mode === "auto" ? "auto" : "manual";
    const result = await runSecondOpinion({
      question: args.question,
      worker_draft: args.worker_draft,
      prior_context: args.prior_context,
      model: args.model,
      resolution_mode,
    });
    // Compose a Markdown response so smaller worker models that paste
    // tool output verbatim get a clean block instead of a JSON wall.
    // Mirrors what the verify_answer path does with summary_md.
    return {
      content: [
        { type: "text" as const, text: renderConsultMarkdown(result, resolution_mode) },
      ],
    };
  }

  throw new Error(`Unknown tool: ${toolName}`);
}

// ---------------------------------------------------------------------------
// /second response composer
// ---------------------------------------------------------------------------
//
// Workers tend to paste tool output verbatim; if we send back the raw
// JSON-stringified ConsultOutput, chat ends up with a JSON wall. We
// compose a clean Markdown block that includes both legs' answers, the
// pre-rendered analysis table, and (in auto mode) the synthesised
// final answer. The structured fields are still available to anyone
// hitting the MCP API directly -- they just don't get pasted into
// chat by accident.
function renderConsultMarkdown(
  result: Awaited<ReturnType<typeof runSecondOpinion>>,
  mode: "manual" | "auto"
): string {
  const parts: string[] = [];
  const dual = result.dual_opinion;

  // Header
  parts.push(`**Second opinion** -- mode=${mode}`);
  if (result.diff_summary && result.diff_summary.trim().length > 0) {
    parts.push("");
    parts.push(`_${result.diff_summary}_`);
  }

  // AMD leg
  parts.push("");
  parts.push(`### ${result.model} (AMD leg)`);
  parts.push("");
  if (result.unavailable) {
    parts.push(`_unavailable: ${result.error ?? "no detail"}_`);
  } else {
    parts.push(result.second_opinion.trim());
  }

  // NVIDIA leg (dual)
  if (dual) {
    parts.push("");
    parts.push(`### ${dual.model} (NVIDIA leg)`);
    parts.push("");
    if (dual.unavailable) {
      parts.push(`_unavailable: ${dual.error ?? "no detail"}_`);
    } else {
      parts.push(dual.second_opinion.trim());
    }
  }

  // Analysis pass
  const analysis = result.analysis;
  if (analysis) {
    parts.push("");
    parts.push(`### Analysis (${analysis.model})`);
    if (analysis.unavailable) {
      parts.push("");
      parts.push(`_analysis unavailable: ${analysis.error ?? "no detail"}_`);
    } else {
      if (analysis.agreements && analysis.agreements.length > 0) {
        parts.push("");
        parts.push("**Agreements:**");
        for (const a of analysis.agreements) {
          parts.push(`- ${a}`);
        }
      }
      if (analysis.table_md && analysis.table_md.trim().length > 0) {
        parts.push("");
        parts.push("**Disputes:**");
        parts.push("");
        parts.push(analysis.table_md);
      }
      if (mode === "auto" && analysis.final_answer && analysis.final_answer.trim().length > 0) {
        parts.push("");
        parts.push("### Synthesised final answer");
        parts.push("");
        parts.push(analysis.final_answer.trim());
      }
    }
  }

  // Latency footer
  const totalMs = result.latency_ms + (dual?.latency_ms ?? 0) + (analysis?.latency_ms ?? 0);
  parts.push("");
  parts.push(`_AMD ${result.latency_ms} ms${dual ? ` · NVIDIA ${dual.latency_ms} ms` : ""}${analysis ? ` · analysis ${analysis.latency_ms} ms` : ""} · total wall ≈ ${totalMs} ms_`);

  return parts.join("\n");
}

// ═══════════════════════════════════════════════════════════════════════════
// HTTP transport (Streamable HTTP)
// ═══════════════════════════════════════════════════════════════════════════

const app = express();
// 2026-05-12: was "50mb". The worker model is untrusted (passes
// arbitrary strings via MCP), and the previous limit invited trivial
// DoS via one huge `answer`. MAX_REQUEST_BYTES defaults to 4 MB; raise
// via VERITY_MAX_REQUEST_BYTES if you genuinely need bigger payloads.
app.use(express.json({ limit: MAX_REQUEST_BYTES }));

// Health check — useful for `curl http://localhost:8090/health`.
app.get("/health", (_req, res) => {
  res.json({
    status: "ok",
    name: "verity",
    version: SERVER_VERSION,
    active_sessions: sessionCount(),
  });
});

// Per-session transport bookkeeping. MCP's streamable HTTP keeps the
// session alive across multiple requests using a header.
//
// We track `lastActivity` alongside the transport so a background sweep
// can reclaim sessions that disconnected without firing onclose (network
// blip, LM Studio restart, browser tab closed).
//
// Header key is `mcp-session-id` because that is what the MCP HTTP
// transport spec defines; clients (LM Studio, MCP CLI, anything that
// speaks the protocol) are expected to send this exact header. Not
// configurable.
interface SessionEntry {
  transport: StreamableHTTPServerTransport;
  lastActivity: number;
}
const sessions: Map<string, SessionEntry> = new Map();
const sessionCount = () => sessions.size;

function touch(sessionId: string): void {
  const entry = sessions.get(sessionId);
  if (entry) entry.lastActivity = Date.now();
}

// Background sweep: drop sessions idle longer than SESSION_IDLE_TIMEOUT_MS.
// unref() so the timer doesn't keep the process alive on shutdown.
const sweep = setInterval(() => {
  const cutoff = Date.now() - SESSION_IDLE_TIMEOUT_MS;
  for (const [id, entry] of sessions) {
    if (entry.lastActivity < cutoff) {
      if (VERBOSE_LOGGING) {
        console.error(`[verity] pruning idle session ${id}`);
      }
      try {
        // Best-effort close; implementations vary.
        (entry.transport as unknown as { close?: () => void }).close?.();
      } catch {
        /* ignore */
      }
      sessions.delete(id);
    }
  }
}, SESSION_SWEEP_INTERVAL_MS);
sweep.unref?.();

app.all("/mcp", async (req, res) => {
  try {
    const sessionId = (req.headers["mcp-session-id"] as string) ?? "";

    let transport: StreamableHTTPServerTransport;
    const existing = sessionId ? sessions.get(sessionId) : undefined;
    if (existing) {
      transport = existing.transport;
      existing.lastActivity = Date.now();
    } else {
      transport = new StreamableHTTPServerTransport({
        sessionIdGenerator: () => randomUUID(),
        onsessioninitialized: (id) => {
          sessions.set(id, { transport, lastActivity: Date.now() });
        },
      });
      transport.onclose = () => {
        if (transport.sessionId) sessions.delete(transport.sessionId);
      };
      // Each session gets its own Server instance (see buildServer docs).
      await buildServer().connect(transport);
    }

    await transport.handleRequest(req, res, req.body);
    if (transport.sessionId) touch(transport.sessionId);
  } catch (err) {
    if (VERBOSE_LOGGING) console.error("[/mcp] error:", err);
    if (!res.headersSent) {
      res.status(500).json({
        jsonrpc: "2.0",
        error: { code: -32603, message: "Internal server error" },
      });
    }
  }
});

app.listen(SERVER_PORT, SERVER_HOST, () => {
  // Logging convention: this file (and the rest of Verity) uses
  // `console.error` for all server-side diagnostic output even on the
  // HTTP transport. That's the historical stdio-MCP convention (stdout
  // is reserved for the JSON-RPC channel) and is preserved here for
  // consistency, even though on HTTP transport stdout is unused and
  // would be acceptable too. If a structured logger is ever added,
  // swap every `console.error` together so the convention stays
  // uniform across stdio + HTTP code paths.
  console.error(
    `[verity] MCP server v${SERVER_VERSION} listening on http://${SERVER_HOST}:${SERVER_PORT}/mcp`
  );
  console.error(`[verity] health check at http://${SERVER_HOST}:${SERVER_PORT}/health`);

  // Non-blocking boot warmup. Pays the ONNX cold-load + tiktoken-init costs
  // off the request hot path so the first /verify call hits warm caches.
  // setImmediate yields back to the event loop so app.listen's callback
  // returns immediately; the server accepts connections during warmup.
  setImmediate(async () => {
    const t0 = Date.now();
    const tasks: Array<Promise<void>> = [];
    // Tokenizer warmup is sync internally but wrap to compose with classifier.
    tasks.push(Promise.resolve().then(() => warmupTokenizer()));
    // Skip the ~1 GB DeBERTa load when NLI isn't using it.
    if (NLI_IMPL === "deberta") {
      tasks.push(warmupClassifier());
    }
    try {
      await Promise.all(tasks);
      console.error(`[verity] warmup complete in ${Date.now() - t0}ms`);
    } catch (err) {
      // warmupClassifier / warmupTokenizer already swallow internally; this
      // is a final safety net so an unexpected throw never bubbles to an
      // unhandledRejection.
      console.error(`[verity] warmup error (non-fatal):`, err);
    }
  });
});
