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
  NLI_IMPL,
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
        "Audits an answer with critic models + NLI claim-checking. The " +
        "answer to audit is the `answer` parameter you pass to this tool " +
        "— typically the prose you JUST composed in the current turn, " +
        "NOT something from prior chat history. Returns a ready-to-paste " +
        "Markdown block containing the answer (echoed back), the critic " +
        "verdicts table, findings, and a follow-up prompt.\n" +
        "\n" +
        "WHEN TO CALL: the user typed '/verify', '/verifydeep', or " +
        "'/verifydeeper' anywhere in their latest message. /verify is a " +
        "TOOL TRIGGER — it means call this tool, not 'write a verified " +
        "answer in prose'. Map: /verify→mode='standard', /verifydeep→" +
        "'deep', /verifydeeper→'deeper'.\n" +
        "\n" +
        "ALSO CALL when your prior turn pasted a Verity block that " +
        "ended with the 'Awaiting your reply' prompt AND the user " +
        "replied with '/verifydeeper', 'deeper', 'yes', 'OK', 'sure', " +
        "'go ahead' — that affirmative means mode='deeper' on the same " +
        "answer.\n" +
        "\n" +
        "If the user replies 'redraft', rewrite the answer to address " +
        "the findings, then call this tool again on the rewritten " +
        "answer. The same STRICT SOURCING CONTRACT applies to the " +
        "redraft: every URL in the redraft MUST be fetched first via " +
        "the fetch tool to confirm it resolves. Do NOT 'address' a " +
        "'needs more sources' finding by adding plausible-looking " +
        "URLs you haven't fetched — that is fabrication and makes the " +
        "redraft worse than the original. If you can't find a working " +
        "source for a claim, drop the claim from the redraft.\n" +
        "\n" +
        "Reply 'no' means do nothing.\n" +
        "\n" +
        "FLOW for an inline '/verify' trigger:\n" +
        "  1. Write a substantive prose answer (a paragraph or more) " +
        "to the user's question. For non-trivial factual claims, use " +
        "the fetch tool to confirm 3-7 working source URLs, then cite " +
        "inline as [N], [author], [publisher], [year], [page], [url]. " +
        "Drop facts you can't source; do not invent URLs.\n" +
        "  2. Call verify_answer with question + answer (the " +
        "full prose from step 1, not a summary or placeholder).\n" +
        "  3. The tool returns a Markdown block. PASTE THE ENTIRE BLOCK " +
        "VERBATIM into your chat reply. LM Studio collapses tool-call " +
        "results by default, so the user only sees what is in your " +
        "message body. The block itself starts with the answer echoed " +
        "back, then the table, then findings. The block also tells you " +
        "not to redraft — follow that.\n" +
        "  4. Stop. The block ends with a yes/no follow-up prompt for " +
        "the user; wait for their reply, do not redraft on your own.\n" +
        "\n" +
        "DO NOT: skip the tool call (writing prose with citations is " +
        "NOT a substitute); paraphrase the block (paste verbatim); " +
        "redraft the answer based on findings (the block already asks " +
        "the user about redrafting); invent the table from your own " +
        "reasoning (the table comes from the tool, not from you).",
      // Older verbose description was 18k chars / 297 lines (2026-05-11
      // afternoon). That length fragmented Qwen 3.5 9B's attention and
      // produced unreliable behaviour. Trimmed to ~2.5k chars / 40
      // lines (v4 evening). The agent-preface inside the rendered
      // block carries the "do not redraft / paste verbatim" rules
      // co-located with the data.
      inputSchema: {
        // 2026-05-11 v3 simplification: reduced from 7 params to 4 to
        // make tool calls more reliable. Smaller models (Qwen 3.5 9B)
        // were intermittently dropping the "parameter=" prefix on one
        // param when the schema had many fields — observed failure:
        //   <prior_context> none </parameter>
        // (missing the parameter= prefix). Fewer params = fewer chances
        // for the generator to glitch.
        //
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

    // Derive the previously-exposed schema params from sensible defaults
    // (see inputSchema comment for rationale): task_type=auto always,
    // use_nli=true always, context_mode is derived from prior_context.
    // We still honour them if a caller passes them via the unstable
    // back-channel (older clients, direct HTTP testers).
    const argsRecord = args as Record<string, unknown>;
    const hasPriorContext =
      typeof args.prior_context === "string" && args.prior_context.trim().length > 0;
    const result = await runVerificationPipeline({
      question: args.question,
      answer: args.answer,
      mode: args.mode ?? "standard",
      task_type: (argsRecord.task_type as TaskType) ?? "auto",
      context_mode:
        (argsRecord.context_mode as ContextMode) ??
        (hasPriorContext ? "with_context" : "minimal"),
      prior_context: hasPriorContext ? args.prior_context : undefined,
      use_nli: (argsRecord.use_nli as boolean | undefined) ?? true,
    });

    // Send the pre-rendered Markdown summary as the primary tool-response
    // text. Smaller worker models tend to paste tool output verbatim into
    // chat; returning the JSON-stringified VerifyOutput made them dump
    // raw JSON. Returning summary_md means whatever the worker pastes is
    // already a clean Markdown table -- no extraction step required.
    return {
      content: [
        { type: "text" as const, text: result.summary_md || JSON.stringify(result, null, 2) },
      ],
    };
  }

  if (toolName === "consult_second_opinion") {
    const args = (request.params.arguments ?? {}) as Partial<ConsultInput>;
    if (typeof args.question !== "string") {
      throw new Error("consult_second_opinion requires a 'question' string");
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
app.use(express.json({ limit: "50mb" }));

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
// [ADAPT] If LM Studio uses a different session header key, adjust here.
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

app.listen(SERVER_PORT, () => {
  console.error(
    `[verity] MCP server v${SERVER_VERSION} listening on http://localhost:${SERVER_PORT}/mcp`
  );
  console.error(`[verity] health check at http://localhost:${SERVER_PORT}/health`);

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
