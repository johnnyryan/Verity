/**
 * Parallel cross-family second-opinion consult with analysis pass.
 *
 * Sibling to the /verify pipeline. Whereas /verify is post-answer auditing
 * (the worker has already answered, the critics check the answer), /second
 * is pre-final-answer consultation: the worker asks a different-family model
 * on the secondary GPU for its own independent answer to the same question.
 *
 * Design goals:
 *   - Different model family from the worker to add error-mode diversity.
 *     Default AMD leg is granite3.2:8b (IBM); default NVIDIA leg is the
 *     worker model (Qwen/Alibaba). The cross-family contrast is the point.
 *   - Both GPUs generate in parallel; wall-clock cost ≈ max(amd, nvidia).
 *   - Then a third analysis call on NVIDIA compares the two answers and
 *     emits a structured {agreements, disputes, table_html} object. In
 *     auto mode the analysis also synthesises a final_answer.
 *   - The worker sees both texts + the analysis and decides how to merge.
 *     No reconciliation logic in here beyond the LLM-judged analysis.
 *
 * 2026-04-21 — Phase C extension. Phase A+B (dual dispatch + cheap
 * dispute heuristic) was resurrected from the 2026-04-20 archive. Phase
 * C adds the structured analysis pass.
 *
 * Environment variables (all have sane defaults; see src/config.ts):
 *   CONSULT_DUAL                           "0" reverts to single-Ollama path.
 *   SECOND_OPINION_PRIMARY_ENDPOINT        NVIDIA leg endpoint (default LM Studio).
 *   SECOND_OPINION_PRIMARY_MODEL           NVIDIA leg model (default WORKER_MODEL).
 *   SECOND_OPINION_PRIMARY_API_KEY         NVIDIA leg API key (default "lm-studio").
 *   SECOND_OPINION_CONCURRENT_TIMEOUT_MS   Per-leg dual-dispatch timeout (20 s).
 *   SECOND_OPINION_ANALYSIS_MODEL          Phase C analysis model (default WORKER_MODEL).
 *   SECOND_OPINION_ANALYSIS_TIMEOUT_MS     Analysis-call timeout (15 s).
 *   SECOND_OPINION_ANALYSIS_MAX_TOKENS     Analysis-JSON token cap (800).
 *
 * Wire-format contract (preserved across the 2026-04-21 rewrite):
 *   Legacy: second_opinion, model, latency_ms, diff_summary, unavailable,
 *           error, notes, dual_opinion (from Phase A), disputes (from Phase B).
 *   Added:  analysis, resolution_mode.
 */

import {
  OLLAMA_URL,
  VERBOSE_LOGGING,
  SECOND_OPINION_MODEL,
  SECOND_OPINION_MAX_TOKENS,
  SECOND_OPINION_TEMPERATURE,
  SECOND_OPINION_TIMEOUT_MS,
  SECOND_OPINION_DUAL_ENABLED,
  SECOND_OPINION_PRIMARY_ENDPOINT,
  SECOND_OPINION_PRIMARY_MODEL,
  SECOND_OPINION_PRIMARY_API_KEY,
  SECOND_OPINION_CONCURRENT_TIMEOUT_MS,
  SECOND_OPINION_ANALYSIS_MODEL,
  SECOND_OPINION_ANALYSIS_TIMEOUT_MS,
  SECOND_OPINION_ANALYSIS_MAX_TOKENS,
} from "../config.js";
import { findBalancedJsonObject } from "../critics/parse.js";
import { getLlmClient } from "../llm/client.js";
import { stripReasoningTraces } from "../sanitize.js";

const ANSWERER_SYSTEM_PROMPT = `
You are an independent answerer providing a second opinion. A different model
(the "primary") has been asked the same question. Your answer will be shown
to the primary so it can compare and reconcile.

Answer the user's question briefly, directly, and with a focus on correctness.

Rules:
  - Do not acknowledge or mention the primary model. Do not compare.
  - If you are unsure, say so — hedging is better than confident error.
  - If a claim needs evidence, flag it; do not fabricate citations.
  - Keep your answer short (2–6 sentences by default) unless the question
    clearly warrants more.
  - Output plain text, not JSON, not markdown fences.
`.trim();

export type ResolutionMode = "manual" | "auto";

export interface ConsultInput {
  question: string;
  worker_draft?: string;
  prior_context?: string;
  /** Override default model; must be a tag available on the Ollama server. */
  model?: string;
  /**
   * "manual" (default) returns both answers + analysis so the worker can
   * show both to the user and let them decide. "auto" additionally asks
   * the analysis pass to synthesise a final_answer the worker can present
   * directly.
   */
  resolution_mode?: ResolutionMode;
}

/**
 * Single divergence between the two dual /second answers (cheap heuristic).
 * Diagnostic-only; coexists with the richer LLM-judged `analysis` object.
 */
export interface DisputePoint {
  /**
   * "polarity-mismatch" — the first sentence of each answer leans in
   *   opposing polarity (agree/disagree, yes/no, correct/incorrect, etc.).
   * "no-overlap" — token-overlap Jaccard between the two answers was
   *   below the agreement floor so we couldn't align them sentence-wise.
   */
  kind: "polarity-mismatch" | "no-overlap";
  excerpt_a: string;
  excerpt_b: string;
}

/**
 * Additive dual-path payload. Present only when DUAL mode fires and
 * `input.model` was not supplied.
 */
export interface DualOpinion {
  model: string;
  endpoint: string;
  second_opinion: string;
  latency_ms: number;
  unavailable?: boolean;
  error?: string;
}

/**
 * Structured comparison of the two answers produced by the Phase C
 * analysis pass. Always LLM-judged; present only when both legs of the
 * dual dispatch succeeded.
 */
export interface AnalysisDispute {
  topic: string;
  amd_position: string;
  nvidia_position: string;
}

export interface AnalysisResult {
  agreements: string[];
  disputes: AnalysisDispute[];
  /** Pre-rendered HTML table of disputes (inline styles; renders in LM Studio chat). */
  table_html: string;
  /**
   * 2026-04-21 additive: same content as table_html, rendered as a Markdown
   * pipe table. LM Studio's chat renderer is more reliable with Markdown
   * than with inline-styled HTML on some versions — workers may prefer to
   * render this one verbatim if HTML passthrough isn't styled. Always
   * populated when table_html is; identical fallback strings for the
   * no-disputes / unavailable cases.
   */
  table_md: string;
  /** Only populated when resolution_mode === "auto". */
  final_answer?: string;
  /** Which model produced this analysis. */
  model: string;
  latency_ms: number;
  unavailable?: boolean;
  error?: string;
}

export interface ConsultOutput {
  /** The AMD/Ollama model's answer (legacy field; back-compat with callers). */
  second_opinion: string;
  /** The AMD/Ollama model tag (legacy field). */
  model: string;
  /** Latency for the AMD/Ollama leg (legacy field). */
  latency_ms: number;
  /** One-line heuristic: "agrees", "disagrees", "unsure", etc. (legacy). */
  diff_summary: string;
  /** Optional: raw concern if the second opinion flags something. */
  notes?: string[];
  /** True if the AMD leg failed/timed out; `second_opinion` becomes a reason string. */
  unavailable?: boolean;
  error?: string;
  /**
   * 2026-04-20 additive: NVIDIA/LM-Studio leg. Absent when dual is off or
   * when `input.model` was supplied.
   */
  dual_opinion?: DualOpinion;
  /**
   * 2026-04-20 additive: cheap-heuristic differences between the two dual
   * answers. Always `[]` when dual is absent or when the two answers are
   * materially aligned. Never affects any consensus or verdict.
   */
  disputes: DisputePoint[];
  /**
   * 2026-04-21 additive: structured analysis produced by Phase C. Present
   * only when both dual legs succeeded; absent when dual is off, when
   * `input.model` was supplied (legacy path), or when either leg was
   * unavailable.
   */
  analysis?: AnalysisResult;
  /**
   * 2026-04-21 additive: the mode this call ran in. Echoed so the worker
   * can branch on it (`auto` → present `analysis.final_answer`; `manual`
   * → show both answers and let the user decide).
   */
  resolution_mode?: ResolutionMode;
}

function buildUserMessage(input: ConsultInput): string {
  const parts: string[] = [];
  if (input.prior_context && input.prior_context.trim().length > 0) {
    parts.push(
      "### PRIOR CONTEXT (reference material the question depends on)\n" +
        input.prior_context.trim()
    );
  }
  parts.push("### USER QUESTION\n" + input.question.trim());
  if (input.worker_draft && input.worker_draft.trim().length > 0) {
    parts.push(
      "### (For your information — the primary model's draft answer; " +
        "do not mirror it, produce your own independent answer)\n" +
        input.worker_draft.trim()
    );
  }
  parts.push(
    "### YOUR TASK\nAnswer the user question independently. Plain text only."
  );
  return parts.join("\n\n");
}

/**
 * Very cheap disagreement heuristic. Not meant to be authoritative — the
 * Phase C analysis pass does the real LLM-judged comparison — but gives
 * a one-line flavour for the tool result even when dual is off.
 */
function computeDiffSummary(workerDraft: string | undefined, secondOpinion: string): string {
  if (!workerDraft || workerDraft.trim().length === 0) {
    return "no draft supplied; second opinion is standalone";
  }
  const norm = (s: string) =>
    s
      .toLowerCase()
      .replace(/[^a-z0-9 ]+/g, " ")
      .split(/\s+/)
      .filter((t) => t.length > 3);
  const a = new Set(norm(workerDraft));
  const b = new Set(norm(secondOpinion));
  if (a.size === 0 || b.size === 0) {
    return "agreement indeterminate (short texts)";
  }
  let intersect = 0;
  for (const t of a) if (b.has(t)) intersect += 1;
  const jaccard = intersect / (a.size + b.size - intersect);

  const negations = /\b(no|not|never|incorrect|wrong|false|disagree)\b/i;
  const hedges = /\b(unsure|uncertain|not sure|may|might|could be)\b/i;

  if (jaccard >= 0.45) return `agrees with draft (token overlap ${jaccard.toFixed(2)})`;
  if (negations.test(secondOpinion))
    return `potential disagreement (contains negation, overlap ${jaccard.toFixed(2)})`;
  if (hedges.test(secondOpinion))
    return `second opinion is hedged (overlap ${jaccard.toFixed(2)})`;
  if (jaccard < 0.15)
    return `low overlap with draft (${jaccard.toFixed(2)}); review differences`;
  return `partial overlap (${jaccard.toFixed(2)}); compare carefully`;
}

// ─────────────────────────────────────────────────────────────────────────
// Disputes (Phase B — cheap heuristic, coexists with Phase C analysis)
// ─────────────────────────────────────────────────────────────────────────

const AGREEMENT_FLOOR = 0.40;

const POSITIVE_POLARITY = [
  "agree",
  "correct",
  "yes",
  "pass",
  "right",
  "accurate",
  "true",
  "confirmed",
  "supported",
];
const NEGATIVE_POLARITY = [
  "disagree",
  "incorrect",
  "no",
  "fail",
  "wrong",
  "inaccurate",
  "false",
  "refuted",
  "unsupported",
];

function firstSentence(text: string): string {
  const trimmed = text.trim();
  if (!trimmed) return "";
  const m = trimmed.match(/^[\s\S]*?[.?!](?=\s|$)/);
  return (m ? m[0] : trimmed).trim();
}

function hasAnyWord(text: string, words: readonly string[]): boolean {
  const lower = text.toLowerCase();
  for (const w of words) {
    const re = new RegExp(`\\b${w}\\b`, "i");
    if (re.test(lower)) return true;
  }
  return false;
}

function tokenize(text: string): Set<string> {
  return new Set(
    text
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
  const union = a.size + b.size - intersect;
  return union > 0 ? intersect / union : 0;
}

export function computeDisputes(
  answerA: string,
  answerB: string
): DisputePoint[] {
  const trimmedA = answerA.trim();
  const trimmedB = answerB.trim();
  if (!trimmedA || !trimmedB) return [];

  const sentA = firstSentence(trimmedA);
  const sentB = firstSentence(trimmedB);

  const aHasPos = hasAnyWord(sentA, POSITIVE_POLARITY);
  const aHasNeg = hasAnyWord(sentA, NEGATIVE_POLARITY);
  const bHasPos = hasAnyWord(sentB, POSITIVE_POLARITY);
  const bHasNeg = hasAnyWord(sentB, NEGATIVE_POLARITY);

  const aPolarity = aHasPos && !aHasNeg ? "pos" : aHasNeg && !aHasPos ? "neg" : "";
  const bPolarity = bHasPos && !bHasNeg ? "pos" : bHasNeg && !bHasPos ? "neg" : "";
  if (aPolarity && bPolarity && aPolarity !== bPolarity) {
    return [
      {
        kind: "polarity-mismatch",
        excerpt_a: sentA,
        excerpt_b: sentB,
      },
    ];
  }

  const overlap = jaccard(tokenize(trimmedA), tokenize(trimmedB));
  if (overlap < AGREEMENT_FLOOR) {
    return [
      {
        kind: "no-overlap",
        excerpt_a: sentA || trimmedA.slice(0, 160),
        excerpt_b: sentB || trimmedB.slice(0, 160),
      },
    ];
  }

  return [];
}

// ─────────────────────────────────────────────────────────────────────────
// Back-pressure semaphore (the per-endpoint client cache moved to
// ../llm/client.ts so all five LLM call-sites share a single allocator.)
// ─────────────────────────────────────────────────────────────────────────

const endpointQueue = new Map<string, Promise<unknown>>();

function withEndpointLock<T>(
  endpoint: string,
  fn: () => Promise<T>
): Promise<T> {
  const prior = endpointQueue.get(endpoint) ?? Promise.resolve();
  const next = prior.catch(() => {}).then(fn);
  endpointQueue.set(
    endpoint,
    next.catch(() => {})
  );
  return next;
}

export async function callOneBackend(
  endpoint: string,
  apiKey: string,
  model: string,
  input: ConsultInput,
  timeoutMs: number
): Promise<Omit<ConsultOutput, "disputes" | "dual_opinion" | "analysis" | "resolution_mode">> {
  const start = Date.now();
  const abort = new AbortController();
  const timer = setTimeout(() => abort.abort(), timeoutMs);
  try {
    const response = await getLlmClient({ endpoint, apiKey }).chat.completions.create(
      {
        model,
        messages: [
          { role: "system", content: ANSWERER_SYSTEM_PROMPT },
          { role: "user", content: buildUserMessage(input) },
        ],
        temperature: SECOND_OPINION_TEMPERATURE,
        max_tokens: SECOND_OPINION_MAX_TOKENS,
      },
      { signal: abort.signal }
    );

    const raw = response.choices[0]?.message?.content ?? "";
    const cleaned = stripReasoningTraces(raw).trim();

    if (VERBOSE_LOGGING) {
      console.error(
        `[second-opinion:${model}@${endpoint}] ${Date.now() - start}ms ${cleaned.length}chars`
      );
    }

    return {
      second_opinion: cleaned,
      model,
      latency_ms: Date.now() - start,
      diff_summary: computeDiffSummary(input.worker_draft, cleaned),
    };
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    if (VERBOSE_LOGGING) {
      console.error(`[second-opinion:${model}@${endpoint}] error:`, msg);
    }
    return {
      second_opinion: `(unavailable: ${msg})`,
      model,
      latency_ms: Date.now() - start,
      diff_summary: "second-opinion model unavailable",
      unavailable: true,
      error: msg,
    };
  } finally {
    clearTimeout(timer);
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Phase C — LLM-judged analysis pass
// ─────────────────────────────────────────────────────────────────────────
//
// Once both dual legs return, a third call on NVIDIA compares them.
// The model is asked to emit STRICT JSON with {agreements, disputes,
// [final_answer]}. We then render the disputes as a markdown table
// (pipe-char-escaped, <br> for newlines inside cells).
//
// Parsing is tolerant:
//   - <think>/<reasoning> traces are stripped.
//   - Markdown fences (```json ... ```) are stripped.
//   - Prose preamble ("Thinking Process: ...") is skipped; we find the
//     first `{` and parse from there.
//   - On any parse failure: analysis.unavailable = true, table_html falls
//     back to a fixed string so the worker always has something to show.

const ANALYSIS_SYSTEM_PROMPT = `
You are analyzing two independent answers to the same question to identify
agreements and disputes between them. Output STRICT JSON ONLY. No markdown
fences, no preamble, no "Thinking Process:" narration. JSON object only.

Rules:
  - Be precise about WHAT the two answers disagree on, not how.
  - If the answers largely agree, return an empty disputes array and 2–4
    agreements.
  - Plain text within JSON fields. No markdown inside strings. No newlines
    inside strings.
  - Prefer 0–3 disputes; never more than 5.
  - amd_position and nvidia_position must each be ONE short sentence.
`.trim();

function analysisUserPrompt(args: {
  question: string;
  amdAnswer: string;
  nvidiaAnswer: string;
  primaryModelName: string;
  mode: ResolutionMode;
}): string {
  const { question, amdAnswer, nvidiaAnswer, primaryModelName, mode } = args;

  const schemaLines = [
    "Schema:",
    "{",
    `  "agreements": [string],`,
    `  "disputes": [`,
    `    { "topic": string, "amd_position": string, "nvidia_position": string }`,
    `  ]${mode === "auto" ? "," : ""}`,
  ];
  if (mode === "auto") {
    schemaLines.push(
      `  "final_answer": string   // the most defensible answer synthesised from both`
    );
  }
  schemaLines.push("}");

  const closing =
    mode === "auto"
      ? "Produce the analysis JSON AND include a final_answer field synthesised from both answers."
      : "Produce the analysis JSON.";

  return [
    schemaLines.join("\n"),
    "",
    `Question: ${question.trim()}`,
    "",
    "AMD answer (Granite 3.2 8B):",
    amdAnswer.trim(),
    "",
    `NVIDIA answer (${primaryModelName}):`,
    nvidiaAnswer.trim(),
    "",
    closing,
  ].join("\n");
}

/**
 * HTML-entity-escape a cell value: `&<>"'` escaped, newlines become `<br>`
 * so multi-line content stays on one table row. Inline styles used on the
 * table itself (NOT here) so LM Studio's markdown renderer passes the HTML
 * through with styling intact.
 */
function escapeHtml(value: string): string {
  return (value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;")
    .replace(/\r?\n/g, "<br>")
    .trim();
}

export function renderDisputesTable(
  disputes: AnalysisDispute[],
  primaryModelName: string
): string {
  if (!Array.isArray(disputes) || disputes.length === 0) {
    return "<p><em>No disputes — models agreed.</em></p>";
  }
  const th =
    "padding:6px 10px;text-align:left;border:1px solid #888;";
  const td =
    "padding:6px 10px;border:1px solid #888;vertical-align:top;";
  const tableStyle =
    "border-collapse:collapse;margin:0.5em 0;font-size:0.95em;border:1px solid #888;";
  const headerStyle = "background:#c62828;color:#fff;";
  const header =
    `<thead><tr style="${headerStyle}">` +
    `<th style="${th}">Topic</th>` +
    `<th style="${th}">AMD (Granite 3.2 8B)</th>` +
    `<th style="${th}">NVIDIA (${escapeHtml(primaryModelName)})</th>` +
    `</tr></thead>`;
  const body =
    "<tbody>" +
    disputes
      .map(
        (d) =>
          `<tr>` +
          `<td style="${td}"><strong>${escapeHtml(d.topic)}</strong></td>` +
          `<td style="${td}">${escapeHtml(d.amd_position)}</td>` +
          `<td style="${td}">${escapeHtml(d.nvidia_position)}</td>` +
          `</tr>`
      )
      .join("") +
    "</tbody>";
  return `<table style="${tableStyle}">${header}${body}</table>`;
}

/**
 * Markdown-pipe-table version of the disputes table. Same semantics as
 * renderDisputesTable but safe to render verbatim in any chat UI that
 * handles Markdown (LM Studio's default renderer, GitHub, Slack blocks,
 * etc.). Pipes within cells are escaped to `\|`; newlines become `<br>`
 * (Markdown doesn't support literal newlines mid-cell; `<br>` is the
 * conventional workaround and LM Studio honours it).
 *
 * Fallback strings deliberately match the HTML variant in spirit so a
 * worker can render either one without conditional logic.
 */
export function renderDisputesMarkdown(
  disputes: AnalysisDispute[],
  primaryModelName: string
): string {
  if (!Array.isArray(disputes) || disputes.length === 0) {
    return "_No disputes — models agreed._";
  }
  const escapeCell = (value: string): string =>
    (value ?? "")
      .replace(/\\/g, "\\\\")
      .replace(/\|/g, "\\|")
      .replace(/\r?\n/g, "<br>")
      .trim();
  const rows: string[] = [];
  rows.push(`| Topic | AMD (Granite 3.2 8B) | NVIDIA (${escapeCell(primaryModelName)}) |`);
  rows.push(`| --- | --- | --- |`);
  for (const d of disputes) {
    rows.push(
      `| **${escapeCell(d.topic)}** | ${escapeCell(d.amd_position)} | ${escapeCell(d.nvidia_position)} |`
    );
  }
  return rows.join("\n");
}

/**
 * Tolerant JSON extractor. Strips reasoning traces, then strips markdown
 * fences, then locates the first balanced `{…}` and parses it.
 * Returns null on any failure — the caller handles the unavailable case.
 *
 * 2026-04-21 hardening: qwen3.5-9b emits large <think>…</think> blocks
 * before JSON. If the model truncates mid-think (max_tokens hit), the
 * closing tag is missing and stripReasoningTraces — which requires both
 * tags — leaves the block intact. The draft JSON inside `<think>` then
 * leaks into parsing and fails. We now strip any residual opening
 * `<think>`/`<thinking>`/`<reasoning>` tag (and everything that follows
 * up to the next closing tag OR end-of-string) as a belt-and-braces
 * second pass. Same treatment for the plain-text `Thinking Process:`
 * preamble that some variants emit without tags.
 */
export function extractAnalysisJson(raw: string): unknown | null {
  if (!raw) return null;
  let s = stripReasoningTraces(raw).trim();

  // Second pass: drop any unclosed <think>/<thinking>/<reasoning> opener.
  // Matches `<think>…` with NO closing tag; we strip from the opener to end.
  // If a closing tag appears later in the string (rare but possible — model
  // emits a second <think> that DOES close), stripReasoningTraces already
  // handled the first paired block; this catches the unpaired tail.
  s = s.replace(/<(?:think|thinking|reasoning)>[\s\S]*$/i, "").trim();

  // Plain-text preamble some models emit without tags ("Thinking Process:"
  // followed by prose then the JSON object). The balanced-brace scanner
  // below tolerates `{` inside the preamble, but a leading paragraph
  // would still slow it down — strip it explicitly.
  s = s.replace(/^(?:thinking\s+process\s*:[\s\S]*?)(?=\{|$)/i, "").trim();

  // Strip ```json ... ``` or ``` ... ``` fences.
  s = s.replace(/^```(?:json)?\s*/i, "").replace(/```$/i, "").trim();

  // Use the shared balanced-brace finder, preferring objects that contain
  // an `agreements` key (the canonical top-level field for this schema).
  // The previous indexOf/lastIndexOf span could grab the wrong text if the
  // model emitted prose containing `{` characters before the JSON object.
  const candidate = findBalancedJsonObject(s, { requireKey: "agreements" });
  if (!candidate) return null;
  try {
    return JSON.parse(candidate);
  } catch {
    return null;
  }
}

function isAnalysisDispute(v: unknown): v is AnalysisDispute {
  if (!v || typeof v !== "object") return false;
  const o = v as Record<string, unknown>;
  return (
    typeof o.topic === "string" &&
    typeof o.amd_position === "string" &&
    typeof o.nvidia_position === "string"
  );
}

/**
 * Run the Phase C analysis pass. Swallows all errors — the worker always
 * gets an AnalysisResult with at minimum a table_html fallback string.
 */
export async function runAnalysisPass(args: {
  question: string;
  amdAnswer: string;
  nvidiaAnswer: string;
  primaryModelName: string;
  mode: ResolutionMode;
  endpoint?: string;
  apiKey?: string;
  model?: string;
  timeoutMs?: number;
  maxTokens?: number;
}): Promise<AnalysisResult> {
  const {
    question,
    amdAnswer,
    nvidiaAnswer,
    primaryModelName,
    mode,
  } = args;
  const endpoint = args.endpoint ?? SECOND_OPINION_PRIMARY_ENDPOINT;
  const apiKey = args.apiKey ?? SECOND_OPINION_PRIMARY_API_KEY;
  const model = args.model ?? SECOND_OPINION_ANALYSIS_MODEL;
  const timeoutMs = args.timeoutMs ?? SECOND_OPINION_ANALYSIS_TIMEOUT_MS;
  const maxTokens = args.maxTokens ?? SECOND_OPINION_ANALYSIS_MAX_TOKENS;

  const start = Date.now();
  const unavailable = (error: string): AnalysisResult => ({
    agreements: [],
    disputes: [],
    table_html: "<p><em>analysis unavailable (parse failure)</em></p>",
    table_md: "_analysis unavailable (parse failure)_",
    model,
    latency_ms: Date.now() - start,
    unavailable: true,
    error,
  });

  const abort = new AbortController();
  const timer = setTimeout(() => abort.abort(), timeoutMs);
  try {
    const response = await getLlmClient({ endpoint, apiKey }).chat.completions.create(
      {
        model,
        messages: [
          { role: "system", content: ANALYSIS_SYSTEM_PROMPT },
          {
            role: "user",
            content: analysisUserPrompt({
              question,
              amdAnswer,
              nvidiaAnswer,
              primaryModelName,
              mode,
            }),
          },
        ],
        temperature: 0.1,
        max_tokens: maxTokens,
      },
      { signal: abort.signal }
    );

    const raw = response.choices[0]?.message?.content ?? "";
    const parsed = extractAnalysisJson(raw);
    if (!parsed || typeof parsed !== "object") {
      return {
        ...unavailable("analysis JSON parse failure"),
      };
    }

    const obj = parsed as Record<string, unknown>;
    const agreements = Array.isArray(obj.agreements)
      ? obj.agreements.filter((x): x is string => typeof x === "string")
      : [];
    const disputesRaw = Array.isArray(obj.disputes) ? obj.disputes : [];
    const disputes = disputesRaw.filter(isAnalysisDispute).slice(0, 5);
    const final_answer =
      mode === "auto" && typeof obj.final_answer === "string" && obj.final_answer.trim().length > 0
        ? obj.final_answer.trim()
        : undefined;

    const table_html = renderDisputesTable(disputes, primaryModelName);
    const table_md = renderDisputesMarkdown(disputes, primaryModelName);

    if (VERBOSE_LOGGING) {
      console.error(
        `[second-opinion:analysis:${model}@${endpoint}] ${Date.now() - start}ms ` +
          `agreements=${agreements.length} disputes=${disputes.length}` +
          (mode === "auto" ? ` final_answer=${final_answer ? "yes" : "no"}` : "")
      );
    }

    const result: AnalysisResult = {
      agreements,
      disputes,
      table_html,
      table_md,
      model,
      latency_ms: Date.now() - start,
    };
    if (final_answer !== undefined) result.final_answer = final_answer;
    return result;
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    if (VERBOSE_LOGGING) {
      console.error(`[second-opinion:analysis:${model}@${endpoint}] error:`, msg);
    }
    return unavailable(msg);
  } finally {
    // Clear the abort timer on every exit path. Previously only cleared
    // on success; an early throw left an orphaned timer that fired on a
    // dead AbortController (M2).
    clearTimeout(timer);
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────

export async function runSecondOpinion(
  input: ConsultInput
): Promise<ConsultOutput> {
  const resolution_mode: ResolutionMode = input.resolution_mode ?? "manual";

  // Legacy path: explicit model override OR dual disabled via env.
  // Preserves pre-2026-04-20 wire shape: no dual_opinion, no analysis,
  // disputes: [].
  if (input.model || !SECOND_OPINION_DUAL_ENABLED) {
    const model = input.model ?? SECOND_OPINION_MODEL;
    const single = await withEndpointLock(OLLAMA_URL, () =>
      callOneBackend(OLLAMA_URL, "ollama", model, input, SECOND_OPINION_TIMEOUT_MS)
    );
    return {
      ...single,
      disputes: [],
    };
  }

  // Dual path: fire both GPUs in parallel.
  const ollamaPromise = withEndpointLock(OLLAMA_URL, () =>
    callOneBackend(
      OLLAMA_URL,
      "ollama",
      SECOND_OPINION_MODEL,
      input,
      SECOND_OPINION_CONCURRENT_TIMEOUT_MS
    )
  );
  const primaryPromise = withEndpointLock(SECOND_OPINION_PRIMARY_ENDPOINT, () =>
    callOneBackend(
      SECOND_OPINION_PRIMARY_ENDPOINT,
      SECOND_OPINION_PRIMARY_API_KEY,
      SECOND_OPINION_PRIMARY_MODEL,
      input,
      SECOND_OPINION_CONCURRENT_TIMEOUT_MS
    )
  );

  const [ollamaSettled, primarySettled] = await Promise.allSettled([
    ollamaPromise,
    primaryPromise,
  ]);

  const ollamaResult =
    ollamaSettled.status === "fulfilled"
      ? ollamaSettled.value
      : {
          second_opinion: `(unavailable: ${String(ollamaSettled.reason)})`,
          model: SECOND_OPINION_MODEL,
          latency_ms: 0,
          diff_summary: "second-opinion model unavailable",
          unavailable: true,
          error: String(ollamaSettled.reason),
        };
  const primaryResult =
    primarySettled.status === "fulfilled"
      ? primarySettled.value
      : {
          second_opinion: `(unavailable: ${String(primarySettled.reason)})`,
          model: SECOND_OPINION_PRIMARY_MODEL,
          latency_ms: 0,
          diff_summary: "second-opinion model unavailable",
          unavailable: true,
          error: String(primarySettled.reason),
        };

  const dual: DualOpinion = {
    model: primaryResult.model,
    endpoint: SECOND_OPINION_PRIMARY_ENDPOINT,
    second_opinion: primaryResult.second_opinion,
    latency_ms: primaryResult.latency_ms,
    unavailable: primaryResult.unavailable,
    error: primaryResult.error,
  };

  // Phase B cheap heuristic — runs when both legs succeeded.
  const disputes =
    !ollamaResult.unavailable && !primaryResult.unavailable
      ? computeDisputes(ollamaResult.second_opinion, primaryResult.second_opinion)
      : [];

  // Phase C — LLM-judged analysis. Only when both legs succeeded.
  let analysis: AnalysisResult | undefined;
  if (!ollamaResult.unavailable && !primaryResult.unavailable) {
    analysis = await withEndpointLock(SECOND_OPINION_PRIMARY_ENDPOINT, () =>
      runAnalysisPass({
        question: input.question,
        amdAnswer: ollamaResult.second_opinion,
        nvidiaAnswer: primaryResult.second_opinion,
        primaryModelName: primaryResult.model,
        mode: resolution_mode,
      })
    );
  }

  const out: ConsultOutput = {
    ...ollamaResult,
    dual_opinion: dual,
    disputes,
    resolution_mode,
  };
  if (analysis) out.analysis = analysis;
  return out;
}
