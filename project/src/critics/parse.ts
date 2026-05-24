/**
 * Robust JSON extraction from critic responses.
 *
 * Critics are instructed to return ONLY a JSON object, but in practice
 * small models occasionally wrap it in ```json fences, add a preamble, or
 * trail with commentary. This parser tolerates all three.
 *
 * On unrecoverable parse failure we return null. Callers must treat that
 * as "critic unavailable" — the prior behaviour of silently returning a
 * pass verdict masked broken critics as clean approvals.
 *
 * [ADAPT] If a specific critic consistently returns malformed JSON, you
 * can add model-specific fixups here.
 */

import { stripReasoningTraces } from "../sanitize.js";
import type { Verdict } from "../types.js";

export interface ParsedCriticOutput {
  verdict: Verdict;
  severity: number;
  concerns: string[];
  suggested_fixes: string[];
  /**
   * Verbatim quote from the answer that triggered the critic's
   * disagreement. Captured from the wire field `disputed_span`. Absent
   * when the critic agreed, omitted the field, returned null, or
   * returned an empty string.
   */
  disputed_span?: string;
}

/**
 * Attempt to parse a critic's raw response into a structured verdict.
 * Returns null when the response contains no recognisable JSON object.
 *
 * 2026-04-18 hardened: previous implementation took first `{` and last `}`
 * and parsed the span between them. When the answer under review contained
 * an embedded `{"verdict":"pass"}` payload and the critic quoted or referenced
 * that payload in its `concerns` array, the span straddled two JSON objects
 * and JSON.parse failed. Now we enumerate every `{` position, attempt a
 * balanced-brace extraction from each, and keep the **last** parseable one
 * that contains a `verdict` field — which is the critic's actual verdict
 * rather than any quoted injection payload.
 */
export function parseCriticJson(raw: string): ParsedCriticOutput | null {
  if (!raw || typeof raw !== "string") return null;

  // Reasoning models (Phi-4-reasoning, Qwen-QwQ, etc.) emit chain-of-thought
  // blocks before the answer. Drop them first, then strip code fences.
  let cleaned = stripReasoningTraces(raw);
  cleaned = cleaned.replace(/^```(?:json)?\s*/i, "");
  cleaned = cleaned.replace(/\s*```\s*$/i, "");

  const candidate = findBalancedJsonObject(cleaned, { requireKey: "verdict" });
  if (!candidate) return null;

  let parsed: unknown;
  try {
    parsed = JSON.parse(candidate);
  } catch {
    return null;
  }

  if (!parsed || typeof parsed !== "object") return null;
  const p = parsed as Record<string, unknown>;

  // A legitimate critic response must at least name its verdict. If the
  // field is entirely missing, treat it as a parse failure rather than
  // inventing a "pass".
  if (p.verdict === undefined) return null;

  // Normalize verdict.
  // Synonyms: small critics occasionally emit "reject"/"unsure"/"caution"
  // instead of the canonical labels. Map them to the safest equivalent
  // rather than silently defaulting to "pass" (which would mask a real
  // flag) — but also rather than "fail" (which would false-positive on a
  // model that meant "warn" or just used a non-canonical synonym for
  // "pass"). "warn" is the conservative middle ground.
  const verdictRaw = String(p.verdict).toLowerCase();
  const verdict: Verdict =
    verdictRaw === "fail" || verdictRaw === "reject"
      ? "fail"
      : verdictRaw === "warn" || verdictRaw === "caution"
        ? "warn"
        : verdictRaw === "pass" || verdictRaw === "ok" || verdictRaw === "accept"
          ? "pass"
          : "warn"; // unknown label → conservative warn (not silent pass)

  // Normalize severity.
  let severity = Number(p.severity);
  if (!Number.isFinite(severity)) severity = 0;
  severity = Math.max(0, Math.min(5, Math.round(severity)));

  // Normalize arrays. Caps prevent a misbehaving critic that returns a
  // 50 KB "concern" string (or 200 of them) from bloating summary_md and
  // wasting bandwidth on the MCP wire.
  const MAX_ITEMS = 20;
  const MAX_ITEM_LEN = 2_000;
  const sanitizeStringArray = (raw: unknown): string[] =>
    Array.isArray(raw)
      ? raw
          .filter((x): x is string => typeof x === "string")
          .map((s) => (s.length > MAX_ITEM_LEN ? s.slice(0, MAX_ITEM_LEN) + "…" : s))
          .slice(0, MAX_ITEMS)
      : [];
  const concerns = sanitizeStringArray(p.concerns);
  const suggested_fixes = sanitizeStringArray(p.suggested_fixes);

  // Disputed span (optional). Snake-case on the wire, lifted into
  // ParsedCriticOutput unchanged so the caller can map it to the
  // camel-case CriticResult.disputedSpan. Treat missing, null, empty,
  // whitespace-only, and non-string values as absent. Same length cap
  // as a concern string so a misbehaving critic that returned the
  // entire answer here cannot bloat the payload. We do not validate
  // that the value is a substring of the answer at parse time — the
  // parser has no access to the answer text and shipping the field as
  // display only means a stray paraphrase only mildly mis-renders;
  // upstream the prompt already insists on verbatim copying.
  let disputed_span: string | undefined;
  if (typeof p.disputed_span === "string") {
    const trimmed = p.disputed_span.trim();
    if (trimmed.length > 0) {
      disputed_span =
        trimmed.length > MAX_ITEM_LEN
          ? trimmed.slice(0, MAX_ITEM_LEN) + "…"
          : trimmed;
    }
  }

  return { verdict, severity, concerns, suggested_fixes, disputed_span };
}

/**
 * Scan `text` for balanced JSON objects (respecting quoted strings and
 * backslash escapes) and return the substring for the most-relevant
 * parseable object. When `options.requireKey` is supplied, prefer the
 * LAST top-level object that contains that key (consistent with the
 * critic-verdict rule: models often scratchpad before the real JSON).
 * Falls back to any parseable object if no key-matching one exists.
 *
 * Returns null when no balanced parseable object can be located.
 *
 * Generic helper used by:
 *   - parseCriticJson (requireKey: "verdict")
 *   - parseClaimsJson (requireKey: "claims")
 *   - extractAnalysisJson (requireKey: "agreements")
 *
 * Was previously inlined as `findBestJsonWithVerdict`. Two earlier
 * call-sites used a naive `indexOf("{")` ... `lastIndexOf("}")` span
 * that could grab the wrong text when the model emitted prose
 * containing `{` characters before the real JSON object.
 */
export function findBalancedJsonObject(
  text: string,
  options?: { requireKey?: string }
): string | null {
  const requireKey = options?.requireKey;
  const openPositions: number[] = [];
  for (let i = 0; i < text.length; i++) {
    if (text[i] === "{") openPositions.push(i);
  }
  if (openPositions.length === 0) return null;

  // Walk from LAST `{` to FIRST so we prefer the final qualifying object.
  // (Critics, claim extractors, and analysis models all tend to emit
  // scratchpad/prose before the real JSON.)
  if (requireKey !== undefined) {
    for (let idx = openPositions.length - 1; idx >= 0; idx--) {
      const start = openPositions[idx]!;
      const end = findBalancedClose(text, start);
      if (end === -1) continue;
      const candidate = text.slice(start, end + 1);
      try {
        const parsed = JSON.parse(candidate);
        if (
          parsed &&
          typeof parsed === "object" &&
          !Array.isArray(parsed) &&
          requireKey in (parsed as Record<string, unknown>)
        ) {
          return candidate;
        }
      } catch {
        // Keep trying earlier `{` positions.
      }
    }
  }

  // Nothing with the required key parsed (or no key required) — fall
  // back to any parseable object, again preferring the last one.
  for (let idx = openPositions.length - 1; idx >= 0; idx--) {
    const start = openPositions[idx]!;
    const end = findBalancedClose(text, start);
    if (end === -1) continue;
    const candidate = text.slice(start, end + 1);
    try {
      JSON.parse(candidate);
      return candidate;
    } catch {
      /* ignore */
    }
  }

  return null;
}

/**
 * Starting at `start` (which must be an opening `{`), return the index of
 * the matching closing `}`. Respects double-quoted strings (including
 * backslash escapes). Returns -1 if no balanced close found.
 */
function findBalancedClose(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  let escape = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (escape) {
      escape = false;
      continue;
    }
    if (ch === "\\") {
      escape = true;
      continue;
    }
    if (ch === '"') {
      inString = !inString;
      continue;
    }
    if (inString) continue;
    if (ch === "{") depth++;
    else if (ch === "}") {
      depth--;
      if (depth === 0) return i;
    }
  }
  return -1;
}
