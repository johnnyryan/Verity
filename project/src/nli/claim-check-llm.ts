/**
 * LLM-based claim checker — alternative to DeBERTa NLI for Option C.
 *
 * Uses a small Ollama model (Granite 3.2 2B by default) to classify each
 * extracted claim as supported / contradicted / unsupported against a
 * premise (prior_context) or — when no premise — against world knowledge.
 *
 * Returns the same NliResult shape as the DeBERTa classifier so it can
 * be swapped in transparently via NLI_IMPL=llm.
 *
 * Rationale: DeBERTa-v3-mnli and cross-encoder variants return "neutral"
 * on subtle logical oppositions that a small LLM can actually reason
 * about. Measured in the 2026-04-18 NLI audit: 0 contradictions flagged
 * across explicitly contradictory pairs. See experiments/.
 */
import { NLI_LLM_MODEL, OLLAMA_URL, VERBOSE_LOGGING } from "../config.js";
import type { NliResult } from "../types.js";
import { extractClaims } from "./extract-claims.js";

const SYSTEM_PROMPT = `
You are a claim verification classifier. Given a single CLAIM and a CONTEXT
(which may be empty, meaning "evaluate against common world knowledge"),
classify the claim into exactly one of:
  supported   — the context directly supports or entails the claim
  contradicted — the context directly contradicts the claim, OR (if context
                 is empty) the claim is clearly false by common knowledge
  unsupported — the context is silent on the claim and we have no strong
                 common-knowledge basis to judge either way

Respond with ONLY a single JSON object, nothing else:
{"label": "supported" | "contradicted" | "unsupported", "confidence": 0.0-1.0}

Rules:
- Output JSON only, no prose, no markdown fences.
- Confidence in [0,1]; 1.0 = certain, 0.5 = roughly even, 0.0 = guessing.
- If the claim is vague or subjective, prefer "unsupported" with lower
  confidence.
- If context empty and claim is controversial or context-dependent,
  return "unsupported".
`.trim();

function buildUserPrompt(claim: string, context: string): string {
  const ctx =
    context && context.trim().length > 0
      ? context.trim()
      : "(no context provided; evaluate against common world knowledge)";
  return `CONTEXT:\n${ctx}\n\nCLAIM:\n${claim.trim()}`;
}

interface LlmVerdict {
  label: "supported" | "contradicted" | "unsupported";
  confidence: number;
}

function parseLlmVerdict(text: string): LlmVerdict | null {
  if (!text) return null;
  // Extract the first JSON object; tolerate extra whitespace / fences.
  const match = text.match(/\{[\s\S]*?\}/);
  if (!match) return null;
  try {
    const obj = JSON.parse(match[0]);
    const raw = String(obj.label ?? "").toLowerCase();
    let label: LlmVerdict["label"];
    if (raw.includes("contradict")) label = "contradicted";
    else if (raw.includes("support") || raw.includes("entail"))
      label = "supported";
    else label = "unsupported";
    const confidence = Math.max(0, Math.min(1, Number(obj.confidence ?? 0.5)));
    return { label, confidence };
  } catch {
    return null;
  }
}

async function classifyClaim(
  claim: string,
  context: string
): Promise<LlmVerdict | null> {
  const body = {
    model: NLI_LLM_MODEL,
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: buildUserPrompt(claim, context) },
    ],
    temperature: 0.0,
    max_tokens: 80,
    stream: false,
  };
  const res = await fetch(`${OLLAMA_URL}/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    if (VERBOSE_LOGGING) {
      console.error(
        `[NLI-LLM] Ollama ${res.status} ${await res.text().catch(() => "")}`
      );
    }
    return null;
  }
  const data: any = await res.json().catch(() => null);
  const text = data?.choices?.[0]?.message?.content ?? "";
  return parseLlmVerdict(text);
}

const CONTRADICTION_MIN_CONF = 0.55;
const UNSUPPORTED_MIN_CONF = 0.55;

export async function runLlmClaimCheck(
  answer: string,
  priorContext: string | undefined,
  preExtractedClaims?: string[]
): Promise<NliResult> {
  const claims = preExtractedClaims ?? extractClaims(answer);
  if (claims.length === 0) {
    return {
      ran: true,
      claims_checked: 0,
      contradictions: [],
      unsupported: [],
      notes: "LLM claim-check: no factual claims extracted.",
    };
  }

  const ctx = priorContext ?? "";
  const contradictions: NliResult["contradictions"] = [];
  const unsupported: NliResult["unsupported"] = [];

  // Parallel up to a small pool — Granite 2B handles a few concurrent
  // requests cleanly; going wider risks contention with the critic calls.
  const POOL = 3;
  const queue = claims.slice();
  const workers: Promise<void>[] = [];
  for (let i = 0; i < Math.min(POOL, queue.length); i++) {
    workers.push(
      (async () => {
        while (queue.length > 0) {
          const claim = queue.shift();
          if (!claim) return;
          const verdict = await classifyClaim(claim, ctx).catch(() => null);
          if (!verdict) continue;
          if (
            verdict.label === "contradicted" &&
            verdict.confidence >= CONTRADICTION_MIN_CONF
          ) {
            contradictions.push({
              claim,
              premise_snippet: ctx.slice(0, 200),
              confidence: verdict.confidence,
            });
          } else if (
            verdict.label === "unsupported" &&
            verdict.confidence >= UNSUPPORTED_MIN_CONF
          ) {
            unsupported.push({ claim });
          }
        }
      })()
    );
  }
  await Promise.all(workers);

  return {
    ran: true,
    claims_checked: claims.length,
    contradictions,
    unsupported,
    notes: `LLM claim-check (${NLI_LLM_MODEL}): ${contradictions.length} contradiction(s), ${unsupported.length} unsupported.`,
  };
}
