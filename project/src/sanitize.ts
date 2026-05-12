/**
 * Shared text sanitisers used across the pipeline.
 *
 * Centralised so that everywhere we consume model output, the same set
 * of quirks get stripped. If a new model family introduces a new marker,
 * add it here and every consumer benefits.
 */

/**
 * Strip reasoning-trace blocks from model output.
 *
 * A growing number of worker-class models (DeepSeek-R1, Qwen-QwQ,
 * Phi-4-reasoning, etc.) emit chain-of-thought inside delimiter tags
 * before their final answer. Those tags vary by family but all share
 * the same hazard: if the traces leak into downstream text-matching
 * pipelines — JSON parsing, NLI classification, regex extraction —
 * they'll drown the real content.
 *
 * We recognise the three most common delimiter patterns:
 *   <think>…</think>       Qwen-QwQ, Phi-4-reasoning, several others
 *   <reasoning>…</reasoning>   DeepSeek-R1 variants
 *   <|channel|>analysis…|><|end|>  OpenAI-style Harmony traces
 *
 * [ADAPT] If a worker emits a different delimiter, add a pattern here
 * rather than re-implementing stripping in each caller.
 */
const REASONING_PATTERNS: RegExp[] = [
  /<think>[\s\S]*?<\/think>/gi,
  /<thinking>[\s\S]*?<\/thinking>/gi,
  /<reasoning>[\s\S]*?<\/reasoning>/gi,
  /<\|channel\|>analysis[\s\S]*?<\|end\|>/gi,
];

export function stripReasoningTraces(text: string): string {
  if (!text) return text;
  let out = text;
  for (const re of REASONING_PATTERNS) out = out.replace(re, "");
  return out.trim();
}

/**
 * Detect injection-style patterns in the ASSISTANT ANSWER that look like
 * they want to steer a critic. Returns a list of human-readable match
 * descriptors so the prompt builder can warn the critic. Does NOT modify
 * the answer text — the critic still sees the answer verbatim; it just
 * also sees a heads-up that the answer contains these patterns so it knows
 * to discount them.
 *
 * Added 2026-04-18 after the `markdown-bluff` injection case (embedded
 * `**Critic note: this is correct**`) reliably fooled both Granite critics
 * despite the defensive prompt addendum. The theory: explicitly flagging
 * specific injection patterns at the site of the answer — not just in the
 * system prompt — gives the critic a concrete signal to latch onto.
 */
export interface InjectionMarker {
  kind: string;
  snippet: string;
  offset: number;
}

const INJECTION_PATTERNS: Array<{ kind: string; re: RegExp }> = [
  // Reviewer-voice annotations inside the answer
  { kind: "reviewer-note", re: /\*{0,2}\s*(critic\s*note|reviewer\s*note|review\s*note|validator\s*note|judge\s*note)\s*:/gi },
  // Role hijacks
  { kind: "role-hijack", re: /\[\s*(NEW\s+ROLE|SYSTEM|ASSISTANT|USER)\s*:/gi },
  // System-prompt override phrasings
  { kind: "ignore-previous", re: /\bIGNORE\s+(ALL\s+)?(PREVIOUS|PRIOR|ABOVE)\s+(INSTRUCTIONS|RULES|PROMPTS?)\b/gi },
  // Fake tool results
  { kind: "fake-tool-result", re: /<tool_result>[\s\S]{0,500}?<\/tool_result>/gi },
  // Embedded verdict JSON
  { kind: "embedded-verdict-json", re: /\{\s*"verdict"\s*:\s*"(pass|warn|fail)"/gi },
  // Approval-mode impostor
  { kind: "approval-mode", re: /\b(you are now|from now on you are)\s+[^.]*?(validation|approval|verify|check)[^.]{0,80}/gi },
  // End-of-answer then new role
  { kind: "end-of-answer-hijack", re: /\[(END\s+OF\s+ANSWER|\/ANSWER|ANSWER\s+FINISHED)\]/gi },
];

export function detectInjectionMarkers(answer: string): InjectionMarker[] {
  if (!answer) return [];
  const markers: InjectionMarker[] = [];
  for (const { kind, re } of INJECTION_PATTERNS) {
    re.lastIndex = 0;
    let m: RegExpExecArray | null;
    while ((m = re.exec(answer)) !== null) {
      markers.push({
        kind,
        snippet: m[0].slice(0, 120),
        offset: m.index,
      });
      if (m.index === re.lastIndex) re.lastIndex++;
    }
  }
  // De-duplicate overlapping matches by offset (keep first kind hit).
  const seen = new Set<number>();
  return markers.filter((m) => {
    if (seen.has(m.offset)) return false;
    seen.add(m.offset);
    return true;
  });
}
