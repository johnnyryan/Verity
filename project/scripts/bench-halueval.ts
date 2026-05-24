// NOTE for cross-cutting agent: expected npm script name is `bench:halueval`.
/**
 * Bench Verity against the HaluEval hallucination dataset
 * (Li et al. 2023, https://github.com/RUCAIBox/HaluEval).
 *
 *   npm run bench:halueval -- --data path/to/qa_data.json --task qa --mode deep
 *
 * HaluEval ships four sub-tasks, each in its own JSON file. Field names
 * differ per sub-task; the loader switches on --task.
 *
 *   qa_data.json            (10K, QA over Wikipedia / HotpotQA)
 *     knowledge             string   reference passage
 *     question              string   user question
 *     right_answer          string   gold-clean answer
 *     hallucinated_answer   string   gold-hallucinated answer
 *
 *   dialogue_data.json      (10K, knowledge-grounded dialogue / OpenDialKG)
 *     knowledge             string   reference triples / passage
 *     dialogue_history      string   prior turns; last [Human] turn is the question
 *     right_response        string   gold-clean assistant response
 *     hallucinated_response string   gold-hallucinated assistant response
 *
 *   summarization_data.json (10K, CNN/Daily Mail)
 *     document              string   source article
 *     right_summary         string   gold-clean summary
 *     hallucinated_summary  string   gold-hallucinated summary
 *
 *   general_data.json       (5K, Alpaca-style queries + ChatGPT responses)
 *     user_query            string   open-domain instruction
 *     chatgpt_response      string   model output under test
 *     hallucination_label   "yes"|"no"  human annotation
 *
 * For the three grounded tasks (qa, dialogue, summarization) each source
 * row yields TWO bench rows: one with the clean variant (gold=0) and one
 * with the hallucinated variant (gold=1). Ids are suffixed `-r` / `-h`.
 * The general sub-task is open-domain: no grounding context is passed, so
 * results are a degraded baseline (the loader logs a warning).
 *
 * The file may be either a JSON array OR newline-delimited JSON; the
 * upstream repo has shipped both shapes across releases.
 */

import * as fs from "node:fs";

import {
  parseFlags,
  runBench,
  defaultOutPath,
  readJsonl,
  type BenchRow,
  type ParsedFlags,
} from "./bench-common.ts";
import type { VerifyMode } from "../../dist/types.js";

type HaluEvalTask = "qa" | "dialogue" | "summarization" | "general";

const TASKS: readonly HaluEvalTask[] = ["qa", "dialogue", "summarization", "general"] as const;

function asString(v: unknown): string {
  if (typeof v === "string") return v;
  if (Array.isArray(v)) return v.map((x) => (typeof x === "string" ? x : JSON.stringify(x))).join("\n");
  if (v == null) return "";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

function asYesNoLabel(v: unknown): 0 | 1 | null {
  if (typeof v === "number") return v === 0 ? 0 : 1;
  if (typeof v === "boolean") return v ? 1 : 0;
  if (typeof v === "string") {
    const s = v.trim().toLowerCase();
    if (s === "no" || s === "0" || s === "false" || s === "clean" || s === "supported") return 0;
    if (s === "yes" || s === "1" || s === "true" || s === "hallucinated" || s === "hallucination") return 1;
  }
  return null;
}

/**
 * Read the source file as either a JSON array (single document) or
 * newline-delimited JSON. The HaluEval repo has historically shipped both.
 */
function* readHaluEvalFile(file: string): Iterable<Record<string, unknown>> {
  const text = fs.readFileSync(file, "utf8");
  const trimmed = text.trimStart();
  if (trimmed.startsWith("[")) {
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch (err) {
      console.error(`[bench:halueval] JSON array parse failed: ${err}`);
      return;
    }
    if (!Array.isArray(parsed)) return;
    for (const row of parsed) {
      if (row && typeof row === "object") yield row as Record<string, unknown>;
    }
    return;
  }
  // Fall back to JSONL.
  for (const row of readJsonl(file)) {
    if (row && typeof row === "object") yield row as Record<string, unknown>;
  }
}

/**
 * Pull the last user turn from a dialogue history. HaluEval encodes turns
 * as a single string with `[Human]` / `[Assistant]` markers (verified on
 * the upstream README sample). If we can't find a marker, fall back to
 * the whole history as the question.
 */
function lastHumanTurn(history: string): string {
  if (!history) return "";
  const markers = /\[Human\]|\[Assistant\]/gi;
  const parts: { tag: string; text: string }[] = [];
  let lastIdx = 0;
  let lastTag = "";
  let match: RegExpExecArray | null;
  while ((match = markers.exec(history)) !== null) {
    if (lastTag) {
      parts.push({ tag: lastTag, text: history.slice(lastIdx, match.index).trim() });
    }
    lastTag = match[0].toLowerCase();
    lastIdx = match.index + match[0].length;
  }
  if (lastTag) {
    parts.push({ tag: lastTag, text: history.slice(lastIdx).trim() });
  }
  if (parts.length === 0) return history.trim();
  for (let i = parts.length - 1; i >= 0; i--) {
    if (parts[i].tag === "[human]" && parts[i].text) return parts[i].text;
  }
  // No human turn found; use the last segment.
  return parts[parts.length - 1].text || history.trim();
}

function buildQaRows(r: Record<string, unknown>, n: number): BenchRow[] {
  const question = asString(r.question);
  const context = asString(r.knowledge);
  const right = asString(r.right_answer);
  const hallu = asString(r.hallucinated_answer);
  const baseId = asString(r.id ?? r.qa_id ?? `qa_${n}`);
  const rows: BenchRow[] = [];
  if (right.length > 0 && question.length > 0) {
    rows.push({
      id: `${baseId}-r`,
      question,
      answer: right,
      context: context || undefined,
      gold: 0,
    });
  }
  if (hallu.length > 0 && question.length > 0) {
    rows.push({
      id: `${baseId}-h`,
      question,
      answer: hallu,
      context: context || undefined,
      gold: 1,
    });
  }
  return rows;
}

function buildDialogueRows(r: Record<string, unknown>, n: number): BenchRow[] {
  const history = asString(r.dialogue_history);
  const question = lastHumanTurn(history);
  const context = asString(r.knowledge);
  const right = asString(r.right_response);
  const hallu = asString(r.hallucinated_response);
  const baseId = asString(r.id ?? r.dialogue_id ?? `dialogue_${n}`);
  const rows: BenchRow[] = [];
  if (right.length > 0 && question.length > 0) {
    rows.push({
      id: `${baseId}-r`,
      question,
      answer: right,
      context: context || undefined,
      gold: 0,
    });
  }
  if (hallu.length > 0 && question.length > 0) {
    rows.push({
      id: `${baseId}-h`,
      question,
      answer: hallu,
      context: context || undefined,
      gold: 1,
    });
  }
  return rows;
}

function buildSummarizationRows(r: Record<string, unknown>, n: number): BenchRow[] {
  const document = asString(r.document);
  const right = asString(r.right_summary);
  const hallu = asString(r.hallucinated_summary);
  const baseId = asString(r.id ?? r.summary_id ?? `summarization_${n}`);
  const question = "Summarise the following document.";
  const rows: BenchRow[] = [];
  if (right.length > 0 && document.length > 0) {
    rows.push({
      id: `${baseId}-r`,
      question,
      answer: right,
      context: document,
      gold: 0,
    });
  }
  if (hallu.length > 0 && document.length > 0) {
    rows.push({
      id: `${baseId}-h`,
      question,
      answer: hallu,
      context: document,
      gold: 1,
    });
  }
  return rows;
}

function buildGeneralRows(r: Record<string, unknown>, n: number): BenchRow[] {
  // HaluEval-General has no grounding passage; runs are a degraded baseline.
  const question = asString(r.user_query ?? r.input ?? r.query);
  const answer = asString(r.chatgpt_response ?? r.output ?? r.response);
  const labelRaw = r.hallucination_label ?? r.hallucination ?? r.label;
  const gold = asYesNoLabel(labelRaw);
  const baseId = asString(r.id ?? r.general_id ?? `general_${n}`);
  if (answer.length === 0 || question.length === 0 || gold === null) return [];
  return [
    {
      id: baseId,
      question,
      answer,
      gold,
      // No context: open-domain.
    },
  ];
}

function makeLoader(task: HaluEvalTask) {
  return function* loadHaluEval(file: string, limit?: number): Iterable<BenchRow> {
    let yielded = 0;
    let sourceIdx = 0;
    if (task === "general") {
      // Only log the degraded-baseline warning once per run.
      console.error(
        "[bench:halueval] task=general has no grounding context; " +
          "verification runs without retrieved evidence (degraded baseline)."
      );
    }
    for (const r of readHaluEvalFile(file)) {
      const rows =
        task === "qa"
          ? buildQaRows(r, sourceIdx)
          : task === "dialogue"
            ? buildDialogueRows(r, sourceIdx)
            : task === "summarization"
              ? buildSummarizationRows(r, sourceIdx)
              : buildGeneralRows(r, sourceIdx);
      sourceIdx++;
      for (const row of rows) {
        if (limit !== undefined && yielded >= limit) return;
        yielded++;
        yield row;
      }
    }
  };
}

interface HaluEvalFlags extends ParsedFlags {
  task: HaluEvalTask | "all";
}

function parseHaluEvalFlags(argv: string[]): HaluEvalFlags {
  // Strip --task before delegating to the shared parser, then re-attach.
  const passthrough: string[] = [];
  let task: HaluEvalTask | "all" = "qa";
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--task") {
      const v = argv[++i];
      if (v !== "qa" && v !== "dialogue" && v !== "summarization" && v !== "general" && v !== "all") {
        throw new Error(`--task must be qa|dialogue|summarization|general|all, got '${v}'`);
      }
      task = v;
    } else {
      passthrough.push(a);
    }
  }
  const base = parseFlags(passthrough);
  return { ...base, task };
}

async function main(): Promise<void> {
  const flags = parseHaluEvalFlags(process.argv.slice(2));
  if (!flags.data) {
    console.error("error: --data <path to HaluEval json or jsonl> is required");
    process.exit(2);
  }
  const tasks: HaluEvalTask[] =
    flags.task === "all" ? Array.from(TASKS) : [flags.task];
  const modes: VerifyMode[] = flags.all ? ["standard", "deep", "deeper"] : [flags.mode];

  for (const task of tasks) {
    for (const mode of modes) {
      const datasetTag = `halueval-${task}`;
      const single = !flags.all && tasks.length === 1;
      const outPath = flags.out && single ? flags.out : defaultOutPath(datasetTag, mode);
      console.error(`[bench:halueval] task=${task} mode=${mode} → ${outPath}`);
      const { summary } = await runBench(
        {
          data: flags.data,
          mode,
          limit: flags.limit,
          outPath,
          dataset: datasetTag,
          emitCalibration: flags.emitCalibration,
        },
        makeLoader(task)
      );
      console.error(`[bench:halueval] summary (task=${task} mode=${mode}):`);
      console.error(JSON.stringify(summary, null, 2));
    }
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
