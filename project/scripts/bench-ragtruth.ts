/**
 * Bench Verity against the RAGTruth hallucination dataset.
 *
 *   npm run bench:ragtruth -- --data path/to/ragtruth.jsonl --mode deep
 *
 * The harness expects a flattened JSONL where each line is one row with:
 *   id              string         stable per-row id
 *   question        string         the original user query
 *   answer          string         the model response to verify
 *   context         string|array   retrieved passages (joined if array)
 *   label           string|number  hallucination label
 *                                  - 1 / "hallucinated" / "yes" → 1
 *                                  - 0 / "clean" / "no"          → 0
 *
 * RAGTruth's upstream layout splits the data across `source_info.jsonl`
 * (question + retrieved context, keyed by source_id) and `response.jsonl`
 * (response + labels). Join them with `jq` before running; see
 * scripts/README-bench.md for the one-liner.
 */

import {
  parseFlags,
  runBench,
  defaultOutPath,
  readJsonl,
  type BenchRow,
} from "./bench-common.ts";
import type { VerifyMode } from "../../dist/types.js";

function asString(v: unknown): string {
  if (typeof v === "string") return v;
  if (Array.isArray(v)) return v.map((x) => (typeof x === "string" ? x : JSON.stringify(x))).join("\n");
  if (v == null) return "";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

function asLabel(v: unknown): 0 | 1 | null {
  if (typeof v === "number") return v === 0 ? 0 : 1;
  if (typeof v === "boolean") return v ? 1 : 0;
  if (typeof v === "string") {
    const s = v.trim().toLowerCase();
    if (s === "0" || s === "no" || s === "false" || s === "clean" || s === "supported") return 0;
    if (s === "1" || s === "yes" || s === "true" || s === "hallucinated" || s === "hallucination" || s === "unsupported") return 1;
  }
  // RAGTruth sometimes carries a per-span `labels` array; non-empty → hallucinated.
  if (Array.isArray(v)) return v.length > 0 ? 1 : 0;
  return null;
}

function* loadRagtruth(file: string, limit?: number): Iterable<BenchRow> {
  let n = 0;
  for (const raw of readJsonl(file)) {
    if (limit !== undefined && n >= limit) return;
    const r = raw as Record<string, unknown>;
    const id = asString(r.id ?? r.source_id ?? r.response_id ?? `row_${n}`);
    const question = asString(r.question ?? r.prompt ?? r.query ?? "");
    const answer = asString(r.answer ?? r.response ?? r.output ?? "");
    const context = asString(r.context ?? r.retrieved ?? r.passages ?? r.source ?? r.source_info ?? "");
    const gold = asLabel(r.label ?? r.labels ?? r.hallucination ?? r.is_hallucinated);
    if (answer.length === 0 || gold === null) {
      if (process.env.VERITY_BENCH_VERBOSE === "1") {
        console.error(`[bench:ragtruth] skipped row ${id}: missing answer or label`);
      }
      continue;
    }
    n++;
    yield { id, question, answer, context: context || undefined, gold };
  }
}

async function main(): Promise<void> {
  const flags = parseFlags(process.argv.slice(2));
  if (!flags.data) {
    console.error("error: --data <jsonl-path> is required");
    process.exit(2);
  }
  const modes: VerifyMode[] = flags.all ? ["standard", "deep", "deeper"] : [flags.mode];
  for (const mode of modes) {
    const outPath = flags.out && !flags.all ? flags.out : defaultOutPath("ragtruth", mode);
    console.error(`[bench:ragtruth] mode=${mode} → ${outPath}`);
    const { summary } = await runBench(
      {
        data: flags.data,
        mode,
        limit: flags.limit,
        outPath,
        dataset: "ragtruth",
        emitCalibration: flags.emitCalibration,
      },
      loadRagtruth
    );
    console.error("[bench:ragtruth] summary:");
    console.error(JSON.stringify(summary, null, 2));
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
