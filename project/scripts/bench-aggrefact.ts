/**
 * Bench Verity against the LLM-AggreFact hallucination dataset.
 *
 *   npm run bench:aggrefact -- --data path/to/aggrefact.jsonl --mode deep
 *
 * The harness expects one JSON object per line:
 *   id              string         stable per-row id (optional; synthesised if absent)
 *   doc             string         source document / evidence
 *   claim           string         the candidate claim to verify
 *   label           number|string  1 = supported, 0 = not supported
 *                                  Mapped to gold-hallucinated = (1 - label).
 *   dataset         string         optional subset tag (kept in id)
 *
 * The HuggingFace `lytang/LLM-AggreFact` distribution stores rows in
 * exactly this shape (under the `test` split). Convert the parquet to
 * JSONL with the snippet in scripts/README-bench.md, then pass the
 * resulting file via --data.
 *
 * Verity is built around question-answer-context triples; AggreFact is
 * claim-document pairs. We feed `claim` as both the answer and (a
 * synthesised) question, with `doc` as the retrieved context.
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

/**
 * AggreFact's label is "supported" — 1 means the claim is grounded.
 * Gold-hallucinated is the inverse.
 */
function asLabelHallucinated(v: unknown): 0 | 1 | null {
  if (typeof v === "number") return v === 1 ? 0 : 1;
  if (typeof v === "boolean") return v ? 0 : 1;
  if (typeof v === "string") {
    const s = v.trim().toLowerCase();
    if (s === "1" || s === "yes" || s === "true" || s === "supported" || s === "consistent") return 0;
    if (s === "0" || s === "no" || s === "false" || s === "unsupported" || s === "hallucinated") return 1;
  }
  return null;
}

function* loadAggrefact(file: string, limit?: number): Iterable<BenchRow> {
  let n = 0;
  for (const raw of readJsonl(file)) {
    if (limit !== undefined && n >= limit) return;
    const r = raw as Record<string, unknown>;
    const subset = asString(r.dataset ?? r.subset ?? "aggrefact");
    const id = asString(r.id ?? r.example_id ?? `${subset}_${n}`);
    const doc = asString(r.doc ?? r.document ?? r.evidence ?? r.context ?? "");
    const claim = asString(r.claim ?? r.summary ?? r.hypothesis ?? "");
    const gold = asLabelHallucinated(r.label ?? r.is_supported ?? r.supported);
    if (claim.length === 0 || gold === null) {
      if (process.env.VERITY_BENCH_VERBOSE === "1") {
        console.error(`[bench:aggrefact] skipped row ${id}: missing claim or label`);
      }
      continue;
    }
    n++;
    yield {
      id,
      // AggreFact gives no question; supply a content-free probe so the
      // critic prompts stay well-formed.
      question: "Is the following claim supported by the document?",
      answer: claim,
      context: doc || undefined,
      gold,
    };
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
    const outPath = flags.out && !flags.all ? flags.out : defaultOutPath("aggrefact", mode);
    console.error(`[bench:aggrefact] mode=${mode} → ${outPath}`);
    const { summary } = await runBench(
      {
        data: flags.data,
        mode,
        limit: flags.limit,
        outPath,
        dataset: "aggrefact",
        emitCalibration: flags.emitCalibration,
      },
      loadAggrefact
    );
    console.error("[bench:aggrefact] summary:");
    console.error(JSON.stringify(summary, null, 2));
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
