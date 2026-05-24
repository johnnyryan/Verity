/**
 * Bench Verity against ANAH-v2 (Shanghai AI Lab, NeurIPS 2024).
 *
 *   npm run bench:anah -- --data path/to/anah.jsonl --mode deep --language en
 *
 * Reference: Gu et al., "ANAH-v2: Scaling Analytical Hallucination Annotation
 * of Large Language Models", arXiv:2407.04693. Bilingual, sentence-level.
 *
 * (No npm script yet — expected name is `bench:anah`. See README-bench.md.)
 *
 * Schema. ANAH-v2 publishes per-sentence annotations. The annotator's
 * own eval.py writes one JSONL line per sentence:
 *
 *   { "question":   string,
 *     "response":   string,
 *     "sentence":   string,
 *     "annotation": "ok" | "nofact" | "contradictory" | "unverifiable",
 *     "language":   "en" | "zh" }
 *
 * The reference document lives in a sidecar `question_document.jsonl`
 * keyed by question, so an evaluation-ready flat JSONL ought to carry
 * the document too. We accept both shapes:
 *
 *   (a) Per-sentence rows with an optional `document` / `reference`
 *       field. We group by question + response and stitch sentences
 *       back into one row before sending to Verity.
 *
 *   (b) Per-response rows in the Mask-DPO packed form:
 *         { "question": ..., "response": ..., "document": ...,
 *           "sents": [...], "type": [...], "language": ... }
 *       Each `type[i]` is one of the four label strings above (or the
 *       Chinese equivalent emitted by the v2 annotator).
 *
 * Label mapping (per-sentence → per-row gold):
 *   - "contradictory"  → hallucinated (1)
 *   - "unverifiable"   → hallucinated (1)
 *   - "ok" / "no"      → clean (0)
 *   - "nofact"         → skip sentence (not a factual claim)
 *
 * Per-row aggregation: ANY sentence labelled contradictory or
 * unverifiable flips the row to hallucinated. A row whose every kept
 * sentence is "ok" is clean. A row whose every sentence is "nofact" is
 * skipped entirely.
 */

import {
  parseFlags,
  runBench,
  defaultOutPath,
  readJsonl,
  type BenchRow,
} from "./bench-common.ts";
import type { VerifyMode } from "../../dist/types.js";

// ─── Flag parsing (adds --language on top of the common flags) ────────

type AnahLanguage = "en" | "zh" | "both";

interface AnahFlags {
  data?: string;
  mode: VerifyMode;
  all: boolean;
  limit?: number;
  out?: string;
  emitCalibration?: string;
  language: AnahLanguage;
}

function parseAnahFlags(argv: string[]): AnahFlags {
  // Pull --language out first so the common parser does not choke on it.
  const remaining: string[] = [];
  let language: AnahLanguage = "en";
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--language") {
      const v = argv[++i];
      if (v !== "en" && v !== "zh" && v !== "both") {
        throw new Error(`--language must be en|zh|both, got '${v}'`);
      }
      language = v;
    } else {
      remaining.push(a);
    }
  }
  const base = parseFlags(remaining);
  return {
    data: base.data,
    mode: base.mode,
    all: base.all,
    limit: base.limit,
    out: base.out,
    emitCalibration: base.emitCalibration,
    language,
  };
}

// ─── String / label normalisation ─────────────────────────────────────

function asString(v: unknown): string {
  if (typeof v === "string") return v;
  if (Array.isArray(v)) {
    return v.map((x) => (typeof x === "string" ? x : JSON.stringify(x))).join("\n");
  }
  if (v == null) return "";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

/**
 * Canonical label string. ANAH-v2's annotator emits both English and
 * Chinese tags; we collapse them onto the four-token English set.
 *   "ok" | "contradictory" | "unverifiable" | "nofact" | null (unknown).
 */
function normaliseLabel(raw: unknown): "ok" | "contradictory" | "unverifiable" | "nofact" | null {
  if (typeof raw !== "string") return null;
  const s = raw.trim().toLowerCase().replace(/[<>「」\s]/g, "");
  // Chinese: 无幻觉 / 矛盾 / 无法验证 / 无事实
  if (s.includes("无幻觉") || s === "ok" || s === "no" || s === "nohallucination" || s === "supported") return "ok";
  if (s.includes("矛盾") || s === "contradictory" || s === "contradiction" || s === "contradict") return "contradictory";
  if (s.includes("无法验证") || s === "unverifiable" || s === "unverified") return "unverifiable";
  if (s.includes("无事实") || s === "nofact" || s === "nofacts" || s === "no_fact" || s === "non_factual") return "nofact";
  // The annotator also emits hallucination-type tags in the v1 vocabulary.
  if (s === "no_hallucination") return "ok";
  return null;
}

// ─── Loader ───────────────────────────────────────────────────────────

interface PackedRow {
  id: string;
  question: string;
  response: string;
  document: string;
  /** Per-sentence labels (canonical strings, in the order they arrived). */
  labels: Array<"ok" | "contradictory" | "unverifiable" | "nofact">;
  language: "en" | "zh";
}

/**
 * Read one or more JSONL lines and yield packed-per-row records.
 *
 * Per-sentence rows (one line per sentence) get grouped by question +
 * response. Packed rows (Mask-DPO-style with `sents` and `type`) are
 * consumed in one pass.
 */
function* loadPacked(file: string): Iterable<PackedRow> {
  // For per-sentence rows we accumulate in insertion order so we can
 // emit one PackedRow per (question, response) pair on the fly.
  const grouped = new Map<string, PackedRow>();
  let perSentenceCount = 0;

  for (const raw of readJsonl(file)) {
    const r = raw as Record<string, unknown>;

    // Detect packed form first.
    const sents = r.sents ?? r.sentences ?? null;
    const types = r.type ?? r.types ?? r.labels ?? null;
    if (Array.isArray(sents) && Array.isArray(types) && sents.length === types.length) {
      const language = (asString(r.language).toLowerCase() as "en" | "zh") || "en";
      const labels: PackedRow["labels"] = [];
      for (const t of types) {
        const lab = normaliseLabel(t);
        if (lab) labels.push(lab);
      }
      const id = asString(r.id ?? r.uid ?? `anah_packed_${grouped.size + perSentenceCount}`);
      yield {
        id,
        question: asString(r.question ?? r.prompt ?? ""),
        response: asString(r.response ?? r.answer ?? r.content ?? ""),
        document: asString(r.document ?? r.reference ?? r.documents ?? r.evidence ?? ""),
        labels,
        language: language === "zh" ? "zh" : "en",
      };
      continue;
    }

    // Otherwise treat as per-sentence row.
    const question = asString(r.question ?? r.prompt ?? "");
    const response = asString(r.response ?? r.answer ?? r.content ?? "");
    const annotation = normaliseLabel(r.annotation ?? r.label ?? r.type ?? r.hallucination_type);
    const document = asString(r.document ?? r.reference ?? r.documents ?? r.evidence ?? "");
    const language = (asString(r.language).toLowerCase() as "en" | "zh") || "en";
    if (question.length === 0 || response.length === 0 || annotation === null) {
      if (process.env.VERITY_BENCH_VERBOSE === "1") {
        console.error(`[bench:anah] skipped malformed per-sentence row`);
      }
      continue;
    }
    const key = `${question}${response}`;
    let row = grouped.get(key);
    if (!row) {
      row = {
        id: asString(r.id ?? r.uid ?? `anah_${grouped.size}`),
        question,
        response,
        document,
        labels: [],
        language: language === "zh" ? "zh" : "en",
      };
      grouped.set(key, row);
    } else if (row.document.length === 0 && document.length > 0) {
      // Pick up the document from whichever sentence carries it.
      row.document = document;
    }
    row.labels.push(annotation);
    perSentenceCount++;
  }

  for (const row of grouped.values()) yield row;
}

/**
 * Per-row aggregation:
 *   - any contradictory or unverifiable sentence → hallucinated (1)
 *   - else any "ok" sentence                     → clean (0)
 *   - else (all "nofact")                        → skip row
 */
function aggregateGold(labels: PackedRow["labels"]): 0 | 1 | null {
  let sawOk = false;
  for (const l of labels) {
    if (l === "contradictory" || l === "unverifiable") return 1;
    if (l === "ok") sawOk = true;
  }
  return sawOk ? 0 : null;
}

function* loadAnah(
  file: string,
  language: AnahLanguage,
  limit?: number
): Iterable<BenchRow> {
  let n = 0;
  for (const packed of loadPacked(file)) {
    if (limit !== undefined && n >= limit) return;
    if (language !== "both" && packed.language !== language) continue;
    if (packed.labels.length === 0) continue;
    const gold = aggregateGold(packed.labels);
    if (gold === null) continue; // every sentence was "nofact"
    if (packed.response.length === 0) continue;
    n++;
    yield {
      id: packed.id,
      question: packed.question,
      answer: packed.response,
      context: packed.document.length > 0 ? packed.document : undefined,
      gold,
    };
  }
}

// ─── Main ─────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const flags = parseAnahFlags(process.argv.slice(2));
  if (!flags.data) {
    console.error("error: --data <jsonl-path> is required");
    process.exit(2);
  }
  const modes: VerifyMode[] = flags.all ? ["standard", "deep", "deeper"] : [flags.mode];
  const languages: AnahLanguage[] =
    flags.language === "both" ? ["en", "zh"] : [flags.language];

  for (const lang of languages) {
    for (const mode of modes) {
      const datasetTag = `anah-${lang}`;
      const outPath =
        flags.out && !flags.all && languages.length === 1
          ? flags.out
          : defaultOutPath(datasetTag, mode);
      console.error(`[bench:anah] language=${lang} mode=${mode} → ${outPath}`);
      const { summary } = await runBench(
        {
          data: flags.data,
          mode,
          limit: flags.limit,
          outPath,
          dataset: datasetTag,
          emitCalibration: flags.emitCalibration,
        },
        (file, limit) => loadAnah(file, lang, limit)
      );
      // Footer breadcrumb: spell out the per-row aggregation choice so a
      // future reader doesn't have to re-derive it from this file.
      const fs = await import("node:fs");
      const note =
        `# anah_aggregation: any contradictory|unverifiable sentence → row=hallucinated; ` +
        `all "ok" → row=clean; all "nofact" → row skipped\n` +
        `# anah_language: ${lang}\n`;
      fs.appendFileSync(outPath, note);
      console.error("[bench:anah] summary:");
      console.error(JSON.stringify(summary, null, 2));
    }
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
