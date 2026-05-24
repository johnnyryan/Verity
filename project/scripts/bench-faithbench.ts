// NOTE: expected npm script (added by cross-cutting agent): `bench:faithbench`.
/**
 * Bench Verity against the FaithBench summarisation-hallucination dataset.
 *
 *   npm run bench:faithbench -- --data path/to/batch_1.json --mode deep
 *   npm run bench:faithbench -- --data path/to/data_for_release_dir/ --mode deep
 *
 * FaithBench (Vectara 2024, https://github.com/vectara/FaithBench) ships
 * as one JSON file per annotation batch under `data_for_release/`. Each
 * file is a FaithBenchBatch object:
 *
 *   {
 *     "samples": [
 *       {
 *         "sample_id":  0,
 *         "source":     "<source document>",
 *         "summary":    "<model-generated summary>",
 *         "annotations": [
 *           {
 *             "label": ["Unwanted", "Unwanted.Intrinsic"],   // hierarchical
 *             "summary_span": "...", "summary_start": N, "summary_end": M,
 *             ...annotator metadata...
 *           },
 *           ...
 *         ],
 *         "metadata": { "summarizer": "...", "hhem-2.1": 0.5, ... }
 *       },
 *       ...
 *     ]
 *   }
 *
 * `annotations` is empty for a Consistent sample. The four span-level
 * label classes are Consistent (no annotation), Benign, Questionable,
 * Unwanted (with sub-classes Unwanted.Intrinsic / Unwanted.Extrinsic).
 *
 * --data may point at either a single batch JSON file or a directory
 * containing batch_*.json files (mirrors the upstream `data_for_release/`
 * layout, batches 1..16 excluding 13).
 *
 * Verity wants (question, answer, context); FaithBench is (source,
 * summary). We feed `summary` as the answer, `source` as the prior
 * context, and a fixed summarisation probe as the question. Treat the
 * resulting scores as summary-faithfulness, not end-to-end answer
 * quality (same caveat as bench-aggrefact).
 *
 * ─── Label collapsing ─────────────────────────────────────────────────
 *
 * FaithBench is 4-way at the span level. We need a binary
 * sample-level gold for Verity's convention (1 = hallucinated, 0 = clean).
 * Two protocols are supported via `--label-protocol`:
 *
 *   lenient  (default; brief spec)
 *     Unwanted (any sub)     → hallucinated (1)
 *     Consistent (no annot.) → clean (0)
 *     Questionable           → clean (0)   [ambiguous, scored as clean]
 *     Benign                 → clean (0)   [ambiguous, scored as clean]
 *
 *   strict   (matches upstream `scripts/binarize.py` default config
 *             `aggregation_strategy="worst", hallucinated_classes=
 *             [Questionable, Unwanted, Unwanted_Intrinsic, Unwanted_Extrinsic]`)
 *     Unwanted (any sub)     → hallucinated (1)
 *     Questionable           → hallucinated (1)
 *     Benign                 → clean (0)
 *     Consistent (no annot.) → clean (0)
 *
 * Aggregation across multiple annotations on one sample is "worst"
 * (matching the upstream default): the most severe label wins. The
 * chosen mapping is recorded in the TSV footer so a run can be
 * re-mapped without re-invoking the pipeline if conventions shift.
 */

import * as fs from "node:fs";
import * as path from "node:path";

import {
  parseFlags as parseSharedFlags,
  runBench,
  defaultOutPath,
  type BenchRow,
  type ParsedFlags,
} from "./bench-common.ts";
import type { VerifyMode } from "../../dist/types.js";

// ─── Local types ──────────────────────────────────────────────────────

type FaithBenchLabelClass =
  | "Consistent"
  | "Benign"
  | "Questionable"
  | "Unwanted";

type LabelProtocol = "lenient" | "strict";

interface FaithBenchAnnotation {
  label?: string[];
  summary_span?: string;
  summary_start?: number;
  summary_end?: number;
  [k: string]: unknown;
}

interface FaithBenchSample {
  sample_id?: number | string;
  source?: string;
  summary?: string;
  annotations?: FaithBenchAnnotation[];
  metadata?: Record<string, unknown>;
  [k: string]: unknown;
}

interface FaithBenchBatch {
  samples?: FaithBenchSample[];
  batch_id?: number | string;
  [k: string]: unknown;
}

// ─── Label collapsing ─────────────────────────────────────────────────

const SEVERITY: Record<FaithBenchLabelClass, number> = {
  Consistent: 0,
  Benign: 1,
  Questionable: 2,
  Unwanted: 3,
};

/**
 * Normalise one upstream label string (e.g. "Unwanted.Intrinsic",
 * "unwanted_extrinsic", "Benign") to its top-level class.
 */
function normaliseLabel(raw: string): FaithBenchLabelClass | null {
  const head = raw.split(/[._]/, 1)[0]?.trim().toLowerCase();
  if (!head) return null;
  if (head === "consistent") return "Consistent";
  if (head === "benign") return "Benign";
  if (head === "questionable") return "Questionable";
  if (head === "unwanted") return "Unwanted";
  return null;
}

/**
 * "worst" aggregation: highest-severity span-level label across all
 * annotations on the sample. Empty annotations → Consistent.
 */
function sampleLevelClass(sample: FaithBenchSample): FaithBenchLabelClass {
  const annotations = sample.annotations ?? [];
  let worst: FaithBenchLabelClass = "Consistent";
  let worstSev = SEVERITY.Consistent;
  for (const a of annotations) {
    const labels = a.label ?? [];
    for (const lbl of labels) {
      const cls = normaliseLabel(lbl);
      if (!cls) continue;
      const sev = SEVERITY[cls];
      if (sev > worstSev) {
        worstSev = sev;
        worst = cls;
      }
    }
  }
  return worst;
}

function classToGold(cls: FaithBenchLabelClass, protocol: LabelProtocol): 0 | 1 {
  if (cls === "Unwanted") return 1;
  if (protocol === "strict" && cls === "Questionable") return 1;
  return 0;
}

// ─── Data loading ─────────────────────────────────────────────────────

function readBatchFile(filePath: string): FaithBenchSample[] {
  const text = fs.readFileSync(filePath, "utf8");
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (err) {
    throw new Error(`[bench:faithbench] failed to parse JSON at ${filePath}: ${err}`);
  }
  // Either { samples: [...] } (canonical) or a bare array (defensive).
  if (Array.isArray(parsed)) return parsed as FaithBenchSample[];
  const batch = parsed as FaithBenchBatch;
  return batch.samples ?? [];
}

function* iterDataPath(dataPath: string): Iterable<{ file: string; sample: FaithBenchSample }> {
  const stat = fs.statSync(dataPath);
  if (stat.isDirectory()) {
    const files = fs
      .readdirSync(dataPath)
      .filter((f) => /^batch_\d+\.json$/i.test(f))
      .sort((a, b) => {
        const na = Number(a.match(/\d+/)?.[0] ?? 0);
        const nb = Number(b.match(/\d+/)?.[0] ?? 0);
        return na - nb;
      });
    if (files.length === 0) {
      throw new Error(
        `[bench:faithbench] no batch_*.json files in directory ${dataPath}`
      );
    }
    for (const f of files) {
      const full = path.join(dataPath, f);
      for (const sample of readBatchFile(full)) yield { file: f, sample };
    }
    return;
  }
  for (const sample of readBatchFile(dataPath)) {
    yield { file: path.basename(dataPath), sample };
  }
}

const SUMMARISATION_PROBE =
  "Summarise the following document. Is the summary faithful to the source?";

function makeLoader(protocol: LabelProtocol) {
  return function* loadFaithBench(file: string, limit?: number): Iterable<BenchRow> {
    let n = 0;
    let kept = 0;
    for (const { file: batchFile, sample } of iterDataPath(file)) {
      if (limit !== undefined && kept >= limit) return;
      n++;
      const source = typeof sample.source === "string" ? sample.source : "";
      const summary = typeof sample.summary === "string" ? sample.summary : "";
      if (summary.length === 0) {
        if (process.env.VERITY_BENCH_VERBOSE === "1") {
          console.error(
            `[bench:faithbench] skipped sample ${sample.sample_id ?? n} in ${batchFile}: empty summary`
          );
        }
        continue;
      }
      const cls = sampleLevelClass(sample);
      const gold = classToGold(cls, protocol);
      const sid = sample.sample_id ?? `${batchFile}_${n}`;
      const id = `${path.basename(batchFile, ".json")}_${sid}`;
      kept++;
      yield {
        id,
        question: SUMMARISATION_PROBE,
        answer: summary,
        context: source.length > 0 ? source : undefined,
        gold,
      };
    }
  };
}

// ─── CLI flag parsing ─────────────────────────────────────────────────

interface FaithBenchFlags extends ParsedFlags {
  labelProtocol: LabelProtocol;
}

function parseFlags(argv: string[]): FaithBenchFlags {
  // Strip --label-protocol before delegating; the shared parser would throw.
  const filtered: string[] = [];
  let labelProtocol: LabelProtocol = "lenient";
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--label-protocol") {
      const v = argv[++i];
      if (v !== "lenient" && v !== "strict") {
        throw new Error(`--label-protocol must be lenient|strict, got '${v}'`);
      }
      labelProtocol = v;
    } else {
      filtered.push(a);
    }
  }
  const shared = parseSharedFlags(filtered);
  return { ...shared, labelProtocol };
}

// ─── Entry point ──────────────────────────────────────────────────────

async function main(): Promise<void> {
  const flags = parseFlags(process.argv.slice(2));
  if (!flags.data) {
    console.error(
      "error: --data <path> is required " +
        "(a batch_*.json file or a data_for_release/ directory)"
    );
    process.exit(2);
  }
  const modes: VerifyMode[] = flags.all ? ["standard", "deep", "deeper"] : [flags.mode];
  for (const mode of modes) {
    const outPath = flags.out && !flags.all ? flags.out : defaultOutPath("faithbench", mode);
    console.error(
      `[bench:faithbench] mode=${mode} protocol=${flags.labelProtocol} -> ${outPath}`
    );
    const { summary } = await runBench(
      {
        data: flags.data,
        mode,
        limit: flags.limit,
        outPath,
        dataset: "faithbench",
        emitCalibration: flags.emitCalibration,
      },
      makeLoader(flags.labelProtocol)
    );
    // Append the label-protocol footer so re-mapping is possible later.
    fs.appendFileSync(
      outPath,
      `# label_protocol: ${flags.labelProtocol}\n` +
        `# label_mapping: Unwanted(any sub)=1; Consistent=0; ` +
        (flags.labelProtocol === "strict"
          ? "Questionable=1; Benign=0\n"
          : "Questionable=0; Benign=0\n") +
        `# aggregation: worst-severity across span annotations\n` +
        `# question_probe: ${JSON.stringify(SUMMARISATION_PROBE)}\n`
    );
    console.error("[bench:faithbench] summary:");
    console.error(JSON.stringify(summary, null, 2));
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
