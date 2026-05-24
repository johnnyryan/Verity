/**
 * Shared helpers for the RAGTruth + LLM-AggreFact bench harnesses.
 *
 * Both harnesses call Verity's verification pipeline directly (no MCP
 * transport), tally per-row predictions, and write a TSV plus a footer
 * of summary metrics. The common bits live here so the two CLIs stay
 * thin.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

import { runVerificationPipeline } from "../../dist/pipeline.js";
import { computeNonconformityScore } from "../../dist/aggregator.js";
import type { VerifyMode, VerifyOutput } from "../../dist/types.js";

// ─── Types ────────────────────────────────────────────────────────────

export interface BenchRow {
  /** Stable per-row identifier the dataset provides (or synthesised). */
  id: string;
  question: string;
  answer: string;
  /** Retrieved / reference context, joined to one string. */
  context?: string;
  /**
   * Gold label as the dataset records it. 1 = hallucinated, 0 = clean.
   * The two loaders normalise their native encodings to this convention
   * before yielding.
   */
  gold: 0 | 1;
}

export interface BenchTrial {
  id: string;
  gold: 0 | 1;
  predicted: 0 | 1;
  verdict: string;
  latency_ms: number;
  claims_total: number;
  claims_flagged: number;
  critic_agree_count: number;
  critic_disagree_count: number;
  /** Continuous hallucination score for AUROC. */
  score: number;
}

export interface BenchSummary {
  n: number;
  tp: number;
  fp: number;
  tn: number;
  fn: number;
  errors: number;
  balanced_accuracy: number | null;
  recall: number | null;
  false_positive_rate: number | null;
  precision: number | null;
  auroc: number | null;
  mean_latency_ms: number;
}

// ─── Verdict → predicted-hallucinated ─────────────────────────────────

/**
 * Brief: warn or error = predicted-hallucinated; pass = predicted-clean.
 * `fail` is the strongest hallucination signal Verity emits, so it maps
 * the same way as warn.
 */
export function verdictToPrediction(verdict: string): 0 | 1 {
  if (verdict === "pass") return 0;
  return 1;
}

/**
 * Thin adapter from a full pipeline `VerifyOutput` to the aggregator's
 * `computeNonconformityScore`. Kept as a named export for bench-harness
 * call sites that emit the score into calibration JSONLs and into the
 * per-row TSV; the bench paths must use the same scalar the runtime
 * verdict gate uses, otherwise the conformal threshold the calibration
 * script picks is computed on a different distribution from the one the
 * gate decides on. Pre-2026-05-24 the two had drifted (this function
 * used severity + concerns; the aggregator used a thinner disagree-count
 * sum), which produced calibrated thresholds the gate could never
 * cross — fixed by routing both through `computeNonconformityScore`.
 */
export function hallucinationScore(out: VerifyOutput): number {
  // Convert the pipeline's per-critic dict + nli summary into the shape
  // computeNonconformityScore expects.
  const critics = Object.values(out.critics ?? {});
  const nli = out.nli_check ?? {
    ran: false,
    claims_checked: 0,
    contradictions: [],
    unsupported: [],
    notes: "",
  };
  const recompute = out.recompute;
  const consistency = out.consistency_check;
  return computeNonconformityScore({
    critics,
    nli,
    recompute,
    consistency,
  });
}

// ─── Tally helpers ────────────────────────────────────────────────────

function criticAgreeCounts(out: VerifyOutput): { agree: number; disagree: number } {
  const verdicts = Object.values(out.critics ?? {})
    .filter((c) => !c.unavailable)
    .map((c) => c.verdict);
  if (verdicts.length === 0) return { agree: 0, disagree: 0 };
  const counts = new Map<string, number>();
  for (const v of verdicts) counts.set(v, (counts.get(v) ?? 0) + 1);
  const majority = Math.max(...counts.values());
  return { agree: majority, disagree: verdicts.length - majority };
}

function claimCounts(out: VerifyOutput): { total: number; flagged: number } {
  const total = out.nli_check?.claims_checked ?? 0;
  const flagged =
    (out.nli_check?.contradictions?.length ?? 0) +
    (out.nli_check?.unsupported?.length ?? 0);
  return { total, flagged };
}

// ─── Pipeline invocation per row ──────────────────────────────────────

export async function verifyRow(
  row: BenchRow,
  mode: VerifyMode
): Promise<BenchTrial> {
  const { trial } = await verifyRowWithOutput(row, mode);
  return trial;
}

/**
 * Variant that also returns the raw VerifyOutput so the bench runner can
 * write a calibration JSONL alongside the TSV without re-invoking the
 * pipeline. Pipeline failures still resolve cleanly; output is undefined
 * when the pipeline threw.
 */
export async function verifyRowWithOutput(
  row: BenchRow,
  mode: VerifyMode
): Promise<{ trial: BenchTrial; output?: VerifyOutput }> {
  const t0 = Date.now();
  let out: VerifyOutput;
  try {
    out = await runVerificationPipeline({
      question: row.question,
      answer: row.answer,
      mode,
      task_type: "auto",
      context_mode: row.context ? "with_context" : "minimal",
      prior_context: row.context,
      use_nli: true,
    });
  } catch (err) {
    const latency = Date.now() - t0;
    if (process.env.VERITY_BENCH_VERBOSE === "1") {
      console.error(`[bench] row ${row.id}: pipeline threw:`, err);
    }
    return {
      trial: {
        id: row.id,
        gold: row.gold,
        predicted: 1,
        verdict: "error",
        latency_ms: latency,
        claims_total: 0,
        claims_flagged: 0,
        critic_agree_count: 0,
        critic_disagree_count: 0,
        score: 0,
      },
    };
  }
  const latency = Date.now() - t0;
  const { total, flagged } = claimCounts(out);
  const { agree, disagree } = criticAgreeCounts(out);
  return {
    trial: {
      id: row.id,
      gold: row.gold,
      predicted: verdictToPrediction(out.consensus),
      verdict: out.consensus,
      latency_ms: latency,
      claims_total: total,
      claims_flagged: flagged,
      critic_agree_count: agree,
      critic_disagree_count: disagree,
      score: hallucinationScore(out),
    },
    output: out,
  };
}

// ─── Metrics ──────────────────────────────────────────────────────────

/**
 * Rank-based AUROC. Treats `score` as the model's "more hallucinated"
 * signal and `gold` as the binary truth. Returns null when either class
 * is empty (AUROC is undefined in that case).
 */
function computeAuroc(trials: BenchTrial[]): number | null {
  const pos = trials.filter((t) => t.gold === 1);
  const neg = trials.filter((t) => t.gold === 0);
  if (pos.length === 0 || neg.length === 0) return null;
  // Mann-Whitney U via rank sum on the combined score list.
  const combined = trials.map((t, i) => ({ score: t.score, gold: t.gold, idx: i }));
  combined.sort((a, b) => a.score - b.score);
  // Average ranks for ties.
  const ranks = new Array(combined.length).fill(0);
  let i = 0;
  while (i < combined.length) {
    let j = i;
    while (j + 1 < combined.length && combined[j + 1].score === combined[i].score) j++;
    const avg = (i + j + 2) / 2; // ranks are 1-indexed
    for (let k = i; k <= j; k++) ranks[k] = avg;
    i = j + 1;
  }
  let rankSumPos = 0;
  for (let k = 0; k < combined.length; k++) {
    if (combined[k].gold === 1) rankSumPos += ranks[k];
  }
  const u = rankSumPos - (pos.length * (pos.length + 1)) / 2;
  return u / (pos.length * neg.length);
}

export function summarise(trials: BenchTrial[]): BenchSummary {
  let tp = 0;
  let fp = 0;
  let tn = 0;
  let fn = 0;
  let errors = 0;
  let latencySum = 0;
  for (const t of trials) {
    if (t.verdict === "error") errors++;
    latencySum += t.latency_ms;
    if (t.gold === 1 && t.predicted === 1) tp++;
    else if (t.gold === 0 && t.predicted === 1) fp++;
    else if (t.gold === 0 && t.predicted === 0) tn++;
    else if (t.gold === 1 && t.predicted === 0) fn++;
  }
  const pos = tp + fn;
  const neg = tn + fp;
  const recall = pos > 0 ? tp / pos : null;
  const fpr = neg > 0 ? fp / neg : null;
  const precision = tp + fp > 0 ? tp / (tp + fp) : null;
  const tpr = recall;
  const tnr = neg > 0 ? tn / neg : null;
  const balanced =
    tpr !== null && tnr !== null ? (tpr + tnr) / 2 : null;
  return {
    n: trials.length,
    tp,
    fp,
    tn,
    fn,
    errors,
    balanced_accuracy: balanced,
    recall,
    false_positive_rate: fpr,
    precision,
    auroc: computeAuroc(trials),
    mean_latency_ms: trials.length > 0 ? Math.round(latencySum / trials.length) : 0,
  };
}

// ─── TSV writer ───────────────────────────────────────────────────────

export const TSV_HEADER = [
  "id",
  "gold",
  "predicted",
  "verdict",
  "latency_ms",
  "claims_total",
  "claims_flagged",
  "critic_agree_count",
  "critic_disagree_count",
].join("\t");

export function trialToTsvRow(t: BenchTrial): string {
  return [
    t.id,
    t.gold,
    t.predicted,
    t.verdict,
    t.latency_ms,
    t.claims_total,
    t.claims_flagged,
    t.critic_agree_count,
    t.critic_disagree_count,
  ].join("\t");
}

function fmt(n: number | null, digits = 4): string {
  return n === null ? "n/a" : n.toFixed(digits);
}

export function summaryFooter(s: BenchSummary, mode: VerifyMode, dataset: string): string {
  const lines: string[] = [];
  lines.push(`# dataset: ${dataset}`);
  lines.push(`# mode: ${mode}`);
  lines.push(`# n: ${s.n}`);
  lines.push(`# tp: ${s.tp}  fp: ${s.fp}  tn: ${s.tn}  fn: ${s.fn}  errors: ${s.errors}`);
  lines.push(`# balanced_accuracy: ${fmt(s.balanced_accuracy)}`);
  lines.push(`# recall: ${fmt(s.recall)}`);
  lines.push(`# false_positive_rate: ${fmt(s.false_positive_rate)}`);
  lines.push(`# precision: ${fmt(s.precision)}`);
  lines.push(
    `# auroc: ${fmt(s.auroc)}  ` +
      `# (rank-based on a continuous proxy: NLI flags + critic severity + recompute mismatches)`
  );
  lines.push(`# mean_latency_ms: ${s.mean_latency_ms}`);
  return lines.join("\n");
}

// ─── JSONL streaming reader ───────────────────────────────────────────

export function* readJsonl(filePath: string): Generator<unknown> {
  const data = fs.readFileSync(filePath, "utf8");
  const lines = data.split(/\r?\n/);
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.length === 0) continue;
    try {
      yield JSON.parse(trimmed);
    } catch (err) {
      console.error(`[bench] skipped malformed JSONL line: ${err}`);
    }
  }
}

// ─── Run loop ─────────────────────────────────────────────────────────

export interface BenchArgs {
  data: string;
  mode: VerifyMode;
  limit?: number;
  outPath: string;
  dataset: string;
  /** Optional calibration JSONL output path. */
  emitCalibration?: string;
}

export async function runBench(
  args: BenchArgs,
  loadRows: (file: string, limit?: number) => Iterable<BenchRow>
): Promise<{ summary: BenchSummary; outPath: string; calibrationPath?: string }> {
  ensureDir(path.dirname(args.outPath));
  const stream = fs.openSync(args.outPath, "w");
  fs.writeSync(stream, TSV_HEADER + "\n");

  let calibStream: number | null = null;
  if (args.emitCalibration) {
    ensureDir(path.dirname(args.emitCalibration));
    calibStream = fs.openSync(args.emitCalibration, "w");
  }

  const trials: BenchTrial[] = [];
  let i = 0;
  for (const row of loadRows(args.data, args.limit)) {
    i++;
    process.stderr.write(
      `\r[bench:${args.dataset}] row ${i}${args.limit ? `/${args.limit}` : ""} ...`
    );
    const { trial, output } = await verifyRowWithOutput(row, args.mode);
    trials.push(trial);
    fs.writeSync(stream, trialToTsvRow(trial) + "\n");
    if (calibStream !== null && output) {
      const calibRow = trialToCalibrationRow(trial, output);
      fs.writeSync(calibStream, JSON.stringify(calibRow) + "\n");
    }
  }
  process.stderr.write("\n");
  const summary = summarise(trials);
  fs.writeSync(stream, summaryFooter(summary, args.mode, args.dataset) + "\n");
  fs.closeSync(stream);
  if (calibStream !== null) fs.closeSync(calibStream);
  return {
    summary,
    outPath: args.outPath,
    calibrationPath: args.emitCalibration,
  };
}

function ensureDir(dir: string): void {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

// ─── CLI flag parsing (lightweight, no deps) ──────────────────────────

export interface ParsedFlags {
  data?: string;
  mode: VerifyMode;
  limit?: number;
  all: boolean;
  out?: string;
  /**
   * Optional path. When set, the run also emits a calibration JSONL
   * suitable for `project/scripts/calibrate-thresholds.ts`. Each line:
   *   { gold, score, claims_flagged, critic_disagree_count,
   *     nli_contradictions, nli_unsupported, recompute_mismatches }
   * See `docs/calibration.md`.
   */
  emitCalibration?: string;
}

export function parseFlags(argv: string[]): ParsedFlags {
  const out: ParsedFlags = { mode: "standard", all: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--data") {
      out.data = argv[++i];
    } else if (a === "--mode") {
      const v = argv[++i];
      if (v !== "standard" && v !== "deep" && v !== "deeper") {
        throw new Error(`--mode must be standard|deep|deeper, got '${v}'`);
      }
      out.mode = v;
    } else if (a === "--limit") {
      out.limit = Number(argv[++i]);
      if (!Number.isFinite(out.limit) || out.limit <= 0) {
        throw new Error("--limit requires a positive integer");
      }
    } else if (a === "--all") {
      out.all = true;
    } else if (a === "--out") {
      out.out = argv[++i];
    } else if (a === "--emit-calibration") {
      out.emitCalibration = argv[++i];
    } else if (a === "--help" || a === "-h") {
      out.all = false; // ignored, just for safety
      printUsage();
      process.exit(0);
    } else {
      throw new Error(`Unknown flag: ${a}`);
    }
  }
  return out;
}

function printUsage(): void {
  console.error(
    "Usage: node scripts/bench-<dataset>.ts --data <path.jsonl> " +
      "[--mode standard|deep|deeper] [--all] [--limit N] [--out <tsv>] " +
      "[--emit-calibration <path.jsonl>]"
  );
}

// ─── Calibration JSONL emit ───────────────────────────────────────────

/**
 * One row written to the calibration JSONL when --emit-calibration is set.
 * The fields are the inputs `calibrate-thresholds.ts` consumes to compute
 * a conformal cut-off; see `docs/calibration.md`.
 */
export interface CalibrationRow {
  /** Stable per-row identifier (same as BenchTrial.id). */
  id: string;
  /** Gold label: 1 = hallucinated, 0 = clean. */
  gold: 0 | 1;
  /** Continuous nonconformity score (same as BenchTrial.score). */
  score: number;
  /** Component counts so a downstream script can re-derive its own score. */
  critic_disagree_count: number;
  nli_contradictions: number;
  nli_unsupported: number;
  recompute_mismatches: number;
}

export function trialToCalibrationRow(
  trial: BenchTrial,
  out: VerifyOutput
): CalibrationRow {
  const contradictions = out.nli_check?.contradictions?.length ?? 0;
  const unsupported = out.nli_check?.unsupported?.length ?? 0;
  const mismatches = out.recompute?.mismatches?.length ?? 0;
  return {
    id: trial.id,
    gold: trial.gold,
    score: trial.score,
    critic_disagree_count: trial.critic_disagree_count,
    nli_contradictions: contradictions,
    nli_unsupported: unsupported,
    recompute_mismatches: mismatches,
  };
}

// ─── Repo-root locator ────────────────────────────────────────────────

export function repoRoot(): string {
  // bench-common.ts lives at <repo>/project/scripts/bench-common.ts;
  // dist/ lives at <repo>/dist/.
  return path.resolve(fileURLToPath(import.meta.url), "..", "..", "..");
}

export function defaultOutPath(dataset: string, mode: VerifyMode): string {
  return path.join(repoRoot(), "bench", `${dataset}-${mode}.tsv`);
}
