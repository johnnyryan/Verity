/**
 * Compute conformally-calibrated warn / fail thresholds from a JSONL
 * calibration set produced by the bench harness (--emit-calibration).
 *
 * Reference: Yadkori et al., "To Believe or Not to Believe Your LLM" /
 * "Conformal Abstention", 2024 (arXiv:2405.01563); Quach et al.,
 * "Conformal Language Modeling", 2023 (arXiv:2306.10193).
 *
 * Mechanism. The user picks alpha (target residual error rate). The script
 * computes the (1-alpha)-quantile of nonconformity scores on the
 * calibration set, separately for the warn and fail gates:
 *
 *   - The FAIL cut-off uses the score distribution of CLEAN answers
 *     (gold=0). We want the cut-off such that, under the exchangeability
 *     assumption, the chance of a clean answer scoring at or above it is
 *     at most alpha. That bounds the false-fail rate.
 *
 *   - The WARN cut-off uses the same distribution at a softer alpha
 *     (default 2*alpha) so warn ranges over more rows than fail.
 *
 * The output JSON is consumed by `project/src/aggregator.ts` at startup;
 * see `docs/calibration.md` for the full operational story.
 *
 * Usage:
 *   node scripts/calibrate-thresholds.ts \
 *     --in    bench/calibration.jsonl  \
 *     --alpha 0.1                       \
 *     --out   project/src/calibrated-thresholds.json \
 *     --dataset ragtruth                \
 *     --warn-alpha 0.2
 *
 * Defaults: alpha=0.1, warn-alpha=2*alpha, out=project/src/calibrated-thresholds.json.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

interface CalibrationRow {
  id?: string;
  gold: 0 | 1;
  score: number;
  critic_disagree_count?: number;
  nli_contradictions?: number;
  nli_unsupported?: number;
  recompute_mismatches?: number;
}

interface CalibrationFlags {
  in?: string;
  out?: string;
  alpha: number;
  warnAlpha?: number;
  dataset?: string;
}

function parseFlags(argv: string[]): CalibrationFlags {
  const out: CalibrationFlags = { alpha: 0.1 };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--in") {
      out.in = argv[++i];
    } else if (a === "--out") {
      out.out = argv[++i];
    } else if (a === "--alpha") {
      out.alpha = Number(argv[++i]);
      if (!Number.isFinite(out.alpha) || out.alpha <= 0 || out.alpha >= 1) {
        throw new Error("--alpha must be a number in (0,1)");
      }
    } else if (a === "--warn-alpha") {
      out.warnAlpha = Number(argv[++i]);
      if (
        !Number.isFinite(out.warnAlpha) ||
        out.warnAlpha <= 0 ||
        out.warnAlpha >= 1
      ) {
        throw new Error("--warn-alpha must be a number in (0,1)");
      }
    } else if (a === "--dataset") {
      out.dataset = argv[++i];
    } else if (a === "--help" || a === "-h") {
      printUsage();
      process.exit(0);
    } else {
      throw new Error(`Unknown flag: ${a}`);
    }
  }
  if (!out.in) {
    throw new Error("--in is required (path to calibration JSONL)");
  }
  return out;
}

function printUsage(): void {
  console.error(
    "Usage: node scripts/calibrate-thresholds.ts " +
      "--in <calibration.jsonl> [--alpha 0.1] [--warn-alpha 0.2] " +
      "[--out <path>] [--dataset <name>]"
  );
}

function readJsonl(filePath: string): CalibrationRow[] {
  const data = fs.readFileSync(filePath, "utf8");
  const rows: CalibrationRow[] = [];
  for (const raw of data.split(/\r?\n/)) {
    const line = raw.trim();
    if (line.length === 0) continue;
    try {
      const obj = JSON.parse(line) as CalibrationRow;
      if (typeof obj.score === "number" && (obj.gold === 0 || obj.gold === 1)) {
        rows.push(obj);
      }
    } catch (err) {
      console.error(`[calibrate] skipped malformed JSONL line: ${err}`);
    }
  }
  return rows;
}

/**
 * (1-alpha)-quantile of `values` using linear interpolation between
 * order statistics — the same convention numpy.quantile uses with
 * `interpolation="linear"`. Returns the largest finite value when alpha
 * is small enough that the conformal-correct rank lands beyond the last
 * order statistic.
 */
function quantile(values: number[], probability: number): number {
  if (values.length === 0) {
    throw new Error("cannot compute a quantile of an empty distribution");
  }
  const sorted = [...values].sort((a, b) => a - b);
  if (sorted.length === 1) return sorted[0]!;
  // Conformal convention: rank index = ceil((n+1)(1-alpha)) - 1 with
  // standard inflation. We use the simpler linear-interp form for
  // readability; the inflation is achieved by selecting min(ceil(rank), n).
  const rank = (sorted.length + 1) * probability - 1;
  const lo = Math.floor(rank);
  const hi = Math.min(Math.ceil(rank), sorted.length - 1);
  if (lo === hi) return sorted[Math.max(0, lo)]!;
  if (lo < 0) return sorted[0]!;
  const frac = rank - lo;
  return sorted[lo]! * (1 - frac) + sorted[hi]! * frac;
}

function computeThresholds(
  rows: CalibrationRow[],
  alpha: number,
  warnAlpha: number
): { fail: number; warn: number; cleanCount: number } {
  const cleanScores = rows.filter((r) => r.gold === 0).map((r) => r.score);
  if (cleanScores.length === 0) {
    throw new Error(
      "calibration set contains no clean (gold=0) examples; cannot compute conformal cut-offs"
    );
  }
  const failCut = quantile(cleanScores, 1 - alpha);
  const warnCut = quantile(cleanScores, 1 - warnAlpha);
  return { fail: failCut, warn: warnCut, cleanCount: cleanScores.length };
}

function defaultOutPath(): string {
  // Default to project/src/calibrated-thresholds.json so the aggregator
  // picks it up at next process boot.
  const here = path.dirname(fileURLToPath(import.meta.url));
  return path.resolve(here, "..", "src", "calibrated-thresholds.json");
}

function main(): void {
  const flags = parseFlags(process.argv.slice(2));
  const alpha = flags.alpha;
  const warnAlpha = flags.warnAlpha ?? Math.min(0.99, alpha * 2);
  if (warnAlpha < alpha) {
    throw new Error(
      `--warn-alpha (${warnAlpha}) must be >= --alpha (${alpha}); ` +
        `warn is the softer gate so its 1-alpha is smaller`
    );
  }

  const rows = readJsonl(flags.in!);
  if (rows.length === 0) {
    console.error(`[calibrate] no usable rows in ${flags.in}`);
    process.exit(2);
  }

  const cuts = computeThresholds(rows, alpha, warnAlpha);

  const output = {
    alpha,
    warn_alpha: warnAlpha,
    calibration_set_size: rows.length,
    clean_set_size: cuts.cleanCount,
    warn_score_threshold: Number(cuts.warn.toFixed(6)),
    fail_score_threshold: Number(cuts.fail.toFixed(6)),
    calibrated_at: new Date().toISOString(),
    dataset: flags.dataset ?? path.basename(flags.in!),
    source_jsonl: path.resolve(flags.in!),
  };

  const outPath = flags.out ? path.resolve(flags.out) : defaultOutPath();
  ensureDir(path.dirname(outPath));
  fs.writeFileSync(outPath, JSON.stringify(output, null, 2) + "\n", "utf8");
  console.error(`[calibrate] wrote ${outPath}`);
  console.error(`[calibrate] alpha=${alpha} warn_alpha=${warnAlpha}`);
  console.error(
    `[calibrate] warn=${output.warn_score_threshold} fail=${output.fail_score_threshold} ` +
      `(over ${cuts.cleanCount} clean rows out of ${rows.length} total)`
  );
}

function ensureDir(dir: string): void {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

main();
