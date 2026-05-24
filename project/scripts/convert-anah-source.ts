/**
 * convert-anah-source.ts
 *
 * One-time converter from the ANAH source dataset (huggingface.co/datasets/opencompass/anah,
 * topic-grouped jsonl) into the flat per-sentence jsonl shape that bench-anah.ts already eats.
 *
 * Source layout per row:
 *   {
 *     "name": "topic name",
 *     "documents": [doc, ...],
 *     "selected_questions": [q1, q2, q3],
 *     "GPT3.5_answers_D": [a1, a2, a3],
 *     "InternLM_answers":  [a1, a2, a3],
 *     "human_GPT3.5_answers_D_ann": [
 *       [ "<要点>...\n<幻觉>无\n<参考>...\n<改正>...", ... ],
 *       ...
 *     ],
 *     "human_InternLM_answers_ann": [...],
 *     "language": "zh" | "en"
 *   }
 *
 * Each annotation entry is a list of tagged strings, one per response sentence.
 * The hallucination value sits inside the <幻觉> ... </幻觉?> block (no closing tag in source;
 * the value is the text up to the next newline). Mapping to the English label vocabulary:
 *
 *   无         -> ok
 *   矛盾       -> contradictory
 *   无法验证   -> unverifiable
 *   无事实     -> nofact
 *
 * Output line shape (what bench-anah.ts expects):
 *   {"question": ..., "response": ..., "sentence": ..., "annotation": "ok|contradictory|...", "language": "zh|en"}
 *
 * Usage:
 *   node project/scripts/convert-anah-source.ts \
 *     --input  C:/AI/verify-data/anah \
 *     --output C:/AI/verify-data/anah/anah-flat.jsonl
 *
 * --input may be a directory of *.jsonl topic files (event/location/person/thing.jsonl)
 * or a single jsonl file.
 */

import * as fs from "node:fs";
import * as path from "node:path";

interface Flags {
  input: string;
  output: string;
}

function parseFlags(argv: string[]): Flags {
  const f: Partial<Flags> = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--input") f.input = argv[++i];
    else if (a === "--output") f.output = argv[++i];
  }
  if (!f.input || !f.output) {
    console.error(
      "Usage: convert-anah-source.ts --input <path> --output <path>",
    );
    process.exit(2);
  }
  return f as Flags;
}

const LABEL_MAP: Record<string, string> = {
  // Chinese
  "无": "ok",
  "矛盾": "contradictory",
  "无法验证": "unverifiable",
  "无事实": "nofact",
  // English (in case the en split uses these directly)
  "ok": "ok",
  "none": "ok",
  "contradictory": "contradictory",
  "unverifiable": "unverifiable",
  "nofact": "nofact",
};

function extractTag(block: string, tag: string): string | null {
  // ANAH tags are <要点>...<幻觉>...<参考>...<改正>...
  // No closing tags; each value runs to the next newline or next <tag>.
  // ASSUMPTION: tag values do not themselves contain `<` (true for the
  // hallucination-label values `无` / `矛盾` / `无法验证` / `无事实` which
  // are the only ones the harness reads). For other tags like `<参考>`
  // whose value may include `<SEP>` markers, the regex truncates at the
  // first `<` — fine for our use (we discard `<参考>` and `<改正>` values)
  // but anyone reusing this function on other fields should widen.
  const re = new RegExp(`<${tag}>([^<\\n]*)`, "u");
  const m = block.match(re);
  return m ? m[1].trim() : null;
}

function mapAnnotation(raw: string): string | null {
  const norm = raw.trim();
  if (norm in LABEL_MAP) return LABEL_MAP[norm];
  // Heuristic fallback for minor punctuation drift.
  const stripped = norm.replace(/[。.，,!?！？]/gu, "");
  if (stripped in LABEL_MAP) return LABEL_MAP[stripped];
  return null;
}

function* readJsonlLines(filePath: string): Generator<unknown> {
  const text = fs.readFileSync(filePath, "utf8");
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      yield JSON.parse(trimmed);
    } catch {
      // skip malformed line
    }
  }
}

function listSourceFiles(input: string): string[] {
  const stat = fs.statSync(input);
  if (stat.isFile()) return [input];
  return fs
    .readdirSync(input)
    .filter((f) => f.endsWith(".jsonl"))
    .map((f) => path.join(input, f));
}

interface FlatRow {
  question: string;
  response: string;
  sentence: string;
  annotation: string;
  language: string;
}

function flattenTopic(topic: any): FlatRow[] {
  const rows: FlatRow[] = [];
  const language = topic.language ?? "en";
  const questions: string[] = topic.selected_questions ?? [];

  const modelChannels = [
    {
      answers: topic["GPT3.5_answers_D"] as string[] | undefined,
      anns: topic["human_GPT3.5_answers_D_ann"] as string[][] | undefined,
    },
    {
      answers: topic["InternLM_answers"] as string[] | undefined,
      anns: topic["human_InternLM_answers_ann"] as string[][] | undefined,
    },
  ];

  for (const channel of modelChannels) {
    if (!channel.answers || !channel.anns) continue;
    const n = Math.min(
      questions.length,
      channel.answers.length,
      channel.anns.length,
    );
    for (let i = 0; i < n; i++) {
      const question = questions[i];
      const response = channel.answers[i];
      const sentenceBlocks = channel.anns[i] ?? [];
      for (const block of sentenceBlocks) {
        if (typeof block !== "string") continue;
        const sentence = extractTag(block, "要点");
        const labelRaw = extractTag(block, "幻觉");
        if (!sentence || !labelRaw) continue;
        const annotation = mapAnnotation(labelRaw);
        if (!annotation) continue;
        rows.push({ question, response, sentence, annotation, language });
      }
    }
  }
  return rows;
}

function main(): void {
  const flags = parseFlags(process.argv.slice(2));
  const inputFiles = listSourceFiles(flags.input);
  const out = fs.createWriteStream(flags.output, { encoding: "utf8" });

  let topicCount = 0;
  let rowCount = 0;
  let labelHist: Record<string, number> = {};

  for (const file of inputFiles) {
    for (const topic of readJsonlLines(file)) {
      topicCount++;
      const rows = flattenTopic(topic as any);
      for (const row of rows) {
        out.write(JSON.stringify(row) + "\n");
        rowCount++;
        labelHist[row.annotation] = (labelHist[row.annotation] ?? 0) + 1;
      }
    }
  }
  out.end();

  console.log(
    `Read ${topicCount} topic rows from ${inputFiles.length} file(s).`,
  );
  console.log(`Emitted ${rowCount} per-sentence rows -> ${flags.output}`);
  console.log(`Label distribution:`, labelHist);
}

main();
