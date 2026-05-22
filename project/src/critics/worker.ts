/**
 * Worker re-sampling client.
 *
 * Calls the worker model (Qwen 3.5 9B by default) running in LM Studio to
 * generate additional samples of the same question — used by the
 * consistency-check signal in deep modes.
 *
 * Separate from the critic clients because:
 *   - It targets the WORKER model, not Critic A
 *   - It uses higher temperature (sampling diversity is the point)
 *   - It does not parse JSON output — it returns raw text
 */

import {
  WORKER_ENDPOINT,
  WORKER_API_KEY,
  WORKER_MODEL_NAME,
  CRITIC_TIMEOUT_MS,
  VERBOSE_LOGGING,
} from "../config.js";
import { getLlmClient } from "../llm/client.js";

export interface WorkerSample {
  text: string;
  latency_ms: number;
  error?: string;
}

/**
 * Generate one fresh sample of an answer to the given question.
 *
 * [ADAPT] If your worker model uses a system prompt that affects answers,
 * you can pass it via the `systemPrompt` parameter. Default is no system
 * prompt — matches the case where the user asked Qwen plainly.
 */
export async function sampleWorker(params: {
  question: string;
  temperature: number;
  maxTokens?: number;
  systemPrompt?: string;
}): Promise<WorkerSample> {
  const start = Date.now();
  // Default 800: this is a re-sample for the consistency signal, not a
  // user-facing answer. It runs N times in parallel during /verifydeep
  // and /verifydeeper, so latency matters more than length. Most
  // re-samples come back well under 600 tokens; 800 leaves headroom
  // for the verbose case without dragging the whole signal out. The
  // worker model's true 8k answer ceiling is not the relevant budget
  // here — over-budget re-samples would only multiply wall-clock with
  // diminishing entropy gains, so cap deliberately below it.
  const { question, temperature, maxTokens = 800, systemPrompt } = params;

  try {
    const abort = new AbortController();
    const timer = setTimeout(() => abort.abort(), CRITIC_TIMEOUT_MS);

    const messages: Array<{ role: "system" | "user"; content: string }> = [];
    if (systemPrompt) messages.push({ role: "system", content: systemPrompt });
    messages.push({ role: "user", content: question });

    const response = await getLlmClient({
      endpoint: WORKER_ENDPOINT,
      apiKey: WORKER_API_KEY,
    }).chat.completions.create(
      {
        model: WORKER_MODEL_NAME,
        messages,
        temperature,
        max_tokens: maxTokens,
        // [ADAPT] top_p and presence_penalty left at defaults; tweak here
        // if you want more or less sample diversity.
      },
      { signal: abort.signal }
    );

    clearTimeout(timer);

    const text = response.choices[0]?.message?.content ?? "";
    if (VERBOSE_LOGGING) {
      console.error(
        `[worker-resample] ${text.length} chars in ${Date.now() - start}ms`
      );
    }
    return { text, latency_ms: Date.now() - start };
  } catch (err) {
    if (VERBOSE_LOGGING) console.error("[worker-resample] error:", err);
    return {
      text: "",
      latency_ms: Date.now() - start,
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

/**
 * Generate N samples in parallel.
 *
 * Note on parallelism: the worker is a single model on a single GPU.
 * LM Studio will queue these requests and serialize them at the inference
 * level — there's no real concurrency. We fire them in parallel only so
 * the HTTP overhead overlaps. Total wall-clock = N * single-sample latency.
 */
export async function sampleWorkerN(params: {
  question: string;
  n: number;
  temperature: number;
  maxTokens?: number;
  systemPrompt?: string;
}): Promise<WorkerSample[]> {
  const promises: Promise<WorkerSample>[] = [];
  for (let i = 0; i < params.n; i++) {
    promises.push(
      sampleWorker({
        question: params.question,
        temperature: params.temperature,
        maxTokens: params.maxTokens,
        systemPrompt: params.systemPrompt,
      })
    );
  }
  return Promise.all(promises);
}
