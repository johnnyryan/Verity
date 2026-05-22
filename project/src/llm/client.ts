/**
 * Shared OpenAI-compatible LLM client factory.
 *
 * Replaces five independent client-allocation strategies that had
 * accumulated across the codebase:
 *   - critics/call-critic.ts   (per-endpoint cache, scoped to that file)
 *   - second-opinion/consult.ts (per-endpoint cache, scoped to that file)
 *   - critics/worker.ts        (module-level singleton, hard-coded URL)
 *   - nli/extract-claims-llm.ts (module-level singleton, hard-coded URL)
 *   - signals/perplexity.ts    (module-level singleton, hard-coded URL)
 *
 * The three module-level singletons captured `LM_STUDIO_URL` at import
 * time, making any later env-var change invisible. Going through this
 * factory means the URL is read fresh on each call (or, more precisely,
 * the URL is whatever the caller supplies — typically a config constant
 * that is itself read at import, but tests can now point a single env
 * var at a mock and it'll be honoured by every call site uniformly).
 *
 * Cache semantics: identical to the old per-file caches. Two calls
 * with the same `endpoint` + `apiKey` get the same OpenAI instance.
 * The cache lives for the process lifetime; there is no eviction.
 * That's fine — there are at most a handful of distinct endpoints in
 * play (LM Studio, Ollama, occasionally per-critic overrides) and the
 * client objects are cheap.
 */

import OpenAI from "openai";
import { createHash } from "node:crypto";

const clientCache = new Map<string, OpenAI>();

/**
 * Get (or lazily allocate) an OpenAI client for the given endpoint.
 *
 * Calls with the same `endpoint` + `apiKey` share an instance.
 *
 * 2026-05-12 (F2): the cache key used to be the literal
 * `${endpoint}|${apiKey}`, which meant the API key sat in the Map's
 * key space and would surface verbatim in any memory dump or core
 * file. Hash the key so the cache still works (same input → same
 * cache key → same client instance) but the credential itself never
 * appears in the key. SHA-256 of a short string is fast and
 * collision-free for this use.
 */
export function getLlmClient(opts: {
  endpoint: string;
  apiKey: string;
}): OpenAI {
  const keyHash = createHash("sha256")
    .update(`${opts.endpoint}|${opts.apiKey}`)
    .digest("hex");
  let c = clientCache.get(keyHash);
  if (!c) {
    c = new OpenAI({ baseURL: opts.endpoint, apiKey: opts.apiKey });
    clientCache.set(keyHash, c);
  }
  return c;
}
