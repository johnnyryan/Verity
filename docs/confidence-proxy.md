# Verity confidence proxy

A transparent, OpenAI-compatible front-door that adds Verity's confidence
gate to the answers an external chat client generates, **without** changing
how you use the client or the model server.

- Source: `project/src/proxy/server.ts` (HTTP wiring) and
  `project/src/proxy/translate.ts` (pure translation logic + the request-shape
  guard).
- Tests: `project/src/__tests__/proxy-translate.test.ts`.
- Config: the `PROXY_*` constants in `project/src/config.ts`.

## What it is

The built-in chat UIs (the LM Studio app, `ollama run`) are sealed: Verity
cannot see the tokens they generate, so it cannot score their confidence.
The proxy closes that gap for the **other** way people chat with a local
model: an external OpenAI-compatible client (Open WebUI, Jan, LibreChat,
AnythingLLM, and similar) pointed at the model server's API port.

Instead of pointing such a client straight at LM Studio
(`http://localhost:1234/v1`), you point it at the proxy
(`http://localhost:1235/v1`). The proxy then sits in the middle:

```
   chat client  ->  confidence proxy (1235)  ->  LM Studio (1234)
                         |
                         +-- for a SCORABLE plain-text request: scores the
                             answer's token logprobs and appends a low-
                             confidence note when warranted
                         +-- for everything else: forwards verbatim, untouched
```

## The rule

The default for every request, including `POST /v1/chat/completions`, is
byte-for-byte pass-through. A chat request is diverted only when it is
"scorable", and a scorable request is graded with an exact, free confidence
note. The two halves are explained below.

### Default: byte-for-byte pass-through

Every request the proxy receives is, by default, forwarded to the upstream
exactly as the client sent it: same method, same path and query, the same body
bytes, and the upstream's response **streamed straight back** with its real
status, content type, `finish_reason`, `usage`, errors, and headers. The client
cannot tell it is not talking to LM Studio directly. Streaming responses keep
their true time-to-first-token because the upstream's SSE bytes are piped
through as they arrive, never buffered.

This covers `/v1/models`, `/v1/embeddings`, a direct `/v1/responses` call, any
future endpoint, **and** any `/v1/chat/completions` request that is not
scorable (see the capability matrix). Nothing is rewritten and no note is
added on these paths.

### The exception: a scorable chat request is graded

A chat request is diverted from pass-through **only** when the request-shape
guard (`isScorable` in `translate.ts`) certifies that the upstream's
`/v1/responses` endpoint can serve it with **full fidelity and zero feature
loss**. `/v1/responses` is the only path that returns per-token logprobs on
LM Studio, so a scorable request is routed through it **exactly once**. That
single generation yields **both** the answer text **and** its real per-token
logprobs, the answer is scored with the shared confidence classifier
(`project/src/signals/confidence.ts`, the same code path that `/verify` uses),
and a one-line note is appended when the answer lands in a `mild`, `low`, or
`very_low` band, for example:

> ⚠️ Elevated model uncertainty: weakest token "Zurich" at 4.2%. Possible
> guessing rather than a known error. Consider `/verifydeep`.

Confident answers (`ok` band) are returned unchanged with no extra text.

**The note is exact and free.** Because the proxy itself drove the one
generation that produced the answer, the logprobs belong to *that* answer, not
to a separate regeneration. There is no second probe and no answer/score
mismatch: the confidence note describes the very tokens the user sees, at no
extra generation cost.

## Capability matrix: what gets a note, what passes through

A request is **scorable** (gets a confidence note) only if **all** of the
following hold. If any fails, the request **passes through ungraded**, verbatim,
with no note. When in doubt, the guard treats a request as **not** scorable, so
a feature is never silently dropped.

| Request shape | Scored (note added)? | Why |
| --- | --- | --- |
| Plain-text messages, no exotic params | **Yes** | `/v1/responses` serves it losslessly; logprobs come from the served answer |
| `temperature`, `top_p`, `max_tokens` / `max_completion_tokens` set | **Yes** | these map cleanly to `/v1/responses` (`temperature`, `top_p`, `max_output_tokens`) |
| `stream: true`, plain text | **Yes** (see Streaming) | replayed as SSE with the note as a final delta |
| Image / vision content (`image_url`, `input_image`) | No, passes through | a text-only translation cannot carry images |
| Audio / file content parts | No, passes through | non-text modality |
| `tools` / `tool_choice` / `functions` / `function_call` | No, passes through | `/v1/responses` uses a different tool schema |
| A message carrying `tool_calls` | No, passes through | replaying a tool-call turn as text would drop the call |
| `response_format` / JSON schema / structured output | No, passes through | structured output uses `text.format`, a different shape |
| `n` greater than 1 | No, passes through | `/v1/responses` returns a single response; `n>1` would collapse |
| `stop` / `stop_sequences` | No, passes through | no stop-sequence parameter on `/v1/responses` |
| `seed` | No, passes through | no deterministic-seed parameter |
| `presence_penalty` / `frequency_penalty` | No, passes through | no penalty parameters |
| `logit_bias` | No, passes through | no per-token bias map |
| `top_k` | No, passes through | not part of the OpenAI sampling surface |
| Client-set `logprobs` / `top_logprobs` | No, passes through | would collide with the logprobs request the proxy sets |
| No messages / empty messages / unparseable body | No, passes through | nothing to score; let the upstream handle it |

### Why the guard is this strict (the `/v1/responses` parameter surface)

The OpenAI **Responses API** is a different primitive from Chat Completions,
not a superset. Verified 2026-05-22 against:

- the OpenAI Responses API reference,
  <https://platform.openai.com/docs/api-reference/responses>;
- LM Studio's Open Responses documentation,
  <https://lmstudio.ai/blog/openresponses> and
  <https://lmstudio.ai/docs/developer/openai-compat/responses>;
- the installed `openai` SDK request type `ResponseCreateParamsBase`
  (`node_modules/openai/resources/responses/responses.d.ts`).

The Responses request body's **only** sampling/length knobs are `temperature`,
`top_p`, and `max_output_tokens`. The Chat Completions knobs `stop`, `seed`,
`presence_penalty`, `frequency_penalty`, `logit_bias`, `top_k`, and `n` have
**no** equivalent on `/v1/responses`. Structured output uses `text.format`
(not `response_format`), tool calling uses a different tool schema, and vision
uses `input_image` content parts. A request that sets any Chat-Completions-only
knob, or asks for tools / structured output / vision / `n>1`, therefore cannot
be routed through `/v1/responses` without silently dropping the feature, so the
guard forwards it verbatim instead.

## Streaming

For a **non-scorable** chat request, and for every non-chat request, streaming
is **native**: the upstream's SSE bytes are piped straight to the client as they
arrive, preserving true time-to-first-token. Nothing is buffered.

For a **scorable** plain-text request with `stream: true`, the proxy currently
**buffers then replays**. The confidence band depends on the *whole*
token-logprob set (the single weakest token might be the last one generated),
so the note can only be emitted once the full answer is known. Ideally the
proxy would still stream the answer's text deltas as they arrive and append the
note as a final delta, but that requires `/v1/responses` streaming to deliver
logprobs **incrementally**, attached to each `response.output_text.delta` event.
As of LM Studio's documented Responses streaming (verified 2026-05-22), logprobs
are documented as a property of the **finalised** `output_text` content object,
not of each streaming delta, and the streaming-events documentation does not
specify a per-delta logprobs field. So for this one path the proxy:

1. drives the upstream `/v1/responses` generation to completion (non-streaming
   upstream), obtaining the full text and logprobs;
2. computes the confidence band and the note;
3. replays the answer to the client as Server-Sent Events: a role-priming
   chunk, incremental answer-content deltas, then the confidence note as a
   single **final** content delta (only when the band is not `ok`), then a
   terminal chunk carrying the **real** `finish_reason`, then `[DONE]`.

The client therefore sees a normal streaming response and never knows the bytes
were buffered. The trade-off is that, **for scorable answers only**,
time-to-first-token equals full generation time. This is the correct trade for
the gated subset: you cannot warn about an answer's confidence until you have
seen all of it. If a future LM Studio build delivers per-delta logprobs, this
path can switch to true incremental streaming with no change to the rest of the
design.

## How to start it

1. Build the project (compiles `src` to `dist`):

   ```bash
   npm run build
   ```

2. Make sure LM Studio is running with its local server enabled (Developer
   tab in the LM Studio app) and a model loaded, listening on its usual
   `http://localhost:1234`.

3. Start the proxy:

   ```bash
   npm run proxy
   ```

   You should see:

   ```
   [proxy] confidence proxy listening on http://127.0.0.1:1235/v1
   [proxy] forwarding to upstream http://localhost:1234/v1
   ```

4. Health check:

   ```bash
   curl http://localhost:1235/healthz
   ```

### Configuration

All knobs live in `project/src/config.ts` and read an environment variable, so
nothing needs editing to retune. The ones you are most likely to touch:

| Constant | Env var | Default | Purpose |
| --- | --- | --- | --- |
| `PROXY_PORT` | `VERITY_PROXY_PORT` | `1235` | Port the proxy listens on. |
| `PROXY_HOST` | `VERITY_PROXY_HOST` | `127.0.0.1` | Bind interface. Set to `0.0.0.0` only if a LAN client (e.g. a phone) must reach it, and you understand the threat model: the proxy has no auth. |
| `PROXY_UPSTREAM_URL` | `VERITY_PROXY_UPSTREAM_URL` | `http://localhost:1234/v1` | The OpenAI-compatible backend to forward to. Must expose `/v1/responses` with logprobs for the confidence gate to apply to scorable requests. |
| `PROXY_RESPONSES_TOP_LOGPROBS` | `VERITY_PROXY_RESPONSES_TOP_LOGPROBS` | `1` | Per-token alternatives requested. The classifier only needs the chosen token's logprob. |
| `PROXY_DEFAULT_MAX_OUTPUT_TOKENS` | `VERITY_PROXY_DEFAULT_MAX_OUTPUT_TOKENS` | `2048` | Output cap when a scorable request omits `max_tokens`. |
| `PROXY_UPSTREAM_TIMEOUT_MS` | `VERITY_PROXY_UPSTREAM_TIMEOUT_MS` | `120000` | Per-generation / per-forward wall-clock timeout. |
| `PROXY_MAX_REQUEST_BYTES` | `VERITY_PROXY_MAX_REQUEST_BYTES` | `16 MiB` | Request body size cap (DoS guard). |

The proxy forwards an `Authorization: Bearer <WORKER_API_KEY>` header on every
upstream call. LM Studio ignores it (harmless); a cloud upstream would need a
real key, set via `WORKER_API_KEY` / `WORKER_ENDPOINT` as documented in the
config file.

The confidence band thresholds are the shared `CONFIDENCE_*` constants in the
config file. Tuning them changes the `/verify` perplexity signal and the proxy
together, by design: there is one source of truth for "what counts as low
confidence".

## Pointing a client at the proxy

Every client below has a single "OpenAI-compatible API base URL" setting. Set
it to the proxy's address instead of LM Studio's:

```
http://localhost:1235/v1
```

If a setting asks for an API key, any non-empty string works (the proxy and LM
Studio both ignore it). Pick the model from the client's model list as usual;
the proxy forwards `/v1/models` so the list is the upstream's real one.

- **Open WebUI**: Settings -> Connections -> OpenAI API. Set the **API Base
  URL** to `http://localhost:1235/v1` and any key (e.g. `lm-studio`). If Open
  WebUI runs in Docker, use `http://host.docker.internal:1235/v1` so the
  container can reach the proxy on the host.
- **Jan**: Settings -> Model Providers (or Engines) -> the OpenAI-compatible /
  remote provider. Set the **Base URL / API URL** to `http://localhost:1235/v1`
  and any key.
- **LibreChat**: in your `librechat.yaml` (or the OpenAI custom endpoint
  config), set `baseURL: "http://localhost:1235/v1"` and any `apiKey`. In Docker,
  use `http://host.docker.internal:1235/v1`.
- **AnythingLLM**: LLM Preference -> "Local AI" / "Generic OpenAI". Set the
  **Base URL** to `http://localhost:1235/v1`, any key, and the model id reported
  by `/v1/models`.

Once configured, chat normally. A plain-text answer the model generated with
shaky token confidence arrives with the appended `Consider /verify...` note;
a confident plain-text answer arrives unchanged; and a request that uses tools,
structured output, vision, `n>1`, or an unsupported parameter is forwarded and
streamed exactly as LM Studio would, ungraded.

## Key limitation: the upstream must expose `/v1/responses` with logprobs

The proxy can only enforce the confidence gate on **scorable** requests when the
backend actually returns per-token logprobs from `/v1/responses`. The state of
play (verified 2026-05-22):

- **LM Studio (0.3.x+)** exposes logprobs via its OpenAI **Responses API**
  (`POST /v1/responses` with `include: ["message.output_text.logprobs"]` and
  `top_logprobs: N`). Its `/v1/chat/completions` and `/v1/completions` return
  `logprobs: null` by design. This proxy targets LM Studio-backed setups.
- **Ollama** currently has **no `/v1/responses` equivalent** and thinner
  logprobs exposure, so the proxy has no numbers to read there. With an Ollama
  upstream, scorable requests would fail at the upstream `/v1/responses` step
  (the proxy returns a clear error explaining this). Non-scorable and non-chat
  requests still pass through, so an Ollama upstream remains usable for those
  paths, just ungraded.

If the upstream returns a non-2xx for `/v1/responses` on a scorable request, the
proxy surfaces the upstream status and a message saying an upstream with
logprobs-bearing `/v1/responses` is required. If `/v1/responses` succeeds but
carries no logprobs (some builds), the classifier treats it as "cannot assess"
and the answer is returned unchanged, with no fabricated warning.

## Notes and follow-ups

- Only the gated, scorable, streaming subset is buffered (see "Streaming");
  everything else streams natively. If LM Studio adds per-delta logprobs to
  `/v1/responses` streaming, the scorable streaming path can switch to true
  incremental streaming.
- True answer-perplexity (rescoring the exact bytes of an answer the model
  produced elsewhere) needs an echo-capable side-car and is out of scope here;
  the proxy generates the scorable answer itself, so the logprobs already match.
- The confidence thresholds (`CONFIDENCE_*`) are first-pass and uncalibrated
  against a labelled corpus. Expect to tune them once you have ground-truth
  confident / unconfident examples.
- When other backends (llama.cpp `llama-server`, vLLM, TGI) gain a
  logprobs-bearing Responses-style endpoint, point `PROXY_UPSTREAM_URL` at them
  and the gate applies to scorable requests there too with no code change.
```
