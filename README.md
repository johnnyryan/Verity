# Verity MCP

LLMs confidently claim things that are manifestly untrue. [Enforce](https://iccl.ie/enforce/) has developed Verity, a tool that helps minimise false claims and fake sources from self-hosted LLMs. It can run on cheap, old hardware. We think it is the first MCP that combines cross-family LLM critics, NLI, deterministic arithmetic recompute, consistency sampling, perplexity, and identifies disputes among these many critics. 

Verity can also produce second opinions. If you have a spare old graphics card Verity can use it to produce second opinions at the same time that your primary LLMs responds. Both answers are then considered by your primary LLM. Once adapted for your hardware, you can easily use Verity in LM Studio. We are also sharing our system prompts, which help minimise LLM mistakes even without Verity. 



***The initial setup assumes a Nvidia 5070ti GPU (2025) and and spare AMD 5700xt (2019) on a 2021 Wintel. This is the reference machine. The design is complicated by the mix of GPUs. The end of this doc describes how to adapt Verity for your own self-hosted LLM setup (single GPU, multi GPU, Apple Silicon, etc).*** 


---

## Basic commands: 

Adding "/verify" at the end of an LM Studio query will check answers from one LLM against different LLMs and an NLI that have been trained in a different way, and have different blind spots. 

"/verifydeep" also repeats the same question x2 (deep) or x5 (deeper) times at higer temperature and checks whether the original claims survive across re-samples. "/verifydeeper" adds scoring of answer tokens, exposing low confidence predictions.

"/second" produces a second opinion and presents differences between LLMs. More commands below. 

---

## How it works at a glance

When you type `/verify`, four things happen at once across two GPUs and the CPU. The worker (the LLM you chat with on your strong GPU) is instructed to strictly source all facts. It will now (or should now) claim only what it can validate with a working URL source. No more made up sources, and far fewer made up facts. Then it hands its last answer to a small Node.js process — the MCP server. That server fans the answer out to two critics (smaller LLMs on the older GPU that re-read the answer with fresh eyes), an NLI claim-checker (a small specialised classifier on the CPU that flags factual contradictions), and a deterministic recompute pass (a non-LLM CPU check that catches arithmetic mistakes). Their findings get aggregated into a single pass / warn / fail verdict and pasted back into the chat. Two of those checks are not LLMs — recompute is plain code, NLI is a 0.4 B-parameter encoder transformer that outputs three numbers per claim. 



---

## Layers of protection against wrongness

**No single check is reliable. The point is that their failure modes don't overlap.** Two LLMs trained on similar data tend to be wrong about the same things — when they agree, they often agree wrong. The NLI classifier was trained on entailment-labelled data instead of helpfulness preferences, so its mistakes look completely different from a chat model's. The recompute pass doesn't have a bias profile at all, because it isn't statistical. When two layers built on different machinery flag the same thing, that is much stronger evidence than any single LLM saying "are you sure?".

                              WORKER ANSWER
                       (the LLM you chat with —
                        Qwen 3.5 9B on the strong GPU)
                             
   ┌────────────────────────────────────────────────────────────────────┐
   │ LAYER 1 · Critic A — IBM Granite 8B (LLM)                          │
   |   How       : different training family from the worker            │
   │   Catches   : subtle code bugs, logic flaws, citation errors       │
   │   Blind to  : mistakes the worker's training data also contains    │
   ├────────────────────────────────────────────────────────────────────┤
   │ LAYER 2 · Critic B — IBM Granite 2B (LLM, distinct corpus)         │
   │   How       : same vendor as A, smaller, faster, different scratch │
   │   Catches   : simple errors, quick confirmations, second voice     │
   │   Blind to  : subtle bugs the 8B sees                              │
   ├────────────────────────────────────────────────────────────────────┤
   │ LAYER 3 · NLI claim-checker — DeBERTa-v3-large (NOT an LLM)        │
   │   How       : encoder transformer, trained on entailment labels    │
   │   Catches   : factual contradictions of supplied prior_context     │
   │   Blind to  : anything without a premise; multi-step reasoning     │
   ├────────────────────────────────────────────────────────────────────┤
   │ LAYER 4 · Recompute pass — regex + arithmetic (NOT an LLM)         │
   │   How       : pure code; deterministic; no model uncertainty       │
   │   Catches   : arithmetic errors, unit conversions — 100% precision │
   │   Blind to  : anything that isn't a closed-form numeric expression │
   ├────────────────────────────────────────────────────────────────────┤
   │   /verifydeep and /verifydeeper add two more layers:               │
   ├────────────────────────────────────────────────────────────────────┤
   │ LAYER 5 · Consistency — re-sample the worker N times               │
   │   How       : ask the same question 2 (deep) or 5 (deeper) times   │
   │   Catches   : low-confidence guessing — answers that flicker       │
   │   Blind to  : consistent overconfidence (always wrong the same way)│
   ├────────────────────────────────────────────────────────────────────┤
   │ LAYER 6 · Perplexity — worker's own logprob entropy                │
   │   How       : score the answer's tokens, flag low-confidence spans │
   │   Catches   : tokens the worker was hesitant about                 │
   │   Blind to  : confident hallucinations                             │
   └────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                   AGGREGATOR — combines layers into
                       PASS / WARN / FAIL / ERROR


The layers are deliberately built on different machinery — a 14 B-class LLM, an 8 B-class LLM from the same family but different scratch corpus, a 0.4 B encoder transformer trained on a different objective, a regex evaluator, a stochastic re-sampler, and a logprob analyser. Six different *kinds* of "wrong" are caught by six different *kinds* of check.

A separate **disputes table** is computed *after* the consensus is decided. It surfaces concerns one critic raised but not the other, so users see disagreement even when the headline verdict is "pass".

---

## The pieces

The verity loads four models at the same time. None of them swap during normal operation; everything stays warm.

### Live lineup

| Role       | Model                                | Family   | Params | Quant  | VRAM    | Device              |
| ---------- | ------------------------------------ | -------- | ------ | ------ | ------- | ------------------- |
| Worker     | Qwen 3.5 9B                          | Alibaba  | 9 B    | Q4_K_M | ~5.5 GB | 5070 Ti / LM Studio |
| Critic A   | IBM Granite 3.2 8B                   | IBM      | 8 B    | Q4_K_M | ~4 GB   | 5700 XT / Ollama    |
| Critic B   | IBM Granite 3.2 2B                   | IBM      | 2 B    | Q4_K_M | ~1.8 GB | 5700 XT / Ollama    |
| NLI check  | DeBERTa-v3-large (cross-encoder)     | Microsoft| 0.4 B  | ONNX   | ~1 GB   | CPU                 |


The reference 
The 5070 Ti only hosts the worker (~5.5 GB of ~16 GB used). The 5700 XT hosts both critics (~5.8 GB of 8 GB, leaving ~2 GB for KV cache).

---

## Quick start (Windows)

You need: Node.js 18+, [LM Studio](https://lmstudio.ai/) 0.3.x or newer with MCP client support running on port 1234, and [Ollama](https://ollama.com/) (Vulkan build for AMD GPUs) installed.

One-time setup (assuming reference machine):

```powershell
# Pull the critics into Ollama
ollama pull granite3.2:8b
ollama pull granite3.2:2b

# Build the Verity MCP server
cd C:\AI\verify
npm install
npm run build

# Drop the two desktop icons for daily use
.\start-verity.ps1 -InstallShortcut       # Start icon (PowerShell-blue)
.\start-verity.ps1 -InstallStopShortcut   # Stop icon (red stop sign)
```


```

Register the MCP server with LM Studio once (Settings → Model Context Protocol):

```json
{
  "mcpServers": {
    "verity": {
      "url": "http://localhost:8090/mcp",
      "timeout": 240000,
      "retries": 1
    }
  }
}
```

Then paste the worker's system prompt (see "Setup in detail" below) into your chat-model preset in LM Studio. That tells the worker to call `verify_answer` when you type `/verify`.

Then chat normally and append `/verify` to any message you want re-checked.

## Suggested LM Studio settings (for reference hardware and models) ##

Model settings: 
Unified KV cache on, K and V cache quant set to Q8, max CPU threads, max practical GPU offload. Offload KV cache to GPU on. Try mmap() on. Keep model in memory on. Flash attention on.

Inference settings: 
Top K sampling: 40. Temperature: 1. Presence penalty: 1.5. Top P sampling: .95. 


## Starting and stopping 

Daily use on referene machine: Run `start-verity.ps1`, which pins Ollama to the AMD card (critical -- see "Setup in detail" for why), starts the Verity MCP server on `:8090`, health-checks both, and leaves a PowerShell window open showing the status. Close the window when ready; both services keep running in the background.

To unload everything: Run 'C:\AI\verify\start-verity.ps1 -Action Stop' to stop the Verity MCP server (which releases the DeBERTa NLI model and tiktoken from CPU RAM), kills `ollama serve` and its child runners (which releases Granite 3.2 8B + 2B from AMD VRAM), then verifies that no Verity-related process remains on either GPU or in CPU memory. It reports each release with a `[OK]` line. LM Studio + any CUDA workloads on NVIDIA are untouched.


---

## Commands

After getting an answer from the worker, type one of these:

### Depth modes

| Command         | Mode     | What runs                                                  | Time     |
| --------------- | -------- | ---------------------------------------------------------- | -------- |
| `/verify`       | standard | 2 critics + NLI + recompute                                | ~3–5 s   |
| `/verifydeep`   | deep     | standard + 2-sample consistency + perplexity rescore       | ~20 s    |
| `/verifydeeper` | deeper   | standard + 5-sample consistency + perplexity (regen FB)    | ~40 s    |

### Context modes

| Command                        | Effect                                              |
| ------------------------------ | --------------------------------------------------- |
| `/verify`                      | Minimal context (question + answer only)            |
| `/verify with context`         | Worker includes relevant prior messages             |
| `/verify full`                 | Pass entire conversation history                    |

### Other modifiers (stack as needed)

| Command                  | Effect                                              |
| ------------------------ | --------------------------------------------------- |
| `/verify no-nli`         | Skip the NLI claim check                            |
| `/verify as code`        | Force task_type=code                                |
| `/verify as prose`       | Force task_type=prose                               |
| `/verify as reasoning`   | Force task_type=reasoning                           |
| `///VERIFY///`           | Distinctive trigger if `/verify` is misread as text |

Modifiers stack: `/verifydeeper as code no-nli with context` is valid. The verdict appears as a new assistant message; your original answer is not modified.

### Second-opinion command

| Command            | Effect                                                                  |
| ------------------ | ----------------------------------------------------------------------- |
| `/second`          | Two cross-family models answer the question in parallel; a third model compares them |
| `/verify /second`  | Combined: `/second` first, then `/verify`                               |

---

### How the signals combine (aggregator rules)

The aggregator applies fixed rules to combine all signals into a single consensus:

```
recompute mismatch:                                     fail
any critic.severity >= 3 or NLI contradicts:            fail
consistency divergence >= 0.5 (deep modes only):        fail
any critic.severity >= 2 or NLI unsupported (>=2):      warn
consistency divergence >= 0.15 or perplexity flagged:   warn
else:                                                    pass
```

When recompute verified an arithmetic claim, NLI contradictions whose claim text contains that expression are dropped (handles the `math-subtle` false-positive).

A separate disputes table is computed after the consensus is decided. It surfaces concerns raised by one critic but not the other (token-Jaccard fuzzy match) plus verdict mismatches. The user always sees disagreement even when the headline verdict is "pass". Disputes never change the verdict — they're a pure diagnostic.

---

## Architecture (using reference machine) 

A picture of where things run. The strong GPU hosts only the worker; both critics share the older GPU; the NLI classifier runs on the CPU (no GPU dependency). A small Node.js MCP server orchestrates them and exposes a single HTTP endpoint to LM Studio.

```
    ┌────────────────────── 5070 Ti (CUDA, 16 GB) ─────────────────────────┐
    │  LM Studio server :1234                                              │
    │  └─ Worker:    Qwen 3.5 9B Q4           (~5.5 GB, always resident)   │
    │      (LM Studio Just-In-Time loading enabled)                        │
    │  Plenty of headroom: Phi-4 14B used to live here too in v1; under    │
    │  current 2-critic design the strong card hosts only the worker.      │
    └──────────────────────────────────────────────────────────────────────┘
    ┌────────────────────── 5700 XT (Vulkan, 8 GB) ────────────────────────┐
    │  Ollama server :11434  (Vulkan build, OLLAMA_MAX_LOADED_MODELS=2)    │
    │  ├─ Critic A:  IBM Granite 3.2 8B Q4   (~4 GB, always resident)      │
    │  └─ Critic B:  IBM Granite 3.2 2B Q4   (~1.8 GB, always resident)    │
    │  ~5.8 GB of 8 GB used; ~2 GB free for KV cache + workspace.          │
    └──────────────────────────────────────────────────────────────────────┘
    ┌────────────────────────── CPU (ONNX Runtime) ────────────────────────┐
    │  NLI classifier: DeBERTa-v3-large (cross-encoder NLI variant)        │
    │  Loaded in-process by the MCP verity server via                    │
    │  @huggingface/transformers (JS port, ONNX backend).                  │
    │  Warmed on server boot via warmupClassifier() so the first /verify   │
    │  doesn't pay the ~1 GB cold-load.                                    │
    └──────────────────────────────────────────────────────────────────────┘
    ┌────────────────── MCP Verity Server (Node.js) ─────────────────────┐
    │  Listens on :8090/mcp                                                │
    │  Registered with LM Studio's MCP client                              │
    │  Two tools:                                                          │
    │    • verify_answer  (post-hoc audit, /verify)               │
    │    • consult_second_opinion  (pre-final consult, /second)            │
    └──────────────────────────────────────────────────────────────────────┘
```

### Request process

Step by step, here is what happens after you type `/verify`:

```
1. User types "/verify" (or "/verify with context") in LM Studio chat.
2. The worker calls the verify_answer MCP tool with the
   preceding question, the previous answer, and optionally the
   selected prior context.
3. The MCP verity server runs the pipeline:
   a. Start critic calls concurrently:
      - HTTP POST to Ollama :11434 → IBM Granite 3.2 8B (Critic A)
      - HTTP POST to Ollama :11434 → IBM Granite 3.2 2B (Critic B)
      (Critics serialize at the hardware level on the 5700 XT,
       but both fire from the orchestrator without waiting.)
   b. In parallel, run claim extraction + NLI classification on the
      worker's answer (unless use_nli=false).
   c. In parallel, run the deterministic recompute pass.
4. Each critic returns a structured verdict:
      { verdict: pass|warn|fail, severity: 0-5, concerns: [...],
        suggested_fixes: [...] }
   Failed critics are marked unavailable. The pipeline continues with
   the surviving critic; consensus="error" only when MORE than
   MAX_UNAVAILABLE_CRITICS critics fail.
5. The aggregator applies its rules (see "How the signals combine"
   above) and computes the disputes table.
6. The tool returns a JSON blob with the critic verdicts, NLI result,
   recompute / consistency / perplexity results, disputes, consensus,
   and a pre-rendered summary_md the worker pastes verbatim.
7. The worker pastes summary_md into the chat.
```

### Expected latency

Standard mode is bottlenecked by the slower critic; deep modes are bottlenecked by worker re-sampling. Critics A and B serialise on the AMD card — there's only one Ollama process and one GPU, so wall-clock for the two-critic stack is roughly Critic A + Critic B back-to-back.

| Stage                                       | Time      |
| ------------------------------------------- | --------- |
| Worker generation (already done)            | n/a       |
| Critic A (Granite 3.2 8B, 5700 XT)          | ~1.5–3 s  |
| Critic B (Granite 3.2 2B, 5700 XT)          | ~0.5–1 s  |
| NLI check (parallel with critics, CPU)      | ~0.5–2 s  |
| Recompute pass (CPU, no LLM)                | < 0.1 s   |
| **Total wall-clock (standard mode)**        | **~3–5 s** |

Deep / deeper modes add consistency re-sampling (worker × N) and a perplexity rescore, both fired in parallel — wall-clock is dominated by the worker re-samples (~16 s for `deep`, ~30–40 s for `deeper`).

### Context handling

The worker can run at up to 64 k context. The critics don't need that much — passing the entire conversation to a critic adds distractors, not signal (Chen et al., 2024 — see "Why this design"). Three modes manage how much gets sent:

- **`/verify`** (default, `context_mode: "minimal"`) — pass only `question + answer + critic_prompt`. Best for code review, math, self-contained prose, anything the answer should stand on its own. Typical input: 2–8 k tokens.
- **`/verify with context`** (`context_mode: "with_context"`) — the worker passes a selected `prior_context` string with the earlier messages the answer depends on (documents, specifications, data, constraints). Worker's system prompt asks it to keep this under ~24 k tokens. Typical input: 10–30 k tokens.
- **`/verify full`** (`context_mode: "full"`) — pass the entire visible conversation. May overflow a critic's context limit; the pipeline truncates from the head if so and reports the truncation in the verdict.

---

## Tools exposed to LM Studio (MCP)

Verity exposes two tools over the standard [Model Context Protocol](https://modelcontextprotocol.io/) — JSON-RPC over HTTP. Any MCP-aware host can use them, not just LM Studio. Each tool's input schema below is what the worker registers with the host.

### Tool 1: `verify_answer`

Post-hoc audit of the worker's most recent answer. The original `/verify` flow.

```typescript
{
  name: "verify_answer",
  description:
    "Run a multi-agent verification pipeline on the most recent question " +
    "and answer. Returns a structured verdict from several critic models " +
    "plus an NLI-based factual-claim check.",
  inputSchema: {
    question: {
      type: "string", required: true,
      description: "The user's most recent question."
    },
    answer: {
      type: "string", required: true,
      description: "Your most recent answer to that question."
    },
    task_type: {
      type: "string", required: false, default: "auto",
      enum: ["code", "prose", "reasoning", "research", "auto"],
      description: "What kind of output to verify. Affects which critic " +
                   "prompts are used. 'auto' lets the pipeline detect it."
    },
    context_mode: {
      type: "string", required: false, default: "minimal",
      enum: ["minimal", "with_context", "full"],
      description: "How much conversation context to include."
    },
    prior_context: {
      type: "string", required: false,
      description: "When context_mode is 'with_context', include earlier " +
                   "messages containing documents, specifications, data, " +
                   "or constraints relevant to the answer. Keep under " +
                   "24 k tokens. Omit small talk."
    },
    use_nli: {
      type: "boolean", required: false, default: true,
      description: "Run the NLI claim checker. Default true."
    }
  }
}
```

Example output (abbreviated):

```json
{
  "critics": {
    "granite_3_2_8b": {
      "verdict":  "warn",
      "severity": 2,
      "concerns": [ "Rate calculation uses annual_rate/12 ..." ],
      "suggested_fixes": [ "Use (1+r)^(1/12)-1 for monthly compounding" ]
    },
    "granite_3_2_2b": {
      "verdict":  "pass",
      "severity": 0,
      "concerns": [],
      "suggested_fixes": []
    }
  },
  "disputes": [
    {
      "kind":        "concern-only-in-a",
      "critic_a_id": "granite_3_2_8b",
      "critic_b_id": "granite_3_2_2b",
      "severity":    "soft"
    }
  ],
  "nli_check":         { "ran": true,  "claims_checked": 0,
                         "contradictions": [], "unsupported": [] },
  "recompute":         { "ran": true,  "expressions_found": 2,
                         "verifications": [{"expr_text":"0.05/12","matches":true}],
                         "mismatches": [] },
  "consistency_check": null,
  "perplexity":        null,
  "consensus":         "warn",
  "summary":           "One reviewer flagged an arithmetic approximation.",
  "summary_md":        "**Verdict: warn** (consensus)\n\n…",
  "latency_ms":        4843,
  "meta": { "mode": "standard", "task_type": "code",
            "context_mode": "minimal", "critics_unavailable": [] }
}
```

**Wire-id history.** Through 2026-05-11 the JSON keys were `phi4_reasoning` and `nemotron_mini` — chosen back when v1 actually ran Phi-4-mini and Nemotron Mini. Both models were swapped to Granite, but the keys lingered. On 2026-05-11 they were renamed to `granite_3_2_8b` and `granite_3_2_2b` so the wire id matches the actual model. If you have older spot-check scripts or archived output, they will reference the legacy keys — those are not breaking errors, just a name change.

### Tool 2: `consult_second_opinion`

Pre-final-answer consultation. The worker calls this *before* committing to a non-trivial answer, to get an independent take from a different-family model on the secondary GPU. Different from `/verify` in two ways:

1. The worker may not have answered yet — it can call this during its own reasoning to get a parallel sanity check.
2. Two cross-family models answer in parallel (one per GPU); a third analysis call compares them and emits a structured agreements / disputes table.

Trigger phrases: `/second`, "second opinion", "ask the other model", or the combined `/verify /second` (in which case `/second` runs first).

```typescript
{
  name: "consult_second_opinion",
  description:
    "Consult two independent cross-family models (one per GPU) in " +
    "parallel for a second opinion on the user's question, then run an " +
    "analysis pass on NVIDIA comparing the two answers.",
  inputSchema: {
    question:        { type: "string",  required: true  },
    worker_draft:    { type: "string",  required: false,
                       description: "Optional in-progress draft; enables " +
                                    "a rough agreement score." },
    prior_context:   { type: "string",  required: false },
    model:           { type: "string",  required: false,
                       description: "Optional Ollama tag override; if set, " +
                                    "forces single-Ollama legacy path." },
    resolution_mode: { type: "string",  enum: ["manual", "auto"],
                       default: "manual" }
  }
}
```

Output: `{ second_opinion, model, dual_opinion, disputes, analysis: { agreements, disputes, table_html, table_md, final_answer? } }`. The analysis pass runs on NVIDIA (default = the worker model) so the AMD Ollama leg can keep generating in parallel. Set `CONSULT_DUAL=0` to revert to the single-Ollama legacy path on hardware where the dual-GPU split isn't available.

---

## Hardware target

The reference machine is an old cheap PC, which has a 2025 NVIDIA card paired with a 2019 AMD card. 

| Component             | Role                                          |
| --------------------- | --------------------------------------------- |
| NVIDIA RTX 5070 Ti    | 16 GB VRAM, CUDA — worker (and v1 Critic A)   |
| AMD Radeon RX 5700 XT | 8 GB VRAM, Vulkan — both Granite critics       |
| Intel UHD 770 iGPU    | Shared system RAM — was planned for NLI       |
| Intel i5 12th gen     | CPU — runs NLI in practice                    |
| OS                    | Windows                                       |

Notes:

- The 5700 XT is RDNA1 and no longer supported by ROCm — Vulkan is the only viable backend. Vulkan on RDNA1 runs at ~60–70 % of CUDA per-watt throughput.
- NLI runs via ONNX Runtime from Node.js (no Python dependency). CPU is fine for DeBERTa-v3-large at ~150 ms per claim. The original plan to push NLI onto the iGPU never paid off — CPU was already fast enough.

---

## Setup in detail (Windows, reference machine)

If you're reproducing this on the reference machine, here's the full step-by-step. Adapting to different hardware is in the next section.

### LM Studio

1. Install LM Studio 0.3.x or newer (MCP client support is required).
2. Download Qwen 3.5 9B Instruct, Q4_K_M.
3. In Settings → Developer → enable **Just-In-Time Model Loading**.
4. In Settings → Developer → Local Server → bind to `127.0.0.1:1234`, start it.
5. Load Qwen 3.5 9B as the primary model.
6. Function-calling / tool-use: enabled for Qwen (native support).

### Ollama (Vulkan, AMD GPU)

1. Install Ollama for Windows. Confirm Vulkan support.
2. **Disable Ollama's tray autostart** in its preferences. The tray app spawns its own `ollama serve` with a broken env (which lands the runners on NVIDIA — see below). The Verity launcher manages Ollama instead.
3. Pull the critic models:
   ```
   ollama pull granite3.2:8b
   ollama pull granite3.2:2b
   ```

Do not start Ollama manually. `start-verity.ps1` starts it with the correct GPU pinning.

#### The GPU-pinning trap 

On the refernce machine, which has both an NVIDIA and an AMD card, Ollama by default detects CUDA at startup and prefers it — even with `OLLAMA_LLM_LIBRARY=vulkan` set. The Granite runners land on the NVIDIA card with VRAM spill into system RAM, and Verity calls time out at the 45 s `CRITIC_TIMEOUT_MS` because Ollama is running at ~12 tok/s instead of the spec ~144 tok/s.

You can't fix this with user-scope env vars because LM Studio (and Lore, if you also run that) needs CUDA visible on the NVIDIA card. The fix has to be **per-process** for Ollama only. That's what `CLI\ollama-amd.ps1` does:

- Sets `CUDA_VISIBLE_DEVICES=''` and `HIP_VISIBLE_DEVICES=0` in its own PowerShell process, then spawns `ollama serve` so the child inherits the scoped env.
- Forces a model load via `/api/generate` and then queries `nvidia-smi --query-compute-apps` to confirm no Ollama runners appeared on NVIDIA.
- If they did, automatically retries once with `VK_DRIVER_FILES` restricted to AMD's Vulkan ICD (hides NVIDIA's Vulkan driver entirely from Ollama's process).
- If they still appear, exits with code `2` and a tail of the launcher log rather than pretending to succeed.

LM Studio, Lore, and any other CUDA workload are untouched because each runs in its own process with its own env block.

Logs and PID files live at `%LOCALAPPDATA%\verity-ollama\`. Nothing is written to the registry, user scope, or as a Windows service. When you stop the launcher, zero persistent state remains.

To check GPU placement at any time:

```powershell
C:\AI\verify\CLI\ollama-amd.ps1 -Action Status
# or just the binding check
C:\AI\verify\CLI\ollama-amd.ps1 -Action Verify
```

Quick API smoke test once it's running: `curl http://localhost:11434/api/tags` should list both Granite tags.

### MCP Verity server

1. Clone or copy this project to `C:\AI\verify`.
2. From the repo root:
   ```powershell
   cd C:\AI\verify
   npm install
   npm run build
   ```
3. Edit `src/config.ts` if any defaults don't match your setup (model names, ports, timeouts). This file is the single point of adaptation.
4. Do **not** run `npm start` directly. Use the orchestrator instead, which also takes care of Ollama pinning:
   ```powershell
   .\start-verity.ps1
   ```
5. Drop a desktop icon for daily use:
   ```powershell
   .\start-verity.ps1 -InstallShortcut
   ```
6. Quick sanity check:
   ```
   curl http://localhost:8090/health
   ```

### Launcher reference

| Command | What it does |
|---|---|
| `start-verity.ps1` | Starts Ollama on AMD + Verity on `:8090`, health-checks both. Default action. |
| `start-verity.ps1 -Action Stop` | Cleanly stops Verity and Ollama. No residue. |
| `start-verity.ps1 -Action Restart` | Stops everything and restarts (e.g. after Windows update). |
| `start-verity.ps1 -Action Status` | Shows what's running, on which GPU. |
| `start-verity.ps1 -InstallShortcut` | One-time: drops a "Verity" icon on your desktop. |
| `CLI\ollama-amd.ps1 -Action Verify` | Just check that Ollama landed on AMD, not NVIDIA. Exit 0 = AMD, exit 2 = NVIDIA. |
| `CLI\ollama-amd.ps1 -Action Logs` | Tail the launcher + Ollama error logs. |

### LM Studio MCP client configuration

In LM Studio → Settings → Model Context Protocol, add:

```json
{
  "mcpServers": {
    "verity": {
      "url": "http://localhost:8090/mcp",
      "timeout": 240000,
      "retries": 1
    }
  }
}
```

The 240 000 ms client timeout matches `PIPELINE_TIMEOUT_MS = 180 000` in `config.ts` with headroom for cold-load. A tighter 60 000 ms (LM Studio's default) can brush the ceiling on `/verifydeeper` and look like a Verity bug when it isn't.

### System prompt for the worker

**Verity is intended to be self-sufficient — you should not need this system prompt for the tools to work, but testing showed that Qwen is tricky.** The `verify_answer` and `consult_second_opinion` tool descriptions carry all the trigger rules, the mandatory-/verify rule, the strict sourcing contract, and the follow-up (redraft / `/verifydeeper` / no) handling. 

The system prompt below is optional — use it if you want the worker to operate with a specific persona (here, "investigative journalist"). The persona is customisable — swap for whatever fits your use case. The trigger and behavioural rules below the blank line duplicate what's already in the tool descriptions and are kept here as belt-and-braces:

```
No small talk. All facts are verified. Do not fabricate. You are allowed to scrape websites. You are an assistant to an investigative journalist who examines large tech corporations. Provide URLs that are working (fetch them to check). When stating facts, provide in-line citation to the source in the following format [source number], [author], [publisher], [year], [page number], [url].

Treat /verify and /second as tool triggers, not English words in the user's question.

When the user types /verify (or /verifydeep / /verifydeeper) anywhere in their message, you are operating under a strict sourcing contract for that turn:
- Every non-trivial fact you state must be backed by a working source. Before you commit a claim, fetch the URL (use the fetch MCP tool) to confirm it resolves. If a URL is dead, do not cite it.
- Provide in-line citations in this exact format: [source number], [author], [publisher], [year], [page number], [url]. If a field is genuinely unavailable (e.g. no page number in a web article), write "n/a" for that field — do not fabricate values.
- A "source that actually backs up the claim" means the linked page contains the specific fact you attributed to it. Do not cite a homepage or a top-level URL when the claim is buried somewhere on the site — link to the exact page.
- Calling verify_answer is then MANDATORY after you have composed the sourced answer. Calling /second proactively does NOT satisfy /verify, and offering /verifydeeper does NOT satisfy /verify. Call the tool, paste the Verity testing block verbatim, then stop.

When using /verify, the first thing to show the user is the Verity testing table that details where critics agree and disagree and the reasons. Highlight the conclusion in bold styled text under the table. Do not redraft the answer based on critics views. Only show the table and conclusion. If critics have significant input, the block itself ends with a yes/no redraft prompt — wait for the user's reply, don't redraft unprompted.

Unless a question or task is extremely trivial, always append /second at the start of the process.

After /verify's table is shown, if the critic table includes any "❓ unable to assess" / "⚠️ warn" / "❌ fail" rows OR any concerns in the Findings section, you may end your turn with one extra line: "Run /verifydeeper for deeper checks? (yes/no)". If the user's next message is yes / OK / sure / go ahead / yeah / please / y / etc., call verify_answer with mode='deeper' immediately on your previous answer — do NOT spend a long thinking pass trying to figure out what they meant; the affirmative reply to your own offer is unambiguous. Do not offer /verifydeeper before /verify has run — let the table's actual findings drive the decision.
```

**How the triggers work:**

| Trigger (user types)       | Tool called               | When in the turn         |
|----------------------------|---------------------------|--------------------------|
| `/verify`                  | `verify_answer`  | After your answer        |
| `/verifydeep`              | `verify_answer` mode=deep | After your answer |
| `/verifydeeper`            | `verify_answer` mode=deeper | After your answer |
| `/second` or "second opinion" | `consult_second_opinion` | Before your answer (parallel) |
| (no trigger, non-trivial Q) | `consult_second_opinion` (proactive) | Before your answer |

`/verify` and `/second` are **complementary, not alternatives**. A `/verify`-flagged turn that also looks non-trivial should call `/second` at the start AND `/verify` at the end. The tool descriptions enforce this — calling `/second` does not satisfy a `/verify` request.

**Modifiers** that can be appended after `/verify`:
- `with context` → `context_mode = "with_context"` (pass the earlier conversation the answer depends on)
- `no-nli` → skip the NLI claim-checker
- `as code` / `as prose` / `as reasoning` / `as research` → override task_type auto-detect

---

## Adapting to different hardware

Settings are in `src/config.ts` and every user-tunable value is marked `[ADAPT]`.

**New + old NVIDIA pair (e.g. RTX 4090 + GTX 1080):**
- LM Studio on the strong card → set `WORKER_MODEL_NAME` to your 9–13 B model.
- Ollama on the old card → set `CRITIC_A_MODEL` and `CRITIC_B_MODEL` to whatever 2–4 GB Q4 quants fit. Keep family diversity (e.g. one Granite, one Phi).

**AMD (old) + NVIDIA (new), like the reference machine:**
- Ollama needs the Vulkan build for the old AMD card in the reference machine.
- On Linux you may need `GGML_VK_VISIBLE_DEVICES=1` to pin Ollama to the AMD card.
- Set `OLLAMA_MAX_LOADED_MODELS=2` so both critics stay resident.

**Apple Silicon (single unified-memory GPU):**
- Drop the dual-backend split — point both `LM_STUDIO_URL` and `OLLAMA_URL` at the same backend. Size critics small enough that the worker stays resident.
- DeBERTa on CPU still works fine.

**Asymmetric (one big, one tiny):**
- Drop to a 1-critic panel — edit `ALL_CRITICS` in `critic-configs.ts` to a single entry, set `MAX_UNAVAILABLE_CRITICS = 0`, accept the loss of cross-critic disputes.
- Or skip Ollama entirely and load a 2 B critic alongside the worker on the big card.
- For `/second`: set `CONSULT_DUAL=0` to revert to a single-model path.

**You will need to change...**
- `CRITIC_TIMEOUT_MS` (45 s default — covers cold JIT load on AMD; faster hardware can lower).
- `PIPELINE_TIMEOUT_MS` — keep at ~3 × the slowest critic.
- `WARN_SEVERITY_THRESHOLD` and `FAIL_SEVERITY_THRESHOLD` — tighten if critics are quiet, loosen if noisy.
- Critic system prompts in `prompts.ts` — second-biggest tuning surface after model choice.

**What stays the same on any hardware:** MCP wiring, aggregator rules (consensus logic + recompute suppression of NLI false-flags), the disputes-surfacing logic, and the recompute / NLI / consistency / perplexity flow. 

---

## Testing without LM Studio

You can POST to the MCP server directly to test the pipeline in isolation. Useful for shaking out a new install before plumbing it into LM Studio.

```bash
curl -X POST http://localhost:8090/mcp \
  -H "Content-Type: application/json" \
  -H "mcp-session-id: test-session" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "verify_answer",
      "arguments": {
        "question": "What is 2+2?",
        "answer": "2+2 equals 5.",
        "task_type": "reasoning",
        "use_nli": false
      }
    }
  }'
```

You should see consensus="fail" with the recompute pass flagging `2+2 = 4 ≠ 5` deterministically. The critics should also flag it, but recompute is the strongest signal — model uncertainty doesn't apply to a regex evaluator.

---

## Files in the repo

| File | Purpose |
|---|---|
| `src/config.ts` | **Start here.** All machine-specific settings (ports, model names, timeouts, thresholds). Every adjustable knob is marked `[ADAPT]`. |
| `src/types.ts` | Shared interfaces (`VerifyInput`, `VerifyOutput`, `CriticResult`, `NliResult`, etc.). |
| `src/prompts.ts` | Critic system prompts by task type (code / prose / reasoning / research) plus auto-detection heuristic. |
| `src/index.ts` | MCP server entry point. HTTP transport on port 8090. Boots the warmup pass. |
| `src/pipeline.ts` | Main orchestration. Fans out, collects, aggregates. |
| `src/aggregator.ts` | Consensus logic combining critic verdicts and signals. Computes the disputes table. |
| `src/sanitize.ts` | Strips `<think>` blocks and other reasoning-trace artefacts from critic output. |
| `src/tokenizer.ts` | tiktoken wrapper for context budgeting. |
| `src/critics/critic-configs.ts` | The current critic fleet (`ALL_CRITICS` array). Add / swap critics here. |
| `src/critics/call-critic.ts` | Dispatches a single critic call against its configured endpoint. |
| `src/critics/worker.ts` | Worker re-sampling for the consistency check. |
| `src/critics/parse.ts` | Tolerant JSON extraction from critic responses. Exports `findBalancedJsonObject` (shared scanner). |
| `src/nli/classifier.ts` | DeBERTa-v3-large cross-encoder NLI via `@huggingface/transformers`. |
| `src/nli/extract-claims.ts` | Heuristic regex claim extraction (standard mode). |
| `src/nli/extract-claims-llm.ts` | Worker-LLM claim extraction (deep modes). |
| `src/nli/claim-check-llm.ts` | LLM-based NLI alternative (`NLI_IMPL=llm`). |
| `src/signals/recompute.ts` | Deterministic arithmetic / unit / enumeration checker. |
| `src/signals/consistency.ts` | SelfCheckGPT-style consistency check. |
| `src/signals/perplexity.ts` | Forward-pass rescore + regen-with-logprobs fallback. |
| `src/second-opinion/consult.ts` | `/second` dual-GPU consultation + Phase-C analysis pass. |
| `src/llm/client.ts` | Shared cached OpenAI client factory used by every LLM call site. |
| `start-verity.ps1` | **Daily-use launcher** at the repo root. Brings the stack up (Ollama on AMD + Verity MCP server). Run `-InstallShortcut` once to drop a desktop icon. |
| `CLI/ollama-amd.ps1` | Per-process AMD-pinned Ollama launcher. Forces GPU verification by probing nvidia-smi after model load. Self-retries with `VK_DRIVER_FILES` if needed. |
| `CLI/cli-verify.ps1` | Direct-call CLI for `/verify` (bypasses LM Studio). |


---


## Problems 

- **Qwen sometimes ignores Verity on first run**
- **Family diversity is thinner than we want** Re-introducing a non-IBM critic when hardware and models allow is on the deferred list below. 
- **Claim extraction is still heuristic in standard mode.** Sentence splitting + filters for numbers / dates / named entities. The deep-mode LLM extractor is  stronger.
- **NLI entailment requires a premise.** Without `prior_context`, NLI is effectively skipped (the pairwise intra-answer mode tested as zero-signal and is now off by default).

## Suggested adaptations

- **Debate rounds.** Critics run once and are aggregated. A second round in which each critic sees the others' verdicts would catch more, at ~2× latency. Deferred.
- **Add a non-IBM critic.** New models are being released regularly. Swap a non Qwen or Granite model in for one of the LLM critics for bettter diversity.
- **Bi-encoder NLI for cheap shortlisting.** Cross-encoder is the right tool for high-precision claim checking but expensive when the claim list is large. A bi-encoder pre-filter could halve NLI cost on long answers. Not measured, not deferred-with-priority.
