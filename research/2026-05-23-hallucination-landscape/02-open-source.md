# Open-source hallucination-detection tooling — landscape vs Verity

Ten entries, ordered from closest comparable to most tangential. All mechanism summaries grounded in the project's own README or docs.

---

## SelfCheckGPT (potsawee/selfcheckgpt, ~613 stars, last release Mar 2024)

**Mechanism.** Zero-resource, black-box hallucination detection that samples multiple stochastic completions from the same generator and scores sentence-level consistency across them. Ships five scoring variants in one package: BERTScore similarity, multiple-choice QA-generation, n-gram overlap, NLI entailment with DeBERTa-v3-large fine-tuned on Multi-NLI, and an LLM-prompt variant where a separate model is asked whether each sentence is "supported by context".

**Notable vs Verity.** This is the canonical self-sampling-for-consistency tool, so Verity's consistency-via-resampling stage is directly in this lineage. Where Verity differs: SelfCheckGPT does not run cross-family LLM critics with a structured agree/disagree disputed-span protocol, has no recompute pass, and produces no perplexity/logprob band classifier. SelfCheckGPT-NLI overlaps with Verity's NLI cross-encoder stage.

**Adoption signal.** Foundational paper (EMNLP 2023), heavily cited, integrated into many downstream frameworks including LangKit; repo itself low star but the technique is referenced widely.

**URL.** https://github.com/potsawee/selfcheckgpt

---

## Lynx (patronus-ai/Lynx-hallucination-detection, ~45 stars, model July 2024)

**Mechanism.** A purpose-built hallucination-judge LLM, not a framework. Llama-3-70B-Instruct (and an 8B sibling) fine-tuned on CovidQA, PubmedQA, DROP and RAGTruth with synthetic semantic-perturbation examples; given (document, question, answer) it outputs a faithfulness verdict and reasoning. Repo holds inference and finetuning scaffolding via mcli / vLLM; the model is the artefact.

**Notable vs Verity.** Same family of approach as Verity's critic stage but mono-model and mono-family (Llama only). Verity's cross-family design (Granite + Ministral) explicitly hedges against the failure mode Lynx cannot detect, where the judge inherits the same blind spots as the generator. No NLI, logprob band, or self-consistency in Lynx.

**Adoption signal.** Repo small (45 stars) but the HF model has reasonable downloads, NVIDIA NeMo-Guardrails integration on day one, and a published paper (arXiv 2407.08488). Patronus AI is a commercial vendor; the OSS release looks like a marketing artefact rather than an actively maintained tool.

**URL.** https://github.com/patronus-ai/Lynx-hallucination-detection

---

## RefChecker (amazon-science/RefChecker, ~429 stars, last release May 2025)

**Mechanism.** Three-stage Decompose-Then-Verify pipeline. An extractor LLM decomposes the response into knowledge triplets (subject, predicate, object). Each triplet goes to a "checker" that emits Entailment / Contradiction / Neutral; checker can be an LLM, AlignScore, or a RoBERTa NLI head. Aggregation rules over per-triplet labels produce the final response-level factuality verdict. Supports zero-context, noisy-context (RAG) and accurate-context settings.

**Notable vs Verity.** Closest architectural analogue to Verity's critic + NLI combination, but ordered differently: RefChecker decomposes first then checks atomic triplets, whereas Verity judges whole claims with a disputed-span citation and uses NLI as a separate cross-encoder check. RefChecker has no logprob/perplexity stage and no self-consistency sampling.

**Adoption signal.** Apache-2.0, Amazon Science backing, 95% inter-annotator-agreement benchmark, active into mid-2025.

**URL.** https://github.com/amazon-science/RefChecker

---

## HHEM-2.1-Open (vectara/hallucination_evaluation_model, parent leaderboard ~3.3k stars, model 2024)

**Mechanism.** A flan-T5-base classifier fine-tuned on AggreFact, RAGTruth and model-generated summaries. Takes a (premise, hypothesis) pair and returns a scalar 0-1 score, "0 = hypothesis is not evidenced by the premise". Two output neurons (hallucinated / consistent). Under 600 MB, ~1.5s for 2k tokens on CPU, no token-length cap.

**Notable vs Verity.** A dedicated NLI-style classifier rather than an LLM judge — directly comparable to Verity's NLI cross-encoder. Much cheaper than Verity's critic stage (CPU-fine), but emits only a scalar, no reasoning, no disputed-span, no perplexity. Verity's pipeline could plausibly use HHEM as a drop-in for its NLI stage.

**Adoption signal.** Vectara also publishes the public hallucination leaderboard built on this model; widely cited and used; HHEM-2.3 (newer) is commercial.

**URL.** https://github.com/vectara/hallucination-leaderboard

---

## Ragas (explodinggradients/ragas, ~14k stars, v0.4.3 Jan 2026)

**Mechanism.** RAG evaluation toolkit. The Faithfulness metric uses an LLM judge to identify claims in the response, then asks the LLM whether each claim "can be inferred from the retrieved context"; final score is supported claims divided by total claims. Ragas also ships `FaithfulnesswithHHEM`, which swaps the per-claim LLM check for Vectara's HHEM-2.1-Open T5 classifier.

**Notable vs Verity.** Same claim-decomposition pattern as RefChecker; LLM-as-judge by default. No cross-family critic redundancy, no logprob band, no self-consistency resample. The HHEM variant is interesting: it shows the same hybrid-NLI thinking Verity uses, just packaged differently.

**Adoption signal.** Very heavily used in RAG-eval pipelines; large community; integrations with LangChain, LlamaIndex, LangSmith.

**URL.** https://github.com/explodinggradients/ragas

---

## TruLens (truera/trulens, ~3.3k stars, v2.8.1 May 2026)

**Mechanism.** Instrumented "RAG Triad" evaluator (groundedness, context-relevance, answer-relevance) plus generic feedback-function machinery. Groundedness "separate[s] the response into individual claims and independently search[es] for evidence" in the retrieved context. Ships both LLM-judge and "groundedness NLI" backends, the latter described as relying on "Medium Language Models (like BERT)" rather than LLMs.

**Notable vs Verity.** The NLI backend is conceptually the same as Verity's NLI stage; the LLM-judge backend uses a single judge with no cross-family redundancy. No logprob/perplexity stage, no self-consistency.

**Adoption signal.** Owned by Snowflake/TruEra, regular releases, decent ecosystem traction.

**URL.** https://github.com/truera/trulens

---

## DeepEval (confident-ai/deepeval, ~15.6k stars, May 2026 release)

**Mechanism.** Pytest-style LLM eval framework with separate Hallucination and Faithfulness metrics. The Hallucination metric is an LLM-as-judge that scores "Number of Contradicted Contexts / Total Number of Contexts" by asking an LLM if each provided context contradicts the output. The Faithfulness metric extracts claims from the output then classifies each as "truthful or contradictory" against retrieval context; score is truthful claims over total. Default backend is OpenAI GPT; custom `DeepEvalBaseLLM` accepted.

**Notable vs Verity.** Thin GPT-as-judge by default. Two single-judge metrics with no NLI, no logprob layer, no self-consistency, no cross-family redundancy. Verity is more architecturally varied at the cost of being heavier.

**Adoption signal.** Large stars, weekly releases, commercial backer (Confident AI).

**URL.** https://github.com/confident-ai/deepeval

---

## FActScore (shmsw25/FActScore, paper repo, EMNLP 2023)

**Mechanism.** Original "Decompose-Then-Verify" framework. Decomposes long-form generation into atomic facts via a generator LLM, then retrieves from a knowledge source (default Wikipedia 2023-04 dump, swappable to user-supplied jsonl) and asks a strong LM whether each atomic fact is supported. Score is percentage of supported atomic facts. Length penalty `gamma` discourages low-fact responses.

**Notable vs Verity.** Reference-grounded rather than retrieval-grounded; assumes a trusted knowledge source. The atomic-fact granularity is finer than Verity's claim-level judgement. No NLI, no logprobs, no consistency sampling, no cross-family critics.

**Adoption signal.** Foundational reference for the entire decompose-then-verify literature; pip-installable; many derivative works (OpenFActScore, PFME, Mask-DPO).

**URL.** https://github.com/shmsw25/FActScore

---

## Guardrails AI (guardrails-ai/guardrails, ~6.9k stars, v0.10.0 Apr 2026)

**Mechanism.** General-purpose Input/Output Guard framework keyed off RAIL specs and a "Guardrails Hub" of pluggable validators. For hallucination specifically the relevant Hub validators wrap third-party detectors (GroundedAI, ProvenanceLLM, ProvenanceEmbeddings, etc.) rather than implementing a novel detector. The framework itself is the orchestration layer; it does function-calling or prompt-engineering to force compliant output.

**Notable vs Verity.** Apples-to-oranges. Guardrails is a policy/validation harness; Verity is a detection pipeline. Verity could plausibly run inside a Guardrails Output Guard as a custom validator.

**Adoption signal.** Healthy stars, regular releases, commercial Guardrails Hub plus 2025 Guardrails Index benchmark.

**URL.** https://github.com/guardrails-ai/guardrails

---

## NVIDIA NeMo Guardrails (NVIDIA/NeMo-Guardrails, ~6.2k stars, v0.22.0 May 2026)

**Mechanism.** Programmable conversational guardrails defined in Colang DSL. Implements hallucination detection as two specific output rails. `self check facts` prompts the same LLM to verify factual accuracy of its own response against retrieved context (single-LLM self-check). `self check hallucination` similarly self-prompts for hallucination flags. Integrates Patronus Lynx as a third-party hallucination rail.

**Notable vs Verity.** Self-check rails are single-LLM checks against the same model that generated the response — the exact failure mode cross-family critics are designed to avoid. No NLI, logprob, or sampling-consistency layer in the rails themselves, although the Lynx integration adds a fine-tuned judge.

**Adoption signal.** NVIDIA-maintained, frequent releases, deeply integrated with NIM endpoints.

**URL.** https://github.com/NVIDIA/NeMo-Guardrails

---

## Outlines + Guidance + LMQL (constrained-generation family)

**Mechanism.** Three sibling libraries that constrain *generation* via logit masking and FSA/CFG-based token filtering, so the model can only emit strings matching a schema, regex, JSON, or Pydantic type. Outlines (~13.9k stars, v1.3.0 May 2026) and Guidance (~21.5k stars, v0.3.2 Mar 2026) are actively developed; LMQL (~4.2k stars) has been quiet since late 2023.

**Notable vs Verity.** Prevention rather than detection. Eliminates *structural* hallucination (malformed JSON, invented enum values, type errors) but cannot stop the model populating a well-formed field with a fabricated fact. Orthogonal to and composable with Verity.

**Adoption signal.** Outlines and Guidance both have large stars, active maintenance, and broad use in agentic stacks. LMQL is largely superseded.

**URLs.** https://github.com/dottxt-ai/outlines | https://github.com/guidance-ai/guidance | https://github.com/eth-sri/lmql

---

## WhyLabs LangKit (whylabs/langkit, ~990 stars, last release v0.0.35 Nov 2024)

**Mechanism.** A whylogs-compatible text-metrics toolkit that extracts signals from prompt/response pairs. The hallucination module is described as a "consistency check between responses" — i.e. a SelfCheckGPT-style re-sample-and-compare wrapper rather than a novel detector. Other modules cover toxicity, PII, sentiment, themes.

**Notable vs Verity.** Self-consistency reimplementation, packaged for monitoring rather than gating. No NLI, no logprob band, no cross-family critics, no claim decomposition. Appears stale (no 2025/2026 release).

**Adoption signal.** Modest stars, no release in over a year, WhyLabs commercial backing but apparent low momentum on the OSS module.

**URL.** https://github.com/whylabs/langkit

---

## Quick cross-cutting note on LangChain evaluators

LangChain's built-in `labeled_criteria` and `embedding_distance` evaluators come up in this space but are not hallucination detectors. `labeled_criteria` is a generic LLM-as-judge over user-specified rubrics; `embedding_distance` is cosine-similarity between prediction and reference. They can be wired up to evaluate faithfulness, but the framework provides no purpose-built hallucination metric — users typically delegate to Ragas, DeepEval, or TruLens for that. Worth noting only because the question presupposed they were in the same category; they are not.

---

**Summary of mechanism coverage vs Verity:**

| Tool | Cross-family critics | NLI | Logprobs | Self-consistency |
|---|---|---|---|---|
| Verity | yes | yes | yes (advisory) | yes |
| SelfCheckGPT | — | yes (one variant) | — | yes (core) |
| Lynx | — (single fine-tuned judge) | — | — | — |
| RefChecker | — | yes (one backend) | — | — |
| HHEM | — | yes (T5 classifier) | — | — |
| Ragas | — | optional (HHEM backend) | — | — |
| TruLens | — | yes (NLI backend) | — | — |
| DeepEval | — | — | — | — |
| FActScore | — | — | — | — |
| Guardrails AI | n/a (harness) | n/a | n/a | n/a |
| NeMo Guardrails | — (self-check on same model) | — | — | — |
| Outlines/Guidance/LMQL | n/a (prevention) | n/a | n/a | n/a |
| LangKit | — | — | — | yes (wrapped) |

Verity's distinguishing combination is the *simultaneous* presence of cross-family critics, NLI, advisory logprob banding, and consistency resampling. No OSS project surveyed combines more than two of these.
