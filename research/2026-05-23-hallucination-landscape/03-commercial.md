# Commercial / SaaS hallucination-detection products

Ordered by architectural relevance to Verity (closest mechanism on top). "Closest" means: ensembled cross-family critics, NLI/entailment classifiers, logprob/uncertainty signals, span-level attribution. Pure LLM-as-judge wrappers are ranked lower.

## Cleanlab Trustworthy Language Model (Cleanlab, hosted + on-prem)

**Mechanism.** Wrapper around any base LLM that fuses three signals: consistency across `num_consistency_samples` (default 8) re-sampled responses, self-reflection where the model grades its own output, and token-level logprob measures. Combines aleatoric and epistemic uncertainty into a single 0-1 score with an optional explanation field. Closest in spirit to Verity of any product in this slice.

**Public claims.** Published in ACL. Cleanlab's own benchmarks claim TLM detects wrong responses with higher precision than raw logprobs or LLM-as-judge baselines; they publish per-LLM hallucination-detection benchmark posts on RAGTruth and similar.

**Notable vs Verity.** Same triad (multi-sample consistency + self-judge + logprobs) but no cross-family critic, no separate NLI cross-encoder, and no disputed-span attribution. Operates as a cloud wrapper around a base LLM rather than as a verifier sitting beside one.

**URL.** https://help.cleanlab.ai/tlm/

## Lynx (Patronus AI, hosted + open-weights self-host)

**Mechanism.** Fine-tuned Llama-3 (8B and 70B variants) trained on CovidQA, PubmedQA, DROP and RAGTruth with synthesised perturbations. Given (question, document, answer) it emits PASS/FAIL plus a chain-of-thought rationale in JSON. Checks three faithfulness conditions: every claim entailed by chunks, no extra information, no contradiction. A purpose-built single-judge classifier, not an ensemble.

**Public claims.** 87.4% accuracy on HaluBench (15k samples) for the 70B; 8B claimed to beat GPT-3.5 by 24.5% and Claude-3-Sonnet by 8.6% on HaluBench. Day-one integrations with NeMo Guardrails, MongoDB, Nomic.

**Notable vs Verity.** Single judge, single family (Llama). No NLI cross-encoder, no logprob band, no consistency re-sampling. Strong reasoning output but no disputed-span citation in the Verity sense. Open weights make it the obvious candidate for the Granite/Ministral roster if Verity ever wants a third critic.

**URL.** https://www.patronus.ai/blog/lynx-state-of-the-art-open-source-hallucination-detection-model

## Luna (Galileo, hosted)

**Mechanism.** DeBERTa-large (440M) encoder fine-tuned for span-level token support against retrieved context plus query. Multi-task heads cover context adherence, chunk utilisation, context relevance, security. Long context handled by a chunking window that retains question + response tokens across slices. Closest commercial product to a classic NLI cross-encoder, similar in flavour to Verity's NLI stage.

**Public claims.** Galileo reports 97% cost and 91% latency reduction vs GPT-3.5 LLM-judge baselines, sub-200ms scoring, "85 to 90 percent agreement with human grading" on RAGTruth/HaluBench class benchmarks per the company's own writeups.

**Notable vs Verity.** Produces span-level predictions (Verity has disputed-span citation only). No cross-family LLM critic, no logprob band, no re-sample consistency. Closed-weights hosted-only encoder; Verity is local and open-stack. Architecturally Luna is "Verity's NLI module, productised on its own".

**URL.** https://galileo.ai/blog/introducing-galileo-luna-a-family-of-evaluation-foundation-models

## HHEM (Vectara, hosted + open-weights HHEM-2.1-Open)

**Mechanism.** DeBERTa-v3-base classifier (HHEM 1.0) trained on multi_nli, snli, fever, vitaminc, paws. Pure NLI-style factual-consistency scorer, deliberately not an LLM-as-judge. HHEM-2.1-Open is multilingual (EN/FR/DE), runs in <600MB at FP32, ~1.5s for 2k tokens on CPU. Commercial HHEM-2.3 sits behind Vectara's API.

**Public claims.** HHEM-2.1 claimed to outperform GPT-3.5-Turbo and GPT-4 on the company's own hallucination leaderboard. >100k Hugging Face downloads.

**Notable vs Verity.** NLI-only, single-shot, no LLM critic, no logprob, no consistency. The cheapest most CPU-friendly equivalent to Verity's NLI stage. Verity's full pipeline is richer; HHEM is what you'd reach for if you only wanted the entailment leg.

**URL.** https://huggingface.co/vectara/hallucination_evaluation_model

## Aporia Guardrails (Aporia, hosted + on-prem)

**Mechanism.** Multi-SLM detection engine: several small language models, each specialised on one policy (hallucination, prompt injection, PII), arbitrated together for RAG faithfulness. Of the products surveyed, this is the only one whose vendor-stated architecture is itself an ensemble of small specialists — closest in spirit to Verity's cross-family critic idea, although Aporia does not disclose model families.

**Public claims.** 98% hallucination detection (Aporia's own benchmark), vs 91% NeMo Guardrails and 94% GPT-4o. Mean 0.34s, p90 0.43s latency.

**Notable vs Verity.** Ensemble flavour but no public disclosure that the SLMs come from different families. No NLI stage exposed, no logprob band, no span attribution. Strictly hosted-runtime or on-prem service; Verity's local critics + recompute + perplexity stack is more transparent end-to-end.

**URL.** https://www.aporia.com/hallucination-mitigation/

## Contextual Grounding Check (Amazon Bedrock Guardrails, hosted)

**Mechanism.** Two scores: a Grounding score (response factually entailed by tagged grounding_source) and a Relevance score (response addresses query). Per-chunk relevance, OR-aggregated. Threshold-based filtering 0-0.99. Undisclosed underlying model; AWS positions it as a managed component rather than describing weights.

**Public claims.** AWS marketing claim: "filters over 75% hallucinated responses for RAG and summarization workloads". Independent testing (Caylent) shows grounding scores drop across multi-turn agentic conversations even with full context passed.

**Notable vs Verity.** Per-turn snapshot, no consistency re-sampling, no logprobs exposed, no span-level disputed-citation. Cloud-only, opaque internals. Verity's logprob band + cross-family critics give it more diagnostic depth.

**URL.** https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-contextual-grounding-check.html

## Groundedness Detection (Azure AI Content Safety, hosted)

**Mechanism.** Custom language-model classifier with domain selectors (MEDICAL, GENERIC) and task selectors (QnA, Summarization). Two modes: Non-Reasoning for fast online filtering and Reasoning mode that calls GPT-4o to emit per-segment explanations and an optional auto-correction. English-only at GA.

**Public claims.** No published benchmark numbers in the docs; marketing-grade ("reduces fabricated outputs"). The reasoning-mode correction feature is unique among hosted groundedness APIs.

**Notable vs Verity.** Auto-correction is a feature Verity lacks. No logprob exposure, no re-sample consistency, no cross-family critic. Single-vendor lock-in (Azure region, GPT-4o for reasoning). Closed weights.

**URL.** https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/groundedness

## Check-Grounding API and Gen AI Evaluation (Google Vertex AI, hosted)

**Mechanism.** check-grounding emits a 0-1 support score with citations to supporting facts; requires whole-claim entailment to register as grounded. High-Fidelity Mode adds a Gemini 1.5 Flash variant fine-tuned to anchor on supplied context. Vertex Gen AI Eval adds explicit Faithfulness and Answer-Relevance metrics for offline pipelines.

**Public claims.** Marketing-only. No public benchmark numbers tied to check-grounding itself; performance attribution rolled into broader Vertex RAG releases.

**Notable vs Verity.** Whole-claim entailment + citations is similar to Verity's span attribution in intent. No logprob band, no cross-family critic (Gemini all the way down), no re-sample consistency. Cloud-only.

**URL.** https://cloud.google.com/generative-ai-app-builder/docs/check-grounding

## Patronus Evaluators API and Judge (Patronus AI, hosted)

**Mechanism.** Self-serve evaluation API bundling Lynx for hallucination plus configurable LLM-as-judge "Judge" evaluators with active learning (thumbs up/down). Pre-built criteria for context relevance, answer relevance, hallucination, enterprise PII, toxicity. Glider is a separate closed-weights judge model.

**Public claims.** $10/$20 per 1000 calls. Customer case studies: Algomo doubled hallucination-detection score 0.375 -> 0.69 using Lynx; Gamma "1000+ hours saved".

**Notable vs Verity.** Combines a tuned classifier (Lynx) with an LLM judge. No cross-family ensemble at runtime, no logprob signal, no NLI stage, no disputed-span attribution. Verity stays local; Patronus is a hosted commercial endpoint.

**URL.** https://www.patronus.ai/blog/patronus-evaluators

## Fact-checking rails in NeMo Guardrails (NVIDIA, hosted + on-prem)

**Mechanism.** NeMo Guardrails ships two output-rail fact-checking modes: Self-Check (entailment-style LLM prompt, default) and AlignScore integration (RoBERTa-based factual-consistency model from Zha et al.). AlignScore averages ~220ms with threshold 0.7. Optional integration with Patronus Lynx and Cleanlab TLM via plug-ins.

**Public claims.** NVIDIA evaluates on a 100-triple MSMARCO subset; AlignScore claimed competitive with the Self-Check LLM rail at lower latency. Marketing rather than peer-reviewed benchmarking for the rail itself.

**Notable vs Verity.** AlignScore is a single-model NLI; Self-Check is single-LLM entailment. No cross-family ensemble, no logprobs exposed, no consistency re-sampling at rail level. Verity sits beside the LLM; NeMo is an inline policy gateway.

**URL.** https://docs.nvidia.com/nemo/guardrails/latest/configure-rails/guardrail-catalog/fact-checking.html

## Trustwise Optimize:ai (Trustwise, hosted)

**Mechanism.** Marketing describes "6 agents, 12 SLMs, 14 datasets, 30 guardrail modules, 1,100 mapped controls, 20K red team prompts, 21M synthetic personas" producing a runtime trust score and policy-shield enforcement (Prompt Shield, Compliance Shield, Brand Shield, Cost Shield, Carbon Shield). Specific hallucination algorithm undisclosed; marketing only.

**Public claims.** Customer case studies: 40% improvement in hallucination detection at a healthcare platform, 20% at a global professional services firm, 80% LLM cost reduction. 2025 Gartner Cool Vendor for Agentic AI in Banking.

**Notable vs Verity.** Multi-SLM hint suggests ensemble structure, but no public detail on cross-family choice, NLI, logprobs, or span attribution. Sold as a runtime governance product rather than a verifier per se. Verity is more transparent and locally inspectable.

**URL.** https://trustwise.ai/

## Pythia (Wisecube AI, hosted + open-source library)

**Mechanism.** Knowledge-triplet extraction (subject, predicate, object) from both the LLM response and a reference source, followed by triplet-level classification into Entailment / Contradiction / Missing / Neutral. Aggregate report uses an A-F grade mapped to a numeric score. Structurally distinct from NLI sentence-level entailment: works at extracted-relation granularity.

**Public claims.** Wisecube's own writeups; no widely cited external benchmark. Targets healthcare research workflows.

**Notable vs Verity.** Triplet-level granularity is finer than Verity's claim-level decomposition; could be complementary. No cross-family critic, no logprob signal, no consistency re-sampling. Single-judge pipeline.

**URL.** https://askpythia.ai/blog/evaluating-llm-hallucination-detectors

## Honourable mentions (LLM-judge wrappers, marketing-grade or out of scope)

- **Arize Phoenix HallucinationEvaluator** — pure LLM-as-judge template (you supply the judge model), Apache 2.0, span-attached via OpenInference. Thin wrapper. https://arize.com/docs/ax/evaluate/llm-as-a-judge/arize-evaluators-llm-as-a-judge/hallucinations
- **Galileo ChainPoll** — chain-of-thought LLM-judge ensembled via polling. The Correctness and Context Adherence metrics that pre-date Luna. Cost-optimised LLM-judge, not a classifier. https://docs.galileo.ai/galileo-ai-research/chainpoll
- **W&B Weave** — LLM-judge templates plus tracing; you choose the judge model. Hallucination detection here is a UX feature, not a vendor algorithm. https://wandb.ai/site/articles/llm-observability/
- **WhyLabs LangKit** — response-consistency hallucination module: re-samples N responses via OpenAI, scores LLM-judge similarity + semantic similarity, combines into final score. Mechanism overlaps with Verity's consistency stage but is OpenAI-only and bolt-on. https://github.com/whylabs/langkit/blob/main/langkit/docs/modules.md
- **Lakera Guard** — primarily prompt injection / jailbreak / PII / OWASP LLM Top 10. Hallucination is marketing-listed but not the strength; no published mechanism for output-side groundedness. https://www.lakera.ai/product-updates/lakera-guard-overview
- **OpenAI Evals** — open-source eval framework with model-graded templates. Not a productised hallucination detector; a harness teams use to build their own LLM-judge graders. https://evals.openai.com/
- **Guardrails AI Hub** — Apache 2.0 validator library; relevant factuality validators include `LLMCritic`, `ProvenanceLLM`, `LogicCheck`, and a MiniCheck wrapper (hosted Bespoke API). LLM-judge or embedding-overlap based; no native ensemble. https://guardrailsai.com/hub

## Architectural takeaways for Verity

- **Closest analogue overall**: Cleanlab TLM. Same triad of consistency re-sampling + self-reflection + logprobs. What it lacks is cross-family critics and an explicit NLI stage. Verity's recombination of a separate NLI cross-encoder with two cross-family critics is unusually integrated for the commercial landscape.
- **Closest analogue to Verity's NLI module**: Vectara HHEM-2.1-Open (DeBERTa, CPU-friendly, open weights) and Galileo Luna (DeBERTa-large, span-level).
- **Closest analogue to Verity's two-critic ensemble**: nothing in the public commercial space discloses cross-family choice. Aporia and Trustwise gesture at multi-SLM but won't say which families. This appears to be a genuinely under-occupied niche.
- **Closest analogue to Verity's logprob band**: only Cleanlab TLM exposes logprob-derived uncertainty as a first-class scoring signal. The hosted gateways (Bedrock, Azure, Vertex) hide it entirely.
- **Closest analogue to Verity's disputed-span citation**: Galileo Luna (token-span predictions) and Vertex check-grounding (claim citations to facts).

No commercial product surveyed combines all four signals (cross-family critics + NLI + logprobs + span attribution) the way Verity does. Most products productise one stage well; Verity's distinguishing pitch would be the unified local pipeline.
