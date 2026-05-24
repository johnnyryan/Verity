# Verity benchmark candidates: hallucination and factuality datasets

Ordered easiest-to-wire-up at the top, hardest at the bottom. "Easy" = ships claim + grounding doc + binary/categorical label and matches Verity's claim-vs-context interface directly.

## RAGTruth (2024, Niu et al., ParticleMedia)

**Task.** RAG hallucination detection. Each example is `(prompt, retrieved_context, model_response)` across three sub-tasks: open-domain QA on MS MARCO passages, data-to-text on Yelp business JSON, news summarisation on CNN/DM. Span-level and response-level hallucination labels; 18,000 annotated responses from six LLMs (GPT-3.5/4, Mistral-7B, three Llama-2 variants), test set ~1,800 datapoints. Grounded (claim vs document).

**SOTA / notable scores.** HHEM-2.1-Open trained on RAGTruth is widely deployed; Bespoke-MiniCheck-7B and RT4CHART (F1 0.776 vs prior best 0.424 on RAGTruth++) are current strong detectors. HalluGuard's 84% balanced accuracy is the figure to beat.

**Verity compatibility.** Direct fit. `verify_answer(claim=response, context=retrieved_context)` and map agree/disagree → supported/hallucinated. Response-level eval is one line of glue; span-level is harder but optional.

**URL.** https://arxiv.org/abs/2401.00396 · https://github.com/ParticleMedia/RAGTruth

## LLM-AggreFact (2024, Tang et al.)

**Task.** Aggregation of 11 factual-consistency datasets (AggreFact, TofuEval, Wice, ClaimVerify, Reveal, FactCheck-GPT, ExpertQA, Lfqa, etc.) into a unified `(document, claim, binary_label)` format. Sentence-level human-annotated factual errors. Sources span Wikipedia, interviews, web text; domains include news, dialogue, science, healthcare. Grounded.

**SOTA / notable scores.** Bespoke-MiniCheck-7B leads; MiniCheck-FT5 (770M) reaches GPT-4 accuracy at ~400x lower cost. HalluGuard reports 76% balanced accuracy.

**Verity compatibility.** Direct fit and the obvious primary benchmark. Drop-in for the existing claim-vs-context interface, unified format means one harness covers 11 sub-tasks. Leaderboard at llm-aggrefact.github.io provides comparator points.

**URL.** https://huggingface.co/datasets/lytang/LLM-AggreFact · https://arxiv.org/abs/2404.10774

## FaithBench (2024, Bao et al., Vectara)

**Task.** Summarisation hallucination detection over 10 modern LLMs across 8 families, layered on Vectara's leaderboard summaries with span-level human annotations and four labels (Consistent, Benign, Questionable, Unwanted). `(source_doc, summary, label_spans)`. Grounded.

**SOTA / notable scores.** "Even the best hallucination detection models have near 50% accuracies on FaithBench" (quote). FaithJudge (LLM-as-judge w/ few-shot human exemplars) is the 2025 step up.

**Verity compatibility.** Direct fit. Treat each summary as a claim, source as context. Useful precisely because it is hard: a stress-test rather than a leaderboard chase. Small (~700 annotated summaries), fast to run.

**URL.** https://arxiv.org/abs/2410.13210 · https://github.com/vectara/FaithBench

## Vectara Hallucination Leaderboard / HHEM (ongoing)

**Task.** Each LLM summarises 7,700+ short docs (50–24,000 words) across news, tech, science, medicine, legal, sport, business, education. HHEM-2.3 scores each `(source, summary)` pair 0..1; <0.5 = hallucination. Reports hallucination rate and answer rate. Grounded.

**SOTA / notable scores.** Current top: `antgroup/finix_s1_32b` at 1.8% hallucination rate. Cohere Command A, DeepSeek V3.2-Exp, Gemini 2.5 Pro all tracked.

**Verity compatibility.** Direct fit as a detector benchmark: feed the HHEM-evaluated `(source, summary)` pairs into Verity and compute agreement with HHEM. Not a labelled ground-truth set per se but a useful comparator against the de-facto industry signal. Pair with FaithBench (which has gold labels on a subset).

**URL.** https://github.com/vectara/hallucination-leaderboard

## HaluEval (2023, Li et al., RUCAIBox)

**Task.** 30,000 task-specific examples in three sub-tasks (QA seeded from HotpotQA, knowledge-grounded dialogue, summarisation) plus 5,000 general-user-query examples with ChatGPT responses. Each task: 10K hallucinated + 10K non-hallucinated pairs. Output: binary hallucination label. Both grounded (QA, summarisation) and partly open (dialogue).

**SOTA / notable scores.** Detection still hard; the paper reports retrieval + CoT help. ANAH-v2 7B beats GPT-4 on related splits.

**Verity compatibility.** Direct fit for the three grounded sub-tasks; QA needs the HotpotQA passage retrieval to be reproduced (or substituted with the provided "knowledge" field). Wire to Verity's claim-vs-context interface straightforwardly.

**URL.** https://arxiv.org/abs/2305.11747 · https://github.com/RUCAIBox/HaluEval

## ANAH / ANAH-v2 (2024, Ji et al., Shanghai AI Lab)

**Task.** Bilingual (zh/en) generative-QA hallucination annotation. Each model sentence is labelled No/Contradictory/Unverifiable/NoFact against a retrieved reference fragment. ANAH-v2 scales the dataset and provides a 7B annotator model. Grounded.

**SOTA / notable scores.** 7B ANAH-v2 model outperforms GPT-4 at sentence-level annotation; Qwen1.5-14B w/ refs best generator in their eval.

**Verity compatibility.** Good fit but labelling scheme is 4-way not binary; needs mapping (Contradictory + Unverifiable → hallucinated, No → supported, NoFact → skip). Sentence-level granularity matches Verity's per-claim mode.

**URL.** https://arxiv.org/abs/2405.20315 · https://open-compass.github.io/ANAH/

## FELM (2023, Chen et al., HKUST)

**Task.** Fine-grained factuality annotation over ChatGPT responses across five domains: world knowledge, science/tech, math, writing/recommendation, reasoning. 847 examples, segment-level true/false labels plus error types and reference links. Mostly open-domain (no fixed grounding doc, but reference links are provided).

**SOTA / notable scores.** GPT-4 + retrieval + CoT is best baseline; "far from satisfactory" per authors. No dominant SOTA published since.

**Verity compatibility.** Workable: use the provided reference links as context. Math and reasoning subsets do not fit a "claim vs context" framing well (no doc to ground against) so report results per-domain. World knowledge and science subsets are direct fits.

**URL.** https://arxiv.org/abs/2310.00741 · https://hkust-nlp.github.io/felm/

## ExpertQA (2024, Malaviya et al.)

**Task.** 2,177 expert-curated long-form questions across 32 fields, with expert-verified answers and per-claim attributions. `(question, model_answer, claim_spans, attributed_evidence, verification_label)`. Grounded at claim level via the attributed evidence.

**SOTA / notable scores.** Included as a sub-benchmark in LLM-AggreFact; finetuned FlanT5-11B / Llama2-7B-Chat baselines reported. No single headline SOTA outside the AggreFact aggregate.

**Verity compatibility.** Good fit per-claim via LLM-AggreFact's prepared split, less so as the raw ExpertQA release (need to assemble the claim/evidence pairs yourself). Use the AggreFact slice.

**URL.** https://arxiv.org/abs/2309.07852 · https://github.com/chaitanyamalaviya/ExpertQA

## CRAG (2024, Yang et al., Meta KDD Cup)

**Task.** 4,409 factual QA pairs across 5 domains and 8 question categories, plus mock web-search and KG APIs to simulate retrieval. Tasks: web summarisation, KG+web augmentation, end-to-end RAG. Metric: accuracy / hallucination / missing. Grounded once retrieval runs.

**SOTA / notable scores.** Best LLMs alone <=34% accuracy; naive RAG 44%; industry RAG only 63% without hallucination. KDD Cup top solutions used confidence-gated abstain (28.2% hallucination drop).

**Verity compatibility.** Indirect. CRAG benches the whole RAG pipeline; Verity slots in only as a post-hoc check on `(retrieved_context, answer)` pairs. Doable but requires running a retrieval pipeline first. Useful for measuring how much Verity adds to hallucination rate on a realistic RAG stack.

**URL.** https://arxiv.org/abs/2406.04744 · https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024

## FActScore (2023, Min et al.)

**Task.** Metric and dataset for long-form factual precision. Decompose generation into atomic facts, retrieve Wikipedia evidence per fact, label each Supported / Not-Supported, compute % Supported. Original dataset: biographies of 183 entities. Grounded (against Wikipedia).

**SOTA / notable scores.** Automated FActScore agrees with humans within 2%. ChatGPT biographies score 58%.

**Verity compatibility.** Metric is partially reusable: Verity's per-claim verdict is the analogue of the atomic-fact Supported label. But FActScore needs an atomic-fact decomposer upstream of Verity, and Wikipedia retrieval, so it's a two-step build rather than a drop-in eval. Most useful as inspiration for Verity's own scoring.

**URL.** https://arxiv.org/abs/2305.14251 · https://github.com/shmsw25/FActScore

## HalluLens (2025, Bang et al., Meta FAIR)

**Task.** Splits hallucination from factuality. Three new extrinsic tasks (output vs training data, with dynamic test-set generation to avoid leakage) plus revisited intrinsic tasks. Mixed output formats. Open-domain for the extrinsic side; grounded for the intrinsic.

**SOTA / notable scores.** Evaluates 13 instruction-tuned LLMs incl. Llama-3.1 (8B/70B/405B), Qwen-2.5, Claude-3, GPT-4o. No single headline detector SOTA; the paper is itself the benchmark contribution.

**Verity compatibility.** Partial. Intrinsic tasks fit Verity directly. Extrinsic tasks require knowing the model's training data, which Verity has no signal on; the right framing there is "did the model refuse appropriately" rather than "did Verity catch a hallucination". Use the intrinsic slice only.

**URL.** https://arxiv.org/abs/2504.17550

## FACTS Grounding (2025, Jacovi et al., Google DeepMind)

**Task.** 860 long-form (up to 32K-token) prompts each pairing a user request with a full grounding document. Models produce long responses, judged by an ensemble of LLM judges for grounding + eligibility. Public + held-out splits. Grounded.

**SOTA / notable scores.** Active Kaggle leaderboard. Gemini 2.x / GPT-4o / Claude 3.5 / 4 cluster at the top. v2 changed the judges and prompt.

**Verity compatibility.** Indirect but interesting. Verity would be a judge of the responses, comparable to the FACTS LLM-judge ensemble. Long contexts (32K) may stress Verity's NLI cross-encoder. Run as a stretch eval.

**URL.** https://arxiv.org/abs/2501.03200 · https://huggingface.co/datasets/google/FACTS-grounding-public

## SimpleQA (2024, Wei et al., OpenAI)

**Task.** 4,326 short fact-seeking questions adversarially collected against GPT-4. Output: short string answer, graded correct / incorrect / not-attempted by an LLM grader against a single indisputable reference answer. Open-domain (no context provided).

**SOTA / notable scores.** DeepSeek-V3.2-Exp 97.1%, Grok 4 Fast 95.0%, DeepSeek-V3.1 93.4% on the public leaderboard.

**Verity compatibility.** Poor without modification. No retrieved context to verify against; tests parametric recall, not grounding. Could only be benched if Verity is paired with a retriever (then it becomes a CRAG-style eval). Skip unless wiring retrieval.

**URL.** https://arxiv.org/abs/2411.04368 · https://github.com/openai/simple-evals

## TruthfulQA (2021, Lin et al.)

**Task.** 817 questions across 38 categories designed to elicit human misconceptions. Two formats: open-ended generation (judged by GPT-judge or humans) and multiple choice (MC1: pick correct from 4-5; MC2: weighted likelihood; new binary best-vs-best-incorrect). Open-domain.

**SOTA / notable scores.** Top frontier models > 90% MC1 but the benchmark is now widely considered gamed; HalluLens authors argue it is "primarily a factuality benchmark" with known label noise.

**Verity compatibility.** Poor. MC format does not match Verity's claim-vs-context interface. The open-ended variant could be wired by treating the gold "best answer" as pseudo-context, but that is awkward and not what TruthfulQA tests. Skip.

**URL.** https://arxiv.org/abs/2109.07958 · https://github.com/sylinrl/TruthfulQA

## HELM factuality (Stanford CRFM, ongoing)

**Task.** HELM is a framework, not a single benchmark. Factuality-adjacent scenarios include TruthfulQA, a Petroni-2019-style fact-knowledge scenario, and (in HELM Capabilities) MMLU-Pro, GPQA, IFEval. Mostly MC or short-answer. Open-domain.

**SOTA / notable scores.** Maintained leaderboards across 30+ models. No specific factuality-detector SOTA; HELM ranks generators.

**Verity compatibility.** Indirect. HELM is for ranking LLMs, not for benching a verifier. The only relevant use is to consume HELM's scenarios via its harness and bolt Verity on top. Lower priority than running RAGTruth / AggreFact directly.

**URL.** https://github.com/stanford-crfm/helm · https://crfm.stanford.edu/helm/

---

## Recommendation summary

Primary: **LLM-AggreFact** (single unified harness across 11 datasets, direct HalluGuard comparator). Secondary: **RAGTruth** (direct HalluGuard comparator, span-level option). Stress test: **FaithBench** (hard, small). Comparator-only: **Vectara HHEM leaderboard** (industry signal, no gold labels). Long-context stretch: **FACTS Grounding**. Skip for now: TruthfulQA, SimpleQA, HELM (format mismatch).
