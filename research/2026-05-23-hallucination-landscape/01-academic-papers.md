# Hallucination Detection / Mitigation: Peer-Reviewed & arXiv Survey

Ordered by adoptability to Verity (cross-family critics + NLI + recompute + perplexity bands over a `/v1/responses` shim). Top entries plug in with least friction.

---

## SelfCheckGPT (2023, Manakul et al.)

**Mechanism.** Black-box, sample-and-compare: draw N stochastic completions for the same prompt and score each sentence of the base response by how much the samples agree with it. Variants score agreement via BERTScore, NLI, n-gram, multiple-choice QA generated from the base sentence, or "ask another LLM to judge consistency". Quote: "if an LLM has knowledge of a given concept, sampled responses are likely to be similar and contain consistent facts".

**Result.** Highest AUC-PR for sentence-level hallucination on the WikiBio-GPT3 set vs grey-box baselines (exact numbers vary by variant; NLI and LLM-prompt variants top the table).

**Notable vs Verity.** Same family as Verity's "consistency via re-sampling" deep mode but adds a per-sentence NLI/QA agreement score across samples rather than a single re-derivation. Cheapest add-on: graft SelfCheck-NLI into the recompute pass so it runs alongside arithmetic re-derivation.

**URL.** https://arxiv.org/abs/2303.08896

---

## Chain-of-Verification / CoVe (2023, Dhuliawala et al.)

**Mechanism.** Four-stage self-verification by a single model: (i) draft answer; (ii) plan verification questions; (iii) answer each question *independently* (no draft in context) to dodge confirmation bias; (iv) regenerate the final answer conditioned on the verifications. The independence step is load-bearing — answering with the draft visible recovers the original hallucinations.

**Result.** Lower hallucination rate than zero-shot, few-shot, and CoT baselines on Wikidata list questions, MultiSpanQA, and longform biographies (paper reports F1 and FActScore gains).

**Notable vs Verity.** Verity's critics agree/disagree on extracted claims but never plan their own verification questions. CoVe's planner-then-isolated-answerer pattern is a near drop-in replacement for the critic prompt: have Granite plan the queries Ministral answers blind, then aggregate.

**URL.** https://arxiv.org/abs/2309.11495

---

## FActScore (2023, Min et al.)

**Mechanism.** Decompose a long-form generation into atomic facts via an LLM, then validate each atom against a trusted knowledge source (Wikipedia in the paper) with a retrieval+LM scorer. FActScore is "the percentage of atomic facts supported by a reliable knowledge source". Two-stage pipeline: Atomic Fact Generation, then Atomic Fact Validation.

**Result.** Automated scorer matches human FActScore within <2% error. ChatGPT scores 58% on person biographies; retrieval-augmented PerplexityAI substantially higher.

**Notable vs Verity.** Verity's claim extraction exists but its critics judge each claim against a disputed-span citation, not a decomposed atomic fact set. The atomic-decomposition prompt and the per-atom support label are directly portable; the knowledge source can be Verity's retrieved context.

**URL.** https://arxiv.org/abs/2305.14251

---

## Semantic Entropy (2024, Farquhar et al., Nature)

**Mechanism.** Sample multiple generations, cluster them by *meaning* (bidirectional NLI entailment between every pair), then compute Shannon entropy over the cluster probabilities rather than raw token sequences. High semantic entropy flags confabulation regardless of surface variation. "Measure uncertainty about the meanings of generated responses rather than the text itself."

**Result.** Detects confabulations on TriviaQA, SQuAD, BioASQ, NQ-Open, MMLU; outperforms predictive-entropy and lexical-similarity baselines. Subsequent Semantic Entropy Probes (arXiv 2406.15927) approximate SE from hidden states in one forward pass for ~5–10× cost reduction.

**Notable vs Verity.** Verity's perplexity bands measure token-level surprise; semantic entropy measures meaning-level disagreement across samples. The clustering step uses an NLI model Verity already has loaded — could replace or supplement the 4-band classifier with negligible new infrastructure.

**URL.** https://www.nature.com/articles/s41586-024-07421-0

---

## Lookback Lens (2024, Chuang et al.)

**Mechanism.** A linear classifier over the *ratio of attention weights on the input context versus newly generated tokens*, per attention head, per token. Contextual hallucinations correlate with the model attending more to its own output than to the provided context. Trains on ~1–2k annotated examples and transfers across model sizes.

**Result.** Matches richer hidden-state and NLI-based detectors. Classifier-guided decoding reduces XSum summarisation hallucinations by 9.6%.

**Notable vs Verity.** Needs attention-map access — Verity's `/v1/responses` shim does not expose attention. Only adoptable if the local model is loaded with a backend that exposes attention (vLLM/HF). Otherwise: noted for the architectural similarity to HalluGuard's attention-spectral signal.

**URL.** https://arxiv.org/abs/2407.07071

---

## SAFE / LongFact (2024, Wei et al.)

**Mechanism.** Search-Augmented Factuality Evaluator: (1) atomic statement decomposition; (2) decontextualisation so each atom is self-contained; (3) iterative Google-Search query generation; (4) per-atom support decision over the search results. Plus a new F1@K metric trading precision against expected-length recall.

**Result.** SAFE agrees with human annotators on 72% of disputed cases and is "much more cost-effective than human annotators". Benchmarked across 13 LLMs on the LongFact prompt set (38 topics).

**Notable vs Verity.** Verity has no web-search fallback; retrieval is over the local corpus. SAFE's decontextualisation step is the most portable idea — Verity's extracted claims often depend on pronouns or unstated subjects, and rewriting them to be self-contained before the NLI check would tighten entailment.

**URL.** https://arxiv.org/abs/2403.18802

---

## Self-RAG (2024, Asai et al.)

**Mechanism.** Trained model emits reflection tokens that decide (a) whether to retrieve, (b) per retrieved passage whether it is relevant, supported, and useful. Segment-wise beam search picks the continuation maximising a weighted combination of these critique tokens. Two-model training: a Critic supervises a Generator that learns to inline both.

**Result.** Outperforms ChatGPT and Llama2-chat on open-domain QA, reasoning, and fact verification (PopQA, ARC, PubHealth) with a 7B model. ICLR 2024 oral.

**Notable vs Verity.** Self-RAG requires fine-tuning a base model with reflection tokens; Verity is purely inference-time and uses two pre-trained critics instead. The reflection-token taxonomy ("relevant / supported / useful") is a finer-grained label set than Verity's pass/warn/error and could replace the aggregator's three-state output.

**URL.** https://arxiv.org/abs/2310.11511

---

## DoLa (2024, Chuang et al.)

**Mechanism.** Decoding-time. At each step, contrast the next-token logits from the final transformer layer against logits from an early "amateur" layer; subtract the early distribution to amplify factual signal that emerges in deeper layers. No retrieval, no fine-tuning. "Factual knowledge in an LLM has generally been shown to be localized to particular transformer layers."

**Result.** +12–17 absolute points on TruthfulQA over Llama baselines; gains on FACTOR, StrategyQA, GSM8K.

**Notable vs Verity.** Generation-side mitigation, not verification. Requires per-layer logit access which Verity's shim doesn't surface. Useful upstream of Verity (cleaner generations to verify) but not as a verifier component.

**URL.** https://arxiv.org/abs/2309.03883

---

## FacTool (2023, Chern et al.)

**Mechanism.** Task-agnostic tool-augmented framework: extract claims, generate tool queries (Google Search, Python interpreter, code executor, scholar APIs), execute, then aggregate per-claim verdicts. Spans four task families: knowledge QA, code generation, math, scientific literature.

**Result.** Outperforms self-checking baselines on knowledge-QA, code-execution and math-error detection (paper Tables 3–6); for scientific-literature hallucinations, catches >80% of fabricated references.

**Notable vs Verity.** Adds tool routing — Python for math, search for facts — that Verity approximates with a single recompute pass. The per-task router (which tool fits which claim type) is the adoptable idea; Verity could route arithmetic claims to recompute and entity claims to NLI rather than running both on every claim.

**URL.** https://arxiv.org/abs/2307.13528

---

## Multi-Agent Debate (2023, Du et al.)

**Mechanism.** N copies of an LLM independently answer; over multiple rounds each agent sees the others' previous responses and revises. Final answer is the majority converged response. No fine-tuning. Paper observes that arithmetic accuracy rises monotonically with both agent count and round count.

**Result.** Improves GSM8K, MMLU, biographies, and a custom factuality benchmark over CoT and single-agent baselines (ICML 2024).

**Notable vs Verity.** Verity uses two cross-family critics that vote once; MAD uses multiple same-family agents that iterate. The multi-round revision pattern would let Granite see Ministral's verdict and revise — currently Verity's critics work in isolation. Cheap to add: one extra round conditioned on the peer's disagreement.

**URL.** https://arxiv.org/abs/2305.14325

---

## Inference-Time Intervention / ITI (2023, Li et al.)

**Mechanism.** (a) Train linear probes on each attention head's activations against a true/false dataset; (b) keep the heads with high probe accuracy; (c) at inference, shift those heads' activations by α·σ along the "truthful direction" (mean-difference vector). Repeated per token. "Their findings suggest that LLMs may have an internal representation of the likelihood of something being true, even as they produce falsehoods on the surface."

**Result.** Alpaca truthfulness on TruthfulQA from 32.5% → 65.1% (MC1). Negligible inference cost once vectors are precomputed.

**Notable vs Verity.** Needs activation access — not feasible behind Verity's HTTP shim. Mechanistically related to HalluGuard's attention work. Catalogued as the canonical "internal-state steering" reference; mitigation, not detection.

**URL.** https://arxiv.org/abs/2306.03341

---

## Representation Engineering / RepE (2023, Zou et al.)

**Mechanism.** Linear Artificial Tomography (LAT): scan model hidden states for concept-aligned directions (truthfulness, honesty, harmfulness) using contrastive stimulus pairs, then either read (detection) or add (steering) along those directions. Population-level representations, not individual neurons. Generalises Burns CCS and ITI under one framework.

**Result.** Unsupervised honesty steering raises TruthfulQA by +18.1 points zero-shot, outperforming all prior methods at submission time.

**Notable vs Verity.** Again, activation access required. Cited for completeness as the "top-down" alternative to CCS — both are reference points but neither is implementable through Verity's current shim.

**URL.** https://arxiv.org/abs/2310.01405

---

## Conformal Abstention (2024, Yadkori et al.) and Conformal Language Modeling (2023, Quach et al.)

**Mechanism.** Use self-consistency among sampled responses as a model-confidence score, then calibrate an abstention threshold via split conformal prediction so that the residual error rate (hallucinations among non-abstentions) is bounded by a user-chosen α with finite-sample guarantee. Quach et al.'s earlier work calibrates both a stopping rule (when to stop sampling candidates) and a rejection rule (which candidates to drop).

**Result.** Bounds the hallucination rate on TriviaQA, TempSeq, and other open-domain QA datasets while abstaining less than log-prob baselines at matched error budgets.

**Notable vs Verity.** Verity's aggregator emits pass/warn/error from hand-tuned thresholds. Conformal calibration would give those thresholds a provable error bound conditional on a held-out calibration set. Adoption is mostly statistical bookkeeping — no new models needed.

**URL.** https://arxiv.org/abs/2405.01563 (Yadkori), https://arxiv.org/abs/2306.10193 (Quach)

---

## RARR (2023, Gao et al.)

**Mechanism.** Post-hoc attribution and revision: given any text generation, (1) generate verification queries; (2) issue them against a search engine; (3) per evidence, decide whether the text is supported; (4) edit unsupported spans while minimising textual change ("preservation" loss). Trained with handful of demonstrations.

**Result.** Improves attribution F1 substantially while preserving the original wording better than prior edit-based baselines, across PaLM/LaMDA generations (ACL 2023).

**Notable vs Verity.** Verity detects but does not rewrite; RARR closes the loop with edits. The preservation-vs-attribution trade-off is a useful design constraint if Verity ever adds a "suggest fix" output. The query-generation step is similar to CoVe's stage (ii).

**URL.** https://arxiv.org/abs/2210.08726

---

## SAPLMA / "The Internal State Knows When It's Lying" (2023, Azaria & Mitchell)

**Mechanism.** Feedforward classifier (256-128-64) over a *middle* hidden layer's activations of the LLM as it reads a statement. Predicts true/false directly. Middle layers chosen because the final layer optimises for next-token prediction, not factuality.

**Result.** 71–83% accuracy on a balanced true/false set across OPT-6.7B and Llama-2-7B; substantially above sentence-probability baselines.

**Notable vs Verity.** Probe-based; needs hidden states. Catalogued as the original "internal-state probe" paper alongside Burns CCS and ITI; precedes the more recent semantic-entropy-probe work that approximates these signals more cheaply.

**URL.** https://arxiv.org/abs/2304.13734

---

## Discovering Latent Knowledge / CCS (2022, Burns et al.)

**Mechanism.** Contrast-Consistent Search: for a yes/no question, embed both (q+"yes") and (q+"no") completions, train an unsupervised linear probe so that p(yes|q) + p(no|q) ≈ 1 (consistency) and the predictions are confident (away from 0.5). No labels used. Probes hidden states for the "is this true" direction.

**Result.** Outperforms zero-shot accuracy of the underlying LM by ~4% on average across 6 LM families and 10 datasets; importantly, when the LM is prompted to lie, zero-shot drops 9.5% but CCS does not.

**Notable vs Verity.** Foundational citation for "the model knows more than it says". Subsequent work (Farquhar et al. 2312.10029) shows CCS often locks onto prominent distracting features rather than truth — useful caveat. Activation access required, so non-adoptable in current Verity, but key conceptual reference.

**URL.** https://arxiv.org/abs/2212.03827

---

## Survey: A Survey on Hallucination in LLMs (2023, Huang et al.)

**Mechanism.** Not a method. Taxonomy of hallucination causes (data, training, inference), detection methods (uncertainty-based, consistency-based, retrieval-based, fine-tuned classifiers), and mitigation strategies organised by where in the pipeline they intervene. ACM TOIS 2025.

**Result.** N/A — survey, 55 pages, ~1k+ citations.

**Notable vs Verity.** Reference architecture for thinking about where Verity sits in the design space (post-hoc, consistency+retrieval+uncertainty hybrid). Useful for slotting in future ideas without redundancy.

**URL.** https://arxiv.org/abs/2311.05232

Companion older survey: Ji et al. "Survey of Hallucination in NLG" (ACM Comput. Surv. 2023), https://arxiv.org/abs/2202.03629 — pre-LLM-era taxonomy, still the most-cited.

---

## Sources

- [SelfCheckGPT (Manakul et al. 2023)](https://arxiv.org/abs/2303.08896)
- [Chain-of-Verification (Dhuliawala et al. 2023)](https://arxiv.org/abs/2309.11495)
- [FActScore (Min et al. 2023)](https://arxiv.org/abs/2305.14251)
- [Semantic Entropy (Farquhar et al. 2024, Nature)](https://www.nature.com/articles/s41586-024-07421-0)
- [Lookback Lens (Chuang et al. 2024)](https://arxiv.org/abs/2407.07071)
- [SAFE / LongFact (Wei et al. 2024)](https://arxiv.org/abs/2403.18802)
- [Self-RAG (Asai et al. 2024)](https://arxiv.org/abs/2310.11511)
- [DoLa (Chuang et al. 2024)](https://arxiv.org/abs/2309.03883)
- [FacTool (Chern et al. 2023)](https://arxiv.org/abs/2307.13528)
- [Multi-Agent Debate (Du et al. 2023)](https://arxiv.org/abs/2305.14325)
- [Inference-Time Intervention (Li et al. 2023)](https://arxiv.org/abs/2306.03341)
- [Representation Engineering (Zou et al. 2023)](https://arxiv.org/abs/2310.01405)
- [Conformal Abstention (Yadkori et al. 2024)](https://arxiv.org/abs/2405.01563)
- [Conformal Language Modeling (Quach et al. 2023)](https://arxiv.org/abs/2306.10193)
- [RARR (Gao et al. 2023)](https://arxiv.org/abs/2210.08726)
- [SAPLMA (Azaria & Mitchell 2023)](https://arxiv.org/abs/2304.13734)
- [Discovering Latent Knowledge / CCS (Burns et al. 2022)](https://arxiv.org/abs/2212.03827)
- [Huang et al. Hallucination Survey (2023)](https://arxiv.org/abs/2311.05232)
- [Ji et al. NLG Hallucination Survey (2022)](https://arxiv.org/abs/2202.03629)
- [Self-Consistency (Wang et al. 2022)](https://arxiv.org/abs/2203.11171)
- [FELM benchmark (Chen et al. 2023)](https://arxiv.org/abs/2310.00741)
- [MARS (Bakman et al. 2024)](https://arxiv.org/abs/2402.11756)
