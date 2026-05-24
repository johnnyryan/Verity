# Multi-agent debate, critique & self-verification — adjacent prior work for Verity

Ordered by adoptability/displacement relevance to Verity (one round, two cross-family critics, NLI, no debate, no iteration).

---

## Chain-of-Verification / CoVe (2023, Dhuliawala et al.)

**Mechanism.** Same model, four sequential calls per query: (1) draft initial response; (2) plan verification questions targeting the draft's claims; (3) answer each question independently (key trick: answers conditioned only on the question, not the original draft, to break confirmation bias); (4) regenerate final answer using those verified facts. No second model. Variants run steps 2-4 jointly or factored.

**Result.** Reduces hallucinations across Wikidata list questions, MultiSpanQA closed-book, and longform biographies. Method paper, no single headline number in abstract.

**Notable vs Verity.** Closest cousin: same "draft then check" intuition, but CoVe self-decomposes into Q&A, where Verity already has external claims plus retrieved citation. The independent-answer trick (step 3 cannot see the draft) is directly adoptable in Verity's recompute pass to harden it against critic agreement bias. Single-model, so cheaper than cross-family.

**URL.** https://arxiv.org/abs/2309.11495

---

## LM vs LM cross-examination (2023, Cohen et al.)

**Mechanism.** Two LM setup: the claimant model produces an answer; a separate examiner LM runs a multi-turn interrogation, generating follow-up questions designed to surface inconsistency. Variable rounds, ends when examiner reaches a confidence verdict. Aggregation is the examiner's final judgement, not voting.

**Result.** "Outperforms existing methods and baselines, often by a large margin" on four factuality benchmarks (LAMA, TriviaQA, NQ, PopQA).

**Notable vs Verity.** Verity's two critics judge in parallel; cross-examination is sequential and interrogative. Adopting a single examiner round (one critic generates a probe question for the other to answer, then both judge) would add adversarial signal without changing Verity's per-query call count much. Most directly adoptable architectural delta.

**URL.** https://arxiv.org/abs/2305.13281

---

## Multi-Agent Debate / Society of Minds (2023, Du et al.)

**Mechanism.** N homogeneous LLM instances (typically 3 agents, 2-3 rounds in the paper) each produce an answer in round 1, then each round shows every agent the other agents' previous responses and asks them to update. Majority vote or final-round consensus at the end. Same model family throughout in the main experiments; an addendum mixes ChatGPT and Bard.

**Result.** On Biographies, debate moves accuracy from 66% (single agent) to roughly 73%; gains across MMLU, chess move validity, GSM8K. ICML 2024.

**Notable vs Verity.** This is the canonical thing Verity is not. Verity uses one round, no inter-agent visibility, no consensus loop. Adopting one extra round where each critic sees the other's verdict would cost roughly 2x more calls but is the most studied lever in the literature. The factuality bump is real but modest; cost on-device is non-trivial.

**URL.** https://arxiv.org/abs/2305.14325

---

## Encouraging Divergent Thinking / MAD (2023, Liang et al.)

**Mechanism.** Two debaters in "tit-for-tat" stance (one affirmative, one negative) plus a separate judge LLM that manages the debate and decides termination. Each debater can see prior turns. Adaptive break: judge stops debate when convinced. Roughly 0.63x extra inference cost over CoT baseline.

**Result.** Gains on Commonsense MT and Counter-Intuitive Arithmetic Reasoning. Authors flag that "LLMs might not be a fair judge if different LLMs are used for agents" — bias caveat directly relevant to Verity.

**Notable vs Verity.** Adds an explicit judge role on top of debate. Verity's two critics are peers; introducing a judge (a third call, smaller model) could replace Verity's manual aggregation rule. The fairness warning argues against a single critic-family judge.

**URL.** https://arxiv.org/abs/2305.19118

---

## ChatProtect / Self-contradictory hallucinations (2023, Mündler et al.)

**Mechanism.** Three-step pipeline aimed specifically at self-contradiction. For each sentence in a generated response, the system (1) prompts the model to generate alternate sentences in the same context; (2) detects whether alternates contradict the original via an LLM-based detector; (3) iteratively refines to remove contradictions. Black-box, no external DB.

**Result.** Detector ~80% F1 on ChatGPT outputs; mitigation reduces self-contradiction by up to 89.5%. 17.7% of ChatGPT sentences flagged as self-contradictory.

**Notable vs Verity.** Targets intrinsic contradiction (no retrieved context), which is the gap Verity's NLI step already addresses for cited claims. ChatProtect's complement (35.2% of contradictions unverifiable via online text) is exactly Verity's residual failure mode; adopting the same-context alternate-sentence trick could backfill the no-citation case.

**URL.** https://arxiv.org/abs/2305.15852

---

## Self-Refine (2023, Madaan et al.)

**Mechanism.** Single LLM plays three roles in sequence: generator, feedback provider, refiner. Iterate until feedback signals stop or for a fixed budget (typically up to ~4 iterations in experiments). No second model, no debate. Same prompts work across tasks.

**Result.** "~20% absolute on average in task performance" over GPT-3.5/GPT-4 baselines across seven tasks (review rewriting, code optimisation, math reasoning, etc.).

**Notable vs Verity.** Pure self-loop, no cross-family signal. Verity's recompute pass is effectively a one-shot Self-Refine. Going to N iterations is cheap (one model) but the literature on self-refinement of factuality (vs style/reasoning) is much weaker, and Liang's "Degeneration-of-Thought" critique applies: a confident model rarely reverses itself.

**URL.** https://arxiv.org/abs/2303.17651

---

## Self-Consistency (2022, Wang et al.)

**Mechanism.** Same model, single prompt, K sampled chain-of-thought completions at temperature > 0 (K typically 5-40). Majority vote over extracted final answers. No iteration, no debate, no feedback. Aggregation is the only multi-call work.

**Result.** +17.9% on GSM8K, +11.0% SVAMP, +12.2% AQuA, +6.4% StrategyQA, +3.9% ARC-challenge over single-sample CoT.

**Notable vs Verity.** Verity's "consistency via re-sampling" is exactly this pattern at the critic stage. Lifting K beyond a small handful (K>=5) is well-studied and cheap on-device per call; the diminishing-returns curve is published. Direct knob to turn.

**URL.** https://arxiv.org/abs/2203.11171

---

## Replacing Judges with Juries / PoLL (2024, Verga et al.)

**Mechanism.** A panel of N smaller heterogeneous judge LLMs (paper uses Command R, GPT-3.5, Claude Haiku) scores each candidate; aggregation by max-vote or mean. No debate, no inter-judge visibility, no iteration. Single round, fully parallel.

**Result.** Panel matches or beats GPT-4-as-judge on six datasets across three judge settings, "over seven times cheaper than GPT-4," and reduces intra-model bias.

**Notable vs Verity.** This is the closest framework match to Verity's design philosophy: small, cross-family, parallel, cheap. Verity has N=2; PoLL evidence says 3-5 is the sweet spot. Adding a third critic (different family again) is the highest-evidence cost-effective extension. The bias-reduction argument is Verity's exact thesis.

**URL.** https://arxiv.org/abs/2404.18796

---

## ChatEval (2023, Chan et al.)

**Mechanism.** Multi-agent referee team: multiple LLM agents with diverse role personas (e.g. critic, scientist, public, news author) debate over multiple rounds about the quality of a generated response. Communication strategies include one-by-one, simultaneous, and group-discussion. Aggregation by final consensus or judge agent.

**Result.** Beats single-LLM evaluators on alignment with human preferences; diverse role prompts are necessary (identical roles degrade performance). ICLR 2024.

**Notable vs Verity.** Persona diversity within a single model family is an alternative to cross-family Verity uses. If Verity ever needs to drop to a single model (eg edge constraints), persona-diverse critics is the documented fallback. Less compute-efficient than PoLL but more transparent.

**URL.** https://arxiv.org/abs/2308.07201

---

## LLM-as-a-Judge / MT-Bench (2023, Zheng et al.)

**Mechanism.** Single strong LLM (typically GPT-4) scores or pairwise-compares responses on open-ended questions. One call per judgement, no debate, no panel. Paper documents and mitigates position, verbosity, and self-enhancement biases.

**Result.** GPT-4-judge achieves >80% agreement with human raters, matching inter-human agreement (max 82% on MT-Bench).

**Notable vs Verity.** Foundational frame for Verity's critic step. The named biases (position, verbosity, self-enhancement) are direct threats: Verity should randomise critic order and watch for length-bias in claim/citation pairs. The single-judge baseline is what PoLL and Verity beat by going multi-model.

**URL.** https://arxiv.org/abs/2306.05685

---

## FELM (2023, Chen et al.)

**Mechanism.** Benchmark, not a method. Annotates LLM-generated responses with fine-grained factuality labels, error types, and reference links. Evaluators tested include vanilla LLMs, retrieval-augmented LLMs, and CoT-augmented LLMs scoring claims as supported/contradicted.

**Result.** "Current LLMs are far from satisfactory to faithfully detect factual errors"; retrieval helps but does not solve. NeurIPS 2023 Datasets and Benchmarks.

**Notable vs Verity.** This is the benchmark Verity could be measured on. Critically, FELM's finding that even retrieval-augmented LLM evaluators underperform validates Verity's "critic + NLI + retrieval" stack over critic-only designs. Direct evaluation target.

**URL.** https://arxiv.org/abs/2310.00741

---

## Constitutional AI / RLAIF (2022, Bai et al.)

**Mechanism.** Two-phase training, not inference-time verification. SL phase: model self-critiques and revises its own responses against a written constitution; revisions become finetuning data. RL phase: model generates pairwise comparisons used as a preference signal for RLHF-style training. Inference is single-pass after training; the critique loop is offline.

**Result.** Trains a harmless-but-non-evasive assistant with far fewer human labels than RLHF. No single inference-time factuality number.

**Notable vs Verity.** Different problem: Constitutional AI bakes the critic into the generator weights; Verity keeps them separate at inference. Not directly adoptable as an architectural template, but the "written constitution" idea (Verity giving its critics an explicit rubric document instead of just "agree/disagree") is a cheap prompt-engineering import.

**URL.** https://arxiv.org/abs/2212.08073

---

## Reflexion (2023, Shinn et al.)

**Mechanism.** Single agent across multiple trials. After each trial, the agent receives a feedback signal (external scalar or self-generated text), writes a "reflection" into an episodic memory buffer, and uses that memory on the next trial. No model weight updates. Trial count up to ~10 in benchmarks.

**Result.** State-of-the-art on HumanEval pass@1 (91% with GPT-4), gains on HotpotQA and ALFWorld.

**Notable vs Verity.** Designed for sequential decision tasks with explicit reward, not one-shot claim verification. Not architecturally a fit for Verity's per-query budget. Listed for completeness; the "verbal memory" idea has minor relevance if Verity ever accumulates corpus-level priors about common hallucination patterns.

**URL.** https://arxiv.org/abs/2303.11366

---

## Reference-only, lower relevance

**Tree of Thoughts (Yao et al. 2023, https://arxiv.org/abs/2305.10601).** Tree-search over reasoning steps with self-evaluation at each node. NeurIPS 2023. Mostly a reasoning method; factuality gains only as a side effect. Not adoptable for one-shot verification — search cost is prohibitive on-device.

**AutoGen (Wu et al. 2023, https://arxiv.org/abs/2308.08155).** Framework, not a method. Provides conversation primitives for assembling multi-agent verification pipelines including those above. Now in maintenance mode; Microsoft Agent Framework is successor. Relevant only if Verity needs orchestration scaffolding.

**SelfCheckGPT (Manakul et al. 2023, https://arxiv.org/abs/2303.08896).** K-sample consistency check via BERTScore or MQAG between sampled responses. Conceptually identical to Verity's consistency-via-re-sampling step. EMNLP 2023.

**MAD-Fact (Ning et al. 2025, https://arxiv.org/abs/2510.22967).** Recent debate-based long-form factuality evaluator with weighted-claim aggregation. Frontiers of CS 2025. Confirms the debate-for-factuality direction is still active.

---

## Synthesis for Verity

**Cheapest high-evidence upgrade.** PoLL-style third critic (different family again) plus increased K on the consistency sampler. Both are parallel, well-benchmarked, on-device feasible.

**Highest-leverage architectural delta.** CoVe-style independent-question step in the recompute pass (prevents critics anchoring on draft) or LM vs LM examiner round (one critic probes the other's claim).

**Avoid.** Full Du-style debate (multi-round, every-agent-sees-everyone) — 3-5x cost for the documented gains. Self-Refine iteration on critic verdicts — Degeneration-of-Thought makes it unlikely to flip a confident wrong judgement.

**Bias hazards to address now.** Position, verbosity, self-enhancement bias (MT-Bench); single-family judge bias (Liang); critic-anchoring on draft (CoVe). All cheap to mitigate by prompt ordering, length normalisation, and the CoVe step-3 independence trick.
