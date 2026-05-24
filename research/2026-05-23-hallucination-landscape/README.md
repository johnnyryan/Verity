# Hallucination-detection landscape, 2026-05-23

Background research for Verity, run as five parallel research agents.

Five reports, one synthesis. Reports are unedited (terse Orwell style, UK English, mechanism-grounded summaries).

## Reports

- `01-academic-papers.md` — peer-reviewed and arXiv. 18 entries.
- `02-open-source.md` — GitHub-hosted libraries and frameworks. 12 entries.
- `03-commercial.md` — closed-source SaaS products. 12 entries plus honourable mentions.
- `04-multi-agent-debate.md` — systems where multiple LLM invocations or rounds drive verification. 13 entries.
- `05-benchmarks.md` — hallucination-detection datasets. 15 entries.

## Synthesis

### Headline finding

**PoLL — "Replacing Judges with Juries"** (Verga et al. 2024, arXiv:2404.18796) is Verity's exact thesis published. Panel of N small heterogeneous judge LLMs, parallel, no debate, no inter-judge visibility. Matches or beats GPT-4-as-judge at "over seven times cheaper". Their evidence says **N = 3 to 5 is the sweet spot**. Verity is at N = 2.

Closest other architectural sibling is **Cleanlab Trustworthy Language Model** — the only commercial product surveyed that combines three of Verity's four signals (consistency + self-reflection + logprobs). Missing: the cross-family critic stage and the explicit NLI cross-encoder.

### The niche Verity occupies

The combination of (cross-family critics + NLI + logprob band + consistency resample) is **not done elsewhere**:

- No OSS project surveyed combines more than two of the four.
- No commercial product surveyed combines more than three. The one that gets to three (Cleanlab TLM) is missing the cross-family signal.
- Two products (Aporia, Trustwise) gesture at multi-SLM ensembles but will not disclose model families.

This appears to be a genuinely under-occupied niche.

### What to add (priority order)

#### Cheap, high evidence

1. **Third critic, different family again.** PoLL evidence: N = 3 to 5 is the sweet spot; we are at 2. Same architecture, one more model load. Candidate: Llama-3-derived (e.g. Patronus Lynx 8B for the IBM/Mistral/Meta trio), or a smaller Qwen if we want Alibaba in the mix.
2. **CoVe step-3 independence trick in the recompute pass.** Dhuliawala et al. (2023, arXiv:2309.11495) showed that critics anchor on the draft if they can see it. Make the recompute step answer verification questions *without* the draft in context. Pure prompt-engineering change.
3. **Higher K on the consistency sampler.** Wang et al. Self-Consistency (arXiv:2203.11171) and SelfCheckGPT (arXiv:2303.08896) both publish diminishing-returns curves. K ≥ 5 is the well-documented sweet spot. Currently we run small N.

#### Higher leverage architectural deltas

4. **Semantic entropy** (Farquhar et al., Nature 2024). Cluster samples by *meaning* (using the NLI head Verity already has loaded), then compute Shannon entropy over the cluster sizes. Better signal than token-level perplexity for confabulation. Could supplement or replace the 4-band confidence classifier.
5. **Conformal calibration** of the pass/warn/error thresholds. Yadkori et al. 2024 (arXiv:2405.01563) and Quach et al. 2023 (arXiv:2306.10193). Gives the thresholds a provable error bound on a held-out calibration set. Pure statistics, no new models.
6. **LM-vs-LM examiner round** (Cohen et al. 2023, arXiv:2305.13281). One critic generates a probe question for the other to answer, then both judge. Adversarial signal without doubling cost.

#### Bias hazards to fix now (basically free)

- **Randomise critic order** to defeat position bias (MT-Bench finding, Zheng et al. 2023).
- **Length-normalise verdicts** to defeat verbosity bias (MT-Bench).
- The CoVe step-3 trick above also covers the critic-anchoring case (Dhuliawala et al.).
- **Watch for single-family judge bias** if a third critic is added — keep families distinct.

#### Benchmarks to add to the harness

Already deferred in `design.md` § 11: **RAGTruth**, **LLM-AggreFact**.

Additional fits worth adding:

- **FaithBench** (Vectara 2024, arXiv:2410.13210). Small (~700 summaries), hard. "Even the best detectors have near 50% accuracy" — a stress test, not a leaderboard chase.
- **HaluEval** (Li et al. 2023, arXiv:2305.11747). 30k examples across QA, dialogue, summarisation.
- **ANAH-v2** (Shanghai AI Lab 2024, arXiv:2405.20315). Sentence-level, bilingual, 4-way labels needing a small mapping.

Skip: TruthfulQA, SimpleQA, HELM. Format mismatch — multiple choice or open-domain with no context.

### What is out of reach (and why)

The entire internal-state probe family — ITI, RepE, CCS, SAPLMA, DoLa, Lookback Lens, and HalluGuard's spectral and NTK methods — requires attention matrices or gradients. Verity speaks OpenAI-compatible HTTP. The richest signal we get over that wire is logprobs, which is what the confidence classifier already uses. **Logprobs are the ceiling for an API-only tool. We are at it.**

### What is overrated for our use case

- **Full Du-style multi-agent debate** (multiple rounds, every agent sees everyone). 3–5x cost for documented but modest factuality gains. PoLL beats it at lower cost.
- **Self-Refine iteration on critic verdicts.** Liang's "Degeneration-of-Thought" — a confident model rarely reverses itself.
- **Constitutional AI / RLAIF.** Different problem (training-time), not inference-time verification. The "written constitution" prompt idea is borrowable, but the training pipeline is not.
- **Most commercial cloud gateways** (Bedrock, Azure, Vertex). Opaque internals, no logprob exposure, no public benchmark numbers.

### Closest sibling, by category

| Slice | Closest sibling | Mechanism overlap |
|---|---|---|
| Academic methods | PoLL (Verga 2024) | Exact match: small heterogeneous parallel judges. |
| Open source | RefChecker (Amazon Science) | Decompose-then-verify with NLI; missing cross-family + logprobs. |
| Open source NLI module | HHEM-2.1-Open (Vectara) | Could drop-in replace Verity's NLI cross-encoder; CPU-fine. |
| Commercial overall | Cleanlab TLM | Consistency + self-reflection + logprobs; missing cross-family + NLI. |
| Commercial NLI | Galileo Luna | DeBERTa-large span-level — Verity's NLI module sold standalone. |
| Multi-agent | PoLL (same as academic) | Same. |
| Method we already do | SelfCheckGPT-NLI | Identical to Verity's consistency stage; adoptable as a graft into recompute. |

### Sources

All entries in the five report files carry direct URLs (arXiv, GitHub, vendor docs). Five reports total roughly 12,000 words of mechanism summaries.

---

Produced 2026-05-23 by five parallel research agents. No code changes were made during research.
