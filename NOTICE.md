# Third-party datasets used for benchmarking

Verity is benchmarked against the following public hallucination-detection datasets. Each is downloaded by the user under the licence below and stored outside this repository (no dataset content is committed). Required attributions follow.

## RAGTruth

- **Licence**: MIT.
- **Source**: https://github.com/ParticleMedia/RAGTruth
- **Citation**:

> Wu, Y., Zhu, J., Xu, S., Shum, K., Niu, C., Zhong, R., Song, J., & Zhang, T. (2024). RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models. arXiv:2401.00396.

## HaluEval

- **Licence**: MIT.
- **Source**: https://github.com/RUCAIBox/HaluEval
- **Citation**:

> Li, J., Cheng, X., Zhao, W. X., Nie, J.-Y., & Wen, J.-R. (2023). HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models. arXiv:2305.11747.

## ANAH

- **Licence**: Apache 2.0.
- **Source**: https://huggingface.co/datasets/opencompass/anah
- **Citation**:

> Ji, Z., Gu, Y., Zhang, W., Lyu, C., Lin, D., & Chen, K. (2024). ANAH: Analytical Annotation of Hallucinations in Large Language Models. Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Vol. 1: Long Papers), pp. 8135-8158.

## Datasets considered and not downloaded

These appear in the benchmark harness for completeness but are not currently downloaded as part of Verity's default benchmark set, for the licence reasons below.

### LLM-AggreFact

- **Licence**: CC-BY-ND-4.0. Evaluation use is explicitly permitted.
- **Why not auto-downloaded**: HuggingFace gates the download behind a contact form. Users wishing to run `bench:aggrefact` must visit `https://huggingface.co/datasets/lytang/LLM-AggreFact`, accept the terms, and download with their HuggingFace credentials.
- **Citation**:

> Tang, L., Laban, P., & Durrett, G. (2024). MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents. Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. arXiv:2404.10774.

### FaithBench

- **Licence**: CC-BY-NC-SA-4.0 (Attribution + NonCommercial + ShareAlike).
- **Why not auto-downloaded**: The NonCommercial clause exceeds attribution-only and is not auto-fetched. Users wishing to run `bench:faithbench` for non-commercial research may download from `https://github.com/vectara/FaithBench` themselves and observe the licence terms.
- **Citation**:

> Bao, F. S., Li, M., Qu, R., Luo, G., Wan, E., Tang, Y., Fan, W., Tamber, M. S., Kazi, S., Sourabh, V., Qi, M., Tu, R., Xu, C., Gonzales, M., Mendelevitch, O., & Ahmad, A. (2025). FaithBench: A Diverse Hallucination Benchmark for Summarization by Modern LLMs. Proceedings of NAACL HLT 2025.
