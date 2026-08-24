# References

- Karpathy, [autoresearch](https://github.com/karpathy/autoresearch). Primary repository for the bounded mutable-training-file pattern; this project changes its single-metric loop into a guarded evidence vector.
- Dwork et al. (2015), [The reusable holdout](https://pubmed.ncbi.nlm.nih.gov/26250683/) and [Generalization in Adaptive Data Analysis and Holdout Reuse](https://papers.neurips.cc/paper_files/paper/2015/hash/bad5f33780c42f2588878a9d07405083-Abstract.html). Motivation for holdout-access discipline under adaptive analysis.
- Kapoor & Narayanan (2023), [Leakage and the Reproducibility Crisis in ML-based Science](https://arxiv.org/abs/2207.07048), plus Greener et al. (2022), [A guide to machine learning for biologists](https://www.nature.com/articles/s41580-021-00407-0). Leakage and biological ML evaluation guidance.
- Joeres et al. (2025), [Data splitting to avoid information leakage with DataSAIL](https://www.nature.com/articles/s41467-025-58606-8) and [official documentation](https://datasail.readthedocs.io/en/latest/). Future similarity-aware split integration.
- Bingham et al. (2024), [Supervised discovery of interpretable gene programs from single-cell data (Spectra)](https://doi.org/10.1038/s41587-023-01940-3). Example of a method whose biological assumptions cannot be inferred from a generic table.
- Huang et al. (2021), [Therapeutics Data Commons](https://arxiv.org/abs/2102.09548) and [official TDC documentation](https://tdc.readthedocs.io/en/main/). Source and loader for the Davis demo.
- Peidli et al. (2024), [OP3 perturbation-prediction benchmark](https://proceedings.neurips.cc/paper_files/paper/2024/file/24c4d51f3ef48dd2dbab78243ecb26a1-Paper-Datasets_and_Benchmarks_Track.pdf) and [Open Problems benchmarks](https://openproblems.bio/benchmarks/). Reference for future multi-output perturbation evaluation.
- OpenAI, [Skills API documentation](https://developers.openai.com/api/reference/python/resources/skills/methods/create) and [Codex use cases](https://developers.openai.com/codex/use-cases). Official skill and Codex documentation; the repository-scoped skill follows the local `SKILL.md` convention supported by Codex environments.

