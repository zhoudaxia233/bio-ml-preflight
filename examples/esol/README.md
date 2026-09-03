# ESOL: a public molecular-regression workflow check

This example uses the 1,128-row Delaney/ESOL table distributed by the
[official DeepChem loader](https://github.com/deepchem/deepchem/blob/master/deepchem/molnet/load_function/delaney_datasets.py).
It tests the evidence-audit workflow, not a new solubility model or a leaderboard claim.
The source is [Delaney (2004)](https://doi.org/10.1021/ci034243x); the benchmark is
[MoleculeNet](https://doi.org/10.1039/C7SC02664A).

From the repository root:

```bash
uv sync --python 3.11 --extra dev --extra chem
uv run python examples/esol/prepare.py
uv run bio-ml-preflight validate-case examples/esol/case.yaml
uv run bio-ml-preflight eda examples/esol/case.yaml --output reports/esol-eda
uv run bio-ml-preflight run examples/esol/case.yaml --budget smoke --output reports/esol-smoke
```

Preparation downloads about 97 KB once, verifies a pinned SHA-256 on every use,
and writes the analysis CSV and `source.json` under ignored `data/cache/esol/`.
Raw data are not distributed with this repository. The benchmark offers a public
download, but an explicit standalone redistribution license for this CSV was not
located; the DeepChem software license is not asserted as the data license.
No DeepChem installation or dataset-specific runtime adapter is required.

## Fixed protocol and interpretation

- Target: measured aqueous log10 solubility in mol/L, without label rescaling.
- Inputs: Morgan fingerprints of the supplied molecular structures only. Source
  descriptors and the source model's predicted solubility remain audit context.
- Identity: canonical supplied isomeric SMILES, retaining the original text and
  every record. This does not reconstruct unspecified stereochemistry or establish
  exact chemical identity. RDKit version and both data checksums are recorded.
- The pinned table has 1,117 distinct supplied structures; 11 representation
  groups repeat (22 rows), and 6 have different measured targets. The case explicitly
  keeps these observations with limits; it does not average, delete or call them
  verified experimental replicates. Batch, measurement conditions and replicate
  protocols are unavailable.
- Probes: existing dummy, Elastic Net and Extra Trees; seeds 11 and 23; nine
  training-label permutations per fitted non-dummy probe. No post-result tuning.
- Scenarios: a 25% random-record diagnostic and 25% scaffold-group sampling using
  the repository's `GroupShuffleSplit` implementation. Actual scaffold test row
  fractions can differ greatly from 25% because scaffold groups differ in size.
  Acyclic molecules share the empty Murcko scaffold and remain one group.
- RMSE is primary; MAE and Spearman are secondary. The raw RMSE thresholds 1.0/1.5,
  dispersion limit 0.25 and permutation-improvement minimum 0.15 are predeclared
  workflow bounds, not published scientific acceptance standards.
- The original MoleculeNet paper used random splitting for ESOL; the current
  DeepChem loader defaults to scaffold splitting. Our sampling protocol is explicit
  but does not reproduce either publication's exact manifests or leaderboard scores.
- This is retrospective development evaluation. There is no consumed holdout,
  deployment claim, causal conclusion, or requirement for a favorable verdict.

Read `reports/esol-smoke/report.json`, `aggregate_experiments.parquet` and
`capability_matrix.parquet` before interpreting Markdown/HTML. Retain the generated
split manifests with the result; a rerun is not additional independent evidence.

## Observed smoke result

In the verified local run, Extra Trees had the lower error of the two fitted
non-dummy probes in both scenarios:

| Scenario | Median RMSE | Null median RMSE | Across-split SD | Verdict |
| --- | ---: | ---: | ---: | --- |
| Random-record diagnostic | 1.547 | 2.578 | 0.071 | INSUFFICIENT_EVIDENCE |
| Unseen scaffold | 1.765 | 2.643 | 0.323 | INSUFFICIENT_EVIDENCE |

Both corrected permutation p-values were 0.10, the nine-draw resolution floor.
There is baseline signal, but the fixed probes miss the predeclared RMSE bound;
scaffold variation also exceeds the dispersion bound. This does not establish
that the dataset is unlearnable. Random splits repeat 3/7 supplied structures
across train/test; scaffold splits repeat none, with test sizes 138/149.
Measurement reliability and batch confounding remain `NOT_ASSESSABLE`.

All 84 prediction artifacts were finite and their RMSE/MAE independently matched
the aggregate table within 9e-16. The local NumPy 2.2.6 / scikit-learn 1.9.0 stack
emitted matrix-product warnings during Elastic Net fitting. On an unchanged
development-fold diagnostic refit, independent scalar summation reproduced the
intercept and exported predictions within 8e-15; warnings were not suppressed.
