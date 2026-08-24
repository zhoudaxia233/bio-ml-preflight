# Methodology

## Inspection before modeling

Inventory covers schema, missingness, invalid values, constants, duplicate rows/identifiers, target distribution, and fingerprints. Independence reports entity counts and repetition as transparent proxies; it never labels them an exact effective sample size. Replicate or repeated-pair dispersion is a noise warning, not a fabricated ceiling.

Coverage includes declared entities and available batch, plate, time, and treatment fields. Pairwise coverage reports unique entities, density, degrees, repeated pairs, and conflicting labels.

## Validation

Each deterministic manifest separates rows, groups, pair entities, both pair dimensions, time, or supplied assignments. Every manifest receives exact, entity, and pair-overlap diagnostics. Similarity overlap is `NOT_ASSESSABLE` unless an explicit similarity function exists. A future DataSAIL provider can supply similarity-aware assignments.

Preprocessing is fitted inside the training pipeline. Random splitting remains a diagnostic. Entity, double-cold, and time splits are claims about distinct deployment boundaries, not interchangeable difficulty settings.

Entity identity conflicts are resolved before feature generation through an explicit `keep`, `exclude`, or `aggregate` policy. Aggregation refuses inconsistent representations, varying modeled features, and tied binary labels. Group/scaffold scenarios stop before fitting when the identity promised as held out would cross folds.

## Evaluation and decisions

Regression reports MAE, RMSE, Pearson, Spearman, and Kendall. Classification reports balanced accuracy, ROC-AUC and average precision when valid, plus log loss. Grouped rankings add Spearman and NDCG where numeric target relevance supports it. Top-k Jaccard, selection probabilities, unstable-membership fraction, and rank standard deviation expose decisions hidden by global error.

Group-respecting permutations move label blocks between equal-sized independent groups (or shuffle within the sole block). Each split/model now samples nine deterministic null draws. Capability evidence reports the null median, 95th percentile, and the corrected empirical upper-tail p-value instead of treating one chance null fit as representative. Bootstrap intervals resample configured independent units. Learning curves should use increasing counts of independent groups; v0.1 exposes the resampling primitive but does not claim a curve when group coverage is too small.

Capability thresholds live in the case. Verdict rules compare the best controlled baseline, permutation delta, split dispersion, ranking overlap, and random-versus-deployment scenario behavior. Each output carries supporting and opposing evidence, uncertainty, unmet assumptions, numbers, and the cheapest next evidence.

Capability verdicts also consume structured audit and overlap results. Conflicting targets for an explicitly declared single-entity prediction unit, inconsistent entity representations, exact-record overlap, pair overlap, or overlap of the entity promised as held out can cap an otherwise `SUPPORTED` result at `SUPPORTED_WITH_LIMITS`. These findings never upgrade weak model evidence. Measurement reliability and required-but-missing metadata receive separate `NOT_ASSESSABLE` rows so their uncertainty is visible without silently changing an unrelated predictive claim.

For a declared SMILES ladder, character hashes and Morgan fingerprints reuse identical manifests, seeds, target permutations, and model families. Reports show per-representation capability verdicts plus matched-model medians; a change in the best model is therefore visible rather than mislabeled as a pure representation effect. The conservative sensitivity status is the weakest per-representation verdict.
