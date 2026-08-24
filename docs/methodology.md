# Methodology

## Inspection before modeling

Inventory covers schema, missingness, invalid values, constants, duplicate rows/identifiers, target distribution, and fingerprints. Independence reports entity counts and repetition as transparent proxies; it never labels them an exact effective sample size. Replicate or repeated-pair dispersion is a noise warning, not a fabricated ceiling.

Coverage includes declared entities and available batch, plate, time, and treatment fields. Pairwise coverage reports unique entities, density, degrees, repeated pairs, and conflicting labels.

## Validation

Each deterministic manifest separates rows, groups, pair entities, both pair dimensions, time, or supplied assignments. Every manifest receives exact, entity, and pair-overlap diagnostics. Similarity overlap is `NOT_ASSESSABLE` unless an explicit similarity function exists. A future DataSAIL provider can supply similarity-aware assignments.

Preprocessing is fitted inside the training pipeline. Random splitting remains a diagnostic. Entity, double-cold, and time splits are claims about distinct deployment boundaries, not interchangeable difficulty settings.

## Evaluation and decisions

Regression reports MAE, RMSE, Pearson, Spearman, and Kendall. Classification reports balanced accuracy, ROC-AUC and average precision when valid, plus log loss. Grouped rankings add Spearman and NDCG where numeric target relevance supports it. Top-k Jaccard, selection probabilities, unstable-membership fraction, and rank standard deviation expose decisions hidden by global error.

Group-respecting permutations move label blocks between equal-sized independent groups (or shuffle within the sole block). Each split/model now samples nine deterministic null draws. Capability evidence reports the null median, 95th percentile, and the corrected empirical upper-tail p-value instead of treating one chance null fit as representative. Bootstrap intervals resample configured independent units. Learning curves should use increasing counts of independent groups; v0.1 exposes the resampling primitive but does not claim a curve when group coverage is too small.

Capability thresholds live in the case. Verdict rules compare the best controlled baseline, permutation delta, split dispersion, ranking overlap, and random-versus-deployment scenario behavior. Each output carries supporting and opposing evidence, uncertainty, unmet assumptions, numbers, and the cheapest next evidence.
