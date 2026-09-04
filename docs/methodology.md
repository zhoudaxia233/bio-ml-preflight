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

Group-respecting permutations move label blocks between equal-sized independent groups (or shuffle within the sole block). Each split/model now samples nine deterministic null draws. Capability evidence reports the null median, 95th percentile, and the corrected empirical p-value instead of treating one chance null fit as representative. The test counts null results at least as good as the observed metric, including ties: upper tail for correlations/AUC/accuracy, lower tail for MAE/RMSE/log loss. Error-metric verdicts also expose the raw 5th percentile. Bootstrap intervals resample configured independent units. Learning curves should use increasing counts of independent groups; v0.1 exposes the resampling primitive but does not claim a curve when group coverage is too small.

Capability thresholds live in the case. Verdict rules compare the best controlled baseline, permutation delta, split dispersion, ranking overlap, and random-versus-deployment scenario behavior. Each output carries supporting and opposing evidence, uncertainty, unmet assumptions, numbers, and the cheapest next evidence.

Low or unstable baseline performance alone does not identify its cause or establish that more independent samples would help. Next-evidence advice starts with existing development predictions and the declared task and split boundary; a predeclared class-support deficit can still justify a concrete request for additional independent examples. Class-support advice preserves conflict, overlap, and holdout safeguards, and consumed holdouts must not be adapted to or rerun.

MAE, RMSE and log loss are minimized; other supported primary metrics are maximized.
Thresholds remain in raw metric units: for error metrics, `supported_metric` is a stricter
maximum than `limited_metric` (for example, RMSE 1.0 and 1.5). For higher-is-better metrics
they are minimums, with the supported threshold higher. A positive `permutation_delta`
always means improvement: null minus observed error, or observed minus null score.
This metric direction is separate from `task.higher_is_better`, which orders target values
for ranking decisions. Nine draws only resolve p-values down to 0.10; benchmark use does
not turn smoke diagnostics into confirmatory inference.

Confirmatory cases can freeze a model allowlist in the case rather than select the best model from holdout outcomes. A supplied holdout may name a protected entity column; any train/test identity crossing then stops before fitting or manifest persistence. Enabled holdouts record access before target-dependent work in a stable ledger keyed by the dataset checksum, with the case fingerprint stored in every event; changing the report output or case parameters cannot bypass the limit, and an override requires an explicit audited reason. Binary cases may predeclare a minimum test count per class and the independent unit used for that count; a point estimate cannot pass the capability boundary when either holdout class falls below it.

Capability verdicts also consume structured audit and overlap results. Conflicting targets for an explicitly declared single-entity prediction unit, inconsistent entity representations, exact-record overlap, pair overlap, or overlap of the entity promised as held out can cap an otherwise `SUPPORTED` result at `SUPPORTED_WITH_LIMITS`. These findings never upgrade weak model evidence. Measurement reliability and required-but-missing metadata receive separate `NOT_ASSESSABLE` rows so their uncertainty is visible without silently changing an unrelated predictive claim.

For a declared SMILES ladder, character hashes and Morgan fingerprints reuse identical manifests, seeds, target permutations, and model families. Reports show per-representation capability verdicts plus matched-model medians; a change in the best model is therefore visible rather than mislabeled as a pure representation effect. The conservative sensitivity status is the weakest per-representation verdict.
