# Methodology

## Researcher and assistant workflow

The tool assists reviews of biological prediction and ranking claims. It supplies repeatable
computations and provenance; it does not decide what a biological observation means.

1. The researcher and assistant read the paper's methods and data documentation. Record the
   original claim, the narrower claim accessible in the public data, and any diagnostic comparison.
   Confirm prediction units (for example, patient visits), independent units (patients), and
   contexts (plates, sites or sampling times). Column names alone do not establish these roles.
2. Use the existing case declaration and `eda` to check those assumptions against the table.
   An unseen-patient claim requires different identities; comparing the same treatment across
   plates requires shared treatment identities. Missing semantics remain unresolved.
3. Freeze exclusions, splits, metrics and a bounded baseline before inspecting evaluation scores.
   Use `run --budget smoke`; obey holdout access limits. Do not tune the claim to the result.
4. Read JSON and Parquet evidence alongside the saved manifests. Check actual partition counts,
   class support and the executed permutation method before interpreting scores or statuses.
   Use saved predictions for diagnosis instead of repeatedly accessing evaluation labels.
5. The assistant connects these facts to the source evidence and the researcher reviews biological
   assumptions: replicate provenance, valid label exchanges, assay comparability, confounding,
   target relevance and deployment scope. Separate tool defects from task errors and missing data.
   `SUPPORTED` is conditional on configured checks; it is not an automatic paper verdict.

## Inspection before modeling

Inventory covers schema, missingness, invalid values, constants, duplicate rows/identifiers, target distribution, and fingerprints. Independence reports entity counts and repetition as transparent proxies; it never labels them an exact effective sample size. Replicate or repeated-pair dispersion is a noise warning, not a fabricated ceiling.

Coverage includes declared entities and available batch, plate, time, and treatment fields. Pairwise coverage reports unique entities, density, degrees, repeated pairs, and conflicting labels.

## Validation

Each deterministic manifest separates rows, groups, pair entities, both pair dimensions, time, or supplied assignments. Every manifest receives exact, entity, and pair-overlap diagnostics. Similarity overlap is `NOT_ASSESSABLE` unless an explicit similarity function exists. A future DataSAIL provider can supply similarity-aware assignments.

Preprocessing is fitted inside the training pipeline. Random splitting remains a diagnostic. Entity, double-cold, and time splits are claims about distinct deployment boundaries, not interchangeable difficulty settings.

Entity identity conflicts are resolved before feature generation through an explicit `keep`, `exclude`, or `aggregate` policy. Aggregation refuses inconsistent representations, varying modeled features, and tied binary labels. Group/scaffold scenarios stop before fitting when the identity promised as held out would cross folds.

## Evaluation and decisions

Regression reports MAE, RMSE, Pearson, Spearman, and Kendall. Classification reports balanced accuracy, ROC-AUC and average precision when valid, plus log loss. Grouped rankings add Spearman and NDCG where numeric target relevance supports it. Top-k Jaccard, selection probabilities, unstable-membership fraction, and rank standard deviation expose decisions hidden by global error.

With a declared `bootstrap_unit`, permutations move label blocks between equal-sized groups
and shuffle labels within each receiving group, including a group with no equal-sized peer.
Without it, permutations shuffle rows. Runs persist the executed method and unit; older results
without that record are described as unrecorded, not assumed to be grouped. Neither method
establishes exchangeability. In particular, within-group shuffling does not preserve longitudinal
order or visit alignment; the researcher must justify valid exchanges for the biological question.
Each split/model defaults to nine deterministic null draws. Capability evidence reports the null
median, 95th percentile, and corrected empirical p-value. The test counts null results at least as
good as the observed metric, including ties: upper tail for correlations/AUC/accuracy, lower tail
for MAE/RMSE/log loss. Error-metric verdicts also expose the raw 5th percentile.

The runner evaluates learning curves on increasing subsets of whole training groups for group and cold-entity scenarios, keeping the test partition fixed. It skips curves with fewer than four training groups and subsets that fail readiness checks. Reports do not currently include bootstrap confidence intervals; the configured `bootstrap_unit` identifies independent units for permutation controls, overlap audits, and class-support counts.

Capability thresholds live in the case. Verdict rules compare the best controlled baseline,
permutation delta, across-run dispersion, ranking overlap, and random-versus-deployment scenario
behavior. Each output carries supporting and opposing evidence, uncertainty, unmet assumptions,
numbers, and the cheapest next evidence. Partition membership fingerprints ignore seed and row
order without changing the manifest checksums used by holdout locks. Split variation uses one
median per distinct partition within the same scenario, representation and model. Fewer than two
partitions, or missing membership records, gives `NOT_ASSESSABLE`. Multiple partitions describe
observed variation; they do not isolate split effects from training initialization or provide a
confidence interval.
With only one finite run, across-run dispersion is also unavailable rather than zero.

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

Class support counts distinct `(independent unit, class)` pairs. One patient may contribute visits
to both classes; counts are not disjoint cohorts or effective sample sizes. Label consistency is
checked at the declared prediction unit, separately from the independence unit.

Capability verdicts also consume structured audit and overlap results. Conflicting targets for an
explicitly declared single-entity prediction unit, inconsistent entity representations, exact-record
overlap, unexpected pair overlap, or overlap of a promised held-out entity can cap `SUPPORTED` at
`SUPPORTED_WITH_LIMITS`. A compatible explicit `same_entity_across_context` claim treats overlapping
pairs containing that entity as expected, while keeping their raw counts and audited pair columns.
This exception does not suppress exact-record or protected-entity findings, apply to unrelated pairs,
or establish measurement reliability. Older audits without pair-column evidence keep the conservative
overlap treatment. Measurement reliability and required-but-missing metadata receive separate
`NOT_ASSESSABLE` rows. These findings never upgrade weak model evidence.
The random-split unseen-unit warning likewise does not apply when every audited partition is
compatible with an explicit cross-context comparison of that same independent unit. Raw overlap
counts remain available; overlapping identities of other protected units still require review.

For a declared SMILES ladder, character hashes and Morgan fingerprints reuse identical manifests, seeds, target permutations, and model families. Reports show per-representation capability verdicts plus matched-model medians; a change in the best model is therefore visible rather than mislabeled as a pure representation effect. The conservative sensitivity status is the weakest per-representation verdict.
