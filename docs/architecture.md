# Architecture

The vertical slice has one execution path. `CaseSpec` validates intent; a tabular adapter loads data; audits produce structured evidence; split strategies persist checksummed index manifests; leakage-safe pipelines fit bounded probes; evaluation emits test metrics and per-run predictions; stability and rule-based capability functions consume those artifacts; rendering produces Markdown, JSON, HTML, Parquet tables, and static figures.

An optional molecular representation ladder is declared in `FeatureSpec`. The runner materializes each stateless feature frame once, creates each scenario/seed manifest once, and evaluates every representation through the same model suite and manifest. Representation names remain explicit in experiment, prediction, learning-curve, capability, ranking, and sensitivity artifacts. Matched-model medians keep representation effects separate from changes in the best model family.

Contracts point inward: adapters normalize source data, while the runner depends only on the normalized table and case. There is no one-implementation adapter interface. Add an adapter as a plain loader like `data/davis.py`, with schema checks and a source record, then expose it in the CLI.

The B3DB confirmation adapter is the minimal external-validation path: it pins and checksums the public source, normalizes cross-dataset identity to InChIKey, removes overlapping development identities, and emits one combined table with supplied train/holdout assignments. The case fingerprint freezes the representation, model allowlist, metric, threshold, class-support floor, and access limit.

Invariant Autoprobe components are the loader, case, manifests, evaluator, metrics, holdout ledger, and provenance. Its only mutable input is `candidate.yaml`. Generated JSON/Parquet artifacts—not narrative text—are numerical truth.

Future AnnData and LINCS loaders should materialize an analysis table plus representation/artifact references. Future multi-output tasks can add a target-matrix contract and metric aggregation while retaining split, independence, holdout, and capability interfaces.

A learned graph representation is not part of the v0.1 feature-frame contract because it must fit only on training folds rather than materialize once from the full table. Add that separate fit/transform seam only after bounded character/Morgan evidence fails a decision-relevant boundary, independent-compound coverage can support the added capacity, and locked external evidence is available to test the extension.
