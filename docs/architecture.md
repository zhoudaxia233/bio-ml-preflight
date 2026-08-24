# Architecture

The vertical slice has one execution path. `CaseSpec` validates intent; a tabular adapter loads data; audits produce structured evidence; split strategies persist checksummed index manifests; leakage-safe pipelines fit bounded probes; evaluation emits test metrics and per-run predictions; stability and rule-based capability functions consume those artifacts; rendering produces Markdown, JSON, HTML, Parquet tables, and static figures.

Contracts point inward: adapters normalize source data, while the runner depends only on the normalized table and case. There is no one-implementation adapter interface. Add an adapter as a plain loader like `data/davis.py`, with schema checks and a source record, then expose it in the CLI.

Invariant Autoprobe components are the loader, case, manifests, evaluator, metrics, holdout ledger, and provenance. Its only mutable input is `candidate.yaml`. Generated JSON/Parquet artifacts—not narrative text—are numerical truth.

Future AnnData and LINCS loaders should materialize an analysis table plus representation/artifact references. Future multi-output tasks can add a target-matrix contract and metric aggregation while retaining split, independence, holdout, and capability interfaces.

