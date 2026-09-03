# bio-ml-preflight

[中文说明](README.zh-CN.md)

`bio-ml-preflight` asks what a dataset and a proposed prediction or ranking claim can support, where that evidence stops, and what inexpensive evidence would reduce the most important uncertainty. It is an evidence audit and bounded baseline runner—not AutoML, a leaderboard, a biological hypothesis generator, or proof of future performance.

## Quick start

Python 3.11+ and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --extra dev
uv run bio-ml-preflight --help
uv run bio-ml-preflight demo synthetic --budget smoke
```

On macOS environments that hide editable `.pth` files, use the one-command checkout target instead:

```bash
make demo
```

Reports land under `reports/` and contain Markdown, HTML, JSON, figures, split manifests, per-run Parquet predictions, aggregate tables, a representation-sensitivity table, a capability matrix, and provenance.

For an existing table:

```bash
uv run bio-ml-preflight init-case DATA.csv --output case.yaml
# Review and confirm every provisional role in case.yaml.
uv run bio-ml-preflight validate-case case.yaml
uv run bio-ml-preflight run case.yaml --budget smoke
```

With no claim, use discovery mode:

```bash
uv run bio-ml-preflight discover DATA.csv --output discovery/
```

Discovery produces candidate *testable analysis tasks*. It does not infer mechanisms from column names, promote associations to causal claims, or perform “biological discovery.” Every inferred role remains unconfirmed until a researcher reviews it.

After confirming a case, inspect development-data quality without fitting a model:

```bash
uv run bio-ml-preflight eda case.yaml --output reports/example-eda
```

EDA writes `eda.json` and `column_profile.parquet` as structured sources of truth, plus
concise Markdown/HTML reports and missingness, target-distribution, and entity-repetition
figures. It reports identity/label conflicts, repeated entities, leakage candidates,
measurement limits, and missing high-value metadata without deleting outliers or emitting a
universal quality score. Holdout-enabled cases are refused before their tables are read;
non-training rows in an ordinary supplied split are excluded from the profile.

The findings also produce a non-scoring pre-model decision: `READY` when no limitation is
detected, `READY_WITH_LIMITS` when modeling may proceed with explicit warnings or
`NOT_ASSESSABLE` boundaries, and `BLOCKED` when an `ACTION_REQUIRED` issue must be resolved.
`run` enforces the same decision before feature construction and model fitting. Examples of
blocking issues are unconfirmed roles, missing targets or entity IDs, unresolved identity
conflicts, infinite modeled values, and target or post-outcome leakage. Ordinary feature
missingness remains a warning because imputation can stay inside training folds. For a locked
holdout run, declaration-only blockers are checked before access; after the access ledger is
updated, the data-dependent readiness audit uses only supplied training rows. Identity policies
likewise cannot use outcomes from rows reserved by a purely supplied evaluation. Binary targets
must use numeric `0` and `1`, matching the current metric and probability contract. Regression
and ranking targets must be numeric with at least two finite values. Each reserved evaluation
partition receives the same executable data-contract check after any logged holdout access and
before feature construction; its outcomes are never used to select or remove rows. Rows excluded
by every split manifest do not enter run audits, feature construction, predictions, or capability
verdicts.

## Why the case matters

Suitability is conditional on the target, prediction unit, deployment population, generalization axis, decision rule, and validation design. Ten thousand measurements from ten patients are not ten thousand independent samples. Random row splits can place the same patient, molecule, target, batch, or near-duplicate in both sets; they are useful diagnostics but cannot establish unseen-entity performance.

Ranking decisions need decision-level evidence. Similar RMSE values can conceal different top candidates, so reports include top-k overlap, selection probability, and rank variability. No single feasibility score is emitted. Each scenario receives one transparent status: `SUPPORTED`, `SUPPORTED_WITH_LIMITS`, `INSUFFICIENT_EVIDENCE`, `CONTRADICTED`, or `NOT_ASSESSABLE`, with the numbers and assumptions behind it.

## Case schema and validation

The versioned YAML schema is implemented with Pydantic v2 in `contracts/case.py`. Missing scientific metadata is not guessed. Its downstream question becomes `NOT_ASSESSABLE` or explicitly limited. Learned imputation, encoding, and scaling stay inside scikit-learn training pipelines. Split manifests are checksummed and reused.

The baseline suite is deliberately bounded: dummy, linear/logistic, regularized linear, extra trees, histogram gradient boosting, and nearest neighbours. Smoke budgets use a strict subset. Pairwise cases also get entity-mean baselines and deterministic character/sequence representations. A case can explicitly declare a SMILES column and compare the dependency-free character hash with RDKit Morgan fingerprints under identical split manifests and model families. `uv sync --extra chem` enables Morgan fingerprints and scaffold splitting; the base package needs no GPU framework.

## Davis demo

The dataset is never committed. The official TDC loader writes a checksummed cache and source record under gitignored `data/cache/`.

```bash
uv sync --python 3.11 --all-extras
uv run --python 3.11 bio-ml-preflight demo davis --budget smoke
uv run bio-ml-preflight run examples/davis/case.yaml --budget standard
```

The case compares random-pair, cold-drug, and cold-target splits. Davis is public, so any local “holdout” is pseudo-sealed, not truly blinded.

## UCI Parkinsons Telemonitoring demo

The [UCI Parkinsons Telemonitoring dataset](https://archive.ics.uci.edu/dataset/189/parkinson)
contains 5,875 home voice recordings from 42 participants. The adapter pins the official
CC BY 4.0 archive by checksum. The case predicts the official, linearly interpolated
`motor_UPDRS` target from the 16 declared voice measurements only; age, sex, recording time,
and `total_UPDRS` remain audit context rather than predictors.

```bash
uv run bio-ml-preflight demo parkinsons --budget smoke
```

The smoke run makes the independence boundary visible. A random-record diagnostic reached
median Spearman `0.599` (permutation delta `0.237`, empirical p-value `0.10`), but all 42
participants appeared on both sides of those splits. It is therefore
`SUPPORTED_WITH_LIMITS`, not evidence for unseen participants or independent recording
occasions. The source has only 2,501 exact participant-time proxies, with 4,904 of 5,875
records belonging to repeated proxies. Across two participant-grouped smoke splits, participant
overlap was zero and median Spearman fell to `0.195`, with permutation delta `0.070`, p-value
`0.30`, and across-split standard deviation `0.158`; all three limitations are recorded and the
verdict is `INSUFFICIENT_EVIDENCE`. Measurement reliability and batch confounding remain
`NOT_ASSESSABLE`. This retrospective result does not establish future-time transport, clinical
utility, or a causal effect.

## BBB_Martins molecular-classification demo

This second TDC demo predicts a binary BBB label from SMILES and compares a random-compound diagnostic with scaffold-separated validation. Its case explicitly excludes the 12 compound identifiers with conflicting labels or inconsistent SMILES (24 rows) before splitting, and records that policy in every report.

```bash
uv run --python 3.11 --all-extras bio-ml-preflight demo bbb --budget smoke
# Equivalent one-command checkout target:
make demo-bbb
```

The smoke comparison reuses four checksummed manifests and the same logistic/extra-trees suite for character hashes and Morgan fingerprints. On the cached 2,006-row analysis set, median balanced accuracy was 0.733 versus 0.808 for random-compound and 0.709 versus 0.749 for scaffold validation. Both representations produced `SUPPORTED` verdicts in both scenarios, with zero compound overlap; this is retrospective capability evidence, not an external confirmation or a causal result.

The v0.1 ladder stops there. A learned graph representation belongs behind the same feature-frame/evaluation seam only when fixed character/Morgan baselines fail a decision-relevant boundary, enough independent compounds exist to fit it without leakage, and a locked external validation can justify the added model class.

## Graph-model readiness gate

The BBB case now declares a deterministic 2D molecular-graph contract without adding a GNN framework. It fixes canonical graph identity, node/edge features, invalid-structure handling, the independent unit, a case-specific class-support floor, and scaffold validation. The audit also checks canonical-graph—not only compound-ID—overlap on every selected manifest.

```bash
uv run bio-ml-preflight assess-graph-readiness reports/bbb-martins \
  --external-report reports/petbd-external-confirmation \
  --output reports/bbb-graph-readiness
```

This command reads existing JSON and Parquet artifacts; it does not refit a baseline or access the PETBD holdout again. For the current artifacts, all 2,006 development compounds convert to 1,955 canonical graphs, both scaffold manifests have zero canonical-graph overlap and at least 104 negative test compounds, and character/Morgan baselines keep the same `SUPPORTED` verdict. The adequately supported PETBD confirmation does not separate from permutation. The combined verdict is therefore `NOT_JUSTIFIED_BY_CURRENT_EVIDENCE`, not a claim that GNNs can never work: current evidence does not identify fixed molecular representation as the limiting boundary. Random-compound splits contain 23 and 18 overlapping canonical graphs and are excluded from graph-readiness evidence.

## Locked B3DB post-release confirmation

The confirmatory demo freezes the development-selected Morgan representation and logistic probe before evaluating B3DB's 175 post-release records. The adapter pins and checksums the official source, derives comparable RDKit InChIKeys, and removes 10 matching identities from BBB_Martins development while retaining the entire external set. A supplied manifest then contains 1,992 training rows and 175 holdout rows with zero compound overlap. The holdout ledger is keyed by the dataset checksum and records the case fingerprint for every access, so changing the report output or case parameters cannot bypass the one-access limit.

```bash
uv run --python 3.11 --all-extras bio-ml-preflight demo bbb-external
# Equivalent one-command checkout target:
make demo-bbb-external
```

The locked run produced balanced accuracy `0.904`, versus a nine-draw permutation median of `0.496` (delta `0.407`, empirical p-value `0.10`). That favorable point estimate is still `INSUFFICIENT_EVIDENCE`: the external set contains 171 positive but only 4 independent negative compounds, below the predeclared minimum of 20 per class. The cheapest next evidence is at least 16 additional independent negative holdout compounds under the same protocol. This is a public, release-time pseudo-sealed check—not a blinded or assay-time prospective study; measurement reliability and batch confounding remain `NOT_ASSESSABLE`. A post-run governance review strengthened cache verification, provenance, and holdout-ledger enforcement without rerunning the external labels; the original numerical artifacts remain the source of truth.

## Constrained Autoprobe

Autoprobe freezes the case, evaluator, metrics, manifests, holdout policy, and provenance. Only `candidate.yaml` changes. Unlike unrestricted autoresearch, it has bounded experiments, multiple scientific guardrails, a vector result instead of one universal score, and no final-holdout access.

```bash
uv run bio-ml-preflight autoprobe prepare CASE.yaml --run-dir runs/example
uv run bio-ml-preflight autoprobe evaluate runs/example/candidate.yaml
```

## Extension points

A dataset adapter normalizes a source into a DataFrame/Parquet table, records source/version/retrieval/checksum metadata, and returns columns referenced by the same `CaseSpec`; `data/davis.py` is the working example. A future AnnData or LINCS L1000 adapter would do that normalization without changing audits or split contracts. A multi-output gene-expression task would extend target validation and metric aggregation (per-gene, per-cell type, and decision-weighted summaries) while retaining the same independent-unit, split-manifest, holdout, provenance, and capability contracts.

DataSAIL is a future optional split provider: its assignment can be normalized into the existing persisted manifest and evaluated by the same overlap audit. It is intentionally not a v0.1 dependency. Physicochemical descriptors and GNN training are likewise omitted from the base ladder until the graph-readiness evidence identifies a distinct scientific question that character hashes and Morgan fingerprints cannot answer.

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
```

See [architecture](docs/architecture.md), [methodology](docs/methodology.md), [limitations](docs/limitations.md), [Autoprobe](docs/autoresearch.md), and [references](docs/references.md).
