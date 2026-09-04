from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from jinja2 import BaseLoader, Environment

from bio_ml_preflight.audits import audit_dataset
from bio_ml_preflight.contracts import CaseSpec
from bio_ml_preflight.contracts.case import unconfirmed_roles
from bio_ml_preflight.data import read_table
from bio_ml_preflight.features import model_feature_columns
from bio_ml_preflight.reporting.render import json_safe

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Evidence-oriented EDA: {{ case.case_id }}</title>
<style>
:root{font-family:Inter,ui-sans-serif,system-ui,sans-serif;color:#17202a;background:#f5f7fa}
body{max-width:1100px;margin:0 auto;padding:2rem 1.25rem 4rem;line-height:1.5}
header,section{background:#fff;border:1px solid #dfe5ec;border-radius:10px;
padding:1.25rem 1.5rem;margin-bottom:1rem}h1{margin:.1rem 0 .5rem}
h2{margin:.1rem 0 .8rem;font-size:1.25rem}.lede{color:#52606d}
table{border-collapse:collapse;width:100%;font-size:.9rem}td,th{border:1px solid #d8dee6;
padding:.5rem;text-align:left;vertical-align:top}th{background:#eef2f6}
.ACTION_REQUIRED{color:#b42318;font-weight:700}.WARNING{color:#9c5d00;font-weight:700}
.NOT_ASSESSABLE{color:#6b7280;font-weight:700}.INFO{color:#087f5b;font-weight:700}
.BLOCKED{color:#b42318}.READY_WITH_LIMITS{color:#9c5d00}.READY{color:#087f5b}
img{display:block;max-width:100%;height:auto;margin:1rem auto}code{font-family:ui-monospace,
SFMono-Regular,Menlo,monospace}small{color:#52606d}
</style>
</head>
<body>
<header><h1>Evidence-oriented EDA: {{ case.case_id }}</h1>
<p class="lede">Deterministic development-data inspection. No model was trained and no
biological meaning was inferred from column names.</p></header>
<section><h2>Scope</h2><ul>
<li>Rows loaded: {{ scope.rows_loaded }}; rows profiled: {{ scope.rows_profiled }}.</li>
<li>Non-training supplied-split rows excluded: {{ scope.non_training_rows_excluded }}.</li>
<li>Locked holdout labels accessed: <strong>no</strong>.</li>
<li>Pre-model readiness: <strong class="{{ readiness.status }}">{{ readiness.status }}</strong>.
{{ readiness.interpretation }}</li>
</ul></section>
<section><h2>Findings and cheapest next evidence</h2>
<table><thead><tr><th>Status</th><th>Check</th><th>Evidence</th><th>Next evidence</th></tr>
</thead><tbody>{% for row in findings %}<tr>
<td class="{{ row.status }}">{{ row.status }}</td><td>{{ row.check }}</td>
<td>{{ row.evidence }}</td><td>{{ row.cheapest_next_evidence }}</td></tr>{% endfor %}
</tbody></table></section>
<section><h2>Missingness</h2><img src="figures/missingness.png" alt="Missing values by column">
</section><section><h2>Declared target</h2>
<img src="figures/target_distribution.png" alt="Declared target distribution"></section>
<section><h2>Declared independent entities</h2>
<img src="figures/entity_repetition.png" alt="Rows per declared entity"></section>
<section><h2>Column profile preview</h2><p><small>The complete typed table is in
<code>column_profile.parquet</code>.</small></p><table><thead><tr>
<th>Column</th><th>Role</th><th>Type</th><th>Missing</th><th>Unique</th>
<th>Median</th><th>Top count</th></tr></thead><tbody>{% for row in columns %}<tr>
<td>{{ row.column }}</td><td>{{ row.roles }}</td><td>{{ row.physical_type }}</td>
<td>{{ row.missing }} ({{ row.missing_fraction }})</td><td>{{ row.unique }}</td>
<td>{{ row.median }}</td><td>{{ row.top_count }}</td></tr>{% endfor %}</tbody></table></section>
<section><h2>Interpretation boundary</h2><ul>
<li>Descriptive associations do not establish a causal effect.</li>
<li>Missing replicate, batch, time, or deployment metadata is reported, not guessed.</li>
<li>No automatic outlier deletion, embedding, or universal data-quality score is produced.</li>
<li>Use a declared split and bounded baselines to assess predictive capability.</li>
</ul></section>
</body></html>"""


def run_eda(case: CaseSpec, output: Path) -> dict[str, Any]:
    """Profile development data without model fitting or locked-holdout access."""
    if case.holdout.enabled:
        raise ValueError(
            "EDA refuses holdout-enabled cases; create a development-only case without "
            "locked holdout rows"
        )
    frame = read_table(Path(case.data.path)).reset_index(drop=True)
    validate_declared_columns(frame, case)
    development, excluded = development_rows(frame, case)
    audits = audit_dataset(development, case)
    profile = _column_profile(development, case)
    findings = readiness_findings(audits, case)
    readiness = assess_readiness(findings)
    source_record_path = Path(case.data.path).parent / "source.json"
    source_record = (
        json.loads(source_record_path.read_text(encoding="utf-8"))
        if source_record_path.is_file()
        else {}
    )
    structured = json_safe(
        {
            "schema_version": 1,
            "mode": "evidence_oriented_eda",
            "case": case.model_dump(mode="json"),
            "scope": {
                "rows_loaded": len(frame),
                "rows_profiled": len(development),
                "non_training_rows_excluded": excluded,
                "model_trained": False,
                "holdout_labels_accessed": False,
            },
            "dataset_source": source_record,
            "audits": audits,
            "findings": findings,
            "readiness": readiness,
            "column_profile_artifact": "column_profile.parquet",
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    profile.to_parquet(output / "column_profile.parquet", index=False)
    (output / "eda.json").write_text(
        json.dumps(structured, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    _missingness_figure(profile, figures / "missingness.png")
    _target_figure(development, case, figures / "target_distribution.png")
    _entity_figure(development, case, figures / "entity_repetition.png")
    _write_reports(output, case, structured, profile)
    return cast(dict[str, Any], structured)


def validate_declared_columns(frame: pd.DataFrame, case: CaseSpec) -> None:
    required = {case.task.target_column, *case.features.include, *case.features.post_outcome}
    required.update(case.data.fingerprint_columns)
    for entity in case.entities.values():
        required.add(entity.id_column)
        if entity.representation_column:
            required.add(entity.representation_column)
    metadata = case.metadata
    required.update(
        column
        for column in [
            metadata.replicate_id,
            metadata.biological_replicate_id,
            metadata.batch_id,
            metadata.plate_id,
            metadata.time_column,
            metadata.treatment_column,
            case.evaluation.bootstrap_unit,
        ]
        if column
    )
    for scenario in case.generalization_scenarios:
        required.update(
            column
            for column in [
                scenario.group_column,
                scenario.left_column,
                scenario.right_column,
                scenario.split_column,
            ]
            if column
        )
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Case references missing declared columns: {missing}")


def development_row_indices(
    frame: pd.DataFrame,
    case: CaseSpec,
    *,
    require_consistent: bool = False,
) -> pd.Index:
    if any(scenario.strategy != "supplied" for scenario in case.generalization_scenarios):
        return frame.index

    split_columns = list(
        dict.fromkeys(
            scenario.split_column
            for scenario in case.generalization_scenarios
            if scenario.split_column
        )
    )
    missing = sorted(set(split_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Case references missing supplied split columns: {missing}")
    train_masks = [frame[column].astype(str).str.lower().eq("train") for column in split_columns]
    if (case.holdout.enabled or require_consistent) and any(
        not mask.equals(train_masks[0]) for mask in train_masks[1:]
    ):
        raise ValueError("Supplied splits must use one consistent training boundary")
    mask = pd.Series(False, index=frame.index)
    for train_mask in train_masks:
        mask |= train_mask
    indices = frame.index[mask]
    if indices.empty:
        raise ValueError("No development rows remain after excluding supplied test/holdout rows")
    return indices


def development_rows(frame: pd.DataFrame, case: CaseSpec) -> tuple[pd.DataFrame, int]:
    indices = development_row_indices(frame, case)
    development = frame.loc[indices].copy()
    return development.reset_index(drop=True), len(frame) - len(development)


def _column_profile(frame: pd.DataFrame, case: CaseSpec) -> pd.DataFrame:
    roles = _column_roles(case)
    rows: list[dict[str, Any]] = []
    for column in frame.columns:
        series = frame[column]
        counts = series.value_counts(dropna=True)
        numeric = pd.api.types.is_numeric_dtype(series)
        finite = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        rows.append(
            {
                "column": str(column),
                "roles": ", ".join(roles.get(str(column), ["context/unassigned"])),
                "physical_type": str(series.dtype),
                "rows": len(series),
                "non_missing": int(series.notna().sum()),
                "missing": int(series.isna().sum()),
                "missing_fraction": float(series.isna().mean()),
                "unique": int(series.nunique(dropna=True)),
                "constant": bool(series.nunique(dropna=False) <= 1),
                "near_constant": bool(
                    len(series) > 0
                    and not counts.empty
                    and float(counts.iloc[0] / len(series)) >= 0.98
                ),
                "minimum": float(finite.min()) if numeric and finite.notna().any() else None,
                "q05": float(finite.quantile(0.05)) if numeric and finite.notna().any() else None,
                "median": float(finite.median()) if numeric and finite.notna().any() else None,
                "q95": float(finite.quantile(0.95)) if numeric and finite.notna().any() else None,
                "maximum": float(finite.max()) if numeric and finite.notna().any() else None,
                "mean": float(finite.mean()) if numeric and finite.notna().any() else None,
                "standard_deviation": float(finite.std())
                if numeric and finite.notna().sum() > 1
                else None,
                "top_value": str(counts.index[0]) if not counts.empty else None,
                "top_count": int(counts.iloc[0]) if not counts.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def _column_roles(case: CaseSpec) -> dict[str, list[str]]:
    roles: dict[str, list[str]] = {}

    def add(column: str | None, role: str) -> None:
        if column:
            roles.setdefault(column, []).append(role)

    add(case.task.target_column, "target")
    for column in case.features.include:
        add(column, "feature")
    for column in case.features.post_outcome:
        add(column, "post_outcome")
    for name, entity in case.entities.items():
        add(entity.id_column, f"entity:{name}")
        add(entity.representation_column, f"representation:{name}")
    for name in [
        "replicate_id",
        "biological_replicate_id",
        "batch_id",
        "plate_id",
        "time_column",
        "treatment_column",
    ]:
        add(getattr(case.metadata, name), f"metadata:{name}")
    for scenario in case.generalization_scenarios:
        add(scenario.split_column, f"split:{scenario.name}")
    add(case.evaluation.bootstrap_unit, "bootstrap_unit")
    return roles


def declaration_findings(case: CaseSpec) -> list[dict[str, str]]:
    """Return readiness findings that can be decided without reading the dataset."""
    findings: list[dict[str, str]] = []

    def add(status: str, code: str, check: str, evidence: str, next_evidence: str) -> None:
        findings.append(
            {
                "status": status,
                "code": code,
                "check": check,
                "evidence": evidence,
                "cheapest_next_evidence": next_evidence,
            }
        )

    unconfirmed = unconfirmed_roles(case)
    if unconfirmed:
        add(
            "ACTION_REQUIRED",
            "unconfirmed_roles",
            "case roles",
            f"Unconfirmed roles: {', '.join(unconfirmed)}.",
            "Have a researcher confirm the target, prediction unit, features, and entities.",
        )

    configured_features = set(model_feature_columns(case.features.include, case))
    if case.features.include and not configured_features:
        add(
            "ACTION_REQUIRED",
            "no_modeled_features",
            "modeled feature availability",
            "No modeled feature column remains after applying declared exclusions.",
            "Declare at least one predictor that is available at prediction time.",
        )
    if case.task.target_column in configured_features:
        add(
            "ACTION_REQUIRED",
            "target_modeled_as_feature",
            "target leakage",
            f"The declared target {case.task.target_column!r} is also a modeled feature.",
            "Remove the target from features.include before model fitting.",
        )

    modeled_post_outcome = sorted(set(case.features.post_outcome) & configured_features)
    if case.features.post_outcome:
        add(
            "ACTION_REQUIRED" if modeled_post_outcome else "INFO",
            (
                "post_outcome_modeled_as_feature"
                if modeled_post_outcome
                else "declared_post_outcome_context"
            ),
            "declared post-outcome fields",
            f"Declared post-outcome columns: {case.features.post_outcome}; also modeled: "
            f"{modeled_post_outcome}.",
            "Keep these columns out of predictors and retain them only as declared audit context.",
        )
    return findings


def readiness_findings(
    audits: dict[str, Any],
    case: CaseSpec,
    *,
    allow_pending_identity_policies: bool = True,
) -> list[dict[str, str]]:
    inventory = audits["inventory"]
    independence = audits["independence"]
    findings = declaration_findings(case)
    modeled_features = set(audits["leakage"]["modeled_features"])

    def add(status: str, code: str, check: str, evidence: str, next_evidence: str) -> None:
        findings.append(
            {
                "status": status,
                "code": code,
                "check": check,
                "evidence": evidence,
                "cheapest_next_evidence": next_evidence,
            }
        )

    missing_modeled = audits["leakage"]["missing_modeled_features"]
    if missing_modeled:
        add(
            "ACTION_REQUIRED",
            "modeled_features_missing",
            "modeled feature availability",
            f"Configured modeled feature columns are missing: {missing_modeled}.",
            "Correct the case declaration or restore the columns before model fitting.",
        )
    if (
        not modeled_features
        and not missing_modeled
        and not any(row["code"] == "no_modeled_features" for row in findings)
    ):
        add(
            "ACTION_REQUIRED",
            "no_modeled_features",
            "modeled feature availability",
            "No modeled feature column remains after applying declared exclusions.",
            "Declare at least one predictor that is available at prediction time.",
        )

    all_missing_modeled = sorted(
        column
        for column in modeled_features
        if int(inventory["missing"].get(column, 0)) == int(inventory["rows"])
    )
    if modeled_features and set(all_missing_modeled) == modeled_features:
        add(
            "ACTION_REQUIRED",
            "no_observed_modeled_features",
            "modeled feature availability",
            f"Every modeled feature is entirely missing: {all_missing_modeled}.",
            "Restore at least one observed predictor before model fitting.",
        )

    findings.extend(target_readiness_findings(audits, case))
    if inventory["missing"]:
        add(
            "WARNING",
            "missing_values",
            "missing values",
            f"Columns with missing values: {inventory['missing']}.",
            "Confirm why values are missing; fit any imputation inside training folds only.",
        )

    invalid_numeric = inventory["invalid_numeric_values"]
    modeled_invalid = {
        column: invalid_numeric[column]
        for column in sorted(set(invalid_numeric) & modeled_features - {case.task.target_column})
    }
    context_invalid = {
        column: invalid_numeric[column]
        for column in sorted(set(invalid_numeric) - modeled_features - {case.task.target_column})
    }
    if modeled_invalid:
        add(
            "ACTION_REQUIRED",
            "non_finite_modeled_features",
            "non-finite modeled features",
            f"Modeled feature columns with infinite values: {modeled_invalid}.",
            "Trace infinite values to the source and declare a fold-safe handling rule.",
        )
    if context_invalid:
        add(
            "WARNING",
            "non_finite_context",
            "non-finite audit context",
            f"Non-modeled numeric columns with infinite values: {context_invalid}.",
            "Review source encoding; these columns remain excluded from model fitting.",
        )
    if inventory["duplicate_rows"]:
        add(
            "WARNING",
            "duplicate_rows",
            "duplicate rows",
            f"Found {inventory['duplicate_rows']} exact duplicate rows.",
            "Determine whether they are technical repeats, legitimate replicates, or "
            "ingestion errors.",
        )
    modeled_constants = sorted(set(inventory["constant_columns"]) & modeled_features)
    modeled_near_constants = sorted(set(inventory["near_constant_columns"]) & modeled_features)
    if modeled_constants or modeled_near_constants:
        add(
            "WARNING",
            "uninformative_modeled_features",
            "uninformative modeled features",
            f"Constant: {modeled_constants}; near-constant: {modeled_near_constants}.",
            "Confirm coding and remove only after the feature role is reviewed.",
        )
    for name, result in independence["entities"].items():
        missing_identifiers = int(result.get("missing_identifier_rows", 0))
        if missing_identifiers:
            add(
                "ACTION_REQUIRED",
                f"entity_identifier_missing:{name}",
                f"entity identifier completeness {name}",
                f"{missing_identifiers} rows lack the declared {name} identifier.",
                "Recover the identifier or exclude those rows under a documented rule before "
                "claiming independent-unit support.",
            )
        if result.get("status") == "NOT_ASSESSABLE":
            add(
                "NOT_ASSESSABLE",
                f"entity_not_assessable:{name}",
                f"entity {name}",
                str(result.get("reason")),
                "Supply the declared entity identifier.",
            )
            continue
        repeats = int(result.get("duplicate_identifier_rows", 0))
        if repeats:
            add(
                "WARNING",
                f"repeated_entity:{name}",
                f"repeated entity {name}",
                f"{repeats} rows belong to identifiers that occur more than once; "
                f"rows/entity={result.get('rows_per_entity')}.",
                "Use the declared entity—not rows—as the split and uncertainty unit.",
            )
        conflicts = int(result.get("conflicting_target_entities", 0))
        representations = int(result.get("inconsistent_representation_entities", 0))
        if conflicts or representations:
            policy = case.entities[name].identity_conflict_policy
            acknowledged = policy == "keep" or (
                allow_pending_identity_policies and policy in {"exclude", "aggregate"}
            )
            add(
                "WARNING" if acknowledged else "ACTION_REQUIRED",
                f"identity_conflict:{name}",
                f"identity consistency {name}",
                f"Conflicting targets: {conflicts}; inconsistent representations: "
                f"{representations}; declared policy: {policy}.",
                "Review affected identities and explicitly keep, exclude, or aggregate them.",
            )
    pair = independence["pair_structure"]
    pair_conflicts = int(pair.get("conflicting_label_pairs", 0))
    if pair.get("defines_prediction_unit") and pair_conflicts:
        add(
            "ACTION_REQUIRED",
            "pair_target_conflict",
            "pair identity consistency",
            f"{pair_conflicts} declared entity pairs map to multiple target values.",
            "Resolve replicate semantics or aggregate the pair target under a documented rule.",
        )
    suspicious = audits["leakage"]["suspicious_identifier_features"]
    if suspicious:
        add(
            "WARNING",
            "identifier_leakage",
            "identifier leakage",
            f"High-cardinality identifier-like modeled features: {suspicious}.",
            "Confirm these values are available at prediction time and cannot encode outcomes.",
        )
    if audits["measurement"].get("status") == "NOT_ASSESSABLE":
        next_evidence = str(
            audits["measurement"].get(
                "cheapest_next_evidence",
                "Add replicate identifiers or repeated measurements under a declared protocol.",
            )
        )
        add(
            "NOT_ASSESSABLE",
            "measurement_reliability_not_assessable",
            "measurement reliability",
            str(audits["measurement"].get("reason")),
            next_evidence,
        )
    for boundary in audits["missing_high_value_metadata"]:
        add(
            "NOT_ASSESSABLE",
            "missing_high_value_metadata",
            "missing high-value metadata",
            boundary,
            "Add or document the missing metadata; do not infer it from column names.",
        )
    add(
        "INFO",
        "declared_target_support",
        "declared target support",
        f"Observed development-data target distribution: {inventory['target_distribution']}.",
        "Judge adequacy against the declared decision and independent unit, not row count alone.",
    )
    return findings


def target_readiness_findings(audits: dict[str, Any], case: CaseSpec) -> list[dict[str, str]]:
    """Return target-contract findings reusable for training and evaluation partitions."""
    inventory = audits["inventory"]
    distribution = inventory["target_distribution"]
    findings: list[dict[str, str]] = []

    def add(code: str, check: str, evidence: str, next_evidence: str) -> None:
        findings.append(
            {
                "status": "ACTION_REQUIRED",
                "code": code,
                "check": check,
                "evidence": evidence,
                "cheapest_next_evidence": next_evidence,
            }
        )

    target_missing = int(inventory["missing"].get(case.task.target_column, 0))
    if target_missing:
        add(
            "target_missing",
            "target completeness",
            f"The audited target has {target_missing} missing values.",
            "Declare and document a target inclusion rule before model fitting.",
        )
    target_invalid = int(inventory["invalid_numeric_values"].get(case.task.target_column, 0))
    if target_invalid:
        add(
            "non_finite_target",
            "non-finite target values",
            f"The audited target contains {target_invalid} infinite values.",
            "Trace infinite target values to the source and declare an inclusion rule.",
        )
    if case.task.kind == "binary_classification":
        class_counts = distribution["class_counts"]
        if len(class_counts) != 2:
            add(
                "invalid_binary_target_support",
                "binary target support",
                f"The audited target has {len(class_counts)} observed classes: {class_counts}.",
                "Provide exactly two declared target classes before fitting a binary model.",
            )
        if not distribution["zero_one_encoded"]:
            add(
                "invalid_binary_target_encoding",
                "binary target encoding",
                f"The audited target is not encoded as numeric 0/1: {class_counts}.",
                "Map the two declared classes to numeric 0 and 1 before model fitting.",
            )
    elif not distribution["numeric"]:
        target = case.task.target_column
        dtype = inventory["physical_types"][target]
        add(
            "non_numeric_target",
            "numeric target encoding",
            f"The audited {case.task.kind} target {target!r} has non-numeric dtype {dtype!r}.",
            "Encode the declared regression or ranking target as numeric values.",
        )
    elif int(distribution["finite_unique"]) < 2:
        add(
            "invalid_continuous_target_support",
            "continuous target support",
            "The audited regression or ranking target has fewer than two finite values.",
            "Provide a target with at least two distinct finite observed values.",
        )
    return findings


def assess_readiness(findings: list[dict[str, str]]) -> dict[str, Any]:
    """Aggregate auditable findings into a non-scoring model-fit decision."""
    allowed_statuses = {"ACTION_REQUIRED", "WARNING", "NOT_ASSESSABLE", "INFO"}
    unknown = sorted({row["status"] for row in findings} - allowed_statuses)
    if unknown:
        raise ValueError(f"Unknown readiness finding statuses: {unknown}")

    blocking = list(
        dict.fromkeys(row["code"] for row in findings if row["status"] == "ACTION_REQUIRED")
    )
    limiting = list(
        dict.fromkeys(
            row["code"] for row in findings if row["status"] in {"WARNING", "NOT_ASSESSABLE"}
        )
    )
    if blocking:
        status = "BLOCKED"
        interpretation = (
            "Resolve every ACTION_REQUIRED finding before feature construction or model fitting."
        )
    elif limiting:
        status = "READY_WITH_LIMITS"
        interpretation = (
            "Model fitting may proceed, but warnings and unassessable boundaries must remain "
            "explicit."
        )
    else:
        status = "READY"
        interpretation = (
            "No action-required, warning, or unassessable finding was detected in the declared "
            "development data."
        )
    return {
        "status": status,
        "model_fitting_allowed": not blocking,
        "blocking_checks": blocking,
        "limiting_checks": limiting,
        "interpretation": interpretation,
    }


def _write_reports(
    output: Path,
    case: CaseSpec,
    structured: dict[str, Any],
    profile: pd.DataFrame,
) -> None:
    findings = structured["findings"]
    lines = [
        f"# Evidence-oriented EDA: {case.case_id}",
        "",
        "Deterministic development-data inspection. No model was trained and no biological "
        "meaning was inferred from column names.",
        "",
        "## Scope",
        "",
        f"- Rows loaded: `{structured['scope']['rows_loaded']}`.",
        f"- Rows profiled: `{structured['scope']['rows_profiled']}`.",
        f"- Non-training supplied-split rows excluded: "
        f"`{structured['scope']['non_training_rows_excluded']}`.",
        "- Locked holdout labels accessed: `false`.",
        f"- Pre-model readiness: `{structured['readiness']['status']}`.",
        "- Model fitting allowed: "
        f"`{str(structured['readiness']['model_fitting_allowed']).lower()}`.",
        f"- Interpretation: {structured['readiness']['interpretation']}",
        "",
        "## Findings and cheapest next evidence",
        "",
        "| Status | Check | Evidence | Cheapest next evidence |",
        "|---|---|---|---|",
    ]
    lines.extend(
        "| "
        + " | ".join(
            _markdown(row[key])
            for key in [
                "status",
                "check",
                "evidence",
                "cheapest_next_evidence",
            ]
        )
        + " |"
        for row in findings
    )
    lines.extend(
        [
            "",
            "## Key figures",
            "",
            "![Missing values by column](figures/missingness.png)",
            "",
            "![Declared target distribution](figures/target_distribution.png)",
            "",
            "![Rows per declared entity](figures/entity_repetition.png)",
            "",
            "## Structured artifacts",
            "",
            "- `eda.json` contains the audit facts, scope, and findings.",
            "- `column_profile.parquet` contains typed per-column descriptive statistics.",
            "",
            "## Interpretation boundary",
            "",
            "- Descriptive associations do not establish a causal effect.",
            "- Missing biological metadata is reported, not guessed.",
            "- No automatic outlier deletion, embedding, or universal quality score is produced.",
            "- Use a declared split and bounded baselines to assess predictive capability.",
            "",
        ]
    )
    (output / "eda.md").write_text("\n".join(lines), encoding="utf-8")
    html = (
        Environment(loader=BaseLoader(), autoescape=True)
        .from_string(HTML_TEMPLATE)
        .render(
            case=case,
            scope=structured["scope"],
            readiness=structured["readiness"],
            findings=findings,
            columns=profile.head(50).to_dict("records"),
        )
    )
    (output / "eda.html").write_text(html, encoding="utf-8")


def _missingness_figure(profile: pd.DataFrame, path: Path) -> None:
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(8, 4.5))
    missing = profile.loc[profile["missing"].gt(0)].nlargest(30, "missing_fraction")
    if missing.empty:
        axis.text(0.5, 0.5, "No missing values", ha="center", va="center")
        axis.set_axis_off()
    else:
        axis.barh(missing["column"], missing["missing_fraction"], color="#2a6f97")
        axis.set_xlabel("missing fraction")
        axis.set_xlim(0, 1)
        axis.invert_yaxis()
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def _target_figure(frame: pd.DataFrame, case: CaseSpec, path: Path) -> None:
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(7, 4))
    target = frame[case.task.target_column].dropna()
    if case.task.kind == "binary_classification" or not pd.api.types.is_numeric_dtype(target):
        counts = target.astype(str).value_counts().sort_index()
        axis.bar(counts.index, counts.to_numpy(), color="#087f5b")
        axis.set_ylabel("rows")
    else:
        finite = pd.to_numeric(target, errors="coerce")
        finite = finite[np.isfinite(finite)]
        if len(finite):
            axis.hist(finite, bins=min(30, max(5, int(np.sqrt(len(finite))))), color="#087f5b")
            axis.set_ylabel("rows")
        else:
            axis.text(0.5, 0.5, "No finite target values", ha="center", va="center")
            axis.set_axis_off()
    axis.set_xlabel(case.task.target_column)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def _entity_figure(frame: pd.DataFrame, case: CaseSpec, path: Path) -> None:
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(8, 4.5))
    labels, medians, maxima = [], [], []
    for name, entity in case.entities.items():
        counts = frame[entity.id_column].value_counts()
        labels.append(name)
        medians.append(float(counts.median()) if not counts.empty else 0.0)
        maxima.append(float(counts.max()) if not counts.empty else 0.0)
    if not labels:
        axis.text(0.5, 0.5, "No entity identifier declared", ha="center", va="center")
        axis.set_axis_off()
    else:
        positions = np.arange(len(labels))
        axis.bar(positions - 0.18, medians, 0.36, label="median", color="#2a6f97")
        axis.bar(positions + 0.18, maxima, 0.36, label="maximum", color="#d97706")
        axis.set_xticks(positions, labels)
        axis.set_ylabel("rows per identifier")
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    return plt


def _markdown(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
