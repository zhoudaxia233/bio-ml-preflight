from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from jinja2 import BaseLoader, Environment

from bio_ml_preflight.audits import audit_dataset
from bio_ml_preflight.contracts import CaseSpec
from bio_ml_preflight.data import read_table

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
    _validate_declared_columns(frame, case)
    development, excluded = _development_rows(frame, case)
    audits = audit_dataset(development, case)
    profile = _column_profile(development, case)
    findings = _findings(audits, case)
    source_record_path = Path(case.data.path).parent / "source.json"
    source_record = (
        json.loads(source_record_path.read_text(encoding="utf-8"))
        if source_record_path.is_file()
        else {}
    )
    structured = _json_safe(
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


def _validate_declared_columns(frame: pd.DataFrame, case: CaseSpec) -> None:
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


def _development_rows(frame: pd.DataFrame, case: CaseSpec) -> tuple[pd.DataFrame, int]:
    mask = pd.Series(True, index=frame.index)
    split_columns = {
        scenario.split_column
        for scenario in case.generalization_scenarios
        if scenario.strategy == "supplied" and scenario.split_column
    }
    for column in split_columns:
        mask &= frame[column].astype(str).str.lower().eq("train")
    development = frame.loc[mask].copy()
    if development.empty:
        raise ValueError("No development rows remain after excluding supplied test/holdout rows")
    return development.reset_index(drop=True), int((~mask).sum())


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


def _findings(audits: dict[str, Any], case: CaseSpec) -> list[dict[str, str]]:
    inventory = audits["inventory"]
    independence = audits["independence"]
    findings: list[dict[str, str]] = []

    def add(status: str, check: str, evidence: str, next_evidence: str) -> None:
        findings.append(
            {
                "status": status,
                "check": check,
                "evidence": evidence,
                "cheapest_next_evidence": next_evidence,
            }
        )

    unconfirmed = sorted(key for key, value in case.role_confirmation.items() if not value)
    if unconfirmed:
        add(
            "ACTION_REQUIRED",
            "case roles",
            f"Unconfirmed roles: {', '.join(unconfirmed)}.",
            "Have a researcher confirm the target, prediction unit, features, and entities.",
        )
    target_missing = int(inventory["missing"].get(case.task.target_column, 0))
    if target_missing:
        add(
            "ACTION_REQUIRED",
            "target completeness",
            f"The declared target has {target_missing} missing values.",
            "Declare and document a target inclusion rule before model fitting.",
        )
    if inventory["missing"]:
        add(
            "WARNING",
            "missing values",
            f"Columns with missing values: {inventory['missing']}.",
            "Confirm why values are missing; fit any imputation inside training folds only.",
        )
    if inventory["invalid_numeric_values"]:
        add(
            "ACTION_REQUIRED",
            "invalid numeric values",
            f"Non-finite numeric values: {inventory['invalid_numeric_values']}.",
            "Trace non-finite values to the source and declare a handling rule.",
        )
    if inventory["duplicate_rows"]:
        add(
            "WARNING",
            "duplicate rows",
            f"Found {inventory['duplicate_rows']} exact duplicate rows.",
            "Determine whether they are technical repeats, legitimate replicates, or "
            "ingestion errors.",
        )
    modeled_constants = sorted(set(inventory["constant_columns"]) & set(case.features.include))
    modeled_near_constants = sorted(
        set(inventory["near_constant_columns"]) & set(case.features.include)
    )
    if modeled_constants or modeled_near_constants:
        add(
            "WARNING",
            "uninformative modeled features",
            f"Constant: {modeled_constants}; near-constant: {modeled_near_constants}.",
            "Confirm coding and remove only after the feature role is reviewed.",
        )
    for name, result in independence["entities"].items():
        missing_identifiers = int(result.get("missing_identifier_rows", 0))
        if missing_identifiers:
            add(
                "ACTION_REQUIRED",
                f"entity identifier completeness {name}",
                f"{missing_identifiers} rows lack the declared {name} identifier.",
                "Recover the identifier or exclude those rows under a documented rule before "
                "claiming independent-unit support.",
            )
        if result.get("status") == "NOT_ASSESSABLE":
            add(
                "NOT_ASSESSABLE",
                f"entity {name}",
                str(result.get("reason")),
                "Supply the declared entity identifier.",
            )
            continue
        repeats = int(result.get("duplicate_identifier_rows", 0))
        if repeats:
            add(
                "WARNING",
                f"repeated entity {name}",
                f"{repeats} rows belong to identifiers that occur more than once; "
                f"rows/entity={result.get('rows_per_entity')}.",
                "Use the declared entity—not rows—as the split and uncertainty unit.",
            )
        conflicts = int(result.get("conflicting_target_entities", 0))
        representations = int(result.get("inconsistent_representation_entities", 0))
        if conflicts or representations:
            policy = case.entities[name].identity_conflict_policy
            add(
                "ACTION_REQUIRED" if policy is None else "WARNING",
                f"identity consistency {name}",
                f"Conflicting targets: {conflicts}; inconsistent representations: "
                f"{representations}; declared policy: {policy}.",
                "Review affected identities and explicitly keep, exclude, or aggregate them.",
            )
    suspicious = audits["leakage"]["suspicious_identifier_features"]
    if suspicious:
        add(
            "WARNING",
            "identifier leakage",
            f"High-cardinality identifier-like modeled features: {suspicious}.",
            "Confirm these values are available at prediction time and cannot encode outcomes.",
        )
    post_outcome = audits["leakage"]["declared_post_outcome_features"]
    if post_outcome:
        modeled_post_outcome = sorted(set(post_outcome) & set(case.features.include))
        add(
            "ACTION_REQUIRED" if modeled_post_outcome else "INFO",
            "declared post-outcome fields",
            f"Declared post-outcome columns: {post_outcome}; also modeled: {modeled_post_outcome}.",
            "Keep these columns out of predictors and retain them only as declared audit context.",
        )
    if audits["measurement"].get("status") == "NOT_ASSESSABLE":
        add(
            "NOT_ASSESSABLE",
            "measurement reliability",
            str(audits["measurement"].get("reason")),
            "Add replicate identifiers or repeated measurements under a declared protocol.",
        )
    for boundary in audits["missing_high_value_metadata"]:
        add(
            "NOT_ASSESSABLE",
            "missing high-value metadata",
            boundary,
            "Add or document the missing metadata; do not infer it from column names.",
        )
    add(
        "INFO",
        "declared target support",
        f"Observed development-data target distribution: {inventory['target_distribution']}.",
        "Judge adequacy against the declared decision and independent unit, not row count alone.",
    )
    return findings


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


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
