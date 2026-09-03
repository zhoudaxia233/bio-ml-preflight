from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from bio_ml_preflight.audits import (
    apply_identity_conflict_policies,
    audit_dataset,
    audit_graph_readiness_contract,
    audit_overlap,
)
from bio_ml_preflight.contracts import CaseSpec
from bio_ml_preflight.contracts.case import ScenarioSpec
from bio_ml_preflight.data import read_table
from bio_ml_preflight.data.io import file_checksum
from bio_ml_preflight.eda import (
    assess_readiness,
    declaration_findings,
    development_row_indices,
    development_rows,
    readiness_findings,
    validate_declared_columns,
)
from bio_ml_preflight.evaluation.capability import capability_matrix
from bio_ml_preflight.evaluation.metrics import (
    compute_metrics,
    group_respecting_permutation,
    per_group_ranking_metrics,
)
from bio_ml_preflight.features import build_feature_frames
from bio_ml_preflight.models.probes import build_probe_suite, entity_mean_predictions
from bio_ml_preflight.provenance import HoldoutLedger, build_provenance
from bio_ml_preflight.reporting import write_report
from bio_ml_preflight.splits import create_split
from bio_ml_preflight.stability.ranking import ranking_stability, stability_decomposition


def run_case(
    case: CaseSpec,
    output: Path,
    *,
    budget: str = "smoke",
    model_allowlist: set[str] | None = None,
    holdout_override_reason: str | None = None,
) -> dict[str, Any]:
    pre_model_findings = declaration_findings(case)
    pre_model_readiness = assess_readiness(pre_model_findings)
    _require_model_readiness(pre_model_readiness)

    data_path = Path(case.data.path)
    holdout_access: list[dict[str, Any]] = []
    if case.holdout.enabled:
        if any(scenario.strategy != "supplied" for scenario in case.generalization_scenarios):
            raise ValueError("Enabled holdouts require supplied split assignments")
        dataset_checksum = file_checksum(data_path)
        ledger = HoldoutLedger(
            data_path.parent / "holdout_access" / f"{dataset_checksum}.jsonl",
            maximum_accesses=case.holdout.maximum_accesses,
        )
        ledger.record_access(
            actor="bio-ml-preflight",
            purpose=f"locked confirmation run for {case.case_id}",
            override_reason=holdout_override_reason,
            case_fingerprint=case.fingerprint(),
        )
        holdout_access = ledger.entries()
    started = time.perf_counter()
    frame = read_table(data_path).reset_index(drop=True)
    validate_declared_columns(frame, case)
    mutating_identity_policy = any(
        entity.identity_conflict_policy in {"exclude", "aggregate"}
        for entity in case.entities.values()
    )
    policy_indices = development_row_indices(
        frame,
        case,
        require_consistent=mutating_identity_policy,
    )
    frame, identity_consistency = apply_identity_conflict_policies(
        frame, case, policy_indices=policy_indices
    )
    development, _ = development_rows(frame, case)
    pre_model_findings = readiness_findings(
        audit_dataset(development, case),
        case,
        allow_pending_identity_policies=False,
    )
    pre_model_readiness = assess_readiness(pre_model_findings)
    _require_model_readiness(pre_model_readiness)
    seeds = case.evaluation.seeds[:2] if budget == "smoke" else case.evaluation.seeds
    manifests = {}
    for scenario in case.generalization_scenarios:
        for seed in seeds:
            manifest = create_split(frame, scenario, seed)
            manifests[(scenario.name, seed)] = manifest
            partition = frame.iloc[manifest.train_indices].reset_index(drop=True)
            scope = f"training_partition:{scenario.name}:seed_{seed}"
            for finding in readiness_findings(
                audit_dataset(partition, case),
                case,
                allow_pending_identity_policies=False,
            ):
                if finding["status"] == "INFO":
                    continue
                scoped = finding.copy()
                scoped["code"] = f"{scope}:{finding['code']}"
                scoped["check"] = f"{finding['check']} ({scenario.name}, seed {seed})"
                scoped["evidence"] = (
                    f"Training partition {scenario.name}, seed {seed}: " + finding["evidence"]
                )
                pre_model_findings.append(scoped)
    pre_model_readiness = assess_readiness(pre_model_findings)
    _require_model_readiness(pre_model_readiness)

    overlap_results: dict[str, Any] = {}
    for scenario in case.generalization_scenarios:
        for seed in seeds:
            manifest = manifests[(scenario.name, seed)]
            train_indices = np.asarray(manifest.train_indices)
            test_indices = np.asarray(manifest.test_indices)
            overlap = audit_overlap(frame, train_indices, test_indices, case)
            if case.task.kind == "binary_classification":
                counts, unit = _test_target_counts(frame, test_indices, case)
                overlap["test_target_counts"] = counts
                overlap["test_target_count_unit"] = unit
            _validate_protected_entity_isolation(case, scenario, overlap)
            evaluation = frame.iloc[test_indices].reset_index(drop=True)
            scope = f"evaluation_partition:{scenario.name}:seed_{seed}"
            for finding in readiness_findings(
                audit_dataset(evaluation, case),
                case,
                allow_pending_identity_policies=False,
            ):
                if finding["status"] == "INFO":
                    continue
                scoped = finding.copy()
                scoped["code"] = f"{scope}:{finding['code']}"
                scoped["check"] = f"{finding['check']} ({scenario.name}, seed {seed})"
                scoped["evidence"] = (
                    f"Evaluation partition {scenario.name}, seed {seed}: " + finding["evidence"]
                )
                pre_model_findings.append(scoped)
            overlap_results[f"{scenario.name}:{seed}"] = overlap
    pre_model_readiness = assess_readiness(pre_model_findings)
    _require_model_readiness(pre_model_readiness)

    modeling_indices = np.asarray(
        sorted(
            {
                index
                for manifest in manifests.values()
                for index in (*manifest.train_indices, *manifest.test_indices)
            }
        ),
        dtype=np.int64,
    )
    modeling_frame = frame.iloc[modeling_indices]
    audits = audit_dataset(modeling_frame, case)
    audits["inventory"].update(
        {
            "scope": "union of manifest train and test partitions",
            "source_rows_after_identity_policy": int(len(frame)),
            "manifest_excluded_rows": int(len(frame) - len(modeling_indices)),
        }
    )
    audits["identity_consistency"] = identity_consistency
    source_record_path = data_path.parent / "source.json"
    dataset_source = (
        json.loads(source_record_path.read_text(encoding="utf-8"))
        if source_record_path.is_file()
        else {}
    )
    if case.graph_readiness is not None:
        audits["graph_readiness_contract"] = audit_graph_readiness_contract(
            modeling_frame, case, identity_consistency
        )
    feature_frames = build_feature_frames(modeling_frame, case)
    for features in feature_frames.values():
        features.index = modeling_indices
    target = frame[case.task.target_column].to_numpy()
    output.mkdir(parents=True, exist_ok=True)
    prediction_dir = output / "predictions"
    manifest_dir = output / "split_manifests"
    prediction_dir.mkdir(exist_ok=True)
    manifest_dir.mkdir(exist_ok=True)
    effective_allowlist = set(case.evaluation.model_allowlist) or None
    if model_allowlist is not None:
        effective_allowlist = (
            model_allowlist
            if effective_allowlist is None
            else effective_allowlist & model_allowlist
        )
    records: list[dict[str, Any]] = []
    ranking_predictions: list[pd.DataFrame] = []
    learning_records: list[dict[str, Any]] = []
    split_hashes: dict[str, str] = {}
    for scenario in case.generalization_scenarios:
        for seed in seeds:
            manifest = manifests[(scenario.name, seed)]
            manifest_path = manifest_dir / f"{scenario.name}__seed-{seed}.json"
            train_indices = np.asarray(manifest.train_indices)
            test_indices = np.asarray(manifest.test_indices)
            manifest.save(manifest_path)
            split_hashes[f"{scenario.name}:{seed}"] = manifest.fingerprint()
            for representation, features in feature_frames.items():
                suite = build_probe_suite(features, case.task.kind, seed, budget)
                if effective_allowlist is not None:
                    suite = {
                        name: model for name, model in suite.items() if name in effective_allowlist
                    }
                if not suite:
                    raise ValueError(
                        "No configured model is available for the requested task and budget"
                    )
                for model_name, model in suite.items():
                    run_id = f"{scenario.name}__{representation}__{model_name}__seed-{seed}"
                    fit_started = time.perf_counter()
                    prediction_indices = np.sort(np.concatenate([train_indices, test_indices]))
                    test_mask = np.isin(prediction_indices, test_indices)
                    model.fit(features.loc[train_indices], target[train_indices])
                    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                        if case.task.kind == "binary_classification":
                            y_prediction = model.predict_proba(features.loc[prediction_indices])[
                                :, 1
                            ]
                        else:
                            y_prediction = model.predict(features.loc[prediction_indices])
                    y_test = y_prediction[test_mask]
                    runtime = time.perf_counter() - fit_started
                    metrics = compute_metrics(
                        target[test_indices],
                        np.asarray(y_test),
                        case.task.kind,
                        classification_threshold=case.evaluation.classification_threshold,
                    )
                    record = _record(
                        scenario.name,
                        scenario.strategy,
                        seed,
                        model_name,
                        False,
                        runtime,
                        len(train_indices),
                        len(test_indices),
                        metrics,
                        representation=representation,
                        model_configuration=_model_configuration(model),
                    )
                    group_column = _ranking_group_column(case)
                    if group_column and group_column in frame:
                        ranking_direction = 1.0 if case.task.higher_is_better else -1.0
                        test_prediction = pd.DataFrame(
                            {
                                group_column: frame.iloc[test_indices][group_column].to_numpy(),
                                "y_true": ranking_direction * target[test_indices],
                                "y_pred": ranking_direction * y_test,
                            }
                        )
                        record.update(
                            per_group_ranking_metrics(
                                test_prediction,
                                group_column=group_column,
                                k=min(case.decision.k),
                            )
                        )
                    records.append(record)
                    prediction = _prediction_frame(
                        frame,
                        target,
                        np.asarray(y_prediction),
                        prediction_indices,
                        test_indices,
                        run_id,
                        scenario.name,
                        model_name,
                        representation,
                    )
                    prediction.to_parquet(prediction_dir / f"{run_id}.parquet", index=False)
                    if model_name != "dummy":
                        ranking_predictions.append(prediction)
                        groups = _permutation_groups(frame.iloc[train_indices], case)
                        for draw in range(case.evaluation.permutation_draws):
                            permutation_seed = seed + 10_000 + 1_000 * draw
                            permuted_target = group_respecting_permutation(
                                target[train_indices], groups, permutation_seed
                            )
                            perm_model = build_probe_suite(features, case.task.kind, seed, budget)[
                                model_name
                            ]
                            perm_started = time.perf_counter()
                            perm_model.fit(features.loc[train_indices], permuted_target)
                            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                                if case.task.kind == "binary_classification":
                                    perm_prediction = perm_model.predict_proba(
                                        features.loc[test_indices]
                                    )[:, 1]
                                else:
                                    perm_prediction = perm_model.predict(features.loc[test_indices])
                            records.append(
                                _record(
                                    scenario.name,
                                    scenario.strategy,
                                    seed,
                                    model_name,
                                    True,
                                    time.perf_counter() - perm_started,
                                    len(train_indices),
                                    len(test_indices),
                                    compute_metrics(
                                        target[test_indices],
                                        np.asarray(perm_prediction),
                                        case.task.kind,
                                        classification_threshold=(
                                            case.evaluation.classification_threshold
                                        ),
                                    ),
                                    representation=representation,
                                    model_configuration=_model_configuration(perm_model),
                                    permutation_draw=draw,
                                )
                            )
                            permutation_run_id = (
                                f"{scenario.name}__{representation}__{model_name}__"
                                f"permuted-{draw}__seed-{seed}"
                            )
                            permutation_frame = frame.iloc[test_indices].copy()
                            permutation_frame["_row_id"] = test_indices
                            permutation_frame["y_true"] = target[test_indices]
                            permutation_frame["y_pred"] = perm_prediction
                            permutation_frame["is_test"] = True
                            permutation_frame["run_id"] = permutation_run_id
                            permutation_frame["scenario"] = scenario.name
                            permutation_frame["model"] = model_name
                            permutation_frame["representation"] = representation
                            permutation_frame["permuted"] = True
                            permutation_frame["permutation_draw"] = draw
                            permutation_frame.to_parquet(
                                prediction_dir / f"{permutation_run_id}.parquet", index=False
                            )
                group_for_curve = _scenario_group_column(scenario)
                if seed == seeds[0] and group_for_curve and group_for_curve in frame:
                    learning_records.extend(
                        _learning_curve(
                            frame,
                            features,
                            target,
                            train_indices,
                            test_indices,
                            group_for_curve,
                            scenario.name,
                            representation,
                            case,
                            budget,
                            seed,
                            effective_allowlist,
                        )
                    )
            if case.task.kind != "binary_classification" and len(case.entities) >= 2:
                entity_columns = [entity.id_column for entity in case.entities.values()]
                if all(column in frame for column in entity_columns[:2]):
                    baselines = entity_mean_predictions(
                        frame.iloc[train_indices],
                        frame.iloc[test_indices],
                        target=case.task.target_column,
                        left=entity_columns[0],
                        right=entity_columns[1],
                    )
                    for model_name, baseline_prediction in baselines.items():
                        run_id = f"{scenario.name}__{model_name}__seed-{seed}"
                        metrics = compute_metrics(
                            target[test_indices],
                            baseline_prediction,
                            case.task.kind,
                            classification_threshold=case.evaluation.classification_threshold,
                        )
                        records.append(
                            _record(
                                scenario.name,
                                scenario.strategy,
                                seed,
                                model_name,
                                False,
                                0.0,
                                len(train_indices),
                                len(test_indices),
                                metrics,
                                representation="not_applicable",
                                model_configuration=json.dumps(
                                    {"type": model_name}, sort_keys=True
                                ),
                            )
                        )
                        baseline_frame = frame.iloc[test_indices].copy()
                        baseline_frame["_row_id"] = test_indices
                        baseline_frame["y_true"] = target[test_indices]
                        baseline_frame["y_pred"] = baseline_prediction
                        baseline_frame["is_test"] = True
                        baseline_frame["run_id"] = run_id
                        baseline_frame["scenario"] = scenario.name
                        baseline_frame["model"] = model_name
                        baseline_frame["representation"] = "not_applicable"
                        baseline_frame.to_parquet(prediction_dir / f"{run_id}.parquet", index=False)
    experiments = pd.DataFrame(records)
    learning_curve = pd.DataFrame(
        learning_records,
        columns=[
            "scenario",
            "representation",
            "model",
            "independent_group",
            "group_count",
            "train_rows",
            "primary_metric",
        ],
    )
    learning_curve.to_parquet(output / "learning_curve.parquet", index=False)
    ranking_input = (
        pd.concat(ranking_predictions, ignore_index=True) if ranking_predictions else pd.DataFrame()
    )
    ranking, ranking_table, rankings_by_representation = _ranking_results(
        ranking_input, frame, case, list(feature_frames)
    )
    decomposition = stability_decomposition(experiments, case.evaluation.primary_metric)
    capability = _capabilities_by_representation(
        experiments,
        case,
        rankings_by_representation,
        list(feature_frames),
        audits=audits,
        overlap_results=overlap_results,
    )
    representation_sensitivity = _representation_sensitivity(
        capability, experiments, case.evaluation.primary_metric
    )
    provenance = build_provenance(
        case,
        dataset_fingerprint=audits["inventory"]["dataset_fingerprint"],
        split_hashes=split_hashes,
        runtime_seconds=time.perf_counter() - started,
    )
    structured = {
        "schema_version": 1,
        "case": case.model_dump(mode="json"),
        "dataset_source": dataset_source,
        "pre_model_readiness": pre_model_readiness,
        "pre_model_findings": pre_model_findings,
        "audits": audits,
        "split_overlap": overlap_results,
        "experiment_summary": _experiment_summary(experiments, case.evaluation.primary_metric),
        "learning_curve": learning_records,
        "stability_decomposition": decomposition,
        "ranking_stability": ranking,
        "representation_sensitivity": representation_sensitivity,
        "capability_matrix": capability,
        "holdout_access": holdout_access,
        "provenance": provenance,
    }
    write_report(
        output,
        case=case,
        structured=_json_safe(structured),
        experiments=experiments,
        ranking_table=ranking_table,
    )
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    return cast(dict[str, Any], _json_safe(structured))


def _require_model_readiness(readiness: dict[str, Any]) -> None:
    if readiness["model_fitting_allowed"]:
        return
    blockers = ", ".join(readiness["blocking_checks"])
    raise ValueError(
        f"Pre-model readiness gate BLOCKED by: {blockers}. "
        "Resolve ACTION_REQUIRED findings before model fitting."
    )


def _ranking_results(
    predictions: pd.DataFrame,
    frame: pd.DataFrame,
    case: CaseSpec,
    representations: list[str],
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, dict[str, Any]]]:
    group_column = _ranking_group_column(case)
    candidate_column = _ranking_candidate_column(frame, case, group_column)
    if not group_column or not candidate_column:
        unavailable = {
            representation: {
                "status": "NOT_ASSESSABLE",
                "reason": "No grouped decision was declared.",
            }
            for representation in representations
        }
        return (
            {
                "status": "NOT_ASSESSABLE",
                "reason": "No grouped decision was declared.",
                "by_representation": unavailable,
            },
            pd.DataFrame(),
            unavailable,
        )
    by_representation: dict[str, dict[str, Any]] = {}
    tables = []
    for representation in representations:
        subset = (
            predictions[predictions["representation"].eq(representation)]
            if "representation" in predictions
            else pd.DataFrame()
        )
        summary, table = ranking_stability(
            subset,
            group_column=group_column,
            candidate_column=candidate_column,
            k_values=case.decision.k,
            higher_is_better=case.task.higher_is_better,
        )
        by_representation[representation] = summary
        if not table.empty:
            table.insert(0, "representation", representation)
            tables.append(table)
    overall, _ = ranking_stability(
        predictions,
        group_column=group_column,
        candidate_column=candidate_column,
        k_values=case.decision.k,
        higher_is_better=case.task.higher_is_better,
    )
    overall["by_representation"] = by_representation
    ranking_table = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    return overall, ranking_table, by_representation


def _capabilities_by_representation(
    experiments: pd.DataFrame,
    case: CaseSpec,
    rankings: dict[str, dict[str, Any]],
    representations: list[str],
    *,
    audits: dict[str, Any],
    overlap_results: dict[str, Any],
) -> list[dict[str, Any]]:
    scenario_claims = {scenario.name for scenario in case.generalization_scenarios}
    if case.decision.kind == "top_k_per_group":
        scenario_claims.add(
            f"top-{min(case.decision.k)} selection within {case.decision.group_entity}"
        )
    represented: list[dict[str, Any]] = []
    shared: list[dict[str, Any]] = []
    for index, representation in enumerate(representations):
        matrix = capability_matrix(
            experiments[experiments["representation"].eq(representation)],
            case,
            rankings[representation],
            audits=audits,
            overlap_results=overlap_results,
        )
        for verdict in matrix:
            if verdict["claim_or_scenario"] in scenario_claims:
                verdict["representation"] = representation
                represented.append(verdict)
            elif index == 0:
                shared.append(verdict)
    return [*represented, *shared]


def _representation_sensitivity(
    capability: list[dict[str, Any]], experiments: pd.DataFrame, primary_metric: str
) -> list[dict[str, Any]]:
    claims: dict[str, list[dict[str, Any]]] = {}
    for verdict in capability:
        if "representation" in verdict:
            claims.setdefault(str(verdict["claim_or_scenario"]), []).append(verdict)
    status_order = {
        "NOT_ASSESSABLE": 0,
        "CONTRADICTED": 1,
        "INSUFFICIENT_EVIDENCE": 2,
        "SUPPORTED_WITH_LIMITS": 3,
        "SUPPORTED": 4,
    }
    rows = []
    real = experiments[experiments["model"].ne("dummy") & experiments["permuted"].eq(False)]
    for claim, verdicts in claims.items():
        statuses = {row["representation"]: row["status"] for row in verdicts}
        medians = {row["representation"]: row["numbers"].get("median") for row in verdicts}
        best_models = {row["representation"]: row["numbers"].get("best_model") for row in verdicts}
        finite = [float(value) for value in medians.values() if value is not None]
        scenario = real[real["scenario"].eq(claim)]
        matched_models: dict[str, dict[str, float]] = {}
        matched_ranges = []
        for model, model_rows in scenario.groupby("model"):
            model_medians = {
                str(representation): float(values.median())
                for representation, values in model_rows.groupby("representation")[primary_metric]
                if values.notna().any()
            }
            if set(model_medians) == set(statuses):
                matched_models[str(model)] = model_medians
                matched_ranges.append(max(model_medians.values()) - min(model_medians.values()))
        rows.append(
            {
                "claim_or_scenario": claim,
                "status_by_representation": statuses,
                "median_by_representation": medians,
                "best_model_by_representation": best_models,
                "matched_model_medians": matched_models,
                "verdict_changes": len(set(statuses.values())) > 1,
                "capability_median_range": max(finite) - min(finite) if finite else None,
                "maximum_matched_model_range": max(matched_ranges) if matched_ranges else None,
                "conservative_status": min(
                    statuses.values(), key=lambda status: status_order.get(status, -1)
                ),
            }
        )
    return rows


def _record(
    scenario: str,
    strategy: str,
    seed: int,
    model: str,
    permuted: bool,
    runtime: float,
    train_rows: int,
    test_rows: int,
    metrics: dict[str, float | None],
    *,
    representation: str,
    model_configuration: str,
    permutation_draw: int | None = None,
) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "strategy": strategy,
        "representation": representation,
        "seed": seed,
        "model": model,
        "permuted": permuted,
        "runtime_seconds": runtime,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "model_configuration": model_configuration,
        "permutation_draw": permutation_draw,
    } | metrics


def _model_configuration(model: Any) -> str:
    parameters = model.get_params(deep=True)
    compact = {
        key: value
        for key, value in parameters.items()
        if "__" in key and not hasattr(value, "get_params")
    }
    return json.dumps(compact, default=str, sort_keys=True)


def _prediction_frame(
    frame: pd.DataFrame,
    target: npt.NDArray[Any],
    prediction: npt.NDArray[Any],
    row_indices: npt.NDArray[np.int64],
    test_indices: npt.NDArray[np.int64],
    run_id: str,
    scenario: str,
    model: str,
    representation: str,
) -> pd.DataFrame:
    result = cast(pd.DataFrame, frame.iloc[row_indices].copy())
    result["_row_id"] = row_indices
    result["y_true"] = target[row_indices]
    result["y_pred"] = prediction
    result["is_test"] = np.isin(row_indices, test_indices)
    result["run_id"] = run_id
    result["scenario"] = scenario
    result["model"] = model
    result["representation"] = representation
    return result


def _permutation_groups(frame: pd.DataFrame, case: CaseSpec) -> npt.NDArray[Any] | None:
    column = case.evaluation.bootstrap_unit
    return frame[column].to_numpy() if column and column in frame else None


def _ranking_group_column(case: CaseSpec) -> str | None:
    group = case.decision.group_entity
    if not group:
        return None
    return case.entities[group].id_column if group in case.entities else group


def _scenario_group_column(scenario: ScenarioSpec) -> str | None:
    if scenario.strategy == "group":
        return scenario.group_column
    if scenario.strategy == "cold_left":
        return scenario.left_column
    if scenario.strategy == "cold_right":
        return scenario.right_column
    if scenario.strategy == "double_cold":
        return scenario.left_column
    return None


def _learning_curve(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    target: npt.NDArray[Any],
    train_indices: npt.NDArray[np.int64],
    test_indices: npt.NDArray[np.int64],
    group_column: str,
    scenario: str,
    representation: str,
    case: CaseSpec,
    budget: str,
    seed: int,
    model_allowlist: set[str] | None,
) -> list[dict[str, Any]]:
    groups = frame.iloc[train_indices][group_column].dropna().unique()
    if len(groups) < 4:
        return []
    order = np.random.default_rng(seed).permutation(groups)
    suite = build_probe_suite(features, case.task.kind, seed, budget)
    preferred = "logistic" if case.task.kind == "binary_classification" else "elastic_net"
    allowed = set(suite) if model_allowlist is None else set(suite) & model_allowlist
    model_name = (
        preferred
        if preferred in allowed
        else next((name for name in allowed if name != "dummy"), None)
    )
    if model_name is None:
        return []
    rows = []
    for fraction in [0.35, 0.65, 1.0]:
        group_count = max(3, round(fraction * len(order)))
        chosen = set(order[:group_count])
        subset = train_indices[frame.iloc[train_indices][group_column].isin(chosen).to_numpy()]
        subset_readiness = assess_readiness(
            readiness_findings(
                audit_dataset(frame.iloc[subset], case),
                case,
                allow_pending_identity_policies=False,
            )
        )
        if not subset_readiness["model_fitting_allowed"]:
            continue
        model = build_probe_suite(features, case.task.kind, seed, budget)[model_name]
        model.fit(features.loc[subset], target[subset])
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            if case.task.kind == "binary_classification":
                prediction = model.predict_proba(features.loc[test_indices])[:, 1]
            else:
                prediction = model.predict(features.loc[test_indices])
        metrics = compute_metrics(
            target[test_indices],
            np.asarray(prediction),
            case.task.kind,
            classification_threshold=case.evaluation.classification_threshold,
        )
        rows.append(
            {
                "scenario": scenario,
                "representation": representation,
                "model": model_name,
                "independent_group": group_column,
                "group_count": int(group_count),
                "train_rows": int(len(subset)),
                "primary_metric": metrics.get(case.evaluation.primary_metric),
            }
        )
    return rows


def _ranking_candidate_column(
    frame: pd.DataFrame, case: CaseSpec, group_column: str | None
) -> str | None:
    if not group_column or case.decision.kind != "top_k_per_group":
        return None
    other_entities = [
        entity.id_column for entity in case.entities.values() if entity.id_column != group_column
    ]
    for column in [*other_entities, "candidate_id", "sample_id", "_row_id"]:
        if column in frame or column == "_row_id":
            return column
    return None


def _test_target_counts(
    frame: pd.DataFrame,
    test_indices: npt.NDArray[np.int64],
    case: CaseSpec,
) -> tuple[dict[str, int], str]:
    target = case.task.target_column
    unit = case.evaluation.bootstrap_unit
    test = frame.iloc[test_indices]
    if unit and unit in test:
        target_counts = test.groupby(unit, dropna=False)[target].nunique(dropna=False)
        if (target_counts > 1).any():
            raise ValueError(f"Independent test unit {unit!r} maps to conflicting targets")
        support = test.drop_duplicates(unit)
        count_unit = unit
    else:
        support = test
        count_unit = "rows"
    return (
        {
            str(label): int(count)
            for label, count in support[target].value_counts().sort_index().items()
        },
        count_unit,
    )


def _validate_protected_entity_isolation(
    case: CaseSpec, scenario: ScenarioSpec, overlap: dict[str, Any]
) -> None:
    if scenario.strategy not in {"group", "scaffold", "supplied"}:
        return
    grouping_column = scenario.group_column
    prediction_unit = case.task.prediction_unit.lower().replace("-", "_").replace(" ", "_")
    for name, entity in case.entities.items():
        entity_columns = {entity.id_column, entity.representation_column}
        explicitly_grouped = grouping_column is not None and grouping_column in entity_columns
        locked_prediction_entity = (
            scenario.strategy == "supplied"
            and case.holdout.enabled
            and (
                prediction_unit in {name.lower(), entity.id_column.lower().removesuffix("_id")}
                or (
                    case.evaluation.bootstrap_unit is not None
                    and case.evaluation.bootstrap_unit in entity_columns
                )
            )
        )
        if not explicitly_grouped and not locked_prediction_entity:
            continue
        count = overlap["entity_overlap"].get(name, {}).get("count", 0)
        if count:
            raise ValueError(
                f"Scenario {scenario.name!r} cannot provide held-out {name} evidence: "
                f"{count} identifiers cross train and test; resolve identity conflicts first"
            )
    if (
        scenario.strategy == "supplied"
        and case.holdout.enabled
        and len(case.entities) == 2
        and (prediction_unit == "pair" or prediction_unit.endswith("_pair"))
        and overlap.get("pair_overlap")
    ):
        raise ValueError(
            f"Scenario {scenario.name!r} cannot provide held-out pair evidence: "
            f"{overlap['pair_overlap']} pairs cross train and test"
        )


def _experiment_summary(experiments: pd.DataFrame, primary: str) -> list[dict[str, Any]]:
    result = []
    for (scenario, representation, model, permuted), group in experiments.groupby(
        ["scenario", "representation", "model", "permuted"]
    ):
        values = group[primary].dropna()
        result.append(
            {
                "scenario": scenario,
                "representation": representation,
                "model": model,
                "permuted": bool(permuted),
                "median": float(values.median()) if len(values) else None,
                "standard_deviation": float(values.std()) if len(values) > 1 else 0.0,
                "runs": int(len(group)),
            }
        )
    return result


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
