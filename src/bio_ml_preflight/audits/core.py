from __future__ import annotations

import hashlib
import json
from numbers import Real
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from bio_ml_preflight.contracts import CaseSpec
from bio_ml_preflight.data import dataset_fingerprint
from bio_ml_preflight.features import model_feature_columns


def _serial(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def audit_dataset(frame: pd.DataFrame, case: CaseSpec) -> dict[str, Any]:
    target = case.task.target_column
    missing = {column: int(count) for column, count in frame.isna().sum().items() if count}
    constants = [column for column in frame if frame[column].nunique(dropna=False) <= 1]
    near_constants = [
        column
        for column in frame
        if column not in constants
        and frame[column].value_counts(normalize=True, dropna=False).iloc[0] >= 0.98
    ]
    invalid_numeric = {}
    for column in frame:
        values = frame[column]
        if pd.api.types.is_numeric_dtype(values):
            count = int(np.isinf(values.to_numpy(dtype=float, na_value=np.nan)).sum())
        else:
            count = sum(isinstance(value, Real) and bool(np.isinf(value)) for value in values)
        if count:
            invalid_numeric[column] = count
    high_cardinality = [
        column
        for column in frame.select_dtypes(exclude=[np.number])
        if frame[column].nunique(dropna=True) > min(100, max(20, len(frame) // 2))
    ]
    configured_features = model_feature_columns(frame.columns, case)
    modeled_features = [column for column in configured_features if column in frame]
    suspicious = [
        column
        for column in modeled_features
        if ("id" in column.lower() or "token" in column.lower())
        and frame[column].nunique(dropna=True) / max(len(frame), 1) > 0.05
    ]
    entities: dict[str, Any] = {}
    entity_columns: list[str] = []
    for name, entity in case.entities.items():
        column = entity.id_column
        if column not in frame:
            entities[name] = {"status": "NOT_ASSESSABLE", "reason": f"missing {column}"}
            continue
        observed = frame.loc[frame[column].notna()]
        counts = observed[column].value_counts()
        entity_result: dict[str, Any] = {
            "id_column": column,
            "unique": int(counts.size),
            "missing_identifier_rows": int(frame[column].isna().sum()),
            "rows_per_entity": {
                "min": int(counts.min()),
                "median": float(counts.median()),
                "max": int(counts.max()),
            }
            if not counts.empty
            else {},
            "duplicate_identifier_rows": int(counts[counts > 1].sum()),
        }
        if counts.empty:
            entity_result.update(
                {
                    "status": "NOT_ASSESSABLE",
                    "reason": f"all values in declared entity column {column!r} are missing",
                }
            )
            entities[name] = entity_result
            continue
        entity_columns.append(column)
        if _entity_defines_prediction_unit(case, name, entity.id_column):
            target_conflicts = observed.groupby(column)[target].nunique(dropna=True)
            entity_result["conflicting_target_entities"] = int((target_conflicts > 1).sum())
        representation = entity.representation_column
        if representation and representation in frame:
            representation_counts = observed.groupby(column)[representation].nunique(dropna=True)
            entity_result["representation_column"] = representation
            entity_result["inconsistent_representation_entities"] = int(
                (representation_counts > 1).sum()
            )
        elif representation:
            entity_result["representation_consistency"] = {
                "status": "NOT_ASSESSABLE",
                "reason": f"missing {representation}",
            }
        entities[name] = entity_result
    pair_columns = entity_columns[:2]
    pair: dict[str, Any] = {
        "status": "NOT_ASSESSABLE",
        "reason": "fewer than two entities",
        "defines_prediction_unit": _pair_defines_prediction_unit(case),
    }
    if len(pair_columns) == 2:
        complete_pairs = frame.dropna(subset=pair_columns)
        if complete_pairs.empty:
            pair = {
                "status": "NOT_ASSESSABLE",
                "reason": "no rows contain both declared entity identifiers",
                "defines_prediction_unit": _pair_defines_prediction_unit(case),
            }
        else:
            pair_counts = complete_pairs.groupby(pair_columns).size()
            label_conflicts = complete_pairs.groupby(pair_columns)[target].nunique(dropna=True)
            left_unique, right_unique = (
                complete_pairs[column].nunique(dropna=True) for column in pair_columns
            )
            pair = {
                "columns": pair_columns,
                "defines_prediction_unit": _pair_defines_prediction_unit(case),
                "unique_pairs": int(pair_counts.size),
                "repeated_pair_rows": int(pair_counts[pair_counts > 1].sum()),
                "conflicting_label_pairs": int((label_conflicts > 1).sum()),
                "pair_density": float(pair_counts.size / max(left_unique * right_unique, 1)),
                "left_degree": _distribution(
                    complete_pairs.groupby(pair_columns[0])[pair_columns[1]].nunique()
                ),
                "right_degree": _distribution(
                    complete_pairs.groupby(pair_columns[1])[pair_columns[0]].nunique()
                ),
            }
    replicates = _replicate_audit(frame, case)
    coverage = _coverage(frame, case)
    target_values = frame[target]
    target_summary: dict[str, Any]
    if case.task.kind == "binary_classification":
        observed_classes = set(target_values.dropna().unique())
        target_summary = {
            "class_counts": {
                str(key): int(value) for key, value in target_values.value_counts().items()
            },
            "zero_one_encoded": bool(
                pd.api.types.is_numeric_dtype(target_values) and observed_classes <= {0, 1}
            ),
        }
    else:
        target_summary = {
            str(key): _serial(value)
            for key, value in target_values.describe(percentiles=[0.05, 0.5, 0.95]).items()
        }
        target_numeric = bool(pd.api.types.is_numeric_dtype(target_values))
        finite_target = (
            pd.to_numeric(target_values, errors="coerce").replace([np.inf, -np.inf], np.nan)
            if target_numeric
            else pd.Series(dtype=float)
        )
        target_summary.update(
            {
                "numeric": target_numeric,
                "finite_unique": int(finite_target.nunique(dropna=True)),
            }
        )
    independence_warnings = []
    repeated = [
        name
        for name, result in entities.items()
        if result.get("rows_per_entity", {}).get("max", 1) > 1
    ]
    if repeated:
        independence_warnings.append(
            "Rows repeat declared entities; row bootstrap and random row splitting may "
            "overstate independence."
        )
    if any(result.get("missing_identifier_rows", 0) for result in entities.values()):
        independence_warnings.append(
            "Declared entity identifiers contain missing values; those rows cannot establish "
            "independent-unit support."
        )
    if any(result.get("conflicting_target_entities", 0) for result in entities.values()):
        independence_warnings.append(
            "Identical entity identifiers have conflicting targets; resolve label semantics "
            "before confirmatory use."
        )
    if any(result.get("inconsistent_representation_entities", 0) for result in entities.values()):
        independence_warnings.append(
            "Identical entity identifiers map to inconsistent representations; verify entity "
            "identity before confirmatory use."
        )
    if not case.evaluation.bootstrap_unit:
        independence_warnings.append("No bootstrap independent unit is configured.")
    return {
        "inventory": {
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
            "physical_types": {column: str(dtype) for column, dtype in frame.dtypes.items()},
            "missing": missing,
            "constant_columns": constants,
            "near_constant_columns": near_constants,
            "duplicate_rows": int(frame.duplicated().sum()),
            "high_cardinality_categorical": high_cardinality,
            "invalid_numeric_values": invalid_numeric,
            "target_distribution": target_summary,
            "dataset_fingerprint": dataset_fingerprint(frame, case.data.fingerprint_columns),
            "case_fingerprint": case.fingerprint(),
        },
        "independence": {
            "entities": entities,
            "pair_structure": pair,
            "warnings": independence_warnings,
            "interpretation": (
                "Entity counts are transparent proxies, not exact effective sample sizes."
            ),
        },
        "leakage": {
            "modeled_features": modeled_features,
            "missing_modeled_features": sorted(set(configured_features) - set(frame.columns)),
            "suspicious_identifier_features": suspicious,
            "declared_post_outcome_features": case.features.post_outcome,
            "pipeline_guard": "All learned preprocessing is fit inside each training fold.",
        },
        "measurement": replicates,
        "coverage": coverage,
        "missing_high_value_metadata": _missing_metadata(case),
    }


def apply_identity_conflict_policies(
    frame: pd.DataFrame,
    case: CaseSpec,
    *,
    policy_indices: pd.Index | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply explicit entity-level policies before feature building or splitting."""
    source = frame.copy()
    policy_source = (
        source if policy_indices is None else source.loc[source.index.intersection(policy_indices)]
    )
    resolved = frame.copy()
    entity_results: dict[str, Any] = {}
    kept_conflicts = False
    changed = False
    for name, entity in case.entities.items():
        identifier = entity.id_column
        if identifier not in policy_source:
            continue
        observed = policy_source.loc[policy_source[identifier].notna()]
        target_conflicts: set[Any] = set()
        if _entity_defines_prediction_unit(case, name, identifier):
            target_counts = observed.groupby(identifier)[case.task.target_column].nunique(
                dropna=True
            )
            target_conflicts = set(target_counts[target_counts > 1].index)
        representation_conflicts: set[Any] = set()
        representation = entity.representation_column
        if representation and representation in observed:
            representation_counts = observed.groupby(identifier)[representation].nunique(
                dropna=True
            )
            representation_conflicts = set(representation_counts[representation_counts > 1].index)
        affected = target_conflicts | representation_conflicts
        result = {
            "policy": entity.identity_conflict_policy,
            "conflicting_target_entities": len(target_conflicts),
            "inconsistent_representation_entities": len(representation_conflicts),
            "affected_entities": len(affected),
        }
        if not affected:
            entity_results[name] = result
            continue
        policy = entity.identity_conflict_policy
        if policy is None:
            raise ValueError(
                f"Entity {name!r} has {len(target_conflicts)} conflicting-target and "
                f"{len(representation_conflicts)} inconsistent-representation identifiers; "
                "set identity_conflict_policy to keep, exclude, or aggregate"
            )
        affected_mask = resolved[identifier].isin(affected)
        if policy_indices is not None:
            affected_mask &= resolved.index.isin(policy_indices)
        before = len(resolved)
        if policy == "keep":
            kept_conflicts = True
        elif policy == "exclude":
            resolved = resolved.loc[~affected_mask].copy()
            result["excluded_rows"] = before - len(resolved)
            changed = True
        else:
            if representation_conflicts:
                raise ValueError(
                    f"Entity {name!r} cannot aggregate distinct values from "
                    f"{representation!r}; choose keep or exclude"
                )
            resolved = _aggregate_conflicting_targets(resolved, case, identifier, affected_mask)
            result["aggregated_rows"] = before - len(resolved)
            changed = True
        entity_results[name] = result
    if resolved.empty:
        raise ValueError("Identity-conflict policies removed every row")
    status = "ACKNOWLEDGED" if kept_conflicts else "RESOLVED" if changed else "NO_CONFLICTS"
    return resolved.reset_index(drop=True), {
        "status": status,
        "source_rows": len(source),
        "policy_scope_rows": len(policy_source),
        "model_rows": len(resolved),
        "source_dataset_fingerprint": dataset_fingerprint(source, case.data.fingerprint_columns),
        "entities": entity_results,
    }


def audit_graph_readiness_contract(
    frame: pd.DataFrame,
    case: CaseSpec,
    identity_consistency: dict[str, Any],
) -> dict[str, Any]:
    """Validate a declared molecular-graph boundary without training a graph model."""
    graph = case.graph_readiness
    if graph is None:
        return {"status": "NOT_DECLARED"}
    converted_graphs = _canonical_molecular_graphs(frame[graph.structure_column])
    if converted_graphs is None:
        return {
            "status": "NOT_ASSESSABLE",
            "reason": "Graph construction audit requires: uv sync --extra chem",
        }
    canonical, atom_counts, bond_counts = converted_graphs
    invalid_rows = sum(value is None for value in canonical)
    if invalid_rows:
        raise ValueError(
            f"graph_readiness found {invalid_rows} invalid structures under the error policy"
        )

    unit = graph.independent_unit_column
    target = case.task.target_column
    converted = pd.DataFrame(
        {
            "unit": frame[unit].astype(str),
            "target": frame[target],
            "canonical_graph": canonical,
            "atom_count": atom_counts,
            "bond_count": bond_counts,
        }
    )
    graphs_per_unit = converted.groupby("unit", dropna=False)["canonical_graph"].nunique()
    targets_per_unit = converted.groupby("unit", dropna=False)["target"].nunique()
    support = converted.drop_duplicates("unit")
    class_counts = {
        str(label): int(count)
        for label, count in support["target"].value_counts().sort_index().items()
    }
    minimum = graph.minimum_independent_units_per_class
    enough_support = len(class_counts) == 2 and min(class_counts.values()) >= minimum
    graph_conflicts = int((graphs_per_unit > 1).sum())
    target_conflicts = int((targets_per_unit > 1).sum())
    identity_status = str(identity_consistency["status"])
    requirements_met = (
        identity_status in {"NO_CONFLICTS", "RESOLVED"}
        and graph_conflicts == 0
        and target_conflicts == 0
        and enough_support
    )
    identity_pairs = sorted(set(zip(converted["unit"], converted["canonical_graph"], strict=True)))
    fingerprint = hashlib.sha256(
        json.dumps(identity_pairs, separators=(",", ":")).encode()
    ).hexdigest()
    shared_graphs = (
        converted.drop_duplicates(["unit", "canonical_graph"])
        .groupby("canonical_graph")["unit"]
        .nunique()
    )
    metadata = {
        "replicate": any(
            column in frame
            for column in [case.metadata.replicate_id, case.metadata.biological_replicate_id]
            if column
        ),
        "batch": bool(case.metadata.batch_id and case.metadata.batch_id in frame),
        "time": bool(case.metadata.time_column and case.metadata.time_column in frame),
        "treatment": bool(
            case.metadata.treatment_column and case.metadata.treatment_column in frame
        ),
    }
    scenarios = {scenario.name: scenario for scenario in case.generalization_scenarios}
    return {
        "status": "VALID_CONTRACT" if requirements_met else "UNMET_REQUIREMENTS",
        "construction": graph.construction,
        "scope": graph.scope,
        "structure_column": graph.structure_column,
        "graph_identity": graph.graph_identity,
        "invalid_structure_policy": graph.invalid_structure_policy,
        "node_features": graph.node_features,
        "edge_features": graph.edge_features,
        "rows_converted": len(converted),
        "invalid_structure_rows": invalid_rows,
        "unique_independent_units": int(converted["unit"].nunique()),
        "unique_canonical_graphs": int(converted["canonical_graph"].nunique()),
        "topology": {
            "atom_count": _distribution(converted["atom_count"]),
            "bond_count": _distribution(converted["bond_count"]),
        },
        "independent_units_with_multiple_graphs": graph_conflicts,
        "independent_units_with_conflicting_targets": target_conflicts,
        "canonical_graphs_shared_across_independent_units": int((shared_graphs > 1).sum()),
        "conversion_fingerprint": fingerprint,
        "identity_consistency_status": identity_status,
        "support": {
            "unit": unit,
            "class_counts": class_counts,
            "minimum_independent_units_per_class": minimum,
            "meets_minimum": enough_support,
            "interpretation": (
                "Counts are transparent support proxies, not exact effective sample sizes."
            ),
        },
        "evaluation_scenarios": [
            {"name": name, "strategy": scenarios[name].strategy}
            for name in graph.evaluation_scenarios
        ],
        "metadata_available": metadata,
        "metadata_limit": (
            "Missing measurement or deployment metadata remains an evidence limit; graph "
            "construction does not repair it."
        ),
    }


def _canonical_molecular_graphs(
    values: pd.Series,
) -> tuple[list[str | None], list[int | None], list[int | None]] | None:
    try:
        from rdkit import Chem, rdBase
    except ImportError:
        return None
    canonical: list[str | None] = []
    atom_counts: list[int | None] = []
    bond_counts: list[int | None] = []
    with rdBase.BlockLogs():
        for value in values:
            molecule = Chem.MolFromSmiles(str(value))
            if molecule is None or molecule.GetNumAtoms() == 0:
                canonical.append(None)
                atom_counts.append(None)
                bond_counts.append(None)
                continue
            canonical.append(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True))
            atom_counts.append(int(molecule.GetNumAtoms()))
            bond_counts.append(int(molecule.GetNumBonds()))
    return canonical, atom_counts, bond_counts


def _aggregate_conflicting_targets(
    frame: pd.DataFrame,
    case: CaseSpec,
    identifier: str,
    affected_mask: pd.Series,
) -> pd.DataFrame:
    affected = frame.loc[affected_mask]
    if not case.features.include:
        raise ValueError("Aggregate policy requires an explicit features.include list")
    feature_columns = [
        column
        for column in model_feature_columns(frame.columns, case)
        if column in frame and column != case.task.target_column
    ]
    for column in feature_columns:
        counts = affected.groupby(identifier, dropna=False)[column].nunique(dropna=False)
        if (counts > 1).any():
            raise ValueError(
                f"Cannot aggregate {identifier!r}: modeled feature {column!r} varies within "
                "an identifier"
            )
    result = frame.copy()
    drop_indices: list[Any] = []
    target = case.task.target_column
    for indices in affected.groupby(identifier, dropna=False, sort=False).groups.values():
        positions = list(indices)
        values = frame.loc[positions, target]
        if case.task.kind == "binary_classification":
            counts = values.value_counts()
            if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
                raise ValueError(
                    f"Cannot aggregate tied binary labels for {identifier!r}; choose keep or "
                    "exclude"
                )
            aggregate = counts.index[0]
        else:
            aggregate = float(pd.to_numeric(values, errors="raise").mean())
        result.loc[positions[0], target] = aggregate
        drop_indices.extend(positions[1:])
    return result.drop(index=drop_indices)


def _entity_defines_prediction_unit(case: CaseSpec, name: str, identifier: str) -> bool:
    prediction_unit = case.task.prediction_unit.lower().replace("-", "_").replace(" ", "_")
    return prediction_unit in {
        name.lower(),
        identifier.lower().removesuffix("_id"),
    }


def _pair_defines_prediction_unit(case: CaseSpec) -> bool:
    prediction_unit = case.task.prediction_unit.lower().replace("-", "_").replace(" ", "_")
    return len(case.entities) == 2 and (
        prediction_unit == "pair" or prediction_unit.endswith("_pair")
    )


def _distribution(values: pd.Series) -> dict[str, float]:
    return {
        "min": float(values.min()),
        "median": float(values.median()),
        "max": float(values.max()),
    }


def _replicate_audit(frame: pd.DataFrame, case: CaseSpec) -> dict[str, Any]:
    columns = [case.metadata.replicate_id, case.metadata.biological_replicate_id]
    available = [column for column in columns if column and column in frame]
    target = case.task.target_column
    if not available:
        pair_columns = [
            entity.id_column for entity in case.entities.values() if entity.id_column in frame
        ]
        complete_pairs = frame.dropna(subset=pair_columns[:2]) if len(pair_columns) >= 2 else frame
        if (
            len(pair_columns) < 2
            or not complete_pairs.duplicated(pair_columns[:2], keep=False).any()
        ):
            return {
                "status": "NOT_ASSESSABLE",
                "reason": "No replicate metadata or repeated entity-pair measurements.",
            }
        available = pair_columns[:2]
    observed = frame.dropna(subset=available)
    grouped = observed.groupby(available)[target]
    repeated = grouped.size()
    repeated_groups = int((repeated > 1).sum())
    if repeated_groups == 0:
        return {
            "status": "NOT_ASSESSABLE",
            "reason": "Replicate columns are present but no observed identifier repeats.",
            "grouping_columns": available,
            "repeated_groups": 0,
            "missing_group_rows": int(len(frame) - len(observed)),
        }
    conflicting_label_rate = float((grouped.nunique()[repeated > 1] > 1).mean())
    if case.task.kind == "binary_classification":
        return {
            "status": "NOT_ASSESSABLE",
            "reason": (
                "Repeated class labels assess label consistency, not measurement reliability."
            ),
            "grouping_columns": available,
            "repeated_groups": repeated_groups,
            "missing_group_rows": int(len(frame) - len(observed)),
            "label_consistency_assessed": True,
            "conflicting_label_rate": conflicting_label_rate,
            "unmet_assumption": (
                "The measured quantity and replicate protocol needed to estimate reliability "
                "were not declared."
            ),
            "cheapest_next_evidence": (
                "Declare the measured quantity and replicate protocol, then repeat a "
                "representative subset."
            ),
        }
    if pd.api.types.is_numeric_dtype(frame[target]):
        within = grouped.std().dropna()
        means = grouped.mean()
        within_median = float(within.median()) if len(within) else None
        between_dispersion = float(means.std()) if len(means) > 1 else None
    else:
        within_median = None
        between_dispersion = None
    return {
        "status": "ASSESSED",
        "grouping_columns": available,
        "repeated_groups": repeated_groups,
        "missing_group_rows": int(len(frame) - len(observed)),
        "within_group_dispersion_median": within_median,
        "between_group_dispersion": between_dispersion,
        "conflicting_label_rate": conflicting_label_rate,
        "noise_warning": "A dispersion proxy is not a proven noise ceiling.",
    }


def _coverage(frame: pd.DataFrame, case: CaseSpec) -> dict[str, Any]:
    columns = {entity.id_column for entity in case.entities.values() if entity.id_column in frame}
    columns.update(
        column
        for column in [
            case.metadata.batch_id,
            case.metadata.plate_id,
            case.metadata.time_column,
            case.metadata.treatment_column,
        ]
        if column and column in frame
    )
    return {
        column: {
            "unique": int(frame[column].nunique(dropna=True)),
            "missing": int(frame[column].isna().sum()),
        }
        for column in sorted(columns)
    }


def _missing_metadata(case: CaseSpec) -> list[str]:
    missing = []
    if not case.metadata.batch_id:
        missing.append("batch_id: batch confounding is not assessable")
    if not case.metadata.replicate_id and not case.metadata.biological_replicate_id:
        missing.append("replicate identifiers: measurement reliability is limited")
    if (
        any(s.strategy == "time" for s in case.generalization_scenarios)
        and not case.metadata.time_column
    ):
        missing.append("time_column: temporal generalization is not assessable")
    return missing


def audit_overlap(
    frame: pd.DataFrame,
    train_indices: npt.NDArray[np.int64],
    test_indices: npt.NDArray[np.int64],
    case: CaseSpec,
) -> dict[str, Any]:
    train, test = frame.iloc[train_indices], frame.iloc[test_indices]
    entity_overlap: dict[str, Any] = {}
    entity_columns = []
    for name, entity in case.entities.items():
        if entity.id_column in frame:
            entity_columns.append(entity.id_column)
            train_ids, test_ids = set(train[entity.id_column]), set(test[entity.id_column])
            entity_overlap[name] = {
                "count": len(train_ids & test_ids),
                "test_fraction": len(train_ids & test_ids) / max(len(test_ids), 1),
            }
    pair_overlap = None
    if len(entity_columns) >= 2:
        pair_columns = entity_columns[:2]
        train_pairs = set(map(tuple, train[pair_columns].itertuples(index=False, name=None)))
        test_pairs = set(map(tuple, test[pair_columns].itertuples(index=False, name=None)))
        pair_overlap = len(train_pairs & test_pairs)
    exact_hash_columns = case.data.fingerprint_columns or list(frame.columns)
    train_hash = set(pd.util.hash_pandas_object(train[exact_hash_columns], index=False))
    test_hash = set(pd.util.hash_pandas_object(test[exact_hash_columns], index=False))
    result: dict[str, Any] = {
        "train_rows": len(train),
        "test_rows": len(test),
        "exact_duplicate_overlap": len(train_hash & test_hash),
        "entity_overlap": entity_overlap,
        "pair_overlap": pair_overlap,
        "near_duplicate_overlap": {
            "status": "NOT_ASSESSABLE",
            "reason": "No similarity function was configured.",
        },
    }
    graph = case.graph_readiness
    if graph is not None:
        converted = _canonical_molecular_graphs(frame[graph.structure_column])
        if converted is None:
            result["canonical_graph_overlap"] = {
                "status": "NOT_ASSESSABLE",
                "reason": "Graph overlap audit requires: uv sync --extra chem",
            }
        else:
            canonical = converted[0]
            train_graphs = {canonical[index] for index in train_indices}
            test_graphs = {canonical[index] for index in test_indices}
            overlap = train_graphs & test_graphs
            result["canonical_graph_overlap"] = {
                "count": len(overlap),
                "test_fraction": len(overlap) / max(len(test_graphs), 1),
            }
    return result
