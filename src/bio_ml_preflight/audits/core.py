from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from bio_ml_preflight.contracts import CaseSpec
from bio_ml_preflight.data import dataset_fingerprint


def _serial(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def audit_dataset(frame: pd.DataFrame, case: CaseSpec) -> dict[str, Any]:
    target = case.task.target_column
    missing = {column: int(count) for column, count in frame.isna().sum().items() if count}
    numeric = frame.select_dtypes(include=[np.number])
    constants = [column for column in frame if frame[column].nunique(dropna=False) <= 1]
    near_constants = [
        column
        for column in frame
        if column not in constants
        and frame[column].value_counts(normalize=True, dropna=False).iloc[0] >= 0.98
    ]
    invalid_numeric = {
        column: int((~np.isfinite(numeric[column].to_numpy(dtype=float))).sum())
        for column in numeric
        if (~np.isfinite(numeric[column].to_numpy(dtype=float))).any()
    }
    high_cardinality = [
        column
        for column in frame.select_dtypes(exclude=[np.number])
        if frame[column].nunique(dropna=True) > min(100, max(20, len(frame) // 2))
    ]
    suspicious = [
        column
        for column in case.features.include
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
        entity_columns.append(column)
        counts = frame[column].value_counts()
        entity_result: dict[str, Any] = {
            "id_column": column,
            "unique": int(counts.size),
            "rows_per_entity": {
                "min": int(counts.min()),
                "median": float(counts.median()),
                "max": int(counts.max()),
            },
            "duplicate_identifier_rows": int(frame[column].duplicated(keep=False).sum()),
        }
        if _entity_defines_prediction_unit(case, name, entity.id_column):
            target_conflicts = frame.groupby(column, dropna=False)[target].nunique(dropna=True)
            entity_result["conflicting_target_entities"] = int((target_conflicts > 1).sum())
        representation = entity.representation_column
        if representation and representation in frame:
            representation_counts = frame.groupby(column, dropna=False)[representation].nunique(
                dropna=True
            )
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
    pair: dict[str, Any] = {"status": "NOT_ASSESSABLE", "reason": "fewer than two entities"}
    if len(pair_columns) == 2:
        pair_counts = frame.groupby(pair_columns, dropna=False).size()
        label_conflicts = frame.groupby(pair_columns, dropna=False)[target].nunique(dropna=True)
        left_unique, right_unique = (frame[column].nunique(dropna=True) for column in pair_columns)
        pair = {
            "columns": pair_columns,
            "unique_pairs": int(pair_counts.size),
            "repeated_pair_rows": int(pair_counts[pair_counts > 1].sum()),
            "conflicting_label_pairs": int((label_conflicts > 1).sum()),
            "pair_density": float(pair_counts.size / max(left_unique * right_unique, 1)),
            "left_degree": _distribution(frame.groupby(pair_columns[0])[pair_columns[1]].nunique()),
            "right_degree": _distribution(
                frame.groupby(pair_columns[1])[pair_columns[0]].nunique()
            ),
        }
    replicates = _replicate_audit(frame, case)
    coverage = _coverage(frame, case)
    target_values = frame[target]
    target_summary: dict[str, Any]
    if case.task.kind == "binary_classification":
        target_summary = {
            "class_counts": {
                str(key): int(value) for key, value in target_values.value_counts().items()
            }
        }
    else:
        target_summary = {
            str(key): _serial(value)
            for key, value in target_values.describe(percentiles=[0.05, 0.5, 0.95]).items()
        }
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
            "suspicious_identifier_features": suspicious,
            "declared_post_outcome_features": case.features.post_outcome,
            "pipeline_guard": "All learned preprocessing is fit inside each training fold.",
        },
        "measurement": replicates,
        "coverage": coverage,
        "missing_high_value_metadata": _missing_metadata(case),
    }


def apply_identity_conflict_policies(
    frame: pd.DataFrame, case: CaseSpec
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply explicit entity-level policies before feature building or splitting."""
    source = frame.copy()
    resolved = frame.copy()
    entity_results: dict[str, Any] = {}
    kept_conflicts = False
    changed = False
    for name, entity in case.entities.items():
        identifier = entity.id_column
        if identifier not in source:
            continue
        target_conflicts: set[Any] = set()
        if _entity_defines_prediction_unit(case, name, identifier):
            target_counts = source.groupby(identifier, dropna=False)[
                case.task.target_column
            ].nunique(dropna=True)
            target_conflicts = set(target_counts[target_counts > 1].index)
        representation_conflicts: set[Any] = set()
        representation = entity.representation_column
        if representation and representation in source:
            representation_counts = source.groupby(identifier, dropna=False)[
                representation
            ].nunique(dropna=True)
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
        "model_rows": len(resolved),
        "source_dataset_fingerprint": dataset_fingerprint(source, case.data.fingerprint_columns),
        "entities": entity_results,
    }


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
        for column in case.features.include
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
    return len(case.entities) == 1 and prediction_unit in {
        name.lower(),
        identifier.lower().removesuffix("_id"),
    }


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
        if len(pair_columns) < 2 or not frame.duplicated(pair_columns[:2], keep=False).any():
            return {
                "status": "NOT_ASSESSABLE",
                "reason": "No replicate metadata or repeated entity-pair measurements.",
            }
        available = pair_columns[:2]
    grouped = frame.groupby(available, dropna=False)[target]
    within = grouped.std().dropna()
    means = grouped.mean()
    repeated = grouped.size()
    consistency = grouped.apply(lambda values: values.rank().corr(pd.Series(values).rank()))
    return {
        "status": "ASSESSED",
        "grouping_columns": available,
        "repeated_groups": int((repeated > 1).sum()),
        "within_group_dispersion_median": float(within.median()) if len(within) else None,
        "between_group_dispersion": float(means.std()) if len(means) > 1 else None,
        "conflicting_label_rate": float((grouped.nunique() > 1).mean()),
        "rank_consistency_proxy": float(consistency.dropna().median())
        if consistency.notna().any()
        else None,
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
    return {
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
