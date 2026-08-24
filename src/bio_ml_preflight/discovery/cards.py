from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from bio_ml_preflight.data import read_table


def infer_roles(frame: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    numeric = [str(column) for column in frame.select_dtypes(include="number").columns]
    categorical = [str(column) for column in frame.select_dtypes(exclude="number").columns]
    identifiers = [
        str(column)
        for column in frame.columns
        if "id" in str(column).lower()
        or (
            column in frame.select_dtypes(exclude="number").columns
            and frame[column].nunique(dropna=True) == len(frame)
        )
    ]
    return {
        "candidate_numeric_targets": [
            {"column": column, "confirmed": False, "basis": "physical numeric type"}
            for column in numeric
        ],
        "candidate_categorical_targets": [
            {"column": column, "confirmed": False, "basis": "physical categorical type"}
            for column in categorical
            if 2 <= frame[column].nunique(dropna=True) <= 20
        ],
        "candidate_identifiers": [
            {"column": column, "confirmed": False, "basis": "name/uniqueness heuristic"}
            for column in identifiers
        ],
    }


def discover_tasks(data_path: Path, output: Path) -> dict[str, Any]:
    frame = read_table(data_path)
    roles = infer_roles(frame)
    cards = []
    for target in roles["candidate_numeric_targets"][:5]:
        cards.append(
            _card(
                "supervised_regression",
                f"Can declared pre-outcome features predict {target['column']}?",
                target["column"],
                "sample (provisional)",
                "independent sample/entity identifier",
                "group or time split after deployment is declared",
                "MAE/RMSE and rank metric only if a ranking decision is confirmed",
                ["target semantics", "feature timing", "independent unit", "deployment population"],
            )
        )
    for target in roles["candidate_categorical_targets"][:5]:
        cards.append(
            _card(
                "supervised_classification",
                f"Can declared pre-outcome features classify {target['column']}?",
                target["column"],
                "sample (provisional)",
                "independent sample/entity identifier",
                "group split after repeated entities are confirmed",
                "balanced accuracy and average precision",
                ["label meaning", "class definition", "feature timing", "independent unit"],
            )
        )
    identifiers = [item["column"] for item in roles["candidate_identifiers"]]
    if len(identifiers) >= 2 and roles["candidate_numeric_targets"]:
        target = roles["candidate_numeric_targets"][0]["column"]
        cards.extend(
            [
                _card(
                    "pairwise_prediction",
                    f"Can declared pair representations predict {target}?",
                    target,
                    f"{identifiers[0]}–{identifiers[1]} pair (provisional)",
                    "left and right entities",
                    "random-pair, cold-left, cold-right, and double-cold",
                    "regression plus per-group ranking metrics",
                    ["two entity roles", "representations", "pair meaning", "deployment axis"],
                ),
                _card(
                    "ranking_within_groups",
                    f"Can candidates be ranked by {target} within a declared group?",
                    target,
                    "candidate within group",
                    "group entity",
                    "group-respecting split",
                    "top-k overlap, selection probability, rank variability",
                    ["group role", "candidate role", "k", "higher-is-better semantics"],
                ),
            ]
        )
    lower_names = {str(column).lower(): str(column) for column in frame.columns}
    if any("treatment" in name for name in lower_names) and any(
        "control" in name for name in lower_names
    ):
        cards.append(
            _card(
                "treatment_control_association",
                "Is an outcome associated with declared treatment versus control groups?",
                None,
                "biological replicate (must be confirmed)",
                "biological replicate",
                "batch-aware grouped comparison",
                "effect estimate with independent-unit interval",
                [
                    "outcome",
                    "treatment",
                    "control value",
                    "batch",
                    "replicate",
                    "assignment process",
                ],
                reason=(
                    "Observational association is not a causal effect without an "
                    "identifying design."
                ),
            )
        )
    dose_columns = [str(column) for column in frame.columns if "dose" in str(column).lower()]
    time_columns = [
        str(column)
        for column in frame.columns
        if "time" in str(column).lower() or "date" in str(column).lower()
    ]
    if dose_columns or time_columns:
        cards.append(
            _card(
                "dose_or_time_response",
                "Is a declared outcome associated with dose or time under an explicit design?",
                None,
                "biological replicate at dose/time",
                "biological replicate",
                "leave-replicate/batch-out and forward-time validation",
                "dose/time trend with independent-unit uncertainty",
                ["outcome", "dose units", "time origin", "replicates", "batch", "controls"],
            )
        )
    if time_columns and identifiers:
        cards.append(
            _card(
                "longitudinal_prediction",
                "Can earlier declared measurements predict a later declared outcome?",
                None,
                "entity-time observation",
                identifiers[0],
                "forward-time split with entity grouping",
                "task metric at predeclared horizon",
                ["entity", "timestamp", "prediction horizon", "feature availability time"],
            )
        )
    cards.extend(
        [
            _card(
                "batch_confounding_audit",
                "Do declared batches align with labels, treatments, or feature structure?",
                None,
                "independent sample/entity",
                "biological sample",
                "leave-batch-out diagnostic",
                "label/batch predictability and coverage",
                ["batch identifier", "biological replicate identifier"],
            ),
            _card(
                "unsupervised_structure_exploration",
                "What reproducible structure is visible without assigning biological meaning?",
                None,
                "sample",
                "independent sample/entity",
                "resampling stability, not a predictive split",
                "cluster/embedding stability",
                ["feature semantics", "batch", "replicate"],
            ),
        ]
    )
    payload = {
        "mode": "task_discovery",
        "warning": (
            "Candidates are testable analysis templates, not biological mechanisms "
            "or causal claims."
        ),
        "roles": roles,
        "claim_cards": cards,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "discovery.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    for index, card in enumerate(cards, start=1):
        (output / f"claim-card-{index:02d}.yaml").write_text(
            yaml.safe_dump(card, sort_keys=False), encoding="utf-8"
        )
    return payload


def _card(
    family: str,
    question: str,
    target: str | None,
    unit: str,
    independent_unit: str,
    split: str,
    metric: str,
    required_metadata: list[str],
    *,
    reason: str = "Roles and scientific semantics are unconfirmed.",
) -> dict[str, Any]:
    return {
        "family": family,
        "candidate_question": question,
        "prediction_or_analysis_unit": unit,
        "candidate_target": target,
        "required_metadata": required_metadata,
        "independent_unit": independent_unit,
        "realistic_split": split,
        "decision_metric": metric,
        "validation_requirement": "Researcher confirmation followed by deterministic validation",
        "current_status": "NOT_ASSESSABLE",
        "reasons_it_may_be_invalid": [reason],
        "role_confirmation": {"confirmed": False},
    }
