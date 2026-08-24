from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from bio_ml_preflight.contracts.case import ScenarioSpec


@dataclass(frozen=True)
class SplitManifest:
    scenario: str
    strategy: str
    seed: int
    train_indices: list[int]
    test_indices: list[int]
    excluded_indices: list[int]

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self) | {"sha256": self.fingerprint()}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def create_split(
    frame: pd.DataFrame,
    scenario: ScenarioSpec,
    seed: int,
    *,
    test_size: float = 0.25,
) -> SplitManifest:
    positions = np.arange(len(frame))
    strategy = scenario.strategy
    excluded: npt.NDArray[np.int64] = np.array([], dtype=int)
    if strategy in {"random", "random_pair"}:
        train, test = train_test_split(positions, test_size=test_size, random_state=seed)
    elif strategy in {"group", "cold_left", "cold_right"}:
        column = {
            "group": scenario.group_column,
            "cold_left": scenario.left_column,
            "cold_right": scenario.right_column,
        }[strategy]
        if column is None or column not in frame:
            raise ValueError(f"{strategy} split column {column!r} is missing")
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train, test = next(splitter.split(positions, groups=frame[column]))
    elif strategy == "double_cold":
        if scenario.left_column is None or scenario.right_column is None:
            raise ValueError("double_cold needs left_column and right_column")
        rng = np.random.default_rng(seed)
        left_values = frame[scenario.left_column].dropna().unique()
        right_values = frame[scenario.right_column].dropna().unique()
        test_left = set(
            rng.choice(left_values, max(1, round(test_size * len(left_values))), replace=False)
        )
        test_right = set(
            rng.choice(right_values, max(1, round(test_size * len(right_values))), replace=False)
        )
        left_test = frame[scenario.left_column].isin(test_left).to_numpy()
        right_test = frame[scenario.right_column].isin(test_right).to_numpy()
        test_mask = left_test & right_test
        train_mask = ~left_test & ~right_test
        excluded_mask = ~(test_mask | train_mask)
        train, test, excluded = (
            positions[train_mask],
            positions[test_mask],
            positions[excluded_mask],
        )
        if not len(train) or not len(test):
            raise ValueError("double_cold split is infeasible for this pair coverage")
    elif strategy == "time":
        column = scenario.group_column
        if column is None or column not in frame:
            raise ValueError("time strategy uses group_column as its timestamp column")
        order = np.argsort(pd.to_datetime(frame[column]).to_numpy())
        cut = round((1 - test_size) * len(order))
        train, test = order[:cut], order[cut:]
    elif strategy == "supplied":
        column = scenario.split_column
        if column is None or column not in frame:
            raise ValueError("supplied split column is missing")
        values = frame[column].astype(str).str.lower()
        train, test = positions[values.eq("train")], positions[values.isin(["test", "holdout"])]
        excluded = positions[~values.isin(["train", "test", "holdout"])]
        if not len(train) or not len(test):
            raise ValueError("supplied split requires train and test/holdout labels")
    elif strategy == "scaffold":
        column = scenario.group_column
        if column is None or column not in frame:
            raise ValueError("scaffold strategy uses group_column as its SMILES column")
        from bio_ml_preflight.features.lightweight import scaffold_groups

        groups = scaffold_groups(frame[column])
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train, test = next(splitter.split(positions, groups=groups))
    else:
        raise ValueError(f"Unsupported split strategy {strategy}")
    return SplitManifest(
        scenario=scenario.name,
        strategy=strategy,
        seed=seed,
        train_indices=sorted(int(value) for value in train),
        test_indices=sorted(int(value) for value in test),
        excluded_indices=sorted(int(value) for value in excluded),
    )


def load_manifest(path: Path) -> SplitManifest:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop("sha256")
    manifest = SplitManifest(**payload)
    if manifest.fingerprint() != expected:
        raise ValueError(f"Split manifest checksum mismatch: {path}")
    return manifest
