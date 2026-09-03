from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from bio_ml_preflight.contracts import CaseSpec


def hashed_character_features(
    values: Iterable[object], *, prefix: str, n_features: int = 32, ngram: int = 3
) -> pd.DataFrame:
    rows: list[npt.NDArray[np.float64]] = []
    for raw in values:
        text = "" if raw is None or raw is pd.NA else str(raw)
        vector: npt.NDArray[np.float64] = np.zeros(n_features, dtype=float)
        grams = [text[i : i + ngram] for i in range(max(1, len(text) - ngram + 1))]
        for gram in grams:
            digest = hashlib.blake2b(gram.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest, "little") % n_features
            vector[bucket] += 1.0
        norm = np.linalg.norm(vector)
        rows.append(vector / norm if norm else vector)
    return pd.DataFrame(rows, columns=[f"{prefix}_char_{i}" for i in range(n_features)])


def amino_acid_composition(values: Iterable[object], *, prefix: str) -> pd.DataFrame:
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    rows: list[list[float]] = []
    for raw in values:
        text = str(raw).upper()
        length = max(len(text), 1)
        rows.append([text.count(amino_acid) / length for amino_acid in alphabet])
    return pd.DataFrame(rows, columns=[f"{prefix}_aa_{value}" for value in alphabet])


def build_feature_frames(frame: pd.DataFrame, case: CaseSpec) -> dict[str, pd.DataFrame]:
    representations: list[str] = []
    representations.extend(case.features.molecular_representations)
    if not representations:
        representations.append("declared_features")
    return {
        representation: build_feature_frame(
            frame,
            case,
            molecular_representation=(
                None if representation == "declared_features" else representation
            ),
        )
        for representation in representations
    }


def model_feature_columns(columns: Iterable[object], case: CaseSpec) -> list[str]:
    """Resolve the exact columns that feature construction will model."""
    available = [str(column) for column in columns]
    if case.features.include:
        return [column for column in case.features.include if column not in case.features.exclude]

    reserved = {
        case.task.target_column,
        *case.features.exclude,
        *case.features.post_outcome,
    }
    reserved.update(entity.id_column for entity in case.entities.values())
    if case.evaluation.bootstrap_unit:
        reserved.add(case.evaluation.bootstrap_unit)
    if case.decision.group_entity:
        reserved.add(
            case.entities[case.decision.group_entity].id_column
            if case.decision.group_entity in case.entities
            else case.decision.group_entity
        )
    reserved.update(
        column
        for column in [
            case.metadata.replicate_id,
            case.metadata.biological_replicate_id,
            case.metadata.batch_id,
            case.metadata.plate_id,
            case.metadata.time_column,
            case.metadata.treatment_column,
        ]
        if column
    )
    for scenario in case.generalization_scenarios:
        reserved.update(
            column
            for column in [
                scenario.group_column,
                scenario.left_column,
                scenario.right_column,
                scenario.split_column,
            ]
            if column
        )
    return [column for column in available if column not in reserved]


def build_feature_frame(
    frame: pd.DataFrame,
    case: CaseSpec,
    *,
    molecular_representation: str | None = None,
) -> pd.DataFrame:
    columns = model_feature_columns(frame.columns, case)
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Configured feature columns are missing: {missing}")
    features = frame.loc[:, columns].reset_index(drop=True).copy()
    representations = {
        entity.representation_column
        for entity in case.entities.values()
        if entity.representation_column in features.columns
    }
    for column in sorted(value for value in representations if value is not None):
        if column == case.features.smiles_column and molecular_representation == "morgan":
            values = morgan_fingerprints(features[column])
            encoded = pd.DataFrame(
                values,
                columns=[f"{column}_morgan_{index}" for index in range(values.shape[1])],
            )
        elif "seq" in column.lower() or "protein" in column.lower() or "target" in column.lower():
            encoded = amino_acid_composition(features[column], prefix=column)
        else:
            encoded = hashed_character_features(features[column], prefix=column)
        features = pd.concat([features.drop(columns=[column]), encoded], axis=1)
    return features


def morgan_fingerprints(
    values: Iterable[object], *, radius: int = 2, n_bits: int = 1024
) -> npt.NDArray[np.uint8]:
    try:
        from rdkit import Chem, rdBase
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError as error:
        raise RuntimeError("Morgan fingerprints require: uv sync --extra chem") from error
    rows: list[npt.NDArray[np.uint8]] = []
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    with rdBase.BlockLogs():
        for value in values:
            molecule = Chem.MolFromSmiles(str(value))
            if molecule is None:
                rows.append(np.zeros(n_bits, dtype=np.uint8))
                continue
            fingerprint = generator.GetFingerprint(molecule)
            rows.append(np.asarray(fingerprint, dtype=np.uint8))
    return cast(npt.NDArray[np.uint8], np.vstack(rows))


def scaffold_groups(values: Iterable[object]) -> list[str]:
    try:
        from rdkit import Chem, rdBase
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError as error:
        raise RuntimeError("Scaffold splitting requires: uv sync --extra chem") from error
    murcko_smiles: Callable[..., str] = cast(Any, MurckoScaffold.MurckoScaffoldSmiles)
    groups = []
    with rdBase.BlockLogs():
        for value in values:
            molecule = Chem.MolFromSmiles(str(value))
            groups.append(murcko_smiles(mol=molecule) if molecule else "INVALID")
    return groups
