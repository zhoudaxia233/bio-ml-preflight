from __future__ import annotations

import importlib.metadata
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from urllib.request import urlopen

import pandas as pd

from bio_ml_preflight.data.io import file_checksum

B3DB_COMMIT = "75dab1cc607bb7a03f3de5c576cffd223e767844"
B3DB_EXTERNAL_SHA256 = "84940d866f7cb53b361c38e787143c20b65641b57040006e020609f9d6f242cf"
B3DB_EXTERNAL_URL = (
    "https://raw.githubusercontent.com/theochem/B3DB/"
    f"{B3DB_COMMIT}/B3DB/B3DB_classification_external.tsv"
)


def load_bbb_martins(
    cache_dir: Path,
    *,
    loader: Callable[[], pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    table_path = cache_dir / "bbb_martins.parquet"
    metadata_path = cache_dir / "source.json"
    if table_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        _require_checksum(table_path, str(metadata.get("sha256", "")), "cached BBB_Martins table")
        return pd.read_parquet(table_path), metadata
    if loader is None:
        try:
            from tdc.single_pred import ADME
        except ImportError as error:
            raise RuntimeError(
                "The BBB_Martins demo requires the TDC extra. Run: "
                "uv sync --python 3.11 --extra tdc && "
                "uv run --python 3.11 bio-ml-preflight demo bbb --budget smoke"
            ) from error

        def loader() -> pd.DataFrame:
            raw_cache = cache_dir / "tdc_raw"
            raw_cache.mkdir(parents=True, exist_ok=True)
            return cast(
                pd.DataFrame,
                ADME(name="BBB_Martins", path=str(raw_cache)).get_data(),
            )

    raw = loader()
    frame = raw.rename(columns={"Drug_ID": "compound_id", "Drug": "smiles", "Y": "bbb_label"})
    required = ["compound_id", "smiles", "bbb_label"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"TDC BBB_Martins loader returned an unexpected schema; missing {missing}")
    frame = frame[required].copy()
    frame["compound_id"] = frame["compound_id"].astype(str)
    frame["bbb_label"] = pd.to_numeric(frame["bbb_label"], errors="raise").astype(int)
    labels = set(frame["bbb_label"].unique())
    if not labels <= {0, 1} or len(labels) != 2:
        raise ValueError(f"BBB_Martins requires binary labels 0/1; observed {sorted(labels)}")
    frame.insert(0, "sample_id", [f"bbb-{index}" for index in range(len(frame))])
    frame.to_parquet(table_path, index=False)
    try:
        loader_version = importlib.metadata.version("pytdc")
    except importlib.metadata.PackageNotFoundError:
        loader_version = "injected-test-loader"
    metadata = {
        "source": "Therapeutics Data Commons BBB_Martins ADME dataset",
        "loader_package": "pytdc",
        "loader_version": loader_version,
        "retrieval_time": datetime.now(UTC).isoformat(),
        "row_count": len(frame),
        "compound_count": int(frame["compound_id"].nunique()),
        "class_counts": {
            str(label): int(count)
            for label, count in frame["bbb_label"].value_counts().sort_index().items()
        },
        "target_transformation": "none; raw binary TDC Y retained as bbb_label",
        "sha256": file_checksum(table_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return frame, metadata


def load_b3db_external_confirmation(
    cache_dir: Path,
    *,
    development_cache_dir: Path | None = None,
    development_loader: Callable[[], pd.DataFrame] | None = None,
    external_loader: Callable[[], pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Combine BBB_Martins development rows with B3DB's pinned post-release set."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    table_path = cache_dir / "b3db_external_confirmation.parquet"
    metadata_path = cache_dir / "source.json"
    raw_path = cache_dir / "B3DB_classification_external.tsv"
    if table_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        _validate_b3db_cache(table_path, raw_path, metadata)
        return pd.read_parquet(table_path), metadata

    development_dir = development_cache_dir or cache_dir.parent / "bbb_martins"
    development, development_metadata = load_bbb_martins(development_dir, loader=development_loader)
    if external_loader is None:
        if not raw_path.exists():
            with urlopen(B3DB_EXTERNAL_URL, timeout=60) as response:  # noqa: S310
                raw_path.write_bytes(response.read())
        _require_checksum(raw_path, B3DB_EXTERNAL_SHA256, "pinned B3DB external source")
        source_mode = "pinned-official"
        external = pd.read_csv(raw_path, sep="\t")
    else:
        external = external_loader().copy()
        external.to_csv(raw_path, sep="\t", index=False)
        source_mode = "injected-test-loader"

    required = {"SMILES", "CID", "BBB+/BBB-", "reference", "group"}
    missing = sorted(required - set(external.columns))
    if missing:
        raise ValueError(f"B3DB external loader returned an unexpected schema; missing {missing}")
    mapped_labels = external["BBB+/BBB-"].map({"BBB+": 1, "BBB-": 0})
    if mapped_labels.isna().any():
        unexpected = sorted(external.loc[mapped_labels.isna(), "BBB+/BBB-"].astype(str).unique())
        raise ValueError(f"B3DB external labels must be BBB+ or BBB-; observed {unexpected}")

    development_keys = _inchi_keys(development["smiles"])
    external_keys = _inchi_keys(external["SMILES"])
    overlap_keys = sorted(set(development_keys) & set(external_keys))
    development_overlap = development_keys.isin(overlap_keys)
    development_kept = development.loc[~development_overlap].copy()
    development_keys = development_keys.loc[~development_overlap]

    development_table = pd.DataFrame(
        {
            "sample_id": "development-" + development_kept["sample_id"].astype(str),
            "compound_id": development_keys.to_numpy(),
            "source_compound_id": development_kept["compound_id"].astype(str),
            "smiles": development_kept["smiles"].astype(str),
            "bbb_label": development_kept["bbb_label"].astype(int),
            "validation_split": "train",
            "source_dataset": "TDC BBB_Martins",
            "source_reference": None,
            "curation_group": None,
        }
    )
    external_table = pd.DataFrame(
        {
            "sample_id": "b3db-external-" + external["CID"].astype(str),
            "compound_id": external_keys.to_numpy(),
            "source_compound_id": external["CID"].astype(str),
            "smiles": external["SMILES"].astype(str),
            "bbb_label": mapped_labels.astype(int),
            "validation_split": "holdout",
            "source_dataset": "B3DB classification external",
            "source_reference": external["reference"],
            "curation_group": external["group"],
        }
    )
    frame = pd.concat([development_table, external_table], ignore_index=True)
    frame.to_parquet(table_path, index=False)
    metadata = {
        "source": "TDC BBB_Martins development plus B3DB post-release external set",
        "b3db_commit": B3DB_COMMIT,
        "b3db_external_url": B3DB_EXTERNAL_URL,
        "b3db_external_raw_sha256": file_checksum(raw_path),
        "source_mode": source_mode,
        "retrieval_time": datetime.now(UTC).isoformat(),
        "development_source_sha256": development_metadata["sha256"],
        "development_source_rows": len(development),
        "development_identity_overlap_rows_excluded": int(development_overlap.sum()),
        "development_identity_overlap_compounds": len(overlap_keys),
        "development_rows_before_identity_policy": len(development_table),
        "external_rows": len(external_table),
        "external_compounds": int(external_table["compound_id"].nunique()),
        "external_class_counts": {
            str(label): int(count)
            for label, count in external_table["bbb_label"].value_counts().sort_index().items()
        },
        "identity_method": "RDKit InChIKey computed from source SMILES",
        "split_policy": (
            "B3DB identities are retained in holdout; matching BBB_Martins rows are excluded "
            "from development before modeling"
        ),
        "scientific_scope": (
            "Public pseudo-sealed repository-release validation; not blinded and not an "
            "assay-time prospective study. reference and group are preserved source fields, "
            "not inferred batches or replicates."
        ),
        "row_count": len(frame),
        "sha256": file_checksum(table_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return frame, metadata


def _inchi_keys(values: pd.Series) -> pd.Series:
    try:
        from rdkit import Chem, rdBase
    except ImportError as error:
        raise RuntimeError("B3DB identity normalization requires: uv sync --extra chem") from error
    keys: list[str] = []
    invalid: list[int] = []
    inchi_key: Callable[[Any], str] = cast(Any, Chem.MolToInchiKey)
    with rdBase.BlockLogs():
        for position, value in enumerate(values):
            molecule = Chem.MolFromSmiles(str(value))
            key = inchi_key(molecule) if molecule is not None else ""
            if not key:
                invalid.append(position)
            keys.append(key)
    if invalid:
        raise ValueError(f"Cannot derive InChIKey for SMILES rows {invalid[:10]}")
    return pd.Series(keys, index=values.index, dtype="string")


def _validate_b3db_cache(table_path: Path, raw_path: Path, metadata: dict[str, Any]) -> None:
    if metadata.get("b3db_commit") != B3DB_COMMIT:
        raise ValueError("Cached B3DB commit does not match the pinned source")
    _require_checksum(table_path, str(metadata.get("sha256", "")), "combined B3DB table")
    expected_raw = (
        B3DB_EXTERNAL_SHA256
        if metadata.get("source_mode") in {None, "pinned-official"}
        else str(metadata.get("b3db_external_raw_sha256", ""))
    )
    _require_checksum(raw_path, expected_raw, "cached B3DB external source")


def _require_checksum(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or not expected or file_checksum(path) != expected:
        raise ValueError(f"{label} checksum mismatch: {path}")
