from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported table format {suffix!r}; use CSV, TSV, or Parquet")


def dataset_fingerprint(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    selected = frame if not columns else frame.loc[:, columns]
    row_hashes = pd.util.hash_pandas_object(selected, index=True).to_numpy().tobytes()
    schema = "|".join(f"{column}:{selected[column].dtype}" for column in selected)
    return hashlib.sha256(schema.encode() + row_hashes).hexdigest()


def file_checksum(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()
