from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

from bio_ml_preflight.data.io import file_checksum

UCI_PARKINSONS_URL = "https://archive.ics.uci.edu/static/public/189/parkinsons%2Btelemonitoring.zip"
UCI_PARKINSONS_SHA256 = "2f82bb0ef96fa7d8d7edf4d97b89173e07c2cca7723440a64bc38918a83345df"
UCI_PARKINSONS_DOI = "10.24432/C5ZS3N"

SOURCE_COLUMNS: dict[str, str] = {
    "subject#": "subject_id",
    "age": "age",
    "sex": "sex",
    "test_time": "test_time_days",
    "motor_UPDRS": "motor_updrs",
    "total_UPDRS": "total_updrs",
    "Jitter(%)": "jitter_percent",
    "Jitter(Abs)": "jitter_abs",
    "Jitter:RAP": "jitter_rap",
    "Jitter:PPQ5": "jitter_ppq5",
    "Jitter:DDP": "jitter_ddp",
    "Shimmer": "shimmer",
    "Shimmer(dB)": "shimmer_db",
    "Shimmer:APQ3": "shimmer_apq3",
    "Shimmer:APQ5": "shimmer_apq5",
    "Shimmer:APQ11": "shimmer_apq11",
    "Shimmer:DDA": "shimmer_dda",
    "NHR": "nhr",
    "HNR": "hnr",
    "RPDE": "rpde",
    "DFA": "dfa",
    "PPE": "ppe",
}


def load_parkinsons_telemonitoring(
    cache_dir: Path,
    *,
    loader: Callable[[], pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    table_path = cache_dir / "parkinsons_telemonitoring.parquet"
    metadata_path = cache_dir / "source.json"
    raw_path = cache_dir / "parkinsons+telemonitoring.zip"
    if table_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        source_mode = metadata.get("source_mode")
        if source_mode not in {"pinned-official", "injected-test-loader"}:
            raise ValueError(f"unsupported cached Parkinsons source mode: {source_mode!r}")
        if loader is None and source_mode != "pinned-official":
            raise ValueError("cached Parkinsons table was produced by an injected test loader")
        if file_checksum(table_path) != metadata.get("sha256"):
            raise ValueError(f"cached Parkinsons table checksum mismatch: {table_path}")
        if source_mode == "pinned-official" and (
            metadata.get("raw_sha256") != UCI_PARKINSONS_SHA256
            or not raw_path.is_file()
            or file_checksum(raw_path) != UCI_PARKINSONS_SHA256
        ):
            raise ValueError(f"cached Parkinsons source checksum mismatch: {raw_path}")
        return pd.read_parquet(table_path), metadata

    if loader is None:
        if not raw_path.exists():
            with urlopen(UCI_PARKINSONS_URL, timeout=60) as response:  # noqa: S310
                raw_path.write_bytes(response.read())
        if file_checksum(raw_path) != UCI_PARKINSONS_SHA256:
            raise ValueError(f"pinned Parkinsons source checksum mismatch: {raw_path}")
        with ZipFile(raw_path) as archive:
            raw = pd.read_csv(archive.open("parkinsons_updrs.data"))
        source_mode = "pinned-official"
        raw_sha256: str | None = UCI_PARKINSONS_SHA256
    else:
        raw = loader().copy()
        source_mode = "injected-test-loader"
        raw_sha256 = None

    missing = sorted(set(SOURCE_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError(f"UCI Parkinsons loader returned an unexpected schema; missing {missing}")
    frame = raw.rename(columns=SOURCE_COLUMNS)[list(SOURCE_COLUMNS.values())].copy()
    if frame.isna().any().any():
        raise ValueError("UCI Parkinsons required fields contain missing values")
    frame["subject_id"] = (
        pd.to_numeric(frame["subject_id"], errors="raise").astype("Int64").astype(str)
    )
    numeric_columns = [column for column in SOURCE_COLUMNS.values() if column != "subject_id"]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("UCI Parkinsons required numeric fields must be finite")
    frame.insert(0, "sample_id", [f"parkinsons-{index}" for index in range(len(frame))])
    frame.to_parquet(table_path, index=False)

    records_per_participant = frame.groupby("subject_id").size()
    participant_time_sizes = frame.groupby(["subject_id", "test_time_days"]).size()
    metadata = {
        "source": "UCI Parkinsons Telemonitoring",
        "source_url": UCI_PARKINSONS_URL,
        "source_doi": UCI_PARKINSONS_DOI,
        "license": "CC BY 4.0",
        "source_mode": source_mode,
        "raw_sha256": raw_sha256,
        "retrieval_time": datetime.now(UTC).isoformat(),
        "row_count": len(frame),
        "participant_count": int(frame["subject_id"].nunique()),
        "records_per_participant": {
            "min": int(records_per_participant.min()),
            "median": float(records_per_participant.median()),
            "max": int(records_per_participant.max()),
        },
        "participant_time_proxy": {
            "definition": "exact normalized subject_id and test_time_days pair",
            "unique": int(len(participant_time_sizes)),
            "records_in_repeated_groups": int(
                participant_time_sizes[participant_time_sizes.gt(1)].sum()
            ),
            "scope": "transparent repeat proxy; not asserted to be a visit or replicate ID",
        },
        "target_semantics": (
            "Official clinician motor_UPDRS score, linearly interpolated to each voice "
            "recording time; renamed motor_updrs without numerical transformation"
        ),
        "scientific_scope": (
            "Retrospective six-month telemonitoring association. Participant-grouped "
            "validation tests unseen participants within this source; it does not establish "
            "future-time transport, clinical utility, measurement reliability, or causality."
        ),
        "sha256": file_checksum(table_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return frame, metadata
