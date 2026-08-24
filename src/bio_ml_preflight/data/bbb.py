from __future__ import annotations

import importlib.metadata
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from bio_ml_preflight.data.io import file_checksum


def load_bbb_martins(
    cache_dir: Path,
    *,
    loader: Callable[[], pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    table_path = cache_dir / "bbb_martins.parquet"
    metadata_path = cache_dir / "source.json"
    if table_path.exists() and metadata_path.exists():
        return pd.read_parquet(table_path), json.loads(metadata_path.read_text(encoding="utf-8"))
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
