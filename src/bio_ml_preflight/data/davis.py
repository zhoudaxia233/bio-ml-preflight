from __future__ import annotations

import importlib.metadata
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from bio_ml_preflight.data.io import file_checksum


def load_davis(
    cache_dir: Path,
    *,
    loader: Callable[[], pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    table_path = cache_dir / "davis.parquet"
    metadata_path = cache_dir / "source.json"
    if table_path.exists() and metadata_path.exists():
        return pd.read_parquet(table_path), json.loads(metadata_path.read_text(encoding="utf-8"))
    if loader is None:
        try:
            from tdc.multi_pred import DTI
        except ImportError as error:
            raise RuntimeError(
                "The Davis demo requires its optional compatible loader environment. Run: "
                "uv sync --python 3.11 --extra tdc && "
                "uv run --python 3.11 bio-ml-preflight demo davis --budget smoke"
            ) from error

        def loader() -> pd.DataFrame:
            raw_cache = cache_dir / "tdc_raw"
            raw_cache.mkdir(parents=True, exist_ok=True)
            return cast(pd.DataFrame, DTI(name="DAVIS", path=str(raw_cache)).get_data())

    raw = loader()
    aliases = {
        "Drug_ID": "compound_id",
        "Drug": "smiles",
        "Target_ID": "target_id",
        "Target": "sequence",
        "Y": "affinity",
    }
    frame = raw.rename(columns=aliases)
    required = list(aliases.values())
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"TDC Davis loader returned an unexpected schema; missing {missing}")
    frame = frame[required].copy()
    frame.insert(0, "sample_id", [f"davis-{index}" for index in range(len(frame))])
    frame.to_parquet(table_path, index=False)
    try:
        loader_version = importlib.metadata.version("pytdc")
    except importlib.metadata.PackageNotFoundError:
        loader_version = "injected-test-loader"
    metadata = {
        "source": "Therapeutics Data Commons DAVIS DTI dataset",
        "loader_package": "pytdc",
        "loader_version": loader_version,
        "retrieval_time": datetime.now(UTC).isoformat(),
        "row_count": len(frame),
        "compound_count": int(frame["compound_id"].nunique()),
        "target_count": int(frame["target_id"].nunique()),
        "target_transformation": "none; raw TDC Y retained as affinity",
        "sha256": file_checksum(table_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return frame, metadata
