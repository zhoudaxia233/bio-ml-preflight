from pathlib import Path

import pandas as pd

from bio_ml_preflight.data.davis import load_davis


def test_davis_adapter_normalizes_and_records_source(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        {
            "Drug_ID": ["d1", "d2"],
            "Drug": ["CC", "CO"],
            "Target_ID": ["t1", "t2"],
            "Target": ["ACD", "AAA"],
            "Y": [10.0, 20.0],
        }
    )
    frame, metadata = load_davis(tmp_path, loader=lambda: raw)
    assert list(frame.columns) == [
        "sample_id",
        "compound_id",
        "smiles",
        "target_id",
        "sequence",
        "affinity",
    ]
    assert metadata["row_count"] == 2
    assert len(metadata["sha256"]) == 64
    cached, cached_metadata = load_davis(
        tmp_path, loader=lambda: (_ for _ in ()).throw(AssertionError())
    )
    pd.testing.assert_frame_equal(frame, cached)
    assert cached_metadata == metadata
