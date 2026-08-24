from pathlib import Path

import pandas as pd

from bio_ml_preflight.data.bbb import load_bbb_martins


def test_bbb_adapter_normalizes_and_records_source(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        {
            "Drug_ID": ["m1", "m2", "m3"],
            "Drug": ["CC", "CO", "CN"],
            "Y": [1, 0, 1],
        }
    )
    frame, metadata = load_bbb_martins(tmp_path, loader=lambda: raw)
    assert list(frame.columns) == ["sample_id", "compound_id", "smiles", "bbb_label"]
    assert frame["bbb_label"].tolist() == [1, 0, 1]
    assert metadata["row_count"] == 3
    assert metadata["class_counts"] == {"0": 1, "1": 2}
    assert len(metadata["sha256"]) == 64
    cached, cached_metadata = load_bbb_martins(
        tmp_path, loader=lambda: (_ for _ in ()).throw(AssertionError())
    )
    pd.testing.assert_frame_equal(frame, cached)
    assert cached_metadata == metadata
