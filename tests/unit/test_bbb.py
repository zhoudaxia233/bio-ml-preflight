from pathlib import Path

import pandas as pd
import pytest

from bio_ml_preflight.data.bbb import (
    B3DB_COMMIT,
    load_b3db_external_confirmation,
    load_bbb_martins,
)


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
    pd.testing.assert_frame_equal(frame.convert_dtypes(), cached.convert_dtypes())
    assert cached_metadata == metadata

    (tmp_path / "bbb_martins.parquet").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_bbb_martins(tmp_path)


def test_b3db_external_adapter_excludes_development_identity_overlap(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    development = pd.DataFrame(
        {
            "Drug_ID": ["d1", "d2", "d3", "d4"],
            "Drug": ["CC", "CO", "CN", "CCC"],
            "Y": [1, 0, 1, 0],
        }
    )
    external = pd.DataFrame(
        {
            "SMILES": ["CC", "CCO", "CCN"],
            "CID": [1, 2, 3],
            "BBB+/BBB-": ["BBB-", "BBB+", "BBB-"],
            "reference": ["r1", "r2", "r3"],
            "group": ["E", "E", "E"],
        }
    )

    frame, metadata = load_b3db_external_confirmation(
        tmp_path / "external",
        development_cache_dir=tmp_path / "development",
        development_loader=lambda: development,
        external_loader=lambda: external,
    )

    train = frame[frame["validation_split"].eq("train")]
    holdout = frame[frame["validation_split"].eq("holdout")]
    assert len(train) == 3
    assert len(holdout) == 3
    assert set(train["compound_id"]).isdisjoint(holdout["compound_id"])
    assert holdout["bbb_label"].tolist() == [0, 1, 0]
    assert metadata["b3db_commit"] == B3DB_COMMIT
    assert metadata["development_identity_overlap_compounds"] == 1
    assert metadata["external_class_counts"] == {"0": 2, "1": 1}

    cached, cached_metadata = load_b3db_external_confirmation(tmp_path / "external")
    pd.testing.assert_frame_equal(frame.convert_dtypes(), cached.convert_dtypes())
    assert cached_metadata == metadata

    (tmp_path / "external" / "B3DB_classification_external.tsv").write_text(
        "tampered", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_b3db_external_confirmation(tmp_path / "external")
