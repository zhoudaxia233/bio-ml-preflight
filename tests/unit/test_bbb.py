from pathlib import Path

import pandas as pd
import pytest

from bio_ml_preflight.data.bbb import (
    B3DB_COMMIT,
    PETBD_COMMIT,
    load_b3db_external_confirmation,
    load_bbb_martins,
    load_petbd_external_confirmation,
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


def test_petbd_adapter_recomputes_log10_and_excludes_protected_identities(
    tmp_path: Path,
) -> None:
    chem = pytest.importorskip("rdkit.Chem")

    def key(smiles: str) -> str:
        return str(chem.MolToInchiKey(chem.MolFromSmiles(smiles)))

    protected = pd.DataFrame(
        {
            "sample_id": ["train", "old-holdout"],
            "compound_id": [key("CCC"), key("CC")],
            "source_compound_id": ["train", "old-holdout"],
            "smiles": ["CCC", "CC"],
            "bbb_label": [1, 0],
            "validation_split": ["train", "holdout"],
            "source_dataset": ["development", "old confirmation"],
            "source_reference": [None, None],
            "curation_group": [None, None],
        }
    )
    external = pd.DataFrame(
        {
            "compound index": ["overlap", "negative-a", "negative-b", "positive", "missing"],
            "SMILES": ["CC", "CO", "CO", "CN", "CCO"],
            "PMID": [1, 2, 3, 4, 5],
            "animal type": ["rat"] * 5,
            "gender": ["male"] * 5,
            "animal weight (g)": [100] * 5,
            "injection dosage (μCi)": [10] * 5,
            "logBB at60min": [0.0, -4.61, -2.30, 0.0, None],
            "brain at60min": [1.0, 0.01, 0.1, 1.0, None],
            "blood at60min": [1.0, 1.0, 1.0, 1.0, None],
            "ref No": ["r1", "r2", "r3", "r4", "r5"],
            "DOI": ["d1", "d2", "d3", "d4", "d5"],
        }
    )

    frame, metadata = load_petbd_external_confirmation(
        tmp_path,
        protected_loader=lambda: (protected, {"sha256": "p" * 64}),
        external_loader=lambda: external,
    )

    train = frame[frame["validation_split"].eq("train")]
    holdout = frame[frame["validation_split"].eq("holdout")]
    assert train["sample_id"].tolist() == ["train"]
    assert set(holdout["compound_id"]) == {key("CO"), key("CN")}
    by_identity = holdout.set_index("compound_id")
    assert by_identity.loc[key("CO"), "log10_bb"] == pytest.approx(-1.5)
    assert by_identity.loc[key("CO"), "bbb_label"] == 0
    assert by_identity.loc[key("CO"), "measurement_count"] == 2
    assert by_identity.loc[key("CN"), "bbb_label"] == 1
    assert metadata["petbd_commit"] == PETBD_COMMIT
    assert metadata["ratio_measurement_rows"] == 4
    assert metadata["rows_excluded_without_positive_brain_and_blood"] == 1
    assert metadata["external_identity_overlap_compounds_excluded"] == 1
    assert metadata["external_class_counts"] == {"0": 1, "1": 1}
    assert metadata["external_negative_margin_counts"]["below_-1.25"] == 1

    cached, cached_metadata = load_petbd_external_confirmation(tmp_path)
    pd.testing.assert_frame_equal(frame.convert_dtypes(), cached.convert_dtypes())
    assert cached_metadata == metadata

    (tmp_path / "PTBD_v20240912.csv").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_petbd_external_confirmation(tmp_path)
