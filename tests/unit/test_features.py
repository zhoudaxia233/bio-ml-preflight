import sys

import pandas as pd
import pytest

from bio_ml_preflight.cli import synthetic_case
from bio_ml_preflight.contracts.case import EntitySpec
from bio_ml_preflight.features import build_feature_frames, morgan_fingerprints


def _molecular_case(tmp_path):
    case = synthetic_case("no_signal", tmp_path / "unused.parquet")
    case.features.include = ["smiles"]
    case.features.smiles_column = "smiles"
    case.entities = {
        "compound": EntitySpec(id_column="compound_id", representation_column="smiles")
    }
    return case


def test_character_hash_representation_is_deterministic(tmp_path) -> None:
    frame = pd.DataFrame({"compound_id": ["a", "b"], "smiles": ["CCO", "CCN"]})
    case = _molecular_case(tmp_path)
    case.features.molecular_representations = ["character_hash"]

    first = build_feature_frames(frame, case)["character_hash"]
    second = build_feature_frames(frame, case)["character_hash"]
    pd.testing.assert_frame_equal(first, second)
    assert first.shape == (2, 32)


def test_morgan_representation_is_deterministic_when_rdkit_is_installed(tmp_path) -> None:
    pytest.importorskip("rdkit")
    frame = pd.DataFrame({"compound_id": ["a", "b"], "smiles": ["CCO", "CCN"]})
    case = _molecular_case(tmp_path)
    case.features.molecular_representations = ["morgan"]

    first = build_feature_frames(frame, case)["morgan"]
    second = build_feature_frames(frame, case)["morgan"]
    pd.testing.assert_frame_equal(first, second)
    assert first.shape == (2, 1024)


def test_morgan_representation_fails_clearly_without_rdkit(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "rdkit", None)
    with pytest.raises(RuntimeError, match="uv sync --extra chem"):
        morgan_fingerprints(["CCO"])
