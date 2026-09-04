import sys

import pandas as pd
import pytest

from bio_ml_preflight.cli import synthetic_case
from bio_ml_preflight.contracts.case import EntitySpec
from bio_ml_preflight.features import build_feature_frames, morgan_fingerprints


@pytest.mark.parametrize("representation,width", [("character_hash", 32), ("morgan", 1024)])
def test_molecular_representation_is_deterministic(tmp_path, representation, width) -> None:
    if representation == "morgan":
        pytest.importorskip("rdkit")
    frame = pd.DataFrame({"compound_id": ["a", "b"], "smiles": ["CCO", "CCN"]})
    case = synthetic_case("no_signal", tmp_path / "unused.parquet")
    case.features.include = ["smiles"]
    case.features.smiles_column = "smiles"
    case.features.molecular_representations = [representation]
    case.entities = {
        "compound": EntitySpec(id_column="compound_id", representation_column="smiles")
    }

    first = build_feature_frames(frame, case)[representation]
    second = build_feature_frames(frame, case)[representation]
    pd.testing.assert_frame_equal(first, second)
    assert first.shape == (2, width)


def test_morgan_representation_fails_clearly_without_rdkit(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "rdkit", None)
    with pytest.raises(RuntimeError, match="uv sync --extra chem"):
        morgan_fingerprints(["CCO"])
