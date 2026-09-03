import hashlib
import runpy
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from bio_ml_preflight.contracts import load_case
from bio_ml_preflight.features import model_feature_columns


def test_esol_preparation_preserves_rows_and_blocks_corrupt_cache(tmp_path) -> None:
    pytest.importorskip("rdkit")
    root = Path(__file__).resolve().parents[2]
    prepare = runpy.run_path(str(root / "examples/esol/prepare.py"))["prepare"]
    target = "measured log solubility in mols per litre"
    frame = pd.DataFrame(
        {
            "Compound ID": [str(i) for i in range(1128)],
            "smiles": ["CCO ", "OCC"] * 564,
            target: [-1.0, -2.0] * 564,
            "ESOL predicted log solubility in mols per litre": [-1.5] * 1128,
        }
    )
    raw = tmp_path / "delaney-processed.csv"
    frame.to_csv(raw, index=False)
    checksum = hashlib.sha256(raw.read_bytes()).hexdigest()
    with patch.dict(prepare.__globals__, SHA256=checksum):
        with patch("urllib.request.urlopen", side_effect=AssertionError("Cache must be reused")):
            prepared = pd.read_csv(prepare(tmp_path))
        assert len(prepared) == len(frame)
        assert prepared[target].equals(frame[target])
        assert prepared["source_smiles"].equals(frame["smiles"])
        assert prepared["compound_id"].nunique() == 1
        case = load_case(root / "examples/esol/case.yaml")
        assert model_feature_columns(prepared.columns, case) == ["smiles"]
        raw.write_text("corrupt", encoding="utf-8")
        with pytest.raises(ValueError, match="checksum mismatch"):
            prepare(tmp_path)
