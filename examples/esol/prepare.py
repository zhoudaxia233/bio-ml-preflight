"""Prepare the pinned public ESOL table for the ordinary tabular workflow."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from bio_ml_preflight.data.io import file_checksum

URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
SHA256 = "8c06a76f0c6487d29ab0f903e6a7a7139f189ab3c1178f159c8be8964602f189"
TARGET = "measured log solubility in mols per litre"


def prepare(cache: Path) -> Path:
    from rdkit import Chem, rdBase

    cache.mkdir(parents=True, exist_ok=True)
    raw = cache / "delaney-processed.csv"
    if not raw.exists():
        with urllib.request.urlopen(URL, timeout=60) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != SHA256:
            raise ValueError("ESOL download checksum mismatch")
        raw.write_bytes(payload)
    if file_checksum(raw) != SHA256:
        raise ValueError("ESOL cache checksum mismatch")
    frame = pd.read_csv(raw)
    required = ["Compound ID", "smiles", TARGET]
    if len(frame) != 1128 or frame[required].isna().any().any():
        raise ValueError("ESOL source shape or required values changed")
    if not np.isfinite(pd.to_numeric(frame[TARGET], errors="raise")).all():
        raise ValueError("ESOL target must be finite")
    with rdBase.BlockLogs():
        molecules = frame["smiles"].map(Chem.MolFromSmiles)
    if molecules.isna().any():
        raise ValueError("ESOL contains invalid structures; no automatic repair is allowed")
    frame["source_smiles"] = frame["smiles"]
    frame["smiles"] = molecules.map(Chem.MolToSmiles)
    # Supplied structure identity is not proof of complete chemical/stereochemical identity.
    frame["compound_id"] = frame["smiles"]
    output = cache / "esol.csv"
    frame.to_csv(output, index=False)
    grouped = frame.groupby("compound_id")[TARGET]
    source = {
        "source": "MoleculeNet / DeepChem Delaney (ESOL)",
        "source_url": URL,
        "source_doi": "10.1021/ci034243x",
        "benchmark_doi": "10.1039/C7SC02664A",
        "raw_sha256": SHA256,
        "analysis_sha256": file_checksum(output),
        "rdkit_version": rdBase.rdkitVersion,
        "target_semantics": "Measured aqueous log10 solubility in mol/L; predict from SMILES only.",
        "scope": "Retrospective public benchmark workflow check; not blinded confirmation.",
        "source_rows": len(frame),
        "unique_supplied_structures": frame["compound_id"].nunique(),
        "repeated_structure_groups": int((grouped.size() > 1).sum()),
        "conflicting_target_structure_groups": int((grouped.nunique() > 1).sum()),
        "normalization": [
            "Keep every source row, target and auxiliary value; retain original source_smiles.",
            "Canonicalize supplied SMILES and group by that representation, without inferring "
            "missing stereochemistry, assay conditions or replicate identities.",
            "Keep conflicting labels explicitly; no averaging, filtering or target rescaling.",
            "Exclude source-model predictions and all supplied descriptors from model inputs.",
        ],
        "missing_metadata": ["batch", "replicate protocol", "measurement conditions"],
        "data_terms": "Public download supplied by the official benchmark loader. A standalone "
        "CSV redistribution license was not located; retain data locally and do not apply the "
        "DeepChem software license to the data.",
    }
    (cache / "source.json").write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    print(prepare(Path(__file__).resolve().parents[2] / "data/cache/esol"))
