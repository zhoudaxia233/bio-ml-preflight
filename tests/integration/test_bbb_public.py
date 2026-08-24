from pathlib import Path

import pytest

from bio_ml_preflight.contracts import load_case
from bio_ml_preflight.data.bbb import load_bbb_martins
from bio_ml_preflight.runner import run_case


@pytest.mark.network
def test_bbb_martins_public_smoke(tmp_path: Path) -> None:
    data, metadata = load_bbb_martins(tmp_path / "cache")
    case = load_case(Path("examples/bbb_martins/case.yaml"))
    case.data.path = str(tmp_path / "cache" / "bbb_martins.parquet")
    result = run_case(case, tmp_path / "report", budget="smoke")
    rows = {
        (row["claim_or_scenario"], row.get("representation")): row
        for row in result["capability_matrix"]
    }
    assert metadata["row_count"] == len(data)
    for representation in ["character_hash", "morgan"]:
        assert rows[("random_compound", representation)]["status"] in {
            "SUPPORTED",
            "SUPPORTED_WITH_LIMITS",
        }
        assert rows[("scaffold_generalization", representation)]["status"] in {
            "SUPPORTED",
            "SUPPORTED_WITH_LIMITS",
            "INSUFFICIENT_EVIDENCE",
        }
        assert (
            rows[("random_compound", representation)]["numbers"]["median"]
            > rows[("scaffold_generalization", representation)]["numbers"]["median"]
        )
        assert rows[("random_compound", representation)]["numbers"]["permutation_draws"] == 9
