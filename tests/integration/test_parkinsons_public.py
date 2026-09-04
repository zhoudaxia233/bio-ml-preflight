from pathlib import Path

import pytest

from bio_ml_preflight.contracts import load_case
from bio_ml_preflight.data.parkinsons import load_parkinsons_telemonitoring
from bio_ml_preflight.runner import run_case


@pytest.mark.network
def test_parkinsons_public_smoke(tmp_path: Path) -> None:
    data, metadata = load_parkinsons_telemonitoring(tmp_path / "cache")
    case = load_case(Path("examples/parkinsons_telemonitoring/case.yaml"))
    case.data.path = str(tmp_path / "cache" / "parkinsons_telemonitoring.parquet")

    result = run_case(case, tmp_path / "report", budget="smoke")
    overlap = result["split_overlap"]
    capability = {
        row["claim_or_scenario"]: row
        for row in result["capability_matrix"]
        if row["claim_or_scenario"] in {"random_record", "unseen_participant"}
    }

    assert metadata["row_count"] == len(data) == 5875
    assert metadata["participant_count"] == data["subject_id"].nunique() == 42
    assert overlap["random_record:11"]["entity_overlap"]["participant"]["count"] > 0
    assert overlap["unseen_participant:11"]["entity_overlap"]["participant"]["count"] == 0
    assert capability.keys() == {"random_record", "unseen_participant"}
    assert capability["random_record"]["status"] == "SUPPORTED_WITH_LIMITS"
    assert capability["random_record"]["numbers"]["bootstrap_unit_overlap_count"] == 42
    assert capability["unseen_participant"]["status"] == "INSUFFICIENT_EVIDENCE"
    assert capability["unseen_participant"]["numbers"]["permutation_p_value"] == 0.3
    assert any(
        "median spearman=0.195" in value
        for value in capability["unseen_participant"]["evidence_against"]
    )
    assert {
        "Permutation separation is insufficient.",
        "The controlled baseline does not meet the configured usefulness threshold.",
        "Performance is unstable across the evaluated splits.",
    } <= set(capability["unseen_participant"]["unmet_assumptions"])
    assert all(row["numbers"]["permutation_draws"] == 9 for row in capability.values())
    assert "linearly interpolated" in result["dataset_source"]["target_semantics"]
    assert "linearly interpolated" in (tmp_path / "report" / "report.md").read_text(
        encoding="utf-8"
    )
    assert result["dataset_source"]["participant_time_proxy"]["unique"] == 2501
    assert result["dataset_source"]["participant_time_proxy"]["records_in_repeated_groups"] == 4904
    assert (
        result["provenance"]["holdout_description"]
        == "retrospective development evaluation; no holdout"
    )
