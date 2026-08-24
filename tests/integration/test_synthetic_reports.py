from pathlib import Path

import pytest

from bio_ml_preflight.cli import synthetic_case
from bio_ml_preflight.data.synthetic import SyntheticKind, generate_synthetic
from bio_ml_preflight.runner import run_case


@pytest.mark.parametrize("kind", ["stable", "leakage", "no_signal", "ranking_instability"])
def test_synthetic_qualitative_findings(tmp_path: Path, kind: SyntheticKind) -> None:
    path = tmp_path / f"{kind}.parquet"
    generate_synthetic(kind, path)
    result = run_case(synthetic_case(kind, path), tmp_path / f"report-{kind}", budget="smoke")
    statuses = {row["claim_or_scenario"]: row["status"] for row in result["capability_matrix"]}
    if kind == "stable":
        assert statuses["cold_target"] in {"SUPPORTED", "SUPPORTED_WITH_LIMITS"}
        random_pair = next(
            row for row in result["capability_matrix"] if row["claim_or_scenario"] == "random_pair"
        )
        assert random_pair["numbers"]["permutation_draws"] == 9
        assert 0 < random_pair["numbers"]["permutation_p_value"] <= 1
    elif kind == "leakage":
        assert statuses["random_row"] in {"SUPPORTED", "SUPPORTED_WITH_LIMITS"}
        assert statuses["cold_entity"] == "CONTRADICTED"
        assert result["audits"]["leakage"]["suspicious_identifier_features"] == ["entity_token"]
    elif kind == "no_signal":
        assert set(statuses.values()) <= {"INSUFFICIENT_EVIDENCE", "CONTRADICTED"}
    else:
        assert statuses["top-5 selection within target"] == "INSUFFICIENT_EVIDENCE"
    report = tmp_path / f"report-{kind}"
    for artifact in [
        "report.md",
        "report.html",
        "report.json",
        "aggregate_experiments.parquet",
        "ranking_stability.parquet",
        "capability_matrix.parquet",
        "provenance.json",
    ]:
        assert (report / artifact).exists()
    assert list((report / "split_manifests").glob("*.json"))
    assert list((report / "predictions").glob("*.parquet"))
