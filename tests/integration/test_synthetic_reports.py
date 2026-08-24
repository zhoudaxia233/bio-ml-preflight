from pathlib import Path

import pytest

from bio_ml_preflight.cli import synthetic_case
from bio_ml_preflight.data.synthetic import SyntheticKind, generate_synthetic
from bio_ml_preflight.runner import run_case


@pytest.mark.parametrize("kind", ["stable", "leakage", "no_signal", "ranking_instability"])
def test_synthetic_qualitative_findings(tmp_path: Path, kind: SyntheticKind) -> None:
    path = tmp_path / f"{kind}.parquet"
    generate_synthetic(kind, path)
    case = synthetic_case(kind, path)
    result = run_case(case, tmp_path / f"report-{kind}", budget="smoke")
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
        scenario_statuses = {statuses[scenario.name] for scenario in case.generalization_scenarios}
        assert scenario_statuses <= {"INSUFFICIENT_EVIDENCE", "CONTRADICTED"}
    else:
        assert statuses["top-5 selection within target"] == "INSUFFICIENT_EVIDENCE"
    report = tmp_path / f"report-{kind}"
    for artifact in [
        "report.md",
        "report.html",
        "report.json",
        "aggregate_experiments.parquet",
        "representation_sensitivity.parquet",
        "ranking_stability.parquet",
        "capability_matrix.parquet",
        "provenance.json",
    ]:
        assert (report / artifact).exists()
    html = (report / "report.html").read_text(encoding="utf-8")
    assert "<body><pre># Capability boundary report" not in html
    assert "<h1>Capability boundary report:" in html
    assert "<table>" in html
    assert "<h3>Identity-consistency gate</h3>" in html
    assert "Representation sensitivity under identical split manifests" in html
    assert result["audits"]["identity_consistency"]["status"] == "NO_CONFLICTS"
    assert result["representation_sensitivity"]
    scenario_section = html.split("<h2>8. Generalization scenarios</h2>", 1)[1].split(
        "</section>", 1
    )[0]
    capability_section = html.split("<h2>11. Capability matrix</h2>", 1)[1].split("</section>", 1)[
        0
    ]
    assert "measurement reliability" not in scenario_section
    assert "batch confounding" not in scenario_section
    assert "measurement reliability" in capability_section
    assert "batch confounding" in capability_section
    assert list((report / "split_manifests").glob("*.json"))
    assert list((report / "predictions").glob("*.parquet"))
