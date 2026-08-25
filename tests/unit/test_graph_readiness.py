import json
from pathlib import Path

import pandas as pd

from bio_ml_preflight.evaluation.graph_readiness import (
    assess_graph_readiness,
    write_graph_readiness,
)


def test_adequately_supported_external_null_failure_does_not_justify_a_graph_probe(
    tmp_path: Path,
) -> None:
    development = tmp_path / "development"
    development.mkdir()
    (development / "report.json").write_text(
        json.dumps(
            {
                "case": {"case_id": "development"},
                "audits": {
                    "graph_readiness_contract": {
                        "status": "VALID_CONTRACT",
                        "rows_converted": 200,
                        "unique_canonical_graphs": 190,
                        "evaluation_scenarios": [{"name": "scaffold"}],
                        "support": {
                            "unit": "compound_id",
                            "class_counts": {"0": 80, "1": 120},
                            "minimum_independent_units_per_class": 20,
                        },
                        "metadata_available": {
                            "replicate": False,
                            "batch": False,
                            "time": False,
                            "treatment": False,
                        },
                    }
                },
                "split_overlap": {
                    "scaffold:11": {
                        "canonical_graph_overlap": {"count": 0},
                        "test_target_counts": {"0": 20, "1": 30},
                        "test_target_count_unit": "compound_id",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "claim_or_scenario": "scaffold",
                "representation": representation,
                "status": "SUPPORTED",
                "numbers": {
                    "median": median,
                    "best_model": "extra_trees",
                    "permutation_delta": 0.2,
                    "permutation_p_value": 0.1,
                },
            }
            for representation, median in [("character_hash", 0.70), ("morgan", 0.75)]
        ]
    ).to_parquet(development / "capability_matrix.parquet", index=False)
    pd.DataFrame(
        [
            {
                "claim_or_scenario": "scaffold",
                "verdict_changes": False,
                "maximum_matched_model_range": 0.05,
            }
        ]
    ).to_parquet(development / "representation_sensitivity.parquet", index=False)
    pd.DataFrame(columns=["scenario"]).to_parquet(
        development / "learning_curve.parquet", index=False
    )

    external = tmp_path / "external"
    external.mkdir()
    (external / "report.json").write_text(
        json.dumps(
            {
                "case": {
                    "case_id": "external",
                    "generalization_scenarios": [{"name": "external"}],
                    "thresholds": {"maximum_permutation_p_value": 0.1},
                }
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "claim_or_scenario": "external",
                "representation": "morgan",
                "status": "INSUFFICIENT_EVIDENCE",
                "numbers": {
                    "median": 0.57,
                    "permutation_q95": 0.59,
                    "permutation_p_value": 0.3,
                    "test_class_counts": {"0": 37, "1": 500},
                    "minimum_test_class_count_required": 20,
                },
            }
        ]
    ).to_parquet(external / "capability_matrix.parquet", index=False)

    result = assess_graph_readiness(development, external)
    output = tmp_path / "readiness"
    write_graph_readiness(result, output)

    assert result["status"] == "NOT_JUSTIFIED_BY_CURRENT_EVIDENCE"
    assert result["numbers"]["external_confirmation"]["adequately_supported_null_failure"]
    assert (output / "graph_readiness.json").is_file()
    assert pd.read_parquet(output / "graph_readiness.parquet").iloc[0]["status"] == result["status"]
