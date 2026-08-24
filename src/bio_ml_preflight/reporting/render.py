from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import BaseLoader, Environment

REPORT_TEMPLATE = """# Capability boundary report: {{ case.case_id }}

Generated from structured artifacts. This retrospective preflight does not prove future success.

## 1. Intended claim and deployment reality

- Task: `{{ case.task.kind }}` for prediction unit `{{ case.task.prediction_unit }}`.
- Target: `{{ case.task.target_column }}`.
- Generalization scenarios: {{ scenarios }}.
- Decision: `{{ case.decision.kind }}`.

## 2. Data inventory

{{ inventory }}

## 3. Independence and replicate structure

{{ independence }}

## 4. Leakage findings

{{ leakage }}

## 5. Measurement reliability

{{ measurement }}

## 6. Coverage

{{ coverage }}

## 7. Baseline and permutation results

See `aggregate_experiments.parquet` and the metric figure below.

![Scenario metrics](figures/scenario_metrics.png)

Independent-group learning curve (when a realistic group axis is available):

{{ learning_curve }}

## 8. Generalization scenarios

{{ scenario_table }}

## 9. Stability decomposition

{{ stability }}

## 10. Decision-level ranking analysis

{{ ranking }}

![Ranking stability](figures/ranking_stability.png)

## 11. Capability matrix

{{ capability_table }}

## 12. Most informative next data or experiment

{{ next_steps }}

## 13. Limitations

- Findings are conditional on declared roles, available metadata, split construction,
  models, and thresholds.
- Missing biological metadata is reported, not replaced by assumptions.
- Development ranking stability may include fitted-row predictions; use a locked external
  holdout for confirmation.
- Random splits are diagnostic and are not evidence of unseen-entity or future-time generalization.
- No predictive result alone establishes a causal effect.
"""


def _pretty(value: Any) -> str:
    return "```json\n" + json.dumps(value, indent=2, sort_keys=True) + "\n```"


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "No assessable results."
    header = "| " + " | ".join(columns) + " |"
    line = "|" + "|".join("---" for _ in columns) + "|"
    body = [
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |" for row in rows
    ]
    return "\n".join([header, line, *body])


def write_report(
    output: Path,
    *,
    case: Any,
    structured: dict[str, Any],
    experiments: pd.DataFrame,
    ranking_table: pd.DataFrame,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    experiments.to_parquet(output / "aggregate_experiments.parquet", index=False)
    ranking_table.to_parquet(output / "ranking_stability.parquet", index=False)
    capability = structured["capability_matrix"]
    pd.DataFrame(capability).to_parquet(output / "capability_matrix.parquet", index=False)
    (output / "report.json").write_text(
        json.dumps(structured, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    _metric_figure(experiments, figures / "scenario_metrics.png", case.evaluation.primary_metric)
    _ranking_figure(structured["ranking_stability"], figures / "ranking_stability.png")
    scenario_rows = [
        {
            "scenario": row["claim_or_scenario"],
            "status": row["status"],
            "numbers": json.dumps(row["numbers"], sort_keys=True),
        }
        for row in capability
        if not row["claim_or_scenario"].startswith("top-")
    ]
    context = {
        "case": case,
        "scenarios": ", ".join(scenario.name for scenario in case.generalization_scenarios),
        "inventory": _pretty(structured["audits"]["inventory"]),
        "independence": _pretty(structured["audits"]["independence"]),
        "leakage": _pretty(structured["audits"]["leakage"]),
        "measurement": _pretty(structured["audits"]["measurement"]),
        "coverage": _pretty(structured["audits"]["coverage"]),
        "scenario_table": _table(scenario_rows, ["scenario", "status", "numbers"]),
        "learning_curve": _pretty(structured["learning_curve"]),
        "stability": _pretty(structured["stability_decomposition"]),
        "ranking": _pretty(structured["ranking_stability"]),
        "capability_table": _table(
            capability, ["claim_or_scenario", "status", "uncertainty", "cheapest_next_evidence"]
        ),
        "next_steps": "\n".join(
            f"- **{row['claim_or_scenario']}**: {row['cheapest_next_evidence']}"
            for row in capability
        ),
    }
    markdown = (
        Environment(loader=BaseLoader(), autoescape=False)
        .from_string(REPORT_TEMPLATE)
        .render(context)
    )
    (output / "report.md").write_text(markdown, encoding="utf-8")
    html = (
        Environment(loader=BaseLoader(), autoescape=True)
        .from_string(
            """<!doctype html><html><head><meta charset="utf-8">
<title>Capability boundary report</title>
<style>body{max-width:1000px;margin:2rem auto;font:16px system-ui;line-height:1.5;padding:0 1rem}
pre{overflow:auto;background:#f5f5f5;padding:1rem}
table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:.4rem}
img{max-width:100%}</style>
</head><body><pre>{{ markdown }}</pre></body></html>"""
        )
        .render(markdown=markdown)
    )
    (output / "report.html").write_text(html, encoding="utf-8")


def _metric_figure(experiments: pd.DataFrame, path: Path, metric: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.5))
    real = experiments[experiments["permuted"].eq(False)]
    labels, values = [], []
    for (scenario, model), group in real.groupby(["scenario", "model"]):
        finite = group[metric].dropna()
        if len(finite):
            labels.append(f"{scenario}\n{model}")
            values.append(float(finite.median()))
    axis.bar(range(len(values)), values, color="#2a6f97")
    axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    axis.set_ylabel(metric)
    axis.axhline(0, color="black", linewidth=0.7)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def _ranking_figure(ranking: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(6, 3.5))
    values = ranking.get("top_k", {})
    labels = list(values)
    overlaps = [values[label].get("average_pairwise_jaccard") or 0 for label in labels]
    axis.bar(labels, overlaps, color="#d97706")
    axis.set_xlabel("k")
    axis.set_ylabel("average pairwise Jaccard")
    axis.set_ylim(0, 1)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
