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

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Capability boundary report: {{ case.case_id }}</title>
<style>
:root{color-scheme:light;font-family:Inter,ui-sans-serif,system-ui,sans-serif;
color:#17202a;background:#f5f7fa}
body{max-width:1100px;margin:0 auto;padding:2rem 1.25rem 4rem;line-height:1.55}
header,section{background:#fff;border:1px solid #dfe5ec;border-radius:10px;
padding:1.25rem 1.5rem;margin-bottom:1rem}
h1{margin:.1rem 0 .5rem;font-size:2rem}h2{margin:0 0 .8rem;font-size:1.25rem}
.lede{color:#52606d;margin:0}.status{font-weight:700}.status-SUPPORTED{color:#087f5b}
.status-SUPPORTED_WITH_LIMITS{color:#9c5d00}
.status-INSUFFICIENT_EVIDENCE,.status-CONTRADICTED{color:#b42318}
pre{overflow:auto;background:#f3f5f7;border-radius:6px;padding:1rem;font-size:.82rem}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
table{border-collapse:collapse;width:100%;font-size:.9rem}
td,th{border:1px solid #d8dee6;padding:.55rem;text-align:left;vertical-align:top}
th{background:#eef2f6}
img{display:block;max-width:100%;height:auto;margin:1rem auto}li{margin:.25rem 0}
</style>
</head>
<body>
<header>
<h1>Capability boundary report: {{ case.case_id }}</h1>
<p class="lede">
Generated from structured artifacts. This retrospective preflight does not prove future success.
</p>
</header>
<main>
<section><h2>1. Intended claim and deployment reality</h2>
<ul>
<li>
Task: <code>{{ case.task.kind }}</code> for prediction unit
<code>{{ case.task.prediction_unit }}</code>.
</li>
<li>Target: <code>{{ case.task.target_column }}</code>.</li>
<li>Generalization scenarios: {{ scenarios }}.</li>
<li>Decision: <code>{{ case.decision.kind }}</code>.</li>
</ul></section>
<section><h2>2. Data inventory</h2><pre><code>{{ inventory_json }}</code></pre></section>
<section>
<h2>3. Independence and replicate structure</h2>
<pre><code>{{ independence_json }}</code></pre>
</section>
<section><h2>4. Leakage findings</h2><pre><code>{{ leakage_json }}</code></pre></section>
<section><h2>5. Measurement reliability</h2><pre><code>{{ measurement_json }}</code></pre></section>
<section><h2>6. Coverage</h2><pre><code>{{ coverage_json }}</code></pre></section>
<section><h2>7. Baseline and permutation results</h2>
<p>See <code>aggregate_experiments.parquet</code> for individual results.</p>
<img src="figures/scenario_metrics.png" alt="Scenario metric comparison">
<p>Independent-group learning curve:</p><pre><code>{{ learning_curve_json }}</code></pre></section>
<section><h2>8. Generalization scenarios</h2>
<table><thead><tr><th>Scenario</th><th>Status</th><th>Numbers</th></tr></thead><tbody>
{% for row in scenario_rows %}
<tr>
<td>{{ row.scenario }}</td>
<td class="status status-{{ row.status }}">{{ row.status }}</td>
<td><code>{{ row.numbers }}</code></td>
</tr>
{% endfor %}
</tbody></table></section>
<section><h2>9. Stability decomposition</h2><pre><code>{{ stability_json }}</code></pre></section>
<section><h2>10. Decision-level ranking analysis</h2><pre><code>{{ ranking_json }}</code></pre>
<img src="figures/ranking_stability.png" alt="Ranking stability"></section>
<section><h2>11. Capability matrix</h2>
<table><thead><tr>
<th>Claim or scenario</th><th>Status</th><th>Uncertainty</th><th>Cheapest next evidence</th>
</tr></thead><tbody>
{% for row in capability %}
<tr>
<td>{{ row.claim_or_scenario }}</td>
<td class="status status-{{ row.status }}">{{ row.status }}</td>
<td>{{ row.uncertainty }}</td>
<td>{{ row.cheapest_next_evidence }}</td>
</tr>
{% endfor %}
</tbody></table></section>
<section><h2>12. Most informative next data or experiment</h2><ul>
{% for row in capability %}
<li><strong>{{ row.claim_or_scenario }}</strong>: {{ row.cheapest_next_evidence }}</li>
{% endfor %}
</ul></section>
<section><h2>13. Limitations</h2><ul>
<li>
Findings are conditional on declared roles, available metadata, split construction,
models, and thresholds.
</li>
<li>Missing biological metadata is reported, not replaced by assumptions.</li>
<li>
Development ranking stability may include fitted-row predictions; use a locked external holdout
for confirmation.
</li>
<li>
Random splits are diagnostic and are not evidence of unseen-entity or future-time generalization.
</li>
<li>No predictive result alone establishes a causal effect.</li>
</ul></section>
</main>
</body>
</html>
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
        if row["claim_or_scenario"] in {scenario.name for scenario in case.generalization_scenarios}
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
    json_context = {
        "inventory_json": json.dumps(structured["audits"]["inventory"], indent=2, sort_keys=True),
        "independence_json": json.dumps(
            structured["audits"]["independence"], indent=2, sort_keys=True
        ),
        "leakage_json": json.dumps(structured["audits"]["leakage"], indent=2, sort_keys=True),
        "measurement_json": json.dumps(
            structured["audits"]["measurement"], indent=2, sort_keys=True
        ),
        "coverage_json": json.dumps(structured["audits"]["coverage"], indent=2, sort_keys=True),
        "learning_curve_json": json.dumps(structured["learning_curve"], indent=2, sort_keys=True),
        "stability_json": json.dumps(
            structured["stability_decomposition"], indent=2, sort_keys=True
        ),
        "ranking_json": json.dumps(structured["ranking_stability"], indent=2, sort_keys=True),
    }
    html = (
        Environment(loader=BaseLoader(), autoescape=True)
        .from_string(HTML_TEMPLATE)
        .render(
            case=case,
            scenarios=context["scenarios"],
            scenario_rows=scenario_rows,
            capability=capability,
            **json_context,
        )
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
