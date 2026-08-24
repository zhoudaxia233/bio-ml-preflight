from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Literal

import typer

from bio_ml_preflight.autoprobe import evaluate_candidate, prepare_run
from bio_ml_preflight.contracts import CaseSpec, load_case, save_case
from bio_ml_preflight.contracts.case import (
    DataSpec,
    DecisionSpec,
    EntitySpec,
    EvaluationSpec,
    FeatureSpec,
    ScenarioSpec,
    TaskSpec,
    ThresholdSpec,
)
from bio_ml_preflight.data import read_table
from bio_ml_preflight.data.bbb import load_bbb_martins
from bio_ml_preflight.data.davis import load_davis
from bio_ml_preflight.data.synthetic import SyntheticKind, generate_synthetic
from bio_ml_preflight.discovery import discover_tasks
from bio_ml_preflight.discovery.cards import infer_roles
from bio_ml_preflight.runner import run_case

app = typer.Typer(no_args_is_help=True, help="Audit evidence boundaries for biological ML claims.")
demo_app = typer.Typer(no_args_is_help=True, help="Run deterministic public or synthetic demos.")
autoprobe_app = typer.Typer(no_args_is_help=True, help="Constrained candidate-model probing.")
app.add_typer(demo_app, name="demo")
app.add_typer(autoprobe_app, name="autoprobe")


@app.command("init-case")
def init_case(
    data_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("case.yaml"),
) -> None:
    """Create a provisional case from physical types without guessing semantics."""
    frame = read_table(data_path)
    numeric = [str(column) for column in frame.select_dtypes(include="number").columns]
    categorical = [str(column) for column in frame.columns if str(column) not in numeric]
    if numeric:
        target: str = numeric[-1]
        kind: Literal["regression", "binary_classification"] = "regression"
    elif categorical:
        target, kind = categorical[-1], "binary_classification"
    else:
        raise typer.BadParameter("No columns are available")
    identifiers = [str(column) for column in frame.columns if "id" in str(column).lower()]
    entities = {"sample": EntitySpec(id_column=identifiers[0])} if identifiers else {}
    feature_columns = [
        str(column)
        for column in frame.columns
        if str(column) != target and str(column) not in identifiers
    ]
    case = CaseSpec(
        case_id=data_path.stem,
        data=DataSpec(path=str(data_path.resolve())),
        task=TaskSpec(
            kind=kind,
            prediction_unit="UNCONFIRMED",
            target_column=target,
        ),
        entities=entities,
        features=FeatureSpec(include=feature_columns),
        generalization_scenarios=[ScenarioSpec(name="random_diagnostic", strategy="random")],
        role_confirmation={
            "target": False,
            "prediction_unit": False,
            "features": False,
            "entities": False,
        },
    )
    relative_data_path = os.path.relpath(data_path.resolve(), output.parent.resolve())
    save_case(case, output, relative_data_path=relative_data_path)
    typer.echo(f"Wrote provisional case to {output}; inferred roles are explicitly unconfirmed.")
    typer.echo(json.dumps(infer_roles(frame), indent=2))


@app.command("validate-case")
def validate_case(case_yaml: Annotated[Path, typer.Argument(exists=True, readable=True)]) -> None:
    """Validate a versioned case and report unresolved declarations."""
    case = load_case(case_yaml)
    unresolved = sorted(key for key, confirmed in case.role_confirmation.items() if not confirmed)
    result = {
        "valid": True,
        "schema_version": case.schema_version,
        "case_id": case.case_id,
        "case_fingerprint": case.fingerprint(),
        "unconfirmed_roles": unresolved,
        "status": "NOT_ASSESSABLE" if unresolved else "READY_FOR_AUDIT",
    }
    typer.echo(json.dumps(result, indent=2))


@app.command("run")
def run_command(
    case_yaml: Annotated[Path, typer.Argument(exists=True, readable=True)],
    budget: Annotated[str, typer.Option("--budget")] = "smoke",
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    """Run deterministic audits and baseline probes for a declared case."""
    if budget not in {"smoke", "standard"}:
        raise typer.BadParameter("budget must be smoke or standard")
    case = load_case(case_yaml)
    destination = output or Path("reports") / case.case_id
    result = run_case(case, destination, budget=budget)
    typer.echo(f"Report: {destination / 'report.md'}")
    typer.echo(_status_summary(result))


@app.command("discover")
def discover_command(
    data_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("discovery"),
) -> None:
    """Generate reviewable analysis claim cards without biological invention."""
    payload = discover_tasks(data_path, output)
    typer.echo(f"Wrote {len(payload['claim_cards'])} provisional claim cards to {output}")


@demo_app.command("synthetic")
def demo_synthetic(
    budget: Annotated[str, typer.Option("--budget")] = "smoke",
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("reports/synthetic"),
) -> None:
    """Run all four deterministic scientific acceptance cases."""
    root = _project_root()
    data_dir = root / "examples" / "synthetic" / "data"
    for kind in ["stable", "leakage", "no_signal", "ranking_instability"]:
        data_path = data_dir / f"{kind}.parquet"
        generate_synthetic(kind, data_path)  # type: ignore[arg-type]
        case = synthetic_case(kind, data_path)  # type: ignore[arg-type]
        destination = output / kind
        result = run_case(case, destination, budget=budget)
        typer.echo(f"{kind}: {_status_summary(result)}")


@demo_app.command("davis")
def demo_davis(
    budget: Annotated[str, typer.Option("--budget")] = "smoke",
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("reports/davis"),
) -> None:
    """Download/cache Davis through TDC and run pairwise scenarios."""
    root = _project_root()
    data_path = root / "data" / "cache" / "davis" / "davis.parquet"
    load_davis(data_path.parent)
    case = load_case(root / "examples" / "davis" / "case.yaml")
    result = run_case(case, output, budget=budget)
    typer.echo(_status_summary(result))


@demo_app.command("bbb")
def demo_bbb(
    budget: Annotated[str, typer.Option("--budget")] = "smoke",
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("reports/bbb-martins"),
) -> None:
    """Download/cache BBB_Martins through TDC and run molecular classification scenarios."""
    root = _project_root()
    data_path = root / "data" / "cache" / "bbb_martins" / "bbb_martins.parquet"
    load_bbb_martins(data_path.parent)
    case = load_case(root / "examples" / "bbb_martins" / "case.yaml")
    result = run_case(case, output, budget=budget)
    typer.echo(_status_summary(result))


@autoprobe_app.command("prepare")
def autoprobe_prepare(
    case_yaml: Annotated[Path, typer.Argument(exists=True, readable=True)],
    run_dir: Annotated[Path, typer.Option("--run-dir")],
) -> None:
    """Lock the invariant case and create the one mutable candidate file."""
    typer.echo(f"Candidate: {prepare_run(case_yaml, run_dir)}")


@autoprobe_app.command("evaluate")
def autoprobe_evaluate(
    candidate_yaml: Annotated[Path, typer.Argument(exists=True, readable=True)],
) -> None:
    """Evaluate one candidate without accessing final holdout labels."""
    typer.echo(json.dumps(evaluate_candidate(candidate_yaml), indent=2))


def synthetic_case(kind: SyntheticKind, data_path: Path) -> CaseSpec:
    data = DataSpec(path=str(data_path.resolve()))
    confirmed = {"target": True, "prediction_unit": True, "features": True}
    if kind == "stable":
        return CaseSpec(
            case_id="synthetic-stable",
            task=TaskSpec(
                kind="regression", prediction_unit="candidate-target pair", target_column="y"
            ),
            entities={"target": EntitySpec(id_column="target_id")},
            features=FeatureSpec(include=["x1", "x2"]),
            decision=DecisionSpec(kind="top_k_per_group", group_entity="target", k=[5, 10]),
            generalization_scenarios=[
                ScenarioSpec(name="random_pair", strategy="random_pair"),
                ScenarioSpec(name="cold_target", strategy="group", group_column="target_id"),
            ],
            evaluation=EvaluationSpec(
                seeds=[11, 23, 47], primary_metric="spearman", bootstrap_unit="target_id"
            ),
            data=data,
            role_confirmation=confirmed,
        )
    if kind == "leakage":
        return CaseSpec(
            case_id="synthetic-leakage",
            task=TaskSpec(kind="regression", prediction_unit="measurement", target_column="y"),
            entities={"entity": EntitySpec(id_column="entity_id")},
            features=FeatureSpec(include=["entity_token", "uninformative"]),
            generalization_scenarios=[
                ScenarioSpec(name="random_row", strategy="random"),
                ScenarioSpec(name="cold_entity", strategy="group", group_column="entity_id"),
            ],
            evaluation=EvaluationSpec(
                seeds=[11, 23, 47], primary_metric="spearman", bootstrap_unit="entity_id"
            ),
            data=data,
            role_confirmation=confirmed,
        )
    if kind == "no_signal":
        return CaseSpec(
            case_id="synthetic-no-signal",
            task=TaskSpec(kind="regression", prediction_unit="sample", target_column="y"),
            entities={"group": EntitySpec(id_column="group_id")},
            features=FeatureSpec(include=["x1", "x2"]),
            generalization_scenarios=[
                ScenarioSpec(name="random_row", strategy="random"),
                ScenarioSpec(name="cold_group", strategy="group", group_column="group_id"),
            ],
            evaluation=EvaluationSpec(
                seeds=[11, 23, 47], primary_metric="spearman", bootstrap_unit="group_id"
            ),
            data=data,
            role_confirmation=confirmed,
        )
    return CaseSpec(
        case_id="synthetic-ranking-instability",
        task=TaskSpec(
            kind="regression", prediction_unit="candidate-target pair", target_column="y"
        ),
        entities={"target": EntitySpec(id_column="target_id")},
        features=FeatureSpec(include=["x", "nuisance"]),
        decision=DecisionSpec(kind="top_k_per_group", group_entity="target", k=[5, 10]),
        generalization_scenarios=[
            ScenarioSpec(name="random_pair", strategy="random_pair"),
            ScenarioSpec(name="cold_target", strategy="group", group_column="target_id"),
        ],
        evaluation=EvaluationSpec(
            seeds=[11, 23, 47], primary_metric="spearman", bootstrap_unit="target_id"
        ),
        thresholds=ThresholdSpec(stable_top_k=0.85, unstable_top_k=0.75),
        data=data,
        role_confirmation=confirmed,
    )


def _status_summary(result: dict[str, object]) -> str:
    rows = result["capability_matrix"]
    assert isinstance(rows, list)
    parts = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        representation = row.get("representation")
        suffix = f"[{representation}]" if representation else ""
        parts.append(f"{row['claim_or_scenario']}{suffix}={row['status']}")
    return ", ".join(parts)


def _project_root() -> Path:
    current = Path.cwd().resolve()
    candidates = [current, *current.parents, Path(__file__).resolve().parents[2]]
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (candidate / "examples").is_dir():
            return candidate
    raise RuntimeError("Run demo commands from a bio-ml-preflight source checkout")


if __name__ == "__main__":
    app()
