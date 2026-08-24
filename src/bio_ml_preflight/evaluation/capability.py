from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from bio_ml_preflight.contracts import CaseSpec
from bio_ml_preflight.evaluation.metrics import empirical_permutation_summary

Status = str


def capability_matrix(
    experiments: pd.DataFrame,
    case: CaseSpec,
    ranking: dict[str, Any],
) -> list[dict[str, Any]]:
    primary = case.evaluation.primary_metric
    usable = experiments[experiments["model"].ne("dummy")]
    random_values = usable[
        usable["strategy"].isin(["random", "random_pair"]) & usable["permuted"].eq(False)
    ][primary]
    random_median = float(random_values.median()) if random_values.notna().any() else None
    rows = []
    for scenario in case.generalization_scenarios:
        current = usable[usable["scenario"].eq(scenario.name)]
        real = current[current["permuted"].eq(False)]
        control = current[current["permuted"].eq(True)]
        controlled_models = set(control["model"])
        by_model = (
            real[real["model"].isin(controlled_models)].groupby("model")[primary].median().dropna()
        )
        if by_model.empty:
            rows.append(
                _verdict(
                    scenario.name,
                    "NOT_ASSESSABLE",
                    [],
                    ["No applicable model produced a finite primary metric."],
                    "Undefined",
                    ["The requested task/metric could not be evaluated."],
                    "Repair the target, features, or split so one baseline can be evaluated.",
                    {},
                )
            )
            continue
        best_model = str(by_model.idxmax())
        best = real[real["model"].eq(best_model)][primary].dropna()
        perm = control[control["model"].eq(best_model)][primary].dropna()
        metric = float(best.median())
        dispersion = float(best.std()) if len(best) > 1 else 0.0
        model_control = control[control["model"].eq(best_model)]
        if "permutation_draw" in model_control and model_control["permutation_draw"].notna().any():
            null_statistics = (
                model_control.groupby("permutation_draw")[primary].median().dropna().tolist()
            )
        else:
            null_statistics = perm.tolist()
        permutation_summary = empirical_permutation_summary(metric, null_statistics)
        permutation = permutation_summary["median"]
        delta = metric - permutation if permutation is not None else None
        permutation_p_value = permutation_summary["p_value"]
        numbers = {
            "primary_metric": primary,
            "median": metric,
            "dispersion": dispersion,
            "best_model": best_model,
            "permutation_median": permutation,
            "permutation_delta": delta,
            "permutation_q95": permutation_summary["q95"],
            "permutation_p_value": permutation_p_value,
            "permutation_draws": permutation_summary["draws"],
        }
        evidence_for = [f"Best baseline median {primary}={metric:.3f} ({best_model})."]
        evidence_against = []
        unmet = []
        if delta is not None:
            evidence_for.append(f"Delta over grouped permutation control={delta:.3f}.")
        if permutation_p_value is not None:
            evidence_for.append(
                f"Empirical grouped-permutation p={permutation_p_value:.3f} "
                f"from {permutation_summary['draws']} draws."
            )
        if scenario.strategy not in {"random", "random_pair"} and random_median is not None:
            evidence_against.append(f"Random-split median across probes={random_median:.3f}.")
        weak_permutation = (
            delta is None
            or delta < case.thresholds.minimum_permutation_delta
            or permutation_p_value is None
            or permutation_p_value > case.thresholds.maximum_permutation_p_value
        )
        if weak_permutation:
            evidence_against.append("The improvement over permutation is absent or too small.")
        if dispersion > case.thresholds.maximum_dispersion:
            evidence_against.append(
                f"Split dispersion {dispersion:.3f} exceeds the configured limit."
            )
        leakage_contradiction = (
            scenario.strategy not in {"random", "random_pair"}
            and random_median is not None
            and random_median >= case.thresholds.supported_metric
            and metric < case.thresholds.limited_metric
        )
        if leakage_contradiction:
            status = "CONTRADICTED"
            unmet.append(
                "Performance does not transfer from row-random to entity-separated validation."
            )
            next_step = (
                "Collect or reserve outcomes for genuinely unseen entities and validate once."
            )
        elif metric < case.thresholds.limited_metric or weak_permutation:
            status = "INSUFFICIENT_EVIDENCE"
            unmet.append("Signal strength and/or permutation separation is insufficient.")
            next_step = (
                "Add independent units along this scenario, then repeat the fixed validation."
            )
        elif (
            metric >= case.thresholds.supported_metric
            and dispersion <= case.thresholds.maximum_dispersion
        ):
            status = "SUPPORTED"
            next_step = "Confirm once on a prospectively reserved or pseudo-sealed external set."
        else:
            status = "SUPPORTED_WITH_LIMITS"
            unmet.append("Evidence is moderate or sensitive to the sampled split.")
            next_step = "Increase independent entity coverage at the weakest split boundary."
        rows.append(
            _verdict(
                scenario.name,
                status,
                evidence_for,
                evidence_against,
                f"Across-split standard deviation={dispersion:.3f}.",
                unmet,
                next_step,
                numbers,
            )
        )
    if case.decision.kind == "top_k_per_group":
        k = str(min(case.decision.k))
        top = ranking.get("top_k", {}).get(k, {})
        overlap = top.get("average_pairwise_jaccard")
        if overlap is None:
            status = "NOT_ASSESSABLE"
            evidence_against = ["No comparable grouped rankings were available across runs."]
        elif overlap < case.thresholds.unstable_top_k:
            status = "INSUFFICIENT_EVIDENCE"
            evidence_against = [f"Average pairwise top-{k} Jaccard={overlap:.3f}."]
        elif overlap < case.thresholds.stable_top_k:
            status = "SUPPORTED_WITH_LIMITS"
            evidence_against = [f"Average pairwise top-{k} Jaccard={overlap:.3f} is only moderate."]
        else:
            status = "SUPPORTED"
            evidence_against = []
        rows.append(
            _verdict(
                f"top-{k} selection within {case.decision.group_entity}",
                status,
                [] if overlap is None else [f"Observed top-{k} overlap={overlap:.3f}."],
                evidence_against,
                "Development-run membership variation; no final holdout exposure.",
                [] if status == "SUPPORTED" else ["Candidate membership changes across runs."],
                "Measure the borderline candidates again, then evaluate the locked ranking once.",
                {"average_pairwise_jaccard": overlap},
            )
        )
    return rows


def _verdict(
    claim: str,
    status: Status,
    evidence_for: list[str],
    evidence_against: list[str],
    uncertainty: str,
    assumptions: list[str],
    next_step: str,
    numbers: dict[str, Any],
) -> dict[str, Any]:
    return {
        "claim_or_scenario": claim,
        "status": status,
        "evidence_supporting": evidence_for,
        "evidence_against": evidence_against,
        "uncertainty": uncertainty,
        "unmet_assumptions": assumptions,
        "cheapest_next_evidence": next_step,
        "numbers": {
            key: None if isinstance(value, float) and not np.isfinite(value) else value
            for key, value in numbers.items()
        },
    }
