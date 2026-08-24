from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataSpec(StrictModel):
    adapter: Literal["tabular", "davis", "bbb_martins"] = "tabular"
    path: str
    fingerprint_columns: list[str] = Field(default_factory=list)


class TaskSpec(StrictModel):
    kind: Literal["regression", "binary_classification", "ranking"]
    prediction_unit: str
    target_column: str
    higher_is_better: bool = True


class EntitySpec(StrictModel):
    id_column: str
    representation_column: str | None = None


class FeatureSpec(StrictModel):
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    post_outcome: list[str] = Field(default_factory=list)


class MetadataSpec(StrictModel):
    replicate_id: str | None = None
    biological_replicate_id: str | None = None
    batch_id: str | None = None
    plate_id: str | None = None
    time_column: str | None = None
    treatment_column: str | None = None
    control_value: str | float | int | None = None


class DecisionSpec(StrictModel):
    kind: Literal["top_k_per_group", "threshold", "none"] = "none"
    group_entity: str | None = None
    k: list[int] = Field(default_factory=lambda: [5, 10])


class ScenarioSpec(StrictModel):
    name: str
    strategy: Literal[
        "random",
        "group",
        "random_pair",
        "cold_left",
        "cold_right",
        "double_cold",
        "time",
        "supplied",
        "scaffold",
    ]
    group_column: str | None = None
    left_column: str | None = None
    right_column: str | None = None
    split_column: str | None = None

    @model_validator(mode="after")
    def validate_strategy_fields(self) -> ScenarioSpec:
        if self.strategy == "group" and not self.group_column:
            raise ValueError("group strategy requires group_column")
        if self.strategy in {"cold_left", "double_cold"} and not self.left_column:
            raise ValueError(f"{self.strategy} requires left_column")
        if self.strategy in {"cold_right", "double_cold"} and not self.right_column:
            raise ValueError(f"{self.strategy} requires right_column")
        if self.strategy == "supplied" and not self.split_column:
            raise ValueError("supplied strategy requires split_column")
        return self


class EvaluationSpec(StrictModel):
    seeds: list[int] = Field(default_factory=lambda: [11, 23, 47, 83, 131])
    primary_metric: str = "spearman"
    secondary_metrics: list[str] = Field(default_factory=lambda: ["mae", "rmse"])
    bootstrap_unit: str | None = None
    permutation_draws: int = Field(default=9, ge=1, le=100)

    @model_validator(mode="after")
    def seeds_are_unique(self) -> EvaluationSpec:
        if not self.seeds or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("evaluation.seeds must contain unique values")
        return self


class HoldoutSpec(StrictModel):
    enabled: bool = False
    maximum_accesses: int = Field(default=1, ge=1)


class ThresholdSpec(StrictModel):
    supported_metric: float = 0.5
    limited_metric: float = 0.2
    stable_top_k: float = 0.65
    unstable_top_k: float = 0.5
    maximum_dispersion: float = 0.15
    minimum_permutation_delta: float = 0.05
    maximum_permutation_p_value: float = Field(default=0.1, gt=0, le=1)


class CaseSpec(StrictModel):
    schema_version: Literal[1] = 1
    case_id: str
    data: DataSpec
    task: TaskSpec
    entities: dict[str, EntitySpec] = Field(default_factory=dict)
    features: FeatureSpec = Field(default_factory=FeatureSpec)
    metadata: MetadataSpec = Field(default_factory=MetadataSpec)
    decision: DecisionSpec = Field(default_factory=DecisionSpec)
    generalization_scenarios: list[ScenarioSpec]
    evaluation: EvaluationSpec = Field(default_factory=EvaluationSpec)
    holdout: HoldoutSpec = Field(default_factory=HoldoutSpec)
    thresholds: ThresholdSpec = Field(default_factory=ThresholdSpec)
    role_confirmation: dict[str, bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def scenarios_are_unique(self) -> CaseSpec:
        names = [scenario.name for scenario in self.generalization_scenarios]
        if not names or len(names) != len(set(names)):
            raise ValueError("generalization scenario names must be present and unique")
        return self

    def fingerprint(self) -> str:
        payload = self.model_dump_json(exclude_none=False)
        return hashlib.sha256(payload.encode()).hexdigest()


def load_case(path: Path) -> CaseSpec:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    case = CaseSpec.model_validate(raw)
    data_path = Path(case.data.path)
    if not data_path.is_absolute():
        case.data.path = str((path.parent / data_path).resolve())
    return case


def save_case(case: CaseSpec, path: Path, *, relative_data_path: str | None = None) -> None:
    payload = case.model_dump(mode="json", exclude_none=False)
    if relative_data_path is not None:
        payload["data"]["path"] = relative_data_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
