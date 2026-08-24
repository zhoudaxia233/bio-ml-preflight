from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

SyntheticKind = Literal["stable", "leakage", "no_signal", "ranking_instability"]


def generate_synthetic(kind: SyntheticKind, path: Path, seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if kind == "stable":
        groups, per_group = 24, 20
        target_id = np.repeat([f"T{i:02d}" for i in range(groups)], per_group)
        x1, x2 = rng.normal(size=(2, groups * per_group))
        y = 2.2 * x1 - 1.4 * x2 + rng.normal(scale=0.25, size=len(x1))
        frame = pd.DataFrame(
            {
                "sample_id": [f"S{i:04d}" for i in range(len(x1))],
                "target_id": target_id,
                "candidate_id": [f"C{i:04d}" for i in range(len(x1))],
                "x1": x1,
                "x2": x2,
                "y": y,
            }
        )
    elif kind == "leakage":
        groups, repeats = 72, 7
        entity_id = np.repeat([f"E{i:03d}" for i in range(groups)], repeats)
        latent = np.repeat(rng.normal(size=groups), repeats)
        frame = pd.DataFrame(
            {
                "sample_id": [f"S{i:04d}" for i in range(groups * repeats)],
                "entity_id": entity_id,
                "entity_token": entity_id,
                "uninformative": rng.normal(size=groups * repeats),
                "y": latent + rng.normal(scale=0.03, size=groups * repeats),
            }
        )
    elif kind == "no_signal":
        groups, per_group = 30, 12
        frame = pd.DataFrame(
            {
                "sample_id": [f"S{i:04d}" for i in range(groups * per_group)],
                "group_id": np.repeat([f"G{i:02d}" for i in range(groups)], per_group),
                "x1": rng.normal(size=groups * per_group),
                "x2": rng.normal(size=groups * per_group),
                "y": rng.normal(size=groups * per_group),
            }
        )
    else:
        groups, per_group = 30, 24
        x = rng.normal(size=groups * per_group)
        nuisance = rng.normal(size=groups * per_group)
        # A moderate global signal with deliberately crowded/noisy top ranks.
        y = 0.55 * x + rng.normal(scale=1.0, size=len(x))
        frame = pd.DataFrame(
            {
                "sample_id": [f"S{i:04d}" for i in range(len(x))],
                "target_id": np.repeat([f"T{i:02d}" for i in range(groups)], per_group),
                "candidate_id": [f"C{i:04d}" for i in range(len(x))],
                "x": x,
                "nuisance": nuisance,
                "y": y,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame
