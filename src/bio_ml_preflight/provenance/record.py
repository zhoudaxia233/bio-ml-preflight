from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bio_ml_preflight.contracts import CaseSpec


def _git_commit(repository: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def build_provenance(
    case: CaseSpec,
    *,
    dataset_fingerprint: str,
    split_hashes: dict[str, str],
    runtime_seconds: float,
    command_line: list[str] | None = None,
) -> dict[str, Any]:
    packages = {}
    for name in ["bio-ml-preflight", "numpy", "pandas", "scikit-learn", "scipy", "pyarrow"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(Path.cwd()),
        "python": sys.version,
        "packages": packages,
        "operating_system": platform.platform(),
        "case_spec_hash": case.fingerprint(),
        "dataset_fingerprint": dataset_fingerprint,
        "split_manifest_hashes": split_hashes,
        "seeds": case.evaluation.seeds,
        "command_line": command_line or sys.argv,
        "runtime_seconds": runtime_seconds,
        "cpu_count": os.cpu_count(),
        "holdout_description": (
            "pseudo-sealed public benchmark"
            if case.data.adapter == "davis"
            else "locally governed holdout"
        ),
    }
