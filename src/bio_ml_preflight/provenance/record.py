from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bio_ml_preflight.contracts import CaseSpec


def _git_state(repository: Path) -> dict[str, str | bool | int | None]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if revision.returncode != 0:
        return {
            "git_commit": None,
            "git_worktree_dirty": None,
            "git_diff_sha256": None,
            "git_untracked_file_count": None,
            "git_untracked_sha256": None,
        }
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    diff = subprocess.run(
        ["git", "diff", "HEAD", "--binary"],
        cwd=repository,
        check=False,
        capture_output=True,
    )
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=repository,
        check=False,
        capture_output=True,
    )
    dirty = status.returncode == 0 and bool(status.stdout.strip())
    diff_hash = (
        hashlib.sha256(diff.stdout).hexdigest() if diff.returncode == 0 and diff.stdout else None
    )
    untracked_paths = sorted(path for path in untracked.stdout.split(b"\0") if path)
    untracked_hash = _untracked_digest(repository, untracked_paths) if untracked_paths else None
    return {
        "git_commit": revision.stdout.strip(),
        "git_worktree_dirty": dirty if status.returncode == 0 else None,
        "git_diff_sha256": diff_hash,
        "git_untracked_file_count": len(untracked_paths) if untracked.returncode == 0 else None,
        "git_untracked_sha256": untracked_hash if untracked.returncode == 0 else None,
    }


def _untracked_digest(repository: Path, relative_paths: list[bytes]) -> str:
    digest = hashlib.sha256()
    for relative_bytes in relative_paths:
        relative = os.fsdecode(relative_bytes)
        path = repository / relative
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        if path.is_symlink():
            content = os.fsencode(os.readlink(path))
            digest.update(b"symlink\0")
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
            continue
        digest.update(b"file\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def build_provenance(
    case: CaseSpec,
    *,
    dataset_fingerprint: str,
    split_hashes: dict[str, str],
    runtime_seconds: float,
    command_line: list[str] | None = None,
) -> dict[str, Any]:
    git_state = _git_state(Path.cwd())
    packages = {}
    for name in ["bio-ml-preflight", "numpy", "pandas", "scikit-learn", "scipy", "pyarrow"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        **git_state,
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
            if case.data.adapter in {"davis", "bbb_martins", "b3db_external"}
            else "locally governed holdout"
        ),
    }
