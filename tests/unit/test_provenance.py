import subprocess
from pathlib import Path

from bio_ml_preflight.provenance.record import _git_state


def test_git_state_hashes_untracked_file_content(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "--quiet",
            "--allow-empty",
            "-m",
            "initial",
        ],
        cwd=tmp_path,
        check=True,
    )
    assert _git_state(tmp_path)["git_worktree_dirty"] is False

    untracked = tmp_path / "untracked.txt"
    untracked.write_text("first", encoding="utf-8")
    first = _git_state(tmp_path)
    untracked.write_text("second", encoding="utf-8")
    second = _git_state(tmp_path)

    assert first["git_worktree_dirty"] is True
    assert first["git_diff_sha256"] is None
    assert first["git_untracked_file_count"] == 1
    assert first["git_untracked_sha256"] != second["git_untracked_sha256"]
