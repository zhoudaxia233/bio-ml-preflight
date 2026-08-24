from pathlib import Path

from bio_ml_preflight.cli import _project_root


def test_project_root_uses_checkout_from_current_directory(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "pyproject.toml").touch()
    (tmp_path / "examples").mkdir()
    monkeypatch.chdir(tmp_path)
    assert _project_root() == tmp_path
