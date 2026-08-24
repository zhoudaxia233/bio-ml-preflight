UV_PROJECT_ENVIRONMENT ?= venv
UV_CACHE_DIR ?= .uv-cache
MPLCONFIGDIR ?= .matplotlib-cache
PYTHON ?= 3.11
export UV_PROJECT_ENVIRONMENT UV_CACHE_DIR MPLCONFIGDIR

.PHONY: demo demo-bbb demo-davis check

demo:
	uv run --python $(PYTHON) bio-ml-preflight demo synthetic --budget smoke

demo-bbb:
	uv run --python $(PYTHON) --all-extras bio-ml-preflight demo bbb --budget smoke

demo-davis:
	uv run --python $(PYTHON) --all-extras bio-ml-preflight demo davis --budget smoke

check:
	uv run --python $(PYTHON) --extra dev ruff check .
	uv run --python $(PYTHON) --extra dev ruff format --check .
	uv run --python $(PYTHON) --extra dev mypy src
	uv run --python $(PYTHON) --extra dev pytest
