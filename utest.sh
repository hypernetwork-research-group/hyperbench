#!/usr/bin/env bash

command -v uv >/dev/null 2>&1 || { echo "uv command not found; please install uv" >&2; exit 1; }

uv pip uninstall .
uv sync
uv pip install -e .

uv run ruff format
uvx ty check

uv run pytest --cov=hyperbench --cov-report=term-missing
