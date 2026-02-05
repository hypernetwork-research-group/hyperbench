# Installation

## Prerequisites

- [uv](https://github.com/astral-sh/uv)
- [make](https://www.gnu.org/software/make/)

## Build
To build the project, run:
```bash
make
```

### Linter and type checker

Use [Ruff](https://github.com/charliermarsh/ruff) for linting and formatting:

```bash
make lint
```

Use [Ty](https://docs.astral.sh/ty/) for type checking:

```bash
make typecheck
```

Use the `check` target to run both linter and type checker:

```bash
make check
```

### Tests

Use [pytest](https://docs.pytest.org/en/latest/) to run the test suite:

```bash
make test

# Run tests with HTML report
uv run pytest --cov=hyperbench --cov-report=html
```

### Pre-commit hooks

Run the following command to install the pre-commit hook:

```bash
make setup

pre-commit install --config .github/hooks/.pre-commit-config.yaml --hook-type pre-commit --install-hooks --overwrite
```

This will ensure that your code adheres to the project's coding standards before each commit.
