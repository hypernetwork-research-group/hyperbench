# HyperBench

## Commands

### Sync dependencies

Use [uv](https://docs.astral.sh/uv/reference/cli/) to sync dependencies:

```bash
uv sync
```

### Linter

Use [Ruff](https://github.com/charliermarsh/ruff) for linting and formatting:

```bash
uvx ruff check

uvx ruff format
```

### Type checker

Use [Ty](https://docs.astral.sh/ty/) for type checking:

```bash
uvx ty check

# In watch mode
uvx ty check --watch
```

### Tests

Run tests with [pytest](https://docs.pytest.org/en/latest/):

```bash
uv run pytest tests/ --cov=hyperbench --cov-report=term-missing 
# html report
uv run pytest tests/ --cov=hyperbench --cov-report=html
```


### Utilities

Before committing code, run the following command to ensure code quality:

```bash
uv sync && \
uv pip install -e . && \
uv run ruff format && \
uvx ty check && \
uv run pytest tests/ --cov=hyperbench --cov-report=term-missing
```
