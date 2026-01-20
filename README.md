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
