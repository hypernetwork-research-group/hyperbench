# Contributing to Hyperbench

Thank you for your interest in contributing to Hyperbench!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/hyperbench.git`
3. Create a branch: `git checkout -b feat/your-feature-name`

## Development Setup

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
bash utest.sh

# Run type checking
mypy hyperbench
```

## Commit Message Style

Commit messages should follow the [conventional commit specification](https://www.conventionalcommits.org/en/v1.0.0/).

The allowed structural elements are:
- `feat` for new features
- `fix` for bug fixes
- `chore` for build process or tooling changes
- `refactor` for code restructuring
- `docs` for documentation changes

Example: `feat: add support for weighted hypergraphs`

## Branch Naming

Branch names should be descriptive and use hyphens:
- `feat/add-user-authentication`
- `fix/issue-with-database-connection`
- `chore/update-dependencies`
- `refactor/improve-code-structure`
- `docs/update-contributing-guidelines`

## Pull Request Process

1. Ensure all tests pass
2. Update documentation as needed
3. Add tests for new features
4. Follow the existing code style
5. Write clear commit messages

## Code Style

- Follow PEP 8
- Use type hints
- Write docstrings for all public APIs
- Keep functions focused and small

## Testing

Add tests for any new functionality in the appropriate test file under `hyperbench/tests/`.

```bash
# Run all tests with coverage
pytest --cov=hyperbench

# Run specific test file
pytest hyperbench/tests/data/dataset_test.py
```

## Documentation

Update documentation when adding new features or changing APIs. Documentation files are in the `docs/` directory.

## Questions?

Open an issue on GitHub if you have questions or need help.
