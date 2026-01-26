# HyperBench

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![project_license][license-shield]][license-url]
[![codecov](https://codecov.io/github/hypernetwork-research-group/hyperbench/graph/badge.svg?token=XE0TB5JMOS)](https://codecov.io/github/hypernetwork-research-group/hyperbench)

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About the project</a>
    </li>
    <li>
      <a href="#getting-started">Getting started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
            <ul>
                <li><a href="#sync-dependencies">Sync dependencies</a></li>
            </ul>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li>
        <a href="#contributing">Contributing</a>
        <ul>
            <li><a href="#pre-commit-hooks">Pre-commit hooks</a></li>
            <li><a href="#linter">Linter</a></li>
            <li><a href="#type-checker">Type checker</a></li>
            <li><a href="#tests">Tests</a></li>
            <li><a href="#utilities">Utilities</a></li>
        </ul>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About The Project

## Getting Started

### Prerequisites

WIP

### Installation

#### Sync dependencies

Use [uv](https://docs.astral.sh/uv/reference/cli/) to sync dependencies:

```bash
uv sync
```

## Usage

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### Pre-commit hooks

Run the following command to install the pre-commit hook:

```bash
uv sync

pre-commit install --config .github/hooks/.pre-commit-config.yaml --hook-type pre-commit --install-hooks --overwrite
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
uv run pytest --cov=hyperbench --cov-report=term-missing
# html report
uv run pytest --cov=hyperbench --cov-report=html
```

### Utilities

Before committing code, run the following command to ensure code quality:

```bash
uv pip uninstall . && \
uv sync && \
uv pip install -e . && \
uv run ruff format && \
uvx ty check && \
uv run pytest --cov=hyperbench --cov-report=term-missing
```

## License

WIP

## Contact

WIP

## Acknowledgments




[contributors-shield]: https://img.shields.io/github/contributors/hypernetwork-research-group/hyperbench.svg?style=for-the-badge
[contributors-url]: https://github.com/hypernetwork-research-group/hyperbench/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/hypernetwork-research-group/hyperbench.svg?style=for-the-badge
[forks-url]: https://github.com/hypernetwork-research-group/hyperbench/network/members
[stars-shield]: https://img.shields.io/github/stars/hypernetwork-research-group/hyperbench.svg?style=for-the-badge
[stars-url]: https://github.com/hypernetwork-research-group/hyperbench/stargazers
[issues-shield]: https://img.shields.io/github/issues/hypernetwork-research-group/hyperbench.svg?style=for-the-badge
[issues-url]: https://github.com/hypernetwork-research-group/hyperbench/issues
[license-shield]: https://img.shields.io/github/license/hypernetwork-research-group/hyperbench.svg?style=for-the-badge
[license-url]: https://github.com/hypernetwork-research-group/hyperbench/blob/master/LICENSE.txt
