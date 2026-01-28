# HyperBench

[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Contributors][contributors-shield]][contributors-url]

[![Issues][issues-shield]][issues-url]
[![project_license][license-shield]][license-url]

[![codecov](https://codecov.io/github/hypernetwork-research-group/hyperbench/graph/badge.svg?token=XE0TB5JMOS)](https://codecov.io/github/hypernetwork-research-group/hyperbench)

For documentation, please visit [here][docs].

### Installation

WIP

## Usage

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on contributing to the project.

### Build

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

## License

WIP

## Contact

WIP

<!-- LINKS -->
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
[docs]: https://hypernetwork-research-group.github.io/hyperbench/
