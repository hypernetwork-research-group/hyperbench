<h1 align="center">
  <img
    src="docs/assets/horizontal.svg"
    alt="HyperTorch"
    width="760"
  >
</h1>

-----------------
# HyperTorch: A Python library for hypergraph learning and reproducible benchmarking

<p align="center">
  <a href="https://hypernetwork-research-group.github.io/hypertorch/">Documentation</a>
  |
  <a href="https://hypernetwork-research-group.github.io/hypertorch/getting-started/overview">Getting started</a>
  |
  <a href="https://hypernetwork-research-group.github.io/hypertorch/getting-started/tutorials">Tutorial</a>
  |
  <a href="https://hypernetwork-research-group.github.io/hypertorch/development/contribution">Contributing</a>
</p>

| Helpful info and tracking | |
| --- | --- |
| Package | [![Release][release-shield]][release-url] [![License: Apache 2.0][license-shield]][license-url] [![Python][python-shield]][python-url] [![Documentation][docs-shield]][docs-url] |
| Testing | ![Integration tests][daily-ci-shield] ![Released version][weekly-ci-shield] [![Unit tests][unit-testing-shield]][unit-testing-url] |
| Quality assurance | [![codecov][codecov-shield]][codecov-url] [![CodeFactor][codefactor-shield]][codefactor-url] [![Issues][issues-shield]][issues-url] |
| Meta | [![Stargazers][stars-shield]][stars-url] [![Forks][forks-shield]][forks-url] [![Contributors][contributors-shield]][contributors-url] [![Discord][discord-shield]][discord-url] |

## About the project

HyperTorch is a library for hypergraph learning and benchmarking. It provides a standardized workflow for loading hypergraph datasets, training models, evaluating them under comparable settings, and reporting results. The current release focuses on Hyperlink Prediction, with ready-to-run pipelines for established hypergraph baselines.

The library is built around extensibility: datasets are represented in [HIF](https://github.com/HIF-org/HIF-standard) format and converted into typed tensor objects, models can be implemented as standard Lightning modules, and benchmarking is handled through reusable trainers, samplers, metrics, loggers, and result exporters (Markdown/LaTeX). HyperTorch includes preloaded datasets, mini-batch and full-hypergraph data loading, negative sampling utilities, structural feature enrichers, neural components, and built-in models such as HGNN, HNHN, HyperGCN, GCN, MLP/SLP, NHP, Node2Vec, VilLain, and more.

Use HyperTorch to:
- Benchmark existing models across a shared collection of hypergraph datasets.
- Develop custom PyTorch or PyTorch Lightning models and train and compare them against the built-in baselines.
- Integrate new datasets through the HIF format and run the same training, evaluation, and reporting pipeline on them.

## Table of contents

- [Main features](#main-features)
- [Getting started](#getting-started)
    - [Installation](#installation)
    - [Source installation](#source-installation)
    - [TensorBoard support](#tensorboard-support)
    - [Run examples](#run-examples)
- [Contributing](#contributing)
- [Documentation](#documentation)
- [License](#license)
- [Discussion](#discussion)

## Main features

| Feature | What you can do | Highlights | Location |
| :--- | :--- | :--- | :--- |
| **Dataset management** | Load, process, and validate hypergraph datasets | HIF loader/processor, built-in datasets such as Algebra, Cora, Pubmed, DBLP, Amazon, and IMDB | `hypertorch.data` |
| **Splitting, sampling, and batching** | Prepare train/validation/test data and mini-batches | Dataset splitters, node and hyperedge samplers, negative samplers, data loaders | `hypertorch.data` |
| **Feature enrichment** | Enrich node and hyperedge features before training | Laplacian positional encodings, Node2Vec features, hyperedge weights and attributes | `hypertorch.data` |
| **Neural components (NN)** | Build models and pipelines | Layers, aggregators, losses, and activation/normalization helpers | `hypertorch.nn` |
| **Models** | Access hypergraph models | HGNN, HGNNP, HNHN, HyperGCN, GCN, MLP/SLP, NHP, Node2Vec, VilLain, CommonNeighbors | `hypertorch.models` |
| **Hyperlink prediction (HLP) pipelines** | Use ready-to-train hyperlink prediction modules | HLP modules with encoders, configs, losses, and stage metrics for multiple models | `hypertorch.hyperlink_prediction` |
| **Node classification (NC) pipelines** | Use ready-to-train node classification modules | NC modules with encoders, configs, losses, and stage metrics for multiple models | `hypertorch.node_classification` |
| **Training and benchmarking** | Train, compare, checkpoint, and report model runs | Multi-model trainer, schedulers, TensorBoard support, CSV/Markdown/LaTeX result tables | `hypertorch.train` |

## Getting started

### Installation

HyperTorch can be installed from PyPI when you want to use it as a dependency, or from source when you want to contribute or run the latest repository version.

CI pipelines validate CPU installs on Python 3.10 through 3.14 for Linux x86_64, Linux ARM/aarch64, macOS arm64, and Windows x64. Install the matching PyTorch and PyG wheels for your platform (e.g., CUDA) before installing HyperTorch.

For more detailed instructions, see the [installation guide](docs/getting-started/installation.md).

### Source installation

```bash
git clone https://github.com/hypernetwork-research-group/hypertorch.git
cd hypertorch

make setup
```

See the [installation guide](docs/getting-started/installation.md) for platform
notes and dependency ranges.

### TensorBoard support

To include TensorBoard support, also run HyperTorch install command with the TensorBoard extra:

```bash
uv pip install "hypertorch[tensorboard]"
```

When installing from source, run the command:

```bash
make setup-tensorboard
```

### Run examples

You can download the [examples](examples) directory and run the example scripts to get started.

With Python:

```bash
python3 examples/hyperlink_prediction/nhp.py
```

Or with `uv`:

```bash
uv run examples/hyperlink_prediction/nhp.py
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on contributing to the project.

## Documentation

You can find the extensive documentation [here][docs].

Alternatively, you can build the documentation locally with the following commands:

```bash
make docs

# With explicit commands
uv run zensical build --clean -f zensical.toml
uv run zensical serve -f zensical.toml -a 127.0.0.1:8000
```
and open the browser at http://localhost:8000 to access the documentation.

## License

This project is released under the Apache License 2.0 license. See [LICENSE](LICENSE).

## Discussion

Most development discussions take place on GitHub in this repo, via the [GitHub issue tracker][issues].

<!-- LINKS -->
[codecov-shield]: https://codecov.io/github/hypernetwork-research-group/hypertorch/graph/badge.svg?token=XE0TB5JMOS
[codecov-url]: https://codecov.io/github/hypernetwork-research-group/hypertorch
[codefactor-shield]: https://www.codefactor.io/repository/github/hypernetwork-research-group/hypertorch/badge
[codefactor-url]: https://www.codefactor.io/repository/github/hypernetwork-research-group/hypertorch
[contributors-shield]: https://img.shields.io/github/contributors/hypernetwork-research-group/hypertorch.svg?style=flat&label=Contributors
[contributors-url]: https://github.com/hypernetwork-research-group/hypertorch/graphs/contributors
[daily-ci-shield]: https://img.shields.io/github/actions/workflow/status/hypernetwork-research-group/hypertorch/daily_ci.yaml?branch=main&label=Integration%20tests
[discord-shield]: https://img.shields.io/discord/693092516286693387?style=flat&label=Discord
[discord-url]: https://discord.gg/4krTXCWRzD
[docs]: https://hypernetwork-research-group.github.io/hypertorch/
[docs-shield]: https://img.shields.io/badge/docs-latest-blue.svg?label=Documentation
[docs-url]: https://hypernetwork-research-group.github.io/hypertorch/
[forks-shield]: https://img.shields.io/github/forks/hypernetwork-research-group/hypertorch.svg?style=flat&label=Forks
[forks-url]: https://github.com/hypernetwork-research-group/hypertorch/network/members
[issues]: https://github.com/hypernetwork-research-group/hypertorch/issues
[issues-shield]: https://img.shields.io/github/issues/hypernetwork-research-group/hypertorch.svg?style=flat&label=Issues
[issues-url]: https://github.com/hypernetwork-research-group/hypertorch/issues
[license-shield]: https://img.shields.io/badge/License-Apache%202.0-yellow.svg
[license-url]: LICENSE
[python-shield]: https://img.shields.io/badge/python-3.10%2B-blue.svg?label=Python
[python-url]: https://www.python.org/downloads/
[release-shield]: https://img.shields.io/github/v/tag/hypernetwork-research-group/hypertorch?label=Release
[release-url]: https://github.com/hypernetwork-research-group/hypertorch/tags
[stars-shield]: https://img.shields.io/github/stars/hypernetwork-research-group/hypertorch.svg?style=flat&label=Stars
[stars-url]: https://github.com/hypernetwork-research-group/hypertorch/stargazers
[unit-testing-shield]: https://img.shields.io/github/actions/workflow/status/hypernetwork-research-group/hypertorch/coverage.yaml?branch=main&label=Unit%20tests
[unit-testing-url]: https://github.com/hypernetwork-research-group/hypertorch/actions/workflows/coverage.yaml
[weekly-ci-shield]: https://img.shields.io/github/actions/workflow/status/hypernetwork-research-group/hypertorch/weekly_ci.yaml?branch=main&label=Released%20version
