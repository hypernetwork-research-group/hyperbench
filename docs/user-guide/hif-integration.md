# HIF integration

HyperTorch uses [**HIF (Hypergraph Interchange Format)**](https://github.com/HIF-org/HIF-standard) to represent hypergraphs.

Supported inputs:
- `.json` (plain HIF).
- `.json.zst` (Zstandard-compressed HIF).

## Load built-in datasets

Many datasets are available as built-ins (downloaded and cached automatically):

```python
from hypertorch.data import AlgebraDataset, SamplingStrategy

dataset = AlgebraDataset(sampling_strategy=SamplingStrategy.HYPEREDGE)
print(dataset.stats())
```

Built-in dataset classes include `AlgebraDataset`, `AmazonDataset`, `CoraDataset`, `CourseraDataset`, `IMDBDataset`, and more. See the [Data API reference](../api/data.md) for the complete list.

## Load a dataset from a local file

```python
from hypertorch.data import Dataset

dataset = Dataset.from_path("path/to/hypergraph.json.zst")
print(dataset.stats())
```

## Load a dataset from a URL

```python
from hypertorch.data import Dataset

dataset = Dataset.from_url("https://example.com/hypergraph.json.zst")
print(dataset.stats())
```

### Validating HIF

Before loading, you can also check that a plain `.json` file conforms to the HIF schema:

```python
from hypertorch.utils import validate_hif_json

is_valid = validate_hif_json("path/to/hypergraph.json")
print(is_valid)
```

## How HIF maps into HyperTorch

When loaded, HIF data is processed into an `HData` object (see [HData API reference](../api/types.md#hypertorch.types.HData) for details).

## HIF keywords and their effect on processing

HIF reserves a few keywords that HyperTorch interprets specially when converting a hypergraph into tensors. The two most important ones are `weight` and `label`.

### `weight` keyword

The `weight` keyword, placed inside a hyperedge's `attrs` dictionary, controls the **hyperedge weights**:

```json
{
  "edges": [
    { "edge": "e1", "attrs": { "weight": 2.5 } },
    { "edge": "e2", "attrs": {} }
  ]
}
```

How it influences processing:

- HyperTorch reads `attrs["weight"]` for every hyperedge and stores the result in the `hyperedge_weights` tensor of shape `[num_hyperedges]` on the `HData` object.
- Hyperedges that do not declare a `weight` default to `1.0`. This also applies to the self-loop hyperedges that HyperTorch creates for isolated nodes.
- Because `weight` is a numeric attribute, it is **also** collected as a column of the `hyperedge_attr` matrix. If you do not want the weight to double as a hyperedge feature, keep it out of `attrs` or account for it when consuming `hyperedge_attr`.

### `label` keyword

The `label` keyword, placed on a node's `attrs`, provides the supervised target for node-related tasks (e.g., node classification):

```json
{
  "nodes": [
    { "node": 0, "attrs": { "label": "cat" } },
    { "node": 1, "attrs": { "label": "dog" } }
  ]
}
```

How it influences processing:

- For node-related tasks, HyperTorch reads `attrs["label"]` from every node and builds the `y` label tensor of shape `[num_nodes]`, mapping each distinct label to a numeric index.
- The mapping from label strings to numeric indices is stored in `hif_hypergraph.metadata["label_map"]` and can be reversed with `Dataset.to_human_readable_y(...)`.
- `label` is **excluded** from the node feature matrix `x` (it is a target, not a feature). Other numeric attributes are kept as features.
- If only some nodes declare a `label`, processing raises a `ValueError`. If no node declares a `label`, `y` is left as `None`.

## Using HIF

HIF is the interchange format HyperTorch uses to represent hypergraphs, so it is the common entry point for loading data. The typical workflow is:

1. **Load** a HIF file from a built-in dataset, a local path, or a URL.
2. **Process**  the loader parses the HIF structure into an `HIFHypergraph` and converts it into an `HData` object of tensors ready for training.
3. **Use** pass the resulting `Dataset` to a trainer, sampler, or model.

```python
from hypertorch.data import Dataset

dataset = Dataset.from_path("path/to/hypergraph.json.zst")
print(dataset.stats())
```

### Choosing the learning task

The `task` argument controls how HIF is interpreted. For node-related tasks (e.g., `TaskEnum.NODE_CLASSIFICATION`), HyperTorch extracts node `label` attributes into `y` and builds the `label_map`. For hyperedge-related tasks (the default, hyperlink prediction), node labels are ignored and `y` is left as `None`.

```python
from hypertorch.data import Dataset
from hypertorch.types import TaskEnum

dataset = Dataset.from_path(
    "path/to/hypergraph.json.zst",
    task=TaskEnum.NODE_CLASSIFICATION,
)
```

### Accessing the original HIF structure

The `Dataset` keeps a reference to the original `HIFHypergraph`, which you can inspect through the `hif_hypergraph` property. This is useful for reading metadata, the `label_map`, or the raw node/hyperedge attributes after processing.

```python
dataset = Dataset.from_path("path/to/hypergraph.json.zst")
hif = dataset.hif_hypergraph
print(hif.metadata)
```



## Next steps

- Model selection/customization: [Models](models.md).
- Training loop (callbacks, devices, etc.): [Training](training.md).
- Comparing multiple models consistently: [Benchmarking](benchmarking.md).
- Outputs and logging: [Loggers](loggers.md).
- Visualizing runs: [TensorBoard](tensorboard.md).
