# Benchmarking

Benchmarking in HyperTorch typically means:
- Running multiple models on the same dataset split.
- Using the same negative sampling and feature enrichment.
- Producing comparable metrics and summary tables.

## Running the benchmark suite

In the benchmark folder, we provide a `bench_hyperlink_prediction.py` script that runs multiple
models on a given dataset. The script is designed to be run from the command line and accepts
various arguments to customize the benchmarking process.

```bash
bash benchmark/bench.sh hlp -- \
    --datasets citeseer cora pubmed \
    --k-nodes 2 \
    --num-workers 4 \
    --num-features 16 \
    --run 3 \
    --seed 1 2 3 \
    --split-ratios 0.7 0.1 0.2 \
    --test-set-negative-ratio 0.5
```

You can specify:
- `--datasets`: List of datasets to benchmark.
- `--k-nodes`: Number of nodes for negative sampling.
- `--num-workers`: Number of workers for data loading.
- `--num-features`: Number of features for the model.
- `--run`: Number of runs for each model.
- `--seed`: Random seeds used for dataset preparation and PyTorch's seed.
- `--split-ratios`: Ratios for train, validation, and test splits.
- `--task`: Task type. The launcher defaults to `hyperlink-prediction` for `hlp` and
  `node-classification` for `nc`.
- `--test-set-negative-ratio`: Ratio of negative samples in the test set.

## Comparing multiple models

The recommended pattern is to pass multiple `ModelConfig` objects to `MultiModelTrainer`:

```python
from hypertorch.types import ModelConfig
from hypertorch.train import MultiModelTrainer
from hypertorch.hyperlink_prediction import MLPPredictor, NHPPredictor

configs = [
    ModelConfig(
        name="nhp",
        version="maxmin",
        model=NHPPredictor(
            encoder_config={
                "in_channels": 32,
                "hidden_channels": 64,
                "aggregation": "maxmin",
            },
        ),
    ),
    ModelConfig(
        name="mlp",
        version="mean",
        model=MLPPredictor(
            encoder_config={
                "in_channels": 32,
                "out_channels": 32,
                "hidden_channels": 64,
                "num_layers": 3,
                "drop_rate": 0.3,
            },
            aggregation="mean",
        ),
    ),
]

with MultiModelTrainer(model_configs=configs, max_epochs=200, accelerator="auto") as trainer:
    trainer.fit_all(train_dataloader=train_loader, val_dataloader=val_loader)
    trainer.test_all(dataloader=test_loader)
```

## Where results are saved

By default, runs are saved under `hypertorch_logs/`.

The trainer writes comparison tables to:
- `hypertorch_logs/experiment_*/comparison/{overall, test, train, val}.md`.
- `hypertorch_logs/experiment_*/comparison/{overall, test, train, val}.tex`.

## Next steps

- Outputs and logging: [Loggers](loggers.md).
- Visualizing runs: [TensorBoard](tensorboard.md).
