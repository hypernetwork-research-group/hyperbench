"""Example usage of the Hypergraph class with HIF data."""

import json
import gdown
import tempfile

from typing import Any
from enum import Enum
from torch.utils.data import Dataset as TorchDataset
from hyperbench.types.hypergraph import HIFHypergraph
from hyperbench.utils.hif import validate_hif_json


class DatasetNames(Enum):
    """
    Enumeration of available datasets with their Google Drive file IDs.
    """

    ALGEBRA = "1-H21_mZTcbbae4U_yM3xzXX19VhbCZ9C"
    EMAIL_ENRON = "placeholder"
    ARXIV = "placeholder"


class HIFConverter:
    @staticmethod
    def get_dataset_from_hif(dataset_name: str) -> HIFHypergraph:
        """Fetches and returns the HIF hypergraph for the specified dataset.

        Args:
            dataset_name: Name of the dataset to fetch.
        Returns:
            HIFHypergraph: The hypergraph representation of the dataset.
        """

        if dataset_name not in DatasetNames.__members__:
            raise ValueError(f"Dataset '{dataset_name}' not found.")

        file_id = DatasetNames[dataset_name].value
        url = f"https://drive.google.com/uc?id={file_id}"

        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".json", delete=False
        ) as tmp_file:
            output = tmp_file.name
            gdown.download(url=url, output=output, quiet=False, fuzzy=True)

        with open(output, "r") as f:
            hiftext = json.load(f)
        if not validate_hif_json(output):
            raise ValueError(f"Dataset '{dataset_name}' is not HIF-compliant.")

        hypergraph = HIFHypergraph.from_hif(hiftext)
        return hypergraph


class Dataset(TorchDataset):
    def __init__(self) -> None:
        pass

    def __len__(self) -> int:
        pass

    def __getitem__(self, index: int) -> Any:
        pass
