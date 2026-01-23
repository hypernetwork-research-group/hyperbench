"""Example usage of the Hypergraph class with HIF data."""

import json
import gdown
import tempfile
import torch

from enum import Enum
from torch.utils.data import Dataset as TorchDataset
from hyperbench.types.hypergraph import HIFHypergraph
from hyperbench.types.hdata import HData
from hyperbench.utils.hif_utils import validate_hif_json


class DatasetNames(Enum):
    """
    Enumeration of available datasets.
    """

    ALGEBRA = "1"
    EMAIL_ENRON = "2"
    ARXIV = "3"


class HIFConverter:
    """
    Docstring for HIFConverter
    A utility class to load hypergraphs from HIF format.
    """

    @staticmethod
    def load_from_hif(dataset_name: str | None, file_id: str | None) -> HIFHypergraph:
        if dataset_name is None or file_id is None:
            raise ValueError(
                f"Dataset name (provided: {dataset_name}) and file ID (provided: {file_id}) must be provided."
            )
        if dataset_name not in DatasetNames.__members__:
            raise ValueError(f"Dataset '{dataset_name}' not found.")

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
    """
    Base Dataset class for hypergraph datasets, extending PyTorch's Dataset.
    Attributes:
        GDRIVE_FILE_ID (str): Google Drive file ID for the dataset.
        DATASET_NAME (str): Name of the dataset.
        hypergraph (HIFHypergraph): Loaded hypergraph instance.
    Methods:
        download(): Downloads and loads the hypergraph from HIF.
        process(): Processes the hypergraph into HData format.
    """

    # TODO: move as input to __init__()? So that users can provide new ids and names of datasets formatted in HIF
    GDRIVE_FILE_ID = None
    DATASET_NAME = None

    def __init__(self) -> None:
        self.hypergraph: HIFHypergraph = self.download()
        self.hdata: HData = self.process()

    def __len__(self) -> int:
        return len(self.hypergraph.nodes)

    def __getitem__(self, index: int) -> HData:
        # TODO: implement sampling of nodes with given index
        return self.hdata

    def download(self) -> HIFHypergraph:
        """
        Load the hypergraph from HIF format using HIFConverter class.
        """
        hypergraph = HIFConverter.load_from_hif(self.DATASET_NAME, self.GDRIVE_FILE_ID)
        return hypergraph

    def process(self) -> HData:
        """
        Process the loaded hypergraph into HData format, mapping HIF structure to tensors.
        Returns:
            HData: Processed hypergraph data.
        """
        if self.hypergraph is None:
            raise ValueError("Hypergraph is not loaded. Call download() first.")

        num_nodes = len(self.hypergraph.nodes)
        num_edges = len(self.hypergraph.edges)

        x = torch.arange(num_nodes).unsqueeze(1)

        node_ids = []
        edge_ids = []
        for incidence in self.hypergraph.incidences:
            node_id = int(incidence.get("node", 0))
            edge_id = int(incidence.get("edge", 0))
            node_ids.append(node_id)
            edge_ids.append(edge_id)

        edge_index = None
        if len(node_ids) < 1:
            raise ValueError("Hypergraph has no incidences.")

        # edge_index: shape [2, E] where E is number of incidences
        # First row: node IDs, Second row: hyperedge IDs
        edge_index = torch.tensor([node_ids, edge_ids])

        edge_attr = None
        if self.hypergraph.edges and any(
            "attrs" in edge for edge in self.hypergraph.edges
        ):
            edge_attrs = []
            for edge in self.hypergraph.edges:
                attrs = edge.get("attrs", {})
                edge_attrs.append(len(attrs))
            edge_attr = torch.tensor(edge_attrs).unsqueeze(1)

        return HData(x, edge_index, edge_attr, num_nodes, num_edges)


class AlgebraDataset(Dataset):
    DATASET_NAME = "ALGEBRA"
    GDRIVE_FILE_ID = "1-H21_mZTcbbae4U_yM3xzXX19VhbCZ9C"

    def __init__(self) -> None:
        super().__init__()
