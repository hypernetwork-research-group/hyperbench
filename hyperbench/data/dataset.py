"""Example usage of the Hypergraph class with HIF data."""

import json
import os
import gdown
import tempfile
import torch
import zstandard as zstd

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

        dataset_name_lower = dataset_name.lower()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        zst_filename = os.path.join(
            current_dir, "datasets", f"{dataset_name_lower}.json.zst"
        )

        if os.path.exists(zst_filename):
            dctx = zstd.ZstdDecompressor()
            with (
                open(zst_filename, "rb") as input_f,
                tempfile.NamedTemporaryFile(
                    mode="wb", suffix=".json", delete=False
                ) as tmp_file,
            ):
                dctx.copy_stream(input_f, tmp_file)
                output = tmp_file.name
        else:
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
        if hasattr(self, "hypergraph") and self.hypergraph is not None:
            return self.hypergraph
        hypergraph = HIFConverter.load_from_hif(self.DATASET_NAME, self.GDRIVE_FILE_ID)
        return hypergraph

    def process(self) -> HData:
        """
        Process the loaded hypergraph into HData format, mapping HIF structure to tensors.
        Returns:
            HData: Processed hypergraph data.
        """

        num_nodes = len(self.hypergraph.nodes)
        num_edges = len(self.hypergraph.edges)

        x = torch.arange(num_nodes).unsqueeze(1)

        node_set = []
        edge_set = []
        incidences_tuples = []

        for inc in self.hypergraph.incidences:
            node = inc.get("node", 0)
            edge = inc.get("edge", 0)
            if node not in node_set:
                node_set.append(node)
            if edge not in edge_set:
                edge_set.append(edge)
            incidences_tuples.append((node, edge))

        node_id_mapping = {node_id: idx for idx, node_id in enumerate(node_set)}
        edge_id_mapping = {edge_id: idx for idx, edge_id in enumerate(edge_set)}

        node_ids = [node_id_mapping[node] for node, _ in incidences_tuples]
        edge_ids = [edge_id_mapping[edge] for _, edge in incidences_tuples]

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
