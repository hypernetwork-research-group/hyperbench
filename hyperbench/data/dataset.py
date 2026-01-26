"""Example usage of the Hypergraph class with HIF data."""

import json
import os
import gdown
import tempfile
import torch
import zstandard as zstd

from enum import Enum
from typing import List, Tuple
from torch import Tensor
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
    GDRIVE_FILE_ID = None
    DATASET_NAME = None

    def __init__(self) -> None:
        self.hypergraph: HIFHypergraph = self.download()
        self.hdata: HData = self.process()

    def __len__(self) -> int:
        return len(self.hypergraph.nodes)

    def __getitem__(self, index: int | List[int]) -> HData:
        sampled_node_ids_list = self.__get_node_ids_to_sample(index)
        self.__validate_node_ids(sampled_node_ids_list)

        sampled_edge_index, sampled_node_ids, sampled_edge_ids = (
            self.__sample_edge_index(sampled_node_ids_list)
        )

        new_edge_index = self.__new_edge_index(
            sampled_edge_index, sampled_node_ids, sampled_edge_ids
        )

        new_node_features = self.hdata.x[sampled_node_ids]

        new_edge_attr = None
        if self.hdata.edge_attr is not None and len(sampled_edge_ids) > 0:
            new_edge_attr = self.hdata.edge_attr[sampled_edge_ids]

        return HData(
            x=new_node_features,
            edge_index=new_edge_index,
            edge_attr=new_edge_attr,
            num_nodes=len(sampled_node_ids),
            num_edges=len(sampled_edge_ids),
        )

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

    def __get_node_ids_to_sample(self, id: int | List[int]) -> List[int]:
        if isinstance(id, int):
            return [id]

        if isinstance(id, list):
            if len(id) < 1:
                raise ValueError("Index list cannot be empty.")
            elif len(id) > self.__len__():
                raise ValueError(
                    "Index list length cannot exceed number of nodes in the hypergraph."
                )
            return list(set(id))

    def __validate_node_ids(self, node_ids: List[int]) -> None:
        for id in node_ids:
            if id < 0 or id >= self.__len__():
                raise IndexError(
                    f"Node ID {id} is out of bounds (0, {self.__len__() - 1})."
                )

    def __sample_edge_index(
        self,
        sampled_node_ids_list: List[int],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        edge_index = self.hdata.edge_index
        node_ids = edge_index[0]
        edge_ids = edge_index[1]

        sampled_node_ids = torch.tensor(sampled_node_ids_list)

        # Find incidences where the node is in our sampled node set
        # Example: edge_index[0] = [0, 0, 1, 2, 3, 4], sampled_node_ids = [0, 3]
        #          -> incidence_mask = [True, True, False, False, True, False]
        node_incidence_mask = torch.isin(node_ids, sampled_node_ids)

        # Get unique hyperedges that have at least one sampled node
        # Example: edge_index[1] = [0, 0, 0, 1, 2, 2], node_incidence_mask = [False, False, True, True, False, False]
        #          -> sampled_edge_ids = [0, 1] as they connect to sampled nodes
        sampled_edge_ids = edge_ids[node_incidence_mask].unique()

        # Find all incidences for sampled nodes belonging to relevant hyperedges
        # Example: edge_index[1] = [0, 0, 0, 1, 2, 2], sampled_edge_ids = [0, 1]
        #          -> edge_incidence_mask = [True, True, True, True, True, False]
        edge_incidence_mask = torch.isin(edge_ids, sampled_edge_ids)

        # Incidence is kept if node is sampled AND hyperedge is relevant
        incidence_mask = node_incidence_mask & edge_incidence_mask

        # Keep only the incidences that match our mask
        # Example: edge_index = [[0, 0, 1, 2, 3, 4],
        #                        [0, 0, 0, 1, 1, 2]]
        #          incidence_mask = [False, False, True, False, True, False]
        #          -> sampled_edge_index = [[1, 3],
        #                                   [0, 1]]
        sampled_edge_index = edge_index[:, incidence_mask]

        return sampled_edge_index, sampled_node_ids, sampled_edge_ids

    def __new_edge_index(
        self,
        sampled_edge_index: Tensor,
        sampled_node_ids: Tensor,
        sampled_edge_ids: Tensor,
    ) -> Tensor:
        """
        Create new edge_index with 0-based node and edge IDs.
        Args:
            sampled_edge_index: Original edge_index tensor with sampled incidences.
            sampled_node_ids: List of sampled original node IDs.
            sampled_edge_ids: List of sampled original edge IDs.
        Returns:
            New edge_index tensor with 0-based node and edge IDs.
        """
        new_node_ids = self.__to_0based_ids(
            sampled_edge_index[0], sampled_node_ids, self.hdata.num_nodes
        )
        new_edge_ids = self.__to_0based_ids(
            sampled_edge_index[1], sampled_edge_ids, self.hdata.num_edges
        )

        # Example: new_node_ids = [0, 1], new_edge_ids = [0, 1]
        #         -> new_edge_index = [[0, 1],
        #                              [0, 1]]
        new_edge_index = torch.stack([new_node_ids, new_edge_ids], dim=0)
        return new_edge_index

    def __to_0based_ids(
        self,
        original_ids: Tensor,
        ids_to_keep: Tensor,
        n: int,
    ) -> Tensor:
        """
        Map original IDs to 0-based ids.
        Example:
            original_ids: [1, 3]
            ids_to_keep: [3, 7]
            n = 6
            Returned 0-based IDs: [0, 1]
        Args:
            original_ids: Tensor of original IDs.
            n: Total number of original IDs.
            ids_to_keep: List of selected original IDs to be mapped to 0-based.
        Returns:
            Tensor of 0-based ids.
        """
        id_to_0based_id = torch.zeros(n)
        n_ids_to_keep = len(ids_to_keep)
        id_to_0based_id[ids_to_keep] = torch.arange(n_ids_to_keep)
        return id_to_0based_id[original_ids]


class AlgebraDataset(Dataset):
    DATASET_NAME = "ALGEBRA"
    GDRIVE_FILE_ID = "1-H21_mZTcbbae4U_yM3xzXX19VhbCZ9C"
