"""Example usage of the Hypergraph class with HIF data."""

import json
import os
import gdown
import tempfile
import torch
import zstandard as zstd

from enum import Enum
from typing import Any, Dict, List, Tuple
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
        if not isinstance(index, (int, list)):
            raise TypeError("Index must be an integer or a list of integers.")
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

        # x: shape [num_nodes, num_node_features]
        all_attr_keys = set()
        # collect unique numeric attribute keys across all nodes
        for node in self.hypergraph.nodes:
            attrs = node.get("attrs", {})
            numeric_attrs = {
                key: value
                for key, value in attrs.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            all_attr_keys.update(numeric_attrs.keys())

        sorted_attr_keys = sorted(all_attr_keys) if all_attr_keys else None

        # Encode node attributes, using 0.0 when missing
        node_attrs_list = []
        for node in self.hypergraph.nodes:
            attrs = node.get("attrs", {})
            # encode_node_attrs can be overridden in subclasses for custom behavior of the dataset
            feature_vector = self.encode_node_attrs(attrs, sorted_attr_keys)
            node_attrs_list.append(feature_vector)

        # Check if all nodes have attributes
        if node_attrs_list and any(len(na) > 0 for na in node_attrs_list):
            x = torch.stack(node_attrs_list)
        else:
            # Fallback to node indices if no numeric attributes
            x = torch.arange(num_nodes, dtype=torch.float).unsqueeze(1)

        # remap node and edge IDs to 0-based contiguous IDs
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

        # edge-attr: shape [num_edges, num_edge_attributes]
        # single attribute per edge is flattened to shape [num_edges]
        # multiple attributes per edge is shape [num_edges, num_attributes]
        edge_attr = None
        if self.hypergraph.edges and any(
            "attrs" in edge for edge in self.hypergraph.edges
        ):
            all_edge_attr_keys = set()
            for edge in self.hypergraph.edges:
                attrs = edge.get("attrs", {})
                numeric_attrs = {
                    key: value
                    for key, value in attrs.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                }
                all_edge_attr_keys.update(numeric_attrs.keys())

            sorted_edge_attr_keys = (
                sorted(all_edge_attr_keys) if all_edge_attr_keys else None
            )

            edge_attrs = []
            for edge in self.hypergraph.edges:
                attrs = edge.get("attrs", {})
                # Use encode_edge_attrs for extensibility (subclasses can override)
                feature_vector = self.encode_edge_attrs(attrs, sorted_edge_attr_keys)
                edge_attrs.append(feature_vector)

            # Check if any edges have attributes
            if edge_attrs and any(len(ea) > 0 for ea in edge_attrs):
                edge_attr = torch.stack(edge_attrs)

                # Flatten to 1D if only one attribute (PyTorch Geometric standard)
                if edge_attr.shape[1] == 1:
                    edge_attr = edge_attr.squeeze(1)

        return HData(x, edge_index, edge_attr, num_nodes, num_edges)

    def encode_node_attrs(
        self, attrs: Dict[str, Any], attr_keys: List[str] | None = None
    ) -> Tensor:
        """
        Extract and encode numeric node attributes to tensor.
        Non-numeric attributes are discarded. Missing attributes are filled with 0.0.

        Args:
            attrs: Dictionary of node attributes
            attr_keys: Optional list of attribute keys to encode. If provided, ensures
                      consistent ordering and fill missing with 0.0.

        Returns:
            Tensor of numeric attribute values (sorted by key for consistency)
        """
        numeric_attrs = {
            key: value
            for key, value in attrs.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }

        if attr_keys is not None:
            values = [float(numeric_attrs.get(key, 0.0)) for key in attr_keys]
            return torch.tensor(values, dtype=torch.float)

        if not numeric_attrs:
            return torch.tensor([], dtype=torch.float)

        sorted_keys = sorted(numeric_attrs.keys())
        values = [float(numeric_attrs[key]) for key in sorted_keys]

        return torch.tensor(values, dtype=torch.float)

    def encode_edge_attrs(
        self, attrs: Dict[str, Any], attr_keys: List[str] | None = None
    ) -> Tensor:
        """
        Extract and encode numeric edge attributes to tensor.
        Non-numeric attributes are discarded. Missing attributes are filled with 0.0.

        Args:
            attrs: Dictionary of edge attributes
            attr_keys: Optional list of attribute keys to encode. If provided, ensures
                      consistent ordering and fill missing with 0.0.

        Returns:
            Tensor of numeric attribute values (sorted by key for consistency)
        """
        numeric_attrs = {
            key: value
            for key, value in attrs.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }

        if attr_keys is not None:
            values = [float(numeric_attrs.get(key, 0.0)) for key in attr_keys]
            return torch.tensor(values, dtype=torch.float)

        if not numeric_attrs:
            return torch.tensor([], dtype=torch.float)

        sorted_keys = sorted(numeric_attrs.keys())
        values = [float(numeric_attrs[key]) for key in sorted_keys]

        return torch.tensor(values, dtype=torch.float)

    def __get_node_ids_to_sample(self, id: int | List[int]) -> List[int]:
        if isinstance(id, int):
            return [id]
        else:  # list of ints
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
        #          -> node_incidence_mask = [True, True, False, False, True, False]
        node_incidence_mask = torch.isin(node_ids, sampled_node_ids)

        # Get unique hyperedges that have at least one sampled node
        # Example: edge_index[1] = [0, 0, 0, 1, 2, 2], node_incidence_mask = [True, True, False, False, True, False]
        #          -> sampled_edge_ids = [0, 2] as they connect to sampled nodes
        sampled_edge_ids = edge_ids[node_incidence_mask].unique()

        # Find all incidences for sampled nodes belonging to relevant hyperedges
        # Example: edge_index[1] = [0, 0, 0, 1, 2, 2], sampled_edge_ids = [0, 2]
        #          -> edge_incidence_mask = [True, True, True, False, True, True]
        edge_incidence_mask = torch.isin(edge_ids, sampled_edge_ids)

        # Incidence is kept if node is sampled AND hyperedge is relevant
        incidence_mask = node_incidence_mask & edge_incidence_mask

        # Keep only the incidences that match our mask
        # Example: edge_index = [[0, 0, 1, 2, 3, 4],
        #                        [0, 0, 0, 1, 2, 2]],
        #          incidence_mask = [True, True, False, False, True, False]
        #          -> sampled_edge_index = [[0, 0, 3],
        #                                   [0, 0, 2]]
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
        # Example: sampled_edge_index = [[1, 1, 3],
        #                                [0, 2, 2]]
        #          sampled_node_ids = [1, 3],
        #          sampled_edge_ids = [0, 2]
        #          -> new_node_ids = [0, 0, 1], new_edge_ids = [0, 1, 1]
        new_node_ids = self.__to_0based_ids(
            sampled_edge_index[0], sampled_node_ids, self.hdata.num_nodes
        )
        new_edge_ids = self.__to_0based_ids(
            sampled_edge_index[1], sampled_edge_ids, self.hdata.num_edges
        )

        # Example: new_node_ids = [0, 1], new_edge_ids = [0, 1]
        #          -> new_edge_index = [[0, 1],
        #                               [0, 1]]
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
            original_ids: [1, 3, 3, 7]
            ids_to_keep: [3, 7]
            n = 8                            # total number of elements (nodes or edges) in the original hypergraph
            Returned 0-based IDs: [0, 0, 1]  # the size is sum of occurrences of ids_to_keep in original_ids
        Args:
            original_ids: Tensor of original IDs.
            n: Total number of original IDs.
            ids_to_keep: List of selected original IDs to be mapped to 0-based.
        Returns:
            Tensor of 0-based ids.
        """
        id_to_0based_id = torch.zeros(n, dtype=torch.long)
        n_ids_to_keep = len(ids_to_keep)
        id_to_0based_id[ids_to_keep] = torch.arange(n_ids_to_keep)
        return id_to_0based_id[original_ids]


class AlgebraDataset(Dataset):
    DATASET_NAME = "ALGEBRA"
    GDRIVE_FILE_ID = "1-H21_mZTcbbae4U_yM3xzXX19VhbCZ9C"
