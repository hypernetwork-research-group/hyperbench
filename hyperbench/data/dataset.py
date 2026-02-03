import json
import os
import tempfile
import torch
import zstandard as zstd
import requests

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from hyperbench.types.hypergraph import HIFHypergraph
from hyperbench.types.hdata import HData
from hyperbench.utils.hif_utils import validate_hif_json


class DatasetNames(Enum):
    """
    Enumeration of available datasets.
    """

    ALGEBRA = "algebra"
    AMAZON = "amazon"
    CONTACT_HIGH_SCHOOL = "contact-high-school"
    CONTACT_PRIMARY_SCHOOL = "contact-primary-school"
    DBLP = "dblp"
    EMAIL_ENRON = "email-Enron"
    EMAIL_W3C = "email-W3C"
    GEOMETRY = "geometry"
    GOT = "got"
    MUSIC_BLUES_REVIEWS = "music-blues-reviews"
    NBA = "nba"
    NDC_CLASSES = "NDC-classes"
    NDC_SUBSTANCES = "NDC-substances"
    RESTAURANT_REVIEWS = "restaurant-reviews"
    THREADS_ASK_UBUNTU = "threads-ask-ubuntu"
    THREADS_MATH_SX = "threads-math-sx"
    TWITTER = "twitter"
    VEGAS_BARS_REVIEWS = "vegas-bars-reviews"


class HIFConverter:
    """
    Docstring for HIFConverter
    A utility class to load hypergraphs from HIF format.
    """

    @staticmethod
    def load_from_hif(
        dataset_name: Optional[str], save_on_disk: bool = False
    ) -> HIFHypergraph:
        if dataset_name is None:
            raise ValueError(
                f"Dataset name (provided: {dataset_name}) must be provided."
            )
        if dataset_name not in DatasetNames.__members__:
            raise ValueError(f"Dataset '{dataset_name}' not found.")

        dataset_name = DatasetNames[dataset_name].value
        current_dir = os.path.dirname(os.path.abspath(__file__))
        zst_filename = os.path.join(current_dir, "datasets", f"{dataset_name}.json.zst")

        if not os.path.exists(zst_filename):
            github_dataset_repo = f"https://github.com/hypernetwork-research-group/datasets/blob/main/{dataset_name}.json.zst?raw=true"

            response = requests.get(github_dataset_repo)
            if response.status_code != 200:
                raise ValueError(
                    f"Failed to download dataset '{dataset_name}' from GitHub. Status code: {response.status_code}"
                )

            if save_on_disk:
                os.makedirs(os.path.join(current_dir, "datasets"), exist_ok=True)
                with open(zst_filename, "wb") as f:
                    f.write(response.content)
            else:
                # Create temporary file for downloaded zst content
                with tempfile.NamedTemporaryFile(
                    mode="wb", suffix=".json.zst", delete=False
                ) as tmp_zst_file:
                    tmp_zst_file.write(response.content)
                    zst_filename = tmp_zst_file.name

        # Decompress the downloaded zst file
        dctx = zstd.ZstdDecompressor()
        with (
            open(zst_filename, "rb") as input_f,
            tempfile.NamedTemporaryFile(
                mode="wb", suffix=".json", delete=False
            ) as tmp_file,
        ):
            dctx.copy_stream(input_f, tmp_file)
            output = tmp_file.name

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
        DATASET_NAME (str): Name of the dataset.
        hypergraph (HIFHypergraph): Loaded hypergraph instance.
    Methods:
        download(): Downloads and loads the hypergraph from HIF.
        process(): Processes the hypergraph into HData format.
    """

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
        hypergraph = HIFConverter.load_from_hif(self.DATASET_NAME)
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
        # collect all attribute keys to have tensors of same size
        node_attr_keys = self.__collect_attr_keys(
            [node.get("attrs", {}) for node in self.hypergraph.nodes]
        )

        if node_attr_keys:
            x = torch.stack(
                [
                    self.transform_node_attrs(
                        node.get("attrs", {}), attr_keys=node_attr_keys
                    )
                    for node in self.hypergraph.nodes
                ]
            )
        else:
            # Fallback to zeros if no numeric attributes
            x = torch.zeros((num_nodes, 1), dtype=torch.float)

        # remap node and edge IDs to 0-based contiguous IDs
        # Use dict comprehension for faster lookups
        node_set = {}
        edge_set = {}
        node_ids = []
        edge_ids = []

        for inc in self.hypergraph.incidences:
            node = inc.get("node", 0)
            edge = inc.get("edge", 0)

            if node not in node_set:
                node_set[node] = len(node_set)
            if edge not in edge_set:
                edge_set[edge] = len(edge_set)

            node_ids.append(node_set[node])
            edge_ids.append(edge_set[edge])

        if len(node_ids) < 1:
            raise ValueError("Hypergraph has no incidences.")

        # edge_index: shape [2, E] where E is number of incidences
        edge_index = torch.tensor([node_ids, edge_ids], dtype=torch.long)

        # edge-attr: shape [num_edges, num_edge_attributes]
        edge_attr = None
        if self.hypergraph.edges and any(
            "attrs" in edge for edge in self.hypergraph.edges
        ):
            # collect all attribute keys to have tensors of same size
            edge_attr_keys = self.__collect_attr_keys(
                [edge.get("attrs", {}) for edge in self.hypergraph.edges]
            )

            edge_attr = torch.stack(
                [
                    self.transform_edge_attrs(
                        edge.get("attrs", {}), attr_keys=edge_attr_keys
                    )
                    for edge in self.hypergraph.edges
                ]
            )

            # Flatten to 1D if only one attribute (PyTorch Geometric standard)
            # if edge_attr.shape[1] == 1:
            #     edge_attr = edge_attr.squeeze(1)

        return HData(x, edge_index, edge_attr, num_nodes, num_edges)

    def transform_node_attrs(
        self, attrs: Dict[str, Any], attr_keys: Optional[List[str]] = None
    ) -> Tensor:
        return self.transform_attrs(attrs, attr_keys)

    def transform_edge_attrs(
        self, attrs: Dict[str, Any], attr_keys: Optional[List[str]] = None
    ) -> Tensor:
        return self.transform_attrs(attrs, attr_keys)

    def transform_attrs(
        self, attrs: Dict[str, Any], attr_keys: Optional[List[str]] = None
    ) -> Tensor:
        """
        Extract and encode numeric node attributes to tensor.
        Non-numeric attributes are discarded. Missing attributes are filled with 0.0.

        Args:
            attrs: Dictionary of node attributes
            attr_keys: Optional list of attribute keys to encode. If provided, ensures
                      consistent ordering and fill missing with 0.0.

        Returns:
            Tensor of numeric attribute values
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

        values = [float(value) for value in numeric_attrs.values()]
        return torch.tensor(values, dtype=torch.float)

    def __collect_attr_keys(self, attr_keys: List[Dict[str, Any]]) -> List[str]:
        """
        Collect unique numeric attribute keys from a list of attribute dictionaries.
        Args:
            attrs_list: List of attribute dictionaries.
        Returns:
            List of unique numeric attribute keys.
        """
        unique_keys = []
        for attrs in attr_keys:
            for key, value in attrs.items():
                if key not in unique_keys and isinstance(value, (int, float)):
                    unique_keys.append(key)

        return unique_keys

    def __get_node_ids_to_sample(self, id: int | List[int]) -> List[int]:
        if isinstance(id, list):
            if len(id) < 1:
                raise ValueError("Index list cannot be empty.")
            elif len(id) > self.__len__():
                raise ValueError(
                    "Index list length cannot exceed number of nodes in the hypergraph."
                )
            return list(set(id))

        return [id]

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

        sampled_node_ids = torch.tensor(sampled_node_ids_list, device=node_ids.device)

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
        device = original_ids.device

        id_to_0based_id = torch.zeros(n, dtype=torch.long, device=device)
        n_ids_to_keep = len(ids_to_keep)
        id_to_0based_id[ids_to_keep] = torch.arange(n_ids_to_keep, device=device)
        return id_to_0based_id[original_ids]


class AlgebraDataset(Dataset):
    DATASET_NAME = "ALGEBRA"


class DBLPDataset(Dataset):
    DATASET_NAME = "DBLP"


class ThreadsMathsxDataset(Dataset):
    DATASET_NAME = "THREADSMATHSX"
