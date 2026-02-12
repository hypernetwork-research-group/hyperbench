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
from hyperbench.types import HData, HIFHypergraph
from hyperbench.utils import validate_hif_json


class DatasetNames(Enum):
    """
    Enumeration of available datasets.
    """

    ALGEBRA = "algebra"
    AMAZON = "amazon"
    CONTACT_HIGH_SCHOOL = "contact-high-school"
    CONTACT_PRIMARY_SCHOOL = "contact-primary-school"
    CORA = "cora"
    COURSERA = "coursera"
    DBLP = "dblp"
    EMAIL_ENRON = "email-Enron"
    EMAIL_W3C = "email-W3C"
    GEOMETRY = "geometry"
    GOT = "got"
    IMDB = "imdb"
    MUSIC_BLUES_REVIEWS = "music-blues-reviews"
    NBA = "nba"
    NDC_CLASSES = "NDC-classes"
    NDC_SUBSTANCES = "NDC-substances"
    PATENT = "patent"
    PUBMED = "pubmed"
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
    def load_from_hif(dataset_name: Optional[str], save_on_disk: bool = False) -> HIFHypergraph:
        if dataset_name is None:
            raise ValueError(f"Dataset name (provided: {dataset_name}) must be provided.")
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
            tempfile.NamedTemporaryFile(mode="wb", suffix=".json", delete=False) as tmp_file,
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
    DATASET_NAME = None

    def __init__(self) -> None:
        self.hypergraph: HIFHypergraph = self.download()
        self.hdata: HData = self.process()

    def __len__(self) -> int:
        return len(self.hypergraph.nodes)

    def __getitem__(self, index: int | List[int]) -> HData:
        sampled_node_ids_list = self.__get_node_ids_to_sample(index)
        self.__validate_node_ids(sampled_node_ids_list)

        sampled_hyperedge_index, sampled_node_ids, sampled_hyperedge_ids = (
            self.__sample_hyperedge_index(sampled_node_ids_list)
        )

        new_hyperedge_index = self.__new_hyperedge_index(
            sampled_hyperedge_index, sampled_node_ids, sampled_hyperedge_ids
        )

        new_x = self.hdata.x[sampled_node_ids]

        new_edge_attr = None
        if self.hdata.edge_attr is not None and len(sampled_hyperedge_ids) > 0:
            new_edge_attr = self.hdata.edge_attr[sampled_hyperedge_ids]

        return HData(
            x=new_x,
            edge_index=new_hyperedge_index,
            edge_attr=new_edge_attr,
            num_nodes=len(sampled_node_ids),
            num_edges=len(sampled_hyperedge_ids),
        )

    def download(self) -> HIFHypergraph:
        """
        Load the hypergraph from HIF format using HIFConverter class.
        """
        if hasattr(self, "hypergraph") and self.hypergraph is not None:
            return self.hypergraph
        hypergraph = HIFConverter.load_from_hif(self.DATASET_NAME, save_on_disk=True)
        return hypergraph

    def process(self) -> HData:
        """
        Process the loaded hypergraph into HData format, mapping HIF structure to tensors.

        Returns:
            HData: Processed hypergraph data.
        """
        num_nodes = len(self.hypergraph.nodes)

        # x: shape [num_nodes, num_node_features]
        # collect all attribute keys to have tensors of same size
        node_attr_keys = self.__collect_attr_keys(
            [node.get("attrs", {}) for node in self.hypergraph.nodes]
        )

        if node_attr_keys:
            x = torch.stack(
                [
                    self.transform_node_attrs(node.get("attrs", {}), attr_keys=node_attr_keys)
                    for node in self.hypergraph.nodes
                ]
            )
        else:
            # Fallback to ones if no node features, 1 is better as it can help during
            # training (e.g., avoid zero multiplication), especially in first epochs
            x = torch.ones((num_nodes, 1), dtype=torch.float)

        # Remap node IDs to 0-based contiguous IDs matching the x tensor order
        node_set = {node.get("node"): id for id, node in enumerate(self.hypergraph.nodes)}
        # Initialize edge_set only with edges that have incidences
        # to avoid inflating edge count due to isolated nodes/missing incidences
        hyperedge_set = {}

        node_ids = []
        hyperedge_ids = []
        nodes_with_incidences = set()

        for incidence in self.hypergraph.incidences:
            node = incidence.get("node", 0)
            hyperedge = incidence.get("edge", 0)

            if hyperedge not in hyperedge_set:
                # Edges start from 0 and are assigned IDs in the order they are first encountered in incidences
                hyperedge_set[hyperedge] = len(hyperedge_set)

            node_ids.append(node_set[node])
            hyperedge_ids.append(hyperedge_set[hyperedge])
            nodes_with_incidences.add(node_set[node])

        # Handle isolated nodes by assigning them to a new unique hyperedge (self-loop)
        for hyperedge_idx in range(num_nodes):
            if hyperedge_idx not in nodes_with_incidences:
                new_hyperedge_id = len(hyperedge_set)
                # Unique dummy key to reserve the index in hyperedge_set
                hyperedge_set[f"__self_loop_{hyperedge_idx}__"] = new_hyperedge_id

                node_ids.append(hyperedge_idx)
                hyperedge_ids.append(new_hyperedge_id)

        num_hyperedges = len(hyperedge_set)

        # hyperedge_index: shape [2, E] where E is number of incidences
        hyperedge_index = torch.tensor([node_ids, hyperedge_ids], dtype=torch.long)

        # hyperedge-attr: shape [num_hyperedges, num_hyperedge_attributes]
        hyperedge_attr = None
        should_process_hyperedge_attrs = self.hypergraph.edges and any(
            "attrs" in edge for edge in self.hypergraph.edges
        )
        if should_process_hyperedge_attrs:
            hyperedge_id_to_attrs: Dict[Any, Dict[str, Any]] = {
                e.get("edge"): e.get("attrs", {}) for e in self.hypergraph.edges
            }

            hyperedge_attr_keys = self.__collect_attr_keys(list(hyperedge_id_to_attrs.values()))

            # Build attributes in exact order of hyperedge_set indices (0 to num_hyperedges - 1)
            idx_to_id = {
                hyperedge_idx: hyperedge_id for hyperedge_id, hyperedge_idx in hyperedge_set.items()
            }

            attrs = []
            for hyperedge_idx in range(num_hyperedges):
                hyperedge_id = idx_to_id[hyperedge_idx]

                # If it's a real hyperedge, get its attrs, if self-loop, get empty dict
                hyperedge_attrs = hyperedge_id_to_attrs.get(hyperedge_id, {})
                attrs.append(
                    self.transform_hyperedge_attrs(hyperedge_attrs, attr_keys=hyperedge_attr_keys)
                )
            hyperedge_attr = torch.stack(attrs)

        return HData(x, hyperedge_index, hyperedge_attr, num_nodes, num_hyperedges)

    def transform_node_attrs(
        self,
        attrs: Dict[str, Any],
        attr_keys: Optional[List[str]] = None,
    ) -> Tensor:
        return self.transform_attrs(attrs, attr_keys)

    def transform_hyperedge_attrs(
        self,
        attrs: Dict[str, Any],
        attr_keys: Optional[List[str]] = None,
    ) -> Tensor:
        return self.transform_attrs(attrs, attr_keys)

    def transform_attrs(
        self,
        attrs: Dict[str, Any],
        attr_keys: Optional[List[str]] = None,
    ) -> Tensor:
        r"""
        Extract and encode numeric attributes to tensor.
        Non-numeric attributes are discarded. Missing attributes are filled with ``0.0``.

        Args:
            attrs: Dictionary of attributes
            attr_keys: Optional list of attribute keys to encode. If provided, ensures consistent ordering and fill missing with ``0.0``.

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
        """
        Get a list of node IDs to sample based on the provided index.

        Args:
            id: An integer or a list of integers representing node IDs to sample.

        Returns:
            List of node IDs to sample.

        Raises:
            ValueError: If the provided index is invalid (e.g., empty list or list length exceeds number of nodes).
        """
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
        """
        Validate that node IDs are within bounds of the hypergraph.

        Args:
            node_ids: List of node IDs to validate.

        Raises:
            IndexError: If any node ID is out of bounds.
        """
        for id in node_ids:
            if id < 0 or id >= self.__len__():
                raise IndexError(f"Node ID {id} is out of bounds (0, {self.__len__() - 1}).")

    def __sample_hyperedge_index(
        self,
        sampled_node_ids_list: List[int],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        hyperedge_index = self.hdata.edge_index
        node_ids = hyperedge_index[0]
        hyperedge_ids = hyperedge_index[1]

        sampled_node_ids = torch.tensor(sampled_node_ids_list, device=node_ids.device)

        # Find incidences where the node is in our sampled node set
        # Example: hyperedge_index[0] = [0, 0, 1, 2, 3, 4], sampled_node_ids = [0, 3]
        #          -> node_incidence_mask = [True, True, False, False, True, False]
        node_incidence_mask = torch.isin(node_ids, sampled_node_ids)

        # Get unique hyperedges that have at least one sampled node
        # Example: hyperedge_index[1] = [0, 0, 0, 1, 2, 2], node_incidence_mask = [True, True, False, False, True, False]
        #          -> sampled_hyperedge_ids = [0, 2] as they connect to sampled nodes
        sampled_hyperedge_ids = hyperedge_ids[node_incidence_mask].unique()

        # Find all incidences for the sampled hyperedges (not just sampled nodes)
        # Example: hyperedge_index[1] = [0, 0, 0, 1, 2, 2], sampled_hyperedge_ids = [0, 2]
        #          -> hyperedge_incidence_mask = [True, True, True, False, True, True]
        hyperedge_incidence_mask = torch.isin(hyperedge_ids, sampled_hyperedge_ids)

        # Collect all node IDs that appear in the sampled hyperedges
        # Example: hyperedge_index[0] = [0, 0, 1, 2, 3, 4], hyperedge_incidence_mask = [True, True, True, False, True, True]
        #          -> node_ids_in_sampled_hyperedge = [0, 1, 3, 4]
        node_ids_in_sampled_hyperedge = node_ids[hyperedge_incidence_mask].unique()

        # Keep all incidences belonging to the sampled hyperedges
        # Example: hyperedge_index = [[0, 0, 1, 2, 3, 4],
        #                             [0, 0, 0, 1, 2, 2]],
        #          hyperedge_incidence_mask = [True, True, True, False, True, True]
        #          -> sampled_hyperedge_index = [[0, 0, 1, 3, 4],
        #                                        [0, 0, 0, 2, 2]]
        sampled_hyperedge_index = hyperedge_index[:, hyperedge_incidence_mask]
        return sampled_hyperedge_index, node_ids_in_sampled_hyperedge, sampled_hyperedge_ids

    def __new_hyperedge_index(
        self,
        sampled_hyperedge_index: Tensor,
        sampled_node_ids: Tensor,
        sampled_hyperedge_ids: Tensor,
    ) -> Tensor:
        """
        Create new hyperedge_index with 0-based node and hyperedge IDs.

        Args:
            sampled_hyperedge_index: Original hyperedge_index tensor with sampled incidences.
            sampled_node_ids: List of sampled original node IDs.
            sampled_hyperedge_ids: List of sampled original hyperedge IDs.

        Returns:
            New hyperedge_index tensor with 0-based node and edge IDs.
        """
        # Example: sampled_edge_index = [[1, 1, 3],
        #                                [0, 2, 2]]
        #          sampled_node_ids = [1, 3],
        #          sampled_edge_ids = [0, 2]
        #          -> new_node_ids = [0, 0, 1], new_edge_ids = [0, 1, 1]
        new_node_ids = self.__to_0based_ids(
            sampled_hyperedge_index[0], sampled_node_ids, self.hdata.num_nodes
        )
        new_hyperedge_ids = self.__to_0based_ids(
            sampled_hyperedge_index[1], sampled_hyperedge_ids, self.hdata.num_edges
        )

        # Example: new_node_ids = [0, 1], new_hyperedge_ids = [0, 1]
        #          -> new_hyperedge_index = [[0, 1],
        #                                    [0, 1]]
        new_hyperedge_index = torch.stack([new_node_ids, new_hyperedge_ids], dim=0)
        return new_hyperedge_index

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


class AmazonDataset(Dataset):
    DATASET_NAME = "AMAZON"


class ContactHighSchoolDataset(Dataset):
    DATASET_NAME = "CONTACT_HIGH_SCHOOL"


class ContactPrimarySchoolDataset(Dataset):
    DATASET_NAME = "CONTACT_PRIMARY_SCHOOL"


class CoraDataset(Dataset):
    DATASET_NAME = "CORA"


class CourseraDataset(Dataset):
    DATASET_NAME = "COURSERA"


class DBLPDataset(Dataset):
    DATASET_NAME = "DBLP"


class EmailEnronDataset(Dataset):
    DATASET_NAME = "EMAIL_ENRON"


class EmailW3CDataset(Dataset):
    DATASET_NAME = "EMAIL_W3C"


class GeometryDataset(Dataset):
    DATASET_NAME = "GEOMETRY"


class GOTDataset(Dataset):
    DATASET_NAME = "GOT"


class IMDBDataset(Dataset):
    DATASET_NAME = "IMDB"


class MusicBluesReviewsDataset(Dataset):
    DATASET_NAME = "MUSIC_BLUES_REVIEWS"


class NBADataset(Dataset):
    DATASET_NAME = "NBA"


class NDCClassesDataset(Dataset):
    DATASET_NAME = "NDC_CLASSES"


class NDCSubstancesDataset(Dataset):
    DATASET_NAME = "NDC_SUBSTANCES"


class PatentDataset(Dataset):
    DATASET_NAME = "PATENT"


class PubmedDataset(Dataset):
    DATASET_NAME = "PUBMED"


class RestaurantReviewsDataset(Dataset):
    DATASET_NAME = "RESTAURANT_REVIEWS"


class ThreadsAskUbuntuDataset(Dataset):
    DATASET_NAME = "THREADS_ASK_UBUNTU"


class ThreadsMathsxDataset(Dataset):
    DATASET_NAME = "THREADS_MATH_SX"


class TwitterDataset(Dataset):
    DATASET_NAME = "TWITTER"


class VegasBarsReviewsDataset(Dataset):
    DATASET_NAME = "VEGAS_BARS_REVIEWS"
