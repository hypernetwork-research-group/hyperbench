import torch

from abc import ABC, abstractmethod
from enum import Enum
from torch import Tensor
from typing import List, Set
from hyperbench.types import HData
from hyperbench.utils import to_0based_ids


class NegativeSamplingSchedule(Enum):
    """When to run negative sampling during training."""

    FIRST_EPOCH = "first_epoch"  # Only at epoch 0, cached for all subsequent epochs
    EVERY_N_EPOCHS = "every_n_epochs"  # Every N epochs (N provided separately)
    EVERY_EPOCH = "every_epoch"  # Negatives generated every epoch


class NegativeSampler(ABC):
    @abstractmethod
    def sample(self, data: HData) -> HData:
        """
        Abstract method for negative sampling.

        Args:
            data: HData
                The input data object containing graph or hypergraph information.

        Returns:
            HData: The negative samples as a new HData object.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class RandomNegativeSampler(NegativeSampler):
    """
    A random negative sampler.

    Args:
        num_negative_samples (int): Number of negative hyperedges to generate.
        num_nodes_per_sample (int): Number of nodes per negative hyperedge.

    Raises:
        ValueError: If either argument is not positive.
    """

    def __init__(self, num_negative_samples: int, num_nodes_per_sample: int):
        if num_negative_samples <= 0:
            raise ValueError(f"num_negative_samples must be positive, got {num_negative_samples}.")
        if num_nodes_per_sample <= 0:
            raise ValueError(f"num_nodes_per_sample must be positive, got {num_nodes_per_sample}.")

        super().__init__()
        self.num_negative_samples = num_negative_samples
        self.num_nodes_per_sample = num_nodes_per_sample

    def sample(self, data: HData) -> HData:
        """
        Generate negative hyperedges by randomly sampling unique node IDs.
        Node IDs are sampled from the same node space as the input data, and the new negative hyperedge IDs
        start from the original number of hyperedges in the input data to avoid ID conflicts.
        The resulting negative samples are returned as a new :class:`HData` object with remapped 0-based node and hyperedge IDs, and mappings to the original IDs.

        Example:
            >>> num_negative_samples = 2, num_nodes_per_sample = 3
            >>> negative_hyperedge_index = [[0, 0, 1, 2, 3, 4],
                                           [0, 1, 1, 0, 1, 0]]
                The negative hyperedge 0 connects nodes 0, 2, 3.
                The second negative hyperedge 1 connects nodes 0, 1, 4.
            >>> negative_x = data.x[[0, 1, 2, 3, 4]]
            >>> negative_hyperedge_attr = random attributes for the 2 negative hyperedges (if data.edge_attr is not None)
            >>> node_local_to_global = [0, 1, 2, 3, 4]
            >>> edge_local_to_global = [3, 4]  # if data.num_edges was 3, the new negative hyperedges will have global IDs 3 and 4
        Args:
            data (HData): The input data object containing node and hyperedge information.

        Returns:
            A new :class:`HData` instance containing the negative samples. It contains node and hyperedges remapped to 0-based IDs,
            and the ``node_local_to_global`` and ``edge_local_to_global`` mappings for the negative samples, so that they can be
            mapped back to the original node and hyperedge IDs in the input data.

        Raises:
            ValueError: If ``num_nodes_per_sample`` is greater than the number of available nodes.
        """
        if self.num_nodes_per_sample > data.num_nodes:
            raise ValueError(
                f"Asked to create samples with {self.num_nodes_per_sample} nodes, but only {data.num_nodes} nodes are available."
            )

        device = data.x.device

        negative_node_ids: Set[int] = set()
        sampled_hyperedge_indexes: List[Tensor] = []
        sampled_hyperedge_attrs: List[Tensor] = []

        new_hyperedge_id_offset = data.num_edges
        for new_hyperedge_id in range(self.num_negative_samples):
            # Sample with multinomial without replacement to ensure unique node ids
            # and assign each node id equal probability of being selected by setting all of them to 1
            # Example: num_nodes_per_sample=3, max_node_id=5
            #          -> possible output: [2, 0, 4]
            equal_probabilities = torch.ones(data.num_nodes, device=device)
            sampled_node_ids = torch.multinomial(
                equal_probabilities, self.num_nodes_per_sample, replacement=False
            )

            # Example: sampled_node_ids = [2, 0, 4], new_hyperedge_id=0, new_hyperedge_id_offset=3
            #          -> hyperedge_index = [[2, 0, 4],
            #                                [3, 3, 3]]  # this is sampled_hyperedge_id_tensor
            sampled_hyperedge_id_tensor = torch.full(
                (self.num_nodes_per_sample,),
                new_hyperedge_id + new_hyperedge_id_offset,
                device=device,
            )
            sampled_hyperedge_index = torch.stack(
                [sampled_node_ids, sampled_hyperedge_id_tensor], dim=0
            )
            sampled_hyperedge_indexes.append(sampled_hyperedge_index)

            # Example: nodes = [0, 1, 2],
            #          sampled_node_ids_0 = [0, 1], sampled_node_ids_1 = [1, 2],
            #          -> negative_node_ids = {0, 1, 2}
            negative_node_ids.update(sampled_node_ids.tolist())

            if data.edge_attr is not None:
                random_edge_attr = torch.randn_like(data.edge_attr[0])
                sampled_hyperedge_attrs.append(random_edge_attr)

        sorted_negative_node_ids = torch.tensor(sorted(negative_node_ids), device=device)

        # Example: new_hyperedge_id_offset = 3 (if data.num_edges was 3)
        #          num_negative_samples = 2
        #          -> num_hyperedges_including_negatives = 5
        num_hyperedges_including_negatives = new_hyperedge_id_offset + self.num_negative_samples
        negative_hyperedge_ids = torch.arange(
            new_hyperedge_id_offset,
            num_hyperedges_including_negatives,
            device=device,
        )

        negative_hyperedge_index = self.__new_negative_hyperedge_index(
            sampled_hyperedge_indexes,
            sorted_negative_node_ids,
            data.num_nodes,
            negative_hyperedge_ids,
            num_hyperedges_including_negatives,
        )

        negative_x = data.x[sorted_negative_node_ids]
        negative_hyperedge_attr = (
            torch.stack(sampled_hyperedge_attrs, dim=0) if data.edge_attr is not None else None
        )
        return HData(
            x=negative_x,
            edge_index=negative_hyperedge_index,
            edge_attr=negative_hyperedge_attr,
            num_nodes=len(negative_node_ids),
            num_edges=self.num_negative_samples,
            node_local_to_global=sorted_negative_node_ids,
            edge_local_to_global=negative_hyperedge_ids,
        ).with_y_zeros()

    def __new_negative_hyperedge_index(
        self,
        sampled_hyperedge_indexes: List[Tensor],
        negative_node_ids: Tensor,
        num_nodes: int,
        negative_hyperedge_ids: Tensor,
        num_hyperedges: int,
    ) -> Tensor:
        """
        Concatenate, sort, and remap the sampled hyperedge indexes for negative samples.

        Args:
            sampled_hyperedge_indexes (List[Tensor]): List of hyperedge index tensors for each negative sample.
            negative_node_ids (Tensor): Tensor of negative node IDs.
            num_nodes (int): The number of nodes in the negative sample.
            negative_hyperedge_ids (Tensor): Tensor of negative hyperedge IDs.
            num_hyperedges (int): The number of hyperedges in the negative sample.

        Returns:
            Tensor: The concatenated, sorted, and remapped hyperedge index tensor.
        """
        negative_hyperedge_index = torch.cat(sampled_hyperedge_indexes, dim=1)
        node_ids_order = negative_hyperedge_index[0].argsort()

        # Example: negative_hyperedge_index before sorting: [[2, 0, 4, 0, 1, 3],
        #                                                    [3, 3, 3, 4, 4, 4]]
        #          -> negative_hyperedge_index after sorting: [[0, 0, 1, 2, 3, 4],
        #                                                      [3, 4, 4, 3, 4, 3]]
        negative_hyperedge_index = negative_hyperedge_index[:, node_ids_order]

        # Example: negative_hyperedge_index after sorting: [[0, 0, 1, 2, 3, 4],
        #                                                    [3, 4, 4, 3, 4, 3]]
        #          global_to_local_ids = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
        #          -> negative_hyperedge_index after remapping: [[0, 0, 1, 2, 3, 4],
        #                                                        [3, 4, 4, 3, 4, 3]]
        negative_hyperedge_index[0] = to_0based_ids(
            negative_hyperedge_index[0],
            negative_node_ids,
            num_nodes,
        )

        # Example: negative_hyperedge_index after remapping nodes: [[0, 0, 1, 2, 3, 4],
        #                                                           [3, 4, 4, 3, 4, 3]]
        #          negative_hyperedge_ids = [3, 4]
        #          -> negative_hyperedge_index after remapping hyperedges: [[0, 0, 1, 2, 3, 4],
        #                                                                   [0, 0, 1, 0, 1, 0]]
        negative_hyperedge_index[1] = to_0based_ids(
            negative_hyperedge_index[1],
            negative_hyperedge_ids,
            num_hyperedges,
        )

        return negative_hyperedge_index
