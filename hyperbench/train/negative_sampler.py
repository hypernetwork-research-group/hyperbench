import torch

from abc import ABC, abstractmethod
from enum import Enum
from torch import Tensor
from typing import List, Set
from hyperbench.types import HData


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

        Args:
            data (HData): The input data object containing node and hyperedge information.

        Returns:
            HData: A new HData object containing the negative samples.

        Raises:
            ValueError: If num_nodes_per_sample is greater than the number of available nodes.
        """
        if self.num_nodes_per_sample > data.num_nodes:
            raise ValueError(
                f"Asked to create samples with {self.num_nodes_per_sample} nodes, but only {data.num_nodes} nodes are available."
            )

        negative_node_ids: Set[int] = set()
        sampled_hyperedge_indexes: List[Tensor] = []
        sampled_hyperedge_attrs: List[Tensor] = []

        device = data.x.device
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
            #                                [3, 3, 3]]
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

        negative_x = data.x[sorted(negative_node_ids)]
        negative_hyperedge_index = self.__new_negative_hyperedge_index(sampled_hyperedge_indexes)
        negative_hyperedge_attr = (
            torch.stack(sampled_hyperedge_attrs, dim=0) if data.edge_attr is not None else None
        )

        return HData(
            x=negative_x,
            edge_index=negative_hyperedge_index,
            edge_attr=negative_hyperedge_attr,
            num_nodes=len(negative_node_ids),
            num_edges=self.num_negative_samples,
        )

    def __new_negative_hyperedge_index(self, sampled_hyperedge_indexes: List[Tensor]) -> Tensor:
        """
        Concatenate and sort the sampled hyperedge indexes for negative samples.

        Args:
            sampled_hyperedge_indexes (Tensor): List of hyperedge index tensors for each negative sample.

        Returns:
            Tensor: The concatenated and sorted hyperedge index tensor.
        """
        negative_hyperedge_index = torch.cat(sampled_hyperedge_indexes, dim=1)
        node_ids_order = negative_hyperedge_index[0].argsort()

        # Example: negative_hyperedge_index before sorting: [[2, 0, 4, 0, 1, 3],
        #                                                    [3, 3, 3, 4, 4, 4]]
        #          -> negative_hyperedge_index after sorting: [[0, 0, 1, 2, 3, 4],
        #                                                      [3, 4, 4, 3, 4, 3]]
        negative_hyperedge_index = negative_hyperedge_index[:, node_ids_order]
        return negative_hyperedge_index
