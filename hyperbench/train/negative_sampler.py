import torch

from typing import List, Set
from torch import Tensor
from hyperbench.types import HData


class NegativeSampler:
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
            raise ValueError(
                f"num_negative_samples must be positive, got {num_negative_samples}."
            )
        if num_nodes_per_sample <= 0:
            raise ValueError(
                f"num_nodes_per_sample must be positive, got {num_nodes_per_sample}."
            )

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
        sampled_edge_indexes: List[Tensor] = []
        sampled_edge_attrs: List[Tensor] = []

        device = data.x.device
        new_edge_id_offset = data.num_edges
        for new_edge_id in range(self.num_negative_samples):
            # Sample with multinomial without replacement to ensure unique node ids
            # and assign each node id equal probability of being selected by setting all of them to 1
            # Example: num_nodes_per_sample=3, max_node_id=5
            #          -> possible output: [2, 0, 4]
            equal_probabilities = torch.ones(data.num_nodes, device=device)
            sampled_node_ids = torch.multinomial(
                equal_probabilities, self.num_nodes_per_sample, replacement=False
            )

            # Example: sampled_node_ids = [2, 0, 4], new_edge_id=0, new_edge_id_offset=3
            #          -> edge_index = [[2, 0, 4],
            #                           [3, 3, 3]]
            sampled_edge_id_tensor = torch.full(
                (self.num_nodes_per_sample,),
                new_edge_id + new_edge_id_offset,
                device=device,
            )
            sampled_edge_index = torch.stack(
                [sampled_node_ids, sampled_edge_id_tensor], dim=0
            )
            sampled_edge_indexes.append(sampled_edge_index)

            # Example: nodes = [0, 1, 2],
            #          sampled_node_ids_0 = [0, 1], sampled_node_ids_1 = [1, 2],
            #          -> negative_node_ids = {0, 1, 2}
            negative_node_ids.update(sampled_node_ids.tolist())

            if data.edge_attr is not None:
                random_edge_attr = torch.randn_like(data.edge_attr[0])
                sampled_edge_attrs.append(random_edge_attr)

        negative_x = data.x[sorted(negative_node_ids)]
        negative_edge_index = self.__new_negative_edge_index(sampled_edge_indexes)
        negative_edge_attr = (
            torch.stack(sampled_edge_attrs, dim=0)
            if data.edge_attr is not None
            else None
        )

        return HData(
            x=negative_x,
            edge_index=negative_edge_index,
            edge_attr=negative_edge_attr,
            num_nodes=len(negative_node_ids),
            num_edges=self.num_negative_samples,
        )

    def __new_negative_edge_index(self, sampled_edge_indexes: List[Tensor]) -> Tensor:
        """
        Concatenate and sort the sampled edge indexes for negative samples.

        Args:
            sampled_edge_indexes (Tensor): List of edge index tensors for each negative sample.

        Returns:
            Tensor: The concatenated and sorted edge index tensor.
        """
        negative_edge_index = torch.cat(sampled_edge_indexes, dim=1)
        node_ids_order = negative_edge_index[0].argsort()

        # Example: negative_edge_index before sorting: [[2, 0, 4, 0, 1, 3],
        #                                               [3, 3, 3, 4, 4, 4]]
        #          -> negative_edge_index after sorting: [[0, 0, 1, 2, 3, 4],
        #                                                 [3, 4, 4, 3, 4, 3]]
        negative_edge_index = negative_edge_index[:, node_ids_order]
        return negative_edge_index
