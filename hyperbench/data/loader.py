import torch

from typing import List, Optional, Tuple
from torch import Tensor
from torch.utils.data import DataLoader as TorchDataLoader
from hyperbench.data import Dataset
from hyperbench.types import HData


class DataLoader(TorchDataLoader):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: Optional[bool] = False,
        **kwargs,
    ) -> None:
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate,
            **kwargs,
        )

    def collate(self, batch: List[HData]) -> HData:
        """Collates a list of HData objects into a single batched HData object.

        This function combines multiple separate hypergraph samples into a single
        batched representation suitable for mini-batch training. It handles:
        - Concatenating node features from all samples
        - Concatenating and offsetting hyperedge from all samples
        - Concatenating edge attributes from all samples, if present

        Example:
            Given batch = [HData_0, HData_1].

            For node features:
                HData_0.x.shape = (3, 64) # 3 nodes with 64 features
                HData_1.x.shape = (2, 64) # 2 nodes with 64 features

                The output will be HData with:
                    x.shape = (5, 64) # All 5 nodes concatenated

            For edge index:
                HData_0 (3 nodes, 2 hyperedges):
                    edge_index = [[0, 1, 1, 2], # Nodes 0, 1, 1, 2
                                  [0, 0, 1, 1]] # Hyperedge 0 contains {0,1}, Hyperedge 1 contains {1,2}

                HData_1 (2 nodes, 1 hyperedge):
                    edge_index = [[0, 1], # Nodes 0, 1
                                  [0, 0]] # Hyperedge 0 contains {0,1}

                Batched result:
                    edge_index = [[0, 1, 1, 2, 3, 4], # Node indices: original then offset by 3, so 0->3, 1->4
                                  [0, 0, 1, 1, 2, 2]] # Hyperedge IDs: original then offset by 2, so 0->2, 0->2
                                   ^^^^^^^^^^  ^^^^
                                   Sample 0    Sample 1 (nodes +3, edges +2)

        Args:
            batch: List of HData objects to collate.

        Returns:
            HData: A single HData object containing the batched data.
        """
        node_features, total_nodes = self.__batch_node_features(batch)
        edge_index, edge_attr, total_edges = self.__batch_edges(batch)

        batched_data = HData(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=total_nodes,
            num_edges=total_edges,
        )

        return batched_data

    def __batch_node_features(self, batch: List[HData]) -> Tuple[Tensor, int]:
        """Concatenates node features from all samples in the batch.

        Example:
            With shape being (num_nodes_in_sample, num_features).

            If batch contains 3 sample with node features:
                Sample 0: x = [[1, 2], [3, 4]]           , shape: (2, 2)
                Sample 1: x = [[5, 6]]                   , shape: (1, 2)
                Sample 2: x = [[7, 8], [9, 10], [11, 12]], shape: (3, 2)

            Result:
                x: [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]
                shape: (6, 2), where 6 = 2 + 1 + 3 total nodes.

        Args:
            batch: List of HData objects.

        Returns:
            Tensor: Concatenated node features with shape (total_nodes, num_features).
        """
        per_sample_x = [data.x for data in batch]

        # Stack all nodes along the node dimension from all samples into a single tensor
        batched_x = torch.cat(per_sample_x, dim=0)
        total_nodes = batched_x.size(0)

        return batched_x, total_nodes

    def __batch_edges(self, batch: List[HData]) -> Tuple[Tensor, Optional[Tensor], int]:
        """Batches hyperedge indices and attributes, adjusting indices for concatenated nodes.
        Hyperedge indices must be offset so they point to the correct nodes in the batched node tensor.

        Example:
            Sample 0 (3 nodes, 2 hyperedges):
                edge_index = [[0, 1, 1, 2], # Nodes 0, 1, 1, 2
                              [0, 0, 1, 1]] # Hyperedge 0 contains {0,1}, Hyperedge 1 contains {1,2}
                node_offset = 0
                edge_offset = 0

            Sample 1 (2 nodes, 1 hyperedge):
                edge_index = [[0, 1], # Nodes 0, 1
                              [0, 0]] # Hyperedge 0 contains {0,1}
                node_offset = 3 # Previous samples have 3 nodes total
                edge_offset = 2 # Previous samples have 2 hyperedges total
            Result:
                edge_index = [[0, 1, 1, 2, 3, 4], # Node indices: original then offset by 3, so 0->3, 1->4
                              [0, 0, 1, 1, 2, 2]] # Hyperedge IDs: original then offset by 2, so 0->2, 0->2
                               ^^^^^^^^^^  ^^^^
                               Sample 0    Sample 1 (nodes +3, edges +2)

        Args:
            batch: List of HData objects.

        Returns:
            Tuple containing:
                - batched_edge_index: Concatenated and offset hyperedge indices, or None
                - batched_edge_attr: Concatenated hyperedge attributes, or None
                - total_edges: Total number of hyperedges across all batched samples
        """
        edge_indexes = []
        edge_attrs = []
        node_offset = 0
        edge_offset = 0

        for data in batch:
            # Offset nodes and hyperedge IDs (indices) in edge_index
            offset_edge_index = data.edge_index.clone()
            offset_edge_index[0] += node_offset
            offset_edge_index[1] += edge_offset
            edge_indexes.append(offset_edge_index)

            if data.edge_attr is not None:
                edge_attrs.append(data.edge_attr)

            # Offset calculations for next sample based on the max hyperedge ID as it indicates the number of hyperedges
            max_edge_id = (
                data.edge_index[1].max().item() if data.edge_index.size(1) > 0 else -1
            )
            edge_offset += (
                data.num_edges if data.num_edges is not None else max_edge_id + 1
            )

            # Offset calculations for next sample based on x[0] as x has shape (num_nodes, num_features), so 0 provides the number of nodes
            node_offset += (
                data.num_nodes if data.num_nodes is not None else data.x.size(0)
            )

        # Concatenate all edge_index tensors along the incidence dimension, so that we get a shape of (2, total_edges)
        batched_edge_index = torch.cat(edge_indexes, dim=1)
        max_edge_id = int(
            (
                batched_edge_index[1].max().item()
                if batched_edge_index.size(1) > 0
                else -1
            )
        )
        total_edges = max_edge_id + 1

        batched_edge_attr = None
        if len(edge_attrs) > 0:
            # Concatenate hyperedge attributes along dimension 0 (the hyperedge dimension)
            # edge_attr typically has shape (num_edges, num_edge_features)
            # Result shape: (total_edges, num_edge_features)
            batched_edge_attr = torch.cat(edge_attrs, dim=0)

        return batched_edge_index, batched_edge_attr, total_edges
