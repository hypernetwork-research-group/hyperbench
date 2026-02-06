import torch

from torch import Tensor
from typing import List


class Graph:
    """A simple graph data structure using edge list representation."""

    def __init__(self, edges: List[List[int]]):
        self.edges = edges

    @property
    def num_nodes(self) -> int:
        """Return the number of nodes in the graph."""
        nodes = set()
        for edge in self.edges:
            nodes.update(edge)
        return len(nodes)

    @property
    def num_edges(self) -> int:
        """Return the number of edges in the graph."""
        return len(self.edges)

    def remove_selfloops(self):
        """
        Remove self-loops from the graph.

        Returns:
            List of edges without self-loops.
        """
        if self.num_edges == 0:
            return

        graph_edges_tensor = torch.tensor(self.edges, dtype=torch.long)

        # Example: edges = [[0, 1],
        #                   [1, 1],
        #                   [2, 3]] shape (|E|, 2)
        #          -> no_selfloop_mask = [True, False, True]
        #          -> edges without self-loops = [[0, 1],
        #                                         [2, 3]]
        no_selfloop_mask = graph_edges_tensor[:, 0] != graph_edges_tensor[:, 1]
        self.edges = graph_edges_tensor[no_selfloop_mask].tolist()

    def to_edge_index(self) -> Tensor:
        """
        Convert the graph to edge index representation.

        Returns:
            edge_index: Tensor of shape (2, |E|) representing edges.
        """
        if self.num_edges == 0:
            return torch.empty((2, 0), dtype=torch.long)

        # Example: edges = [[0, 1],
        #                   [1, 2],
        #                   [2, 3]] shape (|E|, 2)
        #          ->  edge_index = [[0, 1, 2],
        #                            [1, 2, 3]] shape (2, |E|)
        edge_index = torch.tensor(self.edges, dtype=torch.long).t()
        return edge_index

    @classmethod
    def from_directed_to_undirected_edge_index(
        cls,
        edge_index: Tensor,
        with_selfloops: bool = False,
    ) -> Tensor:
        """
        Convert a directed edge index to an undirected edge index by adding reverse edges.

        Args:
            edge_index: Tensor of shape ``(2, |E|)`` representing directed edges.
            with_selfloops: Whether to add self-loops to each node. Defaults to ``False``.

        Returns:
            The undirected edge index tensor of shape ``(2, |E'|)``. If ``with_selfloops`` is ``True``, self-loops are added.
        """
        src, dest = edge_index[0], edge_index[1]
        src, dest = torch.cat([src, dest]), torch.cat([dest, src])

        # Example: edge_index = [[0, 1, 2],
        #                        [1, 0, 3]]
        #          -> after torch.stack([...], dim=0):
        #             undirected_edge_index = [[0, 1, 2, 1, 0, 3],
        #                                      [1, 0, 3, 0, 1, 2]]
        #          -> after torch.unique(..., dim=1):
        #             undirected_edge_index = [[0, 1, 2, 3],
        #                                      [1, 0, 3, 2]]
        undirected_edge_index: Tensor = torch.stack([src, dest], dim=0).to(
            edge_index.device
        )
        undirected_edge_index = cls.__remove_duplicate_edges(undirected_edge_index)

        if with_selfloops:
            # num_nodes assumes that the node indices in edge_index are in the range [0, num_nodes-1],
            # as this is the default logic in the library dataset preprocessing.
            num_nodes = int(undirected_edge_index.max().item()) + 1
            src, dest = undirected_edge_index[0], undirected_edge_index[1]

            # Add self-loops: A_hat = A + I (works as we assume node indices are in [0, num_nodes-1])
            selfloop_indices = torch.arange(num_nodes, device=edge_index.device)
            src = torch.cat([src, selfloop_indices])
            dest = torch.cat([dest, selfloop_indices])
            undirected_edge_index = torch.stack([src, dest], dim=0)
            undirected_edge_index = cls.__remove_duplicate_edges(undirected_edge_index)

        return undirected_edge_index

    @classmethod
    def __remove_duplicate_edges(cls, edge_index: Tensor) -> Tensor:
        """Remove duplicate edges from the edge index."""
        # Example: edge_index = [[0, 1, 2, 2, 0, 3, 2],
        #                        [1, 0, 3, 2, 1, 2, 2]], shape (2, |E| = 7)
        #          -> after torch.unique(..., dim=1):
        #             edge_index = [[0, 1, 2, 2, 3],
        #                           [1, 0, 3, 2, 2]], shape (2, |E'| = 5)
        return torch.unique(edge_index, dim=1)
