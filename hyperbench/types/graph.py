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

    def remove_self_loops(self):
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
        #          -> no_self_loop_mask = [True, False, True]
        #          -> edges without self-loops = [[0, 1],
        #                                         [2, 3]]
        no_self_loop_mask = graph_edges_tensor[:, 0] != graph_edges_tensor[:, 1]
        self.edges = graph_edges_tensor[no_self_loop_mask].tolist()

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
