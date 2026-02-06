import pytest
import json
import torch

from hyperbench.types import HIFHypergraph, Hypergraph
from hyperbench.tests import MOCK_BASE_PATH


def test_build_HIFHypergraph_instance():
    with open(f"{MOCK_BASE_PATH}/algebra.hif.json", "r") as f:
        hiftext = json.load(f)

    hypergraph = HIFHypergraph.from_hif(hiftext)

    assert isinstance(hypergraph, HIFHypergraph)


@pytest.mark.parametrize(
    "edges, expected_edges",
    [
        ([], []),  # Empty hypergraph
        ([[0]], [[0]]),  # Single node in single edge
        ([[0, 1, 2]], [[0, 1, 2]]),  # Single edge with multiple nodes
        ([[0, 1], [2, 3, 4], [5]], [[0, 1], [2, 3, 4], [5]]),  # Multiple edges
        (
            [[0, 1, 2], [1, 2, 3], [2, 3, 4]],
            [[0, 1, 2], [1, 2, 3], [2, 3, 4]],
        ),  # Multiple overlapping edges
        ([[0, 0, 1]], [[0, 0, 1]]),  # Duplicate node within edge
        ([[9, 2, 5, 1]], [[9, 2, 5, 1]]),  # Unordered nodes
    ],
)
def test_init_preserves_edges(edges, expected_edges):
    hypergraph = Hypergraph(edges)
    assert hypergraph.edges == expected_edges


@pytest.mark.parametrize(
    "edges, expected_num_nodes",
    [
        ([], 0),  # Empty hypergraph
        ([[0]], 1),  # Single node in single edge
        ([[0, 1, 2]], 3),  # Multiple nodes in single edge
        ([[0], [1], [2]], 3),  # Three singleton edges
        ([[0], [1], [1]], 2),  # Three singleton edges, two overlapping
        ([[0, 1], [2, 3]], 4),  # Two disjoint edges
        ([[0, 1], [1, 2]], 3),  # Two overlapping edges
        ([[0, 1, 2], [1, 2, 3]], 4),  # Overlapping edges with multiple nodes
        ([[0, 1, 2], [3, 4, 5], [6, 7, 8]], 9),  # Multiple disjoint edges
        ([[5, 10, 15]], 3),  # Non-contiguous node IDs
        ([[0, 0, 1]], 2),  # Edge with duplicate node
        ([[0, 1], [0, 1, 2]], 3),  # One edge is subset of another
        ([[9, 2, 5, 1]], 4),  # Unordered node IDs
    ],
)
def test_num_nodes(edges, expected_num_nodes):
    hypergraph = Hypergraph(edges)
    assert hypergraph.num_nodes == expected_num_nodes


@pytest.mark.parametrize(
    "edges, expected_num_edges",
    [
        ([], 0),  # Empty hypergraph
        ([[0]], 1),  # Single edge with one node
        ([[0, 1, 2]], 1),  # Single edge with multiple nodes
        ([[0], [1], [2]], 3),  # Three singleton edges
        ([[0, 1], [2, 3]], 2),  # Two disjoint edges
        ([[0, 1], [1, 2]], 2),  # Two overlapping edges
        ([[0, 1, 2], [1, 2, 3], [3, 4]], 3),  # Three edges with overlap
    ],
)
def test_num_edges(edges, expected_num_edges):
    hypergraph = Hypergraph(edges)
    assert hypergraph.num_edges == expected_num_edges


@pytest.mark.parametrize(
    "edge_index_data, expected_edges",
    [
        # Empty hypergraph
        ([[[], []]], []),
        # Single node, single edge
        ([[[0], [0]]], [[0]]),
        # Multiple nodes, single edge
        ([[[0, 1, 2, 3], [0, 0, 0, 0]]], [[0, 1, 2, 3]]),
        # Multiple edges, each with single node
        ([[[0, 1, 2], [0, 1, 2]]], [[0], [1], [2]]),
        # Two edges with multiple nodes each
        ([[[0, 1, 2, 3], [0, 0, 1, 1]]], [[0, 1], [2, 3]]),
        # Complex structure with varying edge sizes
        ([[[0, 1, 2, 3, 4, 5], [0, 0, 1, 2, 2, 2]]], [[0, 1], [2], [3, 4, 5]]),
    ],
)
def test_from_edge_index_parametrized(edge_index_data, expected_edges):
    nodes, edges = edge_index_data[0]
    edge_index = torch.tensor([nodes, edges], dtype=torch.long)
    hypergraph = Hypergraph.from_edge_index(edge_index)

    assert hypergraph.edges == expected_edges
