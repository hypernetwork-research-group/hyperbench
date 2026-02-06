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
        pytest.param([], [], id="empty_hypergraph"),
        pytest.param([[0]], [[0]], id="single_node_single_edge"),
        pytest.param(
            [[0, 1, 2]],
            [[0, 1, 2]],
            id="single_edge_multiple_nodes",
        ),
        pytest.param(
            [[0, 1], [2, 3, 4], [5]],
            [[0, 1], [2, 3, 4], [5]],
            id="multiple_edges",
        ),
        pytest.param(
            [[0, 1, 2], [1, 2, 3], [2, 3, 4]],
            [[0, 1, 2], [1, 2, 3], [2, 3, 4]],
            id="multiple_overlapping_edges",
        ),
        pytest.param([[0, 0, 1]], [[0, 0, 1]], id="duplicate_node_within_edge"),
        pytest.param([[9, 2, 5, 1]], [[9, 2, 5, 1]], id="unordered_nodes"),
    ],
)
def test_init_preserves_edges(edges, expected_edges):
    hypergraph = Hypergraph(edges)
    assert hypergraph.edges == expected_edges


@pytest.mark.parametrize(
    "edges, expected_num_nodes",
    [
        pytest.param([], 0, id="empty_hypergraph"),
        pytest.param([[0]], 1, id="single_node_single_edge"),
        pytest.param([[0, 1, 2]], 3, id="multiple_nodes_single_edge"),
        pytest.param([[0], [1], [2]], 3, id="three_singleton_edges"),
        pytest.param([[0], [1], [1]], 2, id="three_singleton_edges_two_overlapping"),
        pytest.param([[0, 1], [2, 3]], 4, id="two_disjoint_edges"),
        pytest.param([[0, 1], [1, 2]], 3, id="two_overlapping_edges"),
        pytest.param(
            [[0, 1, 2], [1, 2, 3]],
            4,
            id="overlapping_edges_multiple_nodes",
        ),
        pytest.param(
            [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
            9,
            id="multiple_disjoint_edges",
        ),
        pytest.param([[5, 10, 15]], 3, id="non_contiguous_node_ids"),
        pytest.param([[0, 0, 1]], 2, id="edge_with_duplicate_node"),
        pytest.param([[0, 1], [0, 1, 2]], 3, id="edge_subset_of_another"),
        pytest.param([[9, 2, 5, 1]], 4, id="unordered_node_ids"),
    ],
)
def test_num_nodes(edges, expected_num_nodes):
    hypergraph = Hypergraph(edges)
    assert hypergraph.num_nodes == expected_num_nodes


@pytest.mark.parametrize(
    "edges, expected_num_edges",
    [
        pytest.param([], 0, id="empty_hypergraph"),
        pytest.param([[0]], 1, id="single_edge_one_node"),
        pytest.param([[0, 1, 2]], 1, id="single_edge_multiple_nodes"),
        pytest.param([[0], [1], [2]], 3, id="three_singleton_edges"),
        pytest.param([[0, 1], [2, 3]], 2, id="two_disjoint_edges"),
        pytest.param([[0, 1], [1, 2]], 2, id="two_overlapping_edges"),
        pytest.param(
            [[0, 1, 2], [1, 2, 3], [3, 4]],
            3,
            id="three_edges_with_overlap",
        ),
    ],
)
def test_num_edges(edges, expected_num_edges):
    hypergraph = Hypergraph(edges)
    assert hypergraph.num_edges == expected_num_edges


@pytest.mark.parametrize(
    "edge_index_data, expected_edges",
    [
        pytest.param([[[], []]], [], id="empty_hypergraph"),
        pytest.param([[[0], [0]]], [[0]], id="single_node_single_edge"),
        pytest.param(
            [[[0, 1, 2, 3], [0, 0, 0, 0]]],
            [[0, 1, 2, 3]],
            id="multiple_nodes_single_edge",
        ),
        pytest.param(
            [[[0, 1, 2], [0, 1, 2]]],
            [[0], [1], [2]],
            id="multiple_edges_single_nodes",
        ),
        pytest.param(
            [[[0, 1, 2, 3], [0, 0, 1, 1]]],
            [[0, 1], [2, 3]],
            id="two_edges_multiple_nodes",
        ),
        pytest.param(
            [[[0, 1, 2, 3, 4, 5], [0, 0, 1, 2, 2, 2]]],
            [[0, 1], [2], [3, 4, 5]],
            id="complex_varying_edge_sizes",
        ),
    ],
)
def test_from_edge_index_parametrized(edge_index_data, expected_edges):
    nodes, edges = edge_index_data[0]
    edge_index = torch.tensor([nodes, edges], dtype=torch.long)
    hypergraph = Hypergraph.from_edge_index(edge_index)

    assert hypergraph.edges == expected_edges
