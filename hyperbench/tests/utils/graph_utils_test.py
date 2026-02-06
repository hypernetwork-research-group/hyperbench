import pytest
import torch

from hyperbench.utils import reduce_to_graph_edge_index
from hyperbench.utils.graph_utils import get_sparse_adjacency_matrix


@pytest.fixture(autouse=True)
def seed():
    """Fix random seed for deterministic projections."""
    torch.manual_seed(42)


def test_get_sparse_adjacency_matrix_returns_sparse_tensor():
    edge_index = torch.tensor([[0, 1], [1, 0]])
    result = get_sparse_adjacency_matrix(edge_index, num_nodes=2)

    assert result.is_sparse


@pytest.mark.parametrize(
    "edge_index, num_nodes",
    [
        pytest.param(torch.tensor([[0, 1], [1, 0]]), 2, id="2_nodes"),
        pytest.param(torch.tensor([[0, 1, 2], [1, 2, 0]]), 4, id="4_nodes_3_edges"),
        pytest.param(torch.tensor([[], []], dtype=torch.long), 5, id="5_nodes_empty"),
    ],
)
def test_get_sparse_adjacency_matrix_shape(edge_index, num_nodes):
    result = get_sparse_adjacency_matrix(edge_index, num_nodes=num_nodes)

    assert result.shape == (num_nodes, num_nodes)


def test_get_sparse_adjacency_matrix_empty_edge_index():
    """Empty edge_index produces all-zero adjacency matrix when converted to dense."""
    edge_index = torch.tensor([[], []], dtype=torch.long)
    result = get_sparse_adjacency_matrix(edge_index, num_nodes=3)
    dense = result.to_dense()

    assert torch.all(dense == 0)


@pytest.mark.parametrize(
    "edge_index, num_nodes, expected_entries",
    [
        pytest.param(
            torch.tensor([[0], [2]]),
            3,
            [(0, 2, 1.0)],
            id="single_directed_edge",
        ),
        pytest.param(
            torch.tensor([[0, 1], [1, 0]]),
            2,
            [(0, 1, 1.0), (1, 0, 1.0)],
            id="undirected_edge",
        ),
        pytest.param(
            torch.tensor([[1], [1]]),
            3,
            [(1, 1, 1.0)],
            id="self_loop",
        ),
        pytest.param(
            torch.tensor([[0, 1, 2], [1, 2, 0]]),
            3,
            [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)],
            id="triangle_directed",
        ),
        pytest.param(
            torch.tensor([[0, 1, 2, 2], [1, 2, 0, 1]]),
            3,
            [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0), (2, 1, 1.0)],
            id="multiple_edges_between_nodes",
        ),
        pytest.param(
            torch.tensor([[0, 1, 2, 2], [1, 2, 0, 0]]),
            3,
            [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 2.0)],  # Duplicate edges are summed
            id="duplicate_edges_to_same_target",
        ),
    ],
)
def test_get_sparse_adjacency_matrix_entries(edge_index, num_nodes, expected_entries):
    result = get_sparse_adjacency_matrix(edge_index, num_nodes=num_nodes)
    dense = result.to_dense()

    for row, col, val in expected_entries:
        assert dense[row, col] == val


def test_get_sparse_adjacency_matrix_preserves_device():
    edge_index = torch.tensor([[0], [1]], device="cpu")

    result = get_sparse_adjacency_matrix(edge_index, num_nodes=2)

    assert result.device == edge_index.device


@pytest.mark.parametrize(
    "edge_index, num_nodes, isolated_nodes",
    [
        pytest.param(
            torch.tensor([[0], [1]]),
            4,
            [2, 3],
            id="two_isolated_nodes",
        ),
        pytest.param(
            torch.tensor([[0, 1], [1, 0]]),
            5,
            [2, 3, 4],
            id="three_isolated_nodes",
        ),
    ],
)
def test_get_sparse_adjacency_matrix_isolated_nodes(
    edge_index, num_nodes, isolated_nodes
):
    """Nodes not in edge_index have zero rows and columns."""
    result = get_sparse_adjacency_matrix(edge_index, num_nodes=num_nodes)
    dense = result.to_dense()

    for node in isolated_nodes:
        assert torch.all(dense[node, :] == 0)
        assert torch.all(dense[:, node] == 0)


@pytest.mark.parametrize(
    "x, edge_index, with_mediators, expected_num_edges",
    [
        pytest.param(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([[0, 1], [0, 0]]),
            False,
            1,  # One hyperedge, so one graph edge, no mediators to create additional edges
            id="single_hyperedge_2_nodes_no_mediators",
        ),
        pytest.param(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([[0, 1], [0, 0]]),
            True,
            # Only 2 nodes and both are extremes (argmin/argmax)
            # No mediators exist (mediators are nodes that are neither argmin nor argmax)
            # So, with mediators enabled and no mediators -> 0 edges produced
            0,
            id="single_hyperedge_2_nodes_with_mediators_produces_no_edges",
        ),
        pytest.param(
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            torch.tensor([[0, 1, 2], [0, 0, 0]]),
            False,
            1,  # One hyperedge, so one graph edge, no mediators to create additional edges
            id="single_hyperedge_3_nodes_no_mediators",
        ),
        pytest.param(
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            torch.tensor([[0, 1, 2], [0, 0, 0]]),
            True,
            2,  # If argmin = 0 and argmax = 2, mediator 1 creates 2 edges [0,1] and [1,2]
            id="single_hyperedge_3_nodes_with_mediators",
        ),
        pytest.param(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [1.0, 1.0]]),
            torch.tensor([[0, 1, 2, 3], [0, 0, 0, 0]]),
            True,
            # 2 nodes are extremes (argmin/argmax), 2 are mediators
            # Each mediator connects to both extremes: 2 mediators * 2 edges = 4 edges
            4,
            id="single_hyperedge_4_nodes_with_mediators",
        ),
        pytest.param(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [1.0, 1.0]]),
            torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]]),
            False,
            # Two hyperedges, each with 2 nodes -> 2 graph edges,
            # there are no mediators to create additional edges
            2,
            id="two_hyperedges_no_mediators",
        ),
        pytest.param(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.5, 0.5, 0.0],
                    [0.0, 0.5, 0.5],
                ]
            ),
            torch.tensor([[0, 1, 2, 2, 3, 4], [0, 0, 0, 1, 1, 1]]),
            True,
            # Hyperedge 0 has 3 nodes -> 1 mediator -> 2 edges
            # Hyperedge 1 has 3 nodes -> 1 mediator -> 2 edges
            # -> 4 edges, there are no mediators to create additional edges
            4,
            id="two_hyperedges_3_nodes_each_with_mediators",
        ),
    ],
)
def test_reduce_to_graph_edge_count(x, edge_index, with_mediators, expected_num_edges):
    result = reduce_to_graph_edge_index(
        x, edge_index, with_mediators=with_mediators, remove_selfloops=False
    )

    assert result.shape[1] == expected_num_edges


@pytest.mark.parametrize(
    "x, edge_index",
    [
        pytest.param(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([[0, 1], [0, 0]]),
            id="2_nodes_1_hyperedge",
        ),
        pytest.param(
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            torch.tensor([[0, 1, 2], [0, 0, 0]]),
            id="3_nodes_1_hyperedge",
        ),
        pytest.param(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [1.0, 1.0]]),
            torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]]),
            id="4_nodes_2_hyperedges",
        ),
    ],
)
def test_reduce_to_graph_output_has_two_rows(x, edge_index):
    result = reduce_to_graph_edge_index(x, edge_index)

    assert result.shape[0] == 2


def test_reduce_to_graph_output_dtype_is_long():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    edge_index = torch.tensor([[0, 1], [0, 0]])

    result = reduce_to_graph_edge_index(x, edge_index)

    assert result.dtype == torch.long


def test_reduce_to_graph_output_nodes_are_within_bounds():
    """All node indices in the output are valid indices from the input node set."""
    x = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.0]]
    )
    edge_index = torch.tensor([[0, 1, 2, 1, 2, 3], [0, 0, 0, 1, 1, 1]])

    result = reduce_to_graph_edge_index(x, edge_index)
    num_nodes = x.shape[0]

    assert result.min() >= 0
    assert result.max() < num_nodes


def test_reduce_to_graph_removes_selfloops():
    # Duplicate node in hyperedge forces a self-loop: projections are identical,
    # so argmax and argmin both select index 0, producing edge [0, 0].
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    edge_index = torch.tensor([[0, 0], [0, 0]])

    result = reduce_to_graph_edge_index(x, edge_index, remove_selfloops=True)

    # Either zero or one edge remains, the reason why one edge may remain is that
    # after removing self-loops, there could be multiple hyperedges projecting to
    # the same graph edge, which would be kept as a single edge
    # Example: hyperedges [[0,1,1],[0,0,2]] both project to graph edge [0,2]
    assert result.shape[1] <= 1

    if result.shape[1] > 0:
        # If any edges remain, check that no self-loops are present
        assert not torch.any(result[0] == result[1]).item()


def test_reduce_to_graph_keeps_selfloops_when_disabled():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    edge_index = torch.tensor([[0, 0], [0, 0]])

    result = reduce_to_graph_edge_index(x, edge_index, remove_selfloops=False)

    assert result.shape[1] == 1  # One node, one hyperedge
    assert result[0, 0] == result[1, 0]  # Self-loop edge [0, 0] is preserved


def test_reduce_to_graph_raises_on_single_node_hyperedge():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    edge_index = torch.tensor([[0], [0]])

    with pytest.raises(
        ValueError, match="The number of vertices in an hyperedge must be >= 2."
    ):
        reduce_to_graph_edge_index(x, edge_index)
