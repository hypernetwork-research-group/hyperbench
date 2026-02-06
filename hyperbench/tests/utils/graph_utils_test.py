import pytest
import torch
import warnings

from hyperbench.utils import (
    get_sparse_adjacency_matrix,
    get_sparse_normalized_degree_matrix,
    get_sparse_normalized_laplacian,
    reduce_to_graph_edge_index,
    smoothing_with_gcn_laplacian_matrix,
    to_undirected_edge_index,
)


@pytest.fixture(autouse=True)
def suppress_sparse_csr_warning():
    """
    Suppress PyTorch sparse CSR beta warning.
    It could be avoided by doing sparse @ dense, as it doesn't trigger CSR warning.
    However, it's inefficient for large graphs.

    Example:
        ```
        AD = torch.sparse.mm(A, D.to_dense())
        L = torch.sparse.mm(D, AD).to_sparse_coo()
        ```
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sparse CSR tensor support is in beta state",
            category=UserWarning,
        )
        yield


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


def test_get_sparse_normalized_degree_matrix_returns_sparse_tensor():
    edge_index = torch.tensor([[0, 1], [1, 0]])
    result = get_sparse_normalized_degree_matrix(edge_index, num_nodes=2)

    assert result.is_sparse


@pytest.mark.parametrize(
    "edge_index, num_nodes",
    [
        pytest.param(torch.tensor([[0, 1], [1, 0]]), 2, id="2_nodes"),
        pytest.param(torch.tensor([[0, 1, 2], [1, 2, 0]]), 4, id="4_nodes_3_edges"),
        pytest.param(
            torch.tensor([[], []], dtype=torch.long), 5, id="5_nodes_no_edges"
        ),
    ],
)
def test_get_sparse_normalized_degree_matrix_shape(edge_index, num_nodes):
    result = get_sparse_normalized_degree_matrix(edge_index, num_nodes=num_nodes)

    assert result.shape == (num_nodes, num_nodes)


def test_get_sparse_normalized_degree_matrix_is_diagonal():
    """All non-zero entries are on the diagonal."""
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    result = get_sparse_normalized_degree_matrix(edge_index, num_nodes=3)
    dense = result.to_dense()

    # Off-diagonal entries should be zero
    for i in range(3):
        for j in range(3):
            if i != j:
                assert dense[i, j] == 0


@pytest.mark.parametrize(
    "edge_index, num_nodes, expected_diagonal",
    [
        pytest.param(
            torch.tensor([[0, 1], [1, 0]]),
            2,
            [1.0, 1.0],  # degree 1 -> 1^-0.5 = 1
            id="degree_1_each",
        ),
        pytest.param(
            torch.tensor([[0, 0, 1], [1, 2, 0]]),
            3,
            # degrees [2, 1, 0] -> [2**-0.5 == 1 / 2**0.5, 1.0, 0] -> [0.707, 1, 0]
            [1 / (2**0.5), 1.0, 0.0],
            id="mixed_degrees",
        ),
        pytest.param(
            torch.tensor([[0, 0, 0, 0], [1, 2, 3, 4]]),
            5,
            [0.5, 0.0, 0.0, 0.0, 0.0],  # degree 4 -> 4^-0.5 = 0.5, others are isolated
            id="single_hub_node",
        ),
    ],
)
def test_get_sparse_normalized_degree_matrix_diagonal_values(
    edge_index, num_nodes, expected_diagonal
):
    result = get_sparse_normalized_degree_matrix(edge_index, num_nodes=num_nodes)
    dense = result.to_dense()

    for i, expected_val in enumerate(expected_diagonal):
        assert torch.isclose(dense[i, i], torch.tensor(expected_val), atol=1e-6)


def test_get_sparse_normalized_degree_matrix_isolated_nodes_are_zero():
    """Isolated nodes (degree 0) have 0 on diagonal, not inf."""
    edge_index = torch.tensor([[0], [1]])

    result = get_sparse_normalized_degree_matrix(edge_index, num_nodes=4)
    dense = result.to_dense()

    # Nodes 2 and 3 are isolated
    assert dense[2, 2] == 0
    assert dense[3, 3] == 0
    # No inf values
    assert not torch.any(torch.isinf(dense))


def test_get_sparse_normalized_degree_matrix_empty_edge_index():
    """Empty edge_index produces all-zero matrix (all nodes isolated)."""
    edge_index = torch.tensor([[], []], dtype=torch.long)

    result = get_sparse_normalized_degree_matrix(edge_index, num_nodes=3)
    dense = result.to_dense()

    assert torch.all(dense == 0)


def test_get_sparse_normalized_degree_matrix_preserves_device():
    edge_index = torch.tensor([[0], [1]], device="cpu")

    result = get_sparse_normalized_degree_matrix(edge_index, num_nodes=2)

    assert result.device == edge_index.device


def test_get_sparse_normalized_laplacian_returns_sparse_tensor():
    edge_index = torch.tensor([[0, 1], [1, 0]])

    result = get_sparse_normalized_laplacian(edge_index)

    assert result.is_sparse


@pytest.mark.parametrize(
    "edge_index, num_nodes",
    [
        pytest.param(torch.tensor([[0, 1], [1, 0]]), 2, id="2_nodes"),
        pytest.param(torch.tensor([[0, 1, 2], [1, 2, 0]]), 4, id="4_nodes"),
        pytest.param(torch.tensor([[0, 1], [1, 0]]), None, id="2_nodes_inferred"),
    ],
)
def test_get_sparse_normalized_laplacian_shape(edge_index, num_nodes):
    result = get_sparse_normalized_laplacian(edge_index, num_nodes=num_nodes)
    expected_num_nodes = num_nodes if num_nodes else edge_index.max().item() + 1

    assert result.shape == (expected_num_nodes, expected_num_nodes)


def test_get_sparse_normalized_laplacian_is_symmetric():
    """GCN Laplacian L = D^-1/2 * A * D^-1/2 is symmetric."""
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])

    result = get_sparse_normalized_laplacian(edge_index)
    dense = result.to_dense()

    assert torch.allclose(dense, dense.T, atol=1e-6)


def test_get_sparse_normalized_laplacian_self_loop_diagonal():
    """Single node graph has diagonal value 1 (self-loop normalized)."""
    edge_index = torch.tensor([[0], [0]])

    result = get_sparse_normalized_laplacian(edge_index, num_nodes=1)
    dense = result.to_dense()

    # Self-loop only: degree = 1, so D^-1/2 * A * D^-1/2 = 1 * 1 * 1 = 1
    assert torch.isclose(dense[0, 0], torch.tensor(1.0), atol=1e-6)


@pytest.mark.parametrize(
    "edge_index, num_nodes, expected_row_sum",
    [
        pytest.param(
            torch.tensor([[0, 1], [1, 0]]),
            2,
            1.0,  # Each node has degree 2 (edge + self-loop), diagonal = 1/2 each
            id="connected_graph",
        ),
        pytest.param(
            torch.tensor([[0, 1, 2], [1, 2, 0]]),
            3,
            1.0,  # Triangle: each node degree 3 (2 edges + self-loop), diag = 1/3 each
            id="triangle_graph",
        ),
    ],
)
def test_get_sparse_normalized_laplacian_row_sum(
    edge_index, num_nodes, expected_row_sum
):
    """
    For connected graphs with self-loops, GCN normalization makes the
    laplacian matrix row-stochastic: every row sums to 1.0.
    """
    result = get_sparse_normalized_laplacian(edge_index, num_nodes=num_nodes)
    dense = result.to_dense()

    # Each row should sum to 1 for connected graphs with self-loops
    for i in range(num_nodes):
        assert torch.isclose(dense[i].sum(), torch.tensor(expected_row_sum), atol=1e-6)


def test_get_sparse_normalized_laplacian_preserves_device():
    edge_index = torch.tensor([[0, 1], [1, 0]], device="cpu")

    result = get_sparse_normalized_laplacian(edge_index)

    assert result.device == edge_index.device


def test_get_sparse_normalized_laplacian_no_nan_or_inf():
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])

    result = get_sparse_normalized_laplacian(edge_index, num_nodes=4)
    dense = result.to_dense()

    assert not torch.any(torch.isnan(dense))
    assert not torch.any(torch.isinf(dense))


def test_get_sparse_normalized_laplacian_has_0_for_isolated_nodes():
    edge_index = torch.tensor([[0], [1]])

    result = get_sparse_normalized_laplacian(edge_index, num_nodes=4)
    dense = result.to_dense()

    assert torch.all(dense[2, :] == 0)
    assert torch.all(dense[:, 2] == 0)
    assert torch.all(dense[3, :] == 0)
    assert torch.all(dense[:, 3] == 0)


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


@pytest.mark.parametrize(
    "num_nodes, num_features",
    [
        pytest.param(2, 2, id="2x2"),
        pytest.param(3, 4, id="3x4"),
        pytest.param(5, 1, id="5x1"),
        pytest.param(10, 8, id="10x8"),
    ],
)
def test_smoothing_with_gcn_laplacian_output_shape_matches_x_shape(
    num_nodes, num_features
):
    """Output shape should match input node feature matrix X shape (|V|, C)."""
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.tensor([[i, (i + 1) % num_nodes] for i in range(num_nodes)]).T

    laplacian = get_sparse_normalized_laplacian(edge_index, num_nodes=num_nodes)

    result = smoothing_with_gcn_laplacian_matrix(x, laplacian)

    assert result.shape == x.shape


def test_smoothing_with_gcn_laplacian_with_identity_laplacian_returns_original_x():
    """Smoothing with identity laplacian should return the original features."""
    num_nodes = 3
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    # Create identity matrix as sparse tensor
    indices = torch.arange(num_nodes).unsqueeze(0).repeat(2, 1)
    values = torch.ones(num_nodes)
    identity_laplacian = torch.sparse_coo_tensor(
        indices, values, size=(num_nodes, num_nodes)
    )

    result = smoothing_with_gcn_laplacian_matrix(x, identity_laplacian)

    assert torch.allclose(result, x, atol=1e-6)


def test_smoothing_with_gcn_laplacian_zero_features():
    """Zero features should remain zero after smoothing."""
    x = torch.zeros(3, 2)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    laplacian = get_sparse_normalized_laplacian(edge_index, num_nodes=3)

    result = smoothing_with_gcn_laplacian_matrix(x, laplacian)
    print(result)

    assert torch.allclose(result, torch.zeros_like(x), atol=1e-6)


def test_smoothing_with_gcn_laplacian_single_node_returns_original_x():
    """Single node with self-loop should return the original features."""
    x = torch.tensor([[1.0, 2.0]])
    edge_index = torch.tensor([[0], [0]])  # Self-loop
    laplacian = get_sparse_normalized_laplacian(edge_index, num_nodes=1)

    result = smoothing_with_gcn_laplacian_matrix(x, laplacian)
    print(laplacian.to_dense(), result)

    # Single node with self-loop has L[0,0] = 1, so result = 1 * x = x
    # as the laplacian is [[1.0]], so:
    # result = L @ x = [[1.0]] @ [[1.0, 2.0]] = [[1.0 * 1.0, 1.0 * 2.0]] = [[1.0, 2.0]] = x
    assert torch.allclose(result, x, atol=1e-6)


def test_smoothing_with_gcn_laplacian_preserves_x_device():
    device = torch.device("cpu")

    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device)
    edge_index = torch.tensor([[0, 1], [1, 0]], device=device)
    laplacian = get_sparse_normalized_laplacian(edge_index, num_nodes=2)

    result = smoothing_with_gcn_laplacian_matrix(x, laplacian)

    assert result.device == x.device


def test_smoothing_with_gcn_laplacian_preserves_x_dtype():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    edge_index = torch.tensor([[0, 1], [1, 0]])
    laplacian = get_sparse_normalized_laplacian(edge_index, num_nodes=2)

    result = smoothing_with_gcn_laplacian_matrix(x, laplacian)

    assert result.dtype == x.dtype


def test_smoothing_with_gcn_laplacian_no_nan_or_inf():
    x = torch.randn(5, 3)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    laplacian = get_sparse_normalized_laplacian(edge_index, num_nodes=5)

    result = smoothing_with_gcn_laplacian_matrix(x, laplacian)

    assert not torch.any(torch.isnan(result))
    assert not torch.any(torch.isinf(result))


def test_smoothing_with_gcn_laplacian_returns_expected_x():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    edge_index = torch.tensor([[0, 1], [1, 0]])
    laplacian = get_sparse_normalized_laplacian(edge_index, num_nodes=2)

    result = smoothing_with_gcn_laplacian_matrix(x, laplacian)

    # For 2 nodes with bidirectional edge, GCN adds self-loops, so each node has degree 2.
    # The GCN Laplacian L = D^-1/2 * A_hat * D^-1/2 = [[0.5, 0.5],
    #                                                  [0.5, 0.5]]
    # L @ x = [[0.5*1 + 0.5*3, 0.5*2 + 0.5*4],
    #          [0.5*1 + 0.5*3, 0.5*2 + 0.5*4]]
    #       = [[2.0, 3.0],
    #          [2.0, 3.0]]
    expected = torch.tensor([[2.0, 3.0], [2.0, 3.0]])

    assert torch.allclose(result, expected, atol=1e-6)


def test_smoothing_with_gcn_laplacian_is_equal_for_zero_and_no_drop_rate():
    """drop_rate=0 should produce the same result as no dropout."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    laplacian = get_sparse_normalized_laplacian(edge_index, num_nodes=3)

    result_no_dropout = smoothing_with_gcn_laplacian_matrix(x, laplacian)
    result_zero_dropout = smoothing_with_gcn_laplacian_matrix(
        x, laplacian, drop_rate=0.0
    )

    assert torch.allclose(result_no_dropout, result_zero_dropout, atol=1e-6)


def test_smoothing_with_gcn_laplacian_nonzero_drop_rate_changes_output():
    torch.manual_seed(123)
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    laplacian = get_sparse_normalized_laplacian(edge_index, num_nodes=3)

    result_no_dropout = smoothing_with_gcn_laplacian_matrix(
        x, laplacian.clone(), drop_rate=0.0
    )
    result_with_dropout = smoothing_with_gcn_laplacian_matrix(
        x, laplacian.clone(), drop_rate=0.7
    )

    assert not torch.allclose(result_no_dropout, result_with_dropout, atol=1e-6)


def test_smoothing_with_gcn_laplacian_drop_rate_stochastic():
    """Different seeds should produce different outputs with dropout."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    laplacian = get_sparse_normalized_laplacian(edge_index, num_nodes=3)

    torch.manual_seed(42)
    result1 = smoothing_with_gcn_laplacian_matrix(x, laplacian.clone(), drop_rate=0.5)

    torch.manual_seed(99)
    result2 = smoothing_with_gcn_laplacian_matrix(x, laplacian.clone(), drop_rate=0.5)

    # Different random seeds should produce different dropout masks
    assert not torch.allclose(result1, result2, atol=1e-6)


def test_smoothing_with_gcn_laplacian_influences_connected_nodes():
    """
    Features of connected nodes should be aggregated.
    For a connected graph with GCN normalization, smoothing should mix features from neighbors.
    """
    # Two connected nodes with distinct features
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    edge_index = torch.tensor([[0, 1], [1, 0]])  # Bidirectional edge
    laplacian = get_sparse_normalized_laplacian(edge_index, num_nodes=2)

    result = smoothing_with_gcn_laplacian_matrix(x, laplacian)

    # After smoothing, node 0 should have some of node 1's features and vice versa
    # Row sum of GCN laplacian is 1 for connected graphs, so features are mixed
    assert result[0, 1] > 0  # Node 0 now has some of feature dimension 1 from node 1
    assert result[1, 0] > 0  # Node 1 now has some of feature dimension 0 from node 0


def test_smoothing_with_gcn_laplacian_isolated_nodes_have_zero_features():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    edge_index = torch.tensor([[0], [1]])  # Only nodes 0 and 1 connected
    laplacian = get_sparse_normalized_laplacian(edge_index, num_nodes=3)

    result = smoothing_with_gcn_laplacian_matrix(x, laplacian)

    # Node 2 is isolated, so its output should be zero
    assert torch.allclose(result[2], torch.zeros(2), atol=1e-6)


def test_to_undirected_edge_index_single_directed_edge():
    """A single directed edge (0 -> 1) should produce bidirectional edges."""
    edge_index = torch.tensor([[0], [1]])

    result = to_undirected_edge_index(edge_index)
    edges = set(zip(result[0].tolist(), result[1].tolist()))

    # Should contain both (0, 1) and (1, 0)
    assert (0, 1) in edges
    assert (1, 0) in edges
    assert len(edges) == 2


def test_to_undirected_edge_index_already_undirected_does_not_create_duplicates():
    edge_index = torch.tensor([[0, 1], [1, 0]])

    result = to_undirected_edge_index(edge_index)
    edges = set(zip(result[0].tolist(), result[1].tolist()))

    assert edges == {(0, 1), (1, 0)}


def test_to_undirected_edge_index_removes_duplicate_edges():
    edge_index = torch.tensor([[0, 0, 1], [1, 1, 0]])

    result = to_undirected_edge_index(edge_index)
    edges = set(zip(result[0].tolist(), result[1].tolist()))

    assert edges == {(0, 1), (1, 0)}


def test_to_undirected_edge_index_triangle_directed():
    """
    A directed triangle should become a bidirectional triangle.

    Example:
        Directed cycle: 0 -> 1 -> 2 -> 0
        Bidirectional traingle: 0 <-> 1 <-> 2 <-> 0
    """
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])

    result = to_undirected_edge_index(edge_index)
    edges = set(zip(result[0].tolist(), result[1].tolist()))

    bidirectional_triangle = {(0, 1), (1, 0), (1, 2), (2, 1), (2, 0), (0, 2)}
    assert edges == bidirectional_triangle


def test_to_undirected_edge_index_preserves_self_loops_in_input():
    edge_index = torch.tensor([[0, 1, 1], [1, 0, 1]])  # (1, 1) is a self-loop

    result = to_undirected_edge_index(edge_index)
    edges = set(zip(result[0].tolist(), result[1].tolist()))

    assert (1, 1) in edges


def test_to_undirected_edge_index_empty_edge_index_returns_empty_tensor():
    edge_index = torch.tensor([[], []])

    result = to_undirected_edge_index(edge_index)

    assert result.shape == (2, 0)


def test_to_undirected_edge_index_with_self_loops_adds_all_self_loops():
    edge_index = torch.tensor([[0, 1], [1, 2]])

    result = to_undirected_edge_index(edge_index, with_selfloops=True)
    edges = set(zip(result[0].tolist(), result[1].tolist()))

    # Should have self-loops for nodes 0, 1, 2 (inferred from max index)
    assert (0, 0) in edges
    assert (1, 1) in edges
    assert (2, 2) in edges


def test_to_undirected_edge_index_with_self_loops_does_not_duplicate_self_loops():
    edge_index = torch.tensor([[0, 1], [1, 1]])  # (1, 1) is already a self-loop

    result = to_undirected_edge_index(edge_index, with_selfloops=True)
    edges = list(zip(result[0].tolist(), result[1].tolist()))

    assert (0, 0) in edges
    assert (1, 1) in edges


@pytest.mark.parametrize(
    "with_self_loops",
    [
        pytest.param(True, id="with_self_loops"),
        pytest.param(False, id="without_self_loops"),
    ],
)
def test_to_undirected_edge_index_preserves_device(with_self_loops):
    edge_index = torch.tensor([[0], [1]], device="cpu")

    result = to_undirected_edge_index(edge_index, with_selfloops=with_self_loops)

    assert result.device == edge_index.device


def test_to_undirected_edge_index_disconnected_components():
    # Two disconnected components: (0, 1) and (2, 3)
    edge_index = torch.tensor([[0, 2], [1, 3]])

    result = to_undirected_edge_index(edge_index)
    edges = set(zip(result[0].tolist(), result[1].tolist()))

    expected = {(0, 1), (1, 0), (2, 3), (3, 2)}
    assert edges == expected


@pytest.mark.parametrize(
    "edge_index, expected_num_undirected_edges",
    [
        pytest.param(
            torch.tensor([[0], [1]]),
            2,
            id="single_edge_becomes_two",
        ),
        pytest.param(
            torch.tensor([[0, 1], [1, 0]]),
            2,
            id="bidirectional_stays_two",
        ),
        pytest.param(
            torch.tensor([[0, 1, 2], [1, 2, 0]]),
            6,
            id="directed_triangle_becomes_six",
        ),
        pytest.param(
            torch.tensor([[0, 0], [1, 2]]),
            4,
            id="star_two_edges_becomes_four",
        ),
    ],
)
def test_to_undirected_edge_index_edge_count(edge_index, expected_num_undirected_edges):
    result = to_undirected_edge_index(edge_index)

    assert result.shape[1] == expected_num_undirected_edges


def test_to_undirected_edge_index_dtype_preserved():
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

    result = to_undirected_edge_index(edge_index)

    assert result.dtype == edge_index.dtype
