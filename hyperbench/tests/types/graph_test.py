import pytest
import torch

from hyperbench.types.graph import Graph


@pytest.fixture
def mock_single_edge_graph():
    return Graph([[0, 1]])


@pytest.fixture
def mock_linear_graph():
    # Linear graph: 0-1-2-3
    return Graph([[0, 1], [1, 2], [2, 3]])


@pytest.fixture
def mock_graph_with_only_selfloops():
    return Graph([[0, 0], [1, 1]])


@pytest.fixture
def mock_graph_with_one_selfloop():
    return Graph([[0, 1], [1, 1], [2, 3]])


@pytest.mark.parametrize(
    "graph, expected_edges",
    [
        pytest.param(Graph([]), [], id="empty_graph"),
        pytest.param(Graph([[0, 1]]), [[0, 1]], id="single_edge"),
        pytest.param(
            Graph([[0, 1], [1, 2], [2, 3]]),
            [[0, 1], [1, 2], [2, 3]],
            id="linear_graph",
        ),
    ],
)
def test_init_edges(graph, expected_edges):
    assert graph.edges == expected_edges


@pytest.mark.parametrize(
    "graph, expected_num_nodes",
    [
        pytest.param(Graph([]), 0, id="empty_graph"),
        pytest.param(Graph([[0, 1]]), 2, id="single_edge"),
        pytest.param(Graph([[0, 0]]), 1, id="single_edge_selfloop"),
        pytest.param(Graph([[0, 1], [1, 2], [2, 3]]), 4, id="linear_graph"),
        pytest.param(Graph([[0, 0], [1, 1]]), 2, id="only_selfloops"),
        pytest.param(Graph([[0, 1], [1, 1], [2, 3]]), 4, id="one_selfloop"),
        pytest.param(Graph([[0, 1], [2, 3]]), 4, id="disconnected_graph"),
        pytest.param(
            Graph([[0, 1], [0, 1], [1, 2]]),
            3,
            id="duplicate_edges",
        ),
        pytest.param(Graph([[0, 1], [0, 2], [1, 2]]), 3, id="complete_graph"),
    ],
)
def test_num_nodes(graph, expected_num_nodes):
    assert graph.num_nodes == expected_num_nodes


@pytest.mark.parametrize(
    "graph, expected_num_edges",
    [
        pytest.param(Graph([]), 0, id="empty_graph"),
        pytest.param(Graph([[0, 1]]), 1, id="single_edge"),
        pytest.param(Graph([[0, 0]]), 1, id="single_edge_selfloop"),
        pytest.param(Graph([[0, 1], [1, 2], [2, 3]]), 3, id="linear_graph"),
        pytest.param(Graph([[0, 0], [1, 1]]), 2, id="only_selfloops"),
        pytest.param(Graph([[0, 1], [1, 1], [2, 3]]), 3, id="one_selfloop"),
        pytest.param(Graph([[0, 1], [2, 3]]), 2, id="disconnected_graph"),
        pytest.param(
            Graph([[0, 1], [0, 1], [1, 2]]),
            3,
            id="duplicate_edges",
        ),
        pytest.param(Graph([[0, 1], [0, 2], [1, 2]]), 3, id="complete_graph"),
    ],
)
def test_num_edges(graph, expected_num_edges):
    assert graph.num_edges == expected_num_edges


@pytest.mark.parametrize(
    "graph, expected_edges_after_removal",
    [
        pytest.param(Graph([]), [], id="empty_graph"),
        pytest.param(
            Graph([[0, 1], [2, 3]]),
            [[0, 1], [2, 3]],
            id="no_selfloops",
        ),
        pytest.param(Graph([[0, 0]]), [], id="one_edge_one_selfloop"),
        pytest.param(Graph([[0, 1], [1, 1]]), [[0, 1]], id="one_selfloop"),
        pytest.param(
            Graph([[0, 0], [1, 1], [2, 2]]),
            [],
            id="all_selfloops",
        ),
        pytest.param(
            Graph([[0, 1], [1, 2], [2, 2]]),
            [[0, 1], [1, 2]],
            id="mixed_edges",
        ),
        pytest.param(
            Graph([[0, 0], [0, 1], [1, 1], [1, 2]]),
            [[0, 1], [1, 2]],
            id="mixed_edges_multiple_selfloops",
        ),
        pytest.param(
            Graph([[0, 0], [1, 1], [2, 2], [3, 4]]),
            [[3, 4]],
            id="multiple_consecutive_selfloops",
        ),
    ],
)
def test_remove_selfloops(graph, expected_edges_after_removal):
    graph.remove_selfloops()
    assert graph.edges == expected_edges_after_removal


def test_remove_selfloops_preserves_order():
    graph = Graph([[0, 1], [1, 1], [2, 3], [3, 3], [4, 5]])
    graph.remove_selfloops()
    assert graph.edges == [[0, 1], [2, 3], [4, 5]]


@pytest.mark.parametrize(
    "graph, expected_edge_index",
    [
        pytest.param(
            Graph([]),
            torch.empty((2, 0), dtype=torch.long),
            id="empty_graph",
        ),
        pytest.param(
            Graph([[0, 1]]),
            torch.tensor([[0], [1]], dtype=torch.long),
            id="single_edge",
        ),
        pytest.param(
            Graph([[0, 1], [1, 2]]),
            torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            id="multiple_edges",
        ),
        pytest.param(
            Graph([[0, 1], [1, 2], [2, 3]]),
            torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
            id="linear_graph",
        ),
        pytest.param(
            Graph([[0, 0], [1, 1]]),
            torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
            id="only_selfloops",
        ),
        pytest.param(
            Graph([[0, 1], [0, 1], [1, 2]]),
            torch.tensor([[0, 0, 1], [1, 1, 2]], dtype=torch.long),
            id="duplicate_edges",
        ),
    ],
)
def test_to_edge_index(graph, expected_edge_index):
    edge_index = graph.to_edge_index()
    assert torch.equal(edge_index, expected_edge_index)


def test_to_edge_index_returns_long_dtype(mock_single_edge_graph):
    edge_index = mock_single_edge_graph.to_edge_index()
    assert edge_index.dtype == torch.long


def test_to_edge_index_large_graph():
    edges = [[i, i + 1] for i in range(1000)]
    graph = Graph(edges)

    edge_index = graph.to_edge_index()

    assert edge_index.shape == (2, 1000)
    assert edge_index[0, 0] == 0
    assert edge_index[1, -1] == 1000


def test_to_edge_index_does_not_modify_graph(mock_linear_graph):
    original_edges = [edge[:] for edge in mock_linear_graph.edges]
    _ = mock_linear_graph.to_edge_index()

    assert mock_linear_graph.edges == original_edges


def test_to_edge_index_is_contiguous(mock_single_edge_graph):
    """
    Test that to_edge_index returns a contiguous tensor.

    Example:
        If edges = [[0, 1]], then edge_index = [[0], [1]] should be contiguous.
        If edges = [[0, 1], [1, 2], [2, 3]], then edge_index = [[0, 1, 2], [1, 2, 3]] should be contiguous.
    """
    edge_index = mock_single_edge_graph.to_edge_index()
    assert edge_index.is_contiguous()


def test_to_edge_index_before_and_after_removal_all_selfloops(
    mock_graph_with_only_selfloops,
):
    edge_index_before = mock_graph_with_only_selfloops.to_edge_index()
    assert edge_index_before.shape == (2, 2)

    mock_graph_with_only_selfloops.remove_selfloops()
    edge_index_after = mock_graph_with_only_selfloops.to_edge_index()

    expected = torch.tensor([], dtype=torch.long).reshape(2, 0)

    assert edge_index_after.shape == (2, 0)
    assert torch.equal(edge_index_after, expected)


def test_to_edge_index_before_and_after_removal_one_selfloops(
    mock_graph_with_one_selfloop,
):
    edge_index_before = mock_graph_with_one_selfloop.to_edge_index()
    assert edge_index_before.shape == (2, 3)

    mock_graph_with_one_selfloop.remove_selfloops()
    edge_index_after = mock_graph_with_one_selfloop.to_edge_index()

    expected = torch.tensor([[0, 2], [1, 3]])

    assert edge_index_after.shape == (2, 2)
    assert torch.equal(edge_index_after, expected)


def test_bidirectional_edges():
    graph = Graph([[0, 1], [1, 0]])
    assert graph.num_edges == 2
    assert graph.num_nodes == 2

    edge_index = graph.to_edge_index()

    expected = torch.tensor([[0, 1], [1, 0]])

    assert torch.equal(edge_index, expected)


def test_star_graph():
    """Test star graph (all edges connected to central node)."""
    graph = Graph([[0, 1], [0, 2], [0, 3], [0, 4]])
    assert graph.num_nodes == 5
    assert graph.num_edges == 4

    edge_index = graph.to_edge_index()

    assert edge_index.shape == (2, 4)


def test_cyclic_graph():
    """Test cyclic graph (a closed loop)."""
    graph = Graph([[0, 1], [1, 2], [2, 3], [3, 0]])
    assert graph.num_nodes == 4
    assert graph.num_edges == 4

    edge_index = graph.to_edge_index()

    assert edge_index.shape == (2, 4)


def test_from_directed_to_undirected_edge_index_single_directed_edge():
    """A single directed edge (0 -> 1) should produce bidirectional edges."""
    edge_index = torch.tensor([[0], [1]])

    result = Graph.from_directed_to_undirected_edge_index(edge_index)
    edges = set(zip(result[0].tolist(), result[1].tolist()))

    # Should contain both (0, 1) and (1, 0)
    assert (0, 1) in edges
    assert (1, 0) in edges
    assert len(edges) == 2


def test_from_directed_to_undirected_edge_index_already_undirected_does_not_create_duplicates():
    edge_index = torch.tensor([[0, 1], [1, 0]])

    result = Graph.from_directed_to_undirected_edge_index(edge_index)
    edges = set(zip(result[0].tolist(), result[1].tolist()))

    assert edges == {(0, 1), (1, 0)}


def test_from_directed_to_undirected_edge_index_removes_duplicate_edges():
    edge_index = torch.tensor([[0, 0, 1], [1, 1, 0]])

    result = Graph.from_directed_to_undirected_edge_index(edge_index)
    edges = set(zip(result[0].tolist(), result[1].tolist()))

    assert edges == {(0, 1), (1, 0)}


def test_from_directed_to_undirected_edge_index_triangle_directed():
    """
    A directed triangle should become a bidirectional triangle.

    Example:
        Directed cycle: 0 -> 1 -> 2 -> 0
        Bidirectional traingle: 0 <-> 1 <-> 2 <-> 0
    """
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])

    result = Graph.from_directed_to_undirected_edge_index(edge_index)
    edges = set(zip(result[0].tolist(), result[1].tolist()))

    bidirectional_triangle = {(0, 1), (1, 0), (1, 2), (2, 1), (2, 0), (0, 2)}
    assert edges == bidirectional_triangle


def test_from_directed_to_undirected_edge_index_preserves_selfloops_in_input():
    edge_index = torch.tensor([[0, 1, 1], [1, 0, 1]])  # (1, 1) is a self-loop

    result = Graph.from_directed_to_undirected_edge_index(edge_index)
    edges = set(zip(result[0].tolist(), result[1].tolist()))

    assert (1, 1) in edges


def test_from_directed_to_undirected_edge_index_empty_edge_index_returns_empty_tensor():
    edge_index = torch.tensor([[], []])

    result = Graph.from_directed_to_undirected_edge_index(edge_index)

    assert result.shape == (2, 0)


def test_from_directed_to_undirected_edge_index_with_selfloops_adds_all_selfloops():
    edge_index = torch.tensor([[0, 1], [1, 2]])

    result = Graph.from_directed_to_undirected_edge_index(
        edge_index, with_selfloops=True
    )
    edges = set(zip(result[0].tolist(), result[1].tolist()))

    # Should have self-loops for nodes 0, 1, 2 (inferred from max index)
    assert (0, 0) in edges
    assert (1, 1) in edges
    assert (2, 2) in edges


def test_from_directed_to_undirected_edge_index_with_selfloops_does_not_duplicate_selfloops():
    edge_index = torch.tensor([[0, 1], [1, 1]])  # (1, 1) is already a self-loop

    result = Graph.from_directed_to_undirected_edge_index(
        edge_index, with_selfloops=True
    )
    edges = list(zip(result[0].tolist(), result[1].tolist()))

    assert (0, 0) in edges
    assert (1, 1) in edges


@pytest.mark.parametrize(
    "with_selfloops",
    [
        pytest.param(True, id="with_selfloops"),
        pytest.param(False, id="without_selfloops"),
    ],
)
def test_from_directed_to_undirected_edge_index_preserves_device(with_selfloops):
    edge_index = torch.tensor([[0], [1]], device="cpu")

    result = Graph.from_directed_to_undirected_edge_index(
        edge_index, with_selfloops=with_selfloops
    )

    assert result.device == edge_index.device


def test_from_directed_to_undirected_edge_index_disconnected_components():
    # Two disconnected components: (0, 1) and (2, 3)
    edge_index = torch.tensor([[0, 2], [1, 3]])

    result = Graph.from_directed_to_undirected_edge_index(edge_index)
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
def test_from_directed_to_undirected_edge_index_edge_count(
    edge_index, expected_num_undirected_edges
):
    result = Graph.from_directed_to_undirected_edge_index(edge_index)

    assert result.shape[1] == expected_num_undirected_edges


def test_from_directed_to_undirected_edge_index_dtype_preserved():
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

    result = Graph.from_directed_to_undirected_edge_index(edge_index)

    assert result.dtype == edge_index.dtype
