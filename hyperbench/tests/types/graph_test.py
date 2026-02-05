import pytest
import torch

from hyperbench.types.graph import Graph


@pytest.fixture
def mock_empty_graph():
    return Graph([])


@pytest.fixture
def mock_single_edge_graph():
    return Graph([[0, 1]])


@pytest.fixture
def mock_linear_graph():
    # Linear graph: 0-1-2-3
    return Graph([[0, 1], [1, 2], [2, 3]])


@pytest.fixture
def mock_graph_with_only_self_loops():
    return Graph([[0, 0], [1, 1]])


@pytest.fixture
def mock_graph_with_one_self_loop():
    return Graph([[0, 1], [1, 1], [2, 3]])


def test_init_empty_edges(mock_empty_graph):
    assert mock_empty_graph.edges == []


def test_init_single_edge(mock_single_edge_graph):
    assert mock_single_edge_graph.edges == [[0, 1]]


def test_init_multiple_edges(mock_linear_graph):
    assert mock_linear_graph.edges == [[0, 1], [1, 2], [2, 3]]


@pytest.mark.parametrize(
    "graph, expected_num_nodes",
    [
        (Graph([]), 0),  # Empty graph
        (Graph([[0, 1]]), 2),  # Single edge
        (Graph([[0, 0]]), 1),  # Single edge, self-loop
        (Graph([[0, 1], [1, 2], [2, 3]]), 4),  # Linear graph
        (Graph([[0, 0], [1, 1]]), 2),  # Graph with only self-loops
        (Graph([[0, 1], [1, 1], [2, 3]]), 4),  # Graph with one self-loop
        (Graph([[0, 1], [2, 3]]), 4),  # Disconnected graph
        (Graph([[0, 1], [0, 1], [1, 2]]), 3),  # Graph with duplicate edges
        (Graph([[0, 1], [0, 2], [1, 2]]), 3),  # Complete graph
    ],
)
def test_num_nodes(graph, expected_num_nodes):
    assert graph.num_nodes == expected_num_nodes


@pytest.mark.parametrize(
    "graph, expected_num_edges",
    [
        (Graph([]), 0),  # Empty graph
        (Graph([[0, 1]]), 1),  # Single edge
        (Graph([[0, 0]]), 1),  # Single edge, self-loop
        (Graph([[0, 1], [1, 2], [2, 3]]), 3),  # Linear graph
        (Graph([[0, 0], [1, 1]]), 2),  # Graph with only self-loops
        (Graph([[0, 1], [1, 1], [2, 3]]), 3),  # Graph with one self-loop
        (Graph([[0, 1], [2, 3]]), 2),  # Disconnected graph
        (Graph([[0, 1], [0, 1], [1, 2]]), 3),  # Graph with duplicate edges
        (Graph([[0, 1], [0, 2], [1, 2]]), 3),  # Complete graph
    ],
)
def test_num_edges(graph, expected_num_edges):
    assert graph.num_edges == expected_num_edges


@pytest.mark.parametrize(
    "graph, expected_edges_after_removal",
    [
        (Graph([]), []),  # Empty graph
        (Graph([[0, 1], [2, 3]]), [[0, 1], [2, 3]]),  # No self-loops
        (Graph([[0, 0]]), []),  # One edge, one self-loop
        (Graph([[0, 1], [1, 1]]), [[0, 1]]),  # One self-loop
        (Graph([[0, 0], [1, 1], [2, 2]]), []),  # All self-loops
        (Graph([[0, 1], [1, 2], [2, 2]]), [[0, 1], [1, 2]]),  # Mixed edges
        (
            Graph([[0, 0], [0, 1], [1, 1], [1, 2]]),
            [[0, 1], [1, 2]],
        ),  # Mixed edges with multiple self-loops
        (
            Graph([[0, 0], [1, 1], [2, 2], [3, 4]]),
            [[3, 4]],
        ),  # Multiple consecutive self-loops
    ],
)
def test_remove_self_loops(graph, expected_edges_after_removal):
    """Test removing self-loops for various graph configurations."""
    graph.remove_self_loops()
    assert graph.edges == expected_edges_after_removal


def test_remove_self_loops_preserves_order():
    graph = Graph([[0, 1], [1, 1], [2, 3], [3, 3], [4, 5]])
    graph.remove_self_loops()
    assert graph.edges == [[0, 1], [2, 3], [4, 5]]


@pytest.mark.parametrize(
    "graph, expected_edge_index",
    [
        (Graph([]), torch.empty((2, 0), dtype=torch.long)),  # Empty graph
        (Graph([[0, 1]]), torch.tensor([[0], [1]], dtype=torch.long)),  # Single edge
        (
            Graph([[0, 1], [1, 2]]),
            torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        ),  # Multiple edges
        (
            Graph([[0, 1], [1, 2], [2, 3]]),
            torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        ),  # Linear graph
        (
            Graph([[0, 0], [1, 1]]),
            torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
        ),  # Graph with only self-loops
        (
            Graph([[0, 1], [0, 1], [1, 2]]),
            torch.tensor([[0, 0, 1], [1, 1, 2]], dtype=torch.long),
        ),  # Graph with duplicate edges
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
    """Test that to_edge_index returns a contiguous tensor."""
    edge_index = mock_single_edge_graph.to_edge_index()
    assert edge_index.is_contiguous()


def test_to_edge_index_before_and_after_removal_all_self_loops(
    mock_graph_with_only_self_loops,
):
    edge_index_before = mock_graph_with_only_self_loops.to_edge_index()
    assert edge_index_before.shape == (2, 2)

    mock_graph_with_only_self_loops.remove_self_loops()
    edge_index_after = mock_graph_with_only_self_loops.to_edge_index()

    expected = torch.tensor([], dtype=torch.long).reshape(2, 0)

    assert edge_index_after.shape == (2, 0)
    assert torch.equal(edge_index_after, expected)


def test_to_edge_index_before_and_after_removal_one_self_loops(
    mock_graph_with_one_self_loop,
):
    edge_index_before = mock_graph_with_one_self_loop.to_edge_index()
    assert edge_index_before.shape == (2, 3)

    mock_graph_with_one_self_loop.remove_self_loops()
    edge_index_after = mock_graph_with_one_self_loop.to_edge_index()

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
