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
def mock_graph_with_only_self_loops():
    return Graph([[0, 0], [1, 1]])


@pytest.fixture
def mock_graph_with_one_self_loop():
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
        pytest.param(Graph([[0, 0]]), 1, id="single_edge_self_loop"),
        pytest.param(Graph([[0, 1], [1, 2], [2, 3]]), 4, id="linear_graph"),
        pytest.param(Graph([[0, 0], [1, 1]]), 2, id="only_self_loops"),
        pytest.param(Graph([[0, 1], [1, 1], [2, 3]]), 4, id="one_self_loop"),
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
        pytest.param(Graph([[0, 0]]), 1, id="single_edge_self_loop"),
        pytest.param(Graph([[0, 1], [1, 2], [2, 3]]), 3, id="linear_graph"),
        pytest.param(Graph([[0, 0], [1, 1]]), 2, id="only_self_loops"),
        pytest.param(Graph([[0, 1], [1, 1], [2, 3]]), 3, id="one_self_loop"),
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
            id="no_self_loops",
        ),
        pytest.param(Graph([[0, 0]]), [], id="one_edge_one_self_loop"),
        pytest.param(Graph([[0, 1], [1, 1]]), [[0, 1]], id="one_self_loop"),
        pytest.param(
            Graph([[0, 0], [1, 1], [2, 2]]),
            [],
            id="all_self_loops",
        ),
        pytest.param(
            Graph([[0, 1], [1, 2], [2, 2]]),
            [[0, 1], [1, 2]],
            id="mixed_edges",
        ),
        pytest.param(
            Graph([[0, 0], [0, 1], [1, 1], [1, 2]]),
            [[0, 1], [1, 2]],
            id="mixed_edges_multiple_self_loops",
        ),
        pytest.param(
            Graph([[0, 0], [1, 1], [2, 2], [3, 4]]),
            [[3, 4]],
            id="multiple_consecutive_self_loops",
        ),
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
            id="only_self_loops",
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
