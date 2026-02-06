import torch

from torch import Tensor
from typing import List, Optional
from hyperbench.types import Graph, Hypergraph

from .sparse_utils import sparse_dropout


def get_sparse_adjacency_matrix(edge_index: Tensor, num_nodes: int) -> Tensor:
    """
    Compute the sparse adjacency matrix from a graph edge index.
    To get the normalized adjacency matrix, add self-loops to the edge_index.


    Args:
        edge_index: Edge index tensor of shape (2, |E|).
        num_nodes: The number of nodes in the graph.

    Returns:
        The sparse adjacency matrix of shape (num_nodes, num_nodes).
    """
    src, dest = edge_index

    # Example: undirected_edge_index = [[0, 1, 2, 3],
    #                                   [1, 0, 3, 2]]
    #         -> adj_values = [1, 1, 1, 1]
    #         -> adj_indices = [[0, 1, 2, 3],
    #                           [1, 0, 3, 2]]
    #                  0  1  2  3
    #         -> A = [[0, 1, 0, 0], 1
    #                 [1, 0, 0, 0], 0
    #                 [0, 0, 0, 1], 3
    #                 [0, 0, 1, 0]] 2
    # Note: We don't have duplicate edges in undirected_edge_index, but
    # even if we did, torch.sparse_coo_tensor would sum them up automatically
    adj_values = torch.ones(src.size(0), device=edge_index.device)
    adj_indices = torch.stack([src, dest], dim=0)
    adj_matrix = torch.sparse_coo_tensor(
        adj_indices, adj_values, (num_nodes, num_nodes)
    )
    return adj_matrix


def get_sparse_normalized_degree_matrix(edge_index: Tensor, num_nodes: int) -> Tensor:
    device = edge_index.device
    src, _ = edge_index

    # Compute degree for each node, initially degree matrix D has all zeros
    degrees: Tensor = torch.zeros(num_nodes, device=device)

    # Example: src = [0, 1, 2, 1], degree_matrix = [0, 0, 0, 0]
    #          -> degree_matrix[0] += 1 = degree_matrix = [1,0,0,0]
    #          -> degree_matrix[1] += 1 = degree_matrix = [1,1,0,0]
    #          -> degree_matrix[2] += 1 = degree_matrix = [1,1,1,0]
    #          -> degree_matrix[1] += 1 = degree_matrix = [1,2,1,0]
    #          -> final degree_matrix = [1,2,1,0]
    degrees.scatter_add_(dim=0, index=src, src=torch.ones(src.size(0), device=device))

    # Compute D^-1/2 == D^-0.5
    degree_inv_sqrt: Tensor = degrees.pow(-0.5)
    # Handle isolated nodes where degree is 0, which lead to inf values in degree_inv_sqrt
    degree_inv_sqrt[degree_inv_sqrt == float("inf")] = 0

    # Convert degree vector to a diagonal sparse normalized matrix D
    # Example: degree_inv_sqrt = [1, 0.707, 1, 0]
    #          -> diagonal_indices = [[0, 1, 2, 3],
    #                                 [0, 1, 2, 3]]
    #                   0  1      2  3
    #          -> D = [[1, 0,     0, 0], 0
    #                  [0, 0.707, 0, 0], 1
    #                  [0, 0,     1, 0], 2
    #                  [0, 0,     0, 0]] 3
    diagonal_indices = torch.arange(num_nodes, device=device).unsqueeze(0).repeat(2, 1)
    degree_matrix = torch.sparse_coo_tensor(
        indices=diagonal_indices, values=degree_inv_sqrt, size=(num_nodes, num_nodes)
    )
    return degree_matrix


def get_sparse_normalized_laplacian(
    edge_index: Tensor,
    num_nodes: Optional[int] = None,
) -> Tensor:
    """
    Compute the sparse Laplacian matrix from a graph edge index.

    The GCN Laplacian is defined as: L_GCN = D_hat^-1/2 * A_hat * D_hat^-1/2,
    where A_hat = A + I (adjacency with self-loops) and D_hat is the degree matrix of A_hat.

    Args:
        edge_index: Edge index tensor of shape (2, |E|).
        num_nodes: The number of nodes in the graph. If ``None``,
            it will be inferred from ``edge_index`` as ``edge_index.max().item() + 1``

    Returns:
        The sparse symmetrically normalized Laplacian matrix of shape (num_nodes, num_nodes).
    """
    undirected_edge_index = to_undirected_edge_index(edge_index, with_self_loops=True)

    # num_nodes assumes that the node indices in edge_index are in the range [0, num_nodes-1],
    # as this is the default logic in the library dataset preprocessing.
    num_nodes = (
        int(undirected_edge_index.max().item()) + 1 if num_nodes is None else num_nodes
    )

    degree_matrix = get_sparse_normalized_degree_matrix(
        edge_index=undirected_edge_index, num_nodes=num_nodes
    )

    adj_matrix = get_sparse_adjacency_matrix(
        edge_index=undirected_edge_index, num_nodes=num_nodes
    )

    # Compute normalized Laplacian matrix: L = D^-1/2 * A * D^-1/2
    normalized_laplacian_matrix = torch.sparse.mm(
        degree_matrix, torch.sparse.mm(adj_matrix, degree_matrix)
    )
    return normalized_laplacian_matrix.coalesce()


def reduce_to_graph_edge_index(
    x: Tensor,
    edge_index: Tensor,
    with_mediators: bool = False,
    remove_selfloops: bool = True,
) -> Tensor:
    r"""
    Construct a graph from a hypergraph with methods proposed in `HyperGCN: A New Method of Training Graph Convolutional Networks on Hypergraphs <https://arxiv.org/pdf/1809.02589.pdf>`_ paper
    Reference implementation: `source <https://deephypergraph.readthedocs.io/en/latest/_modules/dhg/structure/graphs/graph.html#Graph.from_hypergraph_hypergcn>`_.

    Args:
        x: Node feature matrix. Size ``(|V|, C)``.
        edge_index: Hypergraph edge index. Size ``(2, |E|)``.
        with_mediator: Whether to use mediator to transform the hyperedges to edges in the graph. Defaults to ``False``.
        remove_selfloops: Whether to remove self-loops. Defaults to ``True``.

    Returns:
        The graph edge index. Size ``(2, |E'|)``.
    """
    device = x.device

    hypergraph = Hypergraph.from_edge_index(edge_index)
    hypergraph_edges: List[List[int]] = hypergraph.edges
    graph_edges: List[List[int]] = []

    # Random direction (feature_dim, 1) for projecting nodes in each hyperedge
    # Geometrically, we are choosing a random line through the origin in ℝᵈ, where ᵈ = feature_dim
    random_direction = torch.rand((x.shape[1], 1), device=device)

    for edge in hypergraph_edges:
        num_nodes_in_edge = len(edge)
        if num_nodes_in_edge < 2:
            raise ValueError("The number of vertices in an hyperedge must be >= 2.")

        # projections (num_nodes_in_edge,) contains a scalar value for each node in the hyperedge,
        # indicating its projection on the random vector 'random_direction'.
        # Key idea: If two points are very far apart in ℝᵈ, there is a high probability
        # that a random projection will still separate them
        projections = torch.matmul(x[edge], random_direction).squeeze()

        # The indices of the nodes that the farthest apart in the direction of 'random_direction'
        node_max_proj_idx = torch.argmax(projections)
        node_min_proj_idx = torch.argmin(projections)

        if not with_mediators:  # Just connect the two farthest nodes
            graph_edges.append([edge[node_min_proj_idx], edge[node_max_proj_idx]])
            continue

        for node_idx in range(num_nodes_in_edge):
            if node_idx != node_max_proj_idx and node_idx != node_min_proj_idx:
                graph_edges.append([edge[node_min_proj_idx], edge[node_idx]])
                graph_edges.append([edge[node_max_proj_idx], edge[node_idx]])

    graph = Graph(edges=graph_edges)
    if remove_selfloops:
        graph.remove_self_loops()

    return graph.to_edge_index()


def smoothing_with_gcn_laplacian_matrix(
    x: Tensor,
    laplacian_matrix: Tensor,
    drop_rate: float = 0.0,
) -> Tensor:
    r"""
    Return the feature matrix smoothed with GCN Laplacian matrix.
    Reference implementation: `source <https://deephypergraph.readthedocs.io/en/latest/_modules/dhg/structure/graphs/graph.html#Graph.smoothing_with_GCN>`_.

    Args:
        x: Node feature matrix. Size ``(|V|, C)``.
        drop_rate: Randomly dropout the connections in adjacency matrix with probability ``drop_rate``. Default: ``0.0``.

    Returns:
        The smoothed feature matrix. Size ``(|V|, C)``.
    """
    if drop_rate > 0.0:
        laplacian_matrix = sparse_dropout(laplacian_matrix, drop_rate)
    return laplacian_matrix.matmul(x)


def to_undirected_edge_index(
    edge_index: Tensor, with_self_loops: bool = False
) -> Tensor:
    """
    Convert a directed edge index to an undirected edge index by adding reverse edges.

    Args:
        edge_index: Edge index tensor of shape (2, |E|).
        with_self_loops: Whether to add self-loops to each node. Defaults to ``False``.

    Returns:
        The undirected edge index tensor of shape (2, |E'|). If ``with_self_loops`` is ``True``, self-loops are added.
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
    undirected_edge_index = torch.stack([src, dest], dim=0)
    undirected_edge_index = torch.unique(undirected_edge_index, dim=1)

    if with_self_loops:
        # num_nodes assumes that the node indices in edge_index are in the range [0, num_nodes-1],
        # as this is the default logic in the library dataset preprocessing.
        num_nodes = int(undirected_edge_index.max().item()) + 1
        src, dest = undirected_edge_index[0], undirected_edge_index[1]

        # Add self-loops: A_hat = A + I (works as we assume node indices are in [0, num_nodes-1])
        self_loop_indices = torch.arange(num_nodes, device=edge_index.device)
        src = torch.cat([src, self_loop_indices])
        dest = torch.cat([dest, self_loop_indices])
        undirected_edge_index = torch.stack([src, dest], dim=0)

    return undirected_edge_index
