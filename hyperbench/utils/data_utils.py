import torch

from torch import Tensor


def empty_nodefeatures() -> Tensor:
    return torch.empty((0, 0))


def empty_edgeindex() -> Tensor:
    return torch.empty((2, 0))


def empty_edgeattr(num_edges: int) -> Tensor:
    return torch.empty((num_edges, 0))


def to_non_empty_edgeattr(edge_attr: Tensor | None) -> Tensor:
    num_edges = edge_attr.size(0) if edge_attr is not None else 0
    return empty_edgeattr(num_edges) if edge_attr is None else edge_attr


def to_0based_ids(original_ids: Tensor, ids_to_keep: Tensor, n: int) -> Tensor:
    """
    Map original IDs to 0-based ids.

    Example:
        original_ids: [1, 3, 3, 7]
        ids_to_keep: [3, 7]
        n = 8                            # total number of elements (nodes or edges) in the original hypergraph
        Returned 0-based IDs: [0, 0, 1]  # the size is sum of occurrences of ids_to_keep in original_ids

    Args:
        original_ids: Tensor of original IDs.
        ids_to_keep: List of selected original IDs to be mapped to 0-based.
        n: Total number of original IDs.

    Returns:
        Tensor of 0-based ids.
    """
    device = original_ids.device

    id_to_0based_id = torch.zeros(n, dtype=torch.long, device=device)
    n_ids_to_keep = len(ids_to_keep)
    id_to_0based_id[ids_to_keep] = torch.arange(n_ids_to_keep, device=device)
    return id_to_0based_id[original_ids]
