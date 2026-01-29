import torch

from torch import Tensor
from hyperbench.types import HData, HIFHypergraph


def empty_nodefeatures() -> Tensor:
    return torch.empty((0, 0))


def empty_edgeindex() -> Tensor:
    return torch.empty((2, 0))


def empty_edgeattr(num_edges: int) -> Tensor:
    return torch.empty((num_edges, 0))


def empty_hdata() -> HData:
    return HData(
        x=empty_nodefeatures(),
        edge_index=empty_edgeindex(),
        edge_attr=None,
        num_nodes=0,
        num_edges=0,
    )


def empty_hifhypergraph() -> HIFHypergraph:
    return HIFHypergraph(
        network_type="undirected", nodes=[], edges=[], incidences=[], metadata=None
    )


def to_non_empty_edgeattr(edge_attr: Tensor | None) -> Tensor:
    num_edges = edge_attr.size(0) if edge_attr is not None else 0
    return empty_edgeattr(num_edges) if edge_attr is None else edge_attr
