import pytest
import torch

from torch import Tensor
from hyperbench.utils import (
    empty_nodefeatures,
    empty_edgeindex,
    empty_edgeattr,
    to_non_empty_edgeattr,
    empty_hdata,
    empty_hifhypergraph,
)


def test_empty_edgeindex():
    result = empty_edgeindex()

    assert result.shape == (2, 0)
    assert result.dtype == torch.float32


def test_empty_edgeattr_zero_edges():
    result = empty_edgeattr(num_edges=0)

    assert result.shape == (0, 0)


def test_empty_edgeattr_single_edge():
    result = empty_edgeattr(num_edges=1)

    assert result.shape == (1, 0)


def test_empty_edgeattr_with_edges():
    result = empty_edgeattr(num_edges=5)

    assert result.shape == (5, 0)


def test_to_non_empty_edgeattr_with_none():
    result = to_non_empty_edgeattr(edge_attr=None)

    assert result.shape == (0, 0)


def test_to_non_empty_edgeattr_with_tensor():
    edge_attr = torch.tensor([[0.5], [0.7], [0.9]])
    result = to_non_empty_edgeattr(edge_attr)

    assert torch.equal(result, edge_attr)
    assert result.shape == (3, 1)


def test_to_non_empty_edgeattr_with_empty_tensor():
    edge_attr = torch.empty((0, 3))
    result = to_non_empty_edgeattr(edge_attr)

    assert torch.equal(result, edge_attr)
    assert result.shape == (0, 3)


def test_to_non_empty_edgeattr_with_multi_dimensional():
    edge_attr = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    result = to_non_empty_edgeattr(edge_attr)

    assert torch.equal(result, edge_attr)
    assert result.shape == (2, 3)


def test_empty_nodefeatures():
    result = empty_nodefeatures()

    assert result.shape == (0, 0)


def test_empty_hdata():
    result = empty_hdata()

    assert result.x is not None
    assert isinstance(result.x, Tensor)
    assert result.x.shape == (0, 0)

    assert result.edge_index is not None
    assert isinstance(result.edge_index, Tensor)
    assert result.edge_index.shape == (2, 0)

    assert result.edge_attr is None
    assert result.num_nodes == 0
    assert result.num_edges == 0


def test_empty_hifhypergraph():
    result = empty_hifhypergraph()

    assert result.network_type == "undirected"
    assert result.nodes == []
    assert result.edges == []
    assert result.incidences == []
    assert result.metadata == {}
