import pytest
import torch

from torch import Tensor
from hyperbench.utils import empty_edgeindex, empty_edgeattr, to_non_empty_edgeattr


def test_empty_edgeindex():
    result = empty_edgeindex()

    assert isinstance(result, Tensor)
    assert result.shape == (2, 0)
    assert result.dtype == torch.float32


def test_empty_edgeattr_zero_edges():
    result = empty_edgeattr(0)

    assert isinstance(result, Tensor)
    assert result.shape == (0, 0)
    assert result.dtype == torch.float32


def test_empty_edgeattr_with_edges():
    result = empty_edgeattr(5)

    assert isinstance(result, Tensor)
    assert result.shape == (5, 0)
    assert result.dtype == torch.float32


def test_empty_edgeattr_single_edge():
    result = empty_edgeattr(1)

    assert isinstance(result, Tensor)
    assert result.shape == (1, 0)


def test_to_non_empty_edgeattr_with_none():
    result = to_non_empty_edgeattr(None)

    assert isinstance(result, Tensor)
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
