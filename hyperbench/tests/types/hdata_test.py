import pytest
import torch

from hyperbench.types import HData


@pytest.fixture
def mock_hdata():
    x = torch.randn(5, 4)  # 5 nodes with 4 features each
    hyperedge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 0],  # node IDs
            [0, 0, 1, 1, 2, 2],
        ]
    )  # hyperedge IDs
    hyperedge_attr = torch.randn(3, 2)  # 3 hyperedges with 2 features each

    return HData(x=x, edge_index=hyperedge_index, edge_attr=hyperedge_attr)


def test_hdata_to_cpu(mock_hdata):
    returned = mock_hdata.to("cpu")

    assert returned is mock_hdata
    assert mock_hdata.x.device.type == "cpu"
    assert mock_hdata.edge_index.device.type == "cpu"
    assert mock_hdata.edge_attr is not None
    assert mock_hdata.edge_attr.device.type == "cpu"


def test_hdata_to_cpu_handles_none_edge_attr(mock_hdata):
    mock_hdata.edge_attr = None
    returned = mock_hdata.to("cpu")

    assert returned is mock_hdata
    assert mock_hdata.x.device.type == "cpu"
    assert mock_hdata.edge_index.device.type == "cpu"
    assert mock_hdata.edge_attr is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_hdata_to_cuda(mock_hdata):
    returned = mock_hdata.to("cuda")

    assert returned is mock_hdata
    assert mock_hdata.x.device.type == "cuda"
    assert mock_hdata.edge_index.device.type == "cuda"
    assert mock_hdata.edge_attr is not None
    assert mock_hdata.edge_attr.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_hdata_to_cuda_handles_none_edge_attr(mock_hdata):
    mock_hdata.edge_attr = None
    returned = mock_hdata.to("cuda")

    assert returned is mock_hdata
    assert mock_hdata.x.device.type == "cuda"
    assert mock_hdata.edge_index.device.type == "cuda"
    assert mock_hdata.edge_attr is None


@pytest.mark.skipif(not torch.mps.is_available(), reason="MPS not available")
def test_hdata_to_mps(mock_hdata):
    returned = mock_hdata.to("mps")

    assert returned is mock_hdata
    assert mock_hdata.x.device.type == "mps"
    assert mock_hdata.edge_index.device.type == "mps"
    assert mock_hdata.edge_attr is not None
    assert mock_hdata.edge_attr.device.type == "mps"


@pytest.mark.skipif(not torch.mps.is_available(), reason="MPS not available")
def test_hdata_to_mps_handles_none_edge_attr(mock_hdata):
    mock_hdata.edge_attr = None
    returned = mock_hdata.to("mps")

    assert returned is mock_hdata
    assert mock_hdata.x.device.type == "mps"
    assert mock_hdata.edge_index.device.type == "mps"
    assert mock_hdata.edge_attr is None
