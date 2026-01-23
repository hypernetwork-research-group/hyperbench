import torch
import pytest
from unittest.mock import patch, mock_open
from hyperbench.data import Dataset, HIFConverter
from hyperbench.types import HIFHypergraph

from hyperbench.data.dataset import AlgebraDataset


def test_HIFConverter():
    dataset_name = "ALGEBRA"
    file_id = "1-H21_mZTcbbae4U_yM3xzXX19VhbCZ9C"

    hypergraph = HIFConverter.load_from_hif(dataset_name, file_id)

    assert hypergraph is not None
    assert hasattr(hypergraph, "nodes")
    assert hasattr(hypergraph, "edges")
    assert hasattr(hypergraph, "incidences")
    assert hasattr(hypergraph, "metadata")
    assert hasattr(hypergraph, "network_type")

    assert hypergraph.num_nodes == 423
    assert hypergraph.num_edges == 1268


def test_HIFConverter_invalid_dataset():
    dataset_name = "INVALID_DATASET"
    file_id = "invalid_file_id"

    with pytest.raises(ValueError, match="Dataset 'INVALID_DATASET' not found"):
        HIFConverter.load_from_hif(dataset_name, file_id)


def test_HIFConverter_invalid_hif_format():
    dataset_name = "ALGEBRA"
    file_id = "test_file_id"

    invalid_hif_json = '{"network-type": "undirected", "nodes": []}'

    with (
        patch("hyperbench.data.dataset.gdown.download") as mock_download,
        patch("builtins.open", mock_open(read_data=invalid_hif_json)),
        patch("hyperbench.data.dataset.validate_hif_json", return_value=False),
    ):
        with pytest.raises(ValueError, match="Dataset 'ALGEBRA' is not HIF-compliant"):
            HIFConverter.load_from_hif(dataset_name, file_id)


def test_Dataset_available():
    class TestDataset(Dataset):
        GDRIVE_FILE_ID = "abcde"
        DATASET_NAME = "ALGEBRA"

    dataset = TestDataset()
    assert dataset.GDRIVE_FILE_ID == "abcde"
    assert dataset.DATASET_NAME == "ALGEBRA"
    assert dataset.hypergraph is None


def test_Dataset_not_available():
    class TestDataset(Dataset):
        GDRIVE_FILE_ID = "abcde"
        DATASET_NAME = "unreal"

    dataset = TestDataset()
    assert dataset.GDRIVE_FILE_ID == "abcde"
    assert dataset.DATASET_NAME == "unreal"
    assert dataset.hypergraph is None

    with pytest.raises(ValueError, match="Dataset 'unreal' not found"):
        dataset.download()


def test_Dataset_name_none():
    class TestDataset(Dataset):
        GDRIVE_FILE_ID = "abcde"
        DATASET_NAME = None

    dataset = TestDataset()
    assert dataset.GDRIVE_FILE_ID == "abcde"
    assert dataset.DATASET_NAME is None
    assert dataset.hypergraph is None

    with pytest.raises(
        ValueError,
        match=r"Dataset name \(provided: None\) and file ID \(provided: abcde\) must be provided\.",
    ):
        dataset.download()


def test_Dataset_id_none():
    class TestDataset(Dataset):
        GDRIVE_FILE_ID = None
        DATASET_NAME = "abcde"

    dataset = TestDataset()
    assert dataset.GDRIVE_FILE_ID is None
    assert dataset.DATASET_NAME == "abcde"
    assert dataset.hypergraph is None

    with pytest.raises(
        ValueError,
        match=r"Dataset name \(provided: abcde\) and file ID \(provided: None\) must be provided\.",
    ):
        dataset.download()


def test_Dataset_len():
    class TestDataset(Dataset):
        GDRIVE_FILE_ID = "1-H21_mZTcbbae4U_yM3xzXX19VhbCZ9C"
        DATASET_NAME = "ALGEBRA"

    dataset = TestDataset()
    hypergraph = dataset.download()
    dataset.hypergraph = hypergraph

    assert len(dataset) == hypergraph.num_nodes


def test_download_when_hgypergraph_already_loaded():
    class TestDataset(Dataset):
        GDRIVE_FILE_ID = "1-H21_mZTcbbae4U_yM3xzXX19VhbCZ9C"
        DATASET_NAME = "ALGEBRA"

    dataset = TestDataset()
    hypergraph = dataset.download()
    dataset.hypergraph = hypergraph

    with patch("hyperbench.data.dataset.HIFConverter.load_from_hif") as mock_load:
        returned_hypergraph = dataset.download()
        mock_load.assert_not_called()
        assert returned_hypergraph == hypergraph


def test_Dataset_process():
    class TestDataset(Dataset):
        GDRIVE_FILE_ID = "1-H21_mZTcbbae4U_yM3xzXX19VhbCZ9C"
        DATASET_NAME = "ALGEBRA"

    dataset = TestDataset()
    hypergraph = dataset.download()
    dataset.hypergraph = hypergraph

    hdata = dataset.process()
    assert hdata is not None
    assert hasattr(hdata, "x")
    assert hasattr(hdata, "edge_index")
    assert isinstance(hdata.x, torch.Tensor)
    assert isinstance(hdata.edge_index, torch.Tensor)
    assert hdata.x.dim() == 2
    assert hdata.edge_index.dim() == 2


def test_Dataset_hypergrape_none():
    class TestDataset(Dataset):
        GDRIVE_FILE_ID = "1-H21_mZTcbbae4U_yM3xzXX19VhbCZ9C"
        DATASET_NAME = "ALGEBRA"

    dataset = TestDataset()

    with pytest.raises(
        ValueError, match=r"Hypergraph is not loaded\. Call download\(\) first\."
    ):
        dataset.process()


def test_Dataset_process_no_incidences():
    """Test that process handles empty incidences list."""

    class TestDataset(Dataset):
        GDRIVE_FILE_ID = "test_id"
        DATASET_NAME = "ALGEBRA"

    dataset = TestDataset()

    dataset.hypergraph = HIFHypergraph(
        network_type="undirected",
        nodes=[{"node": "0", "attrs": {}}, {"node": "1", "attrs": {}}],
        edges=[{"edge": "0", "attrs": {}}],
        incidences=[],
    )

    with pytest.raises(ValueError, match=r"Hypergraph has no incidences\."):
        dataset.process()


def test_Dataset_process_with_edge_attributes():
    """Test that process correctly handles edges with attributes."""

    class TestDataset(Dataset):
        GDRIVE_FILE_ID = "test_id"
        DATASET_NAME = "ALGEBRA"

    dataset = TestDataset()

    dataset.hypergraph = HIFHypergraph(
        network_type="undirected",
        nodes=[
            {"node": "0", "attrs": {}},
            {"node": "1", "attrs": {}},
            {"node": "2", "attrs": {}},
        ],
        edges=[
            {"edge": "0", "attrs": {"weight": 1.0, "type": "A"}},
            {"edge": "1", "attrs": {"weight": 2.0}},
        ],
        incidences=[
            {"node": "0", "edge": "0"},
            {"node": "1", "edge": "0"},
            {"node": "2", "edge": "1"},
        ],
    )

    hdata = dataset.process()
    assert hdata is not None
    assert hdata.x.shape[0] == 3
    assert hdata.edge_index.shape[0] == 2
    assert hdata.edge_index.shape[1] == 3
    assert hdata.edge_attr is not None
    assert hdata.edge_attr.shape[0] == 2
    assert hdata.edge_attr[0].item() == 2
    assert hdata.edge_attr[1].item() == 1


def test_Dataset_process_without_edge_attributes():
    """Test that process handles edges without attributes."""

    class TestDataset(Dataset):
        GDRIVE_FILE_ID = "test_id"
        DATASET_NAME = "ALGEBRA"

    dataset = TestDataset()
    dataset.hypergraph = HIFHypergraph(
        network_type="undirected",
        nodes=[{"node": "0", "attrs": {}}, {"node": "1", "attrs": {}}],
        edges=[{"edge": "0"}],
        incidences=[{"node": "0", "edge": "0"}, {"node": "1", "edge": "0"}],
    )

    hdata = dataset.process()
    assert hdata is not None
    assert hdata.edge_index.shape[0] == 2
    assert hdata.edge_index.shape[1] == 2


def test_Dataset_process_edge_index_format():
    """Test that edge_index has correct format [node_ids, edge_ids]."""

    class TestDataset(Dataset):
        GDRIVE_FILE_ID = "test_id"
        DATASET_NAME = "ALGEBRA"

    dataset = TestDataset()
    dataset.hypergraph = HIFHypergraph(
        network_type="undirected",
        nodes=[
            {"node": "0", "attrs": {}},
            {"node": "1", "attrs": {}},
            {"node": "2", "attrs": {}},
        ],
        edges=[{"edge": "0", "attrs": {}}, {"edge": "1", "attrs": {}}],
        incidences=[
            {"node": "0", "edge": "0"},
            {"node": "1", "edge": "0"},
            {"node": "2", "edge": "1"},
        ],
    )

    hdata = dataset.process()
    assert hdata.edge_index.shape == (2, 3)
    assert hdata.edge_index[0].tolist() == [0, 1, 2]
    assert hdata.edge_index[1].tolist() == [0, 0, 1]
