import torch
import pytest
from unittest.mock import patch, mock_open
from hyperbench.data import Dataset, HIFConverter
from hyperbench.types import HIFHypergraph

from hyperbench.data.dataset import AlgebraDataset
from hyperbench.tests.mock import *


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
    dataset_name = "EMAIL_ENRON"
    file_id = "test_file_id"

    invalid_hif_json = '{"network-type": "undirected", "nodes": []}'

    with (
        patch("hyperbench.data.dataset.gdown.download") as mock_download,
        patch("builtins.open", mock_open(read_data=invalid_hif_json)),
        patch("hyperbench.data.dataset.validate_hif_json", return_value=False),
    ):
        with pytest.raises(
            ValueError, match="Dataset 'EMAIL_ENRON' is not HIF-compliant"
        ):
            HIFConverter.load_from_hif(dataset_name, file_id)


def test_dataset_available():
    # mock_list = []
    dataset = AlgebraMockDataset()

    assert dataset.GDRIVE_FILE_ID == "1-H21_mZTcbbae4U_yM3xzXX19VhbCZ9C"
    assert dataset.DATASET_NAME == "ALGEBRA"

    hypergraph = dataset.download()
    assert hypergraph is not None
    assert isinstance(hypergraph, HIFHypergraph)


def test_dataset_not_available():
    mock_list = []

    with pytest.raises(ValueError, match=r"Dataset 'FAKE' not found"):
        dataset = FakeMockDataset(mock_list)


def test_AlgebraDataset_available():
    dataset = AlgebraDataset()

    assert dataset.GDRIVE_FILE_ID == "1-H21_mZTcbbae4U_yM3xzXX19VhbCZ9C"
    assert dataset.DATASET_NAME == "ALGEBRA"
    assert dataset.hypergraph is not None
    assert isinstance(dataset.hypergraph, HIFHypergraph)


def test_dataset_name_none():
    mock_list = []

    with pytest.raises(
        ValueError,
        match=r"Dataset name \(provided: None\) and file ID \(provided: fake_id\) must be provided\.",
    ):
        dataset = FakeMockDataset2(mock_list)


def test_dataset_len():
    dataset = AlgebraMockDataset()

    assert dataset.__len__() == dataset.hypergraph.num_nodes


def test_dataset_hypergraph_none():
    with pytest.raises(
        ValueError, match=r"Hypergraph is not loaded\. Call download\(\) first\."
    ):
        FakeMockDataset3()


def test_dataset_process_no_incidences():
    """Test that process handles empty incidences list."""

    dataset = AlgebraMockDataset()

    dataset.hypergraph = HIFHypergraph(
        network_type="undirected",
        nodes=[{"node": "0", "attrs": {}}, {"node": "1", "attrs": {}}],
        edges=[{"edge": "0", "attrs": {}}],
        incidences=[],
    )

    with pytest.raises(ValueError, match=r"Hypergraph has no incidences\."):
        dataset.process()


def test_dataset_process_with_edge_attributes():
    """Test that process correctly handles edges with attributes."""

    dataset = AlgebraMockDataset()

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


def test_dataset_process_without_edge_attributes():
    """Test that process handles edges without attributes."""

    dataset = AlgebraMockDataset()

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


def test_dataset_process_edge_index_format():
    """Test that edge_index has correct format [node_ids, edge_ids]."""

    dataset = AlgebraMockDataset()

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


def test_dataset_process_random_ids():
    dataset = AlgebraMockDataset()

    dataset.hypergraph = HIFHypergraph(
        network_type="undirected",
        nodes=[
            {"node": "abc", "attrs": {}},
            {"node": "ss", "attrs": {}},
            {"node": "fewao", "attrs": {}},
        ],
        edges=[{"edge": "0", "attrs": {}}, {"edge": "1", "attrs": {}}],
        incidences=[
            {"node": "abc", "edge": "0"},
            {"node": "ss", "edge": "0"},
            {"node": "fewao", "edge": "1"},
        ],
    )
    hdata = dataset.process()
    print(hdata.x)
    print(hdata.edge_index)
    assert hdata.edge_index.shape == (2, 3)
