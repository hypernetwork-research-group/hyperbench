import torch
import pytest
from unittest.mock import patch, mock_open
from hyperbench.data import Dataset, HIFConverter
from hyperbench.types import HIFHypergraph

from hyperbench.data.dataset import AlgebraDataset
from hyperbench.tests.mock import *


@pytest.fixture
def sample_hypergraph():
    return HIFHypergraph(
        network_type="undirected",
        nodes=[{"node": "0"}, {"node": "1"}],
        edges=[{"edge": "0"}],
        incidences=[{"node": "0", "edge": "0"}],
    )


def test_fixture(sample_hypergraph):
    assert sample_hypergraph.network_type == "undirected"
    assert len(sample_hypergraph.nodes) == 2
    assert len(sample_hypergraph.edges) == 1
    assert len(sample_hypergraph.incidences) == 1


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


def test_dataset_not_available():
    class FakeMockDataset(Dataset):
        GDRIVE_FILE_ID = "fake_id"
        DATASET_NAME = "FAKE"

    with pytest.raises(ValueError, match=r"Dataset 'FAKE' not found"):
        FakeMockDataset()


def test_AlgebraDataset_available():
    mock_hypergraph = HIFHypergraph(
        network_type="undirected",
        nodes=[{"node": str(i)} for i in range(423)],
        edges=[{"edge": str(i)} for i in range(1268)],
        incidences=[{"node": "0", "edge": "0"}],
    )

    with patch.object(HIFConverter, "load_from_hif", return_value=mock_hypergraph):
        dataset = AlgebraDataset()

        assert dataset.GDRIVE_FILE_ID == "1-H21_mZTcbbae4U_yM3xzXX19VhbCZ9C"
        assert dataset.DATASET_NAME == "ALGEBRA"
        assert dataset.hypergraph is not None
        assert isinstance(dataset.hypergraph, HIFHypergraph)
        assert dataset.__len__() == dataset.hypergraph.num_nodes


def test_double_download():
    dataset = AlgebraDataset()

    with patch.object(
        HIFConverter,
        "load_from_hif",
        wraps=HIFConverter.load_from_hif,
    ) as mock_load:
        hg1 = dataset.download()
        hg2 = dataset.download()

        assert hg1 is hg2


def test_dataset_name_none():
    class FakeMockDataset(Dataset):
        GDRIVE_FILE_ID = "fake_id"
        DATASET_NAME = None

    with pytest.raises(
        ValueError,
        match=r"Dataset name \(provided: None\) and file ID \(provided: fake_id\) must be provided\.",
    ):
        FakeMockDataset()


def test_dataset_process_no_incidences():
    """Test that process handles empty incidences list."""

    mock_hypergraph = HIFHypergraph(
        network_type="undirected",
        nodes=[{"node": "0", "attrs": {}}, {"node": "1", "attrs": {}}],
        edges=[{"edge": "0", "attrs": {}}],
        incidences=[],
    )

    with pytest.raises(ValueError, match=r"Hypergraph has no incidences\."):
        with patch.object(HIFConverter, "load_from_hif", return_value=mock_hypergraph):
            AlgebraDataset()


def test_dataset_process_with_edge_attributes():
    """Test that process correctly handles edges with attributes."""

    mock_hypergraph = HIFHypergraph(
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

    with patch.object(HIFConverter, "load_from_hif", return_value=mock_hypergraph):
        dataset = AlgebraDataset()

    assert dataset.hdata is not None
    assert dataset.hdata.x.shape[0] == 3
    assert dataset.hdata.edge_index.shape[0] == 2
    assert dataset.hdata.edge_index.shape[1] == 3
    assert dataset.hdata.edge_attr is not None
    assert dataset.hdata.edge_attr.shape[0] == 2
    assert dataset.hdata.edge_attr[0].item() == 2
    assert dataset.hdata.edge_attr[1].item() == 1


def test_dataset_process_without_edge_attributes():
    """Test that process handles edges without attributes."""

    mock_hypergraph = HIFHypergraph(
        network_type="undirected",
        nodes=[{"node": "0", "attrs": {}}, {"node": "1", "attrs": {}}],
        edges=[{"edge": "0"}],
        incidences=[{"node": "0", "edge": "0"}, {"node": "1", "edge": "0"}],
    )

    with patch.object(HIFConverter, "load_from_hif", return_value=mock_hypergraph):
        dataset = AlgebraDataset()

    assert dataset.hdata is not None
    assert dataset.hdata.edge_index.shape[0] == 2
    assert dataset.hdata.edge_index.shape[1] == 2


def test_dataset_process_edge_index_format():
    """Test that edge_index has correct format [node_ids, edge_ids]."""

    mock_hypergraph = HIFHypergraph(
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

    with patch.object(HIFConverter, "load_from_hif", return_value=mock_hypergraph):
        dataset = AlgebraDataset()

    assert dataset.hdata.edge_index.shape == (2, 3)
    assert dataset.hdata.edge_index[0].tolist() == [0, 1, 2]
    assert dataset.hdata.edge_index[1].tolist() == [0, 0, 1]


def test_dataset_process_random_ids():
    mock_hypergraph = HIFHypergraph(
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

    with patch.object(HIFConverter, "load_from_hif", return_value=mock_hypergraph):
        dataset = AlgebraDataset()

    assert dataset.hdata.edge_index.shape == (2, 3)
