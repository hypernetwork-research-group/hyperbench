import requests
import torch
import pytest
from unittest.mock import patch, mock_open
from hyperbench.data import Dataset, HIFConverter
from hyperbench.types import HIFHypergraph, HData

from hyperbench.data.dataset import AlgebraDataset
from hyperbench.tests.mock import *


# Reusable fixture for hypergraph instances used in multiple tests
@pytest.fixture
def sample_hypergraph():
    return HIFHypergraph(
        network_type="undirected",
        nodes=[{"node": "0"}, {"node": "1"}],
        edges=[{"edge": "0"}],
        incidences=[{"node": "0", "edge": "0"}],
    )


@pytest.fixture
def simple_mock_hypergraph():
    """Simple hypergraph with 2 nodes for basic tests."""
    return HIFHypergraph(
        network_type="undirected",
        nodes=[{"node": "0", "attrs": {}}, {"node": "1", "attrs": {}}],
        edges=[{"edge": "0", "attrs": {}}],
        incidences=[{"node": "0", "edge": "0"}],
    )


@pytest.fixture
def three_node_mock_hypergraph():
    """Hypergraph with 3 nodes for validation tests."""
    return HIFHypergraph(
        network_type="undirected",
        nodes=[
            {"node": "0", "attrs": {}},
            {"node": "1", "attrs": {}},
            {"node": "2", "attrs": {}},
        ],
        edges=[{"edge": "0", "attrs": {}}],
        incidences=[{"node": "0", "edge": "0"}],
    )


@pytest.fixture
def three_node_mock_weighted_hypergraph():
    return HIFHypergraph(
        network_type="undirected",
        nodes=[
            {"node": "0", "attrs": {}},
            {"node": "1", "attrs": {}},
            {"node": "2", "attrs": {}},
        ],
        edges=[
            {"edge": "0", "attrs": {"weight": 1.0}},
            {"edge": "1", "attrs": {"weight": 2.0}},
        ],
        incidences=[
            {"node": "0", "edge": "0"},
            {"node": "1", "edge": "0"},
            {"node": "2", "edge": "1"},
        ],
    )


@pytest.fixture
def four_node_mock_hypergraph():
    """Hypergraph with 4 nodes and 2 edges for sampling tests."""
    return HIFHypergraph(
        network_type="undirected",
        nodes=[
            {"node": "0", "attrs": {}},
            {"node": "1", "attrs": {}},
            {"node": "2", "attrs": {}},
            {"node": "3", "attrs": {}},
        ],
        edges=[{"edge": "0", "attrs": {}}, {"edge": "1", "attrs": {}}],
        incidences=[
            {"node": "0", "edge": "0"},
            {"node": "1", "edge": "0"},
            {"node": "2", "edge": "1"},
            {"node": "3", "edge": "1"},
        ],
    )


@pytest.fixture
def five_node_mock_hypergraph():
    """Hypergraph with 5 nodes for duplicate testing."""
    return HIFHypergraph(
        network_type="undirected",
        nodes=[
            {"node": "0", "attrs": {}},
            {"node": "1", "attrs": {}},
            {"node": "2", "attrs": {}},
            {"node": "3", "attrs": {}},
            {"node": "4", "attrs": {}},
        ],
        edges=[{"edge": "0", "attrs": {}}],
        incidences=[{"node": "0", "edge": "0"}],
    )


@pytest.fixture
def no_edge_attr_mock_hypergraph():
    return HIFHypergraph(
        network_type="undirected",
        nodes=[
            {"node": "0", "attrs": {}},
            {"node": "1", "attrs": {}},
        ],
        edges=[{"edge": "0"}],
        incidences=[
            {"node": "0", "edge": "0"},
            {"node": "1", "edge": "0"},
        ],
    )


@pytest.fixture
def multiple_edges_attr_mock_hypergraph():
    return HIFHypergraph(
        network_type="undirected",
        nodes=[
            {"node": "0", "attrs": {}},
            {"node": "1", "attrs": {}},
            {"node": "2", "attrs": {}},
            {"node": "3", "attrs": {}},
        ],
        edges=[
            {"edge": "0", "attrs": {"weight": 1.0}},
            {"edge": "1", "attrs": {"weight": 2.0}},
            {"edge": "2", "attrs": {"weight": 3.0}},
        ],
        incidences=[
            {"node": "0", "edge": "0"},
            {"node": "1", "edge": "0"},
            {"node": "2", "edge": "1"},
            {"node": "3", "edge": "2"},
        ],
    )


def test_fixture(sample_hypergraph):
    assert sample_hypergraph.network_type == "undirected"
    assert len(sample_hypergraph.nodes) == 2
    assert len(sample_hypergraph.edges) == 1
    assert len(sample_hypergraph.incidences) == 1


def test_HIFConverter():
    """Test loading a known HIF dataset using HIFConverter."""
    dataset_name = "ALGEBRA"

    hypergraph = HIFConverter.load_from_hif(dataset_name)

    assert hypergraph is not None
    assert hasattr(hypergraph, "nodes")
    assert hasattr(hypergraph, "edges")
    assert hasattr(hypergraph, "incidences")
    assert hasattr(hypergraph, "metadata")
    assert hasattr(hypergraph, "network_type")

    assert hypergraph.num_nodes == 423
    assert hypergraph.num_edges == 1268


def test_HIFConverter_invalid_dataset():
    """Test loading an invalid dataset"""
    dataset_name = "INVALID_DATASET"

    with pytest.raises(ValueError, match="Dataset 'INVALID_DATASET' not found"):
        HIFConverter.load_from_hif(dataset_name)


def test_HIFConverter_invalid_hif_format():
    """Test loading an invalid HIF format dataset."""
    dataset_name = "ALGEBRA"

    invalid_hif_json = '{"network-type": "undirected", "nodes": []}'

    with (
        patch("hyperbench.data.dataset.requests.get") as mock_get,
        patch("hyperbench.data.dataset.validate_hif_json", return_value=False),
        patch("builtins.open", mock_open(read_data=invalid_hif_json)),
        patch("hyperbench.data.dataset.zstd.ZstdDecompressor"),
    ):
        mock_response = mock_get.return_value
        mock_response.status_code = 200
        mock_response.content = b"mock_zst_content"

        with pytest.raises(ValueError, match="Dataset 'algebra' is not HIF-compliant"):
            HIFConverter.load_from_hif(dataset_name)


def test_HIFConverter_save_on_disk():
    """Test downloading dataset with save_on_disk=True."""
    dataset_name = "ALGEBRA"

    mock_hypergraph = HIFHypergraph(
        network_type="undirected",
        nodes=[{"node": "0"}, {"node": "1"}],
        edges=[{"edge": "0"}],
        incidences=[{"node": "0", "edge": "0"}],
    )

    mock_hif_json = {
        "network-type": "undirected",
        "nodes": [{"node": "0"}, {"node": "1"}],
        "edges": [{"edge": "0"}],
        "incidences": [{"node": "0", "edge": "0"}],
    }

    with (
        patch("hyperbench.data.dataset.requests.get") as mock_get,
        patch("hyperbench.data.dataset.os.path.exists", return_value=False),
        patch("hyperbench.data.dataset.os.makedirs"),
        patch("builtins.open", mock_open()) as mock_file,
        patch("hyperbench.data.dataset.zstd.ZstdDecompressor") as mock_decomp,
        patch("hyperbench.data.dataset.json.load", return_value=mock_hif_json),
        patch("hyperbench.data.dataset.validate_hif_json", return_value=True),
        patch.object(HIFHypergraph, "from_hif", return_value=mock_hypergraph),
    ):
        # Mock successful download
        mock_response = mock_get.return_value
        mock_response.status_code = 200
        mock_response.content = b"mock_zst_content"

        # Mock decompressor
        mock_stream = mock_decomp.return_value.stream_reader.return_value
        mock_stream.__enter__ = lambda self: mock_stream
        mock_stream.__exit__ = lambda self, *args: None

        hypergraph = HIFConverter.load_from_hif(dataset_name, save_on_disk=True)

        assert hypergraph is not None
        assert hypergraph.network_type == "undirected"
        mock_get.assert_called_once()
        # Verify file was written to disk (not temp file)
        assert mock_file.call_count >= 2  # Once for write, once for read


def test_HIFConverter_temp_file():
    """Test downloading dataset with save_on_disk=False (uses temp file)."""
    dataset_name = "ALGEBRA"

    mock_hypergraph = HIFHypergraph(
        network_type="undirected",
        nodes=[{"node": "0"}, {"node": "1"}],
        edges=[{"edge": "0"}],
        incidences=[{"node": "0", "edge": "0"}],
    )

    mock_hif_json = {
        "network-type": "undirected",
        "nodes": [{"node": "0"}, {"node": "1"}],
        "edges": [{"edge": "0"}],
        "incidences": [{"node": "0", "edge": "0"}],
    }

    with (
        patch("hyperbench.data.dataset.requests.get") as mock_get,
        patch("hyperbench.data.dataset.os.path.exists", return_value=False),
        patch("hyperbench.data.dataset.tempfile.NamedTemporaryFile") as mock_temp,
        patch("builtins.open", mock_open()),
        patch("hyperbench.data.dataset.zstd.ZstdDecompressor") as mock_decomp,
        patch("hyperbench.data.dataset.json.load", return_value=mock_hif_json),
        patch("hyperbench.data.dataset.validate_hif_json", return_value=True),
        patch.object(HIFHypergraph, "from_hif", return_value=mock_hypergraph),
    ):
        # Mock successful download
        mock_response = mock_get.return_value
        mock_response.status_code = 200
        mock_response.content = b"mock_zst_content"

        # Mock temp file
        mock_temp_file = mock_temp.return_value.__enter__.return_value
        mock_temp_file.name = "/tmp/fake_temp.json.zst"

        # Mock decompressor
        mock_stream = mock_decomp.return_value.stream_reader.return_value
        mock_stream.__enter__ = lambda self: mock_stream
        mock_stream.__exit__ = lambda self, *args: None

        hypergraph = HIFConverter.load_from_hif(dataset_name, save_on_disk=False)

        assert hypergraph is not None
        assert hypergraph.network_type == "undirected"
        mock_get.assert_called_once()
        # Verify temp file was used
        assert mock_temp.call_count >= 1


def test_HIFConverter_download_failure():
    """Test handling of failed download from GitHub."""
    dataset_name = "ALGEBRA"

    with (
        patch("hyperbench.data.dataset.requests.get") as mock_get,
        patch("hyperbench.data.dataset.os.path.exists", return_value=False),
    ):
        # Mock failed download
        mock_response = mock_get.return_value
        mock_response.status_code = 404

        with pytest.raises(
            ValueError,
            match=r"Failed to download dataset 'algebra' from GitHub\. Status code: 404",
        ):
            HIFConverter.load_from_hif(dataset_name)

        mock_get.assert_called_once_with(
            "https://github.com/hypernetwork-research-group/datasets/blob/main/algebra.json.zst?raw=true"
        )


def test_HIFConverter_network_error():
    """Test handling of network errors during download."""
    dataset_name = "ALGEBRA"

    with (
        patch("hyperbench.data.dataset.requests.get") as mock_get,
        patch("hyperbench.data.dataset.os.path.exists", return_value=False),
    ):
        # Mock network error
        mock_get.side_effect = requests.RequestException("Network error")

        with pytest.raises(requests.RequestException, match="Network error"):
            HIFConverter.load_from_hif(dataset_name)


def test_dataset_not_available():
    """Test loading an unavailable dataset."""

    class FakeMockDataset(Dataset):
        DATASET_NAME = "FAKE"

    with pytest.raises(ValueError, match=r"Dataset 'FAKE' not found"):
        FakeMockDataset()


def test_AlgebraDataset_available():
    """Test loading the ALGEBRA dataset."""

    mock_hypergraph = HIFHypergraph(
        network_type="undirected",
        nodes=[{"node": str(i)} for i in range(423)],
        edges=[{"edge": str(i)} for i in range(1268)],
        incidences=[{"node": "0", "edge": "0"}],
    )

    with patch.object(HIFConverter, "load_from_hif", return_value=mock_hypergraph):
        dataset = AlgebraDataset()

        assert dataset.DATASET_NAME == "ALGEBRA"
        assert dataset.hypergraph is not None
        assert dataset.__len__() == dataset.hypergraph.num_nodes


def test_double_download():
    """Test that downloading the same dataset twice uses local value."""

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
    """Test that ValueError is raised if DATASET_NAME is None."""

    class FakeMockDataset(Dataset):
        DATASET_NAME = None

    with pytest.raises(
        ValueError,
        match=r"Dataset name \(provided: None\) must be provided\.",
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
            {"edge": "0", "attrs": {"weight": 1.0, "type": 2.0}},
            {"edge": "1", "attrs": {"weight": 3.0, "type": 0.1}},
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
    # Two edges with two attributes each: shape [2, 2]
    assert dataset.hdata.edge_attr.shape == (2, 2)
    # Attributes maintain dictionary insertion order (no sorting)

    assert torch.allclose(
        dataset.hdata.edge_attr[0], torch.tensor([1.0, 2.0])
    )  # weight, type
    assert torch.allclose(
        dataset.hdata.edge_attr[1], torch.tensor([3.0, 0.1])
    )  # weight, type


def test_dataset_process_without_edge_attributes(no_edge_attr_mock_hypergraph):
    """Test that process handles edges without attributes."""

    with patch.object(
        HIFConverter, "load_from_hif", return_value=no_edge_attr_mock_hypergraph
    ):
        dataset = AlgebraDataset()

    assert dataset.hdata is not None
    assert dataset.hdata.edge_index.shape[0] == 2
    assert dataset.hdata.edge_index.shape[1] == 2
    assert dataset.hdata.edge_attr is None


def test_dataset_process_edge_index_format(four_node_mock_hypergraph):
    """Test that edge_index has correct format [node_ids, edge_ids]."""

    with patch.object(
        HIFConverter, "load_from_hif", return_value=four_node_mock_hypergraph
    ):
        dataset = AlgebraDataset()

    assert dataset.hdata.edge_index.shape == (2, 4)
    assert torch.allclose(dataset.hdata.edge_index[0], torch.tensor([0, 1, 2, 3]))
    assert torch.allclose(dataset.hdata.edge_index[1], torch.tensor([0, 0, 1, 1]))


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
    assert torch.allclose(dataset.hdata.edge_index[0], torch.tensor([0, 1, 2]))
    assert torch.allclose(dataset.hdata.edge_index[1], torch.tensor([0, 0, 1]))
    assert dataset.hdata.edge_attr is not None
    assert dataset.hdata.edge_attr.shape == (2, 0)  # 2 edges, 0 attributes each


def test_getitem_index_list_empty(simple_mock_hypergraph):
    """Test __getitem__ with empty index list raises ValueError."""
    with patch.object(
        HIFConverter, "load_from_hif", return_value=simple_mock_hypergraph
    ):
        dataset = AlgebraDataset()

    with pytest.raises(ValueError, match="Index list cannot be empty."):
        dataset[[]]


def test_getitem_index_list_too_large(five_node_mock_hypergraph):
    """Test __getitem__ with index list larger than number of nodes raises ValueError."""
    with patch.object(
        HIFConverter, "load_from_hif", return_value=five_node_mock_hypergraph
    ):
        dataset = AlgebraDataset()

    with pytest.raises(
        ValueError,
        match="Index list length cannot exceed number of nodes in the hypergraph.",
    ):
        dataset[[0, 1, 2, 3, 4, 5]]


def test_getitem_index_out_of_bounds(four_node_mock_hypergraph):
    """Test __getitem__ with out-of-bounds index raises IndexError."""
    with patch.object(
        HIFConverter, "load_from_hif", return_value=four_node_mock_hypergraph
    ):
        dataset = AlgebraDataset()

    with pytest.raises(IndexError, match="Node ID 4 is out of bounds."):
        dataset[4]


def test_getitem_single_index(sample_hypergraph):
    """Test __getitem__ with a single index."""

    with patch.object(HIFConverter, "load_from_hif", return_value=sample_hypergraph):
        dataset = AlgebraDataset()

    node_data = dataset[1]
    assert node_data.x.shape[0] == 1
    assert node_data.edge_index.shape == (2, 0)


def test_getitem_list_index(four_node_mock_hypergraph):
    """Test __getitem__ with a list of indices."""

    with patch.object(
        HIFConverter, "load_from_hif", return_value=four_node_mock_hypergraph
    ):
        dataset = AlgebraDataset()

    node_data_list = dataset[[0, 2, 3]]
    assert node_data_list.x.shape[0] == 3
    assert node_data_list.edge_index.shape == (2, 3)


def test_getitem_with_edge_attr(three_node_mock_weighted_hypergraph):
    """Test __getitem__ returns correct edge_attr when present."""

    with patch.object(
        HIFConverter, "load_from_hif", return_value=three_node_mock_weighted_hypergraph
    ):
        dataset = AlgebraDataset()

    node_data = dataset[0]
    assert node_data.x.shape[0] == 1
    assert node_data.edge_index.shape == (2, 1)
    assert node_data.edge_attr is not None
    assert node_data.edge_attr.shape[0] == 1
    assert node_data.edge_attr[0].item() == 1


def test_getitem_without_edge_attr(no_edge_attr_mock_hypergraph):
    """Test __getitem__ returns None for edge_attr when not present."""

    with patch.object(
        HIFConverter, "load_from_hif", return_value=no_edge_attr_mock_hypergraph
    ):
        dataset = AlgebraDataset()

    node_data = dataset[0]
    assert node_data.edge_attr is None


def test_getitem_with_multiple_edges_attr(multiple_edges_attr_mock_hypergraph):
    """Test __getitem__ correctly filters edge_attr for sampled edges."""

    with patch.object(
        HIFConverter, "load_from_hif", return_value=multiple_edges_attr_mock_hypergraph
    ):
        dataset = AlgebraDataset()

    node_data = dataset[[0, 2]]
    assert node_data.edge_attr is not None
    assert node_data.edge_attr.shape[0] == 2
    assert node_data.num_edges == 2
    assert torch.allclose(node_data.edge_attr, torch.tensor([[1.0], [2.0]]))


def test_getitem_edge_attr_empty_sampled_edges():
    """Test edge_attr is None when sampled edges list is empty but condition is handled."""
    mock_hypergraph = HIFHypergraph(
        network_type="undirected",
        nodes=[
            {"node": "0", "attrs": {}},
            {"node": "1", "attrs": {}},
            {"node": "2", "attrs": {}},
            {"node": "3", "attrs": {}},
        ],
        edges=[
            {"edge": "0", "attrs": {"weight": 1.0}},
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

        node_data = dataset[3]
        assert node_data.num_nodes == 1
        assert node_data.edge_index.shape[1] == 0
        assert node_data.edge_attr is None


def test_getitem_edge_attr_no_uniform_edges():
    """Test edge attributes are padded with 0.0 when edges have inconsistent attributes."""
    mock_hypergraph = HIFHypergraph(
        network_type="undirected",
        nodes=[
            {"node": "0", "attrs": {}},
            {"node": "1", "attrs": {}},
            {"node": "2", "attrs": {}},
            {"node": "3", "attrs": {}},
        ],
        edges=[
            {"edge": "0", "attrs": {"weight": 1.0, "abc": 5.0}},
            {"edge": "1", "attrs": {"weight": 2.0}},  # Missing 'abc'
            {"edge": "2", "attrs": {"abc": 3.0}},  # Missing 'weight'
        ],
        incidences=[
            {"node": "0", "edge": "0"},
            {"node": "1", "edge": "0"},
            {"node": "2", "edge": "1"},
            {"node": "3", "edge": "2"},
        ],
    )

    with patch.object(HIFConverter, "load_from_hif", return_value=mock_hypergraph):
        dataset = AlgebraDataset()

    assert dataset.hdata.edge_attr is not None
    assert dataset.hdata.edge_attr.shape == (
        3,
        2,
    )  # 3 edges, 2 features each (weight, abc in insertion order)
    assert torch.allclose(
        dataset.hdata.edge_attr[0], torch.tensor([1.0, 5.0])
    )  # weight=1.0, abc=5.0
    assert torch.allclose(
        dataset.hdata.edge_attr[1], torch.tensor([2.0, 0.0])
    )  # weight=2.0, abc=0.0
    assert torch.allclose(
        dataset.hdata.edge_attr[2], torch.tensor([0.0, 3.0])
    )  # weight=0.0, abc=3.0


def test_transform_attrs_empty_attrs():
    """Test transform_attrs with empty or no numeric attributes."""
    mock_hypergraph = HIFHypergraph(
        network_type="undirected",
        nodes=[{"node": "0", "attrs": {}}],
        edges=[{"edge": "0", "attrs": {}}],
        incidences=[{"node": "0", "edge": "0"}],
    )

    with patch.object(HIFConverter, "load_from_hif", return_value=mock_hypergraph):

        class TestDataset(Dataset):
            DATASET_NAME = "TEST"

        dataset = TestDataset()

        result = dataset.transform_attrs({})
        assert len(result) == 0

        attrs = {"name": "node1", "active": True}
        result = dataset.transform_attrs(attrs)
        assert len(result) == 0


def test_process_with_inconsistent_node_attributes():
    """Test process() pads missing node attributes with 0.0 (insertion order)."""
    mock_hypergraph = HIFHypergraph(
        network_type="undirected",
        nodes=[
            {"node": "0", "attrs": {"weight": 1.0}},  # Missing 'score'
            {"node": "1", "attrs": {"weight": 2.0, "score": 0.8}},
            {"node": "2", "attrs": {"score": 0.5}},  # Missing 'weight'
        ],
        edges=[{"edge": "0", "attrs": {}}],
        incidences=[
            {"node": "0", "edge": "0"},
            {"node": "1", "edge": "0"},
            {"node": "2", "edge": "0"},
        ],
    )

    with patch.object(HIFConverter, "load_from_hif", return_value=mock_hypergraph):

        class TestDataset(Dataset):
            DATASET_NAME = "TEST"

        dataset = TestDataset()

        assert dataset.hdata.x.shape == (
            3,
            2,
        )  # 3 nodes, 2 features each (weight, score in insertion order)
        assert torch.allclose(
            dataset.hdata.x[0], torch.tensor([1.0, 0.0])
        )  # weight=1.0, score=0.0
        assert torch.allclose(
            dataset.hdata.x[1], torch.tensor([2.0, 0.8])
        )  # weight=2.0, score=0.8
        assert torch.allclose(
            dataset.hdata.x[2], torch.tensor([0.0, 0.5])
        )  # weight=0.0, score=0.5


def test_process_with_no_node_attributes_fallback():
    """Test process() falls back to torch zeros when no numeric attributes."""
    mock_hypergraph = HIFHypergraph(
        network_type="undirected",
        nodes=[
            {"node": "0", "attrs": {"name": "node0"}},
            {"node": "1", "attrs": {}},
        ],
        edges=[{"edge": "0", "attrs": {}}],
        incidences=[{"node": "0", "edge": "0"}, {"node": "1", "edge": "0"}],
    )

    with patch.object(HIFConverter, "load_from_hif", return_value=mock_hypergraph):

        class TestDataset(Dataset):
            DATASET_NAME = "TEST"

        dataset = TestDataset()

        assert dataset.hdata.x.shape == (2, 1)
        assert torch.allclose(dataset.hdata.x, torch.tensor([[0.0], [0.0]]))


def test_process_with_single_node_attribute():
    """Test process() with single numeric attribute per node."""
    mock_hypergraph = HIFHypergraph(
        network_type="undirected",
        nodes=[
            {"node": "0", "attrs": {"weight": 1.5}},
            {"node": "1", "attrs": {"weight": 2.5}},
            {"node": "2", "attrs": {"weight": 3.5}},
        ],
        edges=[{"edge": "0", "attrs": {}}],
        incidences=[
            {"node": "0", "edge": "0"},
            {"node": "1", "edge": "0"},
            {"node": "2", "edge": "0"},
        ],
    )

    with patch.object(HIFConverter, "load_from_hif", return_value=mock_hypergraph):

        class TestDataset(Dataset):
            DATASET_NAME = "TEST"

        dataset = TestDataset()

        # Single attribute should remain 2D: [num_nodes, 1]
        assert dataset.hdata.x.shape == (3, 1)
        assert torch.allclose(dataset.hdata.x, torch.tensor([[1.5], [2.5], [3.5]]))


def test_getitem_preserves_node_attributes():
    """Test that __getitem__ correctly samples node attributes."""
    mock_hypergraph = HIFHypergraph(
        network_type="undirected",
        nodes=[
            {"node": "0", "attrs": {"weight": 1.0}},
            {"node": "1", "attrs": {"weight": 2.0}},
            {"node": "2", "attrs": {"weight": 3.0}},
        ],
        edges=[
            {"edge": "0", "attrs": {}},
            {"edge": "1", "attrs": {}},
        ],
        incidences=[
            {"node": "0", "edge": "0"},
            {"node": "1", "edge": "0"},
            {"node": "2", "edge": "1"},
        ],
    )

    with patch.object(HIFConverter, "load_from_hif", return_value=mock_hypergraph):

        class TestDataset(Dataset):
            DATASET_NAME = "TEST"

        dataset = TestDataset()

        node_data = dataset[0]
        assert torch.allclose(node_data.x, torch.tensor([[1.0]]))

        node_data = dataset[[0, 2]]
        assert node_data.x.shape == (2, 1)
        assert torch.allclose(node_data.x[0], torch.tensor([1.0]))
        assert torch.allclose(node_data.x[1], torch.tensor([3.0]))

        node_data = dataset[2]
        assert node_data.x.shape == (1, 1)
        assert torch.allclose(node_data.x, torch.tensor([[3.0]]))


def test_transform_attrs_with_attr_keys_padding():
    """Test transform_attrs pads missing attributes with 0.0 when attr_keys provided."""
    mock_hypergraph = HIFHypergraph(
        network_type="undirected",
        nodes=[{"node": "0", "attrs": {}}],
        edges=[{"edge": "0", "attrs": {}}],
        incidences=[{"node": "0", "edge": "0"}],
    )

    with patch.object(HIFConverter, "load_from_hif", return_value=mock_hypergraph):

        class TestDataset(Dataset):
            DATASET_NAME = "TEST"

        dataset = TestDataset()

        # Test with attr_keys - should pad missing attributes with 0.0
        attrs = {"weight": 1.5}
        result = dataset.transform_attrs(attrs, attr_keys=["score", "weight", "age"])
        assert torch.allclose(
            result, torch.tensor([0.0, 1.5, 0.0])
        )  # score=0.0, weight=1.5, age=0.0

        # Test with all attributes present
        attrs = {"weight": 1.5, "score": 0.8, "age": 25.0}
        result = dataset.transform_attrs(attrs, attr_keys=["age", "score", "weight"])
        assert torch.allclose(
            result, torch.tensor([25.0, 0.8, 1.5])
        )  # age=25.0, score=0.8, weight=1.5

        # Test without attr_keys - maintains insertion order
        attrs = {"weight": 1.5, "score": 0.8}
        result = dataset.transform_attrs(attrs)
        assert torch.allclose(
            result, torch.tensor([1.5, 0.8])
        )  # weight, score (insertion order)
