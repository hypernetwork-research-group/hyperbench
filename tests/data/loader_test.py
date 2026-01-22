import pytest
import torch

from hyperbench.data import DataLoader
from hyperbench.types import HData
from hyperbench import utils
from hyperbench.data import Dataset
from typing import Any, List


class MockDataset(Dataset):
    """Mock dataset for testing DataLoader."""

    def __init__(self, data_list: List[Any]):
        super().__init__()
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]


def test_initialization():
    data = HData(x=torch.randn(3, 4), edge_index=torch.tensor([[0, 1, 2], [0, 0, 1]]))
    dataset = MockDataset([data])
    loader = DataLoader(dataset, batch_size=1)

    assert loader.batch_size == 1
    assert loader.dataset == dataset


def test_initialization_with_custom_params():
    data = HData(x=torch.randn(3, 4), edge_index=torch.tensor([[0, 1, 2], [0, 0, 1]]))
    dataset = MockDataset([data])
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    assert loader.batch_size == 4
    assert loader.dataset == dataset

    # num_workers is used to test that kwargs are passed correctly
    assert loader.num_workers == 0


def test_collate_single_sample():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    edge_index = torch.tensor([[0, 1, 1, 2], [0, 0, 1, 1]])
    edge_attr = torch.tensor([[0.5], [0.7]])

    data = HData(x=x, edge_index=edge_index, edge_attr=edge_attr)
    dataset = MockDataset([data])
    loader = DataLoader(dataset, batch_size=1)

    batched = loader.collate([data])

    assert torch.equal(batched.x, x)
    assert torch.equal(batched.edge_index, edge_index)
    assert torch.equal(utils.to_non_empty_edgeattr(batched.edge_attr), edge_attr)
    assert batched.num_nodes == 3
    assert batched.num_edges == 2


def test_collate_two_samples_no_edge_attr():
    # Sample 0: 3 nodes, 2 hyperedges
    x0 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    edge_index0 = torch.tensor([[0, 1, 1, 2], [0, 0, 1, 1]])
    data0 = HData(x=x0, edge_index=edge_index0)

    # Sample 1: 2 nodes, 1 hyperedge
    x1 = torch.tensor([[7.0, 8.0], [9.0, 10.0]])
    edge_index1 = torch.tensor([[0, 1], [0, 0]])
    data1 = HData(x=x1, edge_index=edge_index1)

    dataset = MockDataset([data0, data1])
    loader = DataLoader(dataset, batch_size=2)

    batched = loader.collate([data0, data1])

    # Check node features are concatenated
    expected_x = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],  # Sample 0
            [7.0, 8.0],
            [9.0, 10.0],  # Sample 1
        ]
    )
    assert torch.equal(batched.x, expected_x)
    assert batched.num_nodes == 5

    # Check edge_index nodes from Sample 1 are offset by 3 and hyperedges are offset by 2
    # Sample 0: nodes [0,1,2], edges [0,1]
    # Sample 1: nodes [3,4], edges [2] (offset by 3 nodes, 2 edges)
    expected_edge_index = torch.tensor([[0, 1, 1, 2, 3, 4], [0, 0, 1, 1, 2, 2]])
    assert torch.equal(batched.edge_index, expected_edge_index)
    assert batched.num_edges == 3

    assert batched.edge_attr is None


def test_collate_two_samples_with_edge_attr():
    # Sample 0: 3 nodes, 2 hyperedges
    x0 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    edge_index0 = torch.tensor([[0, 1, 1, 2], [0, 0, 1, 1]])
    edge_attr0 = torch.tensor([[0.5], [0.7]])
    data0 = HData(x=x0, edge_index=edge_index0, edge_attr=edge_attr0)

    # Sample 1: 2 nodes, 1 hyperedge
    x1 = torch.tensor([[7.0, 8.0], [9.0, 10.0]])
    edge_index1 = torch.tensor([[0, 1], [0, 0]])
    edge_attr1 = torch.tensor([[0.9]])
    data1 = HData(x=x1, edge_index=edge_index1, edge_attr=edge_attr1)

    dataset = MockDataset([data0, data1])
    loader = DataLoader(dataset, batch_size=2)

    batched = loader.collate([data0, data1])

    expected_edge_attr = torch.tensor([[0.5], [0.7], [0.9]])
    assert torch.equal(
        utils.to_non_empty_edgeattr(batched.edge_attr), expected_edge_attr
    )


def test_collate_three_samples():
    # Sample 0: 2 nodes, 2 hyperedges
    data0 = HData(
        x=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        edge_index=torch.tensor([[0, 1], [0, 1]]),
    )

    # Sample 1: 1 node, 1 hyperedge
    data1 = HData(x=torch.tensor([[5.0, 6.0]]), edge_index=torch.tensor([[0], [0]]))

    # Sample 2: 3 nodes, 1 hyperedge
    data2 = HData(
        x=torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]),
        edge_index=torch.tensor([[0, 1, 2], [0, 0, 0]]),
    )

    dataset = MockDataset([data0, data1, data2])
    loader = DataLoader(dataset, batch_size=3)

    batched = loader.collate([data0, data1, data2])

    total_nodes_from_all_samples = (
        2 + 1 + 3
    )  # 6 nodes total, 2 from data0, 1 from data1, 3 from data2
    assert batched.num_nodes == total_nodes_from_all_samples
    assert batched.x.size(0) == total_nodes_from_all_samples

    total_edges_from_all_samples = (
        2 + 1 + 1
    )  # 4 hyperedges total, 2 from data0, 1 from data1, 1 from data2
    assert batched.num_edges == total_edges_from_all_samples

    # Sample 0: nodes [0,1], edges [0,1]
    # Sample 1: nodes [2], edges [2] (offset by 2 nodes, 2 edges)
    # Sample 2: nodes [3,4,5], edges [3] (offset by 3 nodes, 3 edges)
    expected_edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 3, 3]])
    assert torch.equal(batched.edge_index, expected_edge_index)


def test_collate_empty_edge_index():
    data0 = HData(x=torch.empty((1, 0)), edge_index=torch.empty((2, 0)))
    data1 = HData(x=torch.empty((1, 0)), edge_index=torch.empty((2, 0)))

    dataset = MockDataset([data0, data1])
    loader = DataLoader(dataset, batch_size=2)

    batched = loader.collate([data0, data1])

    assert batched.num_nodes == 2
    assert batched.edge_index.size(1) == 0

    assert batched.num_edges == 0  # max_edge_id (-1) + 1 = 0


def test_collate_with_explicit_num_nodes_and_edges():
    data0 = HData(
        x=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        edge_index=torch.tensor([[0, 1], [0, 0]]),
        num_nodes=2,
        num_edges=1,
    )
    data1 = HData(
        x=torch.tensor([[5.0, 6.0]]),
        edge_index=torch.tensor([[0], [0]]),
        num_nodes=1,
        num_edges=1,
    )

    dataset = MockDataset([data0, data1])
    loader = DataLoader(dataset, batch_size=2)

    batched = loader.collate([data0, data1])

    assert batched.num_nodes == 2 + 1  # 3 nodes total, 2 from data0 and 1 from data1
    assert (
        batched.num_edges == 1 + 1
    )  # 2 hyperedges total, 1 from data0 and 1 from data1


def test_iteration_over_dataloader():
    # 5 samples with 2 nodes and 1 hyperedge each
    data_list = [
        HData(x=torch.randn(2, 3), edge_index=torch.tensor([[0, 1], [0, 0]]))
        for _ in range(5)
    ]

    dataset = MockDataset(data_list)
    loader = DataLoader(dataset, batch_size=2)

    batch_count = 0
    for batch in loader:
        batch_count += 1
        assert isinstance(batch, HData)
        assert batch.x.size(1) == 3  # 3 features per node

    assert (
        batch_count == 3
    )  # 5 samples with batch_size=2 should give us 3 batches (2 + 2 + 1)


def test_multi_dimensional_edge_attributes():
    data0 = HData(
        x=torch.tensor([[1.0, 2.0]]),
        edge_index=torch.tensor([[0], [0]]),
        edge_attr=torch.tensor([[0.1, 0.2, 0.3]]),
    )
    data1 = HData(
        x=torch.tensor([[3.0, 4.0]]),
        edge_index=torch.tensor([[0], [0]]),
        edge_attr=torch.tensor([[0.4, 0.5, 0.6]]),
    )

    dataset = MockDataset([data0, data1])
    loader = DataLoader(dataset, batch_size=2)

    batched = loader.collate([data0, data1])

    expected_edge_attr = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    assert torch.equal(
        utils.to_non_empty_edgeattr(batched.edge_attr), expected_edge_attr
    )


def test_mixed_edge_attr_presence():
    data0 = HData(
        x=torch.tensor([[1.0, 2.0]]),
        edge_index=torch.tensor([[0], [0]]),
        edge_attr=torch.tensor([[0.5]]),
    )
    data1_no_edge_attr = HData(
        x=torch.tensor([[3.0, 4.0]]),
        edge_index=torch.tensor([[0], [0]]),
    )

    dataset = MockDataset([data0, data1_no_edge_attr])
    loader = DataLoader(dataset, batch_size=2)

    batched = loader.collate([data0, data1_no_edge_attr])

    # Only data0 has edge_attr, so only that should be in the batch
    expected_edge_attr = torch.tensor([[0.5]])
    assert torch.equal(
        utils.to_non_empty_edgeattr(batched.edge_attr), expected_edge_attr
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
