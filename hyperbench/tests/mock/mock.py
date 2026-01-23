from typing import Any, List
from hyperbench.data import Dataset
from hyperbench import utils
from hyperbench.data.dataset import HIFConverter
from hyperbench.types.hypergraph import HIFHypergraph

MOCK_BASE_PATH = "hyperbench/tests/mock"


class MockDataset(Dataset):
    """Mock dataset for testing DataLoader."""

    def __init__(self, data_list: List[Any]):
        super().__init__()
        self.data_list = data_list
        self.hypergraph = utils.empty_hifhypergraph()  # Not used in this mock
        self.hdata = utils.empty_hdata()  # Not used in this mock

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]

    def download(self):
        # Not implemented for mock as we don't need it
        pass

    def process(self):
        # Not implemented for mock as we don't need it
        pass


class AlgebraMockDataset(Dataset):
    GDRIVE_FILE_ID = "1-H21_mZTcbbae4U_yM3xzXX19VhbCZ9C"
    DATASET_NAME = "ALGEBRA"


class FakeMockDataset(MockDataset):
    GDRIVE_FILE_ID = "fake_id"
    DATASET_NAME = "FAKE"

    def download(self) -> HIFHypergraph:
        hypergraph = HIFConverter.load_from_hif(self.DATASET_NAME, self.GDRIVE_FILE_ID)
        return hypergraph


class FakeMockDataset2(MockDataset):
    GDRIVE_FILE_ID = "fake_id"
    DATASET_NAME = None

    def download(self) -> HIFHypergraph:
        hypergraph = HIFConverter.load_from_hif(self.DATASET_NAME, self.GDRIVE_FILE_ID)
        return hypergraph


class FakeMockDataset3(Dataset):
    GDRIVE_FILE_ID = "fake_id"
    DATASET_NAME = None

    def download(self):
        return None
