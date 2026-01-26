from typing import Any, List
from hyperbench.data import Dataset
from hyperbench import utils


MOCK_BASE_PATH = "hyperbench/tests/mock"


class MockDataset(Dataset):
    def __init__(self, data_list: list[Any]):
        super().__init__()
        self.data_list = data_list
        self.hypergraph = utils.empty_hifhypergraph()  # Not used in this mock
        self.hdata = utils.empty_hdata()  # Not used in this mock

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index: int | List[int]) -> Any:
        if isinstance(index, list):
            return [self.data_list[i] for i in index]
        return self.data_list[index]

    def download(self):
        # Not implemented for mock as we don't need it
        pass

    def process(self):
        # Not implemented for mock as we don't need it
        pass
