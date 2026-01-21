import torch

class HData:
    def __init__(self, x: torch.Tensor):
        self.x = x
        self.edge_index = None
        self.edge_attr = None
        self.num_nodes = None
        self.num_edges = None
