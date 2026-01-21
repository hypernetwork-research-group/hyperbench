import torch


class HData:
    def __init__(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor = None,
        edge_attr: torch.Tensor = None,
        num_nodes: int = None,
        num_edges: int = None,
    ):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.num_nodes = num_nodes
        self.num_edges = num_edges
