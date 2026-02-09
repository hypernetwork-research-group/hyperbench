from torch import Tensor, device
from typing import Optional


class HData:
    """
    Container for hypergraph data.

    Attributes:
        x (Tensor): Node feature matrix of shape [num_nodes, num_features].
        edge_index (Tensor): Hyperedge connectivity in COO format
            of shape [2, num_incidences], where edge_index[0] contains
            node IDs and edge_index[1] contains hyperedge IDs.
        edge_attr (Tensor, optional): Hyperedge feature matrix of shape
            [num_edges, num_edge_features]. Features associated with each
            hyperedge (e.g., weights, timestamps, types).
        num_nodes (int, optional): Number of nodes in the hypergraph.
            If None, inferred as x.size(0).
        num_edges (int, optional): Number of hyperedges in the hypergraph.
            If None, inferred as edge_index[1].max().item() + 1.

    Example:
        >>> x = torch.randn(10, 16)  # 10 nodes with 16 features each
        >>> edge_index = torch.tensor([[0, 0, 1, 1, 1],  # node IDs
        ...                            [0, 1, 2, 3, 4]]) # hyperedge IDs
        >>> data = HData(x, edge_index=edge_index)
    """

    def __init__(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
        num_edges: Optional[int] = None,
    ):
        self.x: Tensor = x

        self.edge_index: Tensor = edge_index

        self.edge_attr: Optional[Tensor] = edge_attr

        self.num_nodes: int = num_nodes if num_nodes is not None else x.size(0)

        max_edge_id = edge_index[1].max().item() if edge_index.size(1) > 0 else -1
        self.num_edges: int = num_edges if num_edges is not None else max_edge_id + 1

    def to(self, device: device | str, non_blocking: bool = False) -> "HData":
        self.x = self.x.to(device=device, non_blocking=non_blocking)
        self.edge_index = self.edge_index.to(device=device, non_blocking=non_blocking)
        if self.edge_attr is not None:
            self.edge_attr = self.edge_attr.to(device=device, non_blocking=non_blocking)
        return self

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"    num_nodes={self.num_nodes},\n"
            f"    num_edges={self.num_edges},\n"
            f"    x_shape={self.x.shape},\n"
            f"    edge_index_shape={self.edge_index.shape},\n"
            f"    edge_attr_shape={self.edge_attr.shape if self.edge_attr is not None else None}\n"
            f")"
        )
