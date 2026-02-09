from torch import Tensor
from typing import Optional, List, Dict, Any, Literal


class HIFHypergraph:
    """
    A hypergraph data structure that supports directed/undirected hyperedges
    with incidence-based representation.
    """

    def __init__(
        self,
        network_type: Optional[Literal["asc", "directed", "undirected"]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        incidences: Optional[List[Dict[str, Any]]] = None,
        nodes: Optional[List[Dict[str, Any]]] = None,
        edges: Optional[List[Dict[str, Any]]] = None,
    ):
        self.network_type = network_type
        self.metadata = metadata if metadata is not None else {}
        self.incidences = incidences if incidences is not None else []
        self.nodes = nodes if nodes is not None else []
        self.edges = edges if edges is not None else []

    @classmethod
    def from_hif(cls, data: Dict[str, Any]) -> "HIFHypergraph":
        """
        Create a Hypergraph from a HIF (Hypergraph Interchange Format).

        Args:
            data: Dictionary with keys: network-type, metadata, incidences, nodes, edges

        Returns:
            Hypergraph instance
        """
        network_type = data.get("network-type") or data.get("network_type")
        metadata = data.get("metadata", {})
        incidences = data.get("incidences", [])
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        return cls(
            network_type=network_type,
            metadata=metadata,
            incidences=incidences,
            nodes=nodes,
            edges=edges,
        )

    @property
    def num_nodes(self) -> int:
        """Return the number of nodes in the hypergraph."""
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        """Return the number of edges in the hypergraph."""
        return len(self.edges)


class Hypergraph:
    """
    A simple hypergraph data structure using edge list representation.
    """

    def __init__(self, edges: List[List[int]]):
        self.edges = edges

    @property
    def num_nodes(self) -> int:
        """Return the number of nodes in the hypergraph."""
        nodes = set()
        for edge in self.edges:
            nodes.update(edge)
        return len(nodes)

    @property
    def num_edges(self) -> int:
        """Return the number of edges in the hypergraph."""
        return len(self.edges)

    @classmethod
    def from_hyperedge_index(cls, hyperedge_index: Tensor) -> "Hypergraph":
        """
        Create a Hypergraph from a hyperedge index representation.

        Args:
            hyperedge_index: Tensor of shape (2, |E|) representing hyperedges, where each column is (node, hyperedge).

        Returns:
            Hypergraph instance
        """
        if hyperedge_index.size(1) < 1:
            return cls(edges=[])

        max_edge_id = int(hyperedge_index[1].max().item())
        edges = [
            hyperedge_index[0, hyperedge_index[1] == edge_id].tolist()
            for edge_id in range(max_edge_id + 1)
        ]
        return cls(edges=edges)
