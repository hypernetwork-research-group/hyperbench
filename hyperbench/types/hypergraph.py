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
