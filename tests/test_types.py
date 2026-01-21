import pytest
from hyperbench.types import HIFHypergraph


def build_HIFHypergraph_instance():
    path_hif = "tests/test_data/algebra.hif.json"

    hypergraph = HIFHypergraph.from_hif_file(path_hif)

    assert isinstance(hypergraph, HIFHypergraph)
