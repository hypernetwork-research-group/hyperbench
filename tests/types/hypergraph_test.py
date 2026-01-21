import json

from hyperbench.types import HIFHypergraph


def test_build_HIFHypergraph_instance():
    with open("tests/mock/algebra.hif.json", "r") as f:
        hiftext = json.load(f)

    hypergraph = HIFHypergraph.from_hif(hiftext)

    assert isinstance(hypergraph, HIFHypergraph)
