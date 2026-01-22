import json

from hyperbench.types import HIFHypergraph
from hyperbench.tests import MOCK_BASE_PATH


def test_build_HIFHypergraph_instance():
    with open(f"{MOCK_BASE_PATH}/algebra.hif.json", "r") as f:
        hiftext = json.load(f)

    hypergraph = HIFHypergraph.from_hif(hiftext)

    assert isinstance(hypergraph, HIFHypergraph)
