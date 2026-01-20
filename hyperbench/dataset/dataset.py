"""Example usage of the Hypergraph class with HIF data."""
import json
import gdown

from hyperbench.types.hypergraph import Hypergraph
from hyperbench.utils.hif import validate_hif_json

#https://drive.google.com/file/d/
id = "1-H21_mZTcbbae4U_yM3xzXX19VhbCZ9C"
output = "hyperbench/dataset/test.json"
gdown.download(id=id, output=output, quiet=False, fuzzy=True)

hiftext = json.load(open(output,'r'))

validate_hif_json(output)

H = Hypergraph.from_hif(hiftext)
print(H.network_type)
print(H.metadata)
print(H.incidences)
print(H.nodes)
print(H.edges)  