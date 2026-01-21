import pytest
from hyperbench.utils import validate_hif_json


def test_validate_hif_json():

    path = "tests/test_data/hif_invalid.json"
    assert not validate_hif_json(path)

