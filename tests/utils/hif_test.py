import requests

from unittest.mock import patch, mock_open, MagicMock
from hyperbench.utils import validate_hif_json


def test_validate_hif_json():
    path_invalid = "tests/mock/hif_not_compliant.json"
    assert not validate_hif_json(path_invalid)

    path_valid = "tests/mock/hif_compliant.json"
    assert validate_hif_json(path_valid)


def test_validate_hif_json_with_url_success():
    path_valid = "tests/mock/hif_compliant.json"

    with patch("hyperbench.utils.hif.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {"type": "object"}  # Minimal valid schema
        mock_get.return_value = mock_response

        validate_hif_json(path_valid)
        mock_get.assert_called_once_with(
            "https://raw.githubusercontent.com/HIF-org/HIF-standard/main/schemas/hif_schema.json",
            timeout=10,
        )


def test_validate_hif_json_with_url_timeout_fallback():
    path_valid = "tests/mock_data/hif_compliant.json"

    with (
        patch("hyperbench.utils.hif.requests.get") as mock_get,
        patch("builtins.open", mock_open(read_data='{"type": "object"}')) as mock_file,
    ):
        mock_get.side_effect = requests.Timeout("Connection timeout")
        validate_hif_json(path_valid)
        # local file was opened
        calls = [str(call) for call in mock_file.call_args_list]
        assert any("../schema/hif_schema.json" in call for call in calls)


def test_validate_hif_json_with_url_request_exception_fallback():
    path_valid = "tests/mock_data/hif_compliant.json"

    with (
        patch("hyperbench.utils.hif.requests.get") as mock_get,
        patch("builtins.open", mock_open(read_data='{"type": "object"}')) as mock_file,
    ):
        mock_get.side_effect = requests.RequestException("Network error")
        validate_hif_json(path_valid)
        # local file was opened
        calls = [str(call) for call in mock_file.call_args_list]
        assert any("../schema/hif_schema.json" in call for call in calls)
