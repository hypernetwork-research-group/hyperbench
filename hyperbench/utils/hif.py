import fastjsonschema
import json
import requests


def validate_hif_json(filename):
    url = "https://raw.githubusercontent.com/HIF-org/HIF-standard/main/schemas/hif_schema.json"
    try:
        schema = requests.get(url, timeout=10).json()
    except (requests.RequestException, requests.Timeout):
        with open("../schema/hif_schema.json", "r") as f:
            schema = json.load(f)
    validator = fastjsonschema.compile(schema)
    hiftext = json.load(open(filename, "r"))
    try:
        validator(hiftext)
        return True
    except Exception as e:
        print(f"Invalid JSON: {e}")
        return False
