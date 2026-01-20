import fastjsonschema
import json
import requests

def validate_hif_json(filename):
    url = "https://raw.githubusercontent.com/HIF-org/HIF-standard/main/schemas/hif_schema.json"
    schema = requests.get(url).json()
    validator = fastjsonschema.compile(schema)
    hiftext = json.load(open(filename,'r'))
    try:
        validator(hiftext)
        print("HIF-Compliant JSON.")
    except Exception as e:
        print(f"Invalid JSON: {e}")
