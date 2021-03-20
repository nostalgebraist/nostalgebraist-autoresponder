import json
import sys
sys.path.append(".")
sys.path.append("gpt-2/")
sys.path.append("gpt-2/src/")
from experimental.ml_layer import handle_request


def lambda_handler(data, lambda_context):
    if data.get("hi"):
        return {}

    return handle_request(data)
