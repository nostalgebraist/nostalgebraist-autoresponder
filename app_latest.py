import json
import sys
import uuid
import time

sys.path.append(".")
sys.path.append("gpt-2/")
sys.path.append("gpt-2/src/")

from experimental.ml_layer import handle_request

lambda_uid = str(uuid.uuid4())


def lambda_handler(data, lambda_context):
    print(f"got request: {repr(data)}")
    print(f"my lambda_uid is {lambda_uid}")

    if 'body' in data:
        data_ = json.loads(data['body'])
        data = data_

    if 'request_id' not in data:
        raise ValueError('no request id')
    request_id = data['request_id']

    if data.get("hi"):
        if 'time_before_responding_sec' in data:
            time.sleep(data['time_before_responding_sec'])
        return {
            "lambda_uid": lambda_uid,
            "request_id": request_id
        }

    return handle_request(data, lambda_uid=lambda_uid)
