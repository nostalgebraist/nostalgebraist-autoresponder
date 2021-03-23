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

    if 'body' in data:
        data_ = json.loads(data['body'])
        data = data_

    if data.get("hi"):
        if 'time_before_responding_sec' in data:
            time.sleep(data['time_before_responding_sec'])
        return {"lambda_uid": lambda_uid, "id": data.get('id')}

    return handle_request(data, lambda_uid=lambda_uid)
