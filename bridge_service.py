"""
Coordinating switchboard that handles communication between the generator, selector, and
tumblr API compontents.
"""
from flask import Flask, request, jsonify

from bot_config import BotSpecificConstants
from experimental.lambda_helpers import request_ml_from_lambda, parse_sns_request
from experimental.lambda_pool_singleton import LAMBDA_POOL

bot_specific_constants = BotSpecificConstants.load()
bridge_service_port = bot_specific_constants.bridge_service_port
bridge_service_url = bot_specific_constants.bridge_service_url

REQUESTS = {}
RESULTS = {}

### FLASK
app = Flask(__name__)


@app.route("/sns", methods=["POST"])
def sns():
    global RESULTS

    data = parse_sns_request(request)
    if 'id' not in data:
        # for warmups
        return jsonify({})

    id_ = data['id']

    if id_ not in RESULTS:
        RESULTS[id_] = []

    RESULTS[id_].append(data)

    if id_ not in LAMBDA_POOL.bridge_ids_to_request_data:
        print(f"unknown id {id_} have ids_ {sorted(LAMBDA_POOL.bridge_ids_to_request_data.keys())}")
        repeat_until_done_signal = False
    else:
        repeat_until_done_signal = LAMBDA_POOL.bridge_ids_to_request_data[id_].get('repeat_until_done_signal', False)
        print(f"repeat_until_done_signal: {repeat_until_done_signal} for {id_}")

    if repeat_until_done_signal:
        # stopping the world in service!  i *think* it's okay...
        LAMBDA_POOL.request(data=LAMBDA_POOL.bridge_ids_to_request_data[id_])

    return jsonify({})


@app.route("/pollml", methods=["GET", "POST"])
def pollml():
    global REQUESTS
    global RESULTS

    if request.method == "POST":
        data = request.json
        for id_ in data.keys():
            if id_ not in RESULTS:
                RESULTS[id_] = []
            RESULTS[id_].append(data[id_])

        return jsonify({})
    elif request.method == "GET":
        return jsonify(REQUESTS)


@app.route("/requestml", methods=["POST"])
def requestml():
    global REQUESTS

    data = request.json
    n_concurrent = 5 if data.get('model') == 'generator' else 1
    request_ml_from_lambda(data, n_concurrent=n_concurrent)

    REQUESTS[data['id']] = data

    return jsonify({})


@app.route("/done", methods=["POST"])
def done():
    global REQUESTS
    global RESULTS

    cleared_from = []

    data = request.json
    if data["id"] in REQUESTS:
        del REQUESTS[data["id"]]
        cleared_from.append("REQUESTS")
    if data["id"] in RESULTS:
        del RESULTS[data["id"]]
        cleared_from.append("RESULTS")

    return jsonify({"cleared_from": cleared_from})


@app.route("/alldone", methods=["POST"])
def alldone():
    global REQUESTS
    global RESULTS

    REQUESTS = {}
    RESULTS = {}

    return jsonify({})


@app.route("/getresult", methods=["POST"])
def getresult():
    global RESULTS

    desired_id = request.form["id"]

    if desired_id in RESULTS:
        response = RESULTS[desired_id]
    else:
        print(f"desired_id: {desired_id} not available, have ids {list(RESULTS.keys())}")
        response = []

    return jsonify(response)


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=bridge_service_port, debug=False)
    except KeyboardInterrupt:
        print("closing session...")
        raise KeyboardInterrupt
