"""
Coordinating switchboard that handles communication between the generator, selector, and
tumblr API compontents.
"""
import json
from flask import Flask, request, jsonify

from bot_config import BotSpecificConstants
from experimental.lambda_helpers import request_ml_from_lambda, parse_sns_request

bot_specific_constants = BotSpecificConstants.load()
bridge_service_port = bot_specific_constants.bridge_service_port
bridge_service_url = bot_specific_constants.bridge_service_url

PROMPT_STACK = {}
RESULT_STACK = {}

### FLASK
app = Flask(__name__)


@app.route("/sns", methods=["POST"])
def sns():
    global RESULT_STACK

    data = parse_sns_request(request)
    id_ = data['id']

    if id_ not in RESULT_STACK:
        RESULT_STACK[id_] = []

    RESULT_STACK[id_].append(data)

    return jsonify({})


@app.route("/pollml", methods=["GET", "POST"])
def pollml():
    global PROMPT_STACK
    global RESULT_STACK

    if request.method == "POST":
        data = request.json
        for id_ in data.keys():
            if id_ not in RESULT_STACK:
                RESULT_STACK[id_] = []
            RESULT_STACK[id_].append(data[id_])

        return jsonify({})
    elif request.method == "GET":
        return jsonify(PROMPT_STACK)


@app.route("/requestml", methods=["POST"])
def requestml():
    data = request.json
    resps = request_ml_from_lambda(data)
    print(resps)

    return jsonify({})


@app.route("/done", methods=["POST"])
def done():
    global PROMPT_STACK
    global RESULT_STACK

    cleared_from = []

    data = request.json
    if data["id"] in PROMPT_STACK:
        del PROMPT_STACK[data["id"]]
        cleared_from.append("PROMPT_STACK")
    if data["id"] in RESULT_STACK:
        del RESULT_STACK[data["id"]]
        cleared_from.append("RESULT_STACK")

    return jsonify({"cleared_from": cleared_from})


@app.route("/alldone", methods=["POST"])
def alldone():
    global PROMPT_STACK
    global RESULT_STACK

    PROMPT_STACK = {}
    RESULT_STACK = {}

    return jsonify({})


@app.route("/getresult", methods=["POST"])
def getresult():
    global RESULT_STACK

    desired_id = request.form["id"]

    if desired_id in RESULT_STACK:
        response = RESULT_STACK[desired_id]
    else:
        # print(f"desired_id: {desired_id} not available, have ids {list(RESULT_STACK.keys())}")
        response = []

    return jsonify(response)


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=bridge_service_port, debug=False)
    except KeyboardInterrupt:
        print("closing session...")
        raise KeyboardInterrupt
