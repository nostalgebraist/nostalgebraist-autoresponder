"""
Coordinating switchboard that handles communication between the generator, selector, and
tumblr API compontents.
"""
import numpy as np
import uuid
from datetime import datetime
from itertools import chain

from flask import Flask, request, jsonify

from bot_config import BotSpecificConstants
from mood import get_mood_by_name, load_logit_diff_sample, estimate_expected_rejections
from bridge_shared import bridge_service_unique_id

from autoresponder_static import DEFAULT_CSC
from autoresponder_static_v8 import timestamp_to_v10_format

bot_specific_constants = BotSpecificConstants.load()
bridge_service_port = bot_specific_constants.bridge_service_port
bridge_service_url = bot_specific_constants.bridge_service_url

TRADE_QUALITY_FOR_SPEED = False

logit_diff_sample_series = load_logit_diff_sample()
EXPECTED_REJECTION_MULT = 0.5 if (not TRADE_QUALITY_FOR_SPEED) else 0.4

# note to self: trying to be more random about textposts / use retention_stack less
# to give the selector more training signal
#
# interventions:
#   - fewer candidates
#   - higher eps in eps_greedy
#   - higher retention_stack proba cutoff (to the point that the stack is usu. small)
TEXTPOST_N_CANDIDATES_TARGET = 15 if (not TRADE_QUALITY_FOR_SPEED) else 12
# TEXTPOST_N_CANDIDATES_TARGET = 20 if (not TRADE_QUALITY_FOR_SPEED) else 18

Q_CHAR = "会"
A_CHAR = "域"
T_CHAR = "职"

UNAME_CHAR = "友"
ORIG_POST_CHAR = "翰"

AB_TEST_SELECTOR = True
AB_TEST_A_SEQUENCE = "\uFFF9"
AB_TEST_B_SEQUENCE = "\uFFF9\uFFFA\uFFFB"

PROMPT_STACK = {}
RESULT_STACK = {}

GENERATION_RESULT_STACK = {}

GENERATIONS_PER_REQUEST = 1

### FLASK
app = Flask(__name__)


@app.route("/pollml", methods=["GET", "POST"])
def pollml():
    global PROMPT_STACK
    global RESULT_STACK

    if request.method == "POST":
        data = request.json
        for id_ in data.keys():
            if id_ not in RESULT_STACK:
                RESULT_STACK[id_] = []
            # print(f"for {id_}, length before: {len(RESULT_STACK[id_])}")
            RESULT_STACK[id_].append(data[id_])

        # for id_ in data.keys():
        #     print(f"for {id_}, length after: {len(RESULT_STACK[id_])}")

        return jsonify({})
    elif request.method == "GET":
        return jsonify(PROMPT_STACK)


@app.route("/requestml", methods=["POST"])
def requestml():
    global PROMPT_STACK

    data = request.json
    PROMPT_STACK[data["id"]] = data

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
