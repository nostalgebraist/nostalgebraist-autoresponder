"""
Coordinating switchboard that handles communication between the generator, selector, and
tumblr API compontents.
"""
from flask import Flask, request, jsonify, make_response

import config.bot_config_singleton
bot_specific_constants = config.bot_config_singleton.bot_specific_constants

bridge_service_port = bot_specific_constants.bridge_service_port
bridge_service_url = bot_specific_constants.bridge_service_url

PROMPT_STACK = {}
RESULT_STACK = {}

PROMPT_DIFFUSION = {}
RESULT_DIFFUSION = {}

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

            if id_ in PROMPT_STACK and PROMPT_STACK[id_].get('incrementing_seed'):
                if 'seed' in PROMPT_STACK[id_].get('kwargs', {}):
                    cur_seed = PROMPT_STACK[id_]['kwargs']['seed']
                    new_seed = cur_seed + 1
                    print(f"incremented seed for {id_}: {cur_seed} -> {new_seed}")
                    PROMPT_STACK[id_]['kwargs']['seed'] = new_seed

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


@app.route("/polldiffusion", methods=["GET"])
def polldiffusion():
    global PROMPT_DIFFUSION

    return jsonify(PROMPT_DIFFUSION)


@app.route("/polldiffusion/<bridge_id>", methods=["POST"])
def polldiffusion_id(bridge_id):
    global PROMPT_DIFFUSION
    global RESULT_DIFFUSION

    RESULT_DIFFUSION[bridge_id] = request.data

    if PROMPT_DIFFUSION is not None and bridge_id == PROMPT_DIFFUSION["id"]:
        PROMPT_DIFFUSION = None
    return jsonify({})


@app.route("/requestdiffusion", methods=["POST"])
def requestdiffusion():
    global PROMPT_DIFFUSION

    data = request.json
    PROMPT_DIFFUSION = data

    return jsonify({})


@app.route("/almostdone", methods=["POST"])
def almostdone():
    global PROMPT_STACK

    data = request.json
    if data["id"] in PROMPT_STACK:
        PROMPT_STACK[data["id"]]["almost_done"] = True

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
    global PROMPT_DIFFUSION

    PROMPT_STACK = {}
    RESULT_STACK = {}
    PROMPT_DIFFUSION = None

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


@app.route("/getresultdiffusion/<bridge_id>", methods=["GET"])
def getresultdiffusion(bridge_id):
    global RESULT_DIFFUSION

    if bridge_id in RESULT_DIFFUSION:
        ret = RESULT_DIFFUSION.pop(bridge_id)
        print((type(ret), len(ret)))
        response = make_response(ret)
        response.headers.set('Content-Type', 'image/png')
        return response
    else:
        return jsonify({})


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=bridge_service_port, debug=False)
    except KeyboardInterrupt:
        print("closing session...")
        raise KeyboardInterrupt
