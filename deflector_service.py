from flask import Flask, request, jsonify
app = Flask(__name__)

from bot_config import BotSpecificConstants

bot_specific_constants = BotSpecificConstants.load()
deflector_service_port = bot_specific_constants.deflector_service_port
bridge_service_port = bot_specific_constants.bridge_service_port
BRIDGE_SERVICE_REMOTE_HOST = bot_specific_constants.BRIDGE_SERVICE_REMOTE_HOST


@app.route("/pollml", methods=["GET", "POST"])
def pollml():
    if request.method == "POST":
        requests.post(f"{BRIDGE_SERVICE_REMOTE_HOST}:{bridge_service_port}/pollml", json=request.json())
        return jsonify({})
    else:
        r = requests.get(f"{BRIDGE_SERVICE_REMOTE_HOST}:{bridge_service_port}/pollml")
        return r.json


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=deflector_service_port, debug=False)
    except KeyboardInterrupt:
        print("closing session...")
        raise KeyboardInterrupt
