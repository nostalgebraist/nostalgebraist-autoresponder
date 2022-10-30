import sys
import time

import requests
from requests.exceptions import ConnectionError, Timeout

import imageio.plugins.ffmpeg
imageio.plugins.ffmpeg.download = lambda: None

sys.path.append("src")
sys.path.append("/content/improved-diffusion")

import api_ml.ml_layer_diffusion


while True:
    # main poll
    api_ml.ml_layer_diffusion.loop_poll(
        period=5,
        n_loops=1
    )

    # check for switch
    host, port = api_ml.ml_layer_diffusion.BRIDGE_SERVICE_REMOTE_HOST, api_ml.ml_layer_diffusion.bridge_service_port
    r = requests.get(
        f"{host}:{port}/pollml",
    )

    time.sleep(0.2)

    data = r.json()
    if not (data is None or len(data) == 0):
        break
