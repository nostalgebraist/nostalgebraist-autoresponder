import sys
import time

import requests
from requests.exceptions import ConnectionError, Timeout

import imageio.plugins.ffmpeg
imageio.plugins.ffmpeg.download = lambda: None

sys.path.append("src")
sys.path.append("/content/improved-diffusion")

import api_ml.ml_layer_diffusion


period, n_loops = 1, 1
after_first_generation = False
time_since_last_generation = 0

while True:
    # main poll
    did_generation = api_ml.ml_layer_diffusion.loop_poll(
        period=period,
        n_loops=n_loops,
    )
    after_first_generation = after_first_generation or did_generation
    time_since_last_generation = (not did_generation) * (time_since_last_generation + period * n_loops)

    # infer switch if we've been waiting a while
    if after_first_generation and (time_since_last_generation >= 10):
        print(f"stopping: time_since_last_generation {time_since_last_generation}s")
        break

    # check for switch
    host, port = api_ml.ml_layer_diffusion.BRIDGE_SERVICE_REMOTE_HOST, api_ml.ml_layer_diffusion.bridge_service_port
    r = requests.get(
        f"{host}:{port}/pollml",
    )

    time.sleep(0.2)

    data = r.json()
    if not (data is None or len(data) == 0):
        break
