import sys
import time

import requests
from requests.exceptions import ConnectionError, Timeout

sys.path.append("/nostalgebraist-autoresponer/src")
sys.path.append("/content/improved-diffusion")

import api_ml.ml_layer_diffusion


while True:
    # main poll
    sys.stdout = fake

    api_ml.ml_layer_diffusion.loop_poll(
        period=5,
        n_loops=1
    )
    sys.stdout = _STDOUT_REF

    # check for switch
    r = requests.get(
        f"{BRIDGE_SERVICE_REMOTE_HOST}:{port}/pollml",
    )

    time.sleep(0.2)

    data = r.json()
    if not (data is None or len(data) == 0):
        break
