import sys
import io
import time

import requests
from requests.exceptions import ConnectionError, Timeout

sys.path.append("/nostalgebraist-autoresponder/src")
sys.path.append("/content/improved-diffusion")

import api_ml.ml_layer_torch


class FakeStream(io.IOBase):
    def write(self, *args, **kwargs): pass


_STDOUT_REF = sys.stdout

fake = FakeStream()

while True:
    # main poll
    sys.stdout = fake

    api_ml.ml_layer_torch.loop_poll(
        period=1,
        n_loops=5
    )
    sys.stdout = _STDOUT_REF

    # check for switch
    r = requests.get(
        f"{BRIDGE_SERVICE_REMOTE_HOST}:{port}/polldiffusion",
    )

    time.sleep(0.2)

    data = r.json()
    if not (data is None or len(data) == 0):
        break
