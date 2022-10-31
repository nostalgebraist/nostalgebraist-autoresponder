import sys
import io
import time

import requests
from requests.exceptions import ConnectionError, Timeout

import imageio.plugins.ffmpeg
imageio.plugins.ffmpeg.download = lambda: None

sys.path.append("src")
sys.path.append("magma")

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
        n_loops=5,
        use_almostdone=False,
    )
    sys.stdout = _STDOUT_REF

    # check for switch
    host, port = api_ml.ml_layer_torch.BRIDGE_SERVICE_REMOTE_HOST, api_ml.ml_layer_torch.bridge_service_port
    r = requests.get(
        f"{host}:{port}/polldiffusion",
    )

    time.sleep(0.2)

    data = r.json()
    if not (data is None or len(data) == 0):
        break
