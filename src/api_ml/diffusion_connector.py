import time
from io import BytesIO

import requests
from PIL import Image

from api_ml.bridge_shared import get_bridge_service_url, bridge_service_unique_id


def request_diffusion(text, **kwargs):
    data = {'prompt': text}
    data.update(kwargs)

    bridge_service_url = get_bridge_service_url()

    bridge_id = bridge_service_unique_id(bridge_service_url, data)

    data_to_send = dict()
    data_to_send.update(data)
    data_to_send["id"] = bridge_id

    requests.post(bridge_service_url + "/requestdiffusion", json=data_to_send)
    return bridge_id


def wait_for_result_diffusion(bridge_id, wait_first_time=40, wait_recheck_time=2):
    bridge_service_url = get_bridge_service_url()
    print("waiting for result", end="... ")
    started_waiting_ts = time.time()
    time.sleep(wait_first_time)
    result = requests.get(bridge_service_url + f"/getresultdiffusion/{bridge_id}")
    n_tries = 0

    def _try_load(result_):
        try:
            data = result_.content
            if not isinstance(data, bytes) or len(data) == 0:
                return
            with BytesIO(data) as b:
                im = Image.open(b)
                im.load()
            return im
        except Exception as e:
            return

    im = _try_load(result)

    while im is None:
        time_to_wait = wait_recheck_time
        n_tries += 1
        print(n_tries, end="... ")
        time.sleep(time_to_wait)
        result = requests.get(bridge_service_url + f"/getresultdiffusion/{bridge_id}")
        im = _try_load(result)

    done_waiting_ts = time.time()
    dt = done_waiting_ts - started_waiting_ts
    print(f"Turnaround time: {dt//60:.0f} min {dt%60:.0f}s")
    return im


def make_image_with_diffusion(text, **kwargs):
    bridge_id = request_diffusion(text, **kwargs)

    return wait_for_result_diffusion(bridge_id)
