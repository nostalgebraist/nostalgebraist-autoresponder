import time
from datetime import datetime
import hashlib
import requests

from bot_config import BotSpecificConstants

bot_specific_constants = BotSpecificConstants.load()

bridge_service_url = bot_specific_constants.bridge_service_url


def bridge_service_unique_id(url, data):
    unique_string = url + str({k: data[k] for k in sorted(data.keys())})
    hashed = hashlib.md5(unique_string.encode("utf-8")).hexdigest()
    return hashed


def wait_for_result(new_id,
                    n_expected=1,
                    wait_first_time=40, wait_recheck_time=5, verbose=True, return_times=False):
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    vprint("waiting for result", end="... ")
    started_waiting_ts = time.time()
    data = {"id": new_id}
    time.sleep(wait_first_time)
    result = requests.post(bridge_service_url + "/getresult", data=data).json()
    n_tries = 0

    while len(result) < n_expected:
        time_to_wait = wait_recheck_time if n_tries < 100 else wait_recheck_time * 10
        n_tries += 1
        vprint(n_tries, end="... ")
        time.sleep(time_to_wait)
        result = requests.post(bridge_service_url + "/getresult", data=data).json()

    done_waiting_ts = time.time()
    dt = done_waiting_ts - started_waiting_ts
    done_waiting_datetime = datetime.now()
    vprint(f"Turnaround time: {dt//60:.0f} min {dt%60:.0f}s")

    if return_times:
        return result, done_waiting_datetime, dt
    return result


def send_alldone():
    requests.post(bridge_service_url + "/alldone")
