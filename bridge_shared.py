import time
import hashlib
import requests

from bot_config import BotSpecificConstants

bot_specific_constants = BotSpecificConstants.load()

bridge_service_url = bot_specific_constants.bridge_service_url


def bridge_service_unique_id(url, data):
    unique_string = url + str({k: data[k] for k in sorted(data.keys())})
    hashed = hashlib.md5(unique_string.encode("utf-8")).hexdigest()
    return hashed


def wait_for_result(new_id, wait_first_time=40, wait_recheck_time=5):
    print("waiting for result", end="... ")
    started_waiting_ts = time.time()
    data = {"id": new_id}
    time.sleep(wait_first_time)
    result = requests.post(bridge_service_url + "/getresult", data=data).json()
    n_tries = 0

    while not result["done"]:
        time_to_wait = wait_recheck_time if n_tries < 100 else wait_recheck_time * 10
        n_tries += 1
        print(n_tries, end="... ")
        time.sleep(time_to_wait)
        result = requests.post(bridge_service_url + "/getresult", data=data).json()

    done_waiting_ts = time.time()
    dt = done_waiting_ts - started_waiting_ts
    print(f"Turnaround time: {dt//60:.0f} min {dt%60:.0f}s")
    return result["result"]


def side_judgments_from_gpt2_service(
    texts,
    v8_timestamps=None,
    v10_timestamps=None,
    wait_first_time=5,
    wait_recheck_time=2.5,
    verbose=False,
):
    if verbose:
        print(f"side_judgements_from_gpt2_service: v10_timestamps={v10_timestamps}")
    data = {"texts": texts}
    if v8_timestamps is not None:
        data["v8_timestamps"] = v8_timestamps
    if v10_timestamps is not None:
        data["v10_timestamps"] = v10_timestamps

    if verbose:
        print(f"side_judgments_from_gpt2_service: data={data}")
    url = bridge_service_url + "/raw_select"
    new_id = bridge_service_unique_id(url, data)

    data_to_send = dict()
    data_to_send.update(data)
    data_to_send["id"] = new_id

    requests.post(url, json=data_to_send)
    result = wait_for_result(new_id, wait_first_time=5, wait_recheck_time=2.5)
    return result
