import time
import hashlib
import requests


def bridge_service_unique_id(url, data):
    unique_string = url + str({k: data[k] for k in sorted(data.keys())})
    hashed = hashlib.md5(unique_string.encode("utf-8")).hexdigest()
    return hashed


def get_bridge_service_url():
    import config.bot_config_singleton
    return config.bot_config_singleton.bot_specific_constants.bridge_service_url


def wait_for_result(new_id, wait_first_time=40, wait_recheck_time=5):
    bridge_service_url = get_bridge_service_url()
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


def send_alldone():
    bridge_service_url = get_bridge_service_url()
    requests.post(bridge_service_url + "/alldone")
