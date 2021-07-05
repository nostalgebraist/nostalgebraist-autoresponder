import os
import json
import time
from typing import Tuple, NamedTuple

import requests

from api_ml.bridge_shared import bridge_service_unique_id, bridge_service_url
from config.autoresponder_config import (
    model_name,
    ckpt_select,
    ckpt_sentiment,
    ckpt_autoreviewer,
)


ModelVersion = NamedTuple("ModelVersion", model_identifier=str, version_identifier=str)

LATEST_MODEL_VERSIONS = (
    ModelVersion("model_name", model_name),
    ModelVersion("ckpt_select", ckpt_select),
    ModelVersion("ckpt_sentiment", ckpt_sentiment),
    ModelVersion("ckpt_autoreviewer", ckpt_autoreviewer),
)

BridgeCacheKey = NamedTuple(
    "BridgeCacheKey", bridge_id=str, model_versions=Tuple[ModelVersion, ...]
)


def make_bridge_cache_key(data: dict) -> BridgeCacheKey:
    bridge_id = bridge_service_unique_id(bridge_service_url, data)
    return BridgeCacheKey(bridge_id=bridge_id, model_versions=LATEST_MODEL_VERSIONS)


def _serialize_bridge_cache_value(v: dict):
    return v[0]["result"]


def _deserialize_bridge_cache_value(vs):
    return [{"result": vs}]


class BridgeCache:
    def __init__(self, cache: dict, last_accessed_time: dict):
        self.cache = cache
        self.last_accessed_time = last_accessed_time

    def query(self, data: dict):
        key = make_bridge_cache_key(data)

        if key not in self.cache:
            response_data = self.call_bridge(data, bridge_id=key.bridge_id)

            true_key = self.key_from_response_data(
                bridge_id=key.bridge_id, response_data=response_data
            )
            key = true_key

            self.cache[key] = response_data

        self.last_accessed_time[key] = time.time()
        return self.cache[key]

    @staticmethod
    def key_from_response_data(response_data: dict, bridge_id: str):
        entry = response_data[0]
        model_info = entry["model_info"]

        model_versions = (
            ModelVersion("model_name", model_info.get("model_name", "")),
            ModelVersion("ckpt_select", model_info.get("ckpt_select", "")),
            ModelVersion("ckpt_sentiment", model_info.get("ckpt_sentiment", "")),
            ModelVersion("ckpt_autoreviewer", model_info.get("ckpt_autoreviewer", "")),
        )

        return BridgeCacheKey(bridge_id=bridge_id, model_versions=model_versions)

    @staticmethod
    def call_bridge(data: dict, bridge_id: str):
        data_to_send = dict()
        data_to_send.update(data)
        data_to_send["id"] = bridge_id

        requests.post(bridge_service_url + "/requestml", json=data_to_send)

        response_data = []
        while len(response_data) == 0:
            time.sleep(1)
            response_data = requests.post(
                bridge_service_url + "/getresult", data={"id": bridge_id}
            ).json()

        requests.post(bridge_service_url + "/done", json={"id": bridge_id})
        return response_data

    def remove_oldest(self, max_hours=2, dryrun=False):
        lat = self.last_accessed_time
        existing = self.cache

        last_allowed_time = time.time() - (3600 * max_hours)

        allowed = {k for k, t in lat.items() if t >= last_allowed_time}

        new = {k: existing[k] for k in existing if k in allowed}

        before_len = len(existing)
        delta_len = before_len - len(new)

        if delta_len > 0:
            if dryrun:
                print(f"remove_oldest: would drop {delta_len} of {before_len} in bridge cache")
            else:
                print(f"remove_oldest: dropping {delta_len} of {before_len} in bridge cache")
                self.cache = new

    def to_json(self):
        return [{"key": k, "value": _serialize_bridge_cache_value(v), "last_accessed_time": self.last_accessed_time[k]}
                for k, v in self.cache.items()]

    @staticmethod
    def from_json(entries: list) -> "BridgeCache":
        def _parse_key(key_json: list) -> BridgeCacheKey:
            bridge_id = key_json[0]
            model_versions = tuple(
                [
                    ModelVersion(model_identifier=mv[0], version_identifier=mv[1])
                    for mv in key_json[1]
                ]
            )
            return BridgeCacheKey(bridge_id=bridge_id, model_versions=model_versions)

        cache = {}
        last_accessed_time = {}
        for entry in entries:
            key = _parse_key(entry["key"])
            cache[key] = _deserialize_bridge_cache_value(entry["value"])
            last_accessed_time[key] = entry["last_accessed_time"]
        return BridgeCache(cache=cache, last_accessed_time=last_accessed_time)

    def save(self, path="data/bridge_cache.json"):
        t1 = time.time()
        self.remove_oldest()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f)
        delta = time.time() - t1
        print(f"saved bridge cache with length {len(self.cache)} in {delta:.2f}s")

    @staticmethod
    def load(path="data/bridge_cache.json") -> "BridgeCache":
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    entries = json.load(f)
                return BridgeCache.from_json(entries)
            except json.decoder.JSONDecodeError:
                print("cache file found, but could not load it")

        print("initializing bridge cache")
        loaded = BridgeCache(cache=dict(), last_accessed_time=dict())
        loaded.remove_oldest()
        return loaded
